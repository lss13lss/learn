import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# 1. 使用 Dataclass 管理配置 (大厂标配)
@dataclass
class ModelArgs:
    dim: int = 4096
    n_heads: int = 32
    head_dim: int = 128  # dim // n_heads
    dropout: float = 0.1
    max_batch_size: int = 32
    max_seq_len: int = 2048

class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.n_heads
        self.head_dim = args.head_dim
        
        # [工业优化 1] Fused QKV
        # 为什么？比起调用 3 次小矩阵乘法，合并成 1 次大矩阵乘法更能跑满 GPU 算力
        # output_dim = 3 * dim (分别对应 Q, K, V)
        self.c_attn = nn.Linear(args.dim, 3 * args.dim, bias=False)
        
        # 输出投影层
        self.c_proj = nn.Linear(args.dim, args.dim, bias=False)
        
        self.dropout = args.dropout
        
        # [工业优化 2] 注册 Buffer 而不是 Parameter
        # register_buffer 用于保存状态（如 Mask），但它不是需要更新梯度的参数
        # 这里的 mask 会随 model.to(device) 自动移动
        mask = torch.tril(torch.ones(args.max_seq_len, args.max_seq_len))
        mask = mask.view(1, 1, args.max_seq_len, args.max_seq_len)
        self.register_buffer("bias", mask)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time(Seq_len), Channel(Dim)

        # 1. Fused QKV 计算
        # qkv shape: [B, T, 3 * C]
        qkv = self.c_attn(x)
        
        # 2. 拆分 Q, K, V
        # split 后 shape: [B, T, C]
        q, k, v = qkv.split(C, dim=2)
        
        # 3. 变形为多头 (Reshape for Multi-head)
        # [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # [工业优化 3] Flash Attention 分支
        # PyTorch 2.0+ 提供了内置的 scaled_dot_product_attention，会自动调用 FlashAttention CUDA 核
        # 这比手写 mask + softmax 快得多，且省显存
        if hasattr(F, 'scaled_dot_product_attention'):
            # 这是一个非常底层的优化，直接跳过了显存读写瓶颈
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, # FlashAttention 内部有 casual mask 优化
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # 传统手动实现 (Fallback)
            # 这里的数学逻辑和你之前写的一样
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        # 4. 还原形状
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 5. 输出投影
        return self.c_proj(y)