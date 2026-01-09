import torch
import torch.nn as nn



def rope_simple(q, k, base=10000.0):
    # q, k: [B, T, H, D] with D even
    B, T, H, D = q.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"
    half = D // 2

    # 计算频率：ω_i = base^{-2i/D}
    idx = torch.arange(half, device=q.device, dtype=q.dtype)
    freq = base ** (-2 * idx / D)  # [half]

    # 位置索引
    pos = torch.arange(T, device=q.device, dtype=q.dtype)  # [T]

    # 相位：pos * freq -> [T, half]
    phase = torch.einsum('t,d->td', pos, freq)

    # 构造 cos/sin，扩展到 [B, T, H, half]
    cos = phase.cos()[None, :, None, :]
    sin = phase.sin()[None, :, None, :]

    def apply_rot(x):
        x1, x2 = x[..., :half], x[..., half:]  # 拆成两半
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return apply_rot(q), apply_rot(k)



from torch import Tensor
from typing import Optional, Tuple

def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device=None, dtype=None) -> Tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0
    half = head_dim // 2
    idx = torch.arange(half, device=device, dtype=dtype)
    freq = base ** (-2 * idx / head_dim)  # [half]
    pos = torch.arange(seq_len, device=device, dtype=dtype)  # [seq_len]
    phase = torch.outer(pos, freq)  # [seq_len, half]
    cos = phase.cos()  # [seq_len, half]
    sin = phase.sin()  # [seq_len, half]
    return cos, sin

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, T, H, D], cos/sin: [T, half]
    B, T, H, D = x.shape
    half = D // 2
    x = x.reshape(B, T, H, half, 2)
    # x[..., 0] 是前半，x[..., 1] 是后半
    cos = cos[:T, None, None, :]  # [T,1,1,half]
    sin = sin[:T, None, None, :]
    x1, x2 = x[..., 0], x[..., 1]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1).reshape(B, T, H, D)

# 使用示例
def rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

# 预构建 cos/sin（可缓存到模型里）
# cos, sin = build_rope_cache(max_seq_len, head_dim, base=10000.0, device=q.device, dtype=q.dtype)
# q_rot, k_rot = rope(q, k, cos, sin)
