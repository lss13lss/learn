import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x):
        # 计算 RMSNorm
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        # 归一化
        x_norm = x / (rms + self.eps)
        # 缩放
        return self.weight * x_norm
    
