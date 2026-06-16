import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Standard RoPE. Returns cos/sin for given position_ids [B,S] -> [B,S,head_dim]."""

    def __init__(self, head_dim, theta=1e6):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        # position_ids [B,S] -> cos/sin [B,S,head_dim]
        freqs = position_ids.float()[..., None] * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x [B,H,S,hd]; cos/sin [B,S,hd]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
