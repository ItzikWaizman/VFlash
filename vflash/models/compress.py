import torch
import torch.nn as nn
from .layers import RMSNorm


class QFormerCompressor(nn.Module):
    """Learnable-query cross-attention compressor (Q-Former style).

    N learnable queries attend over the V projected visual tokens to produce
    N compressed tokens. N (num_queries) sets the compression ratio.
    visual [B,V,D] -> compressed [B,N,D].
    """

    def __init__(self, hidden_size, num_queries, num_heads, depth=1, eps=1e-6):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "q_norm": RMSNorm(hidden_size, eps),
                "kv_norm": RMSNorm(hidden_size, eps),
                "attn": nn.MultiheadAttention(hidden_size, num_heads, batch_first=True),
                "ff_norm": RMSNorm(hidden_size, eps),
                "ff": nn.Sequential(
                    nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(),
                    nn.Linear(4 * hidden_size, hidden_size),
                ),
            }))

    def forward(self, visual, key_padding_mask=None):
        # visual [B,V,D]; key_padding_mask [B,V] True=pad. -> [B,N,D]
        b = visual.shape[0]
        x = self.query.unsqueeze(0).expand(b, -1, -1)
        for blk in self.blocks:
            q = blk["q_norm"](x)
            kv = blk["kv_norm"](visual)
            a, _ = blk["attn"](q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + a
            x = x + blk["ff"](blk["ff_norm"](x))
        return x


def build_compressor(kind, hidden_size, num_queries, num_heads):
    """Returns a compressor module or None (none/full injection)."""
    if kind == "none":
        return None
    if kind == "qformer":
        return QFormerCompressor(hidden_size, num_queries, num_heads)
    if kind in ("mamba", "nsa"):
        raise NotImplementedError(f"visual_compress={kind} is a later-phase variant")
    raise ValueError(f"unknown visual_compress={kind}")
