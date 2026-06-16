from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, RotaryEmbedding, apply_rope, MLP
from .projector import Projector
from .compress import build_compressor

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    FLEX_OK = True
except Exception:
    FLEX_OK = False


@dataclass
class DrafterConfig:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    vocab_size: int
    num_target_layers: int
    target_layer_ids: list
    block_size: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    mask_token_id: int = 0
    visual_compress: str = "none"
    num_queries: int = 128


def repeat_kv(x, n):
    # x [B,Hkv,S,hd] -> [B,Hkv*n,S,hd]
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


class KVInjectionAttention(nn.Module):
    """Queries come only from the noise block; keys/values come from
    [injected context | noise]. This is the DFlash conditioning mechanism.

    Shapes: noise [B,Q,D], context [B,Cn,D] (new context this call)."""

    def __init__(self, cfg: DrafterConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.h, self.hkv, self.hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.groups = self.h // self.hkv
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.h * self.hd, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.hkv * self.hd, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.hkv * self.hd, bias=True)
        self.o_proj = nn.Linear(self.h * self.hd, cfg.hidden_size, bias=False)

    def forward(self, noise, context, q_cos, q_sin, k_cos, k_sin, attn_mask,
                past_key_value=None):
        b, q_len = noise.shape[:2]
        kv_in = noise if context is None else torch.cat([context, noise], dim=1)

        q = self.q_proj(noise).view(b, q_len, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(kv_in).view(b, kv_in.shape[1], self.hkv, self.hd).transpose(1, 2)
        v = self.v_proj(kv_in).view(b, kv_in.shape[1], self.hkv, self.hd).transpose(1, 2)

        q = apply_rope(q, q_cos, q_sin)
        k = apply_rope(k, k_cos, k_sin)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)

        if attn_mask is not None and not torch.is_tensor(attn_mask):  # flex BlockMask
            out = flex_attention(q, k, v, block_mask=attn_mask, scale=self.scale)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

        out = out.transpose(1, 2).reshape(b, q_len, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = KVInjectionAttention(cfg, layer_idx)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, hidden, context, q_cos, q_sin, k_cos, k_sin, attn_mask, past_key_value=None):
        # context is normed once outside; here we norm the noise stream input.
        h = self.self_attn(self.input_layernorm(hidden), context,
                           q_cos, q_sin, k_cos, k_sin, attn_mask, past_key_value)
        hidden = hidden + h
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class DFlashDrafter(nn.Module):
    """Block-diffusion drafter. Owns the trainable parts: layers, projector,
    optional visual compressor. embed_tokens and lm_head are the (frozen) target's."""

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.projector = Projector(len(cfg.target_layer_ids), cfg.hidden_size, cfg.rms_norm_eps)
        self.compressor = build_compressor(
            cfg.visual_compress, cfg.hidden_size, cfg.num_queries, cfg.num_kv_heads)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_context(self, target_hidden, visual_mask=None):
        """target_hidden [1,S,L*D] -> (context [1,Sc,D], n_memory, nonmem_index [Sc-n_memory]).

        When a compressor is set and visual tokens exist, the visual tokens are
        pulled out and compressed into n_memory position-less memory tokens that
        prefix the context; the remaining (text/response) tokens follow in order.
        nonmem_index gives the original sequence index of each non-memory token."""
        s = target_hidden.shape[1]
        device = target_hidden.device
        if self.compressor is None or visual_mask is None or int(visual_mask.sum()) == 0:
            return self.projector(target_hidden), 0, torch.arange(s, device=device)
        vis = self.projector(target_hidden[:, visual_mask])      # [1,V,D]
        comp = self.compressor(vis)                              # [1,N,D]
        txt = self.projector(target_hidden[:, ~visual_mask])     # [1,T,D]
        nonmem_index = torch.arange(s, device=device)[~visual_mask]
        return torch.cat([comp, txt], dim=1), comp.shape[1], nonmem_index

    def forward(self, noise_embedding, target_hidden, visual_mask, anchors, keep,
                block_size, attn_backend="flex"):
        """Training forward (single DDP-wrapped call). Returns draft hidden [1,N*bs,D]."""
        context, n_mem, nonmem_idx = self.encode_context(target_hidden, visual_mask)
        device = context.device
        mem_c = torch.full((n_mem,), -1, dtype=torch.long, device=device)
        mem_r = torch.zeros(n_mem, dtype=torch.long, device=device)
        context_seq_pos = torch.cat([mem_c, nonmem_idx]).long()       # memory -> -1 (always visible)
        context_rope_pos = torch.cat([mem_r, nonmem_idx]).long()      # memory -> 0 (identity rope)

        offsets = torch.arange(block_size, device=device)
        draft_pos = (anchors[:, None] + offsets[None, :]).reshape(-1)  # [N*bs]
        q_pos = draft_pos[None]
        k_pos = torch.cat([context_rope_pos, draft_pos])[None]

        if attn_backend == "flex" and FLEX_OK:
            mask = make_flex_train_mask(anchors, keep, context_seq_pos, block_size, device)
        else:
            mask = make_sdpa_train_mask(anchors, keep, context_seq_pos, block_size, device,
                                        noise_embedding.dtype)
        return self.run(noise_embedding, context, q_pos, k_pos, mask)

    def run(self, noise_embedding, context, q_position_ids, k_position_ids, attn_mask,
            past_key_values=None):
        """Forward the noise stream conditioned on already-encoded context.
        noise_embedding [B,Q,D], context [B,Cn,D] (new ctx this call) or None."""
        q_cos, q_sin = self.rotary(q_position_ids)
        k_cos, k_sin = self.rotary(k_position_ids)
        hidden = noise_embedding
        for layer in self.layers:
            hidden = layer(hidden, context, q_cos, q_sin, k_cos, k_sin, attn_mask, past_key_values)
        return self.norm(hidden)


# --------------------------------------------------------------------------
# Training attention masks (context_seq_pos: per-context-token causal position;
# memory/compressed-visual tokens use -1 so they are always visible).
# --------------------------------------------------------------------------

def make_sdpa_train_mask(anchor_positions, block_keep, context_seq_pos, block_size, device, dtype):
    # anchor_positions [N], block_keep [N], context_seq_pos [Sc]
    n = anchor_positions.shape[0]
    sc = context_seq_pos.shape[0]
    q_len = n * block_size
    q_block = torch.arange(q_len, device=device) // block_size           # [Q]
    anchor_q = anchor_positions[q_block]                                  # [Q]

    ctx_ok = context_seq_pos[None, :] < anchor_q[:, None]                 # [Q,Sc]

    kv_block = torch.arange(q_len, device=device) // block_size          # [Q] (draft kv)
    draft_ok = q_block[:, None] == kv_block[None, :]                      # [Q,Q]

    allow = torch.cat([ctx_ok, draft_ok], dim=1)                         # [Q,Sc+Q]
    allow = allow & block_keep[q_block][:, None]
    mask = torch.zeros(q_len, sc + q_len, device=device, dtype=dtype)
    mask.masked_fill_(~allow, torch.finfo(dtype).min)
    return mask[None, None]                                              # [1,1,Q,Sc+Q]


def make_flex_train_mask(anchor_positions, block_keep, context_seq_pos, block_size, device):
    n = anchor_positions.shape[0]
    sc = context_seq_pos.shape[0]
    q_len = n * block_size

    def mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        anchor = anchor_positions[q_block]
        is_ctx = kv_idx < sc
        ctx_ok = is_ctx & (context_seq_pos[kv_idx.clamp(max=sc - 1)] < anchor)
        kv_block = (kv_idx - sc) // block_size
        draft_ok = (~is_ctx) & (q_block == kv_block)
        return (ctx_ok | draft_ok) & block_keep[q_block]

    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=q_len, KV_LEN=sc + q_len, device=device)
