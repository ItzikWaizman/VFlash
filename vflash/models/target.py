from types import SimpleNamespace

import torch
import torch.nn as nn

from ..utils import extract_context_feature, build_target_layer_ids


def _find_decoder_layers(model, n_layers):
    """Locate the ModuleList of language-model decoder layers (for warm-start)."""
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) == n_layers:
            child = m[0]
            if hasattr(child, "self_attn") and hasattr(child, "mlp"):
                return m
    raise RuntimeError("could not locate decoder layers for warm-start")


class HFTarget:
    """Wraps a HF (V)LM target: frozen forward provider of logits + selected-layer
    hidden states, plus embed_tokens/lm_head for the drafter."""

    def __init__(self, model, processor, target_layer_ids=None):
        self.model = model.eval().requires_grad_(False)
        self.processor = processor
        tc = model.config.get_text_config()
        self.text_cfg = tc
        self.n_layers = tc.num_hidden_layers
        self.target_layer_ids = target_layer_ids or build_target_layer_ids(self.n_layers, 4)
        self.embed_tokens = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        ic = getattr(model.config, "image_token_index", getattr(model.config, "image_token_id", None))
        vc = getattr(model.config, "video_token_index", getattr(model.config, "video_token_id", None))
        self.visual_token_ids = {t for t in (ic, vc) if t is not None}

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def drafter_cfg(self):
        tc = self.text_cfg
        hd = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
        return dict(
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            num_heads=tc.num_attention_heads,
            num_kv_heads=getattr(tc, "num_key_value_heads", tc.num_attention_heads),
            head_dim=hd,
            vocab_size=tc.vocab_size,
            num_target_layers=self.n_layers,
            target_layer_ids=self.target_layer_ids,
            rms_norm_eps=getattr(tc, "rms_norm_eps", 1e-6),
            rope_theta=getattr(tc, "rope_theta", 1e6),
        )

    def decoder_layers(self):
        return _find_decoder_layers(self.model, self.n_layers)

    def visual_mask(self, input_ids):
        # input_ids [1,S] -> bool [S]
        if not self.visual_token_ids:
            return torch.zeros(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        m = torch.zeros_like(input_ids[0], dtype=torch.bool)
        for t in self.visual_token_ids:
            m |= input_ids[0] == t
        return m

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
                past_key_values=None, attention_mask=None, use_cache=False,
                logits_to_keep=0, want_hidden=True, **mm_inputs):
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids,
                past_key_values=past_key_values, attention_mask=attention_mask,
                use_cache=use_cache, output_hidden_states=want_hidden,
                logits_to_keep=logits_to_keep, **mm_inputs,
            )
        target_hidden = last_hidden = None
        if want_hidden:
            target_hidden = extract_context_feature(out.hidden_states, self.target_layer_ids)
            last_hidden = out.hidden_states[-1]
        return SimpleNamespace(logits=out.logits, target_hidden=target_hidden,
                               last_hidden=last_hidden, past_key_values=out.past_key_values)


def warm_start_drafter(drafter, target):
    """Copy weights from target decoder layers (at target_layer_ids) into the
    drafter's KV-injection layers. Projector/compressor stay randomly initialized."""
    src = target.decoder_layers()
    ids = drafter.cfg.target_layer_ids
    for di, ti in enumerate(ids[: len(drafter.layers)]):
        s, d = src[ti], drafter.layers[di]
        sd_s = dict(s.named_parameters())
        sd_d = dict(d.named_parameters())
        for name, p in sd_d.items():
            if name in sd_s and sd_s[name].shape == p.shape:
                p.data.copy_(sd_s[name].data.to(p.dtype))
