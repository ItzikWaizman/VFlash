from contextlib import contextmanager
from types import SimpleNamespace

import torch
from transformers import DynamicCache

from .utils import sample, cuda_time


class _Profiler:
    """Records CUDA-event pairs around named sections with no in-loop syncs;
    one synchronize() at summary() turns them into milliseconds. No-op if off."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.events = {"drafter": [], "target": []}

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        yield
        b.record()
        self.events[name].append((a, b))

    def summary(self):
        if not self.enabled:
            return {"drafter": float("nan"), "target": float("nan")}
        torch.cuda.synchronize()
        return {k: sum(a.elapsed_time(b) for a, b in v) for k, v in self.events.items()}


def _prefill(drafter, target, input_ids, mm_inputs, temperature):
    """Run target prefill (with video), return first token + injected draft context."""
    past_target = DynamicCache()
    out = target.forward(input_ids=input_ids, past_key_values=past_target, use_cache=True,
                         want_hidden=True, logits_to_keep=1, **mm_inputs)
    first_token = sample(out.logits[:, -1:], temperature)            # [1,1]
    visual_mask = target.visual_mask(input_ids)
    context, n_mem, nonmem_idx = drafter.encode_context(out.target_hidden, visual_mask)
    ctx_pos = torch.cat([torch.zeros(n_mem, dtype=torch.long, device=context.device), nonmem_idx])
    return first_token, context, ctx_pos, past_target


@torch.inference_mode()
def spec_generate(drafter, target, input_ids, mm_inputs, max_new_tokens, block_size,
                  mask_token_id, stop_token_ids, temperature=0.0, profile=False):
    drafter.eval()
    device = input_ids.device
    prof = _Profiler(profile and device.type == "cuda")
    S = input_ids.shape[1]
    max_len = S + max_new_tokens
    out_ids = torch.full((1, max_len + block_size), mask_token_id, dtype=torch.long, device=device)
    out_ids[:, :S] = input_ids
    stop = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=device)

    ttft0 = cuda_time()
    first_token, context, ctx_pos, past_target = _prefill(drafter, target, input_ids, mm_inputs, temperature)
    out_ids[:, S] = first_token[0, 0]
    ttft = cuda_time() - ttft0
    draft_ctx_len = int(context.shape[1])                            # injected drafter KV (visual+prompt)

    past_draft = DynamicCache()
    ctx_len = 0
    pending_ctx, pending_pos = context, ctx_pos
    accept_lengths = []
    start = S
    decode0 = cuda_time()

    while start < max_len:
        block_ids = out_ids[:, start:start + block_size].clone()      # [1,bs] pos0 real, rest mask
        noise_emb = target.embed_tokens(block_ids)
        q_pos = torch.arange(start, start + block_size, device=device)
        k_pos = torch.cat([pending_pos, q_pos])[None]
        with prof.section("drafter"):
            hidden = drafter.run(noise_emb, pending_ctx, q_pos[None], k_pos, attn_mask=None,
                                 past_key_values=past_draft)
        ctx_len += pending_ctx.shape[1]
        past_draft.crop(ctx_len)                                      # drop noise keys, keep context
        draft_logits = target.lm_head(hidden[:, 1:])                  # [1,bs-1,V]
        block_ids[:, 1:] = sample(draft_logits, temperature)

        with prof.section("target"):
            tout = target.forward(input_ids=block_ids, position_ids=q_pos[None],
                                  past_key_values=past_target, use_cache=True, want_hidden=True)
        posterior = sample(tout.logits, temperature)                 # [1,bs]
        acc = (block_ids[:, 1:] == posterior[:, :-1]).cumprod(1).sum().item()

        out_ids[:, start:start + acc + 1] = block_ids[:, :acc + 1]
        out_ids[:, start + acc + 1] = posterior[:, acc]
        past_target.crop(start + acc + 1)

        accepted_hidden = tout.target_hidden[:, :acc + 1]
        pending_ctx, _, _ = drafter.encode_context(accepted_hidden, None)
        pending_pos = torch.arange(start, start + acc + 1, device=device)

        accept_lengths.append(acc + 1)
        start += acc + 1
        if stop is not None and torch.isin(out_ids[0, start - acc - 1:start + 1], stop).any():
            break

    decode_t = cuda_time() - decode0
    out_ids = out_ids[:, :min(start + 1, max_len)]
    out_ids = _truncate_stop(out_ids, S, stop)
    n_out = out_ids.shape[1] - S
    timing = prof.summary()
    return SimpleNamespace(output_ids=out_ids.cpu(), num_output_tokens=n_out, ttft=ttft,
                           time_per_output_token=decode_t / max(n_out, 1),
                           acceptance_lengths=accept_lengths, n_blocks=len(accept_lengths),
                           draft_ctx_len=draft_ctx_len,
                           drafter_ms=timing["drafter"], target_ms=timing["target"])


@torch.inference_mode()
def ar_generate(target, input_ids, mm_inputs, max_new_tokens, stop_token_ids, temperature=0.0):
    """Plain autoregressive baseline for speedup reference."""
    device = input_ids.device
    S = input_ids.shape[1]
    stop = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=device)
    past = DynamicCache()
    out = target.forward(input_ids=input_ids, past_key_values=past, use_cache=True,
                        want_hidden=False, logits_to_keep=1, **mm_inputs)
    token = sample(out.logits[:, -1:], temperature)
    gen = [token]
    decode0 = cuda_time()
    for i in range(max_new_tokens - 1):
        pos = torch.tensor([[S + i]], device=device)
        out = target.forward(input_ids=token, position_ids=pos, past_key_values=past,
                            use_cache=True, want_hidden=False)
        token = sample(out.logits[:, -1:], temperature)
        gen.append(token)
        if stop is not None and token.item() in stop_token_ids:
            break
    decode_t = cuda_time() - decode0
    gen_ids = torch.cat(gen, dim=1).cpu()
    return SimpleNamespace(num_output_tokens=len(gen), output_ids=gen_ids,
                           time_per_output_token=decode_t / max(len(gen), 1))


def _truncate_stop(out_ids, S, stop):
    if stop is not None:
        idx = torch.isin(out_ids[0][S:], stop).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            out_ids = out_ids[:, : S + idx[0] + 1]
    return out_ids


def generate(drafter, target, input_ids, mm_inputs, cfg):
    """Dispatch: chain SD or DDtree SD based on cfg['draft_tree']."""
    stop = cfg.get("stop_token_ids")
    if cfg.get("draft_tree"):
        from .tree import ddtree_generate
        return ddtree_generate(drafter, target, input_ids, mm_inputs,
                               cfg["max_new_tokens"], cfg["block_size"], cfg["mask_token_id"],
                               stop, cfg.get("temperature", 0.0), cfg.get("tree_budget"))
    return spec_generate(drafter, target, input_ids, mm_inputs, cfg["max_new_tokens"],
                         cfg["block_size"], cfg["mask_token_id"], stop, cfg.get("temperature", 0.0))
