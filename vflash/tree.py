"""DDtree draft-tree decoding (inference only). Reuses the trained drafter; builds
a best-first tree over the block's per-position draft logits, verifies all nodes in
one target pass via a tree attention mask, then follows the longest accepted branch.

Ported/adapted from ddtree/ddtree.py (Ringel & Romano, "Accelerating Speculative
Decoding with Block Diffusion Draft Trees")."""
import heapq
from types import SimpleNamespace

import numpy as np
import torch
from transformers import DynamicCache

from .utils import sample, cuda_time
from .infer import _prefill, spec_generate, _truncate_stop, _Profiler


def build_tree(draft_logits, budget):
    """draft_logits [depth, V] -> (node_token_ids, node_depths, parents, child_maps, visibility)."""
    if budget <= 0 or draft_logits.shape[0] == 0:
        vis = torch.zeros((1, 1), dtype=torch.bool); vis[0, 0] = True
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                [-1], [dict()], vis)

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])
    logits = draft_logits.float()
    top_logits, top_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    logp = (top_logits - log_z).cpu().numpy()
    ids = top_ids.cpu().numpy()

    first = float(logp[0, 0])
    heap = [(-first, (0,), 0, 1, 0, first)]
    node_tokens = np.empty(budget, np.int64)
    node_depths = np.empty(budget, np.int64)
    parents = np.empty(budget + 1, np.int32); parents[0] = -1
    child_maps = [dict()]
    count = 0

    while heap and count < budget:
        _, ranks, parent, depth, rank, logw = heapq.heappop(heap)
        token = int(ids[depth - 1, rank])
        cur = count + 1
        node_tokens[count] = token
        node_depths[count] = depth
        parents[cur] = parent
        child_maps.append(dict())
        child_maps[parent][token] = cur
        count += 1
        if rank + 1 < topk:
            sib_w = logw - float(logp[depth - 1, rank]) + float(logp[depth - 1, rank + 1])
            heapq.heappush(heap, (-sib_w, ranks[:-1] + (rank + 1,), parent, depth, rank + 1, sib_w))
        if depth < depth_limit:
            child_w = logw + float(logp[depth, 0])
            heapq.heappush(heap, (-child_w, ranks + (0,), cur, depth + 1, 0, child_w))

    cur_len = 1 + count
    vis = np.zeros((cur_len, cur_len), np.bool_)
    vis[0, 0] = True
    for i in range(1, cur_len):
        p = int(parents[i])
        vis[i, :i] = vis[p, :i]
        vis[i, i] = True
    return (torch.from_numpy(node_tokens[:count]), torch.from_numpy(node_depths[:count]),
            parents[:cur_len].tolist(), child_maps, torch.from_numpy(vis))


def compile_tree(root_token, start, node_tokens, node_depths, visibility, past_len, dtype, device):
    cur = 1 + node_tokens.numel()
    ids = torch.empty((1, cur), dtype=torch.long, device=device)
    ids[0, 0] = root_token
    if cur > 1:
        ids[0, 1:] = node_tokens.to(device)
    pos = torch.empty((1, cur), dtype=torch.long, device=device)
    pos[0, 0] = start
    if cur > 1:
        pos[0, 1:] = node_depths.to(device) + start
    mask = torch.full((1, 1, cur, past_len + cur), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask[..., :past_len] = 0
    tree_block = mask[0, 0, :, past_len:past_len + cur]
    tree_block.masked_fill_(visibility.to(device), 0)
    return ids, pos, mask


def follow_tree(child_maps, posterior):
    toks = posterior[0].tolist()
    accepted = [0]
    cur = 0
    nxt = int(toks[0])
    while nxt in child_maps[cur]:
        cur = child_maps[cur][nxt]
        accepted.append(cur)
        nxt = int(toks[cur])
    return accepted, nxt


def compact_cache(cache, past_len, keep):
    if len(keep) == 0:
        cache.crop(past_len)
        return
    keep_t = torch.tensor(keep, dtype=torch.long)
    layers = getattr(cache, "layers", None)
    if layers and hasattr(layers[0], "keys"):
        for layer in layers:
            if layer.keys is None or layer.keys.numel() == 0:
                continue
            _compact(layer.keys, past_len, keep_t.to(layer.keys.device))
            _compact(layer.values, past_len, keep_t.to(layer.values.device))
    elif hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            kt = keep_t.to(cache.key_cache[i].device)
            _compact(cache.key_cache[i], past_len, kt)
            _compact(cache.value_cache[i], past_len, kt)
    cache.crop(past_len + len(keep))


def _compact(tensor, past_len, keep):
    cur = tensor.shape[-2] - past_len
    if cur <= 0 or keep.numel() == cur:
        return
    kept = tensor.narrow(-2, past_len, cur).index_select(-2, keep)
    tensor.narrow(-2, past_len, keep.numel()).copy_(kept)


@torch.inference_mode()
def ddtree_generate(drafter, target, input_ids, mm_inputs, max_new_tokens, block_size,
                    mask_token_id, stop_token_ids, temperature=0.0, tree_budget=None, profile=False):
    if block_size <= 1:
        return spec_generate(drafter, target, input_ids, mm_inputs, max_new_tokens,
                             block_size, mask_token_id, stop_token_ids, temperature, profile)
    drafter.eval()
    device = input_ids.device
    prof = _Profiler(profile and device.type == "cuda")
    S = input_ids.shape[1]
    max_len = S + max_new_tokens
    draft_horizon = block_size - 1
    budget = draft_horizon if tree_budget is None else max(int(tree_budget), 0)
    out_ids = torch.full((1, max_len + 1 + budget), mask_token_id, dtype=torch.long, device=device)
    out_ids[:, :S] = input_ids
    stop = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=device)

    ttft0 = cuda_time()
    first_token, context, ctx_pos, past_target = _prefill(drafter, target, input_ids, mm_inputs, temperature)
    out_ids[:, S] = first_token[0, 0]
    ttft = cuda_time() - ttft0
    draft_ctx_len = int(context.shape[1])

    past_draft = DynamicCache()
    ctx_len = 0
    pending_ctx, pending_pos = context, ctx_pos
    accept_lengths = []
    start = S
    decode0 = cuda_time()

    while start < max_len:
        block_ids = out_ids[:, start:start + block_size].clone()
        noise_emb = target.embed_tokens(block_ids)
        q_pos = torch.arange(start, start + block_size, device=device)
        k_pos = torch.cat([pending_pos, q_pos])[None]
        with prof.section("drafter"):
            hidden = drafter.run(noise_emb, pending_ctx, q_pos[None], k_pos, attn_mask=None,
                                 past_key_values=past_draft)
        ctx_len += pending_ctx.shape[1]
        past_draft.crop(ctx_len)
        draft_logits = target.lm_head(hidden[:, 1:])                 # [1,horizon,V]

        node_tokens, node_depths, parents, child_maps, visibility = build_tree(draft_logits[0], budget)
        verify_ids, verify_pos, mask = compile_tree(
            block_ids[0, 0], start, node_tokens, node_depths, visibility, start, target.dtype, device)

        with prof.section("target"):
            tout = target.forward(input_ids=verify_ids, position_ids=verify_pos, attention_mask=mask,
                                  past_key_values=past_target, use_cache=True, want_hidden=True)
        posterior = sample(tout.logits, temperature)
        accepted, next_token = follow_tree(child_maps, posterior)
        acc_t = torch.tensor(accepted, dtype=torch.long, device=device)
        accepted_tokens = verify_ids.index_select(1, acc_t)

        out_ids[:, start:start + len(accepted)] = accepted_tokens
        out_ids[:, start + len(accepted)] = next_token
        compact_cache(past_target, start, accepted)

        accepted_hidden = tout.target_hidden.index_select(1, acc_t)
        pending_ctx, _, _ = drafter.encode_context(accepted_hidden, None)
        pending_pos = torch.arange(start, start + len(accepted), device=device)

        accept_lengths.append(len(accepted))
        start += len(accepted)
        if stop is not None and torch.isin(out_ids[0, start - len(accepted):start + 1], stop).any():
            break

    decode_t = cuda_time() - decode0
    out_ids = out_ids[:, :max_len]
    out_ids = out_ids[:, out_ids[0] != mask_token_id]
    out_ids = _truncate_stop(out_ids, S, stop)
    n_out = out_ids.shape[1] - S
    timing = prof.summary()
    return SimpleNamespace(output_ids=out_ids.cpu(), num_output_tokens=n_out, ttft=ttft,
                           time_per_output_token=decode_t / max(n_out, 1),
                           acceptance_lengths=accept_lengths, n_blocks=len(accept_lengths),
                           draft_ctx_len=draft_ctx_len,
                           drafter_ms=timing["drafter"], target_ms=timing["target"])
