import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch
import torch._dynamo
import torch.distributed as dist
# flex_attention is a higher-order op; DDP's dynamo graph-splitting optimizer cannot
# handle it (errors under multi-GPU only). Disable it -> single Dynamo bucket per graph.
torch._dynamo.config.optimize_ddp = False
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from .losses import fused_loss
from .monitor import Monitor
from .models.drafter import DrafterConfig, DFlashDrafter
from .models.target import HFTarget, warm_start_drafter
from .utils import (load_config, build_target_layer_ids, seed_everything, cuda_time,
                    maybe_init_distributed, is_dist, is_main, get_rank, get_world_size, fmt_eta)


# --------------------------------------------------------------------------
# Core (importable, target/drafter agnostic) -- exercised by the smoke test.
# --------------------------------------------------------------------------

def sample_anchors(loss_mask, block_size, num_anchors, device):
    """loss_mask [S] -> (anchors [N], keep [N]). Anchors are response positions."""
    S = loss_mask.shape[0]
    valid = (loss_mask[: max(S - block_size, 0) + 1] > 0.5).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return None, None
    perm = valid[torch.randperm(valid.numel(), device=device)][:num_anchors]
    anchors = perm.sort().values
    return anchors, torch.ones_like(anchors, dtype=torch.bool)


def build_noise_embedding(input_ids, anchors, block_size, mask_token_id, embed):
    # [1, N*bs, D]; first token of each block is the real anchor token, rest MASK.
    N = anchors.shape[0]
    ids = torch.full((N, block_size), mask_token_id, dtype=torch.long, device=input_ids.device)
    ids[:, 0] = input_ids[0, anchors]
    return embed(ids.view(1, N * block_size))


def compute_block_loss(drafter, draft_module, target, input_ids, target_hidden, last_hidden,
                       loss_mask, visual_mask, anchors, keep, cfg):
    bs = cfg["block_size"]
    device = input_ids.device
    S = input_ids.shape[1]
    N = anchors.shape[0]

    noise_emb = build_noise_embedding(input_ids, anchors, bs, cfg["mask_token_id"], target.embed_tokens)
    hidden = drafter(noise_embedding=noise_emb, target_hidden=target_hidden, visual_mask=visual_mask,
                     anchors=anchors, keep=keep, block_size=bs, attn_backend=cfg["attn_backend"])
    V = target.lm_head.weight.shape[0]
    draft_logits = target.lm_head(hidden).float().view(N, bs, V)

    offsets = torch.arange(bs, device=device)
    label_idx = anchors[:, None] + offsets[None, :]                  # [N,bs]
    valid = label_idx < S
    safe = label_idx.clamp(max=S - 1)
    target_ids = input_ids[0][safe]                                 # [N,bs]
    kl_idx = (label_idx - 1).clamp(min=0)
    kl_logits = target.lm_head(last_hidden[0][kl_idx].to(target.dtype)).float()   # [N,bs,V]

    pos = offsets[None, :].float()
    w = keep[:, None].float() * valid.float() * (offsets[None, :] > 0).float() * loss_mask[safe]
    if cfg["loss_decay_gamma"] and cfg["loss_decay_gamma"] > 0:
        w = w * torch.exp(-(pos - 1).clamp(min=0) / cfg["loss_decay_gamma"])

    loss, ce, kl = fused_loss(
        draft_logits.reshape(N * bs, V), target_ids.reshape(-1), kl_logits.reshape(N * bs, V),
        w.reshape(-1), cfg["ce_weight"], cfg["kl_weight"], cfg["kl_temp"], cfg["kl_topk"])

    with torch.no_grad():
        pred = draft_logits.argmax(-1)
        correct = (pred == target_ids) & (w > 0)
        pos_acc = []
        for p in range(1, bs):
            d = (w[:, p] > 0).sum()
            pos_acc.append((correct[:, p].sum() / d).item() if d > 0 else float("nan"))
        run = correct[:, 1:].long().cumprod(1).sum(1)
        accept_len = (run + 1).float().mean().item()
    return loss, dict(loss=loss.item(), ce=ce.item(), kl=kl.item(),
                      pos_acc=pos_acc, accept_len=accept_len)


# --------------------------------------------------------------------------
# VLM input building (processor-dependent; validated on cluster).
# --------------------------------------------------------------------------

def build_prompt_inputs(target, sample, device):
    """video + prompt only (the inference/eval starting point).
    Returns (input_ids [1,P], mm_inputs) where mm_inputs holds pixel_values_*."""
    proc = target.processor
    user = {"role": "user", "content": [{"type": "video"},
                                        {"type": "text", "text": sample["prompt"]}]}
    text = proc.apply_chat_template([user], add_generation_prompt=True, tokenize=False)
    enc = proc(text=text, videos=[sample["frames"]], return_tensors="pt")
    enc = {k: _to_device(v, device, target.dtype) for k, v in enc.items()}
    input_ids = enc.pop("input_ids")
    enc.pop("attention_mask", None)                                # bsz=1, no padding; length-agnostic mm only
    return input_ids, enc


def _to_device(v, device, dtype):
    if not torch.is_tensor(v):
        return v
    return v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)


def gen_cache_dir(cfg):
    """Cache the target's generations keyed by (target, frames, max_new_tokens) so all
    drafter variants on the same data reuse it. Override with cfg['gen_cache_dir']."""
    tag = (os.path.basename(cfg["target_model"].rstrip("/")) +
           f"_f{cfg['frames']}_n{cfg['max_new_tokens']}")
    root = cfg.get("gen_cache_dir") or os.path.join(cfg["data_path"], "gen_cache")
    return os.path.join(root, tag)


class GenCache:
    """Disk cache of the target's own generated token ids (tiny ints, ~1KB/sample).
    Shared across DDP ranks via atomic rename; hidden states/logits are never stored."""

    def __init__(self, root, enabled=True):
        self.dir = root
        self.enabled = enabled
        if enabled:
            os.makedirs(self.dir, exist_ok=True)

    def get(self, uid):
        p = os.path.join(self.dir, uid + ".pt")
        if self.enabled and os.path.exists(p):
            try:
                return torch.load(p, weights_only=True)               # plain tensor; avoids pickle FutureWarning
            except Exception:
                return None
        return None

    def put(self, uid, ids):
        if not self.enabled:
            return
        p = os.path.join(self.dir, uid + ".pt")
        tmp = f"{p}.tmp.{os.getpid()}"
        torch.save(ids.detach().cpu().clone(), tmp)               # clone: store only this sample's ids, not the shared batch buffer
        os.replace(tmp, p)


def build_context(target, sample, cfg, device, cache, stop):
    """Build the (full_ids, loss_mask, mm_inputs) the drafter trains over.
    On-policy (default): context = prompt + the target's own greedy generation, so
    training matches lossless speculative-decoding eval. 'gt' uses the dataset answer."""
    input_ids, mm = build_prompt_inputs(target, sample, device)
    P = input_ids.shape[1]
    if cfg.get("context_source", "target") == "gt":
        resp = target.processor.tokenizer(sample["response"], return_tensors="pt",
                                          add_special_tokens=False)["input_ids"][0].to(device)
    else:
        from .infer import ar_generate
        resp = cache.get(sample["uid"])                            # cached as 1-D [L]
        if resp is None:
            resp = ar_generate(target, input_ids, mm, cfg["max_new_tokens"], stop, 0.0).output_ids[0]
            cache.put(sample["uid"], resp)
        resp = resp.to(device)
    full_ids = torch.cat([input_ids, resp[None]], dim=1)           # [1, P+L]
    loss_mask = torch.zeros(full_ids.shape[1], device=device)
    loss_mask[P:] = 1.0
    if full_ids.shape[1] > cfg["max_length"]:
        full_ids = full_ids[:, :cfg["max_length"]]
        loss_mask = loss_mask[:cfg["max_length"]]
    return full_ids, loss_mask, mm


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def build_drafter(target, cfg, device, dtype, visual_compress=None, num_queries=None):
    ids = cfg["target_layer_ids"] or build_target_layer_ids(target.n_layers, cfg["draft_layers"])
    target.target_layer_ids = ids
    dc = target.drafter_cfg()
    dc.update(num_layers=cfg["draft_layers"], block_size=cfg["block_size"], target_layer_ids=ids,
              mask_token_id=cfg["mask_token_id"],
              visual_compress=cfg["visual_compress"] if visual_compress is None else visual_compress,
              num_queries=cfg["num_queries"] if num_queries is None else num_queries)
    drafter = DFlashDrafter(DrafterConfig(**dc)).to(device=device, dtype=dtype)
    warm_start_drafter(drafter, target)
    return drafter


def build_variants(cfg):
    """Variant specs to train together off ONE shared target forward. Each spec only
    differs in the drafter (visual compression); the target trajectory/hidden states
    are identical, so accept-length differences are purely the drafter design."""
    if cfg.get("variants"):
        return [dict(name=v["name"],
                     visual_compress=v.get("visual_compress", cfg["visual_compress"]),
                     num_queries=v.get("num_queries", cfg["num_queries"])) for v in cfg["variants"]]
    return [dict(name=cfg.get("run_name", "baseline"),
                 visual_compress=cfg["visual_compress"], num_queries=cfg["num_queries"])]


class ResumableSampler(Sampler):
    """Deterministic, world-size-agnostic epoch sampler with mid-epoch resume.

    A single GLOBAL order is derived from (base_seed, epoch) only -- NOT from the
    world size -- so the exact set of samples covered is identical no matter how many
    GPUs you launch. `consumed` = how many samples of that global order have already
    been trained this epoch; on resume we drop that prefix and round-robin the REST
    across the current ranks (truncating to an equal per-rank count so DDP stays in
    lockstep). => stop on 1 GPU mid-epoch, resume on N GPUs, with no repeats/skips.

    If num_samples < dataset size, a fixed base permutation selects the active subset
    (stable across epochs); each epoch reshuffles within it.
    """

    def __init__(self, n_total, n_active, world, rank, base_seed):
        self.n_total, self.n_active = n_total, n_active
        self.world, self.rank, self.base_seed = max(world, 1), rank, base_seed
        g = torch.Generator().manual_seed(base_seed)
        self.active = torch.randperm(n_total, generator=g)[:n_active].tolist()   # fixed subset
        self.set_epoch(0, 0)

    def epoch_order(self, epoch):
        g = torch.Generator().manual_seed(self.base_seed + 1 + epoch)
        perm = torch.randperm(self.n_active, generator=g).tolist()
        return [self.active[i] for i in perm]                                    # global indices

    def set_epoch(self, epoch, consumed=0):
        self.epoch, self.consumed = epoch, consumed
        remaining = self.epoch_order(epoch)[consumed:]
        self.per = len(remaining) // self.world                                  # equal per-rank
        self.indices = remaining[: self.per * self.world][self.rank :: self.world]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def save_checkpoint(path, variants, epoch, epoch_consumed, global_step, base_seed, world):
    """One rolling checkpoint with EVERYTHING needed to resume: all variants' weights
    + optimizer states, plus the epoch/consumed position so the covered-sample set is
    exactly reconstructable (epoch_order(epoch)[:epoch_consumed])."""
    ckpt = dict(epoch=epoch, epoch_consumed=epoch_consumed, global_step=global_step,
                base_seed=base_seed, world_size=world,
                variants={v["name"]: dict(model=v["module"].state_dict(),
                                          opt=v["opt"].state_dict()) for v in variants})
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    prog = dict(epoch=epoch, epoch_consumed=epoch_consumed, global_step=global_step,
                variants=[v["name"] for v in variants])
    with open(os.path.join(os.path.dirname(path), "progress.json"), "w") as f:
        json.dump(prog, f, indent=2)


def load_checkpoint(path, variants, map_location):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    for v in variants:
        st = ck["variants"].get(v["name"])
        if st is not None:
            v["module"].load_state_dict(st["model"])
            v["opt"].load_state_dict(st["opt"])
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    for k in ["target_model", "visual_compress", "attn_backend", "output_dir", "data_path"]:
        ap.add_argument(f"--{k.replace('_', '-')}")
    for k in ["draft_layers", "block_size", "num_queries", "num_samples", "frames", "max_length",
              "epochs", "num_anchors", "kl_topk", "log_interval", "eval_interval", "plot_interval",
              "ckpt_interval", "seed"]:
        ap.add_argument(f"--{k.replace('_', '-')}", type=int)
    for k in ["lr", "warmup_ratio", "max_grad_norm", "ce_weight", "kl_weight", "kl_temp",
              "loss_decay_gamma", "temperature"]:
        ap.add_argument(f"--{k.replace('_', '-')}", type=float)
    ap.add_argument("--draft-tree", type=int)
    ap.add_argument("--inject-response-hidden", type=int)
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}

    cfg = load_config(args.config, overrides)
    local_rank = maybe_init_distributed()
    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    seed_everything(cfg["seed"] + max(local_rank, 0))

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    model = AutoVLM.from_pretrained(cfg["target_model"], torch_dtype=dtype,
                                    attn_implementation=cfg.get("attn_impl", "sdpa")).to(device)
    processor = AutoProcessor.from_pretrained(cfg["target_model"])
    target = HFTarget(model, processor)
    if cfg["mask_token_id"] is None:
        tok = processor.tokenizer
        cfg["mask_token_id"] = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    world, rank = get_world_size(), get_rank()
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # One drafter per variant, each trained off the SAME shared target forward.
    variants = []
    for spec in build_variants(cfg):
        module = build_drafter(target, cfg, device, dtype,
                               visual_compress=spec["visual_compress"], num_queries=spec["num_queries"])
        ddp = DDP(module, device_ids=[local_rank], find_unused_parameters=False) if is_dist() else module
        opt = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=cfg["lr"])
        vdir = os.path.join(cfg["output_dir"], spec["name"])
        variants.append(dict(name=spec["name"], drafter=ddp, module=module, opt=opt, vdir=vdir,
                             monitor=Monitor(vdir, cfg["block_size"]) if is_main() else None))
        if is_main():
            n_par = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"[variant] {spec['name']}: visual_compress={spec['visual_compress']} "
                  f"num_queries={spec['num_queries']} ({n_par/1e6:.1f}M trainable)")

    from .data.dataset import VideoInstructDataset, collate_passthrough
    manifest = cfg.get("train_manifest") or os.path.join(cfg["data_path"], "train.jsonl")
    ds = VideoInstructDataset(manifest, cfg["frames"])
    n_total = len(ds)
    n_active = min(cfg.get("num_samples") or n_total, n_total)
    sampler = ResumableSampler(n_total, n_active, world, rank, cfg["seed"])
    loader = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=False,
                        num_workers=cfg.get("num_workers", 6), collate_fn=collate_passthrough,
                        persistent_workers=True, prefetch_factor=4)
    steps_per_epoch = n_active // max(world, 1)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup = int(cfg["warmup_ratio"] * total_steps)
    if is_main():
        print(f"[data] {n_active}/{n_total} active samples | {steps_per_epoch} steps/epoch "
              f"| {cfg['epochs']} epochs -> {total_steps} steps | world={world}")

    def lr_at(s):
        if s < warmup:
            return s / max(warmup, 1)
        prog = (s - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    # Fixed held-out eval set (decoded once on rank 0): identical samples every eval
    # so speedup/accept curves are comparable across steps and across runs.
    eval_set = []
    if is_main():
        eval_path = os.path.join(cfg["data_path"], "eval.jsonl")
        if os.path.exists(eval_path):
            eval_ds = VideoInstructDataset(eval_path, cfg["frames"])
            for i in range(min(cfg.get("eval_samples", 4), len(eval_ds))):
                s = eval_ds[i]
                if s is not None:
                    eval_set.append(s)
            print(f"[eval] fixed held-out eval set: {len(eval_set)} samples")
        else:
            print(f"[eval] WARNING: no eval.jsonl at {eval_path}; eval disabled")

    cache = GenCache(gen_cache_dir(cfg), enabled=cfg.get("gen_cache", True))
    stop = [processor.tokenizer.eos_token_id]
    ckpt_path = os.path.join(cfg["output_dir"], "checkpoint.pt")
    ckpt_interval = cfg.get("ckpt_interval", 100)

    # Auto-resume: copy output_dir (+ gen_cache) to any machine, relaunch -> continues
    # from the exact same epoch/position with no repeated samples (any world size).
    start_epoch, start_consumed, step = 0, 0, 0
    if os.path.exists(ckpt_path):
        ck = load_checkpoint(ckpt_path, variants, device)
        start_epoch, start_consumed, step = ck["epoch"], ck["epoch_consumed"], ck["global_step"]
        if is_main():
            print(f"[resume] from {ckpt_path}: epoch {start_epoch}, consumed {start_consumed}, "
                  f"global step {step}")

    ema_step_t = None
    t_start = time.time()
    tok_sum = {"visual": 0, "prompt": 0, "response": 0, "n": 0,
               "resp_min": 10**9, "resp_max": 0, "prompt_min": 10**9, "prompt_max": 0}
    last_good = None

    for epoch in range(start_epoch, cfg["epochs"]):
        consumed0 = start_consumed if epoch == start_epoch else 0
        sampler.set_epoch(epoch, consumed0)
        epoch_consumed = consumed0
        for sample in loader:
            if sample is None:                                     # decode failed: reuse last good
                if last_good is None:                              # (keeps DDP ranks in lockstep)
                    continue
                sample = last_good
            t0 = cuda_time()
            try:
                input_ids, loss_mask, mm = build_context(target, sample, cfg, device, cache, stop)
            except Exception as e:
                if last_good is None:
                    if is_main():
                        print(f"[skip] sample failed: {e}")
                    continue
                sample = last_good
                input_ids, loss_mask, mm = build_context(target, sample, cfg, device, cache, stop)
            last_good = sample
            anchors, keep = sample_anchors(loss_mask, cfg["block_size"], cfg["num_anchors"], device)
            if anchors is None:
                anchors, keep = sample_anchors(loss_mask, cfg["block_size"], 1, device)
                if anchors is None:
                    epoch_consumed += world
                    step += 1
                    continue
            out = target.forward(want_hidden=True, logits_to_keep=1, input_ids=input_ids, **mm)
            target_hidden = out.target_hidden.clone()              # shared across all variants
            last_hidden = out.last_hidden.clone()
            visual_mask = target.visual_mask(input_ids)

            n_vis = int(visual_mask.sum())
            n_resp = int(loss_mask.sum())
            n_prompt = input_ids.shape[1] - n_vis - n_resp
            tok_sum["visual"] += n_vis
            tok_sum["response"] += n_resp
            tok_sum["prompt"] += n_prompt
            tok_sum["n"] += 1
            tok_sum["resp_min"] = min(tok_sum["resp_min"], n_resp)
            tok_sum["resp_max"] = max(tok_sum["resp_max"], n_resp)
            tok_sum["prompt_min"] = min(tok_sum["prompt_min"], n_prompt)
            tok_sum["prompt_max"] = max(tok_sum["prompt_max"], n_prompt)

            cur_lr = cfg["lr"] * lr_at(step)
            metrics = {}
            for v in variants:                                     # each variant: own loss/grad/step
                loss, m = compute_block_loss(v["drafter"], v["module"], target, input_ids,
                                             target_hidden, last_hidden, loss_mask, visual_mask,
                                             anchors, keep, cfg)
                for g in v["opt"].param_groups:
                    g["lr"] = cur_lr
                v["opt"].zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(v["module"].parameters(), cfg["max_grad_norm"])
                v["opt"].step()
                m["grad_norm"] = float(gnorm)
                metrics[v["name"]] = m

            step_t = cuda_time() - t0
            ema_step_t = step_t if ema_step_t is None else 0.9 * ema_step_t + 0.1 * step_t
            step += 1
            epoch_consumed += world

            n = max(tok_sum["n"], 1)
            avg_tok = {f"avg_{k}": tok_sum[k] / n for k in ("visual", "prompt", "response")}

            if is_main() and step % cfg["log_interval"] == 0:
                eta = fmt_eta(ema_step_t * (total_steps - step))
                for v in variants:
                    m = metrics[v["name"]]
                    m["fwd_ms"] = step_t * 1000
                    m["lr"] = cur_lr
                    m.update(avg_tok)
                    v["monitor"].log_train(step, m)
                parts = " | ".join(f"{v['name']}: loss {metrics[v['name']]['loss']:.3f} "
                                   f"acc {metrics[v['name']]['accept_len']:.2f}" for v in variants)
                print(f"step {step}/{total_steps} | {parts} | {step_t*1000:.0f}ms/step ETA {eta}")
            if is_main() and step % cfg.get("token_stat_interval", 50) == 0:
                print(f"[tokens] step {step} | visual {n_vis} (avg {avg_tok['avg_visual']:.0f}) "
                      f"| prompt {n_prompt} (avg {avg_tok['avg_prompt']:.1f} "
                      f"min {tok_sum['prompt_min']} max {tok_sum['prompt_max']}) "
                      f"| response {n_resp} (avg {avg_tok['avg_response']:.1f} "
                      f"min {tok_sum['resp_min']} max {tok_sum['resp_max']})")
            if is_main() and step % cfg["plot_interval"] == 0:
                for v in variants:
                    v["monitor"].plot()
            if is_main() and step % ckpt_interval == 0:
                save_checkpoint(ckpt_path, variants, epoch, epoch_consumed, step, cfg["seed"], world)
            if step % cfg["eval_interval"] == 0:
                _run_eval(variants, target, eval_set, cfg, device, step)   # only rank 0 does work
                if is_main():
                    _save(variants, cfg, step)
                if is_dist():
                    dist.barrier()                                         # others wait out rank-0 eval

    if is_main():
        _save(variants, cfg, step)
        save_checkpoint(ckpt_path, variants, cfg["epochs"], 0, step, cfg["seed"], world)
        for v in variants:
            v["monitor"].plot()
            v["monitor"].close()
        print(f"done in {fmt_eta(time.time() - t_start)}")


def _mean(xs):
    xs = [x for x in xs if x == x]                                   # drop NaNs
    return sum(xs) / len(xs) if xs else float("nan")


def _run_eval(variants, target, eval_set, cfg, device, step):
    """Eval EACH variant on a fixed held-out set with three decoders:
      - AR (target only)          : speedup reference (computed once, shared)
      - chain SD (vanilla)        : 1 draft path / block
      - tree SD (DDtree)          : best-first draft tree / block (inference-only)
    Logs per-variant speedup, mean accepted length (vanilla + tree), and a latency
    breakdown (drafter vs target ms/block, drafter KV ctx length) so the
    accuracy<->latency tradeoff between the two drafter designs is directly comparable.
    Only rank 0 holds the eval_set; other ranks skip."""
    from .infer import spec_generate, ar_generate
    from .tree import ddtree_generate
    if not is_main() or not eval_set:
        for v in variants:
            v["module"].train()
        return
    stop = [target.processor.tokenizer.eos_token_id]
    for v in variants:
        v["module"].eval()
    aggs = {v["name"]: defaultdict(list) for v in variants}
    for sample in eval_set:
        try:
            input_ids, mm = build_prompt_inputs(target, sample, device)
            ar = ar_generate(target, input_ids, mm, cfg["max_new_tokens"], stop, cfg["temperature"])
            ar_tpot = ar.time_per_output_token
        except Exception as e:
            print(f"[eval] AR failed: {e}")
            continue
        for v in variants:
            try:
                ch = spec_generate(v["module"], target, input_ids, mm, cfg["max_new_tokens"],
                                   cfg["block_size"], cfg["mask_token_id"], stop, cfg["temperature"], profile=True)
                tr = ddtree_generate(v["module"], target, input_ids, mm, cfg["max_new_tokens"],
                                     cfg["block_size"], cfg["mask_token_id"], stop, cfg["temperature"],
                                     cfg.get("tree_budget"), profile=True)
                agg = aggs[v["name"]]
                agg["ar_tps"].append(1 / ar_tpot)
                agg["chain_tps"].append(1 / ch.time_per_output_token)
                agg["tree_tps"].append(1 / tr.time_per_output_token)
                agg["speedup_chain"].append(ar_tpot / max(ch.time_per_output_token, 1e-9))
                agg["speedup_tree"].append(ar_tpot / max(tr.time_per_output_token, 1e-9))
                agg["accept_chain"].append(_mean(ch.acceptance_lengths))
                agg["accept_tree"].append(_mean(tr.acceptance_lengths))
                agg["chain_drafter_ms"].append(ch.drafter_ms / max(ch.n_blocks, 1))
                agg["chain_target_ms"].append(ch.target_ms / max(ch.n_blocks, 1))
                agg["tree_drafter_ms"].append(tr.drafter_ms / max(tr.n_blocks, 1))
                agg["tree_target_ms"].append(tr.target_ms / max(tr.n_blocks, 1))
                agg["draft_ctx_len"].append(ch.draft_ctx_len)
            except Exception as e:
                print(f"[eval][{v['name']}] sample failed: {e}")
    for v in variants:
        agg = aggs[v["name"]]
        if agg.get("speedup_chain"):
            rec = {k: _mean(val) for k, val in agg.items()}
            v["monitor"].log_eval(step, rec)
            print(f"[eval][{v['name']}] step {step} | chain {rec['speedup_chain']:.2f}x "
                  f"(acc {rec['accept_chain']:.2f}) | tree {rec['speedup_tree']:.2f}x "
                  f"(acc {rec['accept_tree']:.2f}) | drafter/blk {rec['chain_drafter_ms']:.1f}ms "
                  f"target/blk {rec['chain_target_ms']:.1f}ms | draftKV {rec['draft_ctx_len']:.0f} tok")
        v["module"].train()


def _save(variants, cfg, step):
    for v in variants:
        os.makedirs(v["vdir"], exist_ok=True)
        torch.save({"step": step, "cfg": cfg, "name": v["name"], "state_dict": v["module"].state_dict()},
                   os.path.join(v["vdir"], "drafter.pt"))


if __name__ == "__main__":
    main()
