import argparse
import math
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .losses import fused_loss
from .monitor import Monitor
from .models.drafter import DrafterConfig, DFlashDrafter
from .models.target import HFTarget, warm_start_drafter
from .utils import (load_config, build_target_layer_ids, seed_everything, cuda_time,
                    maybe_init_distributed, is_dist, is_main, get_world_size, fmt_eta)


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
                return torch.load(p)
            except Exception:
                return None
        return None

    def put(self, uid, ids):
        if not self.enabled:
            return
        p = os.path.join(self.dir, uid + ".pt")
        tmp = f"{p}.tmp.{os.getpid()}"
        torch.save(ids.cpu(), tmp)
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

def build_drafter(target, cfg, device, dtype):
    ids = cfg["target_layer_ids"] or build_target_layer_ids(target.n_layers, cfg["draft_layers"])
    target.target_layer_ids = ids
    dc = target.drafter_cfg()
    dc.update(num_layers=cfg["draft_layers"], block_size=cfg["block_size"], target_layer_ids=ids,
              mask_token_id=cfg["mask_token_id"], visual_compress=cfg["visual_compress"],
              num_queries=cfg["num_queries"])
    drafter = DFlashDrafter(DrafterConfig(**dc)).to(device=device, dtype=dtype)
    warm_start_drafter(drafter, target)
    return drafter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    for k in ["target_model", "visual_compress", "attn_backend", "output_dir", "data_path"]:
        ap.add_argument(f"--{k.replace('_', '-')}")
    for k in ["draft_layers", "block_size", "num_queries", "num_samples", "frames", "max_length",
              "epochs", "num_anchors", "kl_topk", "log_interval", "eval_interval", "plot_interval", "seed"]:
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

    drafter = build_drafter(target, cfg, device, dtype)
    draft_module = drafter
    if is_dist():
        drafter = DDP(drafter, device_ids=[local_rank], find_unused_parameters=False)

    from .data.dataset import VideoInstructDataset, collate_passthrough
    ds = VideoInstructDataset(os.path.join(cfg["data_path"], "train.jsonl"), cfg["frames"])
    sampler = DistributedSampler(ds) if is_dist() else None
    loader = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=sampler is None,
                        num_workers=2, collate_fn=collate_passthrough)

    opt = torch.optim.AdamW([p for p in draft_module.parameters() if p.requires_grad], lr=cfg["lr"])
    total_steps = cfg["epochs"] * len(loader)
    warmup = int(cfg["warmup_ratio"] * total_steps)

    def lr_at(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    monitor = Monitor(cfg["output_dir"], cfg["block_size"]) if is_main() else None
    os.makedirs(cfg["output_dir"], exist_ok=True)
    cache = GenCache(gen_cache_dir(cfg), enabled=cfg.get("gen_cache", True))
    stop = [processor.tokenizer.eos_token_id]
    step = 0
    ema_step_t = None
    t_start = time.time()
    tok_sum = {"visual": 0, "prompt": 0, "response": 0, "n": 0}

    for epoch in range(cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for sample in loader:
            t0 = cuda_time()
            try:
                input_ids, loss_mask, mm = build_context(target, sample, cfg, device, cache, stop)
            except Exception as e:
                if is_main():
                    print(f"[skip] sample failed: {e}")
                continue
            anchors, keep = sample_anchors(loss_mask, cfg["block_size"], cfg["num_anchors"], device)
            if anchors is None:
                continue
            out = target.forward(want_hidden=True, logits_to_keep=1, input_ids=input_ids, **mm)
            target_hidden = out.target_hidden.clone()
            last_hidden = out.last_hidden.clone()
            visual_mask = target.visual_mask(input_ids)

            n_vis = int(visual_mask.sum())
            n_resp = int(loss_mask.sum())
            tok_sum["visual"] += n_vis
            tok_sum["response"] += n_resp
            tok_sum["prompt"] += input_ids.shape[1] - n_vis - n_resp
            tok_sum["n"] += 1

            loss, m = compute_block_loss(drafter, draft_module, target, input_ids, target_hidden,
                                         last_hidden, loss_mask, visual_mask, anchors, keep, cfg)
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * lr_at(step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft_module.parameters(), cfg["max_grad_norm"])
            opt.step()

            step_t = cuda_time() - t0
            ema_step_t = step_t if ema_step_t is None else 0.9 * ema_step_t + 0.1 * step_t
            step += 1

            n = max(tok_sum["n"], 1)
            avg_tok = {f"avg_{k}": tok_sum[k] / n for k in ("visual", "prompt", "response")}

            if is_main() and step % cfg["log_interval"] == 0:
                eta = fmt_eta(ema_step_t * (total_steps - step))
                m["fwd_ms"] = step_t * 1000
                m.update(avg_tok)
                monitor.log_train(step, m)
                print(f"step {step}/{total_steps} loss {m['loss']:.3f} ce {m['ce']:.3f} "
                      f"kl {m['kl']:.3f} acc_len {m['accept_len']:.2f} "
                      f"{step_t * 1000:.0f}ms/step ETA {eta}")
            if is_main() and step % cfg.get("token_stat_interval", 50) == 0:
                print(f"[tokens] step {step} avg visual {avg_tok['avg_visual']:.0f} "
                      f"prompt {avg_tok['avg_prompt']:.0f} response {avg_tok['avg_response']:.0f}")
            if is_main() and step % cfg["plot_interval"] == 0:
                monitor.plot()
            if is_main() and step % cfg["eval_interval"] == 0:
                _run_eval(draft_module, target, sample, cfg, device, monitor, step)
                _save(draft_module, cfg, step)

    if is_main():
        _save(draft_module, cfg, step)
        monitor.plot()
        monitor.close()
        print(f"done in {fmt_eta(time.time() - t_start)}")


def _run_eval(draft_module, target, sample, cfg, device, monitor, step):
    from .infer import spec_generate, ar_generate
    try:
        input_ids, mm = build_prompt_inputs(target, sample, device)
        stop = [target.processor.tokenizer.eos_token_id]
        sd = spec_generate(draft_module, target, input_ids, mm, cfg["max_new_tokens"],
                           cfg["block_size"], cfg["mask_token_id"], stop, cfg["temperature"])
        ar = ar_generate(target, input_ids, mm, cfg["max_new_tokens"], stop, cfg["temperature"])
        speedup = ar.time_per_output_token / max(sd.time_per_output_token, 1e-9)
        monitor.log_eval(step, dict(speedup=speedup,
                                    sd_tps=1 / sd.time_per_output_token,
                                    ar_tps=1 / ar.time_per_output_token,
                                    mean_accept=sum(sd.acceptance_lengths) / max(len(sd.acceptance_lengths), 1)))
        print(f"[eval] step {step} speedup {speedup:.2f}x")
    except Exception as e:
        print(f"[eval] failed: {e}")


def _save(draft_module, cfg, step):
    path = os.path.join(cfg["output_dir"], "drafter.pt")
    torch.save({"step": step, "cfg": cfg, "state_dict": draft_module.state_dict()}, path)


if __name__ == "__main__":
    main()
