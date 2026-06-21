"""Pre-generate the target's own greedy responses and cache their token ids for
on-policy training. Embarrassingly parallel: each SLURM array task takes a strided
shard of the manifest, generates with HF (greedy, batched), and writes its own
<uid>.pt files into <output_dir>/gen_cache. Resumable (skips already-cached uids).

Only token ids are stored (~1KB/sample); hidden states / KL logits are recomputed
online during training. Run AFTER setup, BEFORE train. See README.

SLURM: sbatch --array=0-7 --gres=gpu:1 scripts/pregen.sh experiments/baseline.json
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset

from .dataset import VideoInstructDataset, record_uid
from ..train import GenCache, gen_cache_dir
from ..utils import load_config


def collate_list(batch):
    return [s for s in batch if s is not None]                     # drop failed-decode samples


def _to_device(v, device, dtype):
    if not torch.is_tensor(v):
        return v
    return v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)


def build_batch_inputs(processor, samples, device, dtype):
    """Left-padded batch of (video + prompt) for generation. Left padding keeps all
    rows right-aligned so generated tokens start at the same column for every row."""
    processor.tokenizer.padding_side = "left"
    msgs = [[{"role": "user", "content": [{"type": "video"},
                                          {"type": "text", "text": s["prompt"]}]}] for s in samples]
    texts = [processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
    enc = processor(text=texts, videos=[s["frames"] for s in samples],
                    padding=True, return_tensors="pt")
    return {k: _to_device(v, device, dtype) for k, v in enc.items()}


def greedy_generate(model, enc, max_new_tokens, eos_id):
    """enc holds left-padded inputs -> list of 1-D new-token tensors (eos-truncated)."""
    P = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, pad_token_id=eos_id)
    res = []
    for row in out[:, P:].cpu():
        hit = (row == eos_id).nonzero()
        res.append(row[: hit[0, 0] + 1] if hit.numel() else row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8,
                    help="background video-decode workers (overlaps decode with GPU generation)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = args.manifest or os.path.join(cfg["data_path"], "train.jsonl")
    if args.max_new_tokens:                                             # keep tag consistent with train
        cfg["max_new_tokens"] = args.max_new_tokens
    max_new = cfg["max_new_tokens"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    model = AutoVLM.from_pretrained(cfg["target_model"], torch_dtype=dtype,
                                    attn_implementation=cfg.get("attn_impl", "sdpa")).to(device).eval()
    processor = AutoProcessor.from_pretrained(cfg["target_model"])
    eos = processor.tokenizer.eos_token_id
    cache = GenCache(gen_cache_dir(cfg), enabled=True)

    ds = VideoInstructDataset(manifest, cfg["frames"])
    mine = list(range(args.shard, len(ds), args.num_shards))            # strided -> balanced shards
    todo = [i for i in mine if cache.get(record_uid(ds.records[i])) is None]
    print(f"[pregen] shard {args.shard}/{args.num_shards}: {len(todo)}/{len(mine)} to generate "
          f"(rest cached), batch={args.batch_size}, workers={args.num_workers}, "
          f"max_new={max_new}", flush=True)

    # Decode videos in background workers so CPU decode overlaps GPU generation
    # (serial main-thread decode otherwise idles the GPU ~50% on long videos).
    loader = DataLoader(Subset(ds, todo), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_list,
                        prefetch_factor=2 if args.num_workers > 0 else None,
                        persistent_workers=False)

    done = 0
    for samples in loader:
        if not samples:                                            # whole batch failed to decode
            continue
        try:
            enc = build_batch_inputs(processor, samples, device, dtype)
            outs = greedy_generate(model, enc, max_new, eos)
        except Exception as e:
            torch.cuda.empty_cache()                              # recover OOM before per-sample retry
            print(f"[pregen] batch at {done} failed ({e}); falling back to per-sample", flush=True)
            outs = []
            for s in samples:
                try:
                    enc = build_batch_inputs(processor, [s], device, dtype)
                    outs.append(greedy_generate(model, enc, max_new, eos)[0])
                except Exception as e2:
                    print(f"[pregen] sample {s['uid']} failed: {e2}", flush=True)
                    outs.append(None)
        for s, ids in zip(samples, outs):
            if ids is not None:
                cache.put(s["uid"], ids)
        done += len(samples)
        print(f"[pregen] {done}/{len(todo)}", flush=True)
    print(f"[pregen] shard {args.shard} done", flush=True)


if __name__ == "__main__":
    main()
