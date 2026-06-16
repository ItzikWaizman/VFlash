import argparse
import os

import torch

from .infer import generate, ar_generate
from .models.drafter import DrafterConfig, DFlashDrafter
from .models.target import HFTarget
from .utils import load_config


def load_drafter(ckpt_path, target, device, dtype):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["cfg"]
    target.target_layer_ids = cfg["target_layer_ids"]
    dc = target.drafter_cfg()
    dc.update(num_layers=cfg["draft_layers"], block_size=cfg["block_size"],
              target_layer_ids=cfg["target_layer_ids"], mask_token_id=cfg["mask_token_id"],
              visual_compress=cfg["visual_compress"], num_queries=cfg["num_queries"])
    drafter = DFlashDrafter(DrafterConfig(**dc)).to(device=device, dtype=dtype)
    drafter.load_state_dict(ck["state_dict"])
    return drafter.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--draft-tree", type=int, default=None)
    ap.add_argument("--tree-budget", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.draft_tree is not None:
        cfg["draft_tree"] = bool(args.draft_tree)
    if args.tree_budget is not None:
        cfg["tree_budget"] = args.tree_budget
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    checkpoint = args.checkpoint or os.path.join(cfg["output_dir"], "drafter.pt")
    manifest = args.manifest or cfg["eval_manifest"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    model = AutoVLM.from_pretrained(cfg["target_model"], torch_dtype=dtype,
                                    attn_implementation=cfg.get("attn_impl", "sdpa")).to(device)
    target = HFTarget(model, AutoProcessor.from_pretrained(cfg["target_model"]))
    drafter, dcfg = load_drafter(checkpoint, target, device, dtype)
    cfg["mask_token_id"] = dcfg["mask_token_id"]
    cfg["stop_token_ids"] = [target.processor.tokenizer.eos_token_id]

    from .train import build_prompt_inputs
    from .data.dataset import VideoInstructDataset
    ds = VideoInstructDataset(manifest, cfg["frames"])

    speedups, accepts, sd_tps = [], [], []
    for i in range(min(args.num_eval, len(ds))):
        input_ids, mm = build_prompt_inputs(target, ds[i], device)
        sd = generate(drafter, target, input_ids, mm, cfg)
        ar = ar_generate(target, input_ids, mm, cfg["max_new_tokens"],
                         cfg["stop_token_ids"], cfg["temperature"])
        speedups.append(ar.time_per_output_token / max(sd.time_per_output_token, 1e-9))
        accepts.append(sum(sd.acceptance_lengths) / max(len(sd.acceptance_lengths), 1))
        sd_tps.append(1 / sd.time_per_output_token)

    n = len(speedups)
    mode = "tree" if cfg.get("draft_tree") else "chain"
    print(f"[eval:{mode}] n={n} speedup={sum(speedups)/n:.2f}x "
          f"mean_accept={sum(accepts)/n:.2f} sd_tps={sum(sd_tps)/n:.1f}")


if __name__ == "__main__":
    main()
