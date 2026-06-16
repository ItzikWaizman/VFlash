"""VideoDetailCaption benchmark: generate captions with speculative decoding (chain or
DDtree) and the AR baseline, then report:
  - speedup (AR time-per-token / SD time-per-token)
  - mean accepted length and SD tokens/s
  - lossless% (greedy SD output token-identical to AR -> caption quality is the target's)
  - M: the caption-quality score (avg GPT judge over samples), if a judge is enabled

Greedy SD (temperature=0) is provably lossless vs AR greedy, so M(SD)=M(AR)=M(target);
the value of this script is confirming that and measuring the speedup at fixed quality.

Predictions are also written to a jsonl so M can be scored exactly with the official
lmms-eval `videodetailcaption` task if you prefer that over the built-in judge.

Manifest format (one json/line): {"video": path, "prompt": question, "response": reference}.
"""
import argparse
import json
import os

import torch

from .infer import generate, ar_generate
from .evaluate import load_drafter
from .models.target import HFTarget
from .utils import load_config


def judge_openai(question, reference, prediction, model):
    """Returns a 1-5 detail/correctness score from an OpenAI judge (needs OPENAI_API_KEY)."""
    from openai import OpenAI
    client = OpenAI()
    sys = ("You evaluate video detail captions. Given the question, a reference answer, and a "
           "prediction, rate how well the prediction matches the reference in correctness and "
           "detail. Reply with a single integer 1-5 (5=best).")
    user = f"Question: {question}\nReference: {reference}\nPrediction: {prediction}\nScore (1-5):"
    r = client.chat.completions.create(model=model, temperature=0,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}])
    txt = r.choices[0].message.content.strip()
    return float(next(c for c in txt if c.isdigit()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--manifest", default=None, help="VideoDetailCaption manifest jsonl")
    ap.add_argument("--num-eval", type=int, default=100)
    ap.add_argument("--draft-tree", type=int, default=None)
    ap.add_argument("--tree-budget", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=None, help="predictions jsonl (default outputs/<run>/videodetailcaption.jsonl)")
    ap.add_argument("--judge", choices=["none", "openai"], default="none")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.draft_tree is not None:
        cfg["draft_tree"] = bool(args.draft_tree)
    if args.tree_budget is not None:
        cfg["tree_budget"] = args.tree_budget
    cfg["max_new_tokens"] = args.max_new_tokens
    checkpoint = args.checkpoint or os.path.join(cfg["output_dir"], "drafter.pt")
    manifest = args.manifest or cfg.get("benchmark_manifest", "data/cache/videodetailcaption/test.jsonl")
    out_path = args.out or os.path.join(cfg["output_dir"], "videodetailcaption.jsonl")

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
    cfg["temperature"] = 0.0                                   # greedy => lossless, paper protocol
    cfg["stop_token_ids"] = [target.processor.tokenizer.eos_token_id]
    tok = target.processor.tokenizer

    from .train import build_target_inputs
    from .data.dataset import VideoInstructDataset
    ds = VideoInstructDataset(manifest, cfg["frames"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fout = open(out_path, "w")
    speedups, accepts, sd_tps, lossless, scores = [], [], [], [], []

    for i in range(min(args.num_eval, len(ds))):
        sample = ds[i]
        inputs, _ = build_target_inputs(target, sample, cfg["max_length"], device)
        input_ids = inputs.pop("input_ids")
        S = input_ids.shape[1]
        sd = generate(drafter, target, input_ids, inputs, cfg)
        ar = ar_generate(target, input_ids, inputs, cfg["max_new_tokens"],
                         cfg["stop_token_ids"], 0.0)

        sd_resp = sd.output_ids[0, S:]
        n = min(sd_resp.numel(), ar.output_ids.shape[1])
        lossless.append(bool(torch.equal(sd_resp[:n], ar.output_ids[0, :n])))
        speedups.append(ar.time_per_output_token / max(sd.time_per_output_token, 1e-9))
        accepts.append(sum(sd.acceptance_lengths) / max(len(sd.acceptance_lengths), 1))
        sd_tps.append(1 / sd.time_per_output_token)

        prediction = tok.decode(sd_resp, skip_special_tokens=True)
        fout.write(json.dumps({"video": sample["video"], "question": sample["prompt"],
                               "prediction": prediction, "reference": sample["response"]}) + "\n")
        if args.judge == "openai":
            try:
                scores.append(judge_openai(sample["prompt"], sample["response"], prediction, args.judge_model))
            except Exception as e:
                print(f"[judge] sample {i} failed: {e}")
    fout.close()

    k = len(speedups)
    avg = lambda x: sum(x) / max(len(x), 1)
    mode = "tree" if cfg.get("draft_tree") else "chain"
    print(f"\n=== VideoDetailCaption [{mode}] n={k} ===")
    print(f"speedup        {avg(speedups):.2f}x")
    print(f"mean_accept    {avg(accepts):.2f}")
    print(f"sd_tokens/s    {avg(sd_tps):.1f}")
    print(f"lossless       {100*avg([float(x) for x in lossless]):.1f}%")
    if scores:
        print(f"M (judge avg)  {avg(scores):.3f}  (judge={args.judge_model})")
    else:
        print(f"M              not scored; run lmms-eval videodetailcaption on {out_path}")


if __name__ == "__main__":
    main()
