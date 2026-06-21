"""Overlay metrics from two (or more) training runs so the accuracy<->latency
tradeoff between the straight-forward drafter and the Q-Former-compressed drafter
is directly comparable.

Usage:
    python -m vflash.plot_compare --runs outputs/baseline outputs/qformer \
        --labels baseline qformer --out outputs/compare
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir):
    """Read metrics.jsonl into separate train/eval step-aligned series."""
    train, ev = {"step": []}, {"step": []}
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return train, ev
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tgt = ev if r.get("eval") else train
            tgt["step"].append(r.get("step"))
            for k, v in r.items():
                if k in ("step", "eval") or isinstance(v, (list, dict)):
                    continue
                tgt.setdefault(k, [])
                while len(tgt[k]) < len(tgt["step"]) - 1:           # pad missing keys with NaN
                    tgt[k].append(float("nan"))
                tgt[k].append(v)
    for tgt in (train, ev):
        n = len(tgt["step"])
        for k in tgt:
            while len(tgt[k]) < n:
                tgt[k].append(float("nan"))
    return train, ev


def panel(out_dir, name, runs, source, key, ylabel, title, hline=None):
    has = any(key in runs[lbl][source] for lbl in runs)
    if not has:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl in runs:
        d = runs[lbl][source]
        if key in d:
            ax.plot(d["step"], d[key], marker=".", linewidth=1, label=lbl)
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("step"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, name), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", default="outputs/compare")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    os.makedirs(args.out, exist_ok=True)

    runs = {}
    for lbl, run in zip(labels, args.runs):
        tr, ev = load(run)
        runs[lbl] = {"train": tr, "eval": ev}

    panel(args.out, "cmp_speedup_chain.png", runs, "eval", "speedup_chain",
          "speedup (x)", "Chain SD speedup vs AR", hline=1.0)
    panel(args.out, "cmp_speedup_tree.png", runs, "eval", "speedup_tree",
          "speedup (x)", "Tree SD speedup vs AR", hline=1.0)
    panel(args.out, "cmp_accept_chain.png", runs, "eval", "accept_chain",
          "mean accepted length", "Accepted length (chain, eval)")
    panel(args.out, "cmp_accept_tree.png", runs, "eval", "accept_tree",
          "mean accepted length", "Accepted length (tree, eval)")
    panel(args.out, "cmp_drafter_ms.png", runs, "eval", "chain_drafter_ms",
          "ms / block", "Drafter latency per block (eval)")
    panel(args.out, "cmp_draft_ctx_len.png", runs, "eval", "draft_ctx_len",
          "tokens", "Drafter KV context length")
    panel(args.out, "cmp_loss.png", runs, "train", "loss", "loss", "Training loss")
    panel(args.out, "cmp_accept_train.png", runs, "train", "accept_len",
          "mean accepted length", "Accepted length (train estimate)")
    panel(args.out, "cmp_avg_response.png", runs, "train", "avg_response",
          "tokens (running mean)", "Response tokens per sample")
    print(f"[compare] wrote plots to {args.out}")


if __name__ == "__main__":
    main()
