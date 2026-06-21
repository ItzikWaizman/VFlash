import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Monitor:
    """Accumulates training/eval metrics, appends metrics.jsonl, and rewrites a
    fixed set of plots in place (each plot is cumulative up to the current step).

    Train series:  loss/ce/kl, grad_norm, per-position draft accuracy, train-est
                   accepted length, step time.
    Eval series:   chain & tree speedup, chain & tree mean accepted length, and a
                   latency breakdown (drafter vs target ms per block, drafter KV size).
    """

    def __init__(self, output_dir, block_size):
        self.dir = output_dir
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.block_size = block_size
        self.jsonl = open(os.path.join(output_dir, "metrics.jsonl"), "a")
        self.train_steps = []
        self.train = defaultdict(list)                               # key -> list aligned to train_steps
        self.pos_acc = []
        self.eval_steps = []
        self.eval = defaultdict(list)                               # key -> list aligned to eval_steps

    def log_train(self, step, rec):
        self.jsonl.write(json.dumps({"step": step, **rec}) + "\n")
        self.jsonl.flush()
        self.train_steps.append(step)
        self.pos_acc.append(rec.get("pos_acc", []))
        for k in ("loss", "ce", "kl", "accept_len", "fwd_ms", "grad_norm", "lr",
                  "avg_visual", "avg_prompt", "avg_response"):
            self.train[k].append(rec.get(k, float("nan")))

    def log_eval(self, step, rec):
        self.jsonl.write(json.dumps({"step": step, "eval": True, **rec}) + "\n")
        self.jsonl.flush()
        self.eval_steps.append(step)
        for k in ("speedup_chain", "speedup_tree", "accept_chain", "accept_tree",
                  "chain_drafter_ms", "chain_target_ms", "tree_drafter_ms", "tree_target_ms",
                  "ar_tps", "chain_tps", "tree_tps", "draft_ctx_len"):
            self.eval[k].append(rec.get(k, float("nan")))

    def plot(self):
        self._plot_pos_acc()
        self._line("loss.png", self.train_steps,
                   [("loss", self.train["loss"]), ("ce", self.train["ce"]), ("kl", self.train["kl"])],
                   "step", "loss", "Training loss")
        self._line("grad_norm.png", self.train_steps, [("grad_norm", self.train["grad_norm"])],
                   "step", "grad norm", "Gradient norm")
        self._line("accepted_length_train.png", self.train_steps,
                   [("train est.", self.train["accept_len"])],
                   "step", "mean accepted length", "Accepted length (train estimate)")
        self._line("timing.png", self.train_steps, [("ms/step", self.train["fwd_ms"])],
                   "step", "step time (ms)", "Train step time")
        self._line("token_composition.png", self.train_steps,
                   [("visual", self.train["avg_visual"]), ("prompt", self.train["avg_prompt"]),
                    ("response", self.train["avg_response"])],
                   "step", "tokens (running mean)", "Token composition per sample", logy=True)
        if self.eval_steps:
            self._line("speedup.png", self.eval_steps,
                       [("chain", self.eval["speedup_chain"]), ("tree", self.eval["speedup_tree"])],
                       "step", "speedup (x)", "Speedup vs AR (eval)", hline=1.0)
            self._line("accepted_length_eval.png", self.eval_steps,
                       [("chain", self.eval["accept_chain"]), ("tree", self.eval["accept_tree"])],
                       "step", "mean accepted length", "Accepted length (eval)")
            self._line("latency_breakdown.png", self.eval_steps,
                       [("chain drafter", self.eval["chain_drafter_ms"]),
                        ("chain target", self.eval["chain_target_ms"]),
                        ("tree drafter", self.eval["tree_drafter_ms"]),
                        ("tree target", self.eval["tree_target_ms"])],
                       "step", "ms / block", "Per-block latency breakdown (eval)")

    def _plot_pos_acc(self):
        if not self.pos_acc or not self.pos_acc[-1]:
            return
        n = len(self.pos_acc[-1])
        fig, ax = plt.subplots(figsize=(7, 4))
        for p in range(n):
            ys = [a[p] if p < len(a) else float("nan") for a in self.pos_acc]
            ax.plot(self.train_steps, ys, label=f"pos {p + 1}", linewidth=1)
        ax.set_xlabel("step"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
        ax.set_title("Per-position draft accuracy")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(self.plot_dir, "accuracy_per_position.png"), dpi=110)
        plt.close(fig)

    def _line(self, name, xs, series, xl, yl, title, hline=None, logy=False):
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, ys in series:
            ax.plot(xs, ys, marker=".", linewidth=1, label=label)
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=0.8)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        if len(series) > 1:
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(self.plot_dir, name), dpi=110)
        plt.close(fig)

    def close(self):
        self.jsonl.close()
