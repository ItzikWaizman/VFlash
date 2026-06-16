import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Monitor:
    """Accumulates training/eval metrics, appends metrics.jsonl, and rewrites a
    fixed set of plots in place (each plot is cumulative up to the current step)."""

    def __init__(self, output_dir, block_size):
        self.dir = output_dir
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.block_size = block_size
        self.jsonl = open(os.path.join(output_dir, "metrics.jsonl"), "a")
        self.train_steps, self.pos_acc, self.accept, self.fwd_ms = [], [], [], []
        self.eval_steps, self.speedup, self.sd_tps, self.ar_tps = [], [], [], []

    def log_train(self, step, rec):
        rec = {"step": step, **rec}
        self.jsonl.write(json.dumps(rec) + "\n")
        self.jsonl.flush()
        self.train_steps.append(step)
        self.pos_acc.append(rec.get("pos_acc", []))
        self.accept.append(rec.get("accept_len", float("nan")))
        self.fwd_ms.append(rec.get("fwd_ms", float("nan")))

    def log_eval(self, step, rec):
        self.jsonl.write(json.dumps({"step": step, "eval": True, **rec}) + "\n")
        self.jsonl.flush()
        self.eval_steps.append(step)
        self.speedup.append(rec.get("speedup", float("nan")))
        self.sd_tps.append(rec.get("sd_tps", float("nan")))
        self.ar_tps.append(rec.get("ar_tps", float("nan")))

    def plot(self):
        self._plot_pos_acc()
        self._line("accepted_length.png", self.train_steps, self.accept,
                   "step", "mean accepted length", "Accepted length (train est.)")
        self._line("timing.png", self.train_steps, self.fwd_ms,
                   "step", "forward time (ms)", "Train step time")
        if self.eval_steps:
            self._line("speedup.png", self.eval_steps, self.speedup,
                       "step", "speedup (x)", "SD vs AR speedup")

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

    def _line(self, name, xs, ys, xl, yl, title):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker=".", linewidth=1)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        fig.tight_layout(); fig.savefig(os.path.join(self.plot_dir, name), dpi=110)
        plt.close(fig)

    def close(self):
        self.jsonl.close()
