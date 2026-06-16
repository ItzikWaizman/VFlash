# VFlash

Block-diffusion speculative drafting for video LLMs. VFlash trains a small
DFlash-style drafter whose attention layers are conditioned (via KV injection) on
the target VLM's hidden states over the visual + prompt context, so it can propose
whole blocks of response tokens that the target verifies in one pass. The study
target is the expressivity (accepted length) vs. complexity (draft KV size / latency)
tradeoff for long video context.

Target/draft: `llava-onevision-qwen2-7b-ov-hf` (frozen) + a 4-layer warm-started
drafter. Data: `lmms-lab/LLaVA-Video-178K`. Online self-distillation (CE + KL).

Training is **on-policy**: the drafter is distilled over the target's *own* greedy
generations (matching lossless speculative-decoding eval), not the dataset ground
truth. Those generations are pre-computed once (see step 1) and cached as token ids;
hidden states and KL logits are always recomputed online (never stored).

## Two pluggable features (both behind knobs)
- `--visual-compress qformer` + `--num-queries N`: learnable queries cross-attend
  over the target's visual hidden states -> N position-less memory tokens that
  replace the full visual context injected into the drafter. Trained end-to-end.
- `--draft-tree 1` + `--tree-budget B`: DDtree decoding (inference only, no training
  change) — builds a best-first tree over the block's draft logits, verifies all
  nodes in one target pass, follows the longest accepted branch.

## Layout
```
experiments/baseline.json     full experiment config (none compression)
experiments/qformer.json      full experiment config (Q-Former compression)
vflash/
  models/target.py            frozen HF (V)LM wrapper: logits + selected hidden states
  models/drafter.py           KV-injection block-diffusion drafter + training masks
  models/projector.py         target-hidden (L*D) -> drafter-dim D
  models/compress.py          Q-Former visual compressor (none / qformer)
  losses.py                   fused CE + KL
  train.py                    online training loop (DDP, anchors, rolling checkpoint, ETA)
  infer.py                    speculative decoding (chain) + AR baseline
  tree.py                     DDtree draft-tree decoding
  monitor.py                  metrics.jsonl + live-overwritten plots
  data/setup.py               HF downloads + manifest builder
  data/pregen.py              batched greedy pre-generation -> gen_cache (SLURM array)
  data/dataset.py             video frames + manifest dataset
scripts/                      setup.sh, pregen.sh, train.sh, infer.sh, benchmark.sh
tests/smoke_test.py           CPU smoke test (local only, not tracked in git)
```

Each `scripts/*.sh` is both the local runner and the sbatch target. They take an
experiment config as the first argument (default `experiments/baseline.json`); all
hyperparameters live in that `.json`. New experiment = new json under `experiments/`.

## Run order

0. Setup (download models + a LLaVA-Video-178K subset, build manifest):
```
bash scripts/setup.sh experiments/baseline.json
```

1. Pre-generate the target's greedy responses into `gen_cache` (one GPU per shard;
   resumable, skips cached). Optional but strongly recommended — it removes the slow
   autoregressive generation from the training loop. Locally it runs one shard:
```
bash scripts/pregen.sh experiments/baseline.json
```
On SLURM, fan out as an array (e.g. 8 shards, 1 GPU each): see the SLURM section.
If skipped, training generates on-policy lazily on the first epoch and caches as it goes.

2. Train the drafter (multi-GPU; reads gen_cache, 100% cache hits after step 1):
```
bash scripts/train.sh experiments/baseline.json     # full visual injection
bash scripts/train.sh experiments/qformer.json      # Q-Former compression
```
Writes a single rolling checkpoint `outputs/<run>/drafter.pt`, `metrics.jsonl`, and
live plots under `outputs/<run>/plots/` (per-position accuracy, accepted length,
step timing, eval speedup). Also logs running average visual / prompt / response
token counts.

3. Evaluate speedup vs AR (chain or tree):
```
bash scripts/infer.sh experiments/baseline.json                       # chain SD
bash scripts/infer.sh experiments/baseline.json --draft-tree 1 --tree-budget 60   # DDtree
```

4. VideoDetailCaption benchmark (paper-comparable: reports M + speedup):
```
bash scripts/benchmark.sh experiments/baseline.json                   # chain
bash scripts/benchmark.sh experiments/baseline.json --draft-tree 1    # DDtree
bash scripts/benchmark.sh experiments/baseline.json --judge openai    # also compute M
```
Needs `benchmark_manifest` (one json/line: `{"video","prompt","response"}`, response = the
reference caption). Greedy SD is lossless vs AR, so M equals the target's score; the script
verifies this (lossless%) and saves predictions to `outputs/<run>/videodetailcaption.jsonl`.
For the exact paper metric, score that file with lmms-eval's `videodetailcaption` task, or
pass `--judge openai` (needs `OPENAI_API_KEY`) for a built-in GPT judge approximation.

## Key knobs (edit the experiment `.json`; a few are also CLI-overridable)
`visual_compress {none,qformer}` `num_queries` `draft_tree` `tree_budget`
`block_size` `num_anchors` `draft_layers` `inject_response_hidden`
`ce_weight` `kl_weight` `kl_temp` `kl_topk` `loss_decay_gamma`
`epochs` `lr` `frames` `max_length` `attn_backend {flex,sdpa}`
`context_source {target,gt}` `gen_cache` `max_new_tokens`

## SLURM (single-line submit; run from the cloned repo root)

Repo root on the cluster: `/scratch300/itzikwaizman/vflash/VFlash`.
First: `cd /scratch300/itzikwaizman/vflash/VFlash && git pull && mkdir -p output_logs`.

Setup (downloads):
```
sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=04:00:00 --cpus-per-task=2 --mem=16G -o output_logs/vflash_setup.out --job-name=vflash_setup --chdir /scratch300/itzikwaizman/vflash/VFlash ./scripts/setup.sh experiments/baseline.json
```
Pre-generation (array of 8 shards, 1 GPU each; `%a` = task id in the log name):
```
sbatch --array=0-7 -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=12:00:00 --cpus-per-task=4 --mem=32G -o output_logs/vflash_pregen_%a.out --job-name=vflash_pregen --chdir /scratch300/itzikwaizman/vflash/VFlash ./scripts/pregen.sh experiments/baseline.json
```
Training (8xA100):
```
sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:8 --time=48:00:00 --cpus-per-task=8 --mem=64G -o output_logs/vflash_train.out --job-name=vflash_train --chdir /scratch300/itzikwaizman/vflash/VFlash ./scripts/train.sh experiments/baseline.json
```
Evaluation:
```
sbatch -A gpu-tad-wolf_v2 -p gpu-tad-pool --qos=owner --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 --mem=32G -o output_logs/vflash_infer.out --job-name=vflash_infer --chdir /scratch300/itzikwaizman/vflash/VFlash ./scripts/infer.sh experiments/baseline.json
```

## Parallelization

Data-parallel via `torchrun --nproc_per_node 8` + PyTorch DDP:
- The frozen 7B target is replicated on each GPU (bf16, ~15GB; runs under `no_grad`).
  Each GPU produces its own target hidden states locally, so there is no cross-GPU
  communication for the target.
- The small drafter (4 layers + projector + optional Q-Former) is wrapped in DDP.
  Only its gradients are all-reduced each step.
- `DistributedSampler` gives each GPU different samples; with batch_size=1 and many
  anchors per sample, the effective batch is `8 x num_anchors` draft blocks/step.

Why DDP over FSDP here: the trainable drafter is tiny, and the frozen target fits in
80GB, so sharding buys little while DDP is simpler and scales near-linearly. (FSDP/
sequence-parallel can be revisited if we scale the target to 72B or push max_length.)

## Smoke test (CPU, no downloads)
```
PYTHONPATH=. python -m tests.smoke_test
```
Exercises the full core (training step, chain SD, DDtree SD, Q-Former compression,
monitor) on a tiny synthetic Qwen2 target.

## Notes
- On-policy: the only thing cached on disk is the target's generated token ids
  (~1KB/sample, under `data/cache/.../gen_cache/<target>_f<frames>_n<max_new>/`). The
  cache is keyed by target+frames, so baseline and qformer runs share it (pregen once).
  Set `context_source: "gt"` for dataset answers, or `gen_cache: false` to regenerate
  every epoch.
- Online training avoids the multi-GB/sample disk cost of storing 25K-token visual
  hidden states needed for offline KL.
- `flex` attention is used for the large-context training mask; `sdpa` is the
  CPU/smoke fallback. Inference always uses sdpa (short query blocks).
- `data/setup.py` parses one LLaVA-Video-178K subset; adjust `--data-subset` and the
  manifest builder if the chosen subset's annotation schema differs.
- Models are swappable: any HF causal (V)LM with standard decoder layers works as the
  target; the drafter warm-starts from its layers at `target_layer_ids`.
