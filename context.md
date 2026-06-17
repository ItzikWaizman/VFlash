# VFlash — project context

Read this first if you're picking up the project cold. It's the "why" and the current
state; `README.md` is the "how to run" and `SERVER_SETUP.md` is the "how to stand up a
machine".

## What we're building

VFlash is a speculative-decoding method for video LLMs. The bottleneck we care about is
that video models drag a huge visual context (tens of thousands of tokens) through every
decode step, so the KV cache is enormous and generation is slow. Existing video SD work
(SpecVLM, ParallelVLM) is training-free and leans on pruning visual tokens for the
drafter. We're going the other way: a *trained* drafter in the spirit of DFlash.

The drafter is a small block-diffusion model. Instead of running its own forward pass
over the whole video, its attention layers are conditioned on the target VLM's hidden
states via KV injection — the target's hidden states (from a few selected layers,
projected down to the drafter's width) become the keys/values the drafter attends to.
The drafter then proposes a whole block of response tokens at once, and the target
verifies them in a single pass (standard speculative decoding accept/repair).

The research question — and the thing the monitoring is built around — is the tradeoff
between **expressivity** (higher average accepted length) and **complexity** (draft KV
size / per-block latency) as we change how much visual context the drafter gets. More
injected context should mean better guesses but slower blocks; we want the sweet spot.

## The papers this builds on (all PDFs are in the parent folder)

- **DFlash** — the core idea we're porting: block-diffusion drafter conditioned on target
  hidden states through KV injection. We adapt it from text to video.
- **SpecVLM / ParallelVLM** — the video SD baselines we compare against; they prune visual
  tokens and are training-free. Their benchmark suite is what we report on.
- **Fast-dVLM, Vamba, VideoNSA** — sources for *future* visual-compression variants of the
  drafter (Mamba over visual tokens, sparse attention). Not implemented yet.
- **"Accelerating Speculative Decoding with Block Diffusion Draft Trees" (ddtree)** — the
  tree-decoding (DDtree) we implemented as an inference-only knob.

## Key design decisions (don't silently undo these)

- **On-policy training.** The drafter is distilled against the target's *own* greedy
  generations, NOT the dataset ground-truth responses. This matches how lossless
  speculative decoding is evaluated. There was an earlier bug where training used GT
  responses — that was deliberately removed. `context_source: "target"` is the intended
  path; `"gt"` is only a fallback.
- **Nothing heavy is cached on disk.** We only cache the target's generated *token ids*
  (~1KB/sample, the `gen_cache`). Hidden states and KL logits are recomputed online every
  step — storing them would be many GB/sample. This is why training is online.
- **Single rolling checkpoint.** Disk is tight; we keep one `drafter.pt` and overwrite it,
  not per-epoch snapshots.
- **Loss is fused CE + KL** (weights `ce_weight`, `kl_weight`, plus `kl_temp`, `kl_topk`).
- **Warm start.** The drafter initializes from the target's layers at `target_layer_ids`.
- **Generic models.** Anything HF-causal-(V)LM with standard decoder layers should drop in
  as the target; nothing should hardcode LLaVA specifics beyond the data setup.

## Models & data

- Target: `llava-hf/llava-onevision-qwen2-7b-ov-hf` (frozen, bf16, ~15GB, Qwen2 backbone).
- Drafter: 4 layers, warm-started from the target.
- Train data: `lmms-lab/LLaVA-Video-178K`. The manifest builder prioritizes caption data,
  then open-ended QA, and drops multiple-choice (we want long responses — short ones don't
  exercise speedup).
- Eval/benchmark: VideoDetailCaption (reports M + speedup), with the broader
  SpecVLM/ParallelVLM suite (MVBench, MLVU, LongVideoBench, VideoMME) as the target set.

## The two variants behind knobs

1. **Visual compression** (`visual_compress`): `none` injects all visual hidden states;
   `qformer` cross-attends learnable queries over them to produce `num_queries`
   position-less memory tokens (one shared compressor for all drafter layers, trained
   end-to-end). This is the main expressivity/complexity lever. Future: Mamba, sparse attn.
2. **DDtree** (`draft_tree`, `tree_budget`): inference-only best-first draft tree, verified
   in one target pass. No training change.

## Code map (details in README "Layout")

`vflash/models/{target,drafter,projector,compress}.py`, `losses.py`, `train.py`,
`infer.py`, `tree.py`, `monitor.py`, `data/{setup,pregen,dataset}.py`. Configs live in
`experiments/*.json` (one file = one experiment; CLI can override a few knobs). The
monitor writes `metrics.jsonl` and live-overwritten plots (per-position accuracy, accepted
length, step timing, eval speedup, plus running token-count averages).

## Running it

- Pipeline order: setup -> pregen -> train -> infer -> benchmark. See `README.md`.
- Parallelism is DDP via `torchrun` (frozen target replicated per GPU, only the tiny
  drafter's grads all-reduced). FSDP/TP is the path if we ever scale the target to 72B.
- Two script sets: `scripts/*.sh` are the **cluster/SLURM** runners (they `source` the
  university env — do not repurpose them for other machines, and leave them as-is) and
  double as sbatch targets. `scripts/local/*.sh` are for a plain SSH box and assume you've
  already activated a conda/venv.

## Conventions

- Commit messages: `Title: <meaningful title>`, optional second line `Description: <...>`.
  Prefer several small commits over one giant one. Do not add AI/co-author trailers.
- Code style: compact, real tensor shapes in comments for non-obvious ops, no narration
  comments. Don't make it read like it was machine-generated.
- `tests/` (smoke test) is intentionally not tracked in git.
- Disk is the recurring constraint — think before writing anything large to disk.

## Status / where things stand

Core pipeline is implemented and smoke-tested. On-policy training, gen_cache (with a
SLURM-array pregen), Q-Former compression, DDtree, and the monitor are all in. The
attention masks and CE/KL alignment were audited. The immediate practical work is
environment/run plumbing on new servers and the first real training + benchmark runs to
get the expressivity-vs-complexity numbers.
