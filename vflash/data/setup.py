"""Ground setup: download target/draft models and one or more LLaVA-Video-178K
subsets, then write flat manifest JSONLs (train + held-out eval) that dataset.py
consumes.

Run order: this is step 0. See README.

The LLaVA-Video-178K repo is sharded by source; each subset folder holds annotation
json(s) plus videos packed as *.tar.gz shards. For every requested subset we download
its annotation json(s) + video shards, extract the shards into videos/<subset>/
(deleting each tar after extraction to save disk), then parse the annotations into
{video, prompt, response} records. Multiple-choice (mc) annotations are dropped (we
want long responses); captions are prioritized over open-ended QA. Records are deduped
by uid (md5(video||prompt)), a deterministic held-out eval split is carved off, and the
rest is capped to num_samples for train.jsonl.
"""
import argparse
import json
import os
import random
import tarfile

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from .dataset import record_uid

DATASET_REPO = "lmms-lab/LLaVA-Video-178K"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def download_models(target, baseline, cache_dir):
    for repo in (target, baseline):
        if not repo:
            continue
        print(f"[setup] downloading model {repo}")
        snapshot_download(repo_id=repo, cache_dir=cache_dir,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"])


def download_dataset(subset, cache_dir, data_path):
    """Fetch the subset's annotation json(s) (small) and video shards (*.tar.gz),
    extracting each shard into videos/<subset>/ then deleting the tar to save disk.
    Returns (json_paths, video_dir)."""
    files = [f for f in list_repo_files(DATASET_REPO, repo_type="dataset")
             if f.startswith(subset + "/")]
    json_files = [f for f in files if f.endswith(".json")]
    tar_files = sorted(f for f in files if f.endswith(".tar.gz"))

    json_paths = [hf_hub_download(DATASET_REPO, f, repo_type="dataset", cache_dir=cache_dir)
                  for f in json_files]

    video_dir = os.path.join(data_path, "videos", subset)
    os.makedirs(video_dir, exist_ok=True)
    for i, f in enumerate(tar_files):
        done = os.path.join(video_dir, "." + os.path.basename(f) + ".done")
        if os.path.exists(done):
            continue
        print(f"[setup] ({i + 1}/{len(tar_files)}) download+extract {os.path.basename(f)}")
        # Reuse the HF cache so partial downloads resume; delete the blob after
        # extraction to reclaim disk (peak extra space stays ~1 shard).
        tar_path = hf_hub_download(DATASET_REPO, f, repo_type="dataset", cache_dir=cache_dir)
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(video_dir)
        for p in {tar_path, os.path.realpath(tar_path)}:
            try:
                os.remove(p)
            except OSError:
                pass
        open(done, "w").close()
    return json_paths, video_dir


def parse_conversations(conv):
    """[{from:human,value:..}, {from:gpt,value:..}, ...] -> first (prompt, response)."""
    prompt = response = None
    for turn in conv:
        role, val = turn.get("from"), turn.get("value", "")
        if role == "human" and prompt is None:
            prompt = val.replace("<image>", "").replace("<video>", "").strip()
        elif role == "gpt" and prompt is not None and response is None:
            response = val.strip()
            break
    return prompt, response


def index_videos(video_dir):
    """basename -> absolute path, so manifest resolution is robust to tar layout."""
    idx = {}
    for root, _, names in os.walk(video_dir):
        for n in names:
            if n.lower().endswith(VIDEO_EXTS):
                idx.setdefault(n, os.path.join(root, n))
    return idx


def _ann_priority(path):
    """captions (long, caption-benchmark style) first, then open-ended QA."""
    return 0 if "cap" in os.path.basename(path).lower() else 1


def _is_mc(path):
    """multiple-choice annotations -> short answers; dropped (we want long responses)."""
    return "_mc" in os.path.basename(path).lower() or "mc_" in os.path.basename(path).lower()


def collect_records(json_paths, video_dir, exclude_ids, captions_only=False):
    """Parse one subset's annotations into resolved {video,prompt,response} records,
    captions first. With captions_only=True, keep only caption (long-response)
    annotations and drop open-ended QA. Returns a list; dedup/cap/split happen in caller."""
    idx = index_videos(video_dir)
    out = []
    files = [p for p in json_paths if not _is_mc(p)]
    if captions_only:
        files = [p for p in files if _ann_priority(p) == 0]
    for jf in sorted(files, key=_ann_priority):
        try:
            data = json.load(open(jf))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            vid = item.get("video") or item.get("video_path")
            conv = item.get("conversations")
            if not vid or not conv:
                continue
            vpath = os.path.join(video_dir, vid)
            if not os.path.exists(vpath):
                vpath = idx.get(os.path.basename(vid))
            if not vpath or not os.path.exists(vpath):
                continue
            if os.path.splitext(os.path.basename(vid))[0] in exclude_ids:
                continue
            prompt, response = parse_conversations(conv)
            if prompt and response:
                out.append({"video": vpath, "prompt": prompt, "response": response})
    return out


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def split_and_write(records, train_path, eval_path, num_samples, eval_holdout, seed):
    """Dedup by uid, carve off a random held-out eval split (disjoint), then take up to
    num_samples for train with captions prioritized. Deterministic given seed."""
    seen, uniq = set(), []
    for r in records:
        u = record_uid(r)
        if u in seen:
            continue
        seen.add(u)
        uniq.append(r)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_eval = min(eval_holdout, max(len(uniq) - num_samples, 0)) if num_samples else min(eval_holdout, len(uniq))
    if n_eval == 0 and eval_holdout:
        n_eval = min(eval_holdout, len(uniq))                      # tiny pool: still hold some out
    eval_recs = uniq[:n_eval]
    pool = uniq[n_eval:]
    pool.sort(key=lambda r: 0 if len(r["response"]) >= 200 else 1)  # caption-like (long) first
    train_recs = pool[:num_samples] if num_samples else pool
    _write_jsonl(train_path, train_recs)
    if eval_holdout:
        _write_jsonl(eval_path, eval_recs)
    print(f"[setup] {len(uniq)} unique records -> train {len(train_recs)} ({train_path}), "
          f"eval {len(eval_recs)} ({eval_path})")


def main():
    import json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment .json")
    ap.add_argument("--models-only", action="store_true")
    ap.add_argument("--subsets", default=None, help="comma-separated subsets (overrides config)")
    ap.add_argument("--exclude-ids-file", default=None, help="newline-separated held-out video ids")
    args = ap.parse_args()
    cfg = _json.load(open(args.config))

    target = cfg["target_model"]
    baseline = cfg.get("baseline_draft_model")
    if args.subsets:
        subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    else:
        subsets = cfg.get("data_subsets") or [cfg["data_subset"]]
    num_samples = cfg["num_samples"]
    eval_holdout = cfg.get("eval_holdout", 0)
    train_path = os.path.join(cfg["data_path"], "train.jsonl")
    eval_path = cfg.get("eval_manifest") or os.path.join(cfg["data_path"], "eval.jsonl")
    hf_cache = os.environ.get("HF_HOME", "hf_cache")

    download_models(target, baseline, hf_cache)
    if args.models_only:
        return
    exclude = set()
    if args.exclude_ids_file and os.path.exists(args.exclude_ids_file):
        exclude = {l.strip() for l in open(args.exclude_ids_file) if l.strip()}

    records = []
    for subset in subsets:
        print(f"[setup] === subset {subset} ===")
        json_paths, video_dir = download_dataset(subset, hf_cache, cfg["data_path"])
        sub = collect_records(json_paths, video_dir, exclude, cfg.get("captions_only", False))
        print(f"[setup] {subset}: {len(sub)} resolved records")
        records.extend(sub)
    split_and_write(records, train_path, eval_path, num_samples, eval_holdout, cfg.get("seed", 0))


if __name__ == "__main__":
    main()
