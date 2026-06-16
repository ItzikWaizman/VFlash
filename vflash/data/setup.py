"""Ground setup: download target/draft models and a LLaVA-Video-178K subset,
then write a flat manifest JSONL that dataset.py consumes.

Run order: this is step 0. See README.

The LLaVA-Video-178K repo is sharded by source; each subset folder holds annotation
json(s) plus videos packed as *.tar.gz shards. We download one subset, extract the
shards into videos/<subset>/ (deleting each tar after extraction to save disk), parse
the annotations into {video, prompt, response} records, cap to num_samples, and keep
only records whose video resolves on disk.
"""
import argparse
import json
import os
import tarfile

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

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


def build_manifest(json_paths, video_dir, out_path, num_samples, exclude_ids):
    idx = index_videos(video_dir)
    records = []
    for jf in sorted(json_paths):
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
                records.append({"video": vpath, "prompt": prompt, "response": response})
            if len(records) >= num_samples:
                break
        if len(records) >= num_samples:
            break
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[setup] wrote {len(records)} records -> {out_path}")


def main():
    import json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment .json")
    ap.add_argument("--models-only", action="store_true")
    ap.add_argument("--exclude-ids-file", default=None, help="newline-separated held-out video ids")
    args = ap.parse_args()
    cfg = _json.load(open(args.config))

    target = cfg["target_model"]
    baseline = cfg.get("baseline_draft_model")
    subset = cfg["data_subset"]
    num_samples = cfg["num_samples"]
    manifest = os.path.join(cfg["data_path"], "train.jsonl")
    hf_cache = os.environ.get("HF_HOME", "hf_cache")

    download_models(target, baseline, hf_cache)
    if args.models_only:
        return
    exclude = set()
    if args.exclude_ids_file and os.path.exists(args.exclude_ids_file):
        exclude = {l.strip() for l in open(args.exclude_ids_file) if l.strip()}
    json_paths, video_dir = download_dataset(subset, hf_cache, cfg["data_path"])
    build_manifest(json_paths, video_dir, manifest, num_samples, exclude)


if __name__ == "__main__":
    main()
