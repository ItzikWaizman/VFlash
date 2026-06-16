"""Ground setup: download target/draft models and a LLaVA-Video-178K subset,
then write a flat manifest JSONL that dataset.py consumes.

Run order: this is step 0. See README.

The LLaVA-Video-178K repo is sharded by academic source; each subset folder holds
annotation json(s) plus the referenced videos. We download one subset, parse its
annotations into {video, prompt, response} records, cap to --num-samples, and keep
only records whose video file exists locally.
"""
import argparse
import glob
import json
import os

from huggingface_hub import snapshot_download


def download_models(target, baseline, cache_dir):
    for repo in (target, baseline):
        if not repo:
            continue
        print(f"[setup] downloading model {repo}")
        snapshot_download(repo_id=repo, cache_dir=cache_dir,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"])


def download_dataset(subset, cache_dir, out_root):
    print(f"[setup] downloading LLaVA-Video-178K subset '{subset}'")
    local = snapshot_download(
        repo_id="lmms-lab/LLaVA-Video-178K", repo_type="dataset",
        cache_dir=cache_dir, allow_patterns=[f"{subset}/*"],
    )
    return os.path.join(local, subset)


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


def build_manifest(subset_dir, out_path, num_samples, exclude_ids):
    records = []
    for jf in sorted(glob.glob(os.path.join(subset_dir, "**", "*.json"), recursive=True)):
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
            vpath = os.path.join(subset_dir, vid)
            if not os.path.exists(vpath):
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
    subset_dir = download_dataset(subset, hf_cache, manifest)
    build_manifest(subset_dir, manifest, num_samples, exclude)


if __name__ == "__main__":
    main()
