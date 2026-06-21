"""Download the 7B target model + selected video subsets from HF.

Run this on the remote cluster BEFORE pregen. No GPU needed.
After this finishes, symlink VFlash/data -> VFlash_data so relative
paths in the manifests resolve correctly, then sbatch pregen.

Cluster usage (run from ~/VFlash):

    # download model + 4 academic video subsets (no symlinks needed)
    python scripts/fetch_subset_videos_hf.py

    # then submit pregen
    sbatch scripts/pregen_remote.sh remote_pregen/remote_academic.jsonl

Everything lands under /scratch300/itzikwaizman/VFlash/:
    hf_cache/                            <- 7B model weights (~15 GB)
    data/cache/llava_video_178k/videos/  <- extracted videos (~300 GB)

Disk budget:
    7B model  : ~15 GB
    4 academic : ~249 GB compressed -> ~300 GB extracted
    Peak extra : one shard at a time (~5 GB) -> total peak ~320 GB
"""
import argparse
import os
import tarfile

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

DATASET_REPO = "lmms-lab/LLaVA-Video-178K"
TARGET_MODEL  = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

ACADEMIC_SUBSETS = [
    "0_30_s_academic_v0_1",
    "30_60_s_academic_v0_1",
    "1_2_m_academic_v0_1",
    "2_3_m_academic_v0_1",
]

# Defaults matched to the cluster layout the user described
VFLASH_ROOT = "/scratch300/itzikwaizman/VFlash"
DEFAULT_HF_CACHE = f"{VFLASH_ROOT}/hf_cache"        # model weights
DEFAULT_DATA_PATH = f"{VFLASH_ROOT}/data/cache/llava_video_178k"  # videos


def _rm_blob(path):
    for p in {path, os.path.realpath(path)}:
        try:
            os.remove(p)
        except OSError:
            pass


def download_model(hf_cache):
    print(f"\n[fetch] === downloading target model {TARGET_MODEL} ===", flush=True)
    snapshot_download(
        repo_id=TARGET_MODEL,
        cache_dir=hf_cache,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
    )
    used = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(hf_cache) for f in fs
    )
    print(f"[fetch] model done | hf_cache on-disk: {used/1e9:.1f} GB", flush=True)


def download_subset(subset, data_path, hf_cache):
    video_dir = os.path.join(data_path, "videos", subset)
    os.makedirs(video_dir, exist_ok=True)

    shards = sorted(
        f for f in list_repo_files(DATASET_REPO, repo_type="dataset")
        if f.startswith(subset + "/") and f.endswith(".tar.gz")
    )
    print(f"\n[fetch] === {subset}: {len(shards)} shards -> {video_dir} ===", flush=True)

    for i, sf in enumerate(shards):
        done_marker = os.path.join(video_dir, "." + os.path.basename(sf) + ".done")
        if os.path.exists(done_marker):
            print(f"[fetch]   shard {i+1}/{len(shards)} already done, skipping", flush=True)
            continue

        print(f"[fetch]   shard {i+1}/{len(shards)}: downloading {os.path.basename(sf)} ...", flush=True)
        try:
            tar_path = hf_hub_download(
                DATASET_REPO, sf, repo_type="dataset", cache_dir=hf_cache
            )
        except Exception as e:
            print(f"[fetch]   download failed: {e}", flush=True)
            continue

        print(f"[fetch]   extracting ...", flush=True)
        try:
            with tarfile.open(tar_path, "r:gz") as t:
                t.extractall(video_dir)
        except Exception as e:
            print(f"[fetch]   extract failed: {e}", flush=True)
            _rm_blob(tar_path)
            continue

        _rm_blob(tar_path)
        open(done_marker, "w").close()

        used = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(video_dir) for f in fs
            if not f.startswith(".")
        )
        print(f"[fetch]   shard {i+1}/{len(shards)} done | subset on-disk: {used/1e9:.1f} GB", flush=True)

    print(f"[fetch] {subset}: complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default=",".join(ACADEMIC_SUBSETS),
                    help="comma-separated subset names (default: 4 academic subsets)")
    ap.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                    help="where to write videos/ (default: data/cache/llava_video_178k, "
                         "resolves via symlink VFlash/data -> VFlash_data)")
    ap.add_argument("--hf-cache", default=DEFAULT_HF_CACHE,
                    help=f"HF hub cache dir for model weights and shard blobs (default: {DEFAULT_HF_CACHE})")
    ap.add_argument("--skip-model", action="store_true",
                    help="skip model download (if already present in hf-cache)")
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    print(f"[fetch] hf_cache : {args.hf_cache}", flush=True)
    print(f"[fetch] data_path: {args.data_path}", flush=True)
    print(f"[fetch] subsets  : {subsets}", flush=True)

    if not args.skip_model:
        download_model(args.hf_cache)

    for sub in subsets:
        download_subset(sub, args.data_path, args.hf_cache)

    print("\n[fetch] ALL DONE", flush=True)
    print("[fetch] Next step: sbatch scripts/pregen_remote.sh remote_pregen/remote_academic.jsonl", flush=True)


if __name__ == "__main__":
    main()
