"""Download complete video subsets from lmms-lab/LLaVA-Video-178K.

Unlike fetch_videos_hf.py (which streams through shards to pick individual files),
this script downloads whole subsets shard-by-shard: download one *.tar.gz, extract
all videos, delete the tar, repeat. Peak disk = extracted so far + one shard (~5 GB).

Run this on the remote cluster BEFORE running pregen. No GPU needed.

Example:
    HF_HOME=/scratch300/itzikwaizman/VflashData/hf_cache \\
    python scripts/fetch_subset_videos_hf.py \\
        --subsets 0_30_s_academic_v0_1,30_60_s_academic_v0_1,1_2_m_academic_v0_1,2_3_m_academic_v0_1 \\
        --data-path /scratch300/itzikwaizman/VflashData/data/cache/llava_video_178k
"""
import argparse
import os
import tarfile

from huggingface_hub import hf_hub_download, list_repo_files

DATASET_REPO = "lmms-lab/LLaVA-Video-178K"

ACADEMIC_SUBSETS = [
    "0_30_s_academic_v0_1",
    "30_60_s_academic_v0_1",
    "1_2_m_academic_v0_1",
    "2_3_m_academic_v0_1",
]


def _rm_blob(path):
    for p in {path, os.path.realpath(path)}:
        try:
            os.remove(p)
        except OSError:
            pass


def download_subset(subset, data_path, hf_cache):
    video_dir = os.path.join(data_path, "videos", subset)
    os.makedirs(video_dir, exist_ok=True)

    shards = sorted(
        f for f in list_repo_files(DATASET_REPO, repo_type="dataset")
        if f.startswith(subset + "/") and f.endswith(".tar.gz")
    )
    print(f"[fetch] {subset}: {len(shards)} shards -> {video_dir}", flush=True)

    for i, sf in enumerate(shards):
        done_marker = os.path.join(video_dir, "." + os.path.basename(sf) + ".done")
        if os.path.exists(done_marker):
            print(f"[fetch]   shard {i+1}/{len(shards)} already done, skipping", flush=True)
            continue

        print(f"[fetch]   shard {i+1}/{len(shards)}: downloading {sf} ...", flush=True)
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

        # Running disk usage
        used = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(video_dir) for f in fs
            if not f.startswith(".")
        )
        print(f"[fetch]   shard {i+1}/{len(shards)} done | subset on-disk: {used/1e9:.1f} GB",
              flush=True)

    print(f"[fetch] {subset}: complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subsets",
        default=",".join(ACADEMIC_SUBSETS),
        help="comma-separated subset names (default: all 4 academic subsets)",
    )
    ap.add_argument(
        "--data-path",
        default="data/cache/llava_video_178k",
        help="where to write videos/  (same value as data_path in baseline.json)",
    )
    ap.add_argument(
        "--hf-cache",
        default=os.environ.get("HF_HOME", "hf_cache"),
        help="HF hub cache dir (tars downloaded here, deleted after extraction)",
    )
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    print(f"[fetch] downloading {len(subsets)} subsets: {subsets}", flush=True)
    print(f"[fetch] data_path={args.data_path}  hf_cache={args.hf_cache}", flush=True)

    for sub in subsets:
        download_subset(sub, args.data_path, args.hf_cache)

    print("[fetch] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
