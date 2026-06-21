import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset


def load_video_frames(path, num_frames):
    """Uniformly sample num_frames RGB frames from a video. Returns list[PIL.Image]."""
    import av
    from PIL import Image

    container = av.open(path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    if total <= 0:
        frames = [f.to_image() for f in container.decode(video=0)]
        container.close()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        return [frames[i] for i in idx]

    idx = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    out = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in idx:
            out.append(frame.to_image())
        if len(out) == len(idx):
            break
    container.close()
    while len(out) < num_frames:               # pad short clips
        out.append(out[-1] if out else Image.new("RGB", (336, 336)))
    return out


def record_uid(rec):
    """Stable id from (video, prompt) -> used as the generation-cache key. Computable
    from the manifest record alone (no video decode), so pregen can skip cached samples."""
    return hashlib.md5((rec["video"] + "||" + rec["prompt"]).encode()).hexdigest()[:16]


class VideoInstructDataset(Dataset):
    """Reads a manifest JSONL of {video, prompt, response}; yields raw samples.
    Tokenization/processor application happens in the train loop (needs the VLM
    processor), keeping this dataset processor-agnostic."""

    def __init__(self, manifest, num_frames):
        self.records = [json.loads(l) for l in open(manifest)]
        self.num_frames = num_frames

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        try:
            frames = load_video_frames(r["video"], self.num_frames)
        except Exception as e:                                   # corrupt/unreadable video -> skip
            import sys
            print(f"[dataset] decode failed {r['video']}: {e}", file=sys.stderr, flush=True)
            return None
        return {
            "frames": frames,
            "prompt": r["prompt"],
            "response": r["response"],
            "video": r["video"],
            "uid": record_uid(r),
        }


def collate_passthrough(batch):
    return batch[0]   # bsz=1 online training; one sample per step (may be None -> caller skips)
