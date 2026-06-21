import os
import random
import time
from datetime import timedelta
import yaml
import numpy as np
import torch
import torch.distributed as dist


def load_config(path, overrides=None):
    """Load a yaml config, apply CLI overrides (dict of already-typed values)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def build_target_layer_ids(num_target_layers, num_draft_layers):
    """Spread draft layers across target depth (DFlash convention)."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + i * span / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def extract_context_feature(hidden_states, layer_ids):
    """hidden_states: tuple of (L+1) tensors [B,S,D] from output_hidden_states.
    Returns concat over selected layers -> [B, S, len(ids)*D]. offset=1 skips embeddings."""
    return torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)


def sample(logits, temperature=0.0):
    """logits [B,S,V] -> token ids [B,S]."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    b, s, v = logits.shape
    probs = torch.softmax(logits.view(-1, v) / temperature, dim=-1)
    return torch.multinomial(probs, 1).view(b, s)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return get_rank() == 0


def maybe_init_distributed():
    """Init process group from torchrun env vars; returns local_rank (or -1 if single-proc)."""
    if "RANK" not in os.environ:
        return -1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # long timeout: rank 0 can spend many minutes in eval while others wait at a barrier
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
    return local_rank


def cuda_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def fmt_eta(seconds):
    seconds = int(max(seconds, 0))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"
