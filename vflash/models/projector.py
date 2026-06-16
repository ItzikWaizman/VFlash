import torch.nn as nn
from .layers import RMSNorm


class Projector(nn.Module):
    """Map concatenated target hidden states (len(layer_ids)*D) -> drafter hidden D.

    Mirrors DFlash's fc + hidden_norm, but pulled out of the drafter so visual
    compression can run on already-projected, drafter-dimension features.
    """

    def __init__(self, num_layers_in, hidden_size, eps=1e-6):
        super().__init__()
        self.fc = nn.Linear(num_layers_in * hidden_size, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, target_hidden):
        # target_hidden [B,S,L*D] -> [B,S,D]
        return self.norm(self.fc(target_hidden))
