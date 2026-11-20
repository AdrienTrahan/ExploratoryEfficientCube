import torch
from torch import nn
from torch.nn import functional as F


class FifteenPuzzleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, successor_states):
        encoded = F.one_hot(successor_states.long(), num_classes=16)
        encoded = encoded.flatten(start_dim=1).float()
        return self.network(encoded).squeeze(-1)

    def features(self, successor_states):
        encoded = F.one_hot(successor_states.long(), num_classes=16)
        encoded = encoded.flatten(start_dim=1).float()
        return self.network[:-1](encoded)
