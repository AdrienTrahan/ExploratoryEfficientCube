from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 16),
        )

    def forward(self, states):
        if states.ndim == 2 and states.shape[-1] == 16:
            states = F.one_hot(states.long(), num_classes=16).flatten(start_dim=1)
        if states.ndim != 2 or states.shape[-1] != 16 * 16:
            raise ValueError("states must have shape (batch, 16) or (batch, 256)")
        return self.network(states.float())


class Eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(16, 1)

    def forward(self, embeddings):
        return self.output(embeddings).squeeze(-1)


class Jepa15PuzzleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.evaluator = Eval()

    def forward(self, successor_states):
        embeddings = self.encoder(successor_states)
        return self.evaluator(embeddings), embeddings
