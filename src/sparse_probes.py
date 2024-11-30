import torch
from torch import nn

class ClassificationProbe(nn.Module):

    def __init__(self, in_dim, device):
        super().__init__()

        self.linear = nn.Linear(
            in_features=in_dim,
            out_features=1,
            device=device
        )

    def forward(self, x):

        return torch.sigmoid(self.linear(x))