from torch import nn
from utils import inner_pad

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding = 4),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding = 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding = 2),
            nn.ReLU(),
            nn.Conv2d(32, 3, 5, padding = 2),
        )

    def forward(self, x):
        out = inner_pad(x - 0.5, 2)
        out = self.convs(out)
        return out