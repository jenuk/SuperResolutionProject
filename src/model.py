import torch
from torch import nn

from layers import Residual

class Model(nn.Module):
    def __init__(self, upscaling_factor):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=upscaling_factor, mode="nearest")

        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
        )

        self.features = nn.Sequential(
            *(Residual(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
            ) for _ in range(4)),
            nn.Conv2d(32, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.upscale = nn.Sequential(
            *(nn.Sequential(
                nn.ConvTranspose2d(128, 128, 3, 2, padding=1, output_padding=1),
                nn.ReLU(),
            ) for _ in range(upscaling_factor.bit_length()-1))
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(128+3, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, x):
        up = self.upsample(x)

        out = self.from_rgb(x)
        out = self.features(out)
        out = self.upscale(out)

        out = torch.cat((out, up), axis=1)
        out = self.to_rgb(out)

        return out