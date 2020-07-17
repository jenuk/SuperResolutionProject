import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, upscaling_factor):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=upscaling_factor, mode="nearest")
        self.convs = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 16, 1),
            nn.ReLU(),
            nn.Sequential(
                *(nn.Sequential(
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                ) for _ in range(5))
            ),
            nn.ConvTranspose2d(16, 128, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.to_rgb = nn.Conv2d(128+3, 3, 1)

    def forward(self, x):
        up = self.upsample(x)
        out = self.convs(x)
        out = torch.cat((out, up), axis=1)
        out = self.to_rgb(out)

        return out