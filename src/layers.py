import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class GaussianBlur(nn.Module):
    def __init__(self, radius, sigma = None, use_gpu = False):
        super().__init__()

        self.sigma = radius/3 if sigma is None else sigma
        self.mean = radius
        self.radius = radius

        with torch.no_grad():
            self.kernel = torch.tensor([[[x-self.mean, y-self.mean] for x in range(2*radius+1)] for y in range(2*radius+1)])
            self.kernel = torch.exp(-torch.sum(self.kernel**2, axis=2)/(2*self.sigma**2))
            self.kernel = self.kernel / (2*np.pi*self.sigma**2)
            self.kernel = self.kernel / torch.sum(self.kernel)
            self.kernel = torch.cat((self.kernel.unsqueeze(0),)*3, dim=0).unsqueeze(1)

        if use_gpu:
            self.kernel = self.kernel.cuda()

    def forward(self, img):
        res = F.conv2d(img, self.kernel, stride=1, padding=self.radius, groups=3)

        return res

class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        out = out + x

        return out