from PIL import Image
from torchvision import transforms
from torch import nn
import torch

class Classic(nn.Module):
    """
    Basically a PIL resize, wrapped in a torch model.
    """

    def __init__(self, size, interpolation = Image.BILINEAR):
        super().__init__()

        self.size = size
        self.trafo = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size, interpolation),
            transforms.ToTensor(),
        ])

    def forward(self, x):
        N = x.shape[0]
        res = torch.zeros((N, 3, self.size, self.size))

        # transforms do not support batches
        for i in range(N):
            res[i] = self.trafo(x[i])

        return res