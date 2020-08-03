from PIL import Image
from torchvision import transforms
from torch import nn
import torch

class Classic:
    """
    A wrapper for PIL's resize in batches.
    """

    def __init__(self, output_size, interpolation = Image.BICUBIC, use_gpu = False):
        self.output_size = output_size
        self.trafo = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(output_size, interpolation),
            transforms.ToTensor(),
        ])
        self.use_gpu = use_gpu

    def __call__(self, x):
        N = x.shape[0]
        res = torch.zeros((N, 3, self.output_size, self.output_size))

        # transforms do not support batches
        for i in range(N):
            res[i] = self.trafo(x[i].cpu())

        if self.use_gpu:
            res = res.cuda()

        return res

    def eval(self):
        pass

    def train(self):
        pass

    def cuda(self):
        self.use_gpu = True
        return self