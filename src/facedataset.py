import torch
from torchvision import transforms
from PIL import Image

from layers import GaussianBlur

class FaceDataset(torch.utils.data.Dataset):
    """ A `Dataset` representing the faces from the ffhq dataset."""

    def __init__(self, path, start, end, lower_res, factor, sigma = 1.0, p_flip = 0.5, use_gpu = False):
        """
        Parameters
        ----------
        path : str
            path to the images, should not end with `/`
        start : int
            first image to consider part of the dataset (should be in
            [0, 70'000))
        end : int
            first image (exklusiv) not part of the dataset (should be in
            (start, 70'000])
        lower_res : int
            resolution of low resolution image returned by `__getitem__`
        high_res : int
            resolution of high resolution image returned by `__getitem__`
        p_flip : float, optional
            probability of flipping an image horizontal (default 0.5)
        """

        self.path = path
        self.start = start
        self.end = end
        self.factor = factor
        self.use_gpu = use_gpu
        self.gb = GaussianBlur(int(3*sigma), float(sigma), use_gpu)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p_flip),
            transforms.Resize(lower_res*factor),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        """returns image at idx in two resoultions"""

        idx += self.start
        k = (idx//1000)*1000
        l = idx - k
        img = Image.open(f"{self.path}/{k:05}/{idx:05}.png")

        img = self.transform(img)
        if self.use_gpu:
            img = img.cuda()

        with torch.no_grad():
            lower = self.gb(img.unsqueeze(0))[0, :, ::self.factor, ::self.factor]

        return lower, img

    def __len__(self):
        """returns amount of images in this dataset"""

        return self.end - self.start