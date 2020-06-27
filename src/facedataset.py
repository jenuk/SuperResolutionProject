import torch
from torchvision import transforms
from PIL import Image

class FaceDataset(torch.utils.data.Dataset):
    """ A `Dataset` representing the faces from the ffhq dataset."""

    def __init__(self, path, start, end, lower_res, higher_res, p_flip = 0.5):
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
        self.flip_transform = transforms.RandomHorizontalFlip(p_flip)
        self.resize_lower_transform = transforms.Resize(lower_res)
        self.resize_higher_transform = transforms.Resize(higher_res)
        self.tensor_transform = transforms.ToTensor()

    def __getitem__(self, idx):
        """returns image at idx in two resoultions"""
        idx += self.start
        k = (idx//1000)*1000
        l = idx - k
        img = Image.open(f"{self.path}/{k:05}/{idx:05}.png")
        img = self.flip_transform(img)
        lower = self.resize_lower_transform(img)
        higher = self.resize_higher_transform(img)
        return self.tensor_transform(lower), self.tensor_transform(higher)

    def __len__(self):
        """returns amount of images in this dataset"""
        return self.end - self.start