"""
Converts images from `download_ffhq.py` to pytorch tensors.

Assumes `download_ffhq.py` was run before for thumbnails or full size
images n the same directory. Change `folder` accordingly to either
`images1020x1024` or `thumbnails128x128`.
A new folder in this directory will be created named `tensors`
containing the images as tensors of shape
`(chunk_size*1000, 3, resolution, resolution)`.
`test_size` and `val_size` may be adjusted to change the size of the
test, validation set respectively.
"""

from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm
import math

def save_imgs(start, end, name, folder, chunk_size=70):
    """
    Saves the images in the range `start*1000` to `end*1000`.

    Parameters
    ----------
    start : int
        Starts with images in the subfolder `start*1000`.
    end: int
        Last included subfolder is `(end-1)*1000`.
    name : str
        Name of the tensor saved.
    folder : str
        Directory from which images are taken, i.e. thumbnails or images.
    chunk_size : int, optional
        saved tensors contain at most `chunk_size*1000` images.
    """

    print(f"Start {name}-set")
    to_tensor = transforms.ToTensor()
    for ch in range(start, end, chunk_size):
        print(f"\nChunk {ch//chunk_size + 1}/{math.ceil((end-start)/chunk_size)}")

        imgs = [None]*(min(end - ch, chunk_size)*1000)
        for i in tqdm(range(ch, min(ch+chunk_size, end)), desc="Folder"):
            for j in tqdm(range(1000), desc="Image "):
                imgs[(i-ch)*1000+j] = to_tensor(Image.open(f"{folder}/{i*1000:05}/{i*1000+j:05}.png")).unsqueeze(0)

        print("\nConverting")
        imgs = torch.cat(imgs)
        print("Saving ...",end=" ")
        torch.save(imgs, f"{folder}/tensors/{name}_{ch//chunk_size}.pt")
        print("Saved!")


if __name__ == '__main__':
    folder = "thumbnails128x128"
    chunk_size = 10

    val_size = 10
    test_size = 5
    train_size = 70 - val_size - test_size

    Path("thumbnails128x128/tensors").mkdir(exist_ok = True)

    save_imgs(0, train_size, "train", folder, chunk_size)
    print("\n"*4)
    save_imgs(train_size, train_size+val_size, "validation", folder, chunk_size)
    print("\n"*4)
    save_imgs(train_size+val_size, 70, "test", folder, chunk_size)