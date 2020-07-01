import h5py
import numpy as np

from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path

def make_hdf5(in_folder, out_folder, filename, size):
    Path(out_folder).mkdir(exist_ok = True)

    with h5py.File(f"{out_folder}/{filename}.hdf5", "w") as file:
        train_set = file.create_dataset("train", (55000, 3, size, size), dtype=float)
        validation_set = file.create_dataset("validation", (10000, 3, size, size), dtype=float)
        test_set = file.create_dataset("test", (5000, 3, size, size), dtype=float)

        for i in tqdm(range(20)):
                for j in tqdm(range(1000)):
                    train_set[i*1000 + j] = np.array(Image.open(f"{in_folder}/{i*1000:05}/{i*1000 + j:05}.png")).transpose(2,0,1)

        # for i in tqdm(range(10)):
        #     for j in tqdm(range(1000)):
        #         validation_set[i*1000 + j] = np.array(Image.open(f"{in_folder}/{(i+55)*1000:05}/{(i+55)*1000 + j:05}.png")).transpose(2,0,1)

        # for i in tqdm(range(5)):
        #     for j in tqdm(range(1000)):
        #         test_set[i*1000 + j] = np.array(Image.open(f"{in_folder}/{(i+65)*1000:05}/{(i+65)*1000 + j:05}.png")).transpose(2,0,1)

if __name__ == '__main__':
    in_folder = "thumbnails128x128"
    out_folder = "hdf5"
    filename = "thumbs"
    size = 128

    make_hdf5(in_folder, out_folder, filename, size)