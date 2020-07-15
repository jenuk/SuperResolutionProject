from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

name = "small"
resolution = 256

Path("{name}{resolution}x{resolution}").mkdir(exist_ok = True)

pbar = tqdm(total=70000)
for i in range(70):
    Path(f"{name}{resolution}x{resolution}/{i*1000:05}").mkdir(exist_ok = True)

    for j in range(1000):
        img = Image.open(f"images1024x1024/{i*1000:05}/{i*1000+j:05}.png")
        img = img.resize((resolution, resolution))
        img.save(f"{name}{resolution}x{resolution}/{i*1000:05}/{i*1000+j:05}.png")

        pbar.update(1)

pbar.close()