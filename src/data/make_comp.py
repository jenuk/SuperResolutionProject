"""
samples 100 random images without copyright from validation and test set, to easily asses visual quality

saves them in
comp_validation/ (for validation images)
comp_test/ (for test images)
"""

import random
import json
from pathlib import Path
from shutil import copyfile

random.seed(714)

val_range = (55000, 65000)
test_range = (65000, 70000)
no_samples = 100

print("Loading...", end=" ", flush=True)
with open("ffhq-dataset-v2.json", "r") as file:
    metadata = json.load(file)
print("Processing...", end=" ", flush=True)
license = [""]*70000
for i in range(70000):
    license[i] = metadata[f"{i}"]["metadata"]["license"]
del metadata
print("Done.")

print("Choosing...", end=" ", flush=True)
valid_licenses = {"Public Domain Mark", "Public Domain Dedication (CC0)"}

val_valid = [i for i in range(*val_range) if license[i] in valid_licenses]
test_valid = [i for i in range(*test_range) if license[i] in valid_licenses]

val_choice = random.sample(val_valid, no_samples)
test_choice = random.sample(test_valid, no_samples)
print("Done.")

print("Saving...", end=" ", flush=True)
Path("comp_validation").mkdir(exist_ok = True)
Path("comp_test").mkdir(exist_ok = True)
Path("comp_validation/00000").mkdir(exist_ok = True)
Path("comp_test/00000").mkdir(exist_ok = True)

for k, i in enumerate(val_choice):
    copyfile(f"images1024x1024/{(i//1000)*1000:05}/{i:05}.png", f"comp_validation/00000/{k:05}.png")

for k, i in enumerate(test_choice):
    copyfile(f"images1024x1024/{(i//1000)*1000:05}/{i:05}.png", f"comp_test/00000/{k:05}.png")

print("Done.")