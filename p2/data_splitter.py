""" file to split the data into train, val and test splits """

from pathlib import Path
import random

data_dir = Path("preprocessed_data")
split_dir = Path("splits")
split_dir.mkdir(exist_ok=True)

files = sorted([p.name for p in data_dir.glob("scene_*.npz")])
seed = 42
random.Random(seed).shuffle(files)

n = len(files)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train = files[:n_train]
val = files[n_train:n_train+n_val]
test = files[n_train+n_val:]

for name, arr in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
    (split_dir / name).write_text("\n".join(arr) + "\n")

