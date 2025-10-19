# make_splits.py
from pathlib import Path
import random

project = Path(__file__).resolve().parent
# EDIT these to your paths
real_train_dir = Path("realworld_images/train/images")
real_val_dir   = Path("realworld_images/val/images")
real_test_dir  = Path("realworld_images/test/images")

# Your synthetic images root(s). Add more roots if needed.
synthetic_roots = [
    Path("synthetic_dataset/images/train"),   # example
]

def list_images(root):
    exts = {'.jpg','.jpeg','.png'}
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in exts])

def has_label(img: Path):
    p = str(img)
    if '/images/' in p:
        lab = Path(p.replace('/images/', '/labels/')).with_suffix('.txt')
    elif '\\images\\' in p:
        lab = Path(p.replace('\\images\\', '\\labels\\')).with_suffix('.txt')
    else:
        # No 'images' in path; keep label next to the image
        lab = img.with_suffix('.txt')
    if lab.exists():
        return True
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.touch(exist_ok=True)
    return True

# Real sets
real_train = [p for p in list_images(real_train_dir) if has_label(p)]
real_val   = [p for p in list_images(real_val_dir)   if has_label(p)]
real_test  = [p for p in list_images(real_test_dir)]  # labels optional

# Synthetic train images
synthetic_train = []
for root in synthetic_roots:
    synthetic_train += list_images(root)

# >>> Oversample real train to ~30% of batches <<<
# Simple rule: duplicate real paths K times
K = 3  # start with 2; increase to 3 if real under-represented

def abs_path(p):
    # if already absolute, return it directly
    p = Path(p)
    if p.is_absolute():
        return str(p)
    # otherwise make it absolute relative to the project root
    return str((project / p).resolve())

train_list = [abs_path(p) for p in synthetic_train] + [abs_path(p) for p in real_train] * K
val_list   = [abs_path(p) for p in real_val]
test_list  = [abs_path(p) for p in real_test]

# Shuffle for randomness (optional)
random.shuffle(train_list)

# Write lists
(project/'lists').mkdir(exist_ok=True)
(project/'lists/train.txt').write_text('\n'.join(train_list))
(project/'lists/val.txt').write_text('\n'.join(val_list))
(project/'lists/test.txt').write_text('\n'.join(test_list))

print(f"Wrote {len(train_list)} train (incl. oversample), {len(val_list)} val, {len(test_list)} test")