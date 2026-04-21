import torch
from pytorch_fid import fid_score
import os
import shutil
from pathlib import Path
from PIL import Image

real_dir = Path("/home/fvaleau/HAE/proj/flowers/train")
gen_dir = Path("/home/fvaleau/HAE/FID/hfm_samples_flowers_1_1020_in")
new_test = Path("/home/fvaleau/HAE/FID/flowers_train")

size = (256, 256)
idx = 0
# for img_path in Path(real_dir).rglob("*"):
#     if idx == 1020:
#         break
#     if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
#         continue
#     img = Image.open(img_path).convert("RGB")
#     img = img.resize(size, Image.BILINEAR)
#     img.save(os.path.join(new_test, img_path.name))
#     idx += 1


for class_dir in sorted(real_dir.iterdir()):
    if not class_dir.is_dir():
        continue


    imgs = [
        p for p in sorted(class_dir.iterdir())
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    selected = imgs[:10]

    for img_path in selected:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        img.save(new_test / img_path.name)

        idx += 1

print(f"Processed {idx} images for FID calculation.")


paths = [str(new_test), str(gen_dir)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fid = fid_score.calculate_fid_given_paths(
    paths,
    batch_size=32,
    device=device,
    dims=2048,
    num_workers=0
)

print(f"FID score: {fid:.4f}")
