import os
import io
import json
from PIL import Image
import sys

# Local pyspz import
sys.path.append(os.path.abspath("../404-Repo/spz"))
import pyspz

OUT_DIR = "data/404mini/train"
os.makedirs(OUT_DIR, exist_ok=True)

# Use streaming=True and raw iterator
from datasets import load_dataset

print("Loading 404-Gen/404mini dataset in streaming mode...")
ds = load_dataset("404-Gen/404mini", split="train", streaming=True)
print(ds[0])
for i, ex in enumerate(ds):
    sid = f"{i:07d}"
    sdir = os.path.join(OUT_DIR, sid)
    os.makedirs(sdir, exist_ok=True)

    # Save prompt
    prompt = ex.get("prompt", "").strip()
    with open(os.path.join(sdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt + "\n")

    # Decompress SPZ -> PLY if exists
    spz_bytes = ex.get("spz", None)
    if spz_bytes:
        if isinstance(spz_bytes, dict) and "bytes" in spz_bytes:
            spz_bytes = spz_bytes["bytes"]
        ply_bytes = pyspz.decompress(spz_bytes)
        with open(os.path.join(sdir, "gaussian.ply"), "wb") as f:
            f.write(ply_bytes)

    # Save preview if exists
    img_data = ex.get("image", None)
    if img_data:
        if not isinstance(img_data, Image.Image):
            img_data = Image.open(io.BytesIO(img_data))
        img_data.save(os.path.join(sdir, "preview.png"))

    if i % 500 == 0:
        print(f"Processed {i} samples...")

print("✅ Dataset prep complete. TRELLIS-ready data is in:", OUT_DIR)