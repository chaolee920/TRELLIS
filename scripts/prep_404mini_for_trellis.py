import os, io
from datasets import load_dataset
from PIL import Image
import pyspz  # from the 404-Repo/spz you installed

OUT_DIR = "data/404mini/train"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading 404-Gen/404mini dataset...")
ds = load_dataset("404-Gen/404mini", split="train", streaming=False)

for i, ex in enumerate(ds):
    sid = f"{i:07d}"
    sdir = os.path.join(OUT_DIR, sid)
    os.makedirs(sdir, exist_ok=True)

    # Save prompt
    with open(os.path.join(sdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(ex["prompt"].strip() + "\n")

    # Decompress SPZ -> PLY
    spz_bytes = ex["spz"]
    if isinstance(spz_bytes, dict) and "bytes" in spz_bytes:
        spz_bytes = spz_bytes["bytes"]
    ply_bytes = pyspz.decompress(spz_bytes)
    with open(os.path.join(sdir, "gaussian.ply"), "wb") as f:
        f.write(ply_bytes)

    # Save preview
    if ex.get("image") is not None:
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img))
        img.save(os.path.join(sdir, "preview.png"))

    if i % 500 == 0:
        print(f"Processed {i} samples...")

print("✅ Done. Data ready in:", OUT_DIR)