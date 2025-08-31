import os
import io
from PIL import Image
import sys

# Add your local pyspz path
sys.path.append(os.path.abspath("../404-Repo/spz"))
import pyspz

from datasets import load_dataset

# Output directory for TRELLIS-ready dataset
OUT_DIR = "data/404mini/train"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading 404-Gen/404mini dataset (streaming mode)...")
ds = load_dataset(
    "404-Gen/404mini",
    split="train",
    streaming=True  # avoids Arrow caching and column mismatch errors
)

# Only process columns we need: 'prompt', 'spz', 'image'
required_columns = ["prompt", "spz", "image"]

for i, ex in enumerate(ds):
    sid = f"{i:07d}"
    sdir = os.path.join(OUT_DIR, sid)
    os.makedirs(sdir, exist_ok=True)

    # 1) Save prompt
    prompt = ex.get("prompt", "").strip()
    with open(os.path.join(sdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt + "\n")

    # 2) Decompress SPZ -> PLY
    spz_bytes = ex.get("spz", None)
    if spz_bytes is not None:
        if isinstance(spz_bytes, dict) and "bytes" in spz_bytes:
            spz_bytes = spz_bytes["bytes"]
        ply_bytes = pyspz.decompress(spz_bytes)
        with open(os.path.join(sdir, "gaussian.ply"), "wb") as f:
            f.write(ply_bytes)

    # 3) Save preview image if present
    img = ex.get("image", None)
    if img is not None:
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img))
        img.save(os.path.join(sdir, "preview.png"))

    if i % 500 == 0:
        print(f"Processed {i} samples...")

print("✅ Dataset prep complete. TRELLIS-ready data is in:", OUT_DIR)