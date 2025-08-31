import os, io
from datasets import load_dataset
from PIL import Image

# Add your local pyspz path if not installed
import sys
sys.path.append(os.path.abspath("../404-Repo/spz"))
import pyspz

# Output directory for TRELLIS-ready dataset
OUT_DIR = "data/404mini/train"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading 404-Gen/404mini dataset...")
ds = load_dataset(
    "404-Gen/404mini",
    split="train",
    streaming=False,
    ignore_verifications=True  # skip schema mismatch checks
)

# Keep only the columns we need
columns_to_keep = ["prompt", "spz", "image"]
extra_columns = [c for c in ds.column_names if c not in columns_to_keep]
if extra_columns:
    ds = ds.remove_columns(extra_columns)

for i, ex in enumerate(ds):
    sid = f"{i:07d}"
    sdir = os.path.join(OUT_DIR, sid)
    os.makedirs(sdir, exist_ok=True)

    # 1) Save prompt
    with open(os.path.join(sdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(ex["prompt"].strip() + "\n")

    # 2) Decompress SPZ -> PLY
    spz_bytes = ex["spz"]
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