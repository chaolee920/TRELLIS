To fine-tune TRELLIS for **image-to-3D** generation using a subset of 100 examples from the 404-Gen/404mini dataset, you can adapt the preparation and training pipeline previously outlined. The 404-Gen/404mini dataset includes JSON files (e.g., `antique_toy_train_set_running.json`) with `miner` and `signature`, where the prompt is derived from the JSON file name, and the 3D model is stored in a corresponding `.ply.spz` file (Gaussian splat). For image-to-3D, you’ll use the dataset’s `.png` files (single-view renders) or generate multiview images from the `.ply` files. Since you’re using only 100 examples, this reduces computational requirements, making it feasible to run on a single GPU like an NVIDIA RTX 4090 or A100.

Below, I’ll provide a step-by-step guide tailored for image-to-3D fine-tuning with 100 examples, including updated code structures to handle the dataset’s structure (prompts from JSON file names, models in `.ply.spz`, and renders in `.png`). I’ll assume you’re using TRELLIS’s image-to-3D pipeline and that `.png` files are available for the 100 examples; if not, I’ll include an option to generate renders. The steps align with TRELLIS’s data preparation and training pipeline, optimized for a small subset.

---

### Step-by-Step Guide for Image-to-3D Fine-Tuning (100 Examples)

#### Step 1: Set Up the Environment

Ensure your environment is ready for TRELLIS and the 404-Gen/404mini dataset.

1. **Install Prerequisites**:

   - Python 3.8+, CUDA 11.8 or 12.2, Conda.
   - GPU: NVIDIA RTX 4090 (24GB) or A100 (40GB/80GB) recommended. A single RTX 4090 is sufficient for 100 examples with LoRA.

2. **Clone TRELLIS Repository**:

   ```bash
   git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
   cd TRELLIS
   ```

3. **Install Dependencies**:
   Run the setup script to create a Conda environment and install required packages (PyTorch, Transformers, etc.).

   ```bash
   ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
   conda activate trellis
   ```

4. **Additional Packages for 404-Gen/404mini**:

   ```bash
   pip install huggingface_hub datasets pyspz open3d gsplat
   ```

5. **Verify GPU**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

#### Step 2: Load and Preprocess 100 Examples

Load 100 JSON files and their corresponding `.ply.spz` and `.png` files from the 404-Gen/404mini dataset, derive prompts from JSON file names, and create a TRELLIS-compatible CSV metadata file.

**Code (prepare_train.py)**:
<xaiArtifact artifact_id="9aba347c-6919-4755-9ab7-623ff1bf8a6d" artifact_version_id="f7171cf9-d7ef-451f-a07b-8a9e145a7503" title="prepare_train.py" contentType="text/python">
from huggingface_hub import list_repo_files, hf_hub_download
import os
import pandas as pd
from datasets import Dataset

def load_404mini_dataset(repo_id="404-Gen/404mini", output_dir="datasets/404mini", max_samples=100):
os.makedirs(output_dir, exist_ok=True)
spz_dir = os.path.join(output_dir, "spz")
render_dir = os.path.join(output_dir, "renders")
os.makedirs(spz_dir, exist_ok=True)
os.makedirs(render_dir, exist_ok=True)

    # Get list of files
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    json_files = [f for f in files if f.endswith(".json") and f.startswith("assets/")]
    ply_files = [f for f in files if f.endswith(".ply.spz") and f.startswith("assets/")]
    png_files = [f for f in files if f.endswith(".png") and f.startswith("assets/")]

    data = []
    for json_file in json_files[:max_samples]:  # Limit to 100 examples
        try:
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            category = json_file.split('/')[1]
            ply_file = f"assets/{category}/{base_name}.ply.spz"
            png_file = f"assets/{category}/{base_name}.png"

            if ply_file in ply_files:
                # Derive prompt from file name
                prompt = base_name.replace("_", " ")

                # Download .ply.spz
                spz_path = os.path.join(spz_dir, f"{base_name}.ply.spz")
                hf_hub_download(repo_id=repo_id, filename=ply_file, repo_type="dataset", local_dir=spz_dir)

                # Download .png if available
                render_path = None
                if png_file in png_files:
                    render_path = os.path.join(render_dir, f"{base_name}.png")
                    hf_hub_download(repo_id=repo_id, filename=png_file, repo_type="dataset", local_dir=render_dir)

                data.append({
                    "uid": base_name,
                    "name": base_name,
                    "source": "404mini",
                    "captions": [prompt],  # Optional for image-to-3D
                    "aesthetic_score": 5.0,
                    "model_path": spz_path,
                    "render_path": render_path
                })
            else:
                print(f"Skipping {json_file}: No matching .ply.spz")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if not data:
        raise ValueError("No valid samples found with matching .ply.spz files")

    # Save metadata
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "404mini_100.csv")
    df.to_csv(csv_path, index=False)
    dataset = Dataset.from_pandas(df)

    print(f"Loaded {len(dataset)} valid samples")
    print(f"Metadata saved to {csv_path}")
    return dataset, csv_path

if **name** == "**main**":
try:
dataset, csv_path = load_404mini_dataset(max_samples=100)
print(dataset[:5])
except Exception as e:
print(f"Error: {e}")
</xaiArtifact>

**Run Command**:

```bash
python prepare_train.py
```

**Notes**:

- Limits to 100 JSON files (`json_files[:max_samples]`).
- Derives `prompt` from JSON file name (e.g., `antique_toy_train_set_running` → "antique toy train set running").
- Saves `.ply.spz` and `.png` files to `datasets/404mini/spz/` and `datasets/404mini/renders/`.
- Creates `404mini_100.csv` with columns: `uid`, `name`, `source`, `captions`, `aesthetic_score`, `model_path`, `render_path`.
- If `.png` files are missing, you’ll generate renders in Step 3.

#### Step 3: Decompress .ply.spz Files

Decompress `.ply.spz` to `.ply` for 3D model processing.

**Code (decompress_spz.py)**:
<xaiArtifact artifact_id="bf6124f7-1629-436c-9fd5-780cee6b44d1" artifact_version_id="39df8a77-7c0a-4ad5-a241-072fe3a5e873" title="decompress_spz.py" contentType="text/python">
import pyspz
import os
import pandas as pd

def decompress_spz_files(csv_path, output_dir):
df = pd.read_csv(csv_path)
ply_dir = os.path.join(output_dir, "ply")
os.makedirs(ply_dir, exist_ok=True)

    for idx, row in df.iterrows():
        spz_path = row['model_path']
        ply_path = os.path.join(ply_dir, f"{row['uid']}.ply")
        try:
            with open(spz_path, 'rb') as f:
                compressed = f.read()
            decompressed = pyspz.decompress(compressed, include_normals=True)
            with open(ply_path, 'wb') as f:
                f.write(decompressed)
            df.at[idx, 'model_path'] = ply_path
        except Exception as e:
            print(f"Error decompressing {spz_path}: {e}")

    df.to_csv(csv_path, index=False)
    return df

if **name** == "**main**":
csv_path = "datasets/404mini/404mini_100.csv"
output_dir = "datasets/404mini"
try:
df = decompress_spz_files(csv_path, output_dir)
print(f"Decompressed files saved to {output_dir}/ply")
except Exception as e:
print(f"Error: {e}")
</xaiArtifact>

**Run Command**:

```bash
pip install pyspz
python decompress_spz.py
```

**Notes**:

- Install `pyspz` (`git clone https://github.com/404-repo/spz; pip install .`).
- Updates `model_path` in `404mini_100.csv` to point to `.ply` files.
- 100 samples require ~1–5GB disk space.

#### Step 4: Run Data Preparation Toolkits

Run TRELLIS’s data preparation toolkits to process the 100 examples. For image-to-3D, focus on using `.png` files (from `render_path`) or generating multiview images if `.png` files are missing.

##### 1. Build Metadata

Validate `404mini_100.csv`.

**Run**:

```bash
    python dataset_toolkits/build_metadata1.py 404mini_5 --output_dir datasets/404mini
```

##### 2. Render Multiview Images (Optional)

If `.png` files are available in `render_path`, skip this step and use them for feature extraction. Otherwise, convert `.ply` to `.obj` (meshes) or render `.ply` directly (Gaussian splats).

**Option 1: Use Dataset’s .png Files**
Check if `.png` files exist:

```python
from datasets import Dataset
dataset = Dataset.from_csv("datasets/404mini/404mini_5.csv")
print(dataset.filter(lambda x: x['render_path'] is not None).num_rows)
```

If most samples have `render_path`, proceed to feature extraction.

**Option 2: Convert .ply to Meshes and Render**
Convert `.ply` to `.obj` for TRELLIS’s default renderer.

**Code (convert_ply_to_mesh.py)**:
<xaiArtifact artifact_id="47bc322e-9089-4191-8e8a-89c87ede1cb3" artifact_version_id="282694a4-803b-444e-89c6-64153bf83ec8" title="convert_ply_to_mesh.py" contentType="text/python">
import open3d as o3d
import os
import pandas as pd

def convert*ply_to_mesh(ply_path, obj_path):
pcd = o3d.io.read_point_cloud(ply_path)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
mesh, * = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
o3d.io.write_triangle_mesh(obj_path, mesh)

def main(csv_path, output_dir):
df = pd.read_csv(csv_path)
mesh_dir = os.path.join(output_dir, "meshes")
os.makedirs(mesh_dir, exist_ok=True)

    for idx, row in df.iterrows():
        ply_path = row['model_path']
        obj_path = os.path.join(mesh_dir, f"{row['uid']}.obj")
        try:
            convert_ply_to_mesh(ply_path, obj_path)
            df.at[idx, 'model_path'] = obj_path
        except Exception as e:
            print(f"Error converting {ply_path}: {e}")

    df.to_csv(csv_path, index=False)

if **name** == "**main**":
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()
main(args.csv_path, args.output_dir)
</xaiArtifact>

**Run**:

```bash
pip install open3d
python dataset_toolkits/convert_ply_to_mesh.py --csv_path datasets/404mini/404mini_100.csv --output_dir datasets/404mini
python dataset_toolkits/render.py 404mini_100 --output_dir datasets/404mini --num_views 8
```

**Option 3: Gaussian Splat Rendering**
Render `.ply` files directly using `gsplat`.

**Code (render_gaussian.py)**:
<xaiArtifact artifact_id="48a828b4-e12e-4b05-95b3-c3fb28bb578c" artifact_version_id="24b2d13f-bbad-456f-9b1b-033db44c598b" title="render_gaussian.py" contentType="text/python">
import argparse
import os
import torch
import pandas as pd
import numpy as np
from gsplat import rasterize_gaussians # Placeholder; adjust to gsplat API

def load_gaussian_splat(ply_path): # Implement based on gsplat documentation
pass

def compute_view_matrix(azimuth, elevation, radius): # Implement camera pose
pass

def compute_perspective_matrix(fovy, aspect, near, far): # Implement perspective projection
pass

def render_gaussian_splat(ply_path, output_dir, num_views=8):
points, colors, sh_coeffs = load_gaussian_splat(ply_path)
os.makedirs(output_dir, exist_ok=True)

    height, width = 256, 256
    fovy = np.deg2rad(45)

    for view_idx in range(num_views):
        azimuth = view_idx * 360 / num_views
        view_matrix = compute_view_matrix(azimuth=azimuth, elevation=0, radius=2.0)
        proj_matrix = compute_perspective_matrix(fovy, width/height, near=0.1, far=100.0)
        image = rasterize_gaussians(
            points=points,
            colors=colors,
            sh_coeffs=sh_coeffs,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            image_size=(height, width)
        )
        image.save(os.path.join(output_dir, f"view_{view_idx}.png"))

def main(args):
df = pd.read_csv(os.path.join(args.output_dir, f"{args.dataset_name}.csv"))
render_dir = os.path.join(args.output_dir, "renders")
os.makedirs(render_dir, exist_ok=True)

    for idx, row in df.iterrows():
        ply_path = row['model_path']
        output_subdir = os.path.join(render_dir, row['uid'])
        try:
            render_gaussian_splat(ply_path, output_subdir, args.num_views)
        except Exception as e:
            print(f"Error rendering {ply_path}: {e}")

if **name** == "**main**":
parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_views", type=int, default=8)
args = parser.parse_args()
main(args)
</xaiArtifact>

**Run**:

```bash
pip install gsplat
python dataset_toolkits/render_gaussian.py 404mini_100 --output_dir datasets/404mini --num_views 8
```

##### 3. Extract Features

Extract DINO features from `.png` files (via `render_path`) or multiview images.

**Code (extract_features.py)**:
<xaiArtifact artifact_id="f5061e87-5062-4b18-93e1-cc4a739f4da7" artifact_version_id="8a0c4073-45f6-448a-992d-1fca553c3e8b" title="extract_features.py" contentType="text/python">
import argparse
import os
from transformers import AutoModel, AutoFeatureExtractor
import torch
import pandas as pd
from PIL import Image

def main(args):
df = pd.read_csv(os.path.join(args.output_dir, f"{args.dataset_name}.csv"))
feature_dir = os.path.join(args.output_dir, "features")
os.makedirs(feature_dir, exist_ok=True)

    extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits16")
    model = AutoModel.from_pretrained("facebook/dino-vits16").cuda()

    for idx, row in df.iterrows():
        features = []
        if row['render_path'] and os.path.exists(row['render_path']):
            img_path = row['render_path']
            image = Image.open(img_path).convert("RGB")
            inputs = extractor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            features.append(outputs.last_hidden_state.squeeze().cpu())
        else:
            render_dir = os.path.join(args.output_dir, "renders", row['uid'])
            for view_idx in range(args.num_views):
                img_path = os.path.join(render_dir, f"view_{view_idx}.png")
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert("RGB")
                    inputs = extractor(images=image, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    features.append(outputs.last_hidden_state.squeeze().cpu())

        if features:
            torch.save(features, os.path.join(feature_dir, f"{row['uid']}.pt"))
        else:
            print(f"No images found for {row['uid']}")

if **name** == "**main**":
parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_views", type=int, default=8)
args = parser.parse_args()
main(args)
</xaiArtifact>

**Run**:

```bash
python dataset_toolkits/extract_feature1.py 404mini_5 --output_dir datasets/404mini --num_views 8
```

##### 4. Voxelize, Encode Sparse Structure, and Encode SLAT

These steps are identical to text-to-3D, processing the 3D models.

**Run**:

```bash
python dataset_toolkits/voxelize1.py 404mini_5 --output_dir datasets/404mini
python dataset_toolkits/encode_ss_latent1.py 404mini_5 --output_dir datasets/404mini
python dataset_toolkits/encode_latent1.py 404mini_5 --output_dir datasets/404mini
```

**Notes**:

- For 100 samples, data preparation is fast (~10–30 minutes on RTX 4090, ~5–15 minutes on A100 80GB).
- If `.png` files are available, skip rendering to save time.
- Use `gsplat` for rendering if you prefer not to convert to meshes, but implement `load_gaussian_splat` and camera functions per `gsplat`’s documentation.

#### Step 5: Configure and Run Fine-Tuning

Configure TRELLIS for image-to-3D fine-tuning and train on the 100 examples.

1. **Select Config**:

   - Use an image-to-3D config (e.g., `slat_flow_image_dit_L_64l8p2_fp16.json`).
   - Edit to point to `datasets/404mini` and set batch size (e.g., 4–8 for RTX 4090).

2. **Download Checkpoint**:

   - Use `microsoft/TRELLIS-image-large` from Hugging Face.
   - Place in `path/to/pretrained/image_checkpoint`.

3. **Run Training**:
   ```bash
   python train.py \
     --config configs/generation/slat_flow_image_dit_L_64l8p2_fp16.json \
     --output_dir outputs/finetune_404mini_image_100 \
     --data_dir datasets/404mini \
     --load_dir path/to/pretrained/image_checkpoint \
     --ckpt latest \
     --num_gpus 1
   ```
   python train1.py \
    --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json \
    --output_dir outputs/finetune_404mini_image_5 \
    --data_dir datasets/404mini \
    --load_dir /workspace/.hf_home/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96 \
    --ckpt latest \
    --num_gpus 1
   python train1.py \
   --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json \
   --output_dir outputs/finetune_404mini_image_5 \
   --data_dir datasets/404mini \
   --load_dir /workspace/.hf_home/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96 \
   --ckpt latest \
   --num_gpus 1 \
   --dataset_name 404mini_5
   **Notes**:

- Use `--batch_size 4` or `--gradient_accumulation_steps 2` for RTX 4090 to manage VRAM.
- For A100, increase batch size to 8–16.
- Training 100 samples for 1–5 epochs takes ~10–60 minutes on RTX 4090, ~5–30 minutes on A100 80GB.

#### Step 6: Evaluate

Test the fine-tuned model with an image input.

**Run**:

```bash
python example_image.py --image datasets/404mini/renders/antique_toy_train_set_running.png --load_dir outputs/finetune_404mini_image_100 --ckpt best
```

**Output**: Generates a 3D model (.glb or .ply) based on the input image.

---

### Cost and Time Estimates (100 Examples)

- **GPU**: NVIDIA RTX 4090 (24GB, ~$0.76/hour on Vast.ai or ~$0.10/hour power cost locally) or A100 80GB (~$1.99/hour on RunPod).
- **Data Preparation**:
  - **RTX 4090**: ~10–30 minutes (rendering 100 samples _ 8 views _ ~1–2 seconds, plus ~5–10 seconds for voxelization/features/SLAT).
  - **A100 80GB**: ~5–15 minutes.
  - **Cost**: ~$0.13–$0.38 (RTX 4090, Vast.ai), ~$0.17–$0.50 (A100, RunPod), or ~$0.01–$0.03 (local RTX 4090).
  - **If Using .png**: ~5–10 minutes (skips rendering), ~$0.06–$0.13 (RTX 4090), ~$0.10–$0.20 (A100).
- **Training**:
  - 100 samples, 1–5 epochs, batch size 4: ~100 / 4 = 25 iterations per epoch.
  - **RTX 4090**: ~0.5–1 second/iteration, ~13–125 seconds/epoch, ~1–10 minutes total.
  - **A100 80GB**: ~0.3–0.5 seconds/iteration, ~8–75 seconds/epoch, ~0.5–6 minutes total.
  - **Cost**: ~$0.02–$0.13 (RTX 4090, Vast.ai), ~$0.03–$0.20 (A100, RunPod), or ~$0.01 (local RTX 4090).
- **Total**:
  - **RTX 4090**: ~15–40 minutes, ~$0.15–$0.51 (Vast.ai) or ~$0.02–$0.04 (local).
  - **A100 80GB**: ~6–21 minutes, ~$0.20–$0.70 (RunPod).
  - **With .png**: ~10–20 minutes, ~$0.08–$0.26 (RTX 4090), ~$0.13–$0.40 (A100).

---

### Notes

- **GPU Choice**: RTX 4090 is sufficient for 100 samples with LoRA, cost-effective for prototyping (~$0.15–$0.51 cloud, ~$0.02–$0.04 local). A100 80GB is faster but pricier (~$0.20–$0.70).
- **.png Availability**: If `.png` files are missing, rendering is required, increasing time/cost. Verify with `dataset.filter(lambda x: x['render_path'] is not None).num_rows`.
- **Trellix vs. TRELLIS**: Assumed TRELLIS. If Trellix is a custom model, provide details (e.g., framework, size), and I’ll adjust.
- **Errors**: Handle VRAM issues with smaller batch sizes or gradient accumulation. Share errors for debugging.

Run the steps and let me know if you need help with `gsplat` implementation, config tweaks, or troubleshooting!
