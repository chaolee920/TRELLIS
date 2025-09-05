from datasets import load_dataset_builder, get_dataset_split_names
import os
import json
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
import io

# Define the output directory
data_dir = "/workspace/.hf_home/datasets/404-data/raw"
os.makedirs(data_dir, exist_ok=True)

# Initialize dataset builder
builder = load_dataset_builder("404-Gen/404mini", revision="83d09af0c72cce9b505562e31e408e6abf70cf11")

# Download raw files (this accesses the repository files)
builder.download_and_prepare(
    download_mode="force_redownload",
    output_dir=data_dir
)

# Define expected schema for TRELLIS
features = Features({
    "image": Image(decode=True),
    "ply": Value("binary"),
    "prompt": Value("string")
})

# Process JSON files manually
data = []
json_files = [f for f in os.listdir(os.path.join(data_dir)) if f.endswith(".json")]
for json_file in json_files:
    with open(os.path.join(data_dir, json_file), "r") as f:
        try:
            entry = json.load(f)
            # Keep only required columns, ignore others
            processed_entry = {
                "image": entry.get("image"),
                "ply": entry.get("ply"),
                "prompt": entry.get("prompt")
            }
            # Validate required columns
            if all(k in entry and entry[k] is not None for k in ["image", "ply", "prompt"]):
                data.append(processed_entry)
            else:
                print(f"Skipping {json_file}: Missing required columns")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data, features=features)
dataset.save_to_disk("/path/to/404-data/processed")