from huggingface_hub import list_repo_files, hf_hub_download
import os
import pandas as pd
from datasets import Dataset, load_from_disk

def load_404mini_dataset(repo_id="404-Gen/404mini", output_dir="datasets/404mini", max_samples=None, log_file="download_log.txt"):
    os.makedirs(output_dir, exist_ok=True)
    
    spz_dir = os.path.join(output_dir, "spz")
    render_dir = os.path.join(output_dir, "renders")
    os.makedirs(spz_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    
    # Load start point from log file
    log_path = os.path.join(output_dir, log_file)
    start_index = 0
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                start_index = int(f.read().strip())
                print(f"Resuming from index {start_index}")
            except ValueError:
                print("Invalid log file; starting from index 0")
    
    # Get list of files
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    json_files = [f for f in files if f.endswith(".json") and f.startswith("assets/")]
    ply_files = [f for f in files if f.endswith(".ply.spz") and f.startswith("assets/")]
    png_files = [f for f in files if f.endswith(".png") and f.startswith("assets/")]
    
    # Limit to max_samples if specified
    json_files = json_files[:max_samples] if max_samples is not None else json_files
    print(f"Processing up to {len(json_files)} JSON files from index {start_index}")
    
    # Load existing CSV if it exists
    disk_path = os.path.join(output_dir, f"404mini_{max_samples or 'full'}")
    if os.path.exists(disk_path):
        df_existing = load_from_disk(disk_path)
        data = df_existing.to_dict('records')
    else:
        data = []
    
    # Process files starting from start_index
    for i, json_file in enumerate(json_files[start_index:], start=start_index):
        try:
            id_str = f"item_{i:05d}"
            asset_dir = os.path.join(output_dir, "404-model", id_str)
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            category = json_file.split('/')[1]
            ply_file = f"assets/{category}/{base_name}.ply.spz"
            png_file = f"assets/{category}/{base_name}.png"
            
            if ply_file in ply_files:
                prompt = base_name.replace("_", " ")
                spz_path = os.path.join(spz_dir, 'assets', category, f"{base_name}.ply.spz")
                hf_hub_download(repo_id=repo_id, filename=ply_file, repo_type="dataset", local_dir=spz_dir)
                
                render_path = None
                if png_file in png_files:
                    render_path = os.path.join(render_dir, 'assets', category,  f"{base_name}.png")
                    hf_hub_download(repo_id=repo_id, filename=png_file, repo_type="dataset", local_dir=render_dir)
                
                data.append({
                    "uid": base_name,
                    "name": base_name,
                    "captions": [prompt],
                    "aesthetic_score": 5.0,
                    "model_path": spz_path,
                    "render_path": render_path
                })
                
                # Update log file with current index
                with open(log_path, 'w') as f:
                    f.write(str(i + 1))
                print(f"Processed {json_file}, updated log to index {i + 1}")
            else:
                print(f"Skipping {json_file}: No matching .ply.spz")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not data:
        raise ValueError("No valid samples found with matching .ply.spz files")
    
    # Save metadata
    df = pd.DataFrame(data)
    df.save_to_disk(disk_path, index=False)
    dataset = Dataset.from_pandas(df)
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Metadata saved to {disk_path}")
    return dataset, disk_path

if __name__ == "__main__":
    try:
        dataset, disk_path = load_404mini_dataset(max_samples=5000)  # Set to 5 for 404mini_5.csv
        print(dataset[:5])
    except Exception as e:
        print(f"Error: {e}")