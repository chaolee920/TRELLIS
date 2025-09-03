from huggingface_hub import list_repo_files, hf_hub_download
import os
import pandas as pd
from datasets import Dataset

def load_404mini_dataset(repo_id="404-Gen/404mini", output_dir="datasets/404mini", max_samples=100):
    os.makedirs(output_dir, exist_ok=True)
    spz_dir = os.path.join(output_dir, "spz")
    os.makedirs(spz_dir, exist_ok=True)
    
    # Get list of files
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    json_files = [f for f in files if f.endswith(".json") and f.startswith("assets/")]
    ply_files = [f for f in files if f.endswith(".ply.spz") and f.startswith("assets/")]
    png_files = [f for f in files if f.endswith(".png") and f.startswith("assets/")]
    
    data = []
    for json_file in json_files:
        try:
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            category = json_file.split('/')[1]  # e.g., 'an'
            ply_file = f"assets/{category}/{base_name}.ply.spz"
            png_file = f"assets/{category}/{base_name}.png"
            
            if ply_file in ply_files:
                # Derive prompt from file name
                prompt = base_name.replace("_", " ")
                
                # Download .ply.spz
                spz_path = os.path.join(spz_dir, f"{base_name}.ply.spz")
                hf_hub_download(repo_id=repo_id, filename=ply_file, repo_type="dataset", local_dir=spz_dir)
                
                # Check for .png (optional)
                render_path = None
                if png_file in png_files:
                    render_path = os.path.join(output_dir, "renders", f"{base_name}.png")
                    os.makedirs(os.path.dirname(render_path), exist_ok=True)
                    hf_hub_download(repo_id=repo_id, filename=png_file, repo_type="dataset", local_dir=os.path.dirname(render_path))
                
                data.append({
                    "uid": base_name,
                    "name": base_name,
                    "source": "404mini",
                    "captions": [prompt],
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
    csv_path = os.path.join(output_dir, "404mini.csv")
    df.to_csv(csv_path, index=False)
    dataset = Dataset.from_pandas(df)
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Metadata saved to {csv_path}")
    return dataset, csv_path

if __name__ == "__main__":
    try:
        dataset, csv_path = load_404mini_dataset(max_samples=100)
        print(dataset[:5])
    except Exception as e:
        print(f"Error: {e}")