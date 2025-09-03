# prepare_train.py
from huggingface_hub import list_repo_files, hf_hub_download
import json
import os
from datasets import Dataset
import pandas as pd

def load_404mini_dataset(repo_id="404-Gen/404mini", output_dir="datasets/404mini"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    spz_dir = os.path.join(output_dir, "spz")
    os.makedirs(spz_dir, exist_ok=True)
    
    # Get list of JSON files
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    json_files = [f for f in files if f.endswith(".json") and f.startswith("assets/")]
    
    # Process JSON files
    data = []
    for file in json_files:
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=file, repo_type="dataset")
            with open(local_path, 'r') as f:
                sample = json.load(f)
            
            # Check for required fields
            if 'prompt' in sample and 'model' in sample:
                uid = os.path.basename(file).replace(".json", "")
                spz_path = os.path.join(spz_dir, f"{uid}.ply.spz")
                
                # Save .ply.spz file
                with open(spz_path, 'wb') as f:
                    f.write(sample['model'])  # Assumes model is bytes; decode if base64 string
                data.append({
                    "uid": uid,
                    "name": uid,
                    "source": "404mini",
                    "captions": [sample['prompt']],
                    "aesthetic_score": 5.0,  # Placeholder
                    "model_path": spz_path,
                    "render": sample.get('render', None)  # Optional
                })
            else:
                print(f"Skipping {file}: Missing 'prompt' or 'model'")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Create Dataset and save metadata
    df = pd.DataFrame(data)
    if len(df) == 0:
        raise ValueError("No valid JSON files found with 'prompt' and 'model'")
    
    csv_path = os.path.join(output_dir, "404mini.csv")
    df.to_csv(csv_path, index=False)
    dataset = Dataset.from_pandas(df)
    
    print(f"Loaded {len(dataset)} valid samples")
    print(f"Metadata saved to {csv_path}")
    return dataset, csv_path

if __name__ == "__main__":
    try:
        dataset, csv_path = load_404mini_dataset()
        print(dataset[:5])  # Inspect first 5 samples
    except Exception as e:
        print(f"Error: {e}")