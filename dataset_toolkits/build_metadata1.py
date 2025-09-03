import argparse
import pandas as pd
import os

def main(args):
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    
    # Load and validate CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    required_columns = ['uid', 'name', 'source', 'captions', 'model_path']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in {csv_path}: {required_columns}")
    
    # Validate model paths
    for idx, row in df.iterrows():
        model_path = row['model_path']
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist")
            df.at[idx, 'model_path'] = None
    
    # Validate render paths (for image-to-3D)
    if 'render_path' in df.columns:
        for idx, row in df.iterrows():
            render_path = row['render_path']
            if render_path and not os.path.exists(render_path):
                print(f"Warning: Render path {render_path} does not exist")
                df.at[idx, 'render_path'] = None
    
    # Save processed metadata
    os.makedirs(output_dir, exist_ok=True)
    processed_csv = os.path.join(output_dir, f"{dataset_name}_processed.csv")
    df.to_csv(processed_csv, index=False)
    print(f"Metadata built for {dataset_name} at {processed_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of dataset (e.g., 404mini_100)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    main(args)