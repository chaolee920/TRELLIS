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

if __name__ == "__main__":
    csv_path = "datasets/404mini/404mini.csv"
    output_dir = "datasets/404mini"
    try:
        df = decompress_spz_files(csv_path, output_dir)
        print(f"Decompressed files saved to {output_dir}/ply")
    except Exception as e:
        print(f"Error: {e}")