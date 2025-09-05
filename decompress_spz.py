import pyspz
import os
import pandas as pd

def decompress_spz_files(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    
    
    for idx, row in df.iterrows():
        file_dir = os.path.join(output_dir, "files", row['file_identifier'])
        os.makedirs(file_dir, exist_ok=True)
        spz_path = row['model_path']
        render_path = row['render_path']
        ply_path = os.path.join(file_dir, f"{row['uid']}.ply")
        png_path = os.path.join(file_dir, f"{row['uid']}.png")
        try:
            # Decompress spz and push to file_identifier directory
            with open(spz_path, 'rb') as f:
                compressed = f.read()
            decompressed = pyspz.decompress(compressed, include_normals=True)
            with open(ply_path, 'wb') as f:
                f.write(decompressed)
            # push png to file_identifier directory
            with open(render_path, 'rb') as f:
                png = f.read()
            with open(png_path, 'wb') as f:
                f.write(png)
            df.at[idx, 'model_path'] = ply_path
            df.at[idx, 'render_path'] = png_path
        except Exception as e:
            print(f"Error Pushing {spz_path}, {render_path}: {e}")
    
    df.to_csv(csv_path, index=False)
    return df

if __name__ == "__main__":
    csv_path = "datasets/404mini/404mini_5.csv"
    output_dir = "datasets/404mini"
    try:
        df = decompress_spz_files(csv_path, output_dir)
        print(f"Decompressed files saved to {output_dir}/ply")
    except Exception as e:
        print(f"Error: {e}")