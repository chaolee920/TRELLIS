import argparse
import os
import pandas as pd
import torch
import spconv.pytorch as spconv

def encode_sparse_structure(voxel_path, resolution=64):
    """Encode voxel grid into sparse structural latent."""
    try:
        # Load voxel grid
        voxel_grid = torch.load(voxel_path, weights_only=True)  # [resolution, resolution, resolution]
        if voxel_grid.shape != (resolution, resolution, resolution):
            raise ValueError(f"Invalid voxel grid shape: {voxel_grid.shape}")
        
        # Convert to sparse tensor
        indices = torch.nonzero(voxel_grid).long()  # [N, 3]
        # Add batch dimension: [N, 3] -> [N, 4] with batch index 0
        batch_indices = torch.zeros((indices.shape[0], 1), dtype=torch.long, device=indices.device)
        indices = torch.cat([batch_indices, indices], dim=1)  # [N, 4]
        features = torch.ones((indices.shape[0], 1), dtype=torch.float32)  # Dummy features
        
        sparse_tensor = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=[resolution, resolution, resolution],
            batch_size=1
        )
        
        # Placeholder for further encoding (e.g., octree or convolution)
        return sparse_tensor
    except Exception as e:
        print(f"Error encoding {voxel_path}: {e}")
        return None

def main(args):
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    ss_dir = os.path.join(output_dir, "sparse_structures")
    os.makedirs(ss_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        voxel_path = os.path.join(output_dir, "voxels", f"{row['uid']}.pt")
        if not os.path.exists(voxel_path):
            print(f"Warning: Voxel path {voxel_path} does not exist")
            continue
        
        sparse_tensor = encode_sparse_structure(voxel_path)
        if sparse_tensor is not None:
            output_path = os.path.join(ss_dir, f"{row['uid']}.pt")
            torch.save(sparse_tensor, output_path)
            print(f"Saved sparse structure to {output_path}")
    
    print(f"Sparse structure encoding complete for {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of dataset (e.g., 404mini_5)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution")
    args = parser.parse_args()
    main(args)