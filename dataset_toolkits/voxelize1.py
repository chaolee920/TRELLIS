import argparse
import os
import pandas as pd
import torch
import open3d as o3d
import numpy as np

def voxelize_model(model_path, resolution=64):
    """Voxelize a .ply point cloud into a voxel grid."""
    try:
        if not model_path.endswith('.ply'):
            raise ValueError(f"Unsupported model format: {model_path}")
        
        pcd = o3d.io.read_point_cloud(model_path)
        # Clamp points to [-0.5, 0.5]
        points = np.clip(np.asarray(pcd.points), -0.5 + 1e-6, 0.5 - 1e-6)
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Create voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=1.0 / resolution
        )
        
        # Convert to tensor
        voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        for voxel in voxel_grid.get_voxels():
            idx = voxel.grid_index
            if idx[0] < resolution and idx[1] < resolution and idx[2] < resolution:
                voxels[idx[0], idx[1], idx[2]] = 1.0
        
        return torch.tensor(voxels, dtype=torch.float32)
    except Exception as e:
        print(f"Error voxelizing {model_path}: {e}")
        return None

def main(args):
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    voxel_dir = os.path.join(output_dir, "voxels")
    os.makedirs(voxel_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        model_path = row['model_path']
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist")
            continue
        
        voxel_grid = voxelize_model(model_path)
        if voxel_grid is not None:
            output_path = os.path.join(voxel_dir, f"{row['uid']}.pt")
            torch.save(voxel_grid, output_path)
            print(f"Saved voxel grid to {output_path}")
    
    print(f"Voxelization complete for {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of dataset (e.g., 404mini_5)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    main(args)