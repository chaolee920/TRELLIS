import open3d as o3d
import os
import pandas as pd
from multiprocessing import Pool, cpu_count

def convert_ply_to_mesh(args):
    ply_path, obj_path = args
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        o3d.io.write_triangle_mesh(obj_path, mesh)
        print(f"Converted {ply_path} to {obj_path}")
        return True
    except Exception as e:
        print(f"Error converting {ply_path}: {e}")
        return False

def main(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    tasks = []
    for idx, row in df.iterrows():
        ply_path = row['model_path']
        obj_path = os.path.join(mesh_dir, f"{row['uid']}.obj")
        tasks.append((ply_path, obj_path))
        df.at[idx, 'model_path'] = obj_path
    
    # Parallel processing
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(convert_ply_to_mesh, tasks)
    
    # Update CSV with valid conversions
    df = df[results]  # Keep only successful conversions
    df.to_csv(csv_path, index=False)
    print(f"Processed {len(df)} successful conversions")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.csv_path, args.output_dir)