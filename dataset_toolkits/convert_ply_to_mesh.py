import open3d as o3d
import os
import pandas as pd

def convert_ply_to_mesh(ply_path, obj_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    o3d.io.write_triangle_mesh(obj_path, mesh)

def main(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        ply_path = row['model_path']
        obj_path = os.path.join(mesh_dir, f"{row['uid']}.obj")
        try:
            convert_ply_to_mesh(ply_path, obj_path)
            df.at[idx, 'model_path'] = obj_path
            print('convert done')
        except Exception as e:
            print(f"Error converting {ply_path}: {e}")
    
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.csv_path, args.output_dir)