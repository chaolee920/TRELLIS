import pyspz
import os
import pandas as pd
import open3d as o3d
import trimesh
import torch
from multiprocessing import Pool, cpu_count
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import estimate_pointcloud_normals
from pytorch3d.io import save_obj
def convert_ply_to_mesh(ply_path, file_dir):
    obj_path = os.path.join(file_dir, "model.obj")
    try:
        # Read point cloud (CPU)
        pcd = o3d.io.read_point_cloud(ply_path)
        points = torch.tensor(pcd.points, dtype=torch.float32).cuda()  # Move to GPU

        # Estimate normals on GPU
        pointcloud = Pointclouds(points=[points])
        normals = estimate_pointcloud_normals(pointcloud, neighborhood_size=30)

        # Convert back to Open3D for Poisson reconstruction (CPU fallback)
        pcd.normals = o3d.utility.Vector3dVector(normals[0].cpu().numpy())
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        o3d.io.write_triangle_mesh(obj_path, mesh)

        # Validate and fix mesh (CPU)
        trimesh_mesh = trimesh.load(obj_path)
        if not trimesh_mesh.is_watertight:
            trimesh_mesh.fill_holes()
            trimesh_mesh.export(obj_path)

        print(f"Converted {ply_path} to {obj_path}")
    except Exception as e:
        print(f"Error converting {ply_path}: {e}")
def decompress_spz_files(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    for idx, row in df.iterrows():
        file_dir = os.path.join(output_dir, row['file_identifier'])
        os.makedirs(file_dir, exist_ok=True)
        spz_path = row['model_path']
        render_path = row['render_path']
        ply_path = os.path.join(file_dir, f"gs.ply")
        png_path = os.path.join(file_dir, f"render.png")
        text_path = os.path.join(file_dir, f"prompt.txt")
        try:
            # Decompress spz and push to file_identifier directory
            with open(spz_path, 'rb') as f:
                compressed = f.read()
            decompressed = pyspz.decompress(compressed, include_normals=True)
            with open(ply_path, 'wb') as f:
                f.write(decompressed)
            
            # Convert point cloud to mesh .obj
            obj_path = os.path.join(file_dir, "model.obj")
            
            convert_ply_to_mesh(ply_path, file_dir)



            # push png to file_identifier directory
            with open(render_path, 'rb') as f:
                png = f.read()
            with open(png_path, 'wb') as f:
                f.write(png)
            with open(text_path, 'wb') as f:
                f.write(row['prompt'].encode('utf-8'))
            # df.at[idx, 'model_path'] = obj_path
            # df.at[idx, 'render_path'] = png_path
            # os.remove(spz_path)
            # os.remove(ply_path)
        except Exception as e:
            print(f"Error Pushing : {e}")
    
    df.to_csv(csv_path, index=False)
    return df

if __name__ == "__main__":
    csv_path = "datasets/404mini/metadata.csv"
    output_dir = "datasets/404mini"
    try:
        df = decompress_spz_files(csv_path, output_dir)
        print(f"Decompressed files saved to {output_dir}")
    except Exception as e:
        print(f"Error: {e}")