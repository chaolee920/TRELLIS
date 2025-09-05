import pyspz
import os
import pandas as pd
import open3d as o3d
import trimesh


def decompress_spz_files(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    for idx, row in df.iterrows():
        file_dir = os.path.join(output_dir, row['file_identifier'])
        os.makedirs(file_dir, exist_ok=True)
        spz_path = row['model_path']
        render_path = row['render_path']
        ply_path = os.path.join(file_dir, f"gs.ply")
        png_path = os.path.join(file_dir, f"render.png")
        try:
            # Decompress spz and push to file_identifier directory
            with open(spz_path, 'rb') as f:
                compressed = f.read()
            decompressed = pyspz.decompress(compressed, include_normals=True)
            with open(ply_path, 'wb') as f:
                f.write(decompressed)
            # pc_ply_path = os.path.join(file_dir, "pointcloud.ply")
            # os.system(f"python /workspace/proj-sub17/3DGS-to-PC/gauss_to_pc.py --input {ply_path} --output {pc_ply_path} --no_render_colours")

            # Convert point cloud to mesh .obj
            obj_path = os.path.join(file_dir, "model.obj")
            pcd = o3d.io.read_point_cloud(ply_path)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            o3d.io.write_triangle_mesh(obj_path, mesh)

            # Validate and fix mesh
            trimesh_mesh = trimesh.load(obj_path)
            if not trimesh_mesh.is_watertight:
                trimesh_mesh.fill_holes()
                trimesh_mesh.export(obj_path)



            # push png to file_identifier directory
            with open(render_path, 'rb') as f:
                png = f.read()
            with open(png_path, 'wb') as f:
                f.write(png)
            df.at[idx, 'model_path'] = obj_path
            df.at[idx, 'render_path'] = png_path
            os.remove(spz_path)
            os.remove(ply_path)
            os.remove(pc_ply_path)
        except Exception as e:
            print(f"Error Pushing {spz_path}, {render_path}: {e}")
    
    df.to_csv(csv_path, index=False)
    return df

if __name__ == "__main__":
    csv_path = "datasets/404mini/404mini_5.csv"
    output_dir = "datasets/404mini"
    try:
        df = decompress_spz_files(csv_path, output_dir)
        print(f"Decompressed files saved to {output_dir}")
    except Exception as e:
        print(f"Error: {e}")