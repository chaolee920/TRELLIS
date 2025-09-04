import argparse
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import open3d as o3d
from gsplat import fully_fused_projection, rasterize_to_pixels

def load_gaussian_splat(ply_path):
    """Load .ply file as Gaussian splat (points, colors, scales, rotations, opacities)."""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)  # [N, 3]
    colors = np.asarray(pcd.colors, dtype=np.float32)  # [N, 3]
    
    # Dummy scales, rotations, opacities
    scales = np.ones((points.shape[0], 3), dtype=np.float32) * 0.01
    rotations = np.zeros((points.shape[0], 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # Identity quaternion
    opacities = np.ones((points.shape[0], 1), dtype=np.float32) * 0.9
    
    return (
        torch.tensor(points, device='cuda', dtype=torch.float32),
        torch.tensor(colors, device='cuda', dtype=torch.float32),
        torch.tensor(scales, device='cuda', dtype=torch.float32),
        torch.tensor(rotations, device='cuda', dtype=torch.float32),
        torch.tensor(opacities, device='cuda', dtype=torch.float32)
    )

def compute_camtoworld(azimuth, elevation, radius):
    """Compute camera-to-world matrix."""
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    
    eye = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    z_axis = (target - eye) / np.linalg.norm(target - eye)
    x_axis = np.cross(up, z_axis) / np.linalg.norm(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    camtoworld = np.array([
        [x_axis[0], y_axis[0], z_axis[0], eye[0]],
        [x_axis[1], y_axis[1], z_axis[1], eye[1]],
        [x_axis[2], y_axis[2], z_axis[2], eye[2]],
        [0, 0, 0, 1]
    ])
    return torch.tensor(camtoworld, device='cuda', dtype=torch.float32)

def compute_intrinsics(fovy, width, height):
    """Compute camera intrinsics."""
    f = 1.0 / np.tan(fovy / 2)
    fx = width / (2 * np.tan(fovy / 2))
    fy = height / (2 * np.tan(fovy / 2))
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy

def render_gaussian_splat(ply_path, output_dir, num_views=8):
    """Render multiview images from Gaussian splat .ply file."""
    points, colors, scales, rotations, opacities = load_gaussian_splat(ply_path)
    os.makedirs(output_dir, exist_ok=True)
    
    height, width = 256, 256
    fovy = np.deg2rad(45)
    fx, fy, cx, cy = compute_intrinsics(fovy, width, height)
    
    for view_idx in range(num_views):
        azimuth = view_idx * 360.0 / num_views
        camtoworld = compute_camtoworld(azimuth=azimuth, elevation=0, radius=2.0)
        
        # Project Gaussians
        xys, depths, radii, conics, comp, num_tiles_hit, cov3ds = fully_fused_projection(
            means=points,
            scales=scales,
            quats=rotations,
            camtoworld=camtoworld,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_height=height,
            img_width=width
        )
        
        # Rasterize to pixels
        image, _ = rasterize_to_pixels(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            colors=colors,
            opacities=opacities,
            img_height=height,
            img_width=width
        )  # Shape: [H, W, 3]
        
        # Convert to PIL Image and save
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(image).save(os.path.join(output_dir, f"view_{view_idx}.png"))

def main(args):
    csv_path = os.path.join(args.output_dir, f"{args.dataset_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    render_dir = os.path.join(args.output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        ply_path = row['model_path']
        output_subdir = os.path.join(render_dir, row['uid'])
        try:
            render_gaussian_splat(ply_path, output_subdir, args.num_views)
        except Exception as e:
            print(f"Error rendering {ply_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of dataset (e.g., 404mini_100)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views to render")
    args = parser.parse_args()
    main(args)