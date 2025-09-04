import argparse
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import open3d as o3d
from gsplat import fully_fused_projection, rasterize_to_pixels, quat_scale_to_covar_preci

def load_gaussian_splat(ply_path):
    """Load .ply file as Gaussian splat (points, colors, scales, rotations, opacities)."""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.ascontiguousarray(pcd.points, dtype=np.float32)  # Ensure contiguous
    colors = np.ascontiguousarray(pcd.colors, dtype=np.float32)
    
    # Dummy scales, rotations, opacities
    scales = np.ascontiguousarray(np.ones((points.shape[0], 3), dtype=np.float32) * 0.01)
    rotations = np.ascontiguousarray(np.zeros((points.shape[0], 4), dtype=np.float32))
    rotations[:, 0] = 1.0  # Identity quaternion
    opacities = np.ascontiguousarray(np.ones((points.shape[0], 1), dtype=np.float32) * 0.9)
    
    # Compute covariances
    covars, _ = quat_scale_to_covar_preci(rotations, scales)
    
    return (
        torch.tensor(points, device='cuda', dtype=torch.float32).contiguous(),
        torch.tensor(colors, device='cuda', dtype=torch.float32).contiguous(),
        torch.tensor(covars, device='cuda', dtype=torch.float32).contiguous(),
        torch.tensor(rotations, device='cuda', dtype=torch.float32).contiguous(),
        torch.tensor(scales, device='cuda', dtype=torch.float32).contiguous(),
        torch.tensor(opacities, device='cuda', dtype=torch.float32).contiguous()
    )

def compute_view_matrix(azimuth, elevation, radius):
    """Compute view matrix (world-to-camera)."""
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    
    eye = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    z_axis = (eye - target) / np.linalg.norm(eye - target)
    x_axis = np.cross(up, z_axis) / np.linalg.norm(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    view_matrix = np.array([
        [x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis, eye)],
        [y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis, eye)],
        [z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, eye)],
        [0, 0, 0, 1]
    ])
    return torch.tensor(view_matrix, device='cuda', dtype=torch.float32).contiguous()

def compute_intrinsics(fovy, width, height):
    """Compute camera intrinsics matrix (K)."""
    fx = width / (2 * np.tan(fovy / 2))
    fy = height / (2 * np.tan(fovy / 2))
    cx = width / 2
    cy = height / 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return torch.tensor(K, device='cuda', dtype=torch.float32).contiguous()

def render_gaussian_splat(ply_path, output_dir, num_views=8):
    """Render multiview images from Gaussian splat .ply file."""
    points, colors, covars, quats, scales, opacities = load_gaussian_splat(ply_path)
    os.makedirs(output_dir, exist_ok=True)
    
    height, width = 256, 256
    fovy = np.deg2rad(45)
    K = compute_intrinsics(fovy, width, height)
    
    # Batch view matrices
    viewmats = []
    for view_idx in range(num_views):
        azimuth = view_idx * 360.0 / num_views
        view_matrix = compute_view_matrix(azimuth=azimuth, elevation=0, radius=2.0)
        viewmats.append(view_matrix)
    viewmats = torch.stack(viewmats, dim=0)  # [num_views, 4, 4]
    
    # Project Gaussians (batched)
    try:
        xys, depths, radii, conics, comp, num_tiles_hit, cov3ds = fully_fused_projection(
            means=points,
            covars=covars,
            quats=quats,
            scales=scales,
            viewmats=viewmats,
            Ks=torch.stack([K] * num_views, dim=0),
            width=width,
            height=height,
            near_plane=0.1,
            far_plane=100.0
        )
        
        # Rasterize to pixels (batched)
        images, _ = rasterize_to_pixels(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            colors=colors.repeat(num_views, 1, 1),
            opacities=opacities.repeat(num_views, 1, 1),
            img_height=height,
            img_width=width
        )  # Shape: [num_views, H, W, 3]
        
        # Write example: Save rendered images as PNG
        for view_idx in range(num_views):
            image = (images[view_idx].cpu().numpy() * 255).astype(np.uint8)
            output_path = os.path.join(output_dir, f"view_{view_idx}.png")
            Image.fromarray(image).save(output_path)
            print(f"Saved rendered image to {output_path}")
        
    except Exception as e:
        print(f"Error rendering {ply_path}: {e}")

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