import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import spconv.pytorch as spconv

class SimpleSLATEncoder(nn.Module):
    """Placeholder SLAT encoder for sparse tensors."""
    def __init__(self, input_channels=1, latent_dim=128):
        super().__init__()
        self.conv = spconv.SparseSequential(
            spconv.SparseConv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.ToDense()
        )
        self.fc = nn.Linear(32 * 32 * 32 * 32, latent_dim)  # Adjust for resolution

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def encode_slat_latent(feature_path, resolution=64):
    """Encode features into SLAT latent."""
    try:
        # Load features (list of DINO features from extract_features.py)
        features = torch.load(feature_path, weights_only=True)  # List of [C, H, W]
        if not isinstance(features, list):
            raise ValueError(f"Invalid feature format: {feature_path}")
        
        # Convert to sparse tensor (simplified; assumes features are patch tokens)
        indices = torch.zeros((len(features), 4), dtype=torch.int32, device='cuda')  # [N, 4]
        for i in range(len(features)):
            indices[i, 0] = i  # Batch index
        feats = torch.stack(features, dim=0).mean(dim=(2, 3))  # [N, C]
        
        sparse_tensor = spconv.SparseConvTensor(
            features=feats,
            indices=indices,
            spatial_shape=[resolution, resolution, resolution],
            batch_size=len(features)
        )
        
        # Encode with placeholder encoder
        encoder = SimpleSLATEncoder(input_channels=feats.shape[1]).cuda()
        encoder.eval()
        with torch.no_grad():
            latent = encoder(sparse_tensor)
        return latent.cpu()
    except Exception as e:
        print(f"Error encoding {feature_path}: {e}")
        return None

def main(args):
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    latent_dir = os.path.join(output_dir, "latents")
    os.makedirs(latent_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        feature_path = os.path.join(output_dir, "features", f"{row['uid']}.pt")
        if not os.path.exists(feature_path):
            print(f"Warning: Feature path {feature_path} does not exist")
            continue
        
        latent = encode_slat_latent(feature_path)
        if latent is not None:
            output_path = os.path.join(latent_dir, f"{row['uid']}.pt")
            torch.save(latent, output_path)
            print(f"Saved SLAT latent to {output_path}")
    
    print(f"SLAT encoding complete for {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of dataset (e.g., 404mini_5)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument("--feat_model", type=str, default="dinov2_vitl14_reg", help="Feature model")
    parser.add_argument("--enc_pretrained", type=str, default="microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16", help="Pretrained encoder model")
    args = parser.parse_args()
    main(args)