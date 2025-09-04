import argparse
import os
from transformers import AutoModel, ViTImageProcessor
import torch
import pandas as pd
from PIL import Image

def main(args):
    df = pd.read_csv(os.path.join(args.output_dir, f"{args.dataset_name}.csv"))
    feature_dir = os.path.join(args.output_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)
    
    # Load DINO model with safetensors
    processor = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
    model = AutoModel.from_pretrained("facebook/dino-vits16", use_safetensors=True).cuda()
    
    for idx, row in df.iterrows():
        features = []
        render_path = row.get('render_path')
        if render_path and os.path.exists(render_path):
            # Use single-view .png if available
            img_path = render_path
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model(**inputs)
                features.append(outputs.last_hidden_state.squeeze().cpu())
                print(f"Extracted features from {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        else:
            # Fallback to multiview renders
            render_dir = os.path.join(args.output_dir, "renders", row['uid'])
            for view_idx in range(args.num_views):
                img_path = os.path.join(render_dir, f"view_{view_idx}.png")
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert("RGB")
                        inputs = processor(images=image, return_tensors="pt").to("cuda")
                        with torch.no_grad():
                            outputs = model(**inputs)
                        features.append(outputs.last_hidden_state.squeeze().cpu())
                        print(f"Extracted features from {img_path}")
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        if features:
            output_path = os.path.join(feature_dir, f"{row['uid']}.pt")
            torch.save(features, output_path)
            print(f"Saved features to {output_path}")
        else:
            print(f"No images found for {row['uid']}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=8)
    args = parser.parse_args()
    main(args)