import argparse
import os
from transformers import AutoModel, AutoFeatureExtractor
import torch
import pandas as pd
from PIL import Image

def main(args):
    df = pd.read_csv(os.path.join(args.output_dir, f"{args.dataset_name}.csv"))
    feature_dir = os.path.join(args.output_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)
    
    extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits16")
    model = AutoModel.from_pretrained("facebook/dino-vits16").cuda()
    
    for idx, row in df.iterrows():
        features = []
        if row['render_path'] and os.path.exists(row['render_path']):
            img_path = row['render_path']
            image = Image.open(img_path).convert("RGB")
            inputs = extractor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            features.append(outputs.last_hidden_state.squeeze().cpu())
        else:
            render_dir = os.path.join(args.output_dir, "renders", row['uid'])
            for view_idx in range(args.num_views):
                img_path = os.path.join(render_dir, f"view_{view_idx}.png")
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert("RGB")
                    inputs = extractor(images=image, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    features.append(outputs.last_hidden_state.squeeze().cpu())
        
        if features:
            torch.save(features, os.path.join(feature_dir, f"{row['uid']}.pt"))
        else:
            print(f"No images found for {row['uid']}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=8)
    args = parser.parse_args()
    main(args)