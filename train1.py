import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict
import torch
import torch.multiprocessing as mp
import numpy as np
import random
import pandas as pd

# Placeholder for trellis imports (replace with actual imports if available)
try:
    from trellis import models, trainers
    from trellis.utils.dist_utils import setup_dist
except ImportError:
    print("Warning: trellis module not found; using placeholder")
    models = trainers = setup_dist = None

class Custom404MiniDataset:
    """Custom dataset for 404-Gen/404mini."""
    def __init__(self, data_dir, dataset_name="404mini_5"):
        self.csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} not found")
        self.df = pd.read_csv(self.csv_path)
        self.latent_dir = os.path.join(data_dir, "latents")
        self.feature_dir = os.path.join(data_dir, "features")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        latent_path = os.path.join(self.latent_dir, f"{row['uid']}.pt")
        feature_path = os.path.join(self.feature_dir, f"{row['uid']}.pt")
        if not os.path.exists(latent_path) or not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing latent or feature for {row['uid']}")
        latent = torch.load(latent_path, weights_only=True)
        features = torch.load(feature_path, weights_only=True)
        caption = row['captions'][0] if isinstance(row['captions'], list) else row['captions']
        return {
            'latent': latent,
            'features': features,
            'caption': caption,
            'uid': row['uid']
        }

def find_ckpt(cfg):
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg

def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)

def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary

def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
   -World_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    # Load data
    dataset = Custom404MiniDataset(cfg.data_dir, dataset_name=cfg.dataset_name)

    # Build model
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt)

    # Train
    if not cfg.tryrun