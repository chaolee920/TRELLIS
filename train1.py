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
from transformers import AutoTokenizer

# Placeholder for trellis imports
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
        self.loads = list(range(len(self.df)))  # For BalancedResumableSampler
        self.value_range = (-1.0, 1.0)  # Tuple for trainer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True, max_length=128)
        return {
            'latent': latent,
            'features': features,
            'input_ids': tokens['input_ids'].squeeze(0).to(torch.int64),
            'attention_mask': tokens['attention_mask'].squeeze(0).to(torch.int64),
            'uid': row['uid']
        }

    @staticmethod
    def collate_fn(batch):
        """Batch samples for data loader."""
        latents = [item['latent'] for item in batch]
        features = [item['features'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        uids = [item['uid'] for item in batch]
        
        try:
            latents = torch.stack(latents)
        except:
            latents = latents
        try:
            features = [torch.stack(f) if isinstance(f, list) else f for f in features]
        except:
            features = features
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0).to(torch.int64)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0).to(torch.int64)
        
        # Debug prints
        print("Collated batch:")
        print(f"Latents: {type(latents)}, shapes: {[l.shape for l in latents] if isinstance(latents, list) else latents.shape}")
        print(f"Features: {type(features)}, shapes: {[f.shape for f in features] if isinstance(features, list) else features.shape}")
        print(f"Input IDs: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(f"Attention Masks: {attention_masks.shape}, dtype: {attention_masks.dtype}")
        print(f"UIDs: {uids}")
        
        return {
            'latent': latents,
            'features': features,
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'uid': uids
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
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1 and setup_dist is not None:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)
    setup_rng(rank)
    dataset = Custom404MiniDataset(cfg.data_dir, dataset_name=cfg.dataset_name)
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)
    trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt)
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='404mini_5')
    parser.add_argument('--auto_retry', type=int, default=3)
    parser.add_argument('--tryrun', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=-1)
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='12345')
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    config = json.load(open(opt.config, 'r'))
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f'Error: {e}')
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...')