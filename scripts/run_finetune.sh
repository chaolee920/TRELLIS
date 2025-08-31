#!/bin/bash
torchrun --nproc_per_node=4 trainer.py \
  --config configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16.json \
  --output_dir ../outputs/trellis-text-xl-404mini \
  --data_dir ../data/404mini/train \
  --ckpt latest