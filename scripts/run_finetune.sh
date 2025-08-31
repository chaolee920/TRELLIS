cd ..
torchrun --nproc_per_node=1 train.py \
  --config configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16.json \
  --output_dir ../outputs/trellis-text-xl-404mini \
  --resume_from /workspace/vol_sub17/models/TRELLIS-text-xlarge