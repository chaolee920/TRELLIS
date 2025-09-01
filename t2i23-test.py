import os
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import torch
# Pipeline for Flux
from diffusionkit.mlx import FluxPipeline

from trellis.pipelines import TrellisImageTo3DPipeline
import pybase64
import requests

torch.cuda.empty_cache()


t2i_pipe = FluxPipeline(
  shift=1.0,
  model_version="argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
  low_memory_mode=True,
  a16=True,
  w16=True,
)

# Image Generation
HEIGHT = 512
WIDTH = 512
NUM_STEPS = 4
CFG_WEIGHT = 0

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

prompts_file = open("/workspace/vol_sub17/prompts.txt", "r")
cnt = 0
while cnt < 100 :
    prompt = prompts_file.readline()

    image, _ = t2i_pipe.generate_image(
        prompt,
        cfg_weight=CFG_WEIGHT,
        num_steps=NUM_STEPS,
        latent_size=(HEIGHT // 8, WIDTH // 8),
    )

    outputs = pipeline.run(image,seed=1)

    # Render the outputs
    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample.ply")
    with open("./sample.ply", "rb") as file:
        file_data = file.read()
    encoded_data = pybase64.b64encode(file_data).decode("utf-8")
    validate_url = 'http://127.0.0.1:8094/validate_txt_to_3d_ply'
    response = requests.post(validate_url, json={"prompt": prompt, "data": encoded_data})
    if response.status_code == 200:
        results_validation = response.json()

        validation_score = float(results_validation["score"])
    cnt=cnt+1
