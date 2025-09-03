import os
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import torch

from diffusers import DiffusionPipeline
from trellis.pipelines import TrellisImageTo3DPipeline
import pybase64
import requests
from rembg import remove, new_session

torch.cuda.empty_cache()

model_id = "stabilityai/stable-diffusion-2"
rembg_session = new_session('u2net')

t2i_pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    dtype=torch.float16
).to("cuda")

angles = ["front view", "side view", "back view"]
# t2i_pipe.unet = t2i_pipe.unet.half()
# t2i_pipe.vae = t2i_pipe.vae.half()
# t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

i23_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
i23_pipeline.cuda()

prompts_file = open("/workspace/vol_sub17/prompts.txt", "r")
cnt = 0
while cnt < 300 :
    torch.cuda.empty_cache()
    prompt = prompts_file.readline()
    images = []

    # Generate images for each angle
    for angle in angles:
        # Combine base prompt with angle description
        full_prompt = f"{prompt}, {angle}, white background, 3d style, best quality, high resolution"

        # Generate image
        image = t2i_pipe(
            full_prompt,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]

        # Remove background
        image = remove(image, session=rembg_session)

        images.append(image)
    

    # Run the pipeline
    try:
        outputs = i23_pipeline.run_multi_image(images,seed=1,
            sparse_structure_sampler_params={
                "steps": 30,
                "cfg_strength": 8,
            },
            slat_sampler_params={
                "steps": 30,
                "cfg_strength": 4,
            }
        )
    except ValueError:  #raised if `y` is empty.
        continue

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
        print(validation_score)
    cnt=cnt+1
    print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
    print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")
