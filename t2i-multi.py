from PIL import Image
import torch
from diffusers import DiffusionPipeline
from rembg import remove, new_session

rembg_session = new_session('u2net')
model_id = "stabilityai/stable-diffusion-2"

t2i_pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    dtype=torch.float16
).to("cuda")


prompt = input()
angles = ["front view", "side view", "back view"]

for angle in angles:
    # Combine base prompt with angle description
    full_prompt = f"{prompt}, {angle}, white background, 3d style, best quality, high resolution"

    # Generate image
    image = t2i_pipe(
        full_prompt,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    image = remove(image, session=rembg_session)

    image.save(f"./multi/img_{angle}.png")
    print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
    print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")