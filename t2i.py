from PIL import Image
import torch
from diffusers import DiffusionPipeline
from rembg import remove, new_session

model_id = "stabilityai/stable-diffusion-2"

t2i_pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    dtype=torch.float16
).to("cuda")


prompt = input()

image = t2i_pipe(
    prompt + ", white background, 3d style, best quality, high resolution",
    guidance_scale=15, # Example value, adjust for desired output
    num_inference_steps=25, # Example value, adjust for desired quality/speed
).images[0]

rembg_session = new_session('u2net')
image = remove(image, session=rembg_session)

image.save(f"./output.png")