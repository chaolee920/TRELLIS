from PIL import Image
import torch
from diffusers import DiffusionPipeline
from rembg import remove, new_session

# model_id = "stabilityai/stable-diffusion-2"
model_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

t2i_pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    dtype=torch.float16
).to("cuda")


prompt = input()

image = t2i_pipe(
    prompt + ", white background, 3d style, best quality, high resolution",
    negative_prompt="Text, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
            "Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
            "Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
            "Extra arms, extra legs, fused fingers, too many fingers, long neck",
    guidance_scale=6.0, # Example value, adjust for desired output
    num_inference_steps=25, # Example value, adjust for desired quality/speed
).images[0]

rembg_session = new_session('u2net')
image = remove(image, session=rembg_session)

image.save(f"./output.png")