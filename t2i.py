from PIL import Image
import torch
from diffusers import DiffusionPipeline
from rembg import remove

# model_id = "stabilityai/stable-diffusion-2"
model_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

t2i_pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    dtype=torch.float16
).to("cuda")

t2i_pipe.transformer = t2i_pipe.transformer.half()
t2i_pipe.vae = t2i_pipe.vae.half()
t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

prompt = input()

image = t2i_pipe(
    prompt + ", white background, 3d style, whole body, cartoon asset",
    negative_prompt="Text, flasy, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
            "Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
            "Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
            "Extra arms, extra legs, fused fingers, too many fingers, long neck",
    guidance_scale=7.5, # Example value, adjust for desired output
    num_inference_steps=25, # Example value, adjust for desired quality/speed
).images[0]

image = remove(image, alpha_matting=True, alpha_matting_foreground_threshold=240)

imge = image.resize((512, 512))

image.save(f"./output.png")