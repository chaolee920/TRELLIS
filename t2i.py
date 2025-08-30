from PIL import Image
import torch
from diffusers import DiffusionPipeline
from rembg import remove, new_session

pipe = DiffusionPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled").to("cuda")

prompt = input()
image = pipe(prompt + ", white background, 3D style, best quality", negative_prompt="Text, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
"Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
"Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
"Extra arms, extra legs, fused fingers, too many fingers, long neck", num_inference_steps=25, width=512, height=512,  guidance_scale=3.5).images[0]

output = remove(image, session=new_session(), bgcolor=[255, 255, 255, 0])

output.save('output.png')
