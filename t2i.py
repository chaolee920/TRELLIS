from PIL import Image
import torch
from diffusers import DiffusionPipeline
from rembg import remove, new_session

# pipe = DiffusionPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled").to("cuda")

    # Load the text-to-multi-view pipeline
# text_to_multi_view_pipeline = DiffusionPipeline.from_pretrained(
#     "ashawkey/mvdream-sd2.1-diffusers", # Or another suitable MVDream diffusers port
#     custom_pipeline="dylanebert/multi_view_diffusion",
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
# ).to("cuda") # Move to GPU if available

model_id = "stabilityai/stable-diffusion-2-1-base"

t2i_pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    dtype=torch.float16
).to("cuda")


prompt = input()
# image = pipe(prompt + ", white background, 3D style, best quality", negative_prompt="Text, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
# "Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
# "Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
# "Extra arms, extra legs, fused fingers, too many fingers, long neck", num_inference_steps=25, width=512, height=512,  guidance_scale=3.5).images[0]

# output = remove(image, session=new_session(), bgcolor=[255, 255, 255, 0])

# output.save('output.png')
# Generate images; adjust guidance_scale and num_inference_steps as needed
multi_view_images = t2i_pipe(
    prompt,
    guidance_scale=7.5, # Example value, adjust for desired output
    num_inference_steps=50, # Example value, adjust for desired quality/speed
).images

for i in range(len(multi_view_images)):
    multi_view_images[i].save(f"./MVDREAM/img_{i}.png")