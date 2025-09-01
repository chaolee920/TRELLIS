import os
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import torch

from diffusers import DiffusionPipeline
import pybase64
import requests

torch.cuda.empty_cache()

t2i_pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to("cuda")

pipeline = DiffusionPipeline.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
pipeline.cuda()

prompts_file = open("/workspace/vol_sub17/prompts.txt", "r")
cnt = 0
while cnt < 10 :
    torch.cuda.empty_cache()
    prompt = prompts_file.readline()

    image = t2i_pipe(prompt + ", accurate, consistent, 3D style, whole content", negative_prompt="Text, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
"Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
"Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
"Extra arms, extra legs, fused fingers, too many fingers, long neck", num_inference_steps=20, guidance_scale=3.5).images[0]
    
    print(image)

    image = image.resize((256, 256))

    # Run the pipeline
    try:
        outputs = pipeline.run(image,seed=1)
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