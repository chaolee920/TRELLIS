import os
os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import torch
from diffusers import HunyuanDiTPipeline
from accelerate import Accelerator
from trellis.pipelines import TrellisImageTo3DPipeline
import pybase64
import requests
from time import time
from rembg import remove

torch.cuda.empty_cache()

model_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

accelerator = Accelerator()

t2i_pipe = HunyuanDiTPipeline.from_pretrained(
    model_id,
    dtype=torch.float16
).to("cuda:1")

t2i_pipe.transformer = t2i_pipe.transformer.half()
t2i_pipe.vae = t2i_pipe.vae.half()
t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

i23_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
i23_pipeline.cuda()

prompts_file = open("/workspace/logs/prompts.txt", "r")

for i in range(120):
    prompts_file.readline()

def generate(prompt, guidance_scale=7.5, num_inference_steps=25):
    image = t2i_pipe(
        prompt + ", white background, 3d style, whole body, cartoon asset",
        negative_prompt="Text, flasy, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
                "Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
                "Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
                "Extra arms, extra legs, fused fingers, too many fingers, long neck",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    image = remove(image, alpha_matting=True, alpha_matting_foreground_threshold=240)

    # Run the pipeline
    try:
        outputs = i23_pipeline.run(image,
            sparse_structure_sampler_params={
                "steps": 30,
                "cfg_strength": 8,
            },
            slat_sampler_params={
                "steps": 30,
                "cfg_strength": 4,
            }
        )
    except ValueError:  # raised if `y` is empty.
        return None

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample.ply")


def validate():
    with open("./sample.ply", "rb") as file:
        file_data = file.read()
    encoded_data = pybase64.b64encode(file_data).decode("utf-8")
    validate_url = 'http://127.0.0.1:8094/validate_txt_to_3d_ply'
    response = requests.post(validate_url, json={"prompt": prompt, "data": encoded_data})
    if response.status_code == 200:
        results_validation = response.json()

        validation_score = float(results_validation["score"])
        return validation_score
    else:
        print("Validation failed with status code:", response.status_code)
        return 0

cnt = 0
while cnt < 10 :
    torch.cuda.empty_cache()
    prompt = prompts_file.readline()
    t0 = time()
    generate(prompt, guidance_scale=1.0)

    # validation_score = validate()
    
    # if validation_score < 0.6:
    #     generate(prompt, guidance_scale=3.0)
    
    print(f"=====Final Score: {validate()}, Generation took: {time() - t0}=====")
    cnt=cnt+1
    
    print("Memory usage:")
    print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
    print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")
