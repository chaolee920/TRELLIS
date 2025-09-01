import os
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif
import pybase64
import requests

torch.cuda.empty_cache()


pipeline = ShapEPipeline.from_pretrained("openai/shap-e").to("cuda")

prompts_file = open("/workspace/vol_sub17/prompts.txt", "r")
cnt = 0
while cnt < 100 :
    prompt = prompts_file.readline()


    outputs = pipeline(prompt, guidance_scale=15.0, num_inference_steps=64, size=512, output_type="mesh").meshes[0]

    # Render the outputs
    # Save Gaussians as PLY files
    outputs.export("sample.ply")
    with open("./sample.ply", "rb") as file:
        file_data = file.read()
    encoded_data = pybase64.b64encode(file_data).decode("utf-8")
    validate_url = 'http://127.0.0.1:8094/validate_txt_to_3d_ply'
    response = requests.post(validate_url, json={"prompt": prompt, "data": encoded_data})
    if response.status_code == 200:
        results_validation = response.json()

        validation_score = float(results_validation["score"])
    cnt=cnt+1