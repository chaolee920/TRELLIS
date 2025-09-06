import os
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import torch
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from diffusers import HunyuanDiTPipeline
import pybase64
import requests
from time import time
from rembg import remove

torch.cuda.empty_cache()


torch.cuda.set_device(1)
model_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

t2i_pipe = HunyuanDiTPipeline.from_pretrained(
    model_id,
    dtype=torch.float16
).to("cuda")

t2i_pipe.transformer = t2i_pipe.transformer.half()
t2i_pipe.vae = t2i_pipe.vae.half()
t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

torch.cuda.set_device(2)

i23_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
i23_pipeline.cuda()

torch.cuda.set_device(0)

pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()


def generate_t23(prompt):
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    outputs = pipeline.run(prompt + ", 3d style, whole body, cartoon asset", seed=1,
        sparse_structure_sampler_params={
            "steps": 30,
            "cfg_strength": 8,
        },
        slat_sampler_params={
            "steps": 30,
            "cfg_strength": 4,
        }
    )

    # Render the outputs
    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample.ply")


def generate_t2i23(prompt, guidance_scale=7.5, num_inference_steps=25):
    torch.cuda.set_device(1)
    torch.cuda.empty_cache()
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

    torch.cuda.set_device(2)
    torch.cuda.empty_cache()

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


def validate(prompt):
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


prompts_file = open("/workspace/logs/prompts.txt", "r")
cnt = 0
while cnt < 700 :
    prompt = prompts_file.readline()[:-2]
    t0 = time()
    generate_t23(prompt)
    validation_score = validate(prompt)
    if validation_score < 0.6:
        print(f"Validation score {validation_score} is less than 0.6, regenerating with t2i23...")
        generate_t2i23(prompt, guidance_scale=9.0)
        validation_score = validate(prompt)
    print(f"=====Final Score: {validation_score}, Generation took: {time() - t0}=====")
    cnt=cnt+1
