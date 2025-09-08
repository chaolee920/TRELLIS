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

import logging
import sys
import gc

# Configure basic logging to a file
logging.basicConfig(
    filename='/workspace/logs/test-both.log',  # Name of the log file
    level=logging.INFO,  # Minimum logging level to capture (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
    filemode='w'  # File mode: 'a' for append (default), 'w' for overwrite
)


def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

aggressive_cleanup()
torch.cuda.set_device(0)

pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

aggressive_cleanup()
torch.cuda.set_device(1)

model_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

t2i_pipe = HunyuanDiTPipeline.from_pretrained(
    model_id,
    dtype=torch.float16
).to("cuda")

t2i_pipe.transformer = t2i_pipe.transformer.half()
t2i_pipe.vae = t2i_pipe.vae.half()
t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

aggressive_cleanup()
torch.cuda.set_device(2)

i23_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
i23_pipeline.cuda()


def validate(prompt, result_path="./sample.ply"):
    with open(result_path, "rb") as file:
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


def generate_t23(prompt):
    aggressive_cleanup()
    torch.cuda.set_device(0)
    try:
        outputs = pipeline.run(prompt + ", 3d style, whole body, cartoon asset", seed=1,
            sparse_structure_sampler_params={
                "steps": 30,
                "cfg_strength": 8,
            },
            slat_sampler_params={
                "steps": 30,
                "cfg_strength": 4,
            },
            formats=['gaussian']
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

    # Render the outputs
    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample.ply")
    score = validate(prompt)
    aggressive_cleanup()
    print(f"Score from text-to-3d: {score}")
    logging.info(f"Text-to-3D Score: {score}")
    return (outputs['gaussian'][0], score)


def generate_t2i23(prompt, guidance_scale=7.5, num_inference_steps=25):
    aggressive_cleanup()
    torch.cuda.set_device(1)
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
    aggressive_cleanup()

    torch.cuda.set_device(2)

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
            },
            formats=['gaussian']
        )
    except ValueError:  # raised if `y` is empty.
        return None

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample.ply")
    score = validate(prompt)
    aggressive_cleanup()
    print(f"Score from text-to-image-to-3d: {score}")
    logging.info(f"Text-to-Image-to-3D Score: {score}")
    return (outputs['gaussian'][0], score)


prompts_file = open("/workspace/logs/prompts.txt", "r")

start_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
total_cnt = int(sys.argv[2]) if len(sys.argv) > 2 else 10
for i in range(start_num):
    prompts_file.readline()

cnt = 0

sum_score = 0
zero_cnt = 0

while cnt < total_cnt :
    aggressive_cleanup()
    prompt = prompts_file.readline()[:-2]
    print("=============================================================")

    # Check GPU memory usage
    print("Memory usage:")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        print(f"Device {i}:")
        print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
        print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")

    print(f"====Prompt: {prompt}====")
    logging.info("=============================================================")
    logging.info(f"Processing prompt: {prompt}")
    t0 = time()

    output_t23, score_t23 = generate_t23(prompt)
    output_t2i23, score_t2i23 = generate_t2i23(prompt, guidance_scale=9.0)

    if score_t23 >= score_t2i23:
        best_score = score_t23
        best_gaussian = output_t23
    else:
        best_score = score_t2i23
        best_gaussian = output_t2i23
    
    print(f"====Final Score: {best_score}, Generation took: {time() - t0}====")
    logging.info(f"Final Score: {best_score}, Generation took: {time() - t0}")
    logging.info("=============================================================")
    cnt = cnt + 1
    if best_score < 0.6:
        zero_cnt = zero_cnt + 1
    else:
        sum_score = sum_score + best_score

    # Check GPU memory usage
    print("Memory usage:")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        print(f"Device {i}:")
        print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
        print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")

    print("=============================================================")

prompts_file.close()
logging.info(f"Total Prompts Processed: {total_cnt}")
logging.info(f"Number of Zero Scores: {zero_cnt}")
logging.info(f"Number of Non-Zero Scores: {total_cnt - zero_cnt}")
logging.info(f"Average Scores: {sum_score / total_cnt if total_cnt > 0 else 0}")
logging.info(f"Average Score (excluding zeros): {sum_score / (total_cnt - zero_cnt) if (total_cnt - zero_cnt) > 0 else 0}")