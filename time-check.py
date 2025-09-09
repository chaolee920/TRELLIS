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
import asyncio

import gc

def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

aggressive_cleanup()

pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

aggressive_cleanup()

model_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

t2i_pipe = HunyuanDiTPipeline.from_pretrained(
    model_id,
    dtype=torch.float16
).to("cuda")

t2i_pipe.transformer = t2i_pipe.transformer.half()
t2i_pipe.vae = t2i_pipe.vae.half()
t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

aggressive_cleanup()

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


def _generate_t23_sync(prompt):
    aggressive_cleanup()
    try:
        outputs = pipeline.run(prompt + ", 3d style, whole body, cartoon asset", seed=1,
            sparse_structure_sampler_params={
                "steps": 20,
                "cfg_strength": 8,
            },
            slat_sampler_params={
                "steps": 20,
                "cfg_strength": 4,
            },
            formats=['gaussian']
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

    # Render the outputs
    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample_t23.ply")
    score = validate(prompt, "sample_t23.ply")
    aggressive_cleanup()
    print(f"Score from text-to-3d: {score}")
    return (outputs['gaussian'][0], score)

async def generate_t23(prompt):
    return await asyncio.to_thread(_generate_t23_sync, prompt)


def _generate_t2i23_sync(prompt, guidance_scale=7.5, num_inference_steps=25):
    aggressive_cleanup()
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

    # Run the pipeline
    try:
        outputs = i23_pipeline.run(image,
            sparse_structure_sampler_params={
                "steps": 20,
                "cfg_strength": 8,
            },
            slat_sampler_params={
                "steps": 20,
                "cfg_strength": 4,
            },
            formats=['gaussian']
        )
    except ValueError:  # raised if `y` is empty.
        return None

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample_t2i23.ply")
    score = validate(prompt, "sample_t2i23.ply")
    aggressive_cleanup()
    print(f"Score from text-to-image-to-3d: {score}")
    return (outputs['gaussian'][0], score)

async def generate_t2i23(prompt, guidance_scale=7.5, num_inference_steps=25):
    return await asyncio.to_thread(_generate_t2i23_sync, prompt, guidance_scale, num_inference_steps)


async def main():
    aggressive_cleanup()
    prompt = "pink bicycle"
    print("=============================================================")

    # Check GPU memory usage
    print("Memory usage:")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        print(f"Device {i}:")
        print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
        print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")

    print(f"====Prompt: {prompt}====")
    t0 = time()

    # Run both generation methods concurrently with early termination
    t23_task = asyncio.create_task(generate_t23(prompt))
    t2i23_task = asyncio.create_task(generate_t2i23(prompt, guidance_scale=9.0, num_inference_steps=20))
    
    pending_tasks = {t23_task, t2i23_task}
    task_names = {t23_task: 't23', t2i23_task: 't2i23'}
    
    output_t23, score_t23 = None, 0
    output_t2i23, score_t2i23 = None, 0
    best_gaussian, best_score = None, 0
    early_termination = False
    
    # Process results as they complete
    while pending_tasks:
        done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        
        for completed_task in done:
            try:
                result = await completed_task
                task_type = task_names[completed_task]
                
                if result is not None:
                    gaussian, score = result
                    
                    if task_type == 't23':
                        output_t23, score_t23 = gaussian, score
                        print(f"generate_t23 completed with score: {score}")
                        
                        # Early termination if t23 score > 0.65
                        if score > 0.65:
                            print(f"Early termination: t23 score {score} > 0.65, cancelling t2i23")
                            best_gaussian, best_score = gaussian, score
                            early_termination = True
                            
                            # Cancel remaining tasks
                            for task in pending_tasks:
                                task.cancel()
                            pending_tasks.clear()
                            break
                        
                    elif task_type == 't2i23':
                        output_t2i23, score_t2i23 = gaussian, score
                        print(f"generate_t2i23 completed with score: {score}")
                else:
                    print(f"Generation method {task_type} returned None")
                    
            except Exception as e:
                task_type = task_names[completed_task]
                print(f"Error in {task_type}: {e}")
    
    # If not early terminated, select the best result
    if not early_termination:
        if output_t23 is None and output_t2i23 is None:
            raise Exception("Both generation methods failed")
        elif output_t23 is None:
            best_gaussian = output_t2i23
            best_score = score_t2i23
        elif output_t2i23 is None:
            best_gaussian = output_t23
            best_score = score_t23
        elif score_t23 < score_t2i23:
            best_gaussian = output_t2i23
            best_score = score_t2i23
        else:
            best_gaussian = output_t23
            best_score = score_t23
    
    t1 = time()
    termination_status = "early termination" if early_termination else "both methods completed"
    print(f"====Final Score: {best_score}, Generation took: {t1 - t0}, Status: {termination_status}====")

    # Check GPU memory usage
    print("Memory usage:")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        print(f"Device {i}:")
        print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
        print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")

    print("=============================================================")
    return best_gaussian, best_score

if __name__ == "__main__":
    asyncio.run(main())
