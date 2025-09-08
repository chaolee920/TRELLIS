import os

os.environ['SPCONV_ALGO'] = 'native'

from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response
import uvicorn
import argparse
from time import time
from PIL import Image
import imageio
import torch
import gc
import pybase64
import requests

from omegaconf import OmegaConf
import logging

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

from diffusers import HunyuanDiTPipeline
from rembg import remove


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--config", default="configs/text_mv.yaml")
   
    return parser.parse_args()


args = get_args()
app = FastAPI()


def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Configure basic logging to a file
logging.basicConfig(
    filename='/workspace/logs/serve.log',  # Name of the log file
    level=logging.INFO,  # Minimum logging level to capture (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
    filemode='w'  # File mode: 'a' for append (default), 'w' for overwrite
)

aggressive_cleanup()
# torch.cuda.set_device(0)

pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

aggressive_cleanup()
# torch.cuda.set_device(1)

model_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

t2i_pipe = HunyuanDiTPipeline.from_pretrained(
    model_id,
    dtype=torch.float16
).to("cuda")

t2i_pipe.transformer = t2i_pipe.transformer.half()
t2i_pipe.vae = t2i_pipe.vae.half()
t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

aggressive_cleanup()
# torch.cuda.set_device(2)

i23_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
i23_pipeline.cuda()

def get_config() -> OmegaConf:
    config = OmegaConf.load(args.config)
    return config


# def get_models(config: OmegaConf = Depends(get_config)):
#     return ModelsPreLoader.preload_model(config, "cuda")

def generate_t23(prompt):
    aggressive_cleanup()
    # torch.cuda.set_device(0)
    try:
        outputs = pipeline.run(prompt + ", 3d style, whole body, cartoon asset", seed=1,
            sparse_structure_sampler_params={
                "steps": 20,
                "cfg_strength": 8,
            },
            slat_sampler_params={
                "steps": 20,
                "cfg_strength": 4,
            }
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
    # torch.cuda.set_device(1)
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
    # torch.cuda.set_device(2)

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
            }
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


@app.on_event("startup")
def startup_event() -> None:
    config = get_config()
    

@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    opt:OmegaConf = Depends(get_config),
    #models: list = Depends(get_models),
) -> Response:
    print("=============================================================")
    print(f"====Prompt: {prompt}====")
    logging.info("=============================================================")
    logging.info(f"Processing prompt: {prompt}")
    t0 = time()

    output_t23, score_t23 = generate_t23(prompt)
    if score_t23 > 0.65:
        best_score = score_t23
        best_gaussian = output_t23
    else:
        output_t2i23, score_t2i23 = generate_t2i23(prompt, guidance_scale=9.0, num_inference_steps=20)
        if score_t23 < score_t2i23:
            best_gaussian = output_t2i23
            best_score = score_t2i23
        else:
            best_gaussian = output_t23
            best_score = score_t23
    
    t1 = time()
    print(f"====Final Score: {best_score}, Generation took: {t1 - t0}====")
    logging.info(f"Final Score: {best_score}, Generation took: {t1 - t0}")
    logging.info("=============================================================")

    if best_score < 0.6:
        return None

    buffer = BytesIO()
    best_gaussian.save_ply(buffer)
    buffer.seek(0)
    buffer = buffer.getbuffer()
    t2 = time()
    print(f" Saving and encoding took: {(t2 - t1) / 60.0} min")
    print("=============================================================")

    return Response(buffer, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    opt:OmegaConf = Depends(get_config),
    #models: list = Depends(get_models),
):
    # t1 = time()
    # outputs = app.pipeline.run(prompt, seed=1,
    # # Optional parameters
    # # sparse_structure_sampler_params={
    # #     "steps": 12,
    # #     "cfg_strength": 7.5,
    # # },
    # # slat_sampler_params={
    # #     "steps": 12,
    # #     "cfg_strength": 3,
    # # },
    # )

    # logger.info(f" It took: {(time() - t1) / 60.0} min")
    # logger.info("Generating video.")

    # t2 = time()
    # video_utils = VideoUtils(video_res, video_res, 5, 5, 10, -30, 10)
    # buffer, _ = video_utils.render_video(
    #     processed_data[0],
    #     None,
    #     processed_data[4],
    #     None,
    #     processed_data[3],
    #     processed_data[2],
    #     processed_data[1],
    #     None
    # )
    # logger.info(f" It took: {(time() - t2) / 60.0} min")

    # return StreamingResponse(content=buffer, media_type="video/mp4")
    return None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
