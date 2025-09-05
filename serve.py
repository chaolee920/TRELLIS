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
import pybase64
import requests

from omegaconf import OmegaConf
from loguru import logger

from trellis.pipelines import TrellisImageTo3DPipeline
# from trellis.pipelines import TrellisTextTo3DPipeline
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

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
# pipeline = TrellisTextTo3DPipeline.from_pretrained("/workspace/vol_sub17/models/TRELLIS-text-xlarge")
pipeline.cuda()

t2i_pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", torch_dtype=torch.float16).to("cuda:1")

t2i_pipe.transformer = t2i_pipe.transformer.half()
t2i_pipe.vae = t2i_pipe.vae.half()
t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

def get_config() -> OmegaConf:
    config = OmegaConf.load(args.config)
    return config


# def get_models(config: OmegaConf = Depends(get_config)):
#     return ModelsPreLoader.preload_model(config, "cuda")

def execute_generate(prompt):
    image = t2i_pipe(
        prompt + ", white background, 3d style, whole body, cartoon asset, best quality",
        negative_prompt="Text, flasy, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
                "Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
                "Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
                "Extra arms, extra legs, fused fingers, too many fingers, long neck",
        guidance_scale=7.5,
        num_inference_steps=25,
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
    return outputs['gaussian'][0]


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


@app.on_event("startup")
def startup_event() -> None:
    config = get_config()
    

@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    opt:OmegaConf = Depends(get_config),
    #models: list = Depends(get_models),
) -> Response:
    t0 = time()
    outputs = execute_generate(prompt)
    outputs.save_ply("sample.ply")
    validation_score = validate()
    if validation_score < 0.6:
        outputs = execute_generate(prompt)
    t1 = time()
    print(f" Generation took: {(t1 - t0) / 60.0} min")

    buffer = BytesIO()
    outputs['gaussian'][0].save_ply(buffer)
    buffer.seek(0)
    buffer = buffer.getbuffer()
    t2 = time()
    print(f" Saving and encoding took: {(t2 - t1) / 60.0} min")

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
