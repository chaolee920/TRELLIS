import os

os.environ['SPCONV_ALGO'] = 'native'

from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
from time import time
from PIL import Image
import imageio
import torch

from omegaconf import OmegaConf
from loguru import logger

# from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

from diffusers import DiffusionPipeline
from rembg import remove, new_session


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--config", default="configs/text_mv.yaml")
   
    return parser.parse_args()


args = get_args()
app = FastAPI()

# pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline = TrellisTextTo3DPipeline.from_pretrained("/workspace/vol_sub17/models/TRELLIS-text-xlarge")
pipeline.cuda()

# t2i_pipe = DiffusionPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled", torch_dtype=torch.float16).to("cuda")

# t2i_pipe.transformer = t2i_pipe.transformer.half()
# t2i_pipe.vae = t2i_pipe.vae.half()
# t2i_pipe.text_encoder = t2i_pipe.text_encoder.half()

def get_config() -> OmegaConf:
    config = OmegaConf.load(args.config)
    return config


# def get_models(config: OmegaConf = Depends(get_config)):
#     return ModelsPreLoader.preload_model(config, "cuda")


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
    print("generation started")

    # with torch.cuda.amp.autocast():
    outputs = pipeline.run(prompt + "4k, white background, 3D style, best quality", seed=1)
#         image = t2i_pipe(prompt + ", 4k, white background, 3D style, best quality", negative_prompt="Text, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
# "Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
# "Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
# "Extra arms, extra legs, fused fingers, too many fingers, long neck", num_inference_steps=20,  guidance_scale=3.5).images[0]

#     image = remove(image, session=new_session(), bgcolor=[255, 255, 255, 0])

#     outputs = pipeline.run(image, seed=1,
# # Optional parameters
# #    sparse_structure_sampler_params={
# #        "steps": 25,
# #        "cfg_strength": 6.0,
# #	"cfg_interval": [0.5, 0.95],
# #        "rescale_t": 3.0
# #    },
# #    slat_sampler_params={
# #        "steps": 25,
# #        "cfg_strength": 7.5,
# #        "cfg_interval": [0.5, 0.95],
# #        "rescale_t": 3.0
# #    },
#     )

    print("generation ended")
    t1 = time()
    print(f" Generation took: {(t1 - t0) / 60.0} min")

    buffer = BytesIO()
    outputs['gaussian'][0].save_ply(buffer)
    outputs['gaussian'][0].save_ply('/workspace/vol_sub17/test-ply/result.ply')
    print("saved")
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
