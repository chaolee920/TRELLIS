import os

os.environ['SPCONV_ALGO'] = 'native'

from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
from time import time

from omegaconf import OmegaConf
from loguru import logger

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--config", default="configs/text_mv.yaml")
   
    return parser.parse_args()


args = get_args()
app = FastAPI()

pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

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
    
    outputs = pipeline.run(prompt+", highly detailed", seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 20,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 30,
        "cfg_strength": 6.0,
#        "temperature": 1.0,	
    },
    )
    print("generation ended")
    t1 = time()
    logger.info(f" Generation took: {(t1 - t0) / 60.0} min")

    buffer = BytesIO()
    outputs['gaussian'][0].save_ply(buffer)
    print("saved")
    buffer.seek(0)
    buffer = buffer.getbuffer()
    t2 = time()
    logger.info(f" Saving and encoding took: {(t2 - t1) / 60.0} min")

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
