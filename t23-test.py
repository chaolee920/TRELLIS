import os
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import aiohttp
import torch
from trellis.pipelines import TrellisTextTo3DPipeline
import pybase64
import requests
import asyncio
torch.cuda.empty_cache()


pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()
async def main() :
    file = open("/workspace/vol_sub17/test/prompts.txt", "r")
    cnt = 0
    while cnt < 100 :
        prompt = file.readline()


        outputs = pipeline.run(prompt,seed=1, )

        # Render the outputs
        # Save Gaussians as PLY files
        outputs['gaussian'][0].save_ply("sample.ply")
        with open("./sample.ply", "rb") as file:
            file_data = file.read()
        encoded_data = pybase64.b64encode(file_data).decode("utf-8")
        validate_url = 'http://127.0.0.1:8094/validate_txt_to_3d_ply'
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(validate_url, json={"prompt": prompt, "data": encoded_data}) as response:
                    if response.status == 200:
                        results_validation = await response.json()

                        validation_score = float(results_validation["score"])
            finally:
                cnt=cnt+1

main()