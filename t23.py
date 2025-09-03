import os
#os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

from diffusers import DiffusionPipeline
from rembg import remove, new_session

torch.cuda.empty_cache()

print("A")
pipe = DiffusionPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", dtype=torch.float16).to("cuda")

pipe.transformer = pipe.transformer.half()
pipe.vae = pipe.vae.half()
pipe.text_encoder = pipe.text_encoder.half()

print("B")
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
print("C")

prompt = input()
with torch.cuda.amp.autocast():
    image = pipe(
        prompt + ", white background, 3d style, whole body, cartoon asset",
        negative_prompt="Text, flashy, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
            "Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
            "Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
            "Extra arms, extra legs, fused fingers, too many fingers, long neck", num_inference_steps=25, guidance_scale=7.5).images[0]

output = remove(image, session=new_session(), bgcolor=[255, 255, 255, 0])
output.save("output.png")

output = output.resize((512, 512))

# Run the pipeline
outputs = pipeline.run(
    output,
    seed=1,
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
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply("sample.ply")
