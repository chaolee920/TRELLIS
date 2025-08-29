import os
os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

from diffusers import DiffusionPipeline
from rembg import remove, new_session

print("A")
pipe = DiffusionPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled").to("cuda")
print("B")
pipeline = TrelisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large").cuda()
print("C")

prompt = input()
image = pipe(prompt + ", white background, 3D style, best quality", negative_prompt="Text, close-up, cropped, out of frame, worst quality, low quality, JPEG artifacts, PGLY, repetitive, morbid," \
"Mutilation, extra fingers, mutant hands, poorly drawn hands, poorly drawn faces, mutations, deformities, blurry, dehydrated, poor anatomy," \
"Bad proportions, extra limbs, cloned faces, disfigurement, disgusting proportions, deformed limbs, missing arms, missing legs," \
"Extra arms, extra legs, fused fingers, too many fingers, long neck", num_inference_steps=20, guidance_scale=3.5).images[0]

output = remove(image, session=new_session(), bgcolor=[255, 255, 255, 0])
output.save("output.png")

output = output.resize((512, 512))

# Run the pipeline
outputs = pipeline.run(
    output,
    seed=1
)
# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply("sample.ply")
