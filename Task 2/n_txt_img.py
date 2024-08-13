from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch
from os import device_encoding
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, use_safetensors=True)      #torch_dtype=torch.float16,
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
#cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia
#pipe.enable_model_cpu_offload()
pipe.to("cuda")
prompt = """A beautiful sunset near sea and a small island is there"""
image = pipe(prompt).images[0]
print("[PROMPT]: ",prompt)
plt.imshow(image);
plt.axis('off');