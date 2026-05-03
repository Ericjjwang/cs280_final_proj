import time
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# --- env info ---
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM total: {total_mem:.1f} GB")

torch.cuda.reset_peak_memory_stats()

# --- load model ---
print("\nLoading Zero123++ v1.2 ...")
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)
pipeline.to('cuda')
pipeline.enable_attention_slicing()  # 8GB VRAM safety net

# --- load input ---
cond_url = "https://d.skis.ltd/nrp/sample-data/0_cond.png"
print(f"\nDownloading input image from {cond_url} ...")
cond = Image.open(BytesIO(requests.get(cond_url).content))
print(f"Input size: {cond.size}")

# --- inference ---
print("\nRunning inference (num_inference_steps=36) ...")
t0 = time.time()
result = pipeline(cond, num_inference_steps=36).images[0]
elapsed = time.time() - t0

# --- save ---
out_path = "/home/ericjjwang/projects/cv/outputs/test.png"
result.save(out_path)
print(f"\nSaved to {out_path}")
print(f"Inference time: {elapsed:.1f}s")

if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM used: {peak:.2f} GB")
