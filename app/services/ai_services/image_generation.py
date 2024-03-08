import io
import os
import sys
import time

from dotenv import load_dotenv
_ = load_dotenv()
sys.path.append('./app/services/ai_services/')
sys.path[0] = './app/services/ai_services/'


import random
import tempfile

from PIL import Image
import imageio
import numpy as np
import torch
from diffusers import (AutoPipelineForText2Image, StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image, DiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, StableDiffusionImg2ImgPipeline)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers.utils import load_image

# Register
ENABLED_TASKS = os.environ.get('ENABLED_TASKS', '').split(',')
NO_HALF = os.environ.get('NO_HALF', False)

# Resource 
RESOURCE_CACHE = {}

# Load device
DEVICE = "cpu"
ALLOW_CUDA = True
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Using half: {not NO_HALF}")

if "parrot_sd_task" in ENABLED_TASKS:
    print(f"[INFO] Loading SD1.5 ...")
    if NO_HALF:
        pipeline_sd = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    else:
        pipeline_sd = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16")
    
    pipeline_sd.to(DEVICE)
        
    RESOURCE_CACHE["parrot_sd_task"] = pipeline_sd


if "parrot_sdxl_task" in ENABLED_TASKS:
    print(f"[INFO] Loading SDXL-turbo ...")
    if NO_HALF:
        pipeline_turbo = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo")
    else:
        pipeline_turbo = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        
    pipeline_turbo.to(DEVICE)
    RESOURCE_CACHE["parrot_sdxl_task"] = pipeline_turbo
   
    
if "parrot_sdxl_lightning_task" in ENABLED_TASKS:
    print(f"[INFO] Loading SDXL-lightning ...")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"

    if NO_HALF:
        # Load model.
        unet = UNet2DConditionModel.from_config(base, subfolder="unet")
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
        pipeline_lightning = StableDiffusionXLImg2ImgPipeline.from_pretrained(base)
        pipeline_lightning.to(DEVICE)
    else:
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
        pipeline_lightning = StableDiffusionXLImg2ImgPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to("cuda")
                
    # Ensure sampler uses "trailing" timesteps.
    pipeline_lightning.scheduler = EulerDiscreteScheduler.from_config(pipeline_lightning.scheduler.config, timestep_spacing="trailing")
    RESOURCE_CACHE["parrot_sdxl_lightning_task"] = pipeline_lightning
    

if "parrot_txt2vid_damo_task" in ENABLED_TASKS:
    pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', torch_dtype=torch.float16,variant='fp16').to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    RESOURCE_CACHE["parrot_txt2vid_task"] = pipe


def run_sd(prompt: str, config: dict):
    # Load config
    num_inference_steps = config.get("steps", 50)
    num_inference_steps = min(num_inference_steps, 50)
    guidance_scale = config.get("cfg_scale", 7.5)
    init_image = config.get("init_image", None)
    strength = config.get("strength", 0.75)

    
    if config.get("rotation"):
        rotation = config.get("rotation", "square")
        width, height = 512, 512
        if rotation == "square":
            width, height = 512, 512
        if rotation == "horizontal":
            width, height = 768, 512 
        if rotation == "vertical":
            width, height = 512, 768 
    else:
        width, height = config.get("width", 512), config.get("height", 512)

    negative_prompt = config.get("negative_prompt", "")
    seed = config.get("seed", -1)
    if seed == -1:
        seed = random.randint(0, 1000000)
    generator = torch.Generator().manual_seed(seed)
        
    negative_prompt = config.get("negative_prompt", "")
    
    use_lora = False
    lora_weight_url = config.get("lora_weight_url", "")
    if len(lora_weight_url):
        use_lora = True
    
    saved_dir, filename = "", ""
    if use_lora:
        try:
            saved_dir = "saved_lora_checkpoints"
            filename = f"{int(time.time())}.safetensors"
            download_lora_weight(lora_weight_url, saved_dir, filename)
            RESOURCE_CACHE["parrot_sd_task"].load_lora_weights(saved_dir, weight_name=filename)
        except Exception as e:
            print(e)
            remove(os.path.join(saved_dir, filename))
            RESOURCE_CACHE["parrot_sd_task"].unload_lora_weights()            
            use_lora = False 
            
    if init_image is None:
        # random init image with seed
        np.random.seed(seed)
        init_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        strength = 1.0
        
    image = RESOURCE_CACHE["parrot_sd_task"](
        prompt=prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale, 
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        image=init_image,
        strength=strength,
        generator=generator).images[0]

    # to unfuse the LoRA weights
    if use_lora:
        remove(os.path.join(saved_dir, filename))
        RESOURCE_CACHE["parrot_sd_task"].unload_lora_weights()        
        
    return image 


def run_sdxl(prompt: str, config: dict):
    # Load config
    num_inference_steps = config.get("steps", 1)
    num_inference_steps = min(num_inference_steps, 50)
    guidance_scale = config.get("cfg_scale", 7.5)
    strength = config.get("strength", 0.75)
    
    if config.get("rotation"):
        rotation = config.get("rotation", "square")
        width, height = 512, 512
        if rotation == "square":
            width, height = 1024, 1024
        if rotation == "horizontal":
            width, height = 768, 512 
        if rotation == "vertical":
            width, height = 512, 768 
    else:
        width, height = config.get("width", 1024), config.get("height", 1024)

    negative_prompt = config.get("negative_prompt", "")
    seed = config.get("seed", -1)
    if seed == -1:
        seed = random.randint(0, 1000000)
    generator = torch.Generator().manual_seed(seed)

    negative_prompt = config.get("negative_prompt", "")

    init_image = config.get("init_image", None)
    if init_image is None:
        # random init image with seed
        np.random.seed(seed)
        init_image = np.random.randint(0, 255, (width, height, 3), dtype=np.uint8)
        strength = 1.0

    
    use_lora = False
    lora_weight_url = config.get("lora_weight_url", "")
    if len(lora_weight_url):
        use_lora = True
    
    saved_dir, filename = "", ""
    if use_lora:
        try:
            saved_dir = "saved_lora_checkpoints"
            filename = f"{int(time.time())}.safetensors"
            download_lora_weight(lora_weight_url, saved_dir, filename)
            RESOURCE_CACHE["parrot_sdxl_task"].load_lora_weights(saved_dir, weight_name=filename)
        except Exception as e:
            print(e)
            remove(os.path.join(saved_dir, filename))
            RESOURCE_CACHE["parrot_sdxl_task"].unload_lora_weights()
            use_lora = False 
            
    
    image = RESOURCE_CACHE["parrot_sdxl_task"](
        prompt=prompt, 
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps,
        image=init_image,
        strength=strength,
        generator=generator).images[0]
    
    # to unfuse the LoRA weights
    if use_lora:
        remove(os.path.join(saved_dir, filename))
        RESOURCE_CACHE["parrot_sdxl_task"].unload_lora_weights()

    return image 


def run_sdxl_lightning(prompt: str, config: dict):
    # Load config
    # num_inference_steps = 8
    
    if config.get("rotation"):
        rotation = config.get("rotation", "square")
        width, height = 512, 512
        if rotation == "square":
            width, height = 1024, 1024
        if rotation == "horizontal":
            width, height = 768, 512 
        if rotation == "vertical":
            width, height = 512, 768 
    else:
        width, height = config.get("width", 1024), config.get("height", 1024)

    negative_prompt = config.get("negative_prompt", "")
    seed = config.get("seed", -1)
    if seed == -1:
        seed = random.randint(0, 1000000)
    generator = torch.Generator().manual_seed(seed)

    use_lora = False
    lora_weight_url = config.get("lora_weight_url", "")
    if len(lora_weight_url):
        use_lora = True
    
    saved_dir, filename = "", ""
    if use_lora:
        try:
            saved_dir = "saved_lora_checkpoints"
            filename = f"{int(time.time())}.safetensors"
            download_lora_weight(lora_weight_url, saved_dir, filename)
            RESOURCE_CACHE["parrot_sdxl_task"].load_lora_weights(saved_dir, weight_name=filename)
        except Exception as e:
            print(e)
            remove(os.path.join(saved_dir, filename))
            RESOURCE_CACHE["parrot_sdxl_lightning_task"].unload_lora_weights()            
            use_lora = False 
            

    init_image = config.get("init_image", None)
    strength = config.get("strength", 0.75)
    guidance_scale = config.get("guidance_scale", 7.5)
    num_inference_steps = config.get("steps", 8)

    if init_image is None:
        # random init image with seed
        np.random.seed(seed)
        init_image = np.random.randint(0, 255, (width, height, 3), dtype=np.uint8)
        strength = 1.0


    image = RESOURCE_CACHE["parrot_sdxl_lightning_task"](
        prompt=prompt, 
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps,
        image=init_image,
        strength=strength,
        generator=generator).images[0]
        
    # to unfuse the LoRA weights
    if use_lora:
        remove(os.path.join(saved_dir, filename))
        RESOURCE_CACHE["parrot_sdxl_lightning_task"].unload_lora_weights()
        
    return image 


def run_txt2vid(prompt: str, config: dict):

    fps = config.get("fps", 8)
    seed = config.get("seed", -1)
    num_inference_steps = config.get("steps", 25)
    num_frames = config.get("frames", 16)
    height = config.get("height", 512)
    width = config.get("width", 512)

    if seed == -1:
        seed = random.randint(0, 1000000)

    generator = torch.Generator().manual_seed(seed)
    frames = RESOURCE_CACHE["parrot_txt2vid_task"](
        prompt, 
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        generator=generator
    ).frames
    
    video = to_video(frames, fps)
    return video


def to_video(frames: list[np.ndarray], fps: int) -> io.BytesIO:
    byte_io = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as out_file:
        writer = imageio.get_writer(out_file.name, format='FFMPEG', fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        with open(out_file.name, 'rb') as f:
            byte_io.write(f.read())
    return byte_io
    
    
def download_lora_weight(url, saved_dir, filename):
    import subprocess
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    save_path = os.path.join(saved_dir, filename)
    
    subprocess.run(['wget', '-O', save_path, url], check=True)


def remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


if __name__=="__main__": 
    # test code for sd ===========================================================
    # prompt = "the anime girl"
    # init_image = Image.open("init_image.png").convert("RGB")

    # config = {
    #     "steps": 50,
    #     "cfg_scale": 7.5,
    #     "rotation": "square",
    #     "width": 512,
    #     "height": 512,
    #     "negative_prompt": "",
    #     "seed": -1,
    #     "lora_weight_url": "", 
    #     "init_image": init_image
    # }

    # config1 = {
    #     "steps": 50,
    #     "cfg_scale": 7.5,
    #     "rotation": "square",
    #     "width": 512,
    #     "height": 512,
    #     "negative_prompt": "",
    #     "seed": -1,
    #     "lora_weight_url": "", 
    # }
    # # for sd
    # image = run_sd(prompt, config)
    # image.save("sd.png")
    # ==============================================================================

    # test code for sdxl ===========================================================
    # prompt = "the anime girl"

    # config = {
    #     "steps": 50,
    #     "rotation": "square",
    #     "width": 1024,
    #     "height": 1024,
    #     "negative_prompt": "",
    #     "seed": -1,
    #     "lora_weight_url": "", 
    # }

    # init = load_image("init_image.png").resize((512, 512))
    # config1 = {
    #     "steps": 50,
    #     "rotation": "square",
    #     "width": 1024,
    #     "height": 1024,
    #     "negative_prompt": "",
    #     "seed": -1,
    #     "lora_weight_url": "", 
    #     "init": init, 
    #     "strength": 0.75
    # }

    # # for sdxl
    # image = run_sdxl(prompt, config1)
    # image.save("sdxl.png")
    # ==============================================================================

    # test code for sdxl_lightning =================================================
    prompt = "the anime girl"
    init_image = Image.open("init_image.png").convert("RGB")

    config = {
        "steps": 8,
        "rotation": "square",
        "width": 1024,
        "height": 1024,
        "negative_prompt": "",
        "seed": -1,
        "lora_weight_url": "", 
        "init_image": init_image, 
        "strength": 0.75
    }

    image = run_sdxl_lightning(prompt, config)
    image.save("sdxl_lightning.png")
    