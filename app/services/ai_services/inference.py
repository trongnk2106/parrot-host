import os 
import sys
import time
from dotenv import load_dotenv
_ = load_dotenv()
sys.path.append('./app/services/ai_services/')
sys.path[0] = './app/services/ai_services/'

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import model_loader

from transformers import CLIPTokenizer
from ddpm import DDPMSampler
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0

# Register
ENABLED_TASKS = os.environ.get('ENABLED_TASKS', '').split(',')

# Resouce 
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


if "parrot_sd_task" in ENABLED_TASKS:
    print(f"[INFO] Loading SD1.5 ...")
    pipeline_sd = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16")
    pipeline_sd.to(DEVICE)
    RESOURCE_CACHE["parrot_sd_task"] = pipeline_sd


if "parrot_sdxl_task" in ENABLED_TASKS:
    print(f"[INFO] Loading SDXL-turbo ...")
    pipeline_turbo = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipeline_turbo.to(DEVICE)
    RESOURCE_CACHE["parrot_sdxl_task"] = pipeline_turbo
    
    
if "parrot_encoder_task" in ENABLED_TASKS:
    print(f"[INFO] Loading Encoder module ...")
    tokenizer = CLIPTokenizer("./app/services/ai_services/data/tokenizer_vocab.json", merges_file="./app/services/ai_services/data/tokenizer_merges.txt")
    model_file = "./app/services/ai_services/data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE, only_clip=True)
    models["tokenizer"] = tokenizer
    RESOURCE_CACHE["parrot_encoder_task"] = models


if "parrot_diffuser_task" in ENABLED_TASKS:
    print(f"[INFO] Loading Diffuser module ...")
    tokenizer = CLIPTokenizer("./app/services/ai_services/data/tokenizer_vocab.json", merges_file="./app/services/ai_services/data/tokenizer_merges.txt")
    model_file = "./app/services/ai_services/data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE, only_clip=False)
    models["tokenizer"] = tokenizer
    RESOURCE_CACHE["parrot_diffuser_task"] = models


def run_sd(prompt: str, config: dict):
    # Load config
    num_inference_steps = config.get("steps", 50)
    num_inference_steps = min(num_inference_steps, 50)
    
    rotation = config.get("rotation", "square")
    width, height = 512, 512
    if rotation == "square":
        width, height = 768, 768
    if rotation == "horizontal":
        width, height = 768, 512 
    if rotation == "vertical":
        width, height = 512, 768 
        
    negative_prompt = config.get("negative_prompt", "")
    
    prompt += " high detailed, 8k, high res, (masterpiece:1.4), (highest quality:1.4), (sharp focus:1.2), 8k wallpaper, (tight gap:1.2)"
    negative_prompt += " out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
    
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
            use_lora = False 
            
    image = RESOURCE_CACHE["parrot_sd_task"](
        prompt=prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=7.5, 
        width=width,
        height=height,
        num_inference_steps=num_inference_steps).images[0]

    # to unfuse the LoRA weights
    if use_lora:
        RESOURCE_CACHE["parrot_sd_task"].unet.set_attn_processor(AttnProcessor2_0())
        remove(os.path.join(saved_dir, filename))
        
    return image 


def run_sdxl(prompt: str, config: dict):
    # Load config
    num_inference_steps = config.get("steps", 1)
    num_inference_steps = min(num_inference_steps, 50)
    
    rotation = config.get("rotation", "square")
    width, height = 512, 512
    if rotation == "square":
        width, height = 768, 768
    if rotation == "horizontal":
        width, height = 768, 512 
    if rotation == "vertical":
        width, height = 512, 768 
    negative_prompt = config.get("negative_prompt", "")
    
    prompt += " high detailed, 8k, high res, (masterpiece:1.4), (highest quality:1.4), (sharp focus:1.2), 8k wallpaper, (tight gap:1.2)"
    negative_prompt += " out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
    
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
            use_lora = False 
            
    image = RESOURCE_CACHE["parrot_sdxl_task"](
        prompt=prompt, 
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=0.0, 
        num_inference_steps=1).images[0]
    
    # to unfuse the LoRA weights
    if use_lora:
        RESOURCE_CACHE["parrot_sd_task"].unet.set_attn_processor(AttnProcessor2_0())
        remove(os.path.join(saved_dir, filename))
        
    return image 


def run_encoder(prompt: str, config: dict):
    cond_tokens = RESOURCE_CACHE['parrot_encoder_task']['tokenizer'].batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
    # (Batch_Size, Seq_Len)
    cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=DEVICE)
    cond_context = RESOURCE_CACHE['parrot_encoder_task']['clip'](cond_tokens)
    return cond_context.tolist()


def generate(prompt: str, config: dict):
    
    strength = config.get("strength", 0.5)
    seed = config.get("seed", -1)
    uncond_prompt = config.get("negative_prompt", "")
    do_cfg = True if len(uncond_prompt) else False
    n_inference_steps = config.get("steps", 50)
    n_inference_steps = min(n_inference_steps, 50)
    cfg_scale = config.get("cfg_scale", 7.5)
    
    input_image = None
    
    width, height = 512, 512 
    latents_height = height // 8
    latents_width = height // 8
    
    with torch.no_grad():
        cond_context = torch.tensor(cond_context, device=DEVICE)

        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=DEVICE)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        if do_cfg:
            # Convert into a list of length Seq_Len=77
            uncond_tokens = RESOURCE_CACHE['parrot_diffuser_task']['tokenizer'].batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=DEVICE)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = RESOURCE_CACHE['parrot_diffuser_task']['clip'](uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            context = cond_context

        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(n_inference_steps)

        latents_shape = (1, 4, latents_height, latents_width)

        if input_image:
            input_image_tensor = input_image.resize((width, height))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=DEVICE)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = RESOURCE_CACHE['parrot_diffuser_task']['encoder'](input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=DEVICE)

        # timesteps = tqdm(sampler.timesteps)
        timesteps = sampler.timesteps
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(DEVICE)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = RESOURCE_CACHE['parrot_diffuser_task']['diffusion'](model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)


        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = RESOURCE_CACHE['parrot_diffuser_task']['decoder'](latents)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return Image.fromarray(images[0])
    
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


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
    
    prompt = "the cat and the dog"
    negative_prompt = ""
    
    lora_weight_url = "https://civitai.com/api/download/models/68115"
    # lora_weight_url = "https://civitai.com/api/download/models/268054"
    config = {"rotation": "square", "lora_weight_url": lora_weight_url} # ["horizontal", "vertical", "square"]
    # image = run_sd(prompt, config)
    
    image = run_sd(prompt, config)
    image.save("test.png")
    
    config = {"rotation": "horizontal"} # ["horizontal", "vertical", "square"]
    image = run_sd(prompt, config)
    image.save("test_2.png")