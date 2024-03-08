import requests
import torch
from PIL import Image
from io import BytesIO
import numpy as np

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

# flow image to image
init_image = Image.open("init_image.png").convert("RGB")
init_image = init_image.resize((432,578))

prompt = "a anime girl"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("img2img.png")

# Flow text to image
random_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
images = pipe(prompt=prompt, image=random_image, strength=1.0, guidance_scale=7.5).images
images[0].save("txt2img.png")