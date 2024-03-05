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

import imageio
import numpy as np
import torch
# from diffusers import (AutoPipelineForText2Image, DiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel)
# from huggingface_hub import hf_hub_download
# from safetensors.torch import load_file
# from transformers import AutoProcessor, BarkModel
from transformers import pipeline



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


if "parrot_text2speech_task" in ENABLED_TASKS: 
    print(f"[INFO] Load parrot_text2speech_task")
    repo = "suno/bark"

    pipe = pipeline("text-to-speech", repo, device=DEVICE, half=not NO_HALF)
    RESOURCE_CACHE["parrot_text2speech_task"] = pipe


def run_text2speech(prompt: str, config: dict):
    print(f"[INFO] Run text2speech")
    if "parrot_text2speech_task" not in RESOURCE_CACHE:
        raise Exception("parrot_text2speech_task is not loaded")

    try: 
        pipe = RESOURCE_CACHE["parrot_text2speech_task"]
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Model parrot_text2speech_task is not loaded")

    try: 
        audio_result = pipe(prompt, **config)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Error when inference")
    
    return audio_result