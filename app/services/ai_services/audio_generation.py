import os
import sys

from dotenv import load_dotenv
_ = load_dotenv()
sys.path.append('./app/services/ai_services/')
sys.path[0] = './app/services/ai_services/'


import torch
from transformers import pipeline
from audiocraft.models import AudioGen, MusicGen
from transformers import AutoProcessor, BarkModel


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
print(f"[INFO] Enabled tasks: {ENABLED_TASKS}")


if "parrot_t2s_task" in ENABLED_TASKS: 
    print(f"[INFO] Load parrot_t2s_task")
    repo = "suno/bark"
    processor = AutoProcessor.from_pretrained(repo)
    model = BarkModel.from_pretrained(repo).to(DEVICE)

    RESOURCE_CACHE["parrot_t2s_task"] = (processor, model)
    print(f"[INFO] Load parrot_t2s_task done")


if "parrot_musicgen_task" in ENABLED_TASKS:
    print(f"[INFO] Load parrot_musicgen_task")
    repo = "facebook/musicgen-small"

    model = MusicGen.get_pretrained(repo)
    RESOURCE_CACHE["parrot_musicgen_task"] = model
    print(f"[INFO] Load parrot_musicgen_task done")


if "parrot_audiogen_task" in ENABLED_TASKS:
    print(f"[INFO] Load parrot_audiogen_task")
    repo = "facebook/audiogen-medium"

    model = AudioGen.get_pretrained(repo)
    RESOURCE_CACHE["parrot_audiogen_task"] = model
    print(f"[INFO] Load parrot_audiogen_task done")


def run_text2speech(prompt: str, config: dict):
    print(f"[INFO] Run text2speech")
    if "parrot_t2s_task" not in RESOURCE_CACHE:
        raise Exception("parrot_t2s_task is not loaded")

    try: 
        processor, model = RESOURCE_CACHE["parrot_t2s_task"]
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Model parrot_t2s_task is not loaded")

    try: 
        inputs = processor(prompt).to("cuda")
        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Error when inference")
    
    return audio_array, model.generation_config.sample_rate


def run_musicgen(prompt: str, config: dict):
    print(f"[INFO] Run musicgen")
    if "parrot_musicgen_task" not in RESOURCE_CACHE:
        raise Exception("parrot_musicgen_task is not loaded")

    try:
        model = RESOURCE_CACHE["parrot_musicgen_task"]
        duration = config.get("duration", 5)
        top_k = config.get("top_k", 15)
        top_p = config.get("top_p", 0.9)

        model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Model parrot_audiogen_task is not loaded")

    try:
        audio_result = model.generate([prompt])
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Error when inference")

    return audio_result[0], model.sample_rate


def run_audiogen(prompt: str, config: dict):
    print(f"[INFO] Run audiogen")
    if "parrot_audiogen_task" not in RESOURCE_CACHE:
        raise Exception("parrot_audiogen_task is not loaded")

    try:
        model = RESOURCE_CACHE["parrot_audiogen_task"]
        duration = config.get("duration", 5)
        top_k = config.get("top_k", 15)
        top_p = config.get("top_p", 0.9)

        model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Model parrot_audiogen_task is not loaded")

    try:
        audio_result = model.generate([prompt])
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Error when inference")

    return audio_result[0], model.sample_rate
