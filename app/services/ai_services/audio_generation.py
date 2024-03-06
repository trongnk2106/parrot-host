import os
import sys

from dotenv import load_dotenv
_ = load_dotenv()
sys.path.append('./app/services/ai_services/')
sys.path[0] = './app/services/ai_services/'


import torch
from transformers import pipeline
from audiocraft.models import AudioGen



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

    pipe = pipeline("text-to-speech", repo, device=DEVICE)
    RESOURCE_CACHE["parrot_t2s_task"] = pipe
    print(f"[INFO] Load parrot_t2s_task done")

if "parrot_musicgen_task" in ENABLED_TASKS:
    print(f"[INFO] Load parrot_musicgen_task")
    repo = "facebook/musicgen-small"
    
    synthesiser = pipeline("text-to-audio", repo, device=0)
    RESOURCE_CACHE["parrot_musicgen_task"] = synthesiser
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
        pipe = RESOURCE_CACHE["parrot_t2s_task"]
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Model parrot_t2s_task is not loaded")

    try: 
        audio_result = pipe(prompt, **config)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Error when inference")
    
    return audio_result


def run_musicgen(prompt: str, config: dict):
    print(f"[INFO] Run musicgen")
    if "parrot_musicgen_task" not in RESOURCE_CACHE:
        raise Exception("parrot_musicgen_task is not loaded")

    try: 
        synthesiser = RESOURCE_CACHE["parrot_musicgen_task"]
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Model parrot_musicgen_task is not loaded")

    try: 
        music_result = synthesiser(prompt, **config)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Error when inference")
    
    return music_result


def run_audiogen(prompt: str, config: dict):
    print(f"[INFO] Run audiogen")
    if "parrot_audiogen_task" not in RESOURCE_CACHE:
        raise Exception("parrot_audiogen_task is not loaded")

    try:
        model = RESOURCE_CACHE["parrot_audiogen_task"]
        # model.set_generation_params(duration=config["duration"])
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Model parrot_audiogen_task is not loaded")

    try:
        audio_result = model.generate(prompt, **config)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise Exception("Error when inference")

    return audio_result

if __name__=="__main__":
    # print(f"[INFO] Start audio generation service")
    # print(f"[INFO] enabled_tasks: {ENABLED_TASKS}")
    # print(f"[INFO] RESOURCE_CACHE: {RESOURCE_CACHE}")
    # audio_result = run_text2speech("Hello world", {})
    # print(audio_result)
    # music_result = run_musicgen("Hello world", {})
    # print(music_result)
    audio_result = run_audiogen("Hello world", {})
    print(audio_result)