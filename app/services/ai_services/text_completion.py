import io
import os
import sys
import time

from dotenv import load_dotenv
_ = load_dotenv()
sys.path.append('./app/services/ai_services/')
sys.path[0] = './app/services/ai_services/'

from transformers import AutoTokenizer, pipeline
import torch

# Register
ENABLED_TASKS = os.environ.get('ENABLED_TASKS', '').split(',')

RESOURCE_CACHE = {}

if "parrot_llm_gemma_7b_task" in ENABLED_TASKS:
    print(f"[INFO] Loading Gemma 7B ...")
    model = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline_chat = pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.float16},
        device="cuda",
    )
    RESOURCE_CACHE["parrot_llm_gemma-7b_task"] = {}
    RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["tokenizer"] = tokenizer
    RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"] = pipeline_chat


def run_text_completion_gemma_7b(messages: list, configs: dict):
    prompt = RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"].tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"](
        prompt,
        max_new_tokens=configs.get("max_new_tokens", 256),
        do_sample=True,
        temperature=max(configs.get("temperature", 0.7), 0.01),
        top_k=configs.get("top_k", 50),
        top_p=configs.get("top_p", 0.95),
    )

    return outputs[0]["generated_text"][len(prompt):]
