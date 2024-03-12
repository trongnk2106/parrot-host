import io
import os
import sys
import time
from torch import Tensor


from dotenv import load_dotenv
_ = load_dotenv()
sys.path.append('./app/services/ai_services/')
sys.path[0] = './app/services/ai_services/'

from transformers import AutoTokenizer, pipeline, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# Register
ENABLED_TASKS = os.environ.get('ENABLED_TASKS', '').split(',')

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


if "parrot_gte_task" in ENABLED_TASKS:
    print(f"[INFO] Loading GTE model ...")
    
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
    model = AutoModel.from_pretrained("thenlper/gte-large").to("cuda")

    RESOURCE_CACHE["parrot_gte_task"] = (tokenizer, model)


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

# def average_pool(last_hidden_states: Tensor,
#                     attention_mask: Tensor) -> Tensor:
#         last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#         return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def run_gte_large(prompt: str, configs: dict):

    
    # def average_pool(last_hidden_states: Tensor,
    #                 attention_mask: Tensor) -> Tensor:
    #     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    #     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    try: 
        tokenizer, model = RESOURCE_CACHE["parrot_gte_task"]
    except KeyError:
        raise Exception("GTE large model is not loaded")

    try: 
        
        input_texts = [
            "what is the capital of China?"
        ]
                
       # Tokenize the input texts
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to("cuda")
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])


        return embeddings
    except Exception as e:
        raise Exception(f"Error in GTE large model: {str(e)}")