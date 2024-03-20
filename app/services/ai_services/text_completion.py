import os
import shutil
import sys
from torch import Tensor
import torch.nn.functional as F


from dotenv import load_dotenv
_ = load_dotenv()
sys.path.append('./app/services/ai_services/')
sys.path[0] = './app/services/ai_services/'

import torch
from transformers import (AutoTokenizer, AutoModel,
                        AutoModelForCausalLM,
                        BitsAndBytesConfig,
                        HfArgumentParser,
                        TrainingArguments,
                        logging,
                        pipeline)

from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from app.utils.base import remove_documents
import os
from dotenv import load_dotenv

load_dotenv()


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

config_dict = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "optim": "paged_adamw_32bit",
    "logging_steps": 100,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "constant",
    "max_seq_length": 1024,
    "packing": False,
    "save_strategy": "no",
}


if "parrot_gemma_lora_trainer_task" in ENABLED_TASKS:
    try: 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype="float16", 
            bnb_4bit_use_double_quant=False, 
        )
        from huggingface_hub import login
        login()
        # hf_token = os.environ.get('HUGGINGFACE_API_KEY', "")
        model_name = "google/gemma-7b-it"

        tokenizer = AutoTokenizer.from_pretrained(model_name, )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                    quantization_config = bnb_config,
                                    device_map = "auto", )
        
        RESOURCE_CACHE["parrot_gemma_lora_trainer_task"] = {}
        RESOURCE_CACHE["parrot_gemma_lora_trainer_task"]["tokenizer"] =tokenizer
        RESOURCE_CACHE["parrot_gemma_lora_trainer_task"]["model"] = model
        print(f"[INFO] Load model gemma success.")
    except Exception as e:
        print(f"[ERROR] Load model gemma failed. An error occurred: {e}")

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
    model = AutoModel.from_pretrained("thenlper/gte-large").to(DEVICE)

    RESOURCE_CACHE["parrot_gte_task"] = (tokenizer, model)

if "parrot_mistral_embeddings_task" in ENABLED_TASKS:
    print(f"[INFO] Loading Mistral embeddings ...")
    repo = "mesolitica/mistral-embedding-191m-8k-contrastive"

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(DEVICE)

    RESOURCE_CACHE["parrot_mistral_embeddings_task"] = (tokenizer, model)


def run_gemma_trainer(data:list[str], num_train_epochs: int):
    output_dir = "parrot_gemma_trainer"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else: 
        print(f"Directory {output_dir} already exists")

    try :
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
        )
        try :
            dataset_dict = {"text" : data}
            dataset = Dataset.from_dict(dataset_dict)    
        except:
            print('formatting error')
                
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config_dict['num_train_epochs'] if num_train_epochs is None else num_train_epochs,
            per_device_train_batch_size=config_dict['per_device_train_batch_size'],
            gradient_accumulation_steps=config_dict['gradient_accumulation_steps'],
            optim=config_dict['optim'],
            save_steps=config_dict['save_steps'],
            logging_steps=config_dict['logging_steps'],
            learning_rate=config_dict['learning_rate'],
            weight_decay=config_dict['weight_decay'],
            fp16=config_dict['fp16'],
            bf16=config_dict['bf16'],
            max_grad_norm=config_dict['max_grad_norm'],
            max_steps=config_dict['max_steps'],
            warmup_ratio=config_dict['warmup_ratio'],
            group_by_length=config_dict['group_by_length'],
            lr_scheduler_type=config_dict['lr_scheduler_type'],
            report_to="tensorboard",
        )
        
        
        trainer = SFTTrainer(
            model=RESOURCE_CACHE["parrot_gemma_lora_trainer_task"]["model"],
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            # formatting_func=format_prompts_fn,
            max_seq_length=config_dict['max_seq_length'],
            tokenizer=RESOURCE_CACHE["parrot_gemma_lora_trainer_task"]["tokenizer"],
            args=training_arguments,
            packing=config_dict['packing'],
        )
        trainer.train()
        trainer.model.save_pretrained(output_dir)
    except Exception as e:
        print(f"[ERROR]: Error in Gemma trainer: {str(e)}")
        
    os.system(f"zip -r {output_dir}.zip {output_dir}")
    shutil.rmtree(output_dir)
    return f"{output_dir}.zip"


def run_mistral_embeddings(text: str, configs: dict):

    try: 
        tokenizer, model = RESOURCE_CACHE["parrot_mistral_embeddings_task"]
    except KeyError as err:
        raise Exception(f"Mistral embeddings model is not loaded. {str(err)}")

    try: 
        return_tensors = configs.get("return_tensors", "pt")
        padding = configs.get("padding", True)
        input_ids = tokenizer(
            text, 
            return_tensors = return_tensors,
            padding = padding
        ).to(DEVICE)
        result = model.encode(input_ids).cpu().detach().numpy()
        return result
    except Exception as e:
        raise Exception(f"Error in Mistral embeddings model: {str(e)}")


def run_text_completion_gemma_7b(messages: list, configs: dict):
    if messages[0]['role'] == 'system':
        system_prompt = messages[0]['content']
        messages = messages[1:]
        prompt = RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"].tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt = f"{system_prompt}\n{prompt}"
    else:
        prompt = RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"].tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    outputs = RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"](
        prompt,
        max_new_tokens=min(configs.get("max_new_tokens", 256), 4096),
        do_sample=True,
        temperature=max(configs.get("temperature", 0.7), 0.01),
        top_k=configs.get("top_k", 50),
        top_p=configs.get("top_p", 0.95),
    )

    return outputs[0]["generated_text"][len(prompt):]


def run_gte_large(text: str, configs: dict):
    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    try: 
        tokenizer, model = RESOURCE_CACHE["parrot_gte_task"]
    except KeyError as err:
        raise Exception(f"GTE large model is not loaded. {str(err)}")

    try: 
       # Tokenize the input texts
        batch_dict = tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        result = embeddings.cpu().detach().numpy()
        return result
    except Exception as e:
        raise Exception(f"Error in GTE large model: {str(e)}")