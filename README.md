
# AI Parrot Host
This repository facilitates the establishment of an AI Parrot Host, enabling tasks ranging from image, video, and audio generation to the utilization of Large Language Models (LLMs) and LoRa training. It offers the capability to set up and manage Generate AI models for a diverse array of applications.

## Pre-installation Requirements
This project requires a setup that meets the following criteria to ensure stable and efficient operation.


- **Python**: Version 3.10. Ensure that Python is properly installed and configured on your system. Use the command `python --version` to check your current Python version.

- **CUDA**: Version 12.1 or compatible. 

  #### Checking and Installing CUDA
  - Use the command `nvcc --version` in the terminal to check your current CUDA version. If you do not have CUDA installed, or if it's not version 12.1, please visit NVIDIA's official website to download and install CUDA 12.1.

- **G++** compiler is required. 
  
  #### Checking and Installing G++
  - To check the current version of G++ installed, open a terminal and type `g++ --version`. If G++ is not installed on your system, you will need to install it via your operating system's package manager.

- **GPU**: If your system includes a GPU, should have at least **12GB VRAM** for **image generation** tasks, **48GB VRAM** for **video generation** tasks, and **16GB VRAM** for **LoRA Training** tasks and **LLMs**.

  #### Checking VRAM
  - You can use a tool like `nvidia-smi` on systems with NVIDIA GPUs to check VRAM capacity.


## Installing

### Using Git

```bash
git clone https://github.com/parrotnetwork/parrot-host.git
cd parrot-host
pip install -r requirements.txt
```
Continue with the [Basic Usage](#Basic-Usage) instructions

#### Without git

1. Download [the zipped version](https://github.com/parrotnetwork/parrot-host/archive/refs/heads/main.zip)
2. Extract it to any folder of your choice
3. Continue with the [Basic Usage](#Basic-Usage) instructions

## Basic Usage

### Configure 

1. Login and get your TOKEN 
```bash
python get_token_and_create_env.py --username <your_user_name> --password <your_password>  
```

1. This project supports three types of tasks with Parrot Host

- **Image Generation**
   - `parrot_sd_task`
   - `parrot_sdxl_task`
   - `parrot_sdxl_lightning_task`

- **Video Generation**
   - `parrot_txt2vid_damo_task`

- **Text Generation**
   - `parrot_llm_gemma_7b_task`

- **LoRA Train**
   - `parrot_lora_trainer_task`

### Starting/Stopping

#### Starting the Host
#####  Linux

In the terminal in which it's running

```bash
CUDA_VISBLE_DEVICES=0 sh scripts/parrot_{task_type}.sh
```
Or
```bash
CUDA_VISBLE_DEVICES=0 python server.py
```
#####  Windows
1. Open file .env and uncomment the line: **ENABLED_TASKS = "parrot_sdxl_lightning_task"**
2. You can change **parrot_sdxl_lightning_task** into other tasks.
3. Run `python server.py` to start server.
#### Stopping the Host

* In the terminal in which it's running, simply press `Ctrl+C` together.
* The worker will finish the current jobs before exiting.


## Docker

### Pulling and Starting with the Image

To start using the Parrot Host service, you first need to pull the Docker image from the Docker Hub

```bash
docker pull parrotnetwork/parrot-worker:base-1.0-sd15-sdxl-lora
```

### Usage Docker Compose

### Building
```bash
docker compose build
```

### Starting
```bash
docker compose up -d
```

### Stopping
```bash
docker compose down
```
