# AI Parrot Trainer
This repository allows you to set up a AI Parrot Trainer to generate, LoRa training with SD-1.5, SDXL-turbo.



## Pre-installation Requirements
This project requires a setup that meets the following criteria to ensure stable and efficient operation.


- **Python**: Version 3.10. Ensure that Python is properly installed and configured on your system. Use the command `python --version` to check your current Python version.

- **CUDA**: Version 12.1 or compatible. 

  #### Checking and Installing CUDA
  - Use the command `nvcc --version` in the terminal to check your current CUDA version. If you do not have CUDA installed, or if it's not version 12.1, please visit NVIDIA's official website to download and install CUDA 12.1.

- **G++** compiler is required. 
  
  #### Checking and Installing G++
  - To check the current version of G++ installed, open a terminal and type `g++ --version`. If G++ is not installed on your system, you will need to install it via your operating system's package manager.

- **GPU**: If your system includes a GPU, a minimum of **9GB VRAM** is required for **image generation tasks**. For **LoRA Training tasks**, a minimum of **16GB VRAM** is required.

  #### Checking VRAM
  - You can use a tool like `nvidia-smi` on systems with NVIDIA GPUs to check VRAM capacity.


## Installing

### Using Git

```bash
git clone https://github.com/parrotnetwork/parrot-trainer.git
cd parrot-trainer
pip install -r requirements.txt
```
Continue with the [Basic Usage](#Basic-Usage) instructions

#### Without git

1. Download [the zipped version](https://github.com/parrotnetwork/parrot-trainer/archive/refs/heads/main.zip)
2. Extract it to any folder of your choice
3. Continue with the [Basic Usage](#Basic-Usage) instructions

## Basic Usage

### Configure 

1. Make a copy of `.env_template` to `.env`
1. Edit `.env` and follow the instructions within to fill in your details.
1. This project supports three types of tasks with Parrot Trainer

- **parrot_sd_task**
- **parrot_sdxl_task**
- **parrot_lora_trainner_task**

### Starting/Stopping

#### Starting the Trainer

In the terminal in which it's running

```bash
CUDA_VISBLE_DEVICES=0 sh parrot_worker.sh
```
Or
```bash
CUDA_VISBLE_DEVICES=0 python server.py
```

#### Stopping the Trainer

* In the terminal in which it's running, simply press `Ctrl+C` together.
* The worker will finish the current jobs before exiting.


## Docker

### Pulling and Starting with the Image

To start using the Parrot Trainer service, you first need to pull the Docker image from the Docker Hub

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
