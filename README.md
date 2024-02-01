# AI Parrot Worker
This repository allows you to set up a AI Parrot Worker to generate, LoRa training with SD-1.5, SDXL-turbo.

## Some important details you should know before you start

## Installing

### Using Git

```bash
git clone https://github.com/parrotnetwork/parrot-worker.git
cd parrot-worker
pip install -r requirements.txt
```
Continue with the [Basic Usage](#Basic-Usage) instructions

#### Without git

1. Download [the zipped version](https://github.com/parrotnetwork/parrot-worker/archive/refs/heads/main.zip)
2. Extract it to any folder of your choice
3. Continue with the [Basic Usage](#Basic-Usage) instructions

## Basic Usage

### Configure 

1. Make a copy of `.env_template` to `.env`
1. Edit `.env` and follow the instructions within to fill in your details.

### Starting/Stopping

#### Starting the worker

In the terminal in which it's running

```bash
CUDA_VISBLE_DEVICES=0 sh parrot_worker.sh
```
Or
```bash
CUDA_VISBLE_DEVICES=0 python server.py
```

#### Stopping the worker

* In the terminal in which it's running, simply press `Ctrl+C` together.
* The worker will finish the current jobs before exiting.


## Docker

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