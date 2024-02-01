accelerate launch \
    --num_cpu_threads_per_process=2 "train_network.py" \
    --config_file='./train_network_config.toml'