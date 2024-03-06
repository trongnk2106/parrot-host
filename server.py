import os 

from tasks import celery_app
from tasks import parrot_sd_task, parrot_sdxl_task, parrot_sdxl_lightning_task, parrot_lora_trainer_task, parrot_txt2vid_damo_task, parrot_llm_gemma_7b_task, parrot_t2s_task, parrot_musicgen_task, parrot_audiogen_task

# Register
enabled_tasks = os.environ.get('ENABLED_TASKS', '').split(',')
def register_task(task_func, task_name):
    if task_name in enabled_tasks:
        celery_app.task(
            bind=True,
            max_retries=int(os.environ['CELERY_MAX_RETRIES'])
        )(task_func)

register_task(parrot_sd_task, "parrot_sd_task")
register_task(parrot_sdxl_task, "parrot_sdxl_task")
register_task(parrot_sdxl_lightning_task, "parrot_sdxl_lightning_task")
register_task(parrot_lora_trainer_task, "parrot_lora_trainer_task")
register_task(parrot_txt2vid_damo_task, "parrot_txt2vid_damo_task")
register_task(parrot_llm_gemma_7b_task, "parrot_llm_gemma_7b_task")
register_task(parrot_t2s_task, "parrot_t2s_task")
register_task(parrot_musicgen_task, "parrot_musicgen_task")
register_task(parrot_audiogen_task, "parrot_audiogen_task")


if __name__ == "__main__":
    import sys
    worker_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"worker_number: {worker_number}")
    
    
    queue_name = "sd_queue"

    print(f"enabled_tasks: {enabled_tasks}")
    if "parrot_sd_task" in enabled_tasks:
        queue_name = "sd_queue"
    if "parrot_sdxl_task" in enabled_tasks:
        queue_name = "sdxl_queue"
    if "parrot_sdxl_lightning_task" in enabled_tasks:
        queue_name = "sdxl_lightning_queue"
    if "parrot_lora_trainer_task" in enabled_tasks:
        queue_name = "lora_trainer_queue"
    if "parrot_txt2vid_damo_task" in enabled_tasks:
        queue_name = "txt2vid_damo_queue"
    if "parrot_llm_gemma_7b_task" in enabled_tasks:
        queue_name = "llm_gemma_7b_queue"
    if "parrot_t2s_task" in enabled_tasks:
        queue_name = "t2s_queue"
    if "parrot_musicgen_task" in enabled_tasks:
        queue_name = "musicgen_queue"
    if "parrot_audiogen_task" in enabled_tasks:
        queue_name = "audiogen_queue"
    

    print(f"queue_name: {queue_name}")
    userhost = os.environ.get('USERNAME', 'guest')
    celery_app.worker_main(
        argv=['-A', 'tasks', 'worker', '--loglevel=info', 
            '-c', f'{worker_number}', 
            '-n', f'{queue_name.replace("_queue", "_task")}.{userhost}', 
            '-Q', f'{queue_name}', 
            '--pool', 'solo'
        ]
    )

