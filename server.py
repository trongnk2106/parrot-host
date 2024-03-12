import os 
from tasks import celery_app
from app.utils.const.taskmanager import TASK_MANAGEMENT


# Register
enabled_tasks = os.environ.get('ENABLED_TASKS', '').split(',')

def register_task(task_func):
        celery_app.task(
            bind=True,
            max_retries=int(os.environ['CELERY_MAX_RETRIES'])
        )(task_func)

try: 
    task_name = enabled_tasks[0]
    queue_name = TASK_MANAGEMENT[task_name]["queue_name"]
    load_model_function = TASK_MANAGEMENT[task_name]["load_model_function"]
except: 
    raise ValueError(f"Task {enabled_tasks[0]} is not supported. Please check the enabled_tasks environment variable.")

register_task(load_model_function)


if __name__ == "__main__":
    import sys
    worker_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"worker_number: {worker_number}")
    
    userhost = os.environ.get('USERNAME', 'guest')
    celery_app.worker_main(
        argv=['-A', 'tasks', 'worker', '--loglevel=info', 
            '-c', f'{worker_number}', 
            '-n', f'{queue_name.replace("_queue", "_task")}.{userhost}', 
            '-Q', f'{queue_name}', 
            '--pool', 'solo'
        ]
    )

