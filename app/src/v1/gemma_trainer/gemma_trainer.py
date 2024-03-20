import time

from app.base.exception.exception import show_log
from app.src.v1.backend.api import update_status_for_task, send_done_gemma_trainer_task
from app.services.ai_services.text_completion import run_gemma_trainer
from app.src.v1.schemas.base import GemmaTrainerRequest, DoneGemmaTrainerRequest, UpdateStatusTaskRequest


def gemma_trainer(
        celery_task_id: str,
        request_data: GemmaTrainerRequest,
):
    show_log(
        message="function: gemma_fineturning, "
                f"celery_task_id: {celery_task_id}"
    )
    
    try :
        t0 = time.time()
        output_dir = run_gemma_trainer(
            data = request_data.data,
            num_train_epochs=request_data.num_train_epochs,   
        )
        t1 = time.time()
        show_log(f"Time processed: {t1-t0}")

        print("Done AI part")
        # update task status
        is_success, response, error = update_status_for_task(
            UpdateStatusTaskRequest(
                task_id=request_data.task_id,
                status="COMPLETED",
                result=str(output_dir)
            )
        )

        if not response:
            show_log(
                message="function: gemma_fineturning, "
                f"celery_task_id: {celery_task_id}, "
                f"error: {error}"
            )
            return response
        
        send_done_gemma_trainer_task(
            DoneGemmaTrainerRequest(
                task_id=request_data.task_id,
                url_download=str(output_dir)
            )
        )
        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)
    