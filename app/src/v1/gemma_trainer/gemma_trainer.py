import time
import io
from app.base.exception.exception import show_log
from app.src.v1.backend.api import update_status_for_task, send_done_gemma_trainer_task, update_status_for_gemma
from app.services.ai_services.text_completion import run_gemma_trainer
from app.src.v1.schemas.base import GemmaTrainerRequest, DoneGemmaTrainerRequest, UpdateStatusTaskRequest
from app.utils.services import minio_client

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
            minio_input_paths = request_data['minio_input_paths'],
            num_train_epochs=request_data['num_train_epochs'],   
        )
        
        with open(output_dir, 'rb') as f:
            obj_checkpoint = f.read()
            
        s3_key = f"gemma_trainer/{request_data['task_id']}.zip"
        # TODO : Upload weight to s3 
        # minio_client.minio_upload_file(
        #     content=io.BytesIO(obj_checkpoint),
        #     s3_key=s3_key
        # )
        
        url_download = f"http://103.186.100.242:9000/parrot-prod/{s3_key}"
        
        t1 = time.time()
        show_log(f"Time processed: {t1-t0}")

        # update task status
        is_success, response, error = update_status_for_gemma(
            UpdateStatusTaskRequest(
                task_id=request_data['task_id'],
                status="COMPLETED",
                result=url_download
            )
        )
        if not response:
            show_log(
                message="function: gemma_fineturning, "
                f"celery_task_id: {celery_task_id}, "
                f"error: {error}"
            )
            return response
        
        is_success, response, error = send_done_gemma_trainer_task(
            DoneGemmaTrainerRequest(
                task_id=request_data['task_id'],
                url_download=url_download
            )
        )
        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)
    