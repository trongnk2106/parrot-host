import io
import shutil
import time

from app.base.exception.exception import show_log
from app.src.v1.backend.api import update_status_for_task, send_progress_task, send_done_sd_task
from app.src.v1.schemas.lora_trainner import UpdateStatusTaskRequest, \
    SendProgressTaskRequest, SDRequest, DoneSDRequest
from app.utils.base import remove_documents
from app.utils.services import minio_client

from app.services.ai_services.inference import run_sd

def sd(
        celery_task_id: str,
        request_data: SDRequest,
):
    show_log(
        message="function: sd, "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        result = '' 
        
        t0 = time.time()
        # SD process
        image_result = run_sd(request_data['prompt'], request_data['config'])
        t1 = time.time()
        print("Time generated: ", t1-t0)
        
        # Save the PIL image to the BytesIO object as bytes
        image_bytes_io = io.BytesIO()
        image_result.save(image_bytes_io, format="PNG")
        
        # Upload to MinIO
        s3_key = f"generated_result/{request_data['task_id']}.png"
        minio_client.minio_upload_file(
            content=image_bytes_io,
            s3_key=s3_key
        )
        
        t2 = time.time()
        print("Time upload MinIO", t2-t1)
        
        result = f"http://103.186.100.242:9000/parrot-prod/{s3_key}"
        # update task status
        is_success, response, error = update_status_for_task(
            UpdateStatusTaskRequest(
                task_id=request_data['task_id'],
                status="COMPLETED",
                result=result
            )
        )
        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="SD",
                percent=50
            )
        )

        if not response:
            show_log(
                message="function: sd, "
                        f"celery_task_id: {celery_task_id}, "
                        f"error: Update task status failed",
                level="error"
            )
            return response

        # send done task
        send_done_sd_task(
            request_data=DoneSDRequest(
                task_id=request_data['task_id'],
                url_download=result
            )
        )

        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="SD",
                percent=90
            )
        )
        t3 = time.time()
        print("Time post process:", t3-t2)
        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)
