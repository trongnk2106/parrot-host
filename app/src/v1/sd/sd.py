import io
import time

from app.base.exception.exception import show_log
from app.services.ai_services.image_generation import run_sd
from app.src.v1.backend.api import (send_done_sd_task, send_progress_task, update_status_for_task)
from app.src.v1.schemas.base import (DoneSDRequest, SDRequest, SendProgressTaskRequest, UpdateStatusTaskRequest)
from app.utils.services import minio_client


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
        show_log(f"Time generated: {t1-t0}")
        
        # Save the PIL image to the BytesIO object as bytes
        image_bytes_io = io.BytesIO()
        image_result.save(image_bytes_io, format="PNG")
        
        # Upload to MinIO
        s3_key = f"generated_result/{request_data['task_id']}.png"
        result = minio_client.minio_upload_file(
            content=image_bytes_io,
            s3_key=s3_key
        )
        
        t2 = time.time()
        show_log(f"Time upload to storage {t2-t1}")
        show_log(f"Result URL: {result}")

        # Update task status
        is_success, response, error = update_status_for_task(
            UpdateStatusTaskRequest(
                task_id=request_data['task_id'],
                status="COMPLETED",
                result=result
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
        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)
