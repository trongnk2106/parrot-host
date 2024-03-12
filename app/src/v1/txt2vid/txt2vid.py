import time

from app.base.exception.exception import show_log
from app.src.v1.backend.api import update_status_for_task, send_progress_task, send_done_sd_task
from app.src.v1.schemas.base import UpdateStatusTaskRequest, \
    SendProgressTaskRequest, SDRequest, DoneSDRequest
from app.utils.services import minio_client

from app.services.ai_services.image_generation import run_txt2vid

def txt2vid(
        celery_task_id: str,
        request_data: SDRequest,
):
    show_log(
        message="function: txt2vid"
                f"celery_task_id: {celery_task_id}"
    )
    try:
        result = '' 
        
        t0 = time.time()
        video_byte_io = run_txt2vid(request_data['prompt'], request_data['config'])
        t1 = time.time()
        show_log(f"Time generated: {t1-t0}")

        # Upload to MinIO
        s3_key = f"generated_result/{request_data['task_id']}.mp4"
        result = minio_client.minio_upload_file(
            content=video_byte_io,
            s3_key=s3_key
        )
        
        t2 = time.time()
        show_log(f"Time upload to storage {t2-t1}")
        show_log(f"Result URL: {result}")
        
        # update task status
        is_success, response, error = update_status_for_task(
            UpdateStatusTaskRequest(
                task_id=request_data['task_id'],
                status="COMPLETED",
                result=result
            )
        )

        if not response:
            show_log(
                message="function: txt2vid "
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
