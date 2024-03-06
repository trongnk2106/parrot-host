import io
import time

from app.base.exception.exception import show_log
from app.services.ai_services.audio_generation import run_musicgen
from app.src.v1.backend.api import (send_progress_task, update_status_for_task, send_done_musicgen_task)
from app.src.v1.schemas.base import (DoneMusicGenRequest, MusicGenRequest, SendProgressTaskRequest, UpdateStatusTaskRequest)
from app.utils.services import minio_client


def music(
        celery_task_id: str,
        request_data: MusicGenRequest,
):
    show_log(
        message="function: music, "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        result = '' 
        t0 = time.time()
        # MG process
        music_result = run_musicgen(request_data['prompt'], request_data['config'])
        t1 = time.time()
        print("[INFO] Time generated: ", t1-t0)

        # Save music_result ro the BytesIO object as bytes
        music_bytes_io = io.BytesIO()
        music_result.save(music_bytes_io, format="wav")

        # Upload to MinIO
        s3_key = f"generated_result/{request_data['task_id']}.wav"
        minio_client.minio_upload_file(
            content=music_bytes_io,
            s3_key=s3_key
        )

        t2 = time.time()
        print("[INFO] Time upload to storage", t2-t1)

        result = f"/parrot-prod/{s3_key}"
        # Update task status
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
                task_type="music_gen",
                percent=50
            )
        )

        if not response:
            show_log(
                message="function: music, "
                        f"celery_task_id: {celery_task_id}, "
                        f"error: {error}"
            )
            return False
        
        # Send done task
        send_done_musicgen_task(
            DoneMusicGenRequest(
                task_id=request_data['task_id'],
                task_type="musicgen",
                result=result
            )
        )

        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="musicgen",
                percent=90
            )
        )

        return True, response, None
    except Exception as e:
        print(str(e))
        return False, None, str(e)
     