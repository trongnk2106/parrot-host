import io
import time

from app.base.exception.exception import show_log
from app.services.ai_services.audio_generation import run_audiogen
from app.src.v1.backend.api import (send_progress_task, update_status_for_task, send_done_audiogen_task)
from app.src.v1.schemas.base import (DoneAudioGenRequest, AudioGenRequest, SendProgressTaskRequest, UpdateStatusTaskRequest)
from app.utils.services import minio_client


def audio(
        celery_task_id: str,
        request_data: AudioGenRequest,
):
    show_log(
        message="function: audio, "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        result = '' 
        t0 = time.time()
        # MG process
        audio_result = run_audiogen(request_data['prompt'], request_data['config'])
        t1 = time.time()
        print("[INFO] Time generated: ", t1-t0)

        # Save audio_result ro the BytesIO object as bytes
        audio_bytes_io = io.BytesIO()
        audio_result.save(audio_bytes_io, format="wav")

        # Upload to MinIO
        s3_key = f"generated_result/{request_data['task_id']}.wav"
        minio_client.minio_upload_file(
            content=audio_bytes_io,
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
                task_type="audio_gen",
                percent=50
            )
        )

        if not response:
            show_log(
                message="function: audio, "
                f"celery_task_id: {celery_task_id}, "
                f"error: {error}",
                level="error"
            )
            return response

        # Send done task
        send_done_audiogen_task(
            DoneAudioGenRequest(
                task_id=request_data['task_id'],
                result=result
            )
        )
        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="audio_gen",
                percent=90
            )
        )
        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)