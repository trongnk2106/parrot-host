import os
import io
import time
import scipy
import torchaudio

from app.base.exception.exception import show_log
from app.services.ai_services.audio_generation import run_text2speech
from app.src.v1.backend.api import (send_done_t2s_task, update_status_for_task)
from app.src.v1.schemas.base import (DoneT2SRequest,T2SRequest, UpdateStatusTaskRequest)
from app.utils.services import minio_client


def text2speech(
        celery_task_id: str,
        request_data: T2SRequest,
): 
    show_log(
        message="function: text2speech, "
                f"celery_task_id: {celery_task_id}"
    )

    try: 
        result = '' 
        t0 = time.time()
        # T2S process
        audio_result, sample_rate = run_text2speech(request_data['prompt'], request_data['config'])
        t1 = time.time()
        show_log(f"Time generated: {t1-t0}")
        
        # # Save the AUDIO to the BytesIO object as bytes
        scipy.io.wavfile.write(f"./tmp/{celery_task_id}.wav", rate=sample_rate, data=audio_result)
        # Read the file as bytes
        audio_bytesio = io.BytesIO()
        with open(f"./tmp/{celery_task_id}.wav", "rb") as f:
            audio_bytesio.write(f.read())

        # Upload to MinIO
        s3_key = f"generated_result/{request_data['task_id']}.wav"
        result = minio_client.minio_upload_file(
            content=audio_bytesio,
            s3_key=s3_key
        )
        t2 = time.time()
        os.remove(f"./tmp/{celery_task_id}.wav")
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
                message="function: text2speech, "
                f"celery_task_id: {celery_task_id}, "
                f"error: {error}"
            )
            return response
        
        send_done_t2s_task(
            DoneT2SRequest(
                task_id=request_data['task_id'],
                url_download=result
            )
        )
        return True, response, None
    
    except Exception as e:
        print(str(e))
        return False, None, str(e)


