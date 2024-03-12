import scipy
import io
import os
import torchaudio
import time
from audiocraft.data.audio import audio_write

from app.base.exception.exception import show_log
from app.services.ai_services.audio_generation import run_musicgen
from app.src.v1.backend.api import (update_status_for_task, send_done_musicgen_task)
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
        audio_result, sample_rate = run_musicgen(request_data['prompt'], request_data['config']) #Tensors
        t1 = time.time()
        show_log(f"Time generated: {t1-t0}")

        # Save audio_result t0 file *.wav
        if os.path.exists("./tmp") is False:
            os.makedirs("./tmp")
        audio_write(f"./tmp/{celery_task_id}", audio_result.cpu(), sample_rate, strategy="loudness", loudness_compressor=True)

        # Read file 1.wav and convert to byte buffer -> upload to MinIO
        waveform, sample_rate = torchaudio.load(f'./tmp/{celery_task_id}.wav')
        byte_buffer = io.BytesIO()
        torchaudio.save(byte_buffer, waveform, sample_rate, format='wav')

        # Upload to MinIO
        s3_key = f"generated_result/{request_data['task_id']}.wav"
        result = minio_client.minio_upload_file(
            content=byte_buffer,
            s3_key=s3_key
        )
        t2 = time.time()
        os.remove(f"./tmp/{celery_task_id}.wav")
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
                message="function: music, "
                        f"celery_task_id: {celery_task_id}, "
                        f"error: {error}"
            )
            return False
        
        # Send done task
        send_done_musicgen_task(
            DoneMusicGenRequest(
                task_id=request_data['task_id'],
                url_download=result
            )
        )
        return True, response, None
    except Exception as e:
        print(str(e))
        return False, None, str(e)
     