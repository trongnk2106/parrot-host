import io
import shutil
import os 

from app.base.exception.exception import show_log
from app.src.v1.backend.api import send_done_lora_trainner_task, update_status_for_task, send_progress_task
from app.src.v1.schemas.base import LoraTrainnerRequest, DoneLoraTrainnerRequest, UpdateStatusTaskRequest, \
    SendProgressTaskRequest
from app.utils.base import remove_documents
from app.utils.services import minio_client

import subprocess

def lora_trainner(
        celery_task_id: str,
        request_data: LoraTrainnerRequest,
        input_paths: list[str],
        output_paths: list[str],
        minio_output_paths: list[str],
):
    show_log(
        message="function: lora_trainner, "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        
        url_download = ""
    
        try:
            subprocess.run(
                ["bash", "train_lora.sh"],
                cwd='./app/services/ai_services/trainner-script/'
            )
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with return code {e.returncode}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        checkpoint_path = "resources/output/parrot_lora.safetensors"
        if not os.path.exists(checkpoint_path):
            is_success, response, error = update_status_for_task(
                UpdateStatusTaskRequest(
                    task_id=request_data['task_id'],
                    status="FAILED",
                    result=url_download,
                )
            )  
            
            return False, None, "Failed Lora Trainning"
        
        with open(checkpoint_path, 'rb') as f:
            obj_checkpoint = f.read()

        s3_key = f"generated_result/{request_data['task_id']}.safetensors"
        minio_client.minio_upload_file(
            content=io.BytesIO(obj_checkpoint),
            s3_key=s3_key
        )
        url_download = f"http://103.186.100.242:9000/parrot-prod/{s3_key}"
            
        # update task status
        is_success, response, error = update_status_for_task(
            UpdateStatusTaskRequest(
                task_id=request_data['task_id'],
                status="COMPLETED",
                result=url_download,
            )
        )
        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="LORA_TRAINNER",
                percent=50
            )
        )

        if not response:
            show_log(
                message="function: lora_trainner, "
                        f"celery_task_id: {celery_task_id}, "
                        f"error: Update task status failed",
                level="error"
            )
            return response

        # send done task
        send_done_lora_trainner_task(
            request_data=DoneLoraTrainnerRequest(
                task_id=request_data['task_id'],
                url_download=url_download
            )
        )
        # remove file in local
        input_paths = [os.path.join("resources/input/images", filename) for filename in os.listdir("resources/input/images")]
        output_paths = [os.path.join("resources/output", filename) for filename in os.listdir("resources/output")]
        [remove_documents(path) for path in input_paths]
        [remove_documents(path) for path in output_paths]

        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="LORA_TRAINNER",
                percent=90
            )
        )
        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)
