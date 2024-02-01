from app.base.exception.exception import show_log
from app.src.v1.backend.api import send_done_encoder_task, update_status_for_task, send_progress_task
from app.src.v1.schemas.base_message import BaseMessageEncoderRequest, DoneEncoderTaskRequest
from app.src.v1.schemas.lora_trainner import UpdateStatusTaskRequest, SendProgressTaskRequest
from app.utils.base import get_ip_address

from app.services.ai_services.inference import run_encoder

def encoder(
        request_data: BaseMessageEncoderRequest,
        celery_task_id: str
):
    show_log(
        message="function: refresh_keys "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        node_ip = get_ip_address()

        # CODE HERE
        # v1: encode and save vector
        # TODO: Log config
        '''
        request_data = \
            {
                'prompt': 'the cat and the dog', 
                'index': 0, 
                'task_id': '1d8d9659007248338d6d96808ff6b87f', 
                'sub_task_id': '1d8d9659007248338d6d96808ff6b87f_0', 
                'total_tasks': 1, 
                'task_index': 0
            }
        '''
        # result = request_data['prompt'].upper() + f"_DONE_{node_ip}"

        vector_result = run_encoder(prompt=request_data['prompt'])
        result = ''

        # END
        # update task status
        is_success, response, error = update_status_for_task(
            UpdateStatusTaskRequest(
                task_id=request_data['sub_task_id'],
                status="COMPLETED",
                result=result,
                vector_result=vector_result
            )
        )
        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="IMAGE_GENERATE_BY_PROMPT",
                sub_task_id=request_data['sub_task_id'],
                percent=50
            )
        )
        if not response:
            show_log(
                message="function: encoder, "
                        f"celery_task_id: {celery_task_id}, "
                        f"error: Update task status failed",
                level="error"
            )
            return response
        # send done task
        send_done_encoder_task(
            request_data=DoneEncoderTaskRequest(
                task_id=request_data['task_id'],
                sub_task_id=request_data['sub_task_id'],
                status="COMPLETED",
                node_ip=node_ip,
                result=result
            )
        )
        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="IMAGE_GENERATE_BY_PROMPT",
                sub_task_id=request_data['sub_task_id'],
                percent=90
            )
        )
        return response
    except Exception as ex:
        show_log(
            message="function: encoder, "
                    f"celery_task_id: {celery_task_id}, "
                    f"error: {ex}"
        )
        return None
