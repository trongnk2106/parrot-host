import time

from app.base.exception.exception import show_log
from app.src.v1.schemas.base import DoneGTERequest, GTERequest
from app.src.v1.backend.api import update_status_for_task, send_progress_task, send_done_gte_task
from app.services.ai_services.text_completion import run_gte_large


def text_embedding(
        celery_task_id: str,
        request_data: GTERequest,
):
    show_log(
        message="function: text_embedding "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        response = run_gte_large(request_data['messages'], request_data['config']).tolist()

        if len(response) == 0:
            show_log(
                message="function: text_embedding"
                        f"celery_task_id: {celery_task_id}, "
                        f"error: Update task status failed",
                level="error"
            )
            return response
        
        # send done task
        send_done_gte_task(
            request_data=DoneGTERequest(
                task_id=request_data['task_id'],
                response={
                    "response": response
                }
            )
        )

        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)