import time

from app.base.exception.exception import show_log
from app.src.v1.schemas.base import LLMRequest, DoneLLMRequest
from app.src.v1.backend.api import update_status_for_task, send_progress_task, send_done_llm_task
from app.services.ai_services.text_completion import run_text_completion_mistral_7b
from app.src.v1.schemas.base import LLMRequest, DoneLLMRequest 

def text_completion_mistral(
        celery_task_id: str,
        request_data: LLMRequest,
):
    show_log(
        message="function: text_completion "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        response = run_text_completion_mistral_7b(request_data['messages'], request_data['config'])
        if not response:
            show_log(
                message="function: text_completion"
                        f"celery_task_id: {celery_task_id}, "
                        f"error: Update task status failed",
                level="error"
            )
            return response
        
        # send done task
        send_done_llm_task(
            request_data=DoneLLMRequest(
                task_id=request_data['task_id'],
                response=response
            )
        )

        return True, response, None
    except Exception as e:
        print(e)
        return False, None, str(e)

