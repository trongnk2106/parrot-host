import time

from app.base.exception.exception import show_log
from app.src.v1.schemas.base import LLMRequest, DoneLLMRequest
from app.src.v1.backend.api import update_status_for_task, send_progress_task, send_done_llm_task
from app.services.ai_services.text_completion import run_text_completion_gemma_7b
from app.src.v1.schemas.base import LLMRequest, DoneLLMRequest 

def text_completion(
        celery_task_id: str,
        request_data: LLMRequest,
):
    show_log(
        message="function: text_completion "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        response = run_text_completion_gemma_7b(request_data['messages'], request_data['config'])
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

# def gemma_finetuning(
#         celery_task_id: str,
#         request_data: GemmaFinetuningRequest,
# ):
#     show_log(
#         message="function: gemma_fineturning, "
#                 f"celery_task_id: {celery_task_id}"
#     )
    
#     try :
#         url_download = ""
#         output_dir = ""      
#         try :
#             output_dir = run_gemma_lora(
#                 data = request_data.data_path,
#                 num_train_epochs=request_data.num_train_epochs,   
#             )
#         except Exception as e:
#             print(f"An error occurred: {e}")
#         with open(output_dir, 'rb') as f:
#             obj_checkpoint = f.read()
        
#         s3_key = f"generated_result/{request_data['task_id']}.zip"
#         minio_client.minio_upload_file(
#             content=io.BytesIO(obj_checkpoint),
#             s3_key=s3_key
#         )
        
#         url_download = f"http://103.186.100.242:9000/parrot-prod/{s3_key}"
#         is_success, response, error = update_status_for_task(
#             UpdateStatusTaskRequest(
#                 task_id=request_data['task_id'],
#                 status="COMPLETED",
#                 result=url_download,
#             )
#         )
        
#         send_progress_task(
#             SendProgressTaskRequest(
#                 task_id=request_data['task_id'],
#                 task_type="GEMMA_FINETURNING",
#                 percent=50
#             )
#         )
        
#         if not response:
#             show_log(
#                 message="function: gemma_lora_trainner, "
#                         f"celery_task_id: {celery_task_id}, "
#                         f"error: Update task status failed",
#                 level="error"
#             )
#             return response
        
#         send_done_lora_trainner_task(
#             request_data=DoneLoraTrainnerRequest(
#                 task_id=request_data['task_id'],
#                 url_download=url_download
#             )
#         )
        
#         remove_documents(request_data['data'])     
        
#         send_progress_task(
#             SendProgressTaskRequest(
#                 task_id=request_data['task_id'],
#                 task_type="GEMMA_FINETURNING",
#                 percent=90
#             )
#         )
#         return True, response, None
#     except Exception as e:
#         print(e)
#         return False, None, str(e)