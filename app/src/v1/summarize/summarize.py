from app.base.exception.exception import show_log
from app.src.v1.backend.api import send_done_diffuser_task, update_status_for_task, \
    select_all_child_vectors_info_by_parent_id, send_progress_task
from app.src.v1.schemas.base_message import BaseMessageSummarizeRequest, DoneSummarizeTaskRequest, \
    SelectAllChildVectorsInfoByParentIDRequest
from app.src.v1.schemas.lora_trainner import UpdateStatusTaskRequest, SendProgressTaskRequest
from app.utils.const.redis import REDIS_GET_IMAGE_GENERATE_BY_PROMPT_KEY, REDIS_GET_IMAGE_GENERATE_BY_PROMPT_LIFE_TIME
from app.utils.services import minio_client

from app.services.ai_services.inference import generate
import io


def summarize(
        celery_task_id: str,
        request_data: BaseMessageSummarizeRequest,
):
    show_log(
        message="function: summarize "
                f"celery_task_id: {celery_task_id}"
    )
    response = None
    try:
        # CODE HERE
        # Logic v1: Chỉ 1 luồng -> Lấy vector và gen ảnh
        # select_query = select_vector_info()
        # response = postgres_client.execute_raw_query(
        #     raw_query=select_query,
        #     task_id=request_data['task_id'] + '_0',
        # )
        is_success, response, error = select_all_child_vectors_info_by_parent_id(
            request_data=SelectAllChildVectorsInfoByParentIDRequest(
                parent_id=request_data['task_id']
            )
        )
        if not is_success or not response:
            return False, response, "Select vector info failed"

        vector_result = response[0][0]
        image_result = generate(vector_result)  # Defaut Step
        # Create a BytesIO object
        image_bytes_io = io.BytesIO()
        # Save the PIL image to the BytesIO object as bytes
        image_result.save(image_bytes_io, format="PNG")

        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="IMAGE_GENERATE_BY_PROMPT",
                percent=50
            )
        )

        s3_key = f"generated_result/{request_data['task_id']}.png"
        minio_client.minio_upload_file(
            content=image_bytes_io,
            s3_key=s3_key
        )

        # TODO: Logic v2: 4 luồng -> Mỗi lần đều check xem đủ 4 vectors chưa -> Nếu đủ thì gen ảnh
        # CODE HERE        

        # NOTE: Hard-code 
        result = f"http://103.186.100.242:9000/parrot-prod/{s3_key}"

        if not response:
            return False, response, "Update task status failed"
        # TODO: HARD CODE RESPONSE
        send_done_diffuser_task(
            DoneSummarizeTaskRequest(
                task_id=request_data['task_id'],
                result=result,
                s3_key=s3_key,
                status="COMPLETED"
            )
        )
        send_progress_task(
            SendProgressTaskRequest(
                task_id=request_data['task_id'],
                task_type="IMAGE_GENERATE_BY_PROMPT",
                percent=90
            )
        )

        return True, response, None
    except Exception as ex:
        show_log(
            message="function: encoder, "
                    f"celery_task_id: {celery_task_id}, "
                    f"error: {ex}"
        )
        return False, response, str(ex)
