from app.base.exception.exception import show_log
from app.src.v1.backend.api import send_progress_task
from app.src.v1.encoder.encoder import encoder
from app.src.v1.lora_trainner.lora_trainner import lora_trainner
from app.src.v1.lora_trainner.repo_init_resource_files import init_resource_files, init_resource_files_from_urls
from app.src.v1.schemas.base_message import BaseMessageEncoderRequest, BaseMessageSummarizeRequest
from app.src.v1.schemas.lora_trainner import LoraTrainnerRequest, SendProgressTaskRequest, SDXLRequest, SDRequest
from app.src.v1.sd.sd import sd
from app.src.v1.sdxl.sdxl import sdxl
from app.src.v1.sdxl.sdxl import sdxl_lightning
from app.src.v1.summarize.summarize import summarize


def worker_encoder(
        request_data: BaseMessageEncoderRequest,
        celery_task_id: str,
        celery_task_name: str,
):
    show_log(
        message="function: worker_encoder, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )
    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="IMAGE_GENERATE_BY_PROMPT",
            sub_task_id=request_data['sub_task_id'],
            percent=10
        )
    )
    return encoder(
        request_data=request_data,
        celery_task_id=celery_task_id,
    )


def worker_diffuser(
        celery_task_id: str,
        celery_task_name: str,
        request_data: BaseMessageSummarizeRequest,
):
    show_log(
        message="function: worker_diffuser, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )
    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="IMAGE_GENERATE_BY_PROMPT",
            percent=10
        )
    )
    is_success, response, error = summarize(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_lora_trainner(
        celery_task_id: str,
        celery_task_name: str,
        request_data: LoraTrainnerRequest,
):
    show_log(
        message="function: worker_lora_trainner, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    input_paths, output_paths, minio_output_paths = init_resource_files_from_urls(
        prompt=request_data['prompt'],
        file_urls=request_data['minio_input_paths']
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="LORA_TRAINNER",
            percent=10
        )
    )
    is_success, response, error = lora_trainner(
        celery_task_id=celery_task_id,
        request_data=request_data,
        input_paths=input_paths,
        output_paths=output_paths,
        minio_output_paths=minio_output_paths,
    )
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_sdxl(
        celery_task_id: str,
        celery_task_name: str,
        request_data: SDXLRequest,
):
    show_log(
        message="function: worker_sdxl, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="SDXL",
            percent=10
        )
    )
    is_success, response, error = sdxl(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_sdxl_lightning(
        celery_task_id: str,
        celery_task_name: str,
        request_data: SDXLRequest,
):
    show_log(
        message="function: worker_sdxl_lightning, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="SDXL",
            percent=10
        )
    )
    is_success, response, error = sdxl_lightning(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_sd(
        celery_task_id: str,
        celery_task_name: str,
        request_data: SDRequest,
):
    show_log(
        message="function: worker_sd, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="SD",
            percent=10
        )
    )
    is_success, response, error = sd(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }
