import os

from app.src.v1.backend.api import send_fail_task, send_done_task
from app.src.v1.schemas.lora_trainner import SendFailTaskRequest, SendDoneTaskRequest
from app.src.v1.workers import *
from app.src.v1.schemas.base_message import *
from celery import Celery
from celery.states import SUCCESS, FAILURE
from pathlib import Path

celery_app = Celery(
    os.environ['CELERY_APP_NAME'],
    broker=f"amqp://"
           f"{os.environ['CELERY_SERVICE_WORKER_USERNAME']}:{os.environ['CELERY_SERVICE_WORKER_PASSWORD']}"
           f"@{os.environ['CELERY_SERVICE_WORKER_HOST']}:{os.environ['CELERY_SERVICE_WORKER_PORT']}//",
    broker_connection_retry_on_startup=True,
    result_expires=int(os.environ['CELERY_RESULT_EXPIRES'])
)

Path("resources/input/images").mkdir(parents=True, exist_ok=True)
Path("resources/output/images").mkdir(parents=True, exist_ok=True)


# @celery_app.task(
#     bind=True,
#     max_retries=int(os.environ['CELERY_MAX_RETRIES'])
# )
def parrot_encoder_task(self, request_data: BaseMessageEncoderRequest):
    result = None
    try:
        result = worker_encoder(
            celery_task_id=self.request.id,
            celery_task_name=self.name,
            request_data=request_data
        )
        if not result:
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="IMAGE_GENERATE_BY_PROMPT",
                sub_task_id=request_data['sub_task_id']
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="IMAGE_GENERATE_BY_PROMPT",
                    sub_task_id=request_data['sub_task_id']
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))


# @celery_app.task(
#     bind=True,
#     max_retries=int(os.environ['CELERY_MAX_RETRIES'])
# )
def parrot_diffuser_task(self, request_data):
    result = None
    try:
        result = worker_diffuser(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="IMAGE_GENERATE_BY_PROMPT",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="IMAGE_GENERATE_BY_PROMPT",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))


# @celery_app.task(
#     bind=True,
#     max_retries=int(os.environ['CELERY_MAX_RETRIES'])
# )
def parrot_lora_trainner_task(self, request_data):
    result = None
    try:
        result = worker_lora_trainner(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="LORA_TRAINNER",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="LORA_TRAINNER",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))


# @celery_app.task(
#     bind=True,
#     max_retries=int(os.environ['CELERY_MAX_RETRIES'])
# )
def parrot_sdxl_task(self, request_data):
    result = None
    try:
        result = worker_sdxl(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="SDXL",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="SDXL",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))


# @celery_app.task(
#     bind=True,
#     max_retries=int(os.environ['CELERY_MAX_RETRIES'])
# )
def parrot_sd_task(self, request_data):
    result = None
    try:
        result = worker_sd(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="SD",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="SD",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))

