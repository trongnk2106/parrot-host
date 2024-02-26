import os

import requests
from app.base.exception.exception import show_log
from app.src.v1.schemas.base_message import DoneEncoderTaskRequest, DoneSummarizeTaskRequest, \
    SelectAllChildVectorsInfoByParentIDRequest
from app.src.v1.schemas.lora_trainner import DoneLoraTrainnerRequest, UpdateStatusTaskRequest, SendFailTaskRequest, \
    SendDoneTaskRequest, SendProgressTaskRequest, DoneSDXLRequest, DoneSDRequest
from app.utils.services import JWT_TOKEN


def send_done_encoder_task(request_data: DoneEncoderTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_encoder_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_diffuser_task(request_data: DoneSummarizeTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_diffuser_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            data = response.json().get('data', {}).get('data')
            return True, data, None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_lora_trainner_task(request_data: DoneLoraTrainnerRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_lora_trainner_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def update_status_for_task(request_data: UpdateStatusTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/update_status_for_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def select_all_child_vectors_info_by_parent_id(request_data: SelectAllChildVectorsInfoByParentIDRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/select_all_child_vectors_info_by_parent_id",
            json=request_data.dict()
        )
        data = response.json()
        if response.status_code == 200 and data.get('status') == 'success':
            data = data.get('data')
            return True, data.get('data'), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_fail_task(request_data: SendFailTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/fail_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_task(request_data: SendDoneTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_progress_task(request_data: SendProgressTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/progress_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_sdxl_task(request_data: DoneSDXLRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_sdxl_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e

def send_done_sdxl_lightning_task(request_data: DoneSDXLRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_sdxl_lightning_task",
            json=request_data.dict()
        )
        print(response.status_code, response.json())
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e

def send_done_sd_task(request_data: DoneSDRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_sd_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e
