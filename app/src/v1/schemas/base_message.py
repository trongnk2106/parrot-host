from typing import Any

from pydantic import BaseModel, Field


class BaseMessageEncoderRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    sub_task_id: int = Field(..., description="sub_task_id")
    prompt: str = Field(..., description="prompt")
    config: dict = Field(None, description="config")


class DoneEncoderTaskRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    sub_task_id: str = Field(..., description="sub_task_id")
    status: str = Field(..., description="status")
    node_ip: str = Field(..., description="node_ip")
    result: str = Field(None, description="result")  # Valid string
    vector_result: Any = Field(None, description="vector_result")


class BaseMessageSummarizeRequest(BaseModel):
    task_id: str = Field(..., description="task_id")


class DoneSummarizeTaskRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    s3_key: str = Field(..., description="s3_key")
    status: str = Field(..., description="status")


class SelectAllChildVectorsInfoByParentIDRequest(BaseModel):
    parent_id: str = Field(..., description="parent_id")
