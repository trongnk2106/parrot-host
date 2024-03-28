from app.src.v1.schemas.base import LLMRequest
from app.src.v1.mistral_llm.text_completion import text_completion_mistral

if __name__ == "__main__":

    messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]
    request_data = {
        "task_id": "123",
        "messages": messages,
        "config": {}
    }
    result = text_completion_mistral(
        celery_task_id="123",
        request_data=request_data
    )
