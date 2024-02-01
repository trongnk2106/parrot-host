REDIS_REFRESH_FLAG = "api_key_center_worker:refresh_flag"
REDIS_KEY_EST_TIME = 60  # 1 minutes
REDIS_KEY_LIFE_TIME = 60 * 60 * 24 * 90  # 90 days
REDIS_GET_IMAGE_GENERATE_BY_PROMPT_KEY = "msd:get_image_generate_by_prompt:"
REDIS_GET_IMAGE_GENERATE_BY_PROMPT_LIFE_TIME = 60 * 15  # 5 minutes

REDIS_KEY_TYPES = {
    'gpt-3.5-turbo': {
        'all': 'api_key_center_worker:gpt3.5_turbo_keys',
        'child': 'api_key_center_worker:gpt3.5_turbo_keys:*',
    },
    'gpt-4': {
        'all': 'api_key_center_worker:gpt4_keys',
        'child': 'api_key_center_worker:gpt4_keys:*'
    },
    'gemini': {
        'all': 'api_key_center_worker:gemini_keys',
        'child': 'api_key_center_worker:gemini_keys:*'
    }
}


def get_key_type(key_type: str, is_child: bool = False):
    key_type = key_type.lower().strip()
    if is_child:
        return REDIS_KEY_TYPES[key_type]['child']
    return REDIS_KEY_TYPES[key_type]['all']


def get_worker_task_result(celery_task_id: str):
    return f"api_key_center_backend:results:{celery_task_id}"
