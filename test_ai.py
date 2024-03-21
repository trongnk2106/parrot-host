from app.src.v1.schemas.base import I2VRequest
from app.src.v1.img2vid.img2vid import img2vid

if __name__ == "__main__":
    data = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png'

    config = {
        'fps' : 8,
        'seed' : 42,
        'num_inference_steps' : 25,
        'num_frames' : 16, 
        'height' : 576,
        'width' : 1024,
        'decode_chunk_size' : 8,
        'output_video_path' : "./generated.mp4",
    }
    
    request_data = {
        "task_id": "123",
        "img_url": data,
        "config": config
    }
    result = img2vid(
        celery_task_id="123",
        request_data=request_data
    )
    