from app.src.v1.lora_trainer_v1.v11.lora_trainer import lora_trainer



if __name__=="__main__": 
    result = lora_trainer(
        celery_task_id="123",
        request_data={
            "task_id": "123",
            "prompt": ["the car", "the parrot"],
            "minio_input_paths": ["https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png", "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"]
        }
    )

    print(result)