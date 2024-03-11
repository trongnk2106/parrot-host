from app.src.v1.audio_gen.audio import audio
# from app.src.v1.music_gen.music import music
# from app.src.v1.bark.bark_txt2speech import text2speech


if __name__ == "__main__":
    result = audio(
        celery_task_id="123",
        request_data={
            "task_id": "123",
            "prompt": "dog bark",
            "config": {
                "duration": 5
            }
        }
    )

    print(result)

    # result = music(
    #     celery_task_id="123",
    #     request_data={
    #         "task_id": "123",
    #         "prompt": "rock music",
    #         "config": {
    #             "max_new_tokens": 512, 
    #             "do_sample": True
    #         }
    #     }
    # )

    # print(result)

    # result = text2speech(
    #     celery_task_id="123",
    #     request_data={
    #         "task_id": "123",
    #         "prompt": "hello world",
    #         "config": {
    #             "voice": "en-US-Wavenet-D"
    #         }
    #     }
    # )
    # print(result)