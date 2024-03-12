import time

class UnitTest():
    def test_t2a_audiogen():
        from app.src.v1.audio_gen.audio import audio
        
        t0 = time.time()
        result = audio(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "prompt": "rock music",
                "config": {
                    "duration": 5
                }
            }
        )    
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")

    def test_t2a_musicgen():
        from app.src.v1.music_gen.music import music
        t0 = time.time()

        result = music(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "prompt": "rock music",
                "config": {
                    "duration": 15,                    
                    "max_new_tokens": 512, 
                    "do_sample": True
                }
            }
        )
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")

    def test_t2a_speechgen():  
        from app.src.v1.bark.bark_txt2speech import text2speech
        t0 = time.time()
        result = text2speech(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "prompt": "hello world",
                "config": {
                    "voice": "en-US-Wavenet-D"
                }
            }
        )
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")

    def test_t2a_speechgen():  
        from app.src.v1.bark.bark_txt2speech import text2speech
        t0 = time.time()
        result = text2speech(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "prompt": "hello world",
                "config": {
                    "voice": "en-US-Wavenet-D"
                }
            }
        )
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")


    def test_text_embedding_task():
        from app.src.v1.gte_embedding.gte import text_embedding
        t0 = time.time()
        result = text_embedding(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "text": "the beautifull girl",
                "config": {
                }
            }
        )
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")
    

    def test_t2i_sdxl_lightning_task():
        from app.src.v1.sdxl.sdxl import sdxl_lightning
        t0 = time.time()
        result = sdxl_lightning(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "prompt": "the cake",
                "config": {
                }
            }
        )
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")

    def test_t2i_sdxl_task():
        from app.src.v1.sdxl.sdxl import sdxl
        t0 = time.time()
        result = sdxl(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "prompt": "the car",
                "config": {
                }
            }
        )
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")

    def test_t2i_sd_task():
        from app.src.v1.sd.sd import sd
        t0 = time.time()
        result = sd(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "prompt": "the parrot on tree",
                "config": {
                }
            }
        )
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")

    def test_t2v_task():
        from app.src.v1.txt2vid.txt2vid import txt2vid
        t0 = time.time()
        result = txt2vid(
            celery_task_id="123",
            request_data={
                "task_id": "123",
                "prompt": "the car",
                "config": {
                }
            }
        )
        t1 = time.time()
        print(result)
        print(f"Time proccessed: {t1-t0}s")        


if __name__ == "__main__":
    # UnitTest.test_t2a_audiogen()

    # UnitTest.test_t2a_musicgen()

    # UnitTest.test_t2a_speechgen()    

    # UnitTest.test_text_embedding_task()

    # UnitTest.test_t2i_sdxl_lightning_task()

    # UnitTest.test_t2i_sdxl_task()

    # UnitTest.test_t2i_sd_task()

    UnitTest.test_t2v_task()