# from app.src.v1.audio_gen.audio import audio
# from app.src.v1.music_gen.music import music
# from app.src.v1.bark.bark_txt2speech import text2speech
from app.src.v1.gte_embedding.gte import text_embedding


if __name__ == "__main__":
    # result = audio(
    #     celery_task_id="123",
    #     request_data={
    #         "task_id": "123",
    #         "prompt": "dog bark",
    #         "config": {
    #             "duration": 5
    #         }
    #     }
    # )

    # print(result)

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

    result = text_embedding(
        celery_task_id="123",
        request_data={
            "task_id": "123",
            "messages": "what is the capital of China?",
            "config": {
            }
        }
    )

    print(result)

# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel

# def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# input_texts = [
#     "what is the capital of China?"
# ]

# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
# model = AutoModel.from_pretrained("thenlper/gte-large").to("cuda")

# # Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to("cuda")

# outputs = model(**batch_dict)
# embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# print(embeddings)
