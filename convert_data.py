import os 
from datasets import Dataset
import json
import pandas as pd

languages = {
    'ja': 'Japan',
    'en': 'English',
    'vi': 'Vietnamese',
    'zh-tw': 'Chinese (Taiwanese Mandarin)',
    'zh-cn': 'Chinese (Mainland Mandarin)',
    'ko': 'Korean',
    'fr': 'French',
    'de': 'German',
    'th': 'Thai',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'la': 'Latin',
    'da': 'Danish',
    'hi': 'Hindi',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'pl': 'Polish',
    'es': 'Spanish',
    'tr': 'Turkish'
}

def generate_prompt(row: pd.Series) -> str:
    prompt = """\
<bos><start_of_turn>user
You are a translation expert. Please TRANSALTE the text inside the ``` sign into {} language.
{}
<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(row["dest_lang"], row["origin_text"], row["translated_text"])

    return prompt
def convert_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # print(data)
    df = pd.DataFrame(data)
    df = df.drop(['model_type','workflow', 'created_at', 'id','original_lang' ], axis=1)
    df = df.dropna()
    df['origin_text'] = df['origin_text'].apply(lambda x : " ".join(x))
    df['translated_text'] = df['translated_text'].apply(lambda x : " ".join(x))

    df['dest_lang'] = df['dest_lang'].map(languages)
    df["text"] = df.apply(generate_prompt, axis=1)

    df = df.drop(['origin_text','translated_text','dest_lang'], axis=1)


    df = df.reset_index(drop=True)
    # print(df.head())
    data_dict = {
        "train": Dataset.from_pandas(df)
    }
   
    # save data_dict to json file
    data_dict["train"].to_json("train.jsonl", orient="records", lines=True, force_ascii=False)
    

if __name__ =='__main__':
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(root_dir)
    request_data = {
    "task_id": "123",
    "data_path": "ai_data_1.json",
    "num_train_epochs": 1
}
    
    convert_data(request_data["data_path"])
   
    