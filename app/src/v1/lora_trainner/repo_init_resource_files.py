import os
import urllib.request

from app.utils.services import minio_client


def init_resource_files(s3_input_keys: list[str]):
    input_paths = []
    output_paths = []
    minio_output_paths = []
    for s3_key_in in s3_input_keys:
        filename = s3_key_in[s3_key_in.rfind("/") + 1:]
        src_in = f"resources/input/images/" + filename
        src_out = f"resources/output/images/" + filename
        if not os.path.exists(src_in + filename):
            minio_client.minio_download_to_bytes(s3_key=s3_key_in, output_path=src_in)
        input_paths.append(src_in)
        output_paths.append(src_out)
        minio_output_paths.append(s3_key_in.replace("/input/", "/output/"))

    return input_paths, output_paths, minio_output_paths


def init_resource_files_from_urls(prompt: str, file_urls: list[str]):
    input_paths = []
    output_paths = []
    minio_output_paths = []
    for idx, file_url in enumerate(file_urls):
        filename = file_url[file_url.rfind("/") + 1:].split("?")[0]
        src_in = f"resources/input/images/" + filename
        src_out = f"resources/output/images/" + filename
        if not os.path.exists(src_in + filename):
            download_file(url=file_url, filename=src_in)
        
        filename_txt = filename.split('.')[0] + '.txt'
        with open(os.path.join("resources/input/images/", filename_txt), 'w') as f:
            f.write(prompt)
        f.close()
        
        input_paths.append(src_in)
        output_paths.append(src_out)
        s3_key_in = f"input/images/" + filename
        minio_output_paths.append(s3_key_in.replace("/input/", "/output/"))
    return input_paths, output_paths, minio_output_paths


def download_file(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as ex:
        print(f"download_file error: {ex}")
        return False
