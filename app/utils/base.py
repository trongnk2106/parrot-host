import hashlib
import json
import os
import re
import zipfile
from typing import Union, Any
import urllib.parse
from loguru import logger
from cryptography.fernet import Fernet
import netifaces


def delete_file(file_path: str):
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.error(e)


def json_dump(data):
    try:
        return json.dumps(data)
    except Exception as ex:
        logger.error(str(ex))
        return '{}'


def count_words(string):
    matches = re.findall(r"[\u00ff-\uffff,\s]|[\W_]+", string)
    return len(matches)


def get_language_code(text: str) -> str:
    for char in text:
        if 0x4E00 <= ord(char) <= 0x9FFF:
            return "zh"
        elif 0x0E00 <= ord(char) <= 0x0E7F:
            return "th"
        elif 0xAC00 <= ord(char) <= 0xD7AF:
            return "ko"
        elif 0x3040 <= ord(char) <= 0x309F or 0x30A0 <= ord(char) <= 0x30FF:
            return "ja"
    return "en"


def get_fernet_key():
    key = os.getenv('AI_AUTH_FERNET_KEY')
    key_bytes = key.encode('utf-8')
    fernet = Fernet(key_bytes)
    return fernet


def get_hash(message):
    return hashlib.sha256(message.encode()).hexdigest()


def encrypt(message) -> Union[bool, Any, Any]:
    try:
        if isinstance(message, dict):
            message = json.dumps(message)
        fernet = get_fernet_key()
        encrypted_message = fernet.encrypt(message.encode('utf-8'))
        return encrypted_message.decode('utf-8')
    except Exception as ex:
        logger.error(ex)
        return None


def decrypt(encrypted_message: str, return_json=False) -> Union[bool, Any, Any]:
    try:
        fernet = get_fernet_key()
        encrypted_message = fernet.decrypt(encrypted_message)
        if return_json:
            return True, json.loads(encrypted_message), None
        return True, encrypted_message.strip(), None
    except Exception as ex:
        False, None, str(ex)


# def check_memory_usage(message: Union[str, None] = None):
#     process = psutil.Process()
#     memory_usage = process.memory_info().rss
#
#     # Convert bytes to gigabytes
#     memory_usage_gb = memory_usage / (1024 ** 3)
#     if message:
#         logger.info(f"{message} -> memory_usage_gb: {memory_usage_gb} GB")
#
#     return memory_usage_gb


def remove_documents(file_path: str):
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.error(f"remove_documents -> ex {e}")


def compress_files_to_zip(file_paths: list[str], zip_file_name: str):
    result = False
    try:
        with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname=arcname)
        result = True
    except Exception as ex:
        logger.error(f"compress_files_to_zip -> ex: {ex}")
    finally:
        return result


def is_number(string):
    pattern = r'^[0-9!@#$%^&*()\[\]{}|\\;:"\'<,>.?/-=_+]+$'
    return re.match(pattern, string) is not None


def get_text_translated_with_style(font: str, color: str, content: str, font_size: float = 12):
    return f"""<p style='font-family:{font}; color: {color}; font-size: {font_size}px; letter-spacing: -0.5px; font-weight: 400; border-left:2px solid #c6c6c6; padding-left: 4px; margin-top: 8px;'>{content}</p>"""


def get_encode_password(password):
    if isinstance(password, bytes):
        return urllib.parse.quote_from_bytes(password)
    elif isinstance(password, str):
        return urllib.parse.quote_plus(password)
    else:
        raise TypeError("Password must be bytes or string")


def get_ip_address():
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            ip = addresses[netifaces.AF_INET][0]['addr']
            return ip
