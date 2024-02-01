import os
from dotenv import load_dotenv

from app.services.minio.minio_client import MinioClient

load_dotenv()

minio_client = MinioClient(config={
    "minio_end_point": os.environ.get("MINIO_END_POINT"),
    "minio_access_key_id": os.environ.get("MINIO_ACCESS_KEY_ID"),
    "minio_secret_access_key": os.environ.get("MINIO_SECRET_ACCESS_KEY"),
    "minio_bucket_name": os.environ.get("MINIO_BUCKET_NAME"),
    "minio_secure": True if os.environ.get("MINIO_SECURE") in ["True", "True", "TRUE", '1', 1] else False
})

JWT_TOKEN = os.getenv("JWT_TOKEN")
