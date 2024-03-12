import io
import logging

from minio import Minio

logger = logging.getLogger("minio_logger")


class MinioClient:
    def __init__(self, config: dict):
        self._config = config
        self.minio_url = config.get("minio_end_point")
        self.access_key = config.get("minio_access_key_id")
        self.secret_key = config.get("minio_secret_access_key")
        self.minio_bucket_name = config.get("minio_bucket_name")
        self.secure = True if config.get("minio_secure") in ["True", "True", "TRUE", '1', 1] else False
        self.part_size = 10 * 1024 * 1024
        self._minio_client = Minio(
            self.minio_url,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

    def minio_upload_file(self, content: io.BytesIO, s3_key: str, minio_bucket_name=None, part_size: int = None):
        try:
            content.seek(0)
            self._minio_client.put_object(
                self.minio_bucket_name if not minio_bucket_name else minio_bucket_name,
                object_name=s3_key,
                data=content,
                length=-1,
                part_size=self.part_size if not part_size else part_size,
            )

            return f"https://{self.minio_url}/{self.minio_bucket_name}/{s3_key}"
        except Exception as e:
            raise RuntimeError('Upload error for key "%s".' % s3_key) from e
