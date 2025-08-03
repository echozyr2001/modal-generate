import boto3
import os
import uuid
import logging
import shutil
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from shared.config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """文件管理器，支持本地存储和S3存储"""
    
    def __init__(self, use_s3: bool = True, local_storage_dir: str = "./outputs"):
        self.use_s3 = use_s3
        self.local_storage_dir = local_storage_dir
        
        if use_s3:
            self.s3_client = boto3.client("s3")
            self.bucket_name = settings.s3_bucket_name
        else:
            # 确保本地存储目录存在
            os.makedirs(local_storage_dir, exist_ok=True)
            logger.info(f"Using local storage: {local_storage_dir}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def save_file(self, file_path: str, file_key: Optional[str] = None) -> str:
        """保存文件，返回文件key或路径"""
        if file_key is None:
            file_extension = os.path.splitext(file_path)[1]
            file_key = f"{uuid.uuid4()}{file_extension}"
        
        try:
            if self.use_s3:
                # 上传到S3
                self.s3_client.upload_file(file_path, self.bucket_name, file_key)
                logger.info(f"Successfully uploaded {file_path} to s3://{self.bucket_name}/{file_key}")
                return file_key
            else:
                # 保存到本地
                local_path = os.path.join(self.local_storage_dir, file_key)
                shutil.copy2(file_path, local_path)
                logger.info(f"Successfully saved {file_path} to {local_path}")
                return local_path
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            raise
        finally:
            # 清理临时文件
            if os.path.exists(file_path) and file_path != os.path.join(self.local_storage_dir, file_key):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {file_path}: {e}")
    
    def generate_presigned_url(self, file_key: str, expiration: int = 3600) -> str:
        """生成预签名URL（仅S3模式）"""
        if not self.use_s3:
            return f"file://{file_key}"  # 本地文件路径
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {file_key}: {e}")
            raise


# 为了向后兼容，保留S3Manager类
class S3Manager(FileManager):
    """S3管理器（向后兼容）"""
    
    def __init__(self):
        super().__init__(use_s3=True)
    
    def upload_file(self, file_path: str, s3_key: Optional[str] = None) -> str:
        """上传文件到S3，返回S3 key"""
        return self.save_file(file_path, s3_key)


def ensure_output_dir(output_dir: str = "/tmp/outputs") -> str:
    """确保输出目录存在"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_temp_filepath(output_dir: str, extension: str) -> str:
    """生成临时文件路径"""
    return os.path.join(output_dir, f"{uuid.uuid4()}{extension}")