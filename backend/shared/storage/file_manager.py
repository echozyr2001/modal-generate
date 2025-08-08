import os
import uuid
import shutil
import logging
from typing import Optional
from shared.config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """Unified file manager for local and S3 storage"""
    
    def __init__(self, use_s3: bool = None, local_storage_dir: str = None):
        self.use_s3 = use_s3 if use_s3 is not None else settings.use_s3_storage
        self.local_storage_dir = local_storage_dir or settings.local_storage_dir
        
        if self.use_s3:
            from .s3_manager import S3Manager
            self.s3_manager = S3Manager()
        else:
            os.makedirs(self.local_storage_dir, exist_ok=True)
    
    def save_file(self, file_path: str, file_key: Optional[str] = None, file_type: str = "temp") -> str:
        """Save file and return key/path"""
        if file_key is None:
            file_extension = os.path.splitext(file_path)[1]
            file_key = f"{file_type}/{uuid.uuid4()}{file_extension}"
        
        if self.use_s3:
            return self.s3_manager.upload_file(file_path, file_key)
        else:
            local_path = os.path.join(self.local_storage_dir, file_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy2(file_path, local_path)
            return local_path
    
    def get_file_url(self, key: str, expiration: int = 3600) -> str:
        """Get file URL"""
        if self.use_s3:
            return self.s3_manager.get_presigned_url(key, expiration)
        else:
            return key if os.path.isabs(key) else os.path.join(self.local_storage_dir, key)
    
    def delete_file(self, key: str) -> bool:
        """Delete file"""
        try:
            if self.use_s3:
                return self.s3_manager.delete_file(key)
            else:
                file_path = key if os.path.isabs(key) else os.path.join(self.local_storage_dir, key)
                if os.path.exists(file_path):
                    os.remove(file_path)
                return True
        except Exception as e:
            logger.error(f"Failed to delete file {key}: {e}")
            return False
    
    def file_exists(self, key: str) -> bool:
        """Check if file exists"""
        try:
            if self.use_s3:
                return self.s3_manager.file_exists(key)
            else:
                file_path = key if os.path.isabs(key) else os.path.join(self.local_storage_dir, key)
                return os.path.exists(file_path)
        except Exception as e:
            logger.error(f"Failed to check file existence {key}: {e}")
            return False
    
    def get_storage_mode(self) -> str:
        """Get current storage mode"""
        return "s3" if self.use_s3 else "local"