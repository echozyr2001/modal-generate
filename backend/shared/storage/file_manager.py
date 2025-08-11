import os
import uuid
import shutil
import logging
import base64
from typing import Optional, Dict, Any
from shared.config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """Unified file manager for S3, local, and direct download storage"""
    
    def __init__(self, use_s3: bool = None, local_storage_dir: str = None, storage_mode: str = None):
        self.storage_mode = storage_mode or settings.storage_mode
        self.use_s3 = use_s3 if use_s3 is not None else settings.use_s3_storage
        self.local_storage_dir = local_storage_dir or settings.local_storage_dir
        
        # Override use_s3 based on storage_mode
        if self.storage_mode == "direct_download":
            self.use_s3 = False
        elif self.storage_mode == "s3":
            self.use_s3 = True
        elif self.storage_mode == "local":
            self.use_s3 = False
        
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
        return self.storage_mode
    
    def read_file_as_base64(self, file_path: str) -> str:
        """Read file and return as base64 string"""
        with open(file_path, "rb") as f:
            file_data = f.read()
        return base64.b64encode(file_data).decode('utf-8')
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information including size, format, etc."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Determine file type
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            file_type = 'image'
        elif file_ext in ['.wav', '.mp3', '.flac', '.ogg']:
            file_type = 'audio'
        else:
            file_type = 'unknown'
        
        return {
            'file_name': file_name,
            'file_size': file_size,
            'file_type': file_type,
            'file_extension': file_ext,
            'file_path': file_path
        }
    
    def prepare_direct_download_response(self, file_path: str) -> Dict[str, Any]:
        """Prepare file for direct download - returns file data and metadata"""
        if self.storage_mode != "direct_download":
            raise ValueError("Direct download only available in direct_download mode")
        
        file_info = self.get_file_info(file_path)
        file_data = self.read_file_as_base64(file_path)
        
        # Clean up temporary file after reading
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {e}")
        
        return {
            'file_data': file_data,
            'file_name': file_info['file_name'],
            'file_size': file_info['file_size'],
            'file_type': file_info['file_type'],
            'file_extension': file_info['file_extension'],
            'encoding': 'base64'
        }