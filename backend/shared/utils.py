import boto3
import os
import uuid
import logging
import shutil
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi import HTTPException
from fastapi.responses import FileResponse
from shared.config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """Enhanced file manager supporting unified storage handling (local/S3)"""
    
    def __init__(self, use_s3: bool = None, local_storage_dir: str = None):
        # Use settings defaults if not specified
        self.use_s3 = use_s3 if use_s3 is not None else settings.use_s3_storage
        self.local_storage_dir = local_storage_dir or settings.local_storage_dir
        
        if self.use_s3:
            self.s3_client = boto3.client("s3")
            self.bucket_name = settings.s3_bucket_name
            if not self.bucket_name:
                raise ValueError("S3 bucket name must be configured when using S3 storage")
        else:
            # Ensure local storage directory exists with proper structure
            self._setup_local_storage()
            logger.info(f"Using local storage: {self.local_storage_dir}")
        
        # Initialize metadata tracking
        self.metadata_file = os.path.join(self.local_storage_dir, "metadata.json")
        self._load_metadata()
    
    def _setup_local_storage(self):
        """Setup structured local storage directories"""
        subdirs = ['audio', 'images', 'temp']
        for subdir in subdirs:
            dir_path = os.path.join(self.local_storage_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
    
    def _load_metadata(self):
        """Load file metadata from disk"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self):
        """Save file metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _add_file_metadata(self, key: str, original_path: str, file_type: str, file_size: int):
        """Add metadata for a file"""
        self.metadata[key] = {
            "original_path": original_path,
            "file_type": file_type,
            "file_size": file_size,
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }
        self._save_metadata()
    
    def _update_access_metadata(self, key: str):
        """Update access metadata for a file"""
        if key in self.metadata:
            self.metadata[key]["access_count"] += 1
            self.metadata[key]["last_accessed"] = datetime.now().isoformat()
            self._save_metadata()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def save_file(self, file_path: str, file_key: Optional[str] = None, file_type: str = "temp") -> str:
        """Save file and return key/path
        
        Args:
            file_path: Path to the file to save
            file_key: Optional custom key/filename
            file_type: Type of file for organization (audio, images, temp)
        
        Returns:
            S3 key or local file path
        """
        if file_key is None:
            file_extension = os.path.splitext(file_path)[1]
            file_key = f"{file_type}/{uuid.uuid4()}{file_extension}"
        
        # Get file size for metadata
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        try:
            if self.use_s3:
                # Upload to S3
                self.s3_client.upload_file(file_path, self.bucket_name, file_key)
                logger.info(f"Successfully uploaded {file_path} to s3://{self.bucket_name}/{file_key}")
                result_key = file_key
            else:
                # Save to local with organized structure
                local_path = os.path.join(self.local_storage_dir, file_key)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.copy2(file_path, local_path)
                logger.info(f"Successfully saved {file_path} to {local_path}")
                result_key = local_path
            
            # Add metadata tracking for local storage
            if not self.use_s3:
                self._add_file_metadata(result_key, file_path, file_type, file_size)
            
            return result_key
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            raise
        finally:
            # Clean up temporary file if it's different from destination
            if (os.path.exists(file_path) and 
                file_path != os.path.join(self.local_storage_dir, file_key)):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {file_path}: {e}")
    
    def get_file_url(self, key: str, expiration: int = 3600) -> str:
        """Generate accessible file URL
        
        Args:
            key: File key (S3 key or local path)
            expiration: URL expiration time in seconds (S3 only)
            
        Returns:
            Accessible URL or file path
        """
        # Update access metadata for local storage
        if not self.use_s3:
            self._update_access_metadata(key)
            # For local storage, return the file path
            if os.path.isabs(key):
                return key
            return os.path.join(self.local_storage_dir, key)
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {key}: {e}")
            raise
    
    def delete_file(self, key: str) -> bool:
        """Delete file from storage
        
        Args:
            key: File key (S3 key or local path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.use_s3:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                logger.info(f"Deleted S3 object: {key}")
            else:
                file_path = key if os.path.isabs(key) else os.path.join(self.local_storage_dir, key)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted local file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {key}: {e}")
            return False
    
    def file_exists(self, key: str) -> bool:
        """Check if file exists in storage"""
        try:
            if self.use_s3:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
                return True
            else:
                file_path = key if os.path.isabs(key) else os.path.join(self.local_storage_dir, key)
                return os.path.exists(file_path)
        except Exception:
            return False
    
    def get_file_size(self, key: str) -> Optional[int]:
        """Get file size in bytes"""
        try:
            if self.use_s3:
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
                return response['ContentLength']
            else:
                file_path = key if os.path.isabs(key) else os.path.join(self.local_storage_dir, key)
                return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Failed to get file size for {key}: {e}")
            return None
    
    def serve_file(self, key: str) -> FileResponse:
        """Serve file for local storage mode
        
        Args:
            key: File key (local path)
            
        Returns:
            FastAPI FileResponse
        """
        if self.use_s3:
            raise HTTPException(
                status_code=400, 
                detail="File serving is only available for local storage mode"
            )
        
        file_path = key if os.path.isabs(key) else os.path.join(self.local_storage_dir, key)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Update access metadata
        self._update_access_metadata(key)
        
        # Determine media type based on file extension
        media_type = self._get_media_type(file_path)
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=os.path.basename(file_path)
        )
    
    def _get_media_type(self, file_path: str) -> str:
        """Get media type based on file extension"""
        extension = os.path.splitext(file_path)[1].lower()
        media_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.txt': 'text/plain',
            '.json': 'application/json'
        }
        return media_types.get(extension, 'application/octet-stream')
    
    def get_file_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a file"""
        return self.metadata.get(key)
    
    def list_files(self, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files with metadata
        
        Args:
            file_type: Optional filter by file type (audio, images, temp)
            
        Returns:
            List of file metadata dictionaries
        """
        files = []
        for key, metadata in self.metadata.items():
            if file_type is None or metadata.get("file_type") == file_type:
                files.append({
                    "key": key,
                    **metadata
                })
        return files
    
    def cleanup_old_files(self, max_age_days: int = 7) -> int:
        """Clean up old files based on age
        
        Args:
            max_age_days: Maximum age in days before cleanup
            
        Returns:
            Number of files cleaned up
        """
        if self.use_s3:
            logger.warning("Cleanup not implemented for S3 storage")
            return 0
        
        cleaned_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        files_to_remove = []
        for key, metadata in self.metadata.items():
            try:
                created_at = datetime.fromisoformat(metadata["created_at"]).timestamp()
                if created_at < cutoff_time:
                    files_to_remove.append(key)
            except Exception as e:
                logger.warning(f"Failed to parse creation time for {key}: {e}")
        
        for key in files_to_remove:
            if self.delete_file(key):
                cleaned_count += 1
                self.metadata.pop(key, None)
        
        if cleaned_count > 0:
            self._save_metadata()
            logger.info(f"Cleaned up {cleaned_count} old files")
        
        return cleaned_count
    
    def cleanup_unused_files(self, min_access_count: int = 0) -> int:
        """Clean up files that haven't been accessed
        
        Args:
            min_access_count: Minimum access count to keep file
            
        Returns:
            Number of files cleaned up
        """
        if self.use_s3:
            logger.warning("Cleanup not implemented for S3 storage")
            return 0
        
        cleaned_count = 0
        files_to_remove = []
        
        for key, metadata in self.metadata.items():
            if metadata.get("access_count", 0) <= min_access_count:
                files_to_remove.append(key)
        
        for key in files_to_remove:
            if self.delete_file(key):
                cleaned_count += 1
                self.metadata.pop(key, None)
        
        if cleaned_count > 0:
            self._save_metadata()
            logger.info(f"Cleaned up {cleaned_count} unused files")
        
        return cleaned_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "storage_mode": "S3" if self.use_s3 else "local",
            "total_files": len(self.metadata),
            "file_types": {},
            "total_size_bytes": 0,
            "total_access_count": 0
        }
        
        for metadata in self.metadata.values():
            file_type = metadata.get("file_type", "unknown")
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            stats["total_size_bytes"] += metadata.get("file_size", 0)
            stats["total_access_count"] += metadata.get("access_count", 0)
        
        return stats


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


class CostMonitor:
    """Cost monitoring and tracking utilities"""
    
    def __init__(self):
        self.usage_log: Dict[str, Any] = {}
        self.cost_rates = {
            # GPU cost per hour (approximate)
            "T4": 0.35,
            "L4": 0.60,
            "L40S": 1.20,
            "A100": 2.50,
            "CPU": 0.05
        }
    
    def start_operation(self, operation_id: str, gpu_type: str, service_name: str) -> Dict[str, Any]:
        """Start tracking an operation"""
        operation_data = {
            "operation_id": operation_id,
            "service_name": service_name,
            "gpu_type": gpu_type,
            "start_time": time.time(),
            "cost_rate": self.cost_rates.get(gpu_type, 0.05)
        }
        self.usage_log[operation_id] = operation_data
        logger.info(f"Started operation {operation_id} on {gpu_type} for {service_name}")
        return operation_data
    
    def end_operation(self, operation_id: str) -> Dict[str, Any]:
        """End tracking an operation and calculate cost"""
        if operation_id not in self.usage_log:
            logger.warning(f"Operation {operation_id} not found in usage log")
            return {}
        
        operation_data = self.usage_log[operation_id]
        end_time = time.time()
        duration_hours = (end_time - operation_data["start_time"]) / 3600
        estimated_cost = duration_hours * operation_data["cost_rate"]
        
        operation_data.update({
            "end_time": end_time,
            "duration_seconds": end_time - operation_data["start_time"],
            "duration_hours": duration_hours,
            "estimated_cost": estimated_cost
        })
        
        logger.info(f"Operation {operation_id} completed in {duration_hours:.4f}h, "
                   f"estimated cost: ${estimated_cost:.4f}")
        return operation_data
    
    def get_operation_cost(self, operation_id: str) -> float:
        """Get estimated cost for an operation"""
        if operation_id not in self.usage_log:
            return 0.0
        
        operation_data = self.usage_log[operation_id]
        if "estimated_cost" in operation_data:
            return operation_data["estimated_cost"]
        
        # Calculate current cost if operation is still running
        current_time = time.time()
        duration_hours = (current_time - operation_data["start_time"]) / 3600
        return duration_hours * operation_data["cost_rate"]
    
    def get_total_cost(self, service_name: Optional[str] = None) -> float:
        """Get total estimated cost, optionally filtered by service"""
        total_cost = 0.0
        for operation_data in self.usage_log.values():
            if service_name and operation_data.get("service_name") != service_name:
                continue
            if "estimated_cost" in operation_data:
                total_cost += operation_data["estimated_cost"]
        return total_cost


class TimeoutManager:
    """Timeout management utilities for service operations"""
    
    def __init__(self, default_timeout: int = 600):
        self.default_timeout = default_timeout
        self.active_operations: Dict[str, Dict[str, Any]] = {}
    
    def start_timeout(self, operation_id: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Start timeout tracking for an operation"""
        timeout = timeout_seconds or self.default_timeout
        operation_data = {
            "operation_id": operation_id,
            "start_time": time.time(),
            "timeout_seconds": timeout,
            "deadline": time.time() + timeout
        }
        self.active_operations[operation_id] = operation_data
        logger.info(f"Started timeout tracking for {operation_id} with {timeout}s limit")
        return operation_data
    
    def check_timeout(self, operation_id: str) -> bool:
        """Check if operation has timed out"""
        if operation_id not in self.active_operations:
            return False
        
        operation_data = self.active_operations[operation_id]
        current_time = time.time()
        
        if current_time > operation_data["deadline"]:
            logger.warning(f"Operation {operation_id} has timed out after "
                          f"{operation_data['timeout_seconds']}s")
            return True
        return False
    
    def get_remaining_time(self, operation_id: str) -> float:
        """Get remaining time before timeout"""
        if operation_id not in self.active_operations:
            return 0.0
        
        operation_data = self.active_operations[operation_id]
        remaining = operation_data["deadline"] - time.time()
        return max(0.0, remaining)
    
    def end_timeout(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """End timeout tracking and return operation data"""
        if operation_id not in self.active_operations:
            return None
        
        operation_data = self.active_operations.pop(operation_id)
        operation_data["end_time"] = time.time()
        operation_data["actual_duration"] = operation_data["end_time"] - operation_data["start_time"]
        
        logger.info(f"Ended timeout tracking for {operation_id}, "
                   f"duration: {operation_data['actual_duration']:.2f}s")
        return operation_data
    
    def cleanup_expired(self):
        """Clean up expired timeout entries"""
        current_time = time.time()
        expired_ops = [
            op_id for op_id, data in self.active_operations.items()
            if current_time > data["deadline"]
        ]
        
        for op_id in expired_ops:
            logger.warning(f"Cleaning up expired operation: {op_id}")
            self.active_operations.pop(op_id, None)


# Global instances for shared use
cost_monitor = CostMonitor()
timeout_manager = TimeoutManager()


def create_service_config(service_name: str, config_dict: Dict[str, Any]) -> 'ServiceConfig':
    """Create ServiceConfig from settings dictionary"""
    from .models import ServiceConfig, GPUType
    
    return ServiceConfig(
        service_name=service_name,
        gpu_type=GPUType(config_dict.get("gpu_type", "CPU")),
        scaledown_window=config_dict.get("scaledown_window", 60),
        max_runtime_seconds=config_dict.get("max_runtime_seconds", 600),
        max_concurrent_requests=config_dict.get("max_concurrent_requests", 10)
    )


def get_service_configs() -> Dict[str, 'ServiceConfig']:
    """Get all service configurations from settings"""
    from .config import settings
    
    return {
        "lyrics": create_service_config("lyrics", settings.lyrics_service_config),
        "music": create_service_config("music", settings.music_service_config),
        "image": create_service_config("image", settings.image_service_config),
        "integration": create_service_config("integration", settings.integration_service_config)
    }