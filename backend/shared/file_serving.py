"""
File serving utilities for local storage mode
"""

import os
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .utils import FileManager
from .config import settings


class FileMetadata(BaseModel):
    """File metadata response model"""
    key: str
    file_type: str
    file_size: int
    created_at: str
    access_count: int
    last_accessed: Optional[str] = None


class StorageStats(BaseModel):
    """Storage statistics response model"""
    storage_mode: str
    total_files: int
    file_types: Dict[str, int]
    total_size_bytes: int
    total_access_count: int


class CleanupResult(BaseModel):
    """Cleanup operation result"""
    files_cleaned: int
    message: str


def create_file_serving_router(file_manager: Optional[FileManager] = None) -> APIRouter:
    """Create a file serving router for local storage
    
    Args:
        file_manager: Optional FileManager instance, creates default if None
        
    Returns:
        FastAPI router with file serving endpoints
    """
    router = APIRouter(prefix="/files", tags=["File Serving"])
    
    # Use provided file_manager or create default
    if file_manager is None:
        file_manager = FileManager()
    
    @router.get("/serve/{file_key:path}")
    async def serve_file(file_key: str):
        """Serve a file from local storage
        
        Args:
            file_key: File key or path to serve
            
        Returns:
            File content as FileResponse
        """
        try:
            return file_manager.serve_file(file_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")
    
    @router.get("/metadata/{file_key:path}", response_model=FileMetadata)
    async def get_file_metadata(file_key: str):
        """Get metadata for a specific file
        
        Args:
            file_key: File key to get metadata for
            
        Returns:
            File metadata
        """
        metadata = file_manager.get_file_metadata(file_key)
        if metadata is None:
            raise HTTPException(status_code=404, detail="File metadata not found")
        
        return FileMetadata(key=file_key, **metadata)
    
    @router.get("/list", response_model=List[FileMetadata])
    async def list_files(
        file_type: Optional[str] = Query(None, description="Filter by file type (audio, images, temp)")
    ):
        """List all files with metadata
        
        Args:
            file_type: Optional filter by file type
            
        Returns:
            List of file metadata
        """
        files = file_manager.list_files(file_type)
        return [FileMetadata(**file_data) for file_data in files]
    
    @router.get("/stats", response_model=StorageStats)
    async def get_storage_stats():
        """Get storage statistics
        
        Returns:
            Storage statistics including file counts and sizes
        """
        stats = file_manager.get_storage_stats()
        return StorageStats(**stats)
    
    @router.delete("/cleanup/old")
    async def cleanup_old_files(
        max_age_days: int = Query(7, description="Maximum age in days before cleanup")
    ) -> CleanupResult:
        """Clean up old files
        
        Args:
            max_age_days: Maximum age in days before cleanup
            
        Returns:
            Cleanup result with count of files cleaned
        """
        if file_manager.use_s3:
            raise HTTPException(
                status_code=400, 
                detail="Cleanup is only available for local storage mode"
            )
        
        cleaned_count = file_manager.cleanup_old_files(max_age_days)
        return CleanupResult(
            files_cleaned=cleaned_count,
            message=f"Cleaned up {cleaned_count} files older than {max_age_days} days"
        )
    
    @router.delete("/cleanup/unused")
    async def cleanup_unused_files(
        min_access_count: int = Query(0, description="Minimum access count to keep file")
    ) -> CleanupResult:
        """Clean up unused files
        
        Args:
            min_access_count: Minimum access count to keep file
            
        Returns:
            Cleanup result with count of files cleaned
        """
        if file_manager.use_s3:
            raise HTTPException(
                status_code=400, 
                detail="Cleanup is only available for local storage mode"
            )
        
        cleaned_count = file_manager.cleanup_unused_files(min_access_count)
        return CleanupResult(
            files_cleaned=cleaned_count,
            message=f"Cleaned up {cleaned_count} files with access count <= {min_access_count}"
        )
    
    @router.delete("/{file_key:path}")
    async def delete_file(file_key: str):
        """Delete a specific file
        
        Args:
            file_key: File key to delete
            
        Returns:
            Success message
        """
        success = file_manager.delete_file(file_key)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete file")
        
        return {"message": f"File {file_key} deleted successfully"}
    
    return router


# Global file serving router using default settings
default_file_serving_router = create_file_serving_router()