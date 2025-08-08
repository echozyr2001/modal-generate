from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .base import GenerationMetadata


class GenerateMusicResponse(BaseModel):
    """Unified response model for music generation"""
    audio_file_path: str  # S3 key or local path
    cover_image_file_path: Optional[str] = None  # S3 key or local path
    categories: List[str] = []
    storage_mode: str = "local"  # "s3" or "local"
    generation_metadata: Optional[GenerationMetadata] = None
    service_status: Optional[Dict[str, bool]] = None  # Track which services succeeded
    errors: Optional[List[str]] = None  # Track any errors that occurred
    correlation_id: Optional[str] = None


class LyricsGenerationResponse(BaseModel):
    """Response for lyrics generation"""
    lyrics: str
    word_count: Optional[int] = None
    structure_tags: Optional[List[str]] = None
    metadata: Optional[GenerationMetadata] = None


class PromptGenerationResponse(BaseModel):
    """Response for prompt generation"""
    prompt: str
    tag_count: Optional[int] = None
    detected_genre: Optional[str] = None
    metadata: Optional[GenerationMetadata] = None


class CoverImageGenerationResponse(BaseModel):
    """Response for cover image generation"""
    file_path: str  # S3 key or local path
    image_dimensions: Optional[tuple[int, int]] = None
    file_size_mb: Optional[float] = None
    metadata: Optional[GenerationMetadata] = None


class CategoryGenerationResponse(BaseModel):
    """Response for category generation"""
    categories: List[str]
    primary_genre: Optional[str] = None
    confidence_scores: Optional[Dict[str, float]] = None
    metadata: Optional[GenerationMetadata] = None


# Legacy response models for backward compatibility
class GenerateMusicResponseS3(BaseModel):
    """Legacy S3 response model - deprecated, use GenerateMusicResponse"""
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]


class GenerateMusicResponseLocal(BaseModel):
    """Legacy local response model - deprecated, use GenerateMusicResponse"""
    audio_data: str