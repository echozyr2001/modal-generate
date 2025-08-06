from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
import re


class AudioGenerationBase(BaseModel):
    audio_duration: float = Field(default=180.0, ge=10.0, le=300.0)
    seed: int = Field(default=-1, ge=-1, le=2**31-1)
    guidance_scale: float = Field(default=15.0, ge=1.0, le=30.0)
    infer_step: int = Field(default=60, ge=10, le=100)
    instrumental: bool = False


class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str = Field(..., max_length=1000)
    
    @validator('full_described_song')
    def validate_description(cls, v):
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if len(cleaned) < 5:
            raise ValueError("Description too short")
        return cleaned


class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str = Field(..., max_length=500)
    lyrics: str = Field(..., max_length=2000)


class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str = Field(..., max_length=500)
    described_lyrics: str = Field(..., max_length=1000)


class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]


class GenerateMusicResponse(BaseModel):
    audio_data: str


# 独立服务的请求/响应模型
class LyricsGenerationRequest(BaseModel):
    description: str = Field(..., max_length=1000)
    
    @validator('description')
    def validate_description(cls, v):
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if len(cleaned) < 5:
            raise ValueError("Description too short")
        return cleaned


class LyricsGenerationResponse(BaseModel):
    lyrics: str


class PromptGenerationRequest(BaseModel):
    description: str = Field(..., max_length=1000)
    
    @validator('description')
    def validate_description(cls, v):
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if len(cleaned) < 5:
            raise ValueError("Description too short")
        return cleaned


class PromptGenerationResponse(BaseModel):
    prompt: str


class CoverImageGenerationRequest(BaseModel):
    prompt: str = Field(..., max_length=500)
    style: Optional[str] = Field(default="album cover art", max_length=100)


class CoverImageGenerationResponse(BaseModel):
    s3_key: str


class CategoryGenerationRequest(BaseModel):
    description: str = Field(..., max_length=1000)


class CategoryGenerationResponse(BaseModel):
    categories: List[str]


# Resource Configuration Models
class GPUType(str, Enum):
    """Supported GPU types with their characteristics"""
    CPU = "CPU"
    T4 = "T4"
    L4 = "L4"
    L40S = "L40S"
    A100 = "A100"


class ServiceConfig(BaseModel):
    """Configuration for individual services"""
    service_name: str = Field(..., description="Name of the service")
    gpu_type: GPUType = Field(default=GPUType.CPU, description="GPU type required")
    scaledown_window: int = Field(default=60, ge=15, le=300, description="Scaledown window in seconds")
    max_runtime_seconds: int = Field(default=600, ge=60, le=3600, description="Maximum runtime in seconds")
    max_concurrent_requests: int = Field(default=10, ge=1, le=100, description="Maximum concurrent requests")
    cost_per_hour: float = Field(default=0.05, ge=0.0, description="Estimated cost per hour")
    memory_gb: Optional[int] = Field(default=None, description="Memory requirement in GB")
    
    def __init__(self, **data):
        # Set cost based on GPU type if not explicitly provided
        if 'cost_per_hour' not in data or data['cost_per_hour'] == 0.05:
            gpu_type = data.get('gpu_type', GPUType.CPU)
            gpu_costs = {
                GPUType.CPU: 0.05,
                GPUType.T4: 0.35,
                GPUType.L4: 0.60,
                GPUType.L40S: 1.20,
                GPUType.A100: 2.50
            }
            data['cost_per_hour'] = gpu_costs.get(gpu_type, 0.05)
        super().__init__(**data)


class ResourceLimits(BaseModel):
    """Global resource limits and constraints"""
    max_audio_duration: float = Field(default=300.0, ge=10.0, le=600.0)
    max_prompt_length: int = Field(default=1000, ge=10, le=2000)
    max_lyrics_length: int = Field(default=2000, ge=10, le=5000)
    max_concurrent_operations: int = Field(default=50, ge=1, le=200)
    default_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    cost_alert_threshold: float = Field(default=10.0, ge=0.0, description="Cost threshold for alerts")


class ServiceMetadata(BaseModel):
    """Metadata for service operations"""
    operation_id: str
    service_name: str
    gpu_type: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    estimated_cost: Optional[float] = None
    status: str = "running"
    error_message: Optional[str] = None


class GenerationMetadata(BaseModel):
    """Metadata included in generation responses"""
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    model_info: str = Field(..., description="Model used for generation")
    gpu_type: str = Field(..., description="GPU type used")
    estimated_cost: Optional[float] = Field(default=None, description="Estimated cost in USD")
    operation_id: str = Field(..., description="Unique operation identifier")


# Enhanced Response Models with Metadata
class LyricsGenerationResponseEnhanced(BaseModel):
    lyrics: str
    metadata: GenerationMetadata


class PromptGenerationResponseEnhanced(BaseModel):
    prompt: str
    metadata: GenerationMetadata


class CategoryGenerationResponseEnhanced(BaseModel):
    categories: List[str]
    metadata: GenerationMetadata


class CoverImageGenerationResponseEnhanced(BaseModel):
    file_path: str  # S3 key or local path
    image_dimensions: Optional[tuple[int, int]] = None
    file_size_mb: Optional[float] = None
    metadata: GenerationMetadata


class MusicGenerationResponseEnhanced(BaseModel):
    file_path: str  # S3 key or local path
    audio_duration: float
    file_size_mb: Optional[float] = None
    metadata: GenerationMetadata


# Service Base Response Model
class ServiceResponse(BaseModel):
    """Base response model for all services"""
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[GenerationMetadata] = None
    
    @classmethod
    def success_response(cls, data: Dict[str, Any], metadata: Optional[GenerationMetadata] = None):
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_response(cls, error: str, metadata: Optional[GenerationMetadata] = None):
        return cls(success=False, error=error, metadata=metadata)