from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum
import re


class GPUType(str, Enum):
    """Supported GPU types"""
    CPU = "CPU"
    T4 = "T4"
    L4 = "L4"
    L40S = "L40S"
    A100 = "A100"


class ServiceConfig(BaseModel):
    """Service configuration with necessary attributes"""
    service_name: str
    gpu_type: GPUType = GPUType.CPU
    scaledown_window: int = 60
    max_runtime_seconds: int = 600
    max_concurrent_requests: int = 10
    cost_per_hour: float = 0.05
    memory_gb: Optional[int] = None


class GenerationMetadata(BaseModel):
    """Metadata for generation responses"""
    generation_time: float
    model_info: str
    gpu_type: str
    estimated_cost: Optional[float] = None
    operation_id: str


class AudioGenerationBase(BaseModel):
    """Base model for audio generation requests"""
    audio_duration: float = Field(default=180.0, ge=10.0, le=300.0)
    seed: int = Field(default=-1, ge=-1, le=2**31-1)
    guidance_scale: float = Field(default=15.0, ge=1.0, le=30.0)
    inference_steps: int = Field(default=60, ge=10, le=100)  # Renamed from infer_step
    instrumental: bool = False


def validate_text_input(text: str, min_length: int = 5, max_length: int = 1000) -> str:
    """Common text validation function"""
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")
    
    # Clean text by removing potentially harmful characters
    cleaned = re.sub(r'[<>"\']', '', text.strip())
    
    if len(cleaned) < min_length:
        raise ValueError(f"Text too short after cleaning: {len(cleaned)} < {min_length}")
    
    if len(cleaned) > max_length:
        raise ValueError(f"Text too long: {len(cleaned)} > {max_length}")
    
    return cleaned


def validate_prompt_content(text: str) -> str:
    """Validate prompt content for inappropriate material"""
    inappropriate_keywords = ['nsfw', 'explicit', 'nude', 'sexual']
    if any(keyword in text.lower() for keyword in inappropriate_keywords):
        raise ValueError("Prompt contains inappropriate content")
    return text