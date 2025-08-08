from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from .base import AudioGenerationBase, validate_text_input, validate_prompt_content


class GenerateFromDescriptionRequest(AudioGenerationBase):
    """Request for generating music from description"""
    full_described_song: str = Field(..., max_length=1000)
    
    @field_validator('full_described_song')
    @classmethod
    def validate_description(cls, v):
        return validate_text_input(v, min_length=5, max_length=1000)


class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    """Request for generating music with custom lyrics"""
    prompt: str = Field(..., max_length=500)
    lyrics: str = Field(..., max_length=2000)
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        return validate_text_input(v, min_length=5, max_length=500)
    
    @field_validator('lyrics')
    @classmethod
    def validate_lyrics(cls, v):
        return validate_text_input(v, min_length=10, max_length=2000)


class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    """Request for generating music with described lyrics"""
    prompt: str = Field(..., max_length=500)
    described_lyrics: str = Field(..., max_length=1000)
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        return validate_text_input(v, min_length=5, max_length=500)
    
    @field_validator('described_lyrics')
    @classmethod
    def validate_described_lyrics(cls, v):
        return validate_text_input(v, min_length=5, max_length=1000)


class LyricsGenerationRequest(BaseModel):
    """Request for lyrics generation"""
    description: str = Field(..., max_length=1000)
    style: Optional[str] = Field(default=None, max_length=100)
    language: Optional[str] = Field(default="english", max_length=50)
    mood: Optional[str] = Field(default=None, max_length=100)
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        return validate_text_input(v, min_length=5, max_length=1000)


class PromptGenerationRequest(BaseModel):
    """Request for prompt generation"""
    description: str = Field(..., max_length=1000)
    genre: Optional[str] = Field(default=None, max_length=100)
    instruments: Optional[List[str]] = Field(default=None)
    tempo: Optional[str] = Field(default=None, max_length=50)
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        return validate_text_input(v, min_length=5, max_length=1000)


class CoverImageGenerationRequest(BaseModel):
    """Request for cover image generation"""
    prompt: str = Field(..., min_length=5, max_length=500, description="Description for cover image generation")
    style: Optional[str] = Field(default="album cover art", max_length=100, description="Style preset or custom style")
    width: Optional[int] = Field(default=512, ge=256, le=1024, description="Image width (must be multiple of 64)")
    height: Optional[int] = Field(default=512, ge=256, le=1024, description="Image height (must be multiple of 64)")
    num_inference_steps: Optional[int] = Field(default=4, ge=1, le=10, description="Number of inference steps (1-10 for SDXL-Turbo)")
    seed: Optional[int] = Field(default=None, ge=0, le=2**32-1, description="Random seed for reproducible generation")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        cleaned = validate_text_input(v, min_length=5, max_length=500)
        return validate_prompt_content(cleaned)
    
    @field_validator('style')
    @classmethod
    def validate_style(cls, v):
        if v is not None:
            cleaned = validate_text_input(v, min_length=1, max_length=100)
            return cleaned if cleaned else "album cover art"
        return "album cover art"
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v):
        # Ensure dimensions are multiples of 64 for optimal generation
        if v % 64 != 0:
            v = round(v / 64) * 64
        return max(256, min(1024, v))


class CategoryGenerationRequest(BaseModel):
    """Request for category generation"""
    description: str = Field(..., max_length=1000)
    max_categories: int = Field(default=5, ge=1, le=10)
    include_subgenres: bool = Field(default=True)
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        return validate_text_input(v, min_length=5, max_length=1000)