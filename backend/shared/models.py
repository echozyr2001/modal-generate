from pydantic import BaseModel, Field, validator
from typing import List, Optional
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