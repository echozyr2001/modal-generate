from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # 模型配置
    music_model_checkpoint_dir: str = "/models"
    llm_model_id: str = "Qwen/Qwen2-7B-Instruct"
    image_model_id: str = "stabilityai/sdxl-turbo"
    
    # 生成参数默认值
    default_audio_duration: float = 180.0
    default_guidance_scale: float = 15.0
    default_infer_steps: int = 60
    
    # S3配置
    s3_bucket_name: str = os.environ.get("S3_BUCKET_NAME", "")
    s3_region: Optional[str] = None
    
    # 存储配置
    use_s3_storage: bool = os.environ.get("USE_S3_STORAGE", "false").lower() == "true"
    local_storage_dir: str = os.environ.get("LOCAL_STORAGE_DIR", "./outputs")
    
    # 限制配置
    max_audio_duration: float = 300.0
    max_prompt_length: int = 1000
    max_lyrics_length: int = 2000
    
    # Modal配置
    gpu_type: str = "L40S"
    scaledown_window: int = 15
    
    # 缓存配置
    hf_cache_dir: str = "/.cache/huggingface"
    
    class Config:
        env_file = ".env"


settings = Settings()