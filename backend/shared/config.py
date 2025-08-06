from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os


class Settings(BaseSettings):
    # Model configuration
    music_model_checkpoint_dir: str = "/models"
    llm_model_id: str = "Qwen/Qwen2-7B-Instruct"
    image_model_id: str = "stabilityai/sdxl-turbo"
    
    # Generation parameter defaults
    default_audio_duration: float = 180.0
    default_guidance_scale: float = 15.0
    default_infer_steps: int = 60
    
    # S3 configuration
    s3_bucket_name: str = os.environ.get("S3_BUCKET_NAME", "")
    s3_region: Optional[str] = None
    
    # Storage configuration
    use_s3_storage: bool = os.environ.get("USE_S3_STORAGE", "false").lower() == "true"
    local_storage_dir: str = os.environ.get("LOCAL_STORAGE_DIR", "./outputs")
    
    # Resource limits
    max_audio_duration: float = 300.0
    max_prompt_length: int = 1000
    max_lyrics_length: int = 2000
    max_concurrent_operations: int = 50
    default_timeout_seconds: int = 600
    cost_alert_threshold: float = 10.0
    
    # Service-specific configurations
    lyrics_service_config: Dict[str, Any] = {
        "gpu_type": "T4",
        "scaledown_window": 30,
        "max_runtime_seconds": 60,
        "max_concurrent_requests": 20
    }
    
    music_service_config: Dict[str, Any] = {
        "gpu_type": "L40S",
        "scaledown_window": 60,
        "max_runtime_seconds": 600,
        "max_concurrent_requests": 5
    }
    
    image_service_config: Dict[str, Any] = {
        "gpu_type": "L4",
        "scaledown_window": 45,
        "max_runtime_seconds": 120,
        "max_concurrent_requests": 10
    }
    
    integration_service_config: Dict[str, Any] = {
        "gpu_type": "CPU",
        "scaledown_window": 15,
        "max_runtime_seconds": 900,
        "max_concurrent_requests": 30
    }
    
    # Legacy Modal configuration (for backward compatibility)
    gpu_type: str = "L40S"
    scaledown_window: int = 15
    
    # Cache configuration
    hf_cache_dir: str = "/.cache/huggingface"
    
    # Service URLs (for orchestration)
    lyrics_service_url: str = os.environ.get("LYRICS_SERVICE_URL", "")
    music_service_url: str = os.environ.get("MUSIC_SERVICE_URL", "")
    image_service_url: str = os.environ.get("IMAGE_SERVICE_URL", "")
    
    # Cost monitoring
    enable_cost_monitoring: bool = os.environ.get("ENABLE_COST_MONITORING", "true").lower() == "true"
    cost_log_file: str = os.environ.get("COST_LOG_FILE", "./cost_log.json")
    
    class Config:
        env_file = ".env"


settings = Settings()