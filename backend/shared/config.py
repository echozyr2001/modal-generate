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
    default_inference_steps: int = 60  # Renamed from default_infer_steps
    
    # Storage configuration
    use_s3_storage: bool = os.environ.get("USE_S3_STORAGE", "false").lower() == "true"
    local_storage_dir: str = os.environ.get("LOCAL_STORAGE_DIR", "./outputs")
    s3_bucket_name: str = os.environ.get("S3_BUCKET_NAME", "")
    s3_region: Optional[str] = os.environ.get("S3_REGION")
    
    # Resource limits and constraints
    max_audio_duration: float = 300.0
    max_prompt_length: int = 1000
    max_lyrics_length: int = 2000
    max_concurrent_operations: int = 50
    default_timeout_seconds: int = 600
    
    # Cost monitoring
    enable_cost_monitoring: bool = os.environ.get("ENABLE_COST_MONITORING", "true").lower() == "true"
    cost_log_file: str = os.environ.get("COST_LOG_FILE", "./cost_log.json")
    cost_alert_threshold: float = 10.0
    
    # Cache configuration
    hf_cache_dir: str = "/.cache/huggingface"
    
    # Service URLs (for microservice orchestration)
    lyrics_service_url: str = os.environ.get("LYRICS_SERVICE_URL", "")
    music_service_url: str = os.environ.get("MUSIC_SERVICE_URL", "")
    image_service_url: str = os.environ.get("IMAGE_SERVICE_URL", "")
    
    class Config:
        env_file = ".env"
    
    def get_storage_config(self) -> dict:
        """Get current storage configuration"""
        return {
            "use_s3_storage": self.use_s3_storage,
            "s3_bucket_name": self.s3_bucket_name if self.use_s3_storage else None,
            "local_storage_dir": self.local_storage_dir if not self.use_s3_storage else None,
            "storage_mode": "s3" if self.use_s3_storage else "local"
        }
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service"""
        service_configs = {
            "lyrics": {
                "gpu_type": "T4",
                "scaledown_window": 30,
                "max_runtime_seconds": 60,
                "max_concurrent_requests": 20,
                "cost_per_hour": 0.35,
                "memory_gb": 16
            },
            "music": {
                "gpu_type": "L40S",
                "scaledown_window": 60,
                "max_runtime_seconds": 600,
                "max_concurrent_requests": 5,
                "cost_per_hour": 1.20,
                "memory_gb": 32
            },
            "image": {
                "gpu_type": "L4",
                "scaledown_window": 45,
                "max_runtime_seconds": 120,
                "max_concurrent_requests": 10,
                "cost_per_hour": 0.60,
                "memory_gb": 16
            },
            "integration": {
                "gpu_type": "CPU",
                "scaledown_window": 15,
                "max_runtime_seconds": 900,
                "max_concurrent_requests": 30,
                "cost_per_hour": 0.05,
                "memory_gb": 8
            }
        }
        return service_configs.get(service_name, service_configs["integration"])


settings = Settings()