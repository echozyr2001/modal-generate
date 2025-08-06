"""
Factory for creating Modal service configurations with optimized images and settings.

This module combines the Modal images and service configurations to create
complete service definitions ready for deployment.
"""

from typing import Dict, Any, Optional
import modal
from .modal_images import (
    text_generation_image,
    music_generation_image, 
    image_generation_image,
    integration_service_image,
    model_volumes,
    secrets
)
from .service_config import (
    ServiceConfig,
    ServiceType,
    GPUType,
    TextGenerationConfig,
    MusicGenerationConfig,
    ImageGenerationConfig,
    IntegrationConfig
)


class ModalServiceFactory:
    """Factory for creating Modal service configurations."""
    
    # Mapping of service types to their optimized images
    _service_images = {
        ServiceType.TEXT_GENERATION: text_generation_image,
        ServiceType.MUSIC_GENERATION: music_generation_image,
        ServiceType.IMAGE_GENERATION: image_generation_image,
        ServiceType.INTEGRATION: integration_service_image
    }
    
    # Volume mappings for different service types
    _service_volumes = {
        ServiceType.TEXT_GENERATION: {
            "/.cache/huggingface": model_volumes["huggingface_cache"]
        },
        ServiceType.MUSIC_GENERATION: {
            "/models": model_volumes["ace_step_models"],
            "/.cache/huggingface": model_volumes["huggingface_cache"]
        },
        ServiceType.IMAGE_GENERATION: {
            "/.cache/huggingface": model_volumes["huggingface_cache"],
            "/.cache/diffusers": model_volumes["diffusers_cache"]
        },
        ServiceType.INTEGRATION: {}  # No volumes needed for orchestration
    }
    
    # Secret mappings for different service types
    _service_secrets = {
        ServiceType.TEXT_GENERATION: [secrets["music_gen"]],
        ServiceType.MUSIC_GENERATION: [secrets["music_gen"], secrets["aws_credentials"]],
        ServiceType.IMAGE_GENERATION: [secrets["music_gen"], secrets["aws_credentials"]],
        ServiceType.INTEGRATION: [secrets["music_gen"], secrets["aws_credentials"]]
    }
    
    @classmethod
    def create_modal_config(cls, config: ServiceConfig) -> Dict[str, Any]:
        """Create Modal configuration dictionary from service config."""
        modal_config = {
            "image": cls._service_images[config.service_type],
            "volumes": cls._service_volumes[config.service_type],
            "secrets": cls._service_secrets[config.service_type],
            "scaledown_window": config.scaledown_window_seconds
        }
        
        # Add GPU configuration if not CPU-only
        if config.gpu_type != GPUType.CPU_ONLY:
            modal_config["gpu"] = config.gpu_type.value
        
        # Add memory limits if specified
        if config.resource_limits.max_memory_gb:
            modal_config["memory"] = int(config.resource_limits.max_memory_gb * 1024)  # Convert to MB
        
        # Add timeout configuration
        modal_config["timeout"] = config.resource_limits.timeout_seconds
        
        return modal_config
    
    @classmethod
    def create_app_config(cls, service_name: str, config: ServiceConfig) -> Dict[str, Any]:
        """Create complete Modal app configuration."""
        return {
            "name": service_name,
            "modal_config": cls.create_modal_config(config),
            "service_config": config
        }
    
    @classmethod
    def get_recommended_gpu(cls, service_type: ServiceType, workload_size: str = "medium") -> GPUType:
        """Get recommended GPU type for a service and workload size."""
        recommendations = {
            ServiceType.TEXT_GENERATION: {
                "small": GPUType.CPU_ONLY,
                "medium": GPUType.T4,
                "large": GPUType.T4
            },
            ServiceType.MUSIC_GENERATION: {
                "small": GPUType.L4,
                "medium": GPUType.L40S,
                "large": GPUType.A100
            },
            ServiceType.IMAGE_GENERATION: {
                "small": GPUType.T4,
                "medium": GPUType.L4,
                "large": GPUType.L40S
            },
            ServiceType.INTEGRATION: {
                "small": GPUType.CPU_ONLY,
                "medium": GPUType.CPU_ONLY,
                "large": GPUType.CPU_ONLY
            }
        }
        
        return recommendations[service_type][workload_size]
    
    @classmethod
    def estimate_cost(cls, config: ServiceConfig, runtime_minutes: float) -> Dict[str, float]:
        """Estimate cost for running a service configuration."""
        base_cost = config.cost_estimation.startup_cost_usd
        runtime_cost = config.cost_estimation.cost_per_minute_usd * runtime_minutes
        total_cost = base_cost + runtime_cost
        
        return {
            "startup_cost_usd": base_cost,
            "runtime_cost_usd": runtime_cost,
            "total_cost_usd": total_cost,
            "cost_per_minute_usd": config.cost_estimation.cost_per_minute_usd
        }


# Pre-configured service definitions for common use cases
PREDEFINED_SERVICES = {
    "lyrics-generator": ModalServiceFactory.create_app_config(
        "lyrics-generator",
        TextGenerationConfig(gpu_type=GPUType.T4, scaledown_window_seconds=30)
    ),
    "music-generator-core": ModalServiceFactory.create_app_config(
        "music-generator-core", 
        MusicGenerationConfig(gpu_type=GPUType.L40S, scaledown_window_seconds=60)
    ),
    "cover-image-generator": ModalServiceFactory.create_app_config(
        "cover-image-generator",
        ImageGenerationConfig(gpu_type=GPUType.L4, scaledown_window_seconds=45)
    ),
    "music-generator-integrated": ModalServiceFactory.create_app_config(
        "music-generator-integrated",
        IntegrationConfig(scaledown_window_seconds=15)
    )
}