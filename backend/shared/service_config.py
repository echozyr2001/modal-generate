"""
Service-specific configuration classes for modular AI services.

This module defines configuration classes with GPU types, scaledown windows,
resource limits, and cost estimation as specified in requirements 2.4, 5.1, 5.2, and 5.4.
"""

from enum import Enum
from typing import Dict, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class GPUType(str, Enum):
    """Supported GPU types with their characteristics."""
    CPU_ONLY = "cpu"
    T4 = "T4"
    L4 = "L4" 
    L40S = "L40S"
    A100 = "A100"


class ServiceType(str, Enum):
    """Types of AI services in the modular architecture."""
    TEXT_GENERATION = "text_generation"
    MUSIC_GENERATION = "music_generation"
    IMAGE_GENERATION = "image_generation"
    INTEGRATION = "integration"


class ResourceLimits(BaseModel):
    """Resource limits for cost control and performance optimization."""
    max_runtime_seconds: int = Field(default=600, ge=1, le=3600, description="Maximum runtime per request")
    max_concurrent_requests: int = Field(default=10, ge=1, le=100, description="Maximum concurrent requests")
    max_memory_gb: Optional[float] = Field(default=None, ge=1, le=80, description="Maximum memory usage in GB")
    max_audio_duration: float = Field(default=300.0, ge=1.0, le=600.0, description="Maximum audio duration in seconds")
    timeout_seconds: int = Field(default=300, ge=30, le=1800, description="Request timeout in seconds")


class CostEstimation(BaseModel):
    """Cost estimation parameters for different GPU types."""
    cost_per_hour_usd: float = Field(description="Cost per hour in USD")
    startup_cost_usd: float = Field(default=0.0, description="One-time startup cost")
    
    @property
    def cost_per_minute_usd(self) -> float:
        """Calculate cost per minute from hourly rate."""
        return self.cost_per_hour_usd / 60.0


class ServiceConfig(BaseModel):
    """Base configuration for AI services."""
    service_type: ServiceType
    gpu_type: GPUType
    scaledown_window_seconds: int = Field(ge=10, le=300, description="Time before scaling down idle instances")
    resource_limits: ResourceLimits
    cost_estimation: CostEstimation
    
    # Service-specific parameters
    model_cache_size_gb: float = Field(default=10.0, ge=1.0, le=100.0)
    enable_torch_compile: bool = Field(default=False, description="Enable PyTorch compilation for performance")
    cpu_offload: bool = Field(default=False, description="Enable CPU offloading for memory optimization")
    
    @field_validator('scaledown_window_seconds')
    @classmethod
    def validate_scaledown_window(cls, v, info):
        """Validate scaledown window based on service type and GPU cost."""
        if info.data and 'service_type' in info.data and 'gpu_type' in info.data:
            service_type = info.data['service_type']
            gpu_type = info.data['gpu_type']
            
            # Text services should scale down quickly (CPU/low-cost)
            if service_type == ServiceType.TEXT_GENERATION and v > 60:
                logger.warning(f"Text generation services should have shorter scaledown windows. Got {v}s")
            
            # Music generation should balance cost vs cold start time
            elif service_type == ServiceType.MUSIC_GENERATION and gpu_type in [GPUType.L40S, GPUType.A100] and v < 30:
                logger.warning(f"High-cost GPU services should have longer scaledown windows. Got {v}s")
                
        return v


class TextGenerationConfig(ServiceConfig):
    """Configuration for text generation services (lyrics, prompts, categories)."""
    service_type: Literal[ServiceType.TEXT_GENERATION] = ServiceType.TEXT_GENERATION
    gpu_type: GPUType = Field(default=GPUType.T4, description="CPU or T4 for cost optimization")
    scaledown_window_seconds: int = Field(default=30, description="Fast scaledown for text services")
    
    # Text-specific parameters
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    model_name: str = Field(default="Qwen/Qwen2-7B-Instruct")
    
    def __init__(self, **data):
        if 'resource_limits' not in data:
            data['resource_limits'] = ResourceLimits(
                max_runtime_seconds=60,
                timeout_seconds=90,
                max_concurrent_requests=20
            )
        if 'cost_estimation' not in data:
            # T4 GPU pricing (approximate)
            data['cost_estimation'] = CostEstimation(
                cost_per_hour_usd=0.35 if data.get('gpu_type') == GPUType.T4 else 0.10,
                startup_cost_usd=0.01
            )
        super().__init__(**data)


class MusicGenerationConfig(ServiceConfig):
    """Configuration for music generation services."""
    service_type: Literal[ServiceType.MUSIC_GENERATION] = ServiceType.MUSIC_GENERATION
    gpu_type: GPUType = Field(default=GPUType.L40S, description="High-memory GPU for ACE-Step")
    scaledown_window_seconds: int = Field(default=60, description="Balance cost vs cold start")
    
    # Music-specific parameters
    max_audio_duration: float = Field(default=300.0, ge=1.0, le=600.0)
    default_guidance_scale: float = Field(default=15.0, ge=1.0, le=30.0)
    default_infer_steps: int = Field(default=60, ge=10, le=200)
    enable_overlapped_decode: bool = Field(default=False)
    
    def __init__(self, **data):
        if 'resource_limits' not in data:
            data['resource_limits'] = ResourceLimits(
                max_runtime_seconds=600,
                timeout_seconds=900,
                max_concurrent_requests=5,
                max_memory_gb=48.0,
                max_audio_duration=300.0
            )
        if 'cost_estimation' not in data:
            # L40S GPU pricing (approximate)
            gpu_cost = 2.50 if data.get('gpu_type') == GPUType.L40S else 4.00  # A100
            data['cost_estimation'] = CostEstimation(
                cost_per_hour_usd=gpu_cost,
                startup_cost_usd=0.05
            )
        super().__init__(**data)


class ImageGenerationConfig(ServiceConfig):
    """Configuration for image generation services."""
    service_type: Literal[ServiceType.IMAGE_GENERATION] = ServiceType.IMAGE_GENERATION
    gpu_type: GPUType = Field(default=GPUType.L4, description="Mid-tier GPU for SDXL-Turbo")
    scaledown_window_seconds: int = Field(default=45, description="Balance between cost and performance")
    
    # Image-specific parameters
    default_num_inference_steps: int = Field(default=2, ge=1, le=50)
    default_guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    max_image_size: int = Field(default=1024, ge=256, le=2048)
    batch_size: int = Field(default=1, ge=1, le=8)
    
    def __init__(self, **data):
        if 'resource_limits' not in data:
            data['resource_limits'] = ResourceLimits(
                max_runtime_seconds=120,
                timeout_seconds=180,
                max_concurrent_requests=10,
                max_memory_gb=16.0
            )
        if 'cost_estimation' not in data:
            # L4 GPU pricing (approximate)
            data['cost_estimation'] = CostEstimation(
                cost_per_hour_usd=0.80,
                startup_cost_usd=0.02
            )
        super().__init__(**data)


class IntegrationConfig(ServiceConfig):
    """Configuration for integration/orchestration services."""
    service_type: Literal[ServiceType.INTEGRATION] = ServiceType.INTEGRATION
    gpu_type: Literal[GPUType.CPU_ONLY] = GPUType.CPU_ONLY
    scaledown_window_seconds: int = Field(default=15, description="Fast scaledown for lightweight service")
    
    # Integration-specific parameters
    service_timeout_seconds: int = Field(default=900, description="Timeout for orchestrated requests")
    max_retry_attempts: int = Field(default=3, ge=1, le=10)
    circuit_breaker_threshold: int = Field(default=5, ge=1, le=20)
    
    def __init__(self, **data):
        if 'resource_limits' not in data:
            data['resource_limits'] = ResourceLimits(
                max_runtime_seconds=900,
                timeout_seconds=1200,
                max_concurrent_requests=50
            )
        if 'cost_estimation' not in data:
            # CPU-only pricing
            data['cost_estimation'] = CostEstimation(
                cost_per_hour_usd=0.05,
                startup_cost_usd=0.001
            )
        super().__init__(**data)


class ServiceConfigFactory:
    """Factory for creating service configurations."""
    
    _config_classes = {
        ServiceType.TEXT_GENERATION: TextGenerationConfig,
        ServiceType.MUSIC_GENERATION: MusicGenerationConfig,
        ServiceType.IMAGE_GENERATION: ImageGenerationConfig,
        ServiceType.INTEGRATION: IntegrationConfig
    }
    
    @classmethod
    def create_config(cls, service_type: ServiceType, **kwargs) -> ServiceConfig:
        """Create a service configuration for the specified type."""
        config_class = cls._config_classes.get(service_type)
        if not config_class:
            raise ValueError(f"Unknown service type: {service_type}")
        
        return config_class(**kwargs)
    
    @classmethod
    def get_default_configs(cls) -> Dict[ServiceType, ServiceConfig]:
        """Get default configurations for all service types."""
        return {
            service_type: config_class()
            for service_type, config_class in cls._config_classes.items()
        }
    
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> List[str]:
        """Validate a service configuration and return any warnings."""
        warnings = []
        
        # Check resource limits vs GPU type
        if config.gpu_type == GPUType.CPU_ONLY and config.resource_limits.max_memory_gb and config.resource_limits.max_memory_gb > 32:
            warnings.append("CPU-only services should not require more than 32GB memory")
        
        # Check cost vs performance trade-offs
        if config.service_type == ServiceType.MUSIC_GENERATION and config.gpu_type == GPUType.T4:
            warnings.append("T4 GPU may be insufficient for music generation workloads")
        
        # Check scaledown windows
        if config.gpu_type in [GPUType.L40S, GPUType.A100] and config.scaledown_window_seconds < 30:
            warnings.append("High-cost GPUs should have longer scaledown windows to amortize startup costs")
        
        return warnings


# Default service configurations
DEFAULT_CONFIGS = ServiceConfigFactory.get_default_configs()