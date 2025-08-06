"""
Base classes for modular AI services
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from contextlib import contextmanager

from .models import ServiceConfig, ServiceResponse, GenerationMetadata, GPUType
from .utils import CostMonitor, TimeoutManager, FileManager

logger = logging.getLogger(__name__)


class ServiceBase(ABC):
    """Base class for all AI services"""
    
    def __init__(self, config: ServiceConfig, file_manager: Optional[FileManager] = None):
        self.config = config
        self.file_manager = file_manager or FileManager()
        self.cost_monitor = CostMonitor()
        self.timeout_manager = TimeoutManager(config.max_runtime_seconds)
        self._model_loaded = False
        
        logger.info(f"Initialized {config.service_name} with GPU: {config.gpu_type}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the AI model for this service"""
        pass
    
    @abstractmethod
    def generate(self, request: Any) -> Dict[str, Any]:
        """Generate content based on request"""
        pass
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and status"""
        return {
            "service_name": self.config.service_name,
            "gpu_type": self.config.gpu_type.value,
            "model_loaded": self._model_loaded,
            "max_runtime": self.config.max_runtime_seconds,
            "scaledown_window": self.config.scaledown_window,
            "cost_per_hour": self.config.cost_per_hour
        }
    
    @contextmanager
    def operation_context(self, operation_type: str = "generation"):
        """Context manager for tracking operations with cost and timeout monitoring"""
        operation_id = str(uuid.uuid4())
        
        # Start monitoring
        self.cost_monitor.start_operation(
            operation_id, 
            self.config.gpu_type.value, 
            self.config.service_name
        )
        self.timeout_manager.start_timeout(operation_id, self.config.max_runtime_seconds)
        
        start_time = time.time()
        
        try:
            yield operation_id
        except Exception as e:
            logger.error(f"Operation {operation_id} failed: {e}")
            raise
        finally:
            # End monitoring
            cost_data = self.cost_monitor.end_operation(operation_id)
            timeout_data = self.timeout_manager.end_timeout(operation_id)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Operation {operation_id} completed in {duration:.2f}s, "
                       f"cost: ${cost_data.get('estimated_cost', 0):.4f}")
    
    def create_metadata(self, operation_id: str, model_info: str, 
                       start_time: float) -> GenerationMetadata:
        """Create metadata for generation response"""
        end_time = time.time()
        duration = end_time - start_time
        estimated_cost = self.cost_monitor.get_operation_cost(operation_id)
        
        return GenerationMetadata(
            generation_time=duration,
            model_info=model_info,
            gpu_type=self.config.gpu_type.value,
            estimated_cost=estimated_cost,
            operation_id=operation_id
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": self.config.service_name,
            "model_loaded": self._model_loaded,
            "timestamp": time.time()
        }


class TextGenerationService(ServiceBase):
    """Base class for text generation services (lyrics, prompts, categories)"""
    
    def __init__(self, config: ServiceConfig, file_manager: Optional[FileManager] = None):
        # Text services should use CPU or low-cost GPU
        if config.gpu_type not in [GPUType.CPU, GPUType.T4]:
            logger.warning(f"Text service {config.service_name} using expensive GPU: {config.gpu_type}")
        
        super().__init__(config, file_manager)
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load text generation model"""
        pass
    
    def validate_text_input(self, text: str, max_length: int = 1000) -> str:
        """Validate and clean text input"""
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
        
        # Clean text
        cleaned = text.strip()
        if len(cleaned) > max_length:
            raise ValueError(f"Text input too long: {len(cleaned)} > {max_length}")
        
        return cleaned


class ImageGenerationService(ServiceBase):
    """Base class for image generation services"""
    
    def __init__(self, config: ServiceConfig, file_manager: Optional[FileManager] = None):
        # Image services should use appropriate GPU
        if config.gpu_type == GPUType.CPU:
            logger.warning(f"Image service {config.service_name} using CPU - may be slow")
        
        super().__init__(config, file_manager)
        self.pipeline = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load image generation model"""
        pass
    
    def save_image(self, image, file_type: str = "images") -> str:
        """Save generated image and return file path"""
        import tempfile
        
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file.name, format="PNG")
            
            # Use FileManager to save to final location
            file_path = self.file_manager.save_file(tmp_file.name, file_type=file_type)
            return file_path


class AudioGenerationService(ServiceBase):
    """Base class for audio/music generation services"""
    
    def __init__(self, config: ServiceConfig, file_manager: Optional[FileManager] = None):
        # Audio services typically need high-memory GPU
        if config.gpu_type in [GPUType.CPU, GPUType.T4]:
            logger.warning(f"Audio service {config.service_name} may need more powerful GPU")
        
        super().__init__(config, file_manager)
        self.model = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load audio generation model"""
        pass
    
    def save_audio(self, audio_data, sample_rate: int = 44100, 
                   file_type: str = "audio") -> str:
        """Save generated audio and return file path"""
        import tempfile
        import soundfile as sf
        
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            # Use FileManager to save to final location
            file_path = self.file_manager.save_file(tmp_file.name, file_type=file_type)
            return file_path


class OrchestrationService(ServiceBase):
    """Base class for service orchestration"""
    
    def __init__(self, config: ServiceConfig, service_urls: Dict[str, str],
                 file_manager: Optional[FileManager] = None):
        # Orchestration services should use CPU only
        if config.gpu_type != GPUType.CPU:
            logger.warning(f"Orchestration service using GPU: {config.gpu_type}")
        
        super().__init__(config, file_manager)
        self.service_urls = service_urls
        self._http_client = None
    
    def load_model(self) -> None:
        """Orchestration services don't load models"""
        self._model_loaded = True
        logger.info(f"Orchestration service {self.config.service_name} ready")
    
    @property
    def http_client(self):
        """Lazy initialization of HTTP client"""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def call_service(self, service_name: str, endpoint: str, 
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Call another service with error handling and retries"""
        if service_name not in self.service_urls:
            raise ValueError(f"Service {service_name} not configured")
        
        url = f"{self.service_urls[service_name]}{endpoint}"
        
        try:
            response = await self.http_client.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to call {service_name}{endpoint}: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        if self._http_client:
            await self._http_client.aclose()