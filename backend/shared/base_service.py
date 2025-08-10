"""
Base service class for all Modal services.
Provides common functionality and structure.
"""

import modal
import logging
import time
import uuid
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from shared.models import ServiceConfig, GenerationMetadata
from shared.monitoring import CostMonitor, TimeoutManager
from shared.storage import FileManager
from shared.config import settings

logger = logging.getLogger(__name__)

def create_service_app(service_name: str, service_config: ServiceConfig, 
                      image: modal.Image, volumes: Optional[Dict[str, modal.Volume]] = None,
                      secrets: Optional[list] = None) -> modal.App:
    """Create a standardized Modal app for a service"""
    app = modal.App(service_name)
    
    # Standard configuration
    app_config = {
        "image": image,
        "scaledown_window": service_config.scaledown_window,
        "timeout": service_config.max_runtime_seconds,
    }
    
    # Add GPU if not CPU
    if service_config.gpu_type.value != "CPU":
        app_config["gpu"] = service_config.gpu_type.value
    
    # Add memory if specified
    if service_config.memory_gb:
        app_config["memory"] = service_config.memory_gb * 1024
    
    # Add volumes if provided
    if volumes:
        app_config["volumes"] = volumes
    
    # Add secrets if provided
    if secrets:
        app_config["secrets"] = secrets
    
    return app, app_config


class ServiceMixin:
    """Mixin class that provides common service functionality for Modal services"""
    
    def init_service_components(self, service_config: ServiceConfig):
        """Initialize service components - call this in Modal @enter() method"""
        self.config = service_config
        self.cost_monitor = CostMonitor()
        self.timeout_manager = TimeoutManager(service_config.max_runtime_seconds)
        self.file_manager = FileManager()
        self._model_loaded = False
    
    def start_operation(self, operation_id: str, operation_type: str = "generation"):
        """Start monitoring for an operation"""
        self.cost_monitor.start_operation(
            operation_id, 
            self.config.gpu_type.value, 
            self.config.service_name,
            {"operation_type": operation_type}
        )
        self.timeout_manager.start_timeout(
            operation_id, 
            self.config.max_runtime_seconds,
            operation_type
        )
    
    def end_operation(self, operation_id: str, success: bool = True):
        """End monitoring for an operation"""
        self.cost_monitor.end_operation(operation_id, success)
        self.timeout_manager.end_timeout(operation_id)
    
    def check_timeout(self, operation_id: str) -> bool:
        """Check if operation has timed out"""
        return self.timeout_manager.check_timeout(operation_id)
    
    def cleanup_operation(self, operation_id: str):
        """Clean up monitoring for failed operations"""
        try:
            self.cost_monitor.end_operation(operation_id, success=False)
            self.timeout_manager.end_timeout(operation_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup operation {operation_id}: {e}")
    
    def validate_model_loaded(self):
        """Validate that model is loaded"""
        if not getattr(self, '_model_loaded', False):
            raise RuntimeError("Model not loaded. Service temporarily unavailable.")
    
    def generate_operation_id(self) -> str:
        """Generate unique operation ID"""
        return str(uuid.uuid4())
    
    def create_metadata(self, operation_id: str, model_info: str, start_time: float) -> GenerationMetadata:
        """Create metadata for responses"""
        generation_time = time.time() - start_time
        estimated_cost = generation_time * (self.config.cost_per_hour / 3600)
        
        return GenerationMetadata(
            generation_time=generation_time,
            model_info=model_info,
            gpu_type=self.config.gpu_type.value,
            estimated_cost=estimated_cost,
            operation_id=operation_id
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Standard health check endpoint"""
        return {
            "status": "healthy",
            "service": self.config.service_name,
            "model_loaded": getattr(self, '_model_loaded', False),
            "gpu_type": self.config.gpu_type.value,
            "scaledown_window": self.config.scaledown_window,
            "max_runtime": self.config.max_runtime_seconds,
            "timestamp": time.time()
        }


def standard_error_handler(func):
    """Decorator for standard error handling in service endpoints"""
    def wrapper(*args, **kwargs):
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            return func(*args, operation_id=operation_id, start_time=start_time, **kwargs)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{operation_id}] Operation failed after {duration:.2f}s: {e}")
            raise
    
    return wrapper