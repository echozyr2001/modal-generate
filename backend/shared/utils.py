import os
import uuid
import logging
import time
from typing import Optional
from shared.config import settings

logger = logging.getLogger(__name__)

# Import from new modular structure
from .storage import FileManager, S3Manager
from .monitoring import CostMonitor, TimeoutManager


def ensure_output_dir(output_dir: str = "/tmp/outputs") -> str:
    """Ensure output directory exists"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_temp_filepath(output_dir: str, extension: str) -> str:
    """Generate temporary file path"""
    return os.path.join(output_dir, f"{uuid.uuid4()}{extension}")


def create_metadata(operation_id: str, model_info: str, start_time: float, 
                   gpu_type: str = "CPU", cost_per_hour: float = 0.05) -> 'GenerationMetadata':
    """Create metadata for generation responses"""
    from .models import GenerationMetadata
    
    generation_time = time.time() - start_time
    estimated_cost = generation_time * (cost_per_hour / 3600)
    
    return GenerationMetadata(
        generation_time=generation_time,
        model_info=model_info,
        gpu_type=gpu_type,
        estimated_cost=estimated_cost,
        operation_id=operation_id
    )


# Backward compatibility exports
__all__ = [
    'FileManager',
    'S3Manager', 
    'CostMonitor',
    'TimeoutManager',
    'ensure_output_dir',
    'generate_temp_filepath',
    'create_metadata'
]