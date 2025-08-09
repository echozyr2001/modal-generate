"""
Unified Cover Image Generation Service
Provides cover image generation using SDXL-Turbo.
"""

import modal
import logging
import torch
import time
import uuid
import os
import base64
from typing import Optional, Dict, Any
from PIL import Image
from fastapi import HTTPException

from shared.models import (
    CoverImageGenerationRequest, 
    CoverImageGenerationResponse,
    ServiceConfig,
    GPUType,
    GenerationMetadata
)
from shared.deployment import image_generation_image, hf_volume, music_gen_secrets
from shared.config import settings
from shared.base_service import create_service_app

logger = logging.getLogger(__name__)

# Get service configuration
cover_image_config = ServiceConfig(
    service_name="cover-image-generator",
    **settings.get_service_config("image")
)

# Create Modal app
app, app_config = create_service_app(
    "cover-image-generator",
    cover_image_config,
    image_generation_image,
    volumes={"/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets]
)


@app.cls(**app_config)
class CoverImageGenServer:
    """Unified cover image generation server"""
    
    @modal.enter()
    def load_model(self):
        """Load SDXL-Turbo model for image generation"""
        from diffusers import AutoPipelineForText2Image
        from shared.monitoring import CostMonitor, TimeoutManager
        from shared.storage import FileManager
        
        # Initialize service components (since we can't use __init__)
        self.config = cover_image_config
        self.cost_monitor = CostMonitor()
        self.timeout_manager = TimeoutManager(cover_image_config.max_runtime_seconds)
        self.file_manager = FileManager()
        self._model_loaded = False
        self.model_id = "stabilityai/sdxl-turbo"
        
        logger.info(f"Loading image generation model: {self.model_id}")
        
        try:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir="/.cache/huggingface"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipeline.to("cuda")
                logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                logger.warning("CUDA not available, using CPU")
            
            # Enable memory efficient attention
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            self._model_loaded = True
            logger.info("SDXL-Turbo model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load image generation model: {e}")
            self._model_loaded = False
            raise
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": self.config.service_name,
            "gpu_type": self.config.gpu_type.value,
            "model_id": self.model_id,
            "supported_styles": [
                "album cover art", "vintage", "modern", "abstract", 
                "photorealistic", "illustration", "grunge", "electronic"
            ],
            "image_resolution": "512x512",
            "supported_formats": ["PNG"],
            "batch_limit": 10,
            "scaledown_window": self.config.scaledown_window,
            "max_runtime": self.config.max_runtime_seconds,
            "storage_mode": self.file_manager.get_storage_mode()
        }
    
    def validate_model_loaded(self):
        """Validate that model is loaded"""
        if not getattr(self, '_model_loaded', False):
            raise RuntimeError("Model not loaded. Service temporarily unavailable.")
    
    def generate_operation_id(self) -> str:
        """Generate unique operation ID"""
        return str(uuid.uuid4())
    
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
    
    def get_style_presets(self) -> Dict[str, str]:
        """Get available style presets"""
        return {
            "album cover art": "professional album cover art, high quality, detailed",
            "vintage": "vintage album cover, retro style, aged paper texture",
            "modern": "modern minimalist design, clean lines, contemporary",
            "abstract": "abstract art, geometric shapes, artistic composition",
            "photorealistic": "photorealistic, detailed photography, studio lighting",
            "illustration": "digital illustration, artistic style, vibrant colors",
            "grunge": "grunge style, distressed texture, alternative rock aesthetic",
            "electronic": "electronic music aesthetic, neon colors, futuristic design"
        }
    
    def generate_image_internal(self, request: CoverImageGenerationRequest, operation_id: str) -> dict:
        """Internal image generation method"""
        self.validate_model_loaded()
        
        if self.check_timeout(operation_id):
            raise TimeoutError(f"Operation {operation_id} timed out before generation")
        
        logger.info(f"[{operation_id}] Generating image: {request.prompt[:100]}...")
        
        # Get style preset
        style_presets = self.get_style_presets()
        style_text = style_presets.get(request.style.lower(), request.style)
        
        # Build enhanced prompt
        full_prompt = f"{request.prompt}, {style_text}, high quality, detailed, professional"
        negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, text, watermark"
        
        # Generation parameters
        generation_params = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": request.num_inference_steps or 4,
            "guidance_scale": 0.0,  # SDXL-Turbo works best with guidance_scale=0
            "width": request.width or 512,
            "height": request.height or 512,
            "num_images_per_prompt": 1
        }
        
        # Add seed if provided
        if request.seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(request.seed)
            generation_params["generator"] = generator
        
        try:
            logger.info(f"[{operation_id}] Starting generation with steps={generation_params['num_inference_steps']}")
            image = self.pipeline(**generation_params).images[0]
            logger.info(f"[{operation_id}] Image generation successful")
        except Exception as e:
            logger.error(f"[{operation_id}] Generation failed: {e}")
            # Fallback to simpler generation
            fallback_params = {
                "prompt": full_prompt,
                "num_inference_steps": min(request.num_inference_steps or 2, 2),
                "guidance_scale": 0.0,
                "width": request.width or 512,
                "height": request.height or 512
            }
            if request.seed is not None:
                fallback_params["generator"] = generation_params.get("generator")
            
            try:
                logger.info(f"[{operation_id}] Trying fallback generation")
                image = self.pipeline(**fallback_params).images[0]
                logger.info(f"[{operation_id}] Fallback generation successful")
            except Exception as fallback_error:
                logger.error(f"[{operation_id}] Fallback generation failed: {fallback_error}")
                raise
        
        # Save image
        file_path = self.save_image(image, operation_id)
        
        # Get image metadata
        image_dimensions = image.size
        file_size_mb = self.estimate_file_size(image)
        
        return {
            "file_path": file_path,
            "image_dimensions": image_dimensions,
            "file_size_mb": file_size_mb,
            "style_used": style_text,
            "resolution": f"{image_dimensions[0]}x{image_dimensions[1]}"
        }
    
    def save_image(self, image: Image.Image, operation_id: str) -> str:
        """Save image using FileManager"""
        try:
            # Create filename
            filename = f"{operation_id}.png"
            
            # Save to temporary file first
            temp_path = f"/tmp/{filename}"
            image.save(temp_path, "PNG")
            
            # Save using FileManager
            file_path = self.file_manager.save_file(temp_path, file_type="cover_images")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.info(f"Image saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise RuntimeError(f"Image save failed: {e}")
    
    def estimate_file_size(self, image: Image.Image) -> float:
        """Estimate file size in MB for PNG format"""
        width, height = image.size
        estimated_bytes = width * height * 4  # RGBA
        return round(estimated_bytes / (1024 * 1024), 2)
    
    @modal.fastapi_endpoint(method="POST")
    def generate_cover_image(self, request: CoverImageGenerationRequest) -> CoverImageGenerationResponse:
        """Generate single cover image"""
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Cover image generation: {request.prompt[:100]}...")
        
        # Validate request
        try:
            self.validate_generation_request(request)
        except ValueError as e:
            logger.error(f"[{operation_id}] Request validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Start monitoring
        self.start_operation(operation_id, "image_generation")
        
        try:
            # Generate image
            result = self.generate_image_internal(request, operation_id)
            
            # End monitoring
            self.end_operation(operation_id, success=True)
            
            # Create metadata
            metadata = self.create_metadata(operation_id, f"SDXL-Turbo ({self.model_id})", start_time)
            
            response = CoverImageGenerationResponse(
                file_path=result["file_path"],
                image_dimensions=result["image_dimensions"],
                file_size_mb=result["file_size_mb"],
                metadata=metadata
            )
            
            logger.info(f"[{operation_id}] Cover image generated successfully")
            return response
            
        except TimeoutError as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Generation timed out: {e}")
            raise HTTPException(status_code=408, detail="Image generation timed out")
        
        except torch.cuda.OutOfMemoryError as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] GPU out of memory: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=507, detail="GPU out of memory")
        
        except Exception as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Generation failed: {e}")
            raise HTTPException(status_code=500, detail="Image generation failed")
    
    def validate_generation_request(self, request: CoverImageGenerationRequest):
        """Validate generation request parameters"""
        if not self._model_loaded:
            raise ValueError("Service not ready: Model not loaded")
        
        if len(request.prompt.strip()) < 5:
            raise ValueError("Prompt too short: Must be at least 5 characters")
        
        if len(request.prompt) > 500:
            raise ValueError("Prompt too long: Maximum 500 characters allowed")
        
        if request.width * request.height > 1024 * 1024:
            raise ValueError("Image dimensions too large: Maximum resolution is 1024x1024")
        
        if request.num_inference_steps > 10:
            raise ValueError("Too many inference steps: SDXL-Turbo works best with 1-10 steps")
    
    @modal.fastapi_endpoint(method="GET")
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": self.config.service_name,
            "model_loaded": getattr(self, '_model_loaded', False),
            "gpu_type": self.config.gpu_type.value,
            "scaledown_window": self.config.scaledown_window,
            "max_runtime": self.config.max_runtime_seconds,
            "timestamp": time.time()
        }
    
    @modal.fastapi_endpoint(method="GET")
    def service_info(self) -> Dict[str, Any]:
        """Service information endpoint"""
        return self.get_service_info()
    
    @modal.fastapi_endpoint(method="GET")
    def get_available_styles(self) -> Dict[str, Any]:
        """Get available style presets with descriptions"""
        styles = {
            style: {
                "description": preset,
                "best_for": self.get_style_recommendations().get(style, "General use")
            }
            for style, preset in self.get_style_presets().items()
        }
        
        return {
            "available_styles": styles,
            "custom_style_supported": True,
            "style_mixing_supported": True
        }
    
    def get_style_recommendations(self) -> Dict[str, str]:
        """Get style recommendations"""
        return {
            "album cover art": "General music albums, professional releases",
            "vintage": "Classic rock, jazz, blues, nostalgic themes",
            "modern": "Electronic, pop, indie music",
            "abstract": "Experimental, ambient, avant-garde music",
            "photorealistic": "Portrait-based albums, realistic scenes",
            "illustration": "Concept albums, fantasy themes, animated styles",
            "grunge": "Rock, punk, alternative music",
            "electronic": "EDM, techno, synthwave, electronic music"
        }
    
    @modal.fastapi_endpoint(method="GET")
    def download_image(self, file_key: str) -> dict:
        """Download image as base64 data"""
        from fastapi import HTTPException
        
        storage_path = settings.local_storage_dir
        file_path = os.path.join(storage_path, file_key)
        
        if not os.path.exists(file_path) and os.path.exists(file_key):
            file_path = file_key
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_key}")
        
        try:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            file_size = len(image_bytes)
            
            return {
                "image_data": image_b64,
                "file_size": file_size,
                "filename": os.path.basename(file_path),
                "format": "PNG"
            }
            
        except Exception as e:
            logger.error(f"Failed to read image file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to read image: {e}")


@app.local_entrypoint()
def test_cover_image_service():
    """Test the unified cover image service"""
    import requests
    import time
    
    server = CoverImageGenServer()
    
    print("=== Testing Unified Cover Image Service ===")
    print(f"Service: {cover_image_config.service_name}")
    print(f"GPU: {cover_image_config.gpu_type.value}")
    print("-" * 50)
    
    # Wait for initialization
    print("Waiting for service initialization...")
    time.sleep(10)
    
    try:
        # Test health check
        health_url = server.health_check.get_web_url()
        print(f"Health check: {health_url}")
        
        for attempt in range(3):
            try:
                response = requests.get(health_url, timeout=60)
                response.raise_for_status()
                health = response.json()
                print(f"✓ Health check: {health['status']}, Model loaded: {health['model_loaded']}")
                
                if health['model_loaded']:
                    break
                else:
                    print("Waiting for model to load...")
                    time.sleep(15)
            except Exception as e:
                print(f"Health check attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)
        
        # Test image generation
        print("\nTesting cover image generation...")
        
        test_request = CoverImageGenerationRequest(
            prompt="a vibrant electronic music album cover with neon lights",
            style="electronic",
            width=512,
            height=512,
            num_inference_steps=4
        )
        
        endpoint_url = server.generate_cover_image.get_web_url()
        print(f"Generation URL: {endpoint_url}")
        
        start_time = time.time()
        response = requests.post(endpoint_url, json=test_request.model_dump(), timeout=120)
        duration = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Image generated in {duration:.2f}s")
        print(f"  File path: {result['file_path']}")
        print(f"  Dimensions: {result['image_dimensions']}")
        print(f"  File size: {result['file_size_mb']}MB")
        print(f"  Generation time: {result['metadata']['generation_time']:.2f}s")
        
        # Test style information
        print("\nTesting style information...")
        styles_url = server.get_available_styles.get_web_url()
        styles_response = requests.get(styles_url, timeout=30)
        styles_response.raise_for_status()
        styles_data = styles_response.json()
        
        print(f"✓ Available styles: {len(styles_data['available_styles'])} styles")
        
        print("\n" + "="*50)
        print("Cover image service test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cover_image_service()