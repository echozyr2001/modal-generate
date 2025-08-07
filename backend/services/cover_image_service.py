import modal
import logging
import torch
import time
from typing import Optional, Tuple
from PIL import Image

from shared.models import (
    CoverImageGenerationRequest, 
    CoverImageGenerationResponseEnhanced,
    ServiceConfig,
    GPUType,
    GenerationMetadata
)
from shared.modal_images import image_generation_image
from shared.modal_config import hf_volume, music_gen_secrets
from shared.utils import FileManager
from shared.config import settings

logger = logging.getLogger(__name__)

# Service configuration optimized for mid-tier GPU (T4/L4) with 45-second scaledown
cover_image_config = ServiceConfig(
    service_name="cover-image-generator",
    gpu_type=GPUType.L4,  # Mid-tier GPU for SDXL-Turbo
    scaledown_window=45,  # 45-second scaledown window
    max_runtime_seconds=120,  # 2 minutes max for image generation
    max_concurrent_requests=5,  # Reasonable limit for GPU service
    memory_gb=16,  # Sufficient for SDXL-Turbo
    cost_per_hour=0.60  # L4 cost estimate
)

app = modal.App("cover-image-generator")


class SimpleTimeoutManager:
    """Simple timeout manager for operations"""
    def __init__(self):
        self.operations = {}
    
    def check_timeout(self, operation_id: str) -> bool:
        """Check if operation has timed out"""
        if operation_id in self.operations:
            start_time = self.operations[operation_id]
            elapsed = time.time() - start_time
            return elapsed > cover_image_config.max_runtime_seconds
        return False
    
    def start_timeout(self, operation_id: str):
        """Start timeout tracking for operation"""
        self.operations[operation_id] = time.time()
    
    def end_timeout(self, operation_id: str):
        """End timeout tracking for operation"""
        self.operations.pop(operation_id, None)
    
    def get_elapsed_time(self, operation_id: str) -> float:
        """Get elapsed time for operation"""
        if operation_id in self.operations:
            return time.time() - self.operations[operation_id]
        return 0.0


@app.cls(
    image=image_generation_image,
    gpu=cover_image_config.gpu_type.value,
    volumes={"/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=cover_image_config.scaledown_window,
    timeout=cover_image_config.max_runtime_seconds
)
class CoverImageGenServer:
    # Remove custom __init__ to fix Modal deprecation warning
    # Initialize these in load_model instead

    @modal.enter()
    def load_model(self):
        """Load SDXL-Turbo model optimized for fast image generation"""
        from diffusers import AutoPipelineForText2Image
        import os
        
        # Initialize attributes that were previously in __init__
        self.config = cover_image_config
        self.pipeline = None
        self.model_id = "stabilityai/sdxl-turbo"
        self._model_loaded = False
        
        # Initialize FileManager for storage handling
        # For testing, force local storage to save costs
        settings.switch_to_local_storage("./outputs")
        self.file_manager = FileManager(
            use_s3=settings.use_s3_storage,
            local_storage_dir=settings.local_storage_dir
        )
        
        # Simple timeout manager
        self.timeout_manager = SimpleTimeoutManager()
        
        logger.info(f"Loading image generation model: {self.model_id}")
        logger.info(f"Storage mode: {'S3' if settings.use_s3_storage else 'Local'} ({settings.local_storage_dir})")
        
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

    def generate(self, request: CoverImageGenerationRequest) -> dict:
        """Generate cover image based on request with style customization"""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Enhanced style customization
        style_presets = {
            "album cover art": "professional album cover art, high quality, detailed",
            "vintage": "vintage album cover, retro style, aged paper texture",
            "modern": "modern minimalist design, clean lines, contemporary",
            "abstract": "abstract art, geometric shapes, artistic composition",
            "photorealistic": "photorealistic, detailed photography, studio lighting",
            "illustration": "digital illustration, artistic style, vibrant colors",
            "grunge": "grunge style, distressed texture, alternative rock aesthetic",
            "electronic": "electronic music aesthetic, neon colors, futuristic design"
        }
        
        # Use preset if available, otherwise use custom style
        style_text = style_presets.get(request.style.lower(), request.style)
        
        # Build enhanced prompt with negative prompts for better quality
        full_prompt = f"{request.prompt}, {style_text}, high quality, detailed, professional"
        negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, text, watermark"
        
        logger.info(f"Generating cover image for prompt: {request.prompt[:100]}...")
        logger.debug(f"Full prompt: {full_prompt}")
        
        # Generate image using SDXL-Turbo with request parameters
        generation_params = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": request.num_inference_steps or 4,
            "guidance_scale": 0.0,  # SDXL-Turbo works best with guidance_scale=0
            "width": request.width or 512,
            "height": request.height or 512,
            "num_images_per_prompt": 1
        }
        
        # Add seed if provided for reproducible generation
        if request.seed is not None:
            import torch
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(request.seed)
            generation_params["generator"] = generator
        
        try:
            logger.info(f"Starting image generation with params: steps={generation_params['num_inference_steps']}, size={generation_params['width']}x{generation_params['height']}")
            image = self.pipeline(**generation_params).images[0]
            logger.info("Image generation successful")
        except Exception as e:
            logger.error(f"Image generation failed with full parameters: {e}")
            # Fallback to simpler generation without negative prompt
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
                logger.info(f"Trying fallback generation with simplified params")
                image = self.pipeline(**fallback_params).images[0]
                logger.info("Fallback generation successful")
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {fallback_error}")
                raise
        
        # Save image using FileManager
        file_path = self.save_image(image, file_type="cover_images")
        
        # Get comprehensive image metadata
        image_dimensions = image.size
        file_size_mb = self._estimate_file_size(image)
        
        return {
            "file_path": file_path,
            "image_dimensions": image_dimensions,
            "file_size_mb": file_size_mb,
            "style_used": style_text,
            "resolution": f"{image_dimensions[0]}x{image_dimensions[1]}"
        }
    
    def _estimate_file_size(self, image: Image.Image) -> float:
        """Estimate file size in MB for PNG format"""
        # Rough estimation: width * height * 4 bytes (RGBA) / 1024^2
        width, height = image.size
        estimated_bytes = width * height * 4
        return round(estimated_bytes / (1024 * 1024), 2)
    
    def save_image(self, image: Image.Image, file_type: str = "cover_images") -> str:
        """Save image using FileManager"""
        import uuid
        import os
        
        try:
            # Create filename with UUID
            filename = f"{uuid.uuid4()}.png"
            
            # Ensure output directory exists
            storage_dir = getattr(self.file_manager, 'local_storage_dir', './outputs')
            os.makedirs(storage_dir, exist_ok=True)
            
            # Save to local file
            file_path = os.path.join(storage_dir, filename)
            image.save(file_path, "PNG")
            
            # Verify file was saved
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"Image saved successfully to: {file_path} ({file_size} bytes)")
            else:
                logger.error(f"File was not saved properly: {file_path}")
                raise RuntimeError("Failed to save image file")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise RuntimeError(f"Image save failed: {e}")
    
    def operation_context(self, operation_type: str):
        """Simple operation context manager"""
        import uuid
        from contextlib import contextmanager
        
        @contextmanager
        def context():
            operation_id = str(uuid.uuid4())
            logger.info(f"Starting operation: {operation_type} ({operation_id})")
            
            # Start timeout tracking
            if hasattr(self, 'timeout_manager'):
                self.timeout_manager.start_timeout(operation_id)
            
            try:
                yield operation_id
            finally:
                # End timeout tracking
                if hasattr(self, 'timeout_manager'):
                    elapsed = self.timeout_manager.get_elapsed_time(operation_id)
                    self.timeout_manager.end_timeout(operation_id)
                    logger.info(f"Completed operation: {operation_type} ({operation_id}) in {elapsed:.2f}s")
                else:
                    logger.info(f"Completed operation: {operation_type} ({operation_id})")
        
        return context()
    
    def create_metadata(self, operation_id: str, model_info: str, start_time: float) -> GenerationMetadata:
        """Create metadata for enhanced responses"""
        generation_time = time.time() - start_time
        estimated_cost = generation_time * (cover_image_config.cost_per_hour / 3600) if hasattr(cover_image_config, 'cost_per_hour') else 0.0
        
        return GenerationMetadata(
            operation_id=operation_id,
            model_info=model_info,
            generation_time=generation_time,
            estimated_cost=estimated_cost,
            gpu_type=cover_image_config.gpu_type.value
        )

    @modal.fastapi_endpoint(method="POST")
    def generate_cover_image(self, request: CoverImageGenerationRequest) -> CoverImageGenerationResponseEnhanced:
        """Generate single cover image with comprehensive error handling and validation"""
        logger.info(f"Generating cover image for prompt: {request.prompt[:100]}...")
        
        # Pre-generation validation
        try:
            self._validate_generation_request(request)
        except ValueError as e:
            logger.error(f"Request validation failed: {e}")
            raise ValueError(f"Invalid request: {e}")
        
        with self.operation_context("cover_image_generation") as operation_id:
            start_time = time.time()
            
            # Check timeout periodically during generation
            if self.timeout_manager.check_timeout(operation_id):
                raise TimeoutError(f"Operation {operation_id} timed out")
            
            try:
                # Generate image with timeout protection
                result = self._generate_with_timeout(request, operation_id)
                
                # Create metadata
                metadata = self.create_metadata(
                    operation_id=operation_id,
                    model_info=f"SDXL-Turbo ({self.model_id})",
                    start_time=start_time
                )
                
                response = CoverImageGenerationResponseEnhanced(
                    file_path=result["file_path"],
                    image_dimensions=result["image_dimensions"],
                    file_size_mb=result["file_size_mb"],
                    metadata=metadata
                )
                
                logger.info(f"Cover image generated successfully: {result['file_path']}")
                return response
                
            except TimeoutError as e:
                logger.error(f"Generation timed out: {e}")
                raise TimeoutError("Image generation timed out. Please try again with a simpler prompt.")
            
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU out of memory: {e}")
                # Try to clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory. Please try again or reduce image dimensions.")
            
            except Exception as e:
                logger.error(f"Failed to generate cover image: {e}")
                # Provide user-friendly error messages
                if "NSFW" in str(e) or "safety" in str(e).lower():
                    raise ValueError("Content filtered: Please use a different prompt that doesn't trigger safety filters.")
                elif "connection" in str(e).lower() or "network" in str(e).lower():
                    raise RuntimeError("Network error: Please check your connection and try again.")
                elif "model" in str(e).lower() and "load" in str(e).lower():
                    raise RuntimeError("Model loading error: Service is temporarily unavailable.")
                else:
                    raise RuntimeError(f"Image generation failed: {str(e)[:100]}")
    
    def _validate_generation_request(self, request: CoverImageGenerationRequest) -> None:
        """Validate generation request parameters"""
        # Check if model is loaded
        if not self._model_loaded:
            raise ValueError("Service not ready: Model not loaded")
        
        # Validate prompt length and content
        if len(request.prompt.strip()) < 5:
            raise ValueError("Prompt too short: Must be at least 5 characters")
        
        if len(request.prompt) > 500:
            raise ValueError("Prompt too long: Maximum 500 characters allowed")
        
        # Validate dimensions
        if request.width * request.height > 1024 * 1024:
            raise ValueError("Image dimensions too large: Maximum resolution is 1024x1024")
        
        # Validate inference steps for SDXL-Turbo
        if request.num_inference_steps > 10:
            raise ValueError("Too many inference steps: SDXL-Turbo works best with 1-10 steps")
    
    def _generate_with_timeout(self, request: CoverImageGenerationRequest, operation_id: str) -> dict:
        """Generate image with simple timeout check"""
        # Simple timeout check without signals (which don't work in thread pools)
        start_time = time.time()
        
        # Check timeout before generation
        if self.timeout_manager.check_timeout(operation_id):
            raise TimeoutError(f"Operation {operation_id} timed out before generation")
        
        try:
            result = self.generate(request)
            
            # Check if generation took too long
            generation_time = time.time() - start_time
            if generation_time > self.config.max_runtime_seconds:
                logger.warning(f"Generation took {generation_time:.2f}s, exceeding limit of {self.config.max_runtime_seconds}s")
            
            return result
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Generation failed after {generation_time:.2f}s: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_cover_image_batch(self, requests: list[CoverImageGenerationRequest]) -> dict:
        """Generate multiple cover images with enhanced batch processing and progress tracking"""
        if len(requests) > 10:  # Limit batch size to prevent resource exhaustion
            raise ValueError("Batch size cannot exceed 10 images")
        
        if not requests:
            raise ValueError("Batch request cannot be empty")
        
        logger.info(f"Starting batch generation of {len(requests)} cover images...")
        
        results = []
        failed_requests = []
        batch_start_time = time.time()
        
        # Process each request in the batch
        for i, request in enumerate(requests):
            with self.operation_context(f"batch_cover_image_{i}") as operation_id:
                start_time = time.time()
                
                try:
                    # Generate individual image
                    result = self.generate(request)
                    
                    # Create metadata for this image
                    metadata = self.create_metadata(
                        operation_id=operation_id,
                        model_info=f"SDXL-Turbo ({self.model_id})",
                        start_time=start_time
                    )
                    
                    response = CoverImageGenerationResponseEnhanced(
                        file_path=result["file_path"],
                        image_dimensions=result["image_dimensions"],
                        file_size_mb=result["file_size_mb"],
                        metadata=metadata
                    )
                    
                    results.append(response)
                    logger.info(f"Batch progress: {i+1}/{len(requests)} completed successfully")
                    
                except Exception as e:
                    error_info = {
                        "index": i,
                        "request": request.model_dump(),
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    failed_requests.append(error_info)
                    logger.error(f"Failed to generate image {i+1}/{len(requests)}: {e}")
                    
                    # Continue processing remaining images unless too many failures
                    if len(failed_requests) > len(requests) // 2:  # Fail if more than half fail
                        logger.error(f"Too many failures in batch: {len(failed_requests)}/{len(requests)}")
                        raise RuntimeError(f"Batch processing aborted: {len(failed_requests)} out of {len(requests)} images failed")
        
        batch_duration = time.time() - batch_start_time
        success_count = len(results)
        failure_count = len(failed_requests)
        
        # Calculate batch statistics
        total_file_size = sum(img.file_size_mb or 0 for img in results)
        avg_generation_time = sum(img.metadata.generation_time for img in results) / max(success_count, 1)
        
        batch_summary = {
            "successful_images": results,
            "failed_requests": failed_requests,
            "batch_statistics": {
                "total_requested": len(requests),
                "successful_count": success_count,
                "failed_count": failure_count,
                "success_rate": success_count / len(requests),
                "total_batch_time": batch_duration,
                "average_generation_time": avg_generation_time,
                "total_file_size_mb": total_file_size
            }
        }
        
        logger.info(f"Batch completed: {success_count} successful, {failure_count} failed, "
                   f"total time: {batch_duration:.2f}s")
        
        return batch_summary


    @modal.fastapi_endpoint(method="GET")
    def health_check(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": cover_image_config.service_name,
            "model_loaded": getattr(self, '_model_loaded', False),
            "gpu_type": cover_image_config.gpu_type.value,
            "scaledown_window": cover_image_config.scaledown_window,
            "max_runtime": cover_image_config.max_runtime_seconds,
            "timestamp": time.time()
        }

    @modal.fastapi_endpoint(method="GET")
    def service_info(self) -> dict:
        """Get service information"""
        return {
            "service_name": cover_image_config.service_name,
            "gpu_type": cover_image_config.gpu_type.value,
            "scaledown_window": cover_image_config.scaledown_window,
            "max_runtime": cover_image_config.max_runtime_seconds,
            "model_id": getattr(self, 'model_id', 'stabilityai/sdxl-turbo'),
            "supported_styles": [
                "album cover art", "vintage", "modern", "abstract", 
                "photorealistic", "illustration", "grunge", "electronic"
            ],
            "image_resolution": "512x512",
            "supported_formats": ["PNG"],
            "batch_limit": 10,
            "storage_mode": "Local" if not settings.use_s3_storage else "S3",
            "storage_path": settings.local_storage_dir if not settings.use_s3_storage else settings.s3_bucket_name
        }

    @modal.fastapi_endpoint(method="GET")
    def get_available_styles(self) -> dict:
        """Get available style presets with descriptions"""
        styles = {
            "album cover art": {
                "description": "Professional album cover art with high quality and detailed composition",
                "best_for": "General music albums, professional releases"
            },
            "vintage": {
                "description": "Retro style with aged paper texture and classic design elements",
                "best_for": "Classic rock, jazz, blues, nostalgic themes"
            },
            "modern": {
                "description": "Clean, minimalist design with contemporary aesthetics",
                "best_for": "Electronic, pop, indie music"
            },
            "abstract": {
                "description": "Artistic composition with geometric shapes and abstract elements",
                "best_for": "Experimental, ambient, avant-garde music"
            },
            "photorealistic": {
                "description": "Detailed photography style with studio lighting",
                "best_for": "Portrait-based albums, realistic scenes"
            },
            "illustration": {
                "description": "Digital illustration with vibrant colors and artistic style",
                "best_for": "Concept albums, fantasy themes, animated styles"
            },
            "grunge": {
                "description": "Distressed texture with alternative rock aesthetic",
                "best_for": "Rock, punk, alternative music"
            },
            "electronic": {
                "description": "Futuristic design with neon colors and electronic aesthetics",
                "best_for": "EDM, techno, synthwave, electronic music"
            }
        }
        
        return {
            "available_styles": styles,
            "custom_style_supported": True,
            "style_mixing_supported": True
        }
    
    @modal.fastapi_endpoint(method="GET")
    def serve_file(self, file_key: str):
        """Serve image file for local storage mode"""
        if settings.use_s3_storage:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400, 
                detail="File serving is only available for local storage mode. Use S3 URLs for S3 storage."
            )
        
        # Simple file serving for local mode
        import os
        from fastapi import HTTPException
        from fastapi.responses import FileResponse
        
        file_path = os.path.join(settings.local_storage_dir, file_key)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(file_path)
    
    @modal.fastapi_endpoint(method="GET")
    def download_image(self, file_key: str) -> dict:
        """Download image as base64 data (similar to audio download in main.py)"""
        import os
        import base64
        from fastapi import HTTPException
        
        # Find the file in the storage directory
        storage_dir = getattr(self, 'file_manager', None)
        if storage_dir and hasattr(storage_dir, 'local_storage_dir'):
            storage_path = storage_dir.local_storage_dir
        else:
            storage_path = settings.local_storage_dir
        
        file_path = os.path.join(storage_path, file_key)
        
        # If file_key is a full path, use it directly
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
    
    @modal.fastapi_endpoint(method="GET")
    def get_storage_stats(self) -> dict:
        """Get storage statistics for this service"""
        import os
        
        if settings.use_s3_storage:
            return {"storage_mode": "S3", "bucket": settings.s3_bucket_name}
        
        # Local storage stats
        storage_dir = settings.local_storage_dir
        if not os.path.exists(storage_dir):
            return {"storage_mode": "Local", "directory": storage_dir, "files": 0, "total_size_mb": 0}
        
        files = [f for f in os.listdir(storage_dir) if f.endswith('.png')]
        total_size = sum(os.path.getsize(os.path.join(storage_dir, f)) for f in files)
        
        return {
            "storage_mode": "Local",
            "directory": storage_dir,
            "files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }


@app.local_entrypoint()
def test_cover_generation():
    """Test cover image generation service with comprehensive endpoint testing"""
    import requests
    import time
    import os
    import base64
    
    server = CoverImageGenServer()
    
    print("=== Cover Image Generation Service Test ===")
    print(f"Configuration: {cover_image_config.service_name} on {cover_image_config.gpu_type.value}")
    print(f"Scaledown window: {cover_image_config.scaledown_window}s")
    print(f"Max runtime: {cover_image_config.max_runtime_seconds}s")
    print(f"Storage mode: Local ({settings.local_storage_dir})")
    print("-" * 50)
    
    # Wait for service to initialize
    print("Waiting for service to initialize...")
    time.sleep(5)
    
    try:
        # Test health check first
        print("=== Testing Health Check ===")
        health_url = server.health_check.get_web_url()
        print(f"Health check URL: {health_url}")
        health_response = requests.get(health_url, timeout=30)
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"Health status: {health_data['status']}")
        print(f"Model loaded: {health_data['model_loaded']}")
        
        if not health_data.get('model_loaded', False):
            print("WARNING: Model not loaded yet, waiting...")
            time.sleep(10)
        
        # Test service info endpoint
        print("\n=== Testing Service Info ===")
        info_url = server.service_info.get_web_url()
        info_response = requests.get(info_url, timeout=30)
        info_response.raise_for_status()
        service_info = info_response.json()
        print(f"Service: {service_info['service_name']}")
        print(f"GPU Type: {service_info['gpu_type']}")
        print(f"Storage: {service_info['storage_mode']} ({service_info['storage_path']})")
        print(f"Supported styles: {len(service_info['supported_styles'])} styles")
        
        # Test available styles endpoint
        print("\n=== Testing Available Styles ===")
        styles_url = server.get_available_styles.get_web_url()
        styles_response = requests.get(styles_url, timeout=30)
        styles_response.raise_for_status()
        styles_data = styles_response.json()
        print(f"Available styles: {list(styles_data['available_styles'].keys())}")
        
        # Test single image generation with different styles
        print("\n=== Testing Single Image Generation ===")
        test_cases = [
            ("electronic", "electronic music album cover"),
            ("vintage", "vintage rock album"),
            ("modern", "modern pop album")
        ]
        
        generated_files = []
        
        for style, prompt in test_cases:
            try:
                print(f"Generating {style} style image...")
                request = CoverImageGenerationRequest(
                    prompt=prompt,
                    style=style,
                    width=512,
                    height=512,
                    num_inference_steps=2  # Fast generation for testing
                )
                
                endpoint_url = server.generate_cover_image.get_web_url()
                response = requests.post(endpoint_url, json=request.model_dump(), timeout=120)
                response.raise_for_status()
                cover_response = CoverImageGenerationResponseEnhanced(**response.json())
                
                print(f"✓ Style '{style}' succeeded:")
                print(f"  File: {cover_response.file_path}")
                print(f"  Dimensions: {cover_response.image_dimensions}")
                print(f"  Size: {cover_response.file_size_mb}MB")
                print(f"  Generation time: {cover_response.metadata.generation_time:.2f}s")
                print(f"  Estimated cost: ${cover_response.metadata.estimated_cost:.4f}")
                
                generated_files.append(cover_response.file_path)
                
            except requests.exceptions.Timeout:
                print(f"✗ Style '{style}' timed out")
            except Exception as e:
                print(f"✗ Style '{style}' failed: {e}")
        
        # Test storage stats
        print("\n=== Testing Storage Stats ===")
        stats_url = server.get_storage_stats.get_web_url()
        stats_response = requests.get(stats_url, timeout=30)
        stats_response.raise_for_status()
        storage_stats = stats_response.json()
        print(f"Storage mode: {storage_stats['storage_mode']}")
        print(f"Files generated: {storage_stats.get('files', 0)}")
        print(f"Total size: {storage_stats.get('total_size_mb', 0)}MB")
        
        # Download and save images locally (similar to main.py audio saving)
        print("\n=== Downloading Images to Local Files ===")
        local_files = []
        
        for i, file_path in enumerate(generated_files):
            try:
                # Extract filename from the service path
                filename = os.path.basename(file_path)
                local_filename = f"cover_image_{i+1}_{filename}"
                
                # Use the download_image endpoint to get base64 data
                download_url = server.download_image.get_web_url()
                download_response = requests.get(f"{download_url}?file_key={filename}", timeout=30)
                download_response.raise_for_status()
                
                # Parse the JSON response and decode base64 data
                download_data = download_response.json()
                image_bytes = base64.b64decode(download_data["image_data"])
                
                # Save to local file (similar to main.py audio saving)
                with open(local_filename, "wb") as f:
                    f.write(image_bytes)
                
                file_size = len(image_bytes) / (1024 * 1024)
                print(f"✓ Downloaded {local_filename} ({file_size:.2f}MB)")
                local_files.append(local_filename)
                
            except Exception as e:
                print(f"✗ Failed to download {file_path}: {e}")
        
        # Verify local files exist
        print("\n=== Verifying Downloaded Files ===")
        for local_file in local_files:
            if os.path.exists(local_file):
                file_size = os.path.getsize(local_file) / (1024 * 1024)
                print(f"✓ {local_file} saved locally ({file_size:.2f}MB)")
            else:
                print(f"✗ {local_file} not found locally")
        
        print(f"\n=== Test Summary ===")
        print(f"Generated {len(generated_files)} images successfully")
        print(f"Downloaded {len(local_files)} images to local directory")
        print(f"Service storage: {settings.local_storage_dir}")
        print(f"Local files: {', '.join(local_files) if local_files else 'None'}")
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_cover_generation()