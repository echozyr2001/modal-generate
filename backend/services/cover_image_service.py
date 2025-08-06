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
from shared.modal_images import image_generation_image, model_volumes, secrets
from shared.service_base import ImageGenerationService
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
    memory_gb=16  # Sufficient for SDXL-Turbo
)

app = modal.App("cover-image-generator")


@app.cls(
    image=image_generation_image,
    gpu=cover_image_config.gpu_type.value,
    volumes={
        "/.cache/huggingface": model_volumes["huggingface_cache"],
        "/.cache/diffusers": model_volumes["diffusers_cache"]
    },
    secrets=[secrets["music_gen"], secrets["aws_credentials"]],
    scaledown_window=cover_image_config.scaledown_window,
    timeout=cover_image_config.max_runtime_seconds
)
class CoverImageGenServer(ImageGenerationService):
    def __init__(self):
        # Initialize FileManager for storage handling
        file_manager = FileManager(
            use_s3=settings.use_s3_storage,
            local_storage_dir=settings.local_storage_dir
        )
        super().__init__(cover_image_config, file_manager)
        self.pipeline = None
        self.model_id = "stabilityai/sdxl-turbo"  # SDXL-Turbo for fast generation

    @modal.enter()
    def load_model(self):
        """Load SDXL-Turbo model optimized for fast image generation"""
        from diffusers import AutoPipelineForText2Image
        
        logger.info(f"Loading image generation model: {self.model_id}")
        
        try:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir="/.cache/diffusers"
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
            image = self.pipeline(**generation_params).images[0]
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
        """Generate image with timeout protection"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Generation timed out after {self.config.max_runtime_seconds} seconds")
        
        # Set up timeout signal (Unix systems only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.config.max_runtime_seconds)
        
        try:
            result = self.generate(request)
            return result
        finally:
            # Clear timeout signal
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

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
        return super().health_check()

    @modal.fastapi_endpoint(method="GET")
    def service_info(self) -> dict:
        """Get service information"""
        info = super().get_service_info()
        info.update({
            "model_id": self.model_id,
            "supported_styles": [
                "album cover art", "vintage", "modern", "abstract", 
                "photorealistic", "illustration", "grunge", "electronic"
            ],
            "image_resolution": "512x512",
            "supported_formats": ["PNG"],
            "batch_limit": 10
        })
        return info

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


@app.local_entrypoint()
def test_cover_generation():
    """Test cover image generation service with comprehensive endpoint testing"""
    import requests
    
    server = CoverImageGenServer()
    
    try:
        # Test service info endpoint
        print("=== Testing Service Info ===")
        info_url = server.service_info.get_web_url()
        info_response = requests.get(info_url)
        info_response.raise_for_status()
        service_info = info_response.json()
        print(f"Service: {service_info['service_name']}")
        print(f"GPU Type: {service_info['gpu_type']}")
        print(f"Supported styles: {service_info['supported_styles']}")
        
        # Test available styles endpoint
        print("\n=== Testing Available Styles ===")
        styles_url = server.get_available_styles.get_web_url()
        styles_response = requests.get(styles_url)
        styles_response.raise_for_status()
        styles_data = styles_response.json()
        print(f"Available styles: {list(styles_data['available_styles'].keys())}")
        
        # Test single image generation with different styles
        print("\n=== Testing Single Image Generation ===")
        test_styles = ["electronic", "vintage", "modern"]
        
        for style in test_styles:
            request = CoverImageGenerationRequest(
                prompt=f"{style} music album",
                style=style
            )
            
            endpoint_url = server.generate_cover_image.get_web_url()
            response = requests.post(endpoint_url, json=request.model_dump())
            response.raise_for_status()
            cover_response = CoverImageGenerationResponseEnhanced(**response.json())
            
            print(f"Style '{style}':")
            print(f"  File: {cover_response.file_path}")
            print(f"  Dimensions: {cover_response.image_dimensions}")
            print(f"  Size: {cover_response.file_size_mb}MB")
            print(f"  Generation time: {cover_response.metadata.generation_time:.2f}s")
            print(f"  Estimated cost: ${cover_response.metadata.estimated_cost:.4f}")
        
        # Test batch generation with enhanced response
        print("\n=== Testing Batch Generation ===")
        batch_requests = [
            CoverImageGenerationRequest(prompt="rock music, electric guitar", style="grunge"),
            CoverImageGenerationRequest(prompt="classical symphony", style="vintage"),
            CoverImageGenerationRequest(prompt="ambient electronic", style="abstract"),
        ]
        
        batch_endpoint_url = server.generate_cover_image_batch.get_web_url()
        response = requests.post(batch_endpoint_url, json=[req.model_dump() for req in batch_requests])
        response.raise_for_status()
        batch_result = response.json()
        
        stats = batch_result["batch_statistics"]
        print(f"Batch Results:")
        print(f"  Total requested: {stats['total_requested']}")
        print(f"  Successful: {stats['successful_count']}")
        print(f"  Failed: {stats['failed_count']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Total batch time: {stats['total_batch_time']:.2f}s")
        print(f"  Average generation time: {stats['average_generation_time']:.2f}s")
        print(f"  Total file size: {stats['total_file_size_mb']:.2f}MB")
        
        for i, img in enumerate(batch_result["successful_images"]):
            print(f"  Image {i+1}: {img['file_path']} ({img['metadata']['generation_time']:.2f}s)")
            
        # Test health check
        print("\n=== Testing Health Check ===")
        health_url = server.health_check.get_web_url()
        health_response = requests.get(health_url)
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"Health status: {health_data['status']}")
        print(f"Model loaded: {health_data['model_loaded']}")
        
        print("\n=== All Tests Passed! ===")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_cover_generation()