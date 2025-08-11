"""
Cover Image Generation Service
Provides cover image generation using SDXL-Turbo.
Extracted from main.py to provide dedicated image generation capabilities.
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
from pydantic import BaseModel, Field
from typing import Optional
from shared.deployment import image_generation_image, hf_volume, music_gen_secrets
from shared.config import settings
from shared.base_service import create_service_app, ServiceMixin

logger = logging.getLogger(__name__)


class CoverImageDirectDownloadResponse(BaseModel):
    """Response for direct download mode with embedded file data"""
    file_data: str = Field(..., description="Base64 encoded image data")
    file_name: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File type (image)")
    file_extension: str = Field(..., description="File extension")
    image_dimensions: tuple = Field(..., description="Image dimensions (width, height)")
    file_size_mb: float = Field(..., description="File size in MB")
    metadata: GenerationMetadata

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
class CoverImageGenServer(ServiceMixin):
    """Cover image generation server - extracted from main.py SDXL-Turbo functionality"""
    
    @modal.enter()
    def load_model(self):
        """Load SDXL-Turbo model for image generation"""
        from diffusers import AutoPipelineForText2Image
        
        # Initialize service components using mixin
        self.init_service_components(cover_image_config)
        self.model_id = "stabilityai/sdxl-turbo"
        
        logger.info(f"Loading image generation model: {self.model_id}")
        
        try:
            # Load Stable Diffusion Model (same as main.py)
            self.image_pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16, 
                variant="fp16", 
                cache_dir="/.cache/huggingface"
            )
            self.image_pipe.to("cuda")
            
            self._model_loaded = True
            logger.info("SDXL-Turbo model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load image generation model: {e}")
            self._model_loaded = False
            raise
    
    def generate_thumbnail(self, prompt: str, num_inference_steps: int = 2, 
                          guidance_scale: float = 0.0) -> Image.Image:
        """
        Generate thumbnail image - extracted from main.py
        """
        thumbnail_prompt = f"{prompt}, album cover art"
        image = self.image_pipe(
            prompt=thumbnail_prompt, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale
        ).images[0]
        return image
    
    def generate_and_upload_to_s3(self, prompt: str) -> str:
        """
        Generate image and upload to S3 - extracted from main.py
        Returns S3 key of uploaded image
        """
        import boto3
        
        # Generate thumbnail (same as main.py)
        image = self.generate_thumbnail(prompt)
        
        # Save and upload (same as main.py)
        s3_client = boto3.client("s3")
        bucket_name = os.environ["S3_BUCKET_NAME"]
        
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        image_output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
        image.save(image_output_path)

        image_s3_key = f"{uuid.uuid4()}.png"
        s3_client.upload_file(image_output_path, bucket_name, image_s3_key)
        os.remove(image_output_path)
        
        return image_s3_key
    
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
        
        # Generation parameters
        generation_params = {
            "prompt": full_prompt,
            "num_inference_steps": request.num_inference_steps or 4,
            "guidance_scale": 0.0,  # SDXL-Turbo works best with guidance_scale=0
            "width": request.width or 512,
            "height": request.height or 512,
        }
        
        # Add seed if provided
        if request.seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(request.seed)
            generation_params["generator"] = generator
        
        try:
            logger.info(f"[{operation_id}] Starting generation with steps={generation_params['num_inference_steps']}")
            image = self.image_pipe(**generation_params).images[0]
            logger.info(f"[{operation_id}] Image generation successful")
        except Exception as e:
            logger.error(f"[{operation_id}] Generation failed: {e}")
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
    
    def save_image(self, image: Image.Image, operation_id: str) -> str:
        """Save image using FileManager"""
        try:
            # Create filename
            filename = f"{operation_id}.png"
            
            # Save to temporary file first
            temp_path = f"/tmp/{filename}"
            image.save(temp_path, "PNG")
            
            # In direct download mode, return temp path for later processing
            if self.file_manager.get_storage_mode() == "direct_download":
                logger.info(f"Image prepared for direct download: {temp_path}")
                return temp_path
            else:
                # Save using FileManager for S3/local storage
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
    
    # Service management methods
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": self.config.service_name,
            "gpu_type": self.config.gpu_type.value,
            "model_id": self.model_id,
            "supported_styles": list(self.get_style_presets().keys()),
            "image_resolution": "512x512",
            "supported_formats": ["PNG"],
            "scaledown_window": self.config.scaledown_window,
            "max_runtime": self.config.max_runtime_seconds,
            "storage_mode": self.file_manager.get_storage_mode()
        }
    
    # All common service methods are now provided by ServiceMixin
    
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
    
    # API Endpoints
    @modal.fastapi_endpoint(method="POST")
    def generate_cover_image(self, request: CoverImageGenerationRequest):
        """Generate single cover image - returns different response based on storage mode"""
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
            
            # Return different response based on storage mode
            if self.file_manager.get_storage_mode() == "direct_download":
                # Prepare file for direct download
                download_data = self.file_manager.prepare_direct_download_response(result["file_path"])
                
                response = CoverImageDirectDownloadResponse(
                    file_data=download_data["file_data"],
                    file_name=download_data["file_name"],
                    file_size=download_data["file_size"],
                    file_type=download_data["file_type"],
                    file_extension=download_data["file_extension"],
                    image_dimensions=result["image_dimensions"],
                    file_size_mb=result["file_size_mb"],
                    metadata=metadata
                )
                
                logger.info(f"[{operation_id}] Cover image prepared for direct download")
                return response
            else:
                # Traditional response with file path
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
    
    # Health check removed to save endpoints - use generate endpoint for status
    
    # Endpoints merged to save quota
    
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


@app.local_entrypoint()
def test_cover_image_service():
    """Test the cover image service"""
    import requests
    import time
    
    server = CoverImageGenServer()
    
    print("=== Testing Cover Image Service ===")
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
        
        # Download the generated image to local directory
        try:
            print("\nDownloading generated image...")
            
            # Create local output directory
            local_output_dir = "./test_outputs"
            os.makedirs(local_output_dir, exist_ok=True)
            
            # Use the download endpoint to get the image
            download_url = server.download_image.get_web_url()
            download_response = requests.get(
                download_url, 
                params={"file_path": result['file_path']},
                timeout=30
            )
            download_response.raise_for_status()
            download_data = download_response.json()
            
            # Decode base64 and save locally
            import base64
            image_bytes = base64.b64decode(download_data['image_data'])
            
            remote_filename = os.path.basename(result['file_path'])
            local_file_path = os.path.join(local_output_dir, f"test_cover_{remote_filename}")
            
            with open(local_file_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"  ✓ Image downloaded: {local_file_path}")
            print(f"  ✓ Image size: {download_data['width']}x{download_data['height']}")
            print(f"  ✓ Format: {download_data['format'].upper()}")
            print(f"  ✓ File size: {download_data['file_size_bytes']} bytes")
            
            # Verify the downloaded image
            try:
                from PIL import Image
                with Image.open(local_file_path) as img:
                    print(f"  ✓ Image verified locally: {img.size}, mode: {img.mode}")
            except Exception as e:
                print(f"  ⚠ Could not verify downloaded image: {e}")
                
        except Exception as e:
            print(f"  ⚠ Failed to download image: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        print("Cover image service test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cover_image_service()