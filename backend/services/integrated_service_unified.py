"""
Unified Integrated Service
Orchestrates calls to other services for complete music generation workflow.
"""

import modal
import logging
import time
import asyncio
import httpx
import uuid
from typing import Dict, Any, Optional
from fastapi import HTTPException

from shared.models import (
    GenerateFromDescriptionRequest,
    GenerateMusicResponse,
    LyricsGenerationRequest,
    PromptGenerationRequest,
    CategoryGenerationRequest,
    CoverImageGenerationRequest,
    ServiceConfig,
    GPUType
)
from shared.deployment import base_image, music_gen_secrets
from shared.config import settings
from shared.base_service import create_service_app

logger = logging.getLogger(__name__)

# Get service configuration
integration_config = ServiceConfig(
    service_name="integrated-music-generator",
    **settings.get_service_config("integration")
)

# Create Modal app
app, app_config = create_service_app(
    "integrated-music-generator",
    integration_config,
    base_image,
    secrets=[music_gen_secrets]
)

# Override CPU configuration for integration service
app_config["cpu"] = 2.0
if "gpu" in app_config:
    del app_config["gpu"]  # Integration service doesn't need GPU


@app.cls(**app_config)
class IntegratedMusicGenServer:
    """Unified integrated music generation server"""
    
    @modal.enter()
    def load_model(self):
        """Initialize service (no model loading required)"""
        from shared.monitoring import CostMonitor, TimeoutManager
        from shared.storage import FileManager
        
        # Initialize service components (since we can't use __init__)
        self.config = integration_config
        self.cost_monitor = CostMonitor()
        self.timeout_manager = TimeoutManager(integration_config.max_runtime_seconds)
        self.file_manager = FileManager()
        self._model_loaded = False
        self.service_urls = {
            "lyrics": settings.lyrics_service_url,
            "music": settings.music_service_url,
            "image": settings.image_service_url
        }
        
        logger.info("Initializing integrated service...")
        self._model_loaded = True
        logger.info("Integrated service initialized successfully")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": self.config.service_name,
            "gpu_type": self.config.gpu_type.value,
            "service_type": "orchestration",
            "supported_workflows": ["full_music_generation"],
            "dependent_services": ["lyrics", "music", "image"],
            "service_urls": self.service_urls,
            "scaledown_window": self.config.scaledown_window,
            "max_runtime": self.config.max_runtime_seconds
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
    
    async def call_service(self, service_name: str, endpoint: str, data: dict, 
                          timeout: int = 300) -> dict:
        """Call external service with error handling"""
        if not self.service_urls.get(service_name):
            raise ValueError(f"Service URL not configured for {service_name}")
        
        url = f"{self.service_urls[service_name]}/{endpoint}"
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                logger.info(f"Calling {service_name} service: {url}")
                response = await client.post(url, json=data)
                response.raise_for_status()
                result = response.json()
                logger.info(f"✓ {service_name} service call successful")
                return result
            except httpx.TimeoutException:
                logger.error(f"❌ {service_name} service timeout")
                raise TimeoutError(f"{service_name} service timed out")
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ {service_name} service error: {e.response.status_code}")
                raise RuntimeError(f"{service_name} service error: {e.response.status_code}")
            except Exception as e:
                logger.error(f"❌ {service_name} service call failed: {e}")
                raise RuntimeError(f"{service_name} service failed: {str(e)}")
    
    async def generate_full_workflow(self, request: GenerateFromDescriptionRequest, 
                                   operation_id: str) -> Dict[str, Any]:
        """Execute full music generation workflow"""
        logger.info(f"[{operation_id}] Starting full workflow: {request.full_described_song[:100]}...")
        
        workflow_results = {
            "lyrics": None,
            "prompt": None,
            "categories": None,
            "music": None,
            "cover_image": None,
            "errors": []
        }
        
        try:
            # Step 1: Generate lyrics
            logger.info(f"[{operation_id}] Step 1: Generating lyrics...")
            lyrics_request = LyricsGenerationRequest(description=request.full_described_song)
            lyrics_result = await self.call_service(
                "lyrics", 
                "generate_lyrics", 
                lyrics_request.model_dump()
            )
            workflow_results["lyrics"] = lyrics_result.get("lyrics", "")
            logger.info(f"[{operation_id}] ✓ Lyrics generated")
            
        except Exception as e:
            error_msg = f"Lyrics generation failed: {str(e)}"
            logger.error(f"[{operation_id}] {error_msg}")
            workflow_results["errors"].append(error_msg)
            workflow_results["lyrics"] = "[instrumental]"  # Fallback
        
        try:
            # Step 2: Generate music prompt
            logger.info(f"[{operation_id}] Step 2: Generating music prompt...")
            prompt_request = PromptGenerationRequest(description=request.full_described_song)
            prompt_result = await self.call_service(
                "lyrics",  # Prompt generation is handled by lyrics service
                "generate_prompt", 
                prompt_request.model_dump()
            )
            workflow_results["prompt"] = prompt_result.get("prompt", "")
            logger.info(f"[{operation_id}] ✓ Music prompt generated")
            
        except Exception as e:
            error_msg = f"Prompt generation failed: {str(e)}"
            logger.error(f"[{operation_id}] {error_msg}")
            workflow_results["errors"].append(error_msg)
            workflow_results["prompt"] = "upbeat music"  # Fallback
        
        try:
            # Step 3: Generate categories
            logger.info(f"[{operation_id}] Step 3: Generating categories...")
            categories_request = CategoryGenerationRequest(description=request.full_described_song)
            categories_result = await self.call_service(
                "lyrics",  # Category generation is handled by lyrics service
                "generate_categories", 
                categories_request.model_dump()
            )
            workflow_results["categories"] = categories_result.get("categories", [])
            logger.info(f"[{operation_id}] ✓ Categories generated")
            
        except Exception as e:
            error_msg = f"Category generation failed: {str(e)}"
            logger.error(f"[{operation_id}] {error_msg}")
            workflow_results["errors"].append(error_msg)
            workflow_results["categories"] = ["Unknown"]  # Fallback
        
        # Check if we have minimum requirements for music generation
        if not workflow_results["prompt"] or not workflow_results["lyrics"]:
            raise RuntimeError("Failed to generate minimum requirements for music generation")
        
        try:
            # Step 4: Generate music
            logger.info(f"[{operation_id}] Step 4: Generating music...")
            music_request = {
                "prompt": workflow_results["prompt"],
                "lyrics": workflow_results["lyrics"],
                "audio_duration": request.audio_duration,
                "inference_steps": request.inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "instrumental": request.instrumental
            }
            music_result = await self.call_service(
                "music", 
                "generate_music", 
                music_request,
                timeout=600  # 10 minutes for music generation
            )
            workflow_results["music"] = music_result
            logger.info(f"[{operation_id}] ✓ Music generated")
            
        except Exception as e:
            error_msg = f"Music generation failed: {str(e)}"
            logger.error(f"[{operation_id}] {error_msg}")
            workflow_results["errors"].append(error_msg)
            raise RuntimeError(error_msg)  # Music is critical, fail if it doesn't work
        
        try:
            # Step 5: Generate cover image (optional)
            logger.info(f"[{operation_id}] Step 5: Generating cover image...")
            image_request = CoverImageGenerationRequest(
                prompt=f"album cover for {request.full_described_song}",
                style="album cover art",
                width=512,
                height=512
            )
            image_result = await self.call_service(
                "image", 
                "generate_cover_image", 
                image_request.model_dump()
            )
            workflow_results["cover_image"] = image_result.get("file_path", "")
            logger.info(f"[{operation_id}] ✓ Cover image generated")
            
        except Exception as e:
            error_msg = f"Cover image generation failed: {str(e)}"
            logger.error(f"[{operation_id}] {error_msg}")
            workflow_results["errors"].append(error_msg)
            workflow_results["cover_image"] = None  # Cover image is optional
        
        return workflow_results
    
    @modal.fastapi_endpoint(method="POST")
    def generate_from_description(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponse:
        """Generate complete music package from description"""
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Full music generation: {request.full_described_song[:100]}...")
        
        # Start monitoring
        self.start_operation(operation_id, "full_workflow")
        
        try:
            # Execute workflow
            workflow_results = asyncio.run(
                self.generate_full_workflow(request, operation_id)
            )
            
            # End monitoring
            self.end_operation(operation_id, success=True)
            
            # Create response
            response = GenerateMusicResponse(
                audio_file_path=workflow_results["music"].get("audio_data", "") if workflow_results["music"] else "",
                cover_image_file_path=workflow_results["cover_image"],
                categories=workflow_results["categories"] or ["Unknown"],
                storage_mode="base64" if workflow_results["music"] and "audio_data" in workflow_results["music"] else "local",
                generation_metadata=self.create_metadata(operation_id, "Integrated Workflow", start_time),
                service_status={
                    "lyrics": workflow_results["lyrics"] is not None,
                    "prompt": workflow_results["prompt"] is not None,
                    "categories": bool(workflow_results["categories"]),
                    "music": workflow_results["music"] is not None,
                    "cover_image": workflow_results["cover_image"] is not None
                },
                errors=workflow_results["errors"] if workflow_results["errors"] else None,
                correlation_id=operation_id
            )
            
            logger.info(f"[{operation_id}] Full workflow completed successfully")
            return response
            
        except Exception as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Full workflow failed: {e}")
            raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")
    
    @modal.fastapi_endpoint(method="POST")
    def generate_simple(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponse:
        """Simple generation for testing (placeholder implementation)"""
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Simple generation (placeholder): {request.full_described_song[:100]}...")
        
        # This is a placeholder implementation for testing
        return GenerateMusicResponse(
            audio_file_path="placeholder.wav",
            cover_image_file_path="placeholder.png",
            categories=["placeholder"],
            storage_mode="local",
            generation_metadata=self.create_metadata(operation_id, "Placeholder", start_time),
            correlation_id=operation_id
        )
    
    @modal.fastapi_endpoint(method="GET")
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        health = super().health_check()
        
        # Add service connectivity status
        health["dependent_services"] = {}
        for service_name, url in self.service_urls.items():
            health["dependent_services"][service_name] = {
                "configured": bool(url),
                "url": url if url else "not_configured"
            }
        
        return health
    
    @modal.fastapi_endpoint(method="GET")
    def service_info(self) -> Dict[str, Any]:
        """Service information endpoint"""
        return self.get_service_info()
    
    @modal.fastapi_endpoint(method="GET")
    def workflow_status(self, operation_id: str) -> Dict[str, Any]:
        """Get workflow status (placeholder for future implementation)"""
        return {
            "operation_id": operation_id,
            "status": "completed",  # Placeholder
            "message": "Workflow status tracking not yet implemented"
        }


@app.local_entrypoint()
def test_integrated_service():
    """Test the unified integrated service"""
    import requests
    import time
    
    server = IntegratedMusicGenServer()
    
    print("=== Testing Unified Integrated Service ===")
    print(f"Service: {integration_config.service_name}")
    print(f"GPU: {integration_config.gpu_type.value}")
    print("-" * 50)
    
    # Wait for initialization
    print("Waiting for service initialization...")
    time.sleep(5)
    
    try:
        # Test health check
        health_url = server.health_check.get_web_url()
        print(f"Health check: {health_url}")
        
        response = requests.get(health_url, timeout=30)
        response.raise_for_status()
        health = response.json()
        
        print(f"✓ Health check: {health['status']}")
        print(f"  Service ready: {health['model_loaded']}")
        print(f"  Dependent services:")
        for service, info in health.get("dependent_services", {}).items():
            status = "✓" if info["configured"] else "❌"
            print(f"    {status} {service}: {info['url']}")
        
        # Test service info
        print("\nTesting service info...")
        info_url = server.service_info.get_web_url()
        info_response = requests.get(info_url, timeout=30)
        info_response.raise_for_status()
        service_info = info_response.json()
        
        print(f"✓ Service info retrieved")
        print(f"  Supported workflows: {service_info['supported_workflows']}")
        print(f"  Dependent services: {service_info['dependent_services']}")
        
        # Test simple generation (placeholder)
        print("\nTesting simple generation (placeholder)...")
        simple_request = GenerateFromDescriptionRequest(
            full_described_song="a happy upbeat electronic song about dancing"
        )
        
        simple_url = server.generate_simple.get_web_url()
        simple_response = requests.post(simple_url, json=simple_request.model_dump(), timeout=60)
        simple_response.raise_for_status()
        simple_result = simple_response.json()
        
        print(f"✓ Simple generation successful")
        print(f"  Audio file: {simple_result['audio_file_path']}")
        print(f"  Cover image: {simple_result['cover_image_file_path']}")
        print(f"  Categories: {simple_result['categories']}")
        print(f"  Storage mode: {simple_result['storage_mode']}")
        
        print("\n" + "="*50)
        print("Integrated service test completed!")
        print("Note: Full workflow testing requires other services to be running")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_integrated_service()