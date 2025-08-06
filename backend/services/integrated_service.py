import modal
import logging
import asyncio
import time
import httpx
from typing import Optional, Dict, Any
from shared.models import (
    GenerateFromDescriptionRequest,
    GenerateWithCustomLyricsRequest,
    GenerateWithDescribedLyricsRequest,
    GenerateMusicResponseS3,
    GenerateMusicResponseUnified,
    ServiceConfig,
    GPUType,
    GenerationMetadata
)
from pydantic import BaseModel
from typing import List
from shared.modal_config import base_image, music_gen_secrets
from shared.config import settings
from shared.service_base import OrchestrationService
from shared.utils import FileManager
from shared.service_communication import (
    service_comm_manager, 
    CircuitBreakerConfig,
    generate_correlation_id
)

logger = logging.getLogger(__name__)

app = modal.App("music-generator-integrated")





@app.cls(
    image=base_image,
    secrets=[music_gen_secrets],
    scaledown_window=15,  # Fast scaledown for CPU-only orchestration
    cpu=2.0  # CPU-only orchestration service
)
class IntegratedMusicGenServer:
    @modal.enter()
    def setup(self):
        """Initialize service URLs and configuration"""
        # Service discovery and URL configuration management
        self.service_urls = {
            "lyrics": settings.lyrics_service_url or "https://your-lyrics-service-url",
            "music": settings.music_service_url or "https://your-music-service-url", 
            "image": settings.image_service_url or "https://your-image-service-url"
        }
        
        # Initialize file manager
        self.file_manager = FileManager(
            use_s3=settings.use_s3_storage,
            local_storage_dir=settings.local_storage_dir
        )
        
        # Configure circuit breakers for each service
        self._setup_circuit_breakers()
        
        logger.info(f"Integrated service initialized with URLs: {self.service_urls}")
    
    def operation_context(self, operation_type: str = "generation"):
        """Simple context manager for operation tracking"""
        import contextlib
        import uuid
        
        @contextlib.contextmanager
        def _context():
            operation_id = str(uuid.uuid4())
            start_time = time.time()
            logger.info(f"Starting operation {operation_type} [id: {operation_id}]")
            try:
                yield operation_id
            finally:
                duration = time.time() - start_time
                logger.info(f"Completed operation {operation_type} [id: {operation_id}] in {duration:.2f}s")
        
        return _context()
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        return {
            "status": "healthy",
            "service": "integrated-music-gen",
            "timestamp": time.time(),
            "service_urls": self.service_urls
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": "integrated-music-gen",
            "gpu_type": "CPU",
            "scaledown_window": 15,
            "max_runtime": 900
        }
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for each service with appropriate configurations"""
        # Lyrics service - more tolerant (text generation is usually reliable)
        service_comm_manager.get_or_create_circuit_breaker(
            "lyrics", 
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2,
                timeout_seconds=60
            )
        )
        
        # Music service - less tolerant (expensive GPU operations)
        service_comm_manager.get_or_create_circuit_breaker(
            "music",
            CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=120,
                success_threshold=1,
                timeout_seconds=600
            )
        )
        
        # Image service - moderate tolerance
        service_comm_manager.get_or_create_circuit_breaker(
            "image",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60,
                success_threshold=2,
                timeout_seconds=120
            )
        )

    async def call_lyrics_service(self, endpoint: str, data: dict, 
                                 timeout: int = 60, correlation_id: Optional[str] = None) -> dict:
        """Call lyrics service with timeout handling and retries"""
        return await service_comm_manager.call_service_with_retry(
            service_url=self.service_urls["lyrics"],
            endpoint=f"/{endpoint}",
            data=data,
            service_name="lyrics",
            timeout=timeout,
            max_retries=2,
            correlation_id=correlation_id
        )

    async def call_cover_service(self, data: dict, timeout: int = 120, 
                               correlation_id: Optional[str] = None) -> dict:
        """Call cover image generation service"""
        return await service_comm_manager.call_service_with_retry(
            service_url=self.service_urls["image"],
            endpoint="/generate_cover_image",
            data=data,
            service_name="image",
            timeout=timeout,
            max_retries=2,
            correlation_id=correlation_id
        )

    async def call_music_service(self, data: dict, timeout: int = 600,
                               correlation_id: Optional[str] = None) -> dict:
        """Call music generation service with extended timeout"""
        return await service_comm_manager.call_service_with_retry(
            service_url=self.service_urls["music"],
            endpoint="/generate_music_to_storage",
            data=data,
            service_name="music",
            timeout=timeout,
            max_retries=1,
            correlation_id=correlation_id
        )

    async def generate_complete_music(
        self,
        prompt: str,
        lyrics: str,
        description_for_categorization: str,
        audio_duration: float = 180.0,
        seed: int = -1,
        guidance_scale: float = 15.0,
        infer_step: int = 60,
        instrumental: bool = False,
        correlation_id: Optional[str] = None
    ) -> GenerateMusicResponseS3:
        """Complete music generation workflow with service orchestration"""
        correlation_id = correlation_id or f"gen_{int(time.time())}"
        logger.info(f"Starting complete music generation [correlation_id: {correlation_id}]")
        
        # Track partial results for graceful degradation
        results = {
            "music": None,
            "cover": None,
            "categories": None,
            "errors": []
        }
        
        try:
            # Use parallel service calls with proper error handling
            service_calls = [
                {
                    "service_url": self.service_urls["music"],
                    "endpoint": "/generate_music_to_storage",
                    "data": {
                        "prompt": prompt,
                        "lyrics": lyrics,
                        "audio_duration": audio_duration,
                        "seed": seed,
                        "guidance_scale": guidance_scale,
                        "infer_step": infer_step,
                        "instrumental": instrumental
                    },
                    "service_name": "music",
                    "timeout": 600,
                    "max_retries": 1
                },
                {
                    "service_url": self.service_urls["image"],
                    "endpoint": "/generate_cover_image", 
                    "data": {
                        "prompt": prompt,
                        "style": "album cover art"
                    },
                    "service_name": "image",
                    "timeout": 120,
                    "max_retries": 2
                },
                {
                    "service_url": self.service_urls["lyrics"],
                    "endpoint": "/generate_categories",
                    "data": {
                        "description": description_for_categorization
                    },
                    "service_name": "lyrics",
                    "timeout": 60,
                    "max_retries": 2
                }
            ]
            
            # Execute all calls in parallel
            parallel_results = await service_comm_manager.call_service_batch(
                service_calls, correlation_id
            )
            
            # Process results
            service_names = ["music", "image", "lyrics"]
            for i, (service_name, result) in enumerate(zip(service_names, parallel_results)):
                if result["success"]:
                    if service_name == "lyrics":
                        results["categories"] = result["data"]
                    else:
                        results[service_name] = result["data"]
                    logger.info(f"Successfully completed {service_name} generation [correlation_id: {correlation_id}]")
                else:
                    error_msg = f"Failed {service_name} generation: {result['error']}"
                    results["errors"].append(error_msg)
                    logger.error(f"{error_msg} [correlation_id: {correlation_id}]")
                    
                    # Music generation is required - fail if it fails
                    if service_name == "music":
                        raise Exception(error_msg)
            
            # Build response with available results
            if not results["music"]:
                raise Exception("Music generation failed - cannot proceed")
            
            response = GenerateMusicResponseS3(
                s3_key=results["music"]["file_path"],
                cover_image_s3_key=results["cover"]["file_path"] if results["cover"] else "",
                categories=results["categories"]["categories"] if results["categories"] else []
            )
            
            logger.info(f"Complete music generation finished [correlation_id: {correlation_id}]")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate complete music [correlation_id: {correlation_id}]: {e}")
            raise

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def generate_from_description(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponseS3:
        """Generate complete music work from description - orchestrates all services"""
        import uuid
        correlation_id = str(uuid.uuid4())
        
        logger.info(f"Generating from description: {request.full_described_song[:100]}... [correlation_id: {correlation_id}]")
        
        with self.operation_context("generate_from_description") as operation_id:
            try:
                # 1. Generate music prompt
                prompt_request = {
                    "description": request.full_described_song
                }
                prompt_response = await self.call_lyrics_service(
                    "generate_prompt", prompt_request, correlation_id=correlation_id
                )
                prompt = prompt_response["prompt"]
                
                # 2. Generate lyrics (if not instrumental)
                lyrics = ""
                if not request.instrumental:
                    lyrics_request = {
                        "description": request.full_described_song
                    }
                    lyrics_response = await self.call_lyrics_service(
                        "generate_lyrics", lyrics_request, correlation_id=correlation_id
                    )
                    lyrics = lyrics_response["lyrics"]
                
                # 3. Generate complete music work
                return await self.generate_complete_music(
                    prompt=prompt,
                    lyrics=lyrics,
                    description_for_categorization=request.full_described_song,
                    audio_duration=request.audio_duration,
                    seed=request.seed,
                    guidance_scale=request.guidance_scale,
                    infer_step=request.infer_step,
                    instrumental=request.instrumental,
                    correlation_id=correlation_id
                )
                
            except Exception as e:
                logger.error(f"Failed to generate from description [correlation_id: {correlation_id}]: {e}")
                raise

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def generate_with_lyrics(self, request: GenerateWithCustomLyricsRequest) -> GenerateMusicResponseS3:
        """Generate music with custom lyrics"""
        import uuid
        correlation_id = str(uuid.uuid4())
        
        logger.info(f"Generating with custom lyrics, prompt: {request.prompt[:100]}... [correlation_id: {correlation_id}]")
        
        with self.operation_context("generate_with_lyrics") as operation_id:
            return await self.generate_complete_music(
                prompt=request.prompt,
                lyrics=request.lyrics,
                description_for_categorization=request.prompt,
                audio_duration=request.audio_duration,
                seed=request.seed,
                guidance_scale=request.guidance_scale,
                infer_step=request.infer_step,
                instrumental=request.instrumental,
                correlation_id=correlation_id
            )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def generate_with_described_lyrics(self, request: GenerateWithDescribedLyricsRequest) -> GenerateMusicResponseS3:
        """Generate music with described lyrics"""
        import uuid
        correlation_id = str(uuid.uuid4())
        
        logger.info(f"Generating with described lyrics: {request.described_lyrics[:100]}... [correlation_id: {correlation_id}]")
        
        with self.operation_context("generate_with_described_lyrics") as operation_id:
            try:
                # Generate lyrics (if not instrumental)
                lyrics = ""
                if not request.instrumental:
                    lyrics_request = {
                        "description": request.described_lyrics
                    }
                    lyrics_response = await self.call_lyrics_service(
                        "generate_lyrics", lyrics_request, correlation_id=correlation_id
                    )
                    lyrics = lyrics_response["lyrics"]
                
                return await self.generate_complete_music(
                    prompt=request.prompt,
                    lyrics=lyrics,
                    description_for_categorization=request.prompt,
                    audio_duration=request.audio_duration,
                    seed=request.seed,
                    guidance_scale=request.guidance_scale,
                    infer_step=request.infer_step,
                    instrumental=request.instrumental,
                    correlation_id=correlation_id
                )
                
            except Exception as e:
                logger.error(f"Failed to generate with described lyrics [correlation_id: {correlation_id}]: {e}")
                raise


    @modal.fastapi_endpoint(method="GET")
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint for service monitoring"""
        return self.health_check()
    
    @modal.fastapi_endpoint(method="GET")
    async def service_info(self) -> Dict[str, Any]:
        """Get service information and configuration"""
        return {
            **self.get_service_info(),
            "service_urls": self.service_urls,
            "available_endpoints": [
                "/generate_from_description",
                "/generate_with_lyrics", 
                "/generate_with_described_lyrics",
                "/generate_from_description_unified",
                "/generate_with_lyrics_unified",
                "/generate_with_described_lyrics_unified",
                "/health_check",
                "/service_info",
                "/service_stats",
                "/correlation_history",
                "/cleanup_tracking_data"
            ]
        }
    
    async def check_service_availability(self, service_name: str) -> bool:
        """Check if a service is available"""
        try:
            if service_name not in self.service_urls:
                return False
            
            url = f"{self.service_urls[service_name]}/health_check"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Service {service_name} availability check failed: {e}")
            return False
    
    @modal.fastapi_endpoint(method="GET")
    async def service_stats(self) -> Dict[str, Any]:
        """Get service communication statistics and circuit breaker status"""
        stats = service_comm_manager.get_service_stats()
        
        # Add service availability
        availability = {}
        for service_name in self.service_urls.keys():
            availability[service_name] = await self.check_service_availability(service_name)
        
        return {
            "circuit_breaker_stats": stats,
            "service_availability": availability,
            "active_calls": len(service_comm_manager.active_calls),
            "correlation_tracking_count": len(service_comm_manager.correlation_tracking)
        }
    
    @modal.fastapi_endpoint(method="GET")
    async def correlation_history(self, correlation_id: str) -> Dict[str, Any]:
        """Get call history for a specific correlation ID"""
        calls = service_comm_manager.get_correlation_history(correlation_id)
        
        return {
            "correlation_id": correlation_id,
            "total_calls": len(calls),
            "calls": [
                {
                    "service_name": call.service_name,
                    "endpoint": call.endpoint,
                    "start_time": call.start_time,
                    "duration": call.duration,
                    "success": call.success,
                    "error": call.error,
                    "retry_count": call.retry_count
                }
                for call in calls
            ]
        }
    
    @modal.fastapi_endpoint(method="POST")
    async def cleanup_tracking_data(self, max_age_hours: int = 1) -> Dict[str, Any]:
        """Clean up old tracking data"""
        max_age_seconds = max_age_hours * 3600
        service_comm_manager.cleanup_old_tracking_data(max_age_seconds)
        
        return {
            "message": f"Cleaned up tracking data older than {max_age_hours} hours",
            "remaining_correlations": len(service_comm_manager.correlation_tracking),
            "remaining_active_calls": len(service_comm_manager.active_calls)
        }
    
    async def generate_complete_music_unified(
        self,
        prompt: str,
        lyrics: str,
        description_for_categorization: str,
        audio_duration: float = 180.0,
        seed: int = -1,
        guidance_scale: float = 15.0,
        infer_step: int = 60,
        instrumental: bool = False,
        correlation_id: Optional[str] = None
    ) -> GenerateMusicResponseUnified:
        """Complete music generation with unified response and partial result handling"""
        correlation_id = correlation_id or generate_correlation_id()
        logger.info(f"Starting unified music generation [correlation_id: {correlation_id}]")
        
        # Track service status and results
        service_status = {"music": False, "image": False, "categories": False}
        results = {"music": None, "cover": None, "categories": None}
        errors = []
        
        try:
            # Prepare service calls
            service_calls = [
                {
                    "service_url": self.service_urls["music"],
                    "endpoint": "/generate_music_to_storage",
                    "data": {
                        "prompt": prompt,
                        "lyrics": lyrics,
                        "audio_duration": audio_duration,
                        "seed": seed,
                        "guidance_scale": guidance_scale,
                        "infer_step": infer_step,
                        "instrumental": instrumental
                    },
                    "service_name": "music",
                    "timeout": 600,
                    "max_retries": 1
                },
                {
                    "service_url": self.service_urls["image"],
                    "endpoint": "/generate_cover_image",
                    "data": {
                        "prompt": prompt,
                        "style": "album cover art"
                    },
                    "service_name": "image",
                    "timeout": 120,
                    "max_retries": 2
                },
                {
                    "service_url": self.service_urls["lyrics"],
                    "endpoint": "/generate_categories",
                    "data": {
                        "description": description_for_categorization
                    },
                    "service_name": "lyrics",
                    "timeout": 60,
                    "max_retries": 2
                }
            ]
            
            # Execute all calls in parallel
            parallel_results = await service_comm_manager.call_service_batch(
                service_calls, correlation_id
            )
            
            # Process results with partial handling
            service_names = ["music", "image", "categories"]
            for i, (service_name, result) in enumerate(zip(service_names, parallel_results)):
                if result["success"]:
                    service_status[service_name] = True
                    if service_name == "categories":
                        results["categories"] = result["data"]
                    else:
                        results[service_name] = result["data"]
                    logger.info(f"Successfully completed {service_name} generation [correlation_id: {correlation_id}]")
                else:
                    error_msg = f"Failed {service_name} generation: {result['error']}"
                    errors.append(error_msg)
                    logger.error(f"{error_msg} [correlation_id: {correlation_id}]")
            
            # Music generation is required - fail completely if it fails
            if not service_status["music"]:
                raise Exception("Music generation failed - cannot proceed with workflow")
            
            # Build unified response with partial results
            return GenerateMusicResponseUnified(
                audio_file_path=results["music"]["file_path"],
                cover_image_file_path=results["cover"]["file_path"] if results["cover"] else None,
                categories=results["categories"]["categories"] if results["categories"] else [],
                generation_metadata={
                    "prompt": prompt,
                    "lyrics": lyrics if lyrics else None,
                    "audio_duration": audio_duration,
                    "instrumental": instrumental,
                    "generation_params": {
                        "seed": seed,
                        "guidance_scale": guidance_scale,
                        "infer_step": infer_step
                    }
                },
                service_status=service_status,
                errors=errors,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(f"Failed unified music generation [correlation_id: {correlation_id}]: {e}")
            raise
    
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def generate_from_description_unified(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponseUnified:
        """Generate complete music work from description with unified response format"""
        correlation_id = generate_correlation_id()
        
        logger.info(f"Generating from description (unified): {request.full_described_song[:100]}... [correlation_id: {correlation_id}]")
        
        with self.operation_context("generate_from_description_unified") as operation_id:
            try:
                # 1. Generate music prompt
                prompt_request = {"description": request.full_described_song}
                prompt_response = await self.call_lyrics_service(
                    "generate_prompt", prompt_request, correlation_id=correlation_id
                )
                prompt = prompt_response["prompt"]
                
                # 2. Generate lyrics (if not instrumental)
                lyrics = ""
                if not request.instrumental:
                    lyrics_request = {"description": request.full_described_song}
                    lyrics_response = await self.call_lyrics_service(
                        "generate_lyrics", lyrics_request, correlation_id=correlation_id
                    )
                    lyrics = lyrics_response["lyrics"]
                
                # 3. Generate complete music work with unified response
                return await self.generate_complete_music_unified(
                    prompt=prompt,
                    lyrics=lyrics,
                    description_for_categorization=request.full_described_song,
                    audio_duration=request.audio_duration,
                    seed=request.seed,
                    guidance_scale=request.guidance_scale,
                    infer_step=request.infer_step,
                    instrumental=request.instrumental,
                    correlation_id=correlation_id
                )
                
            except Exception as e:
                logger.error(f"Failed to generate from description (unified) [correlation_id: {correlation_id}]: {e}")
                raise
    
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def generate_with_lyrics_unified(self, request: GenerateWithCustomLyricsRequest) -> GenerateMusicResponseUnified:
        """Generate music with custom lyrics using unified response format"""
        correlation_id = generate_correlation_id()
        
        logger.info(f"Generating with custom lyrics (unified), prompt: {request.prompt[:100]}... [correlation_id: {correlation_id}]")
        
        with self.operation_context("generate_with_lyrics_unified") as operation_id:
            return await self.generate_complete_music_unified(
                prompt=request.prompt,
                lyrics=request.lyrics,
                description_for_categorization=request.prompt,
                audio_duration=request.audio_duration,
                seed=request.seed,
                guidance_scale=request.guidance_scale,
                infer_step=request.infer_step,
                instrumental=request.instrumental,
                correlation_id=correlation_id
            )
    
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def generate_with_described_lyrics_unified(self, request: GenerateWithDescribedLyricsRequest) -> GenerateMusicResponseUnified:
        """Generate music with described lyrics using unified response format"""
        correlation_id = generate_correlation_id()
        
        logger.info(f"Generating with described lyrics (unified): {request.described_lyrics[:100]}... [correlation_id: {correlation_id}]")
        
        with self.operation_context("generate_with_described_lyrics_unified") as operation_id:
            try:
                # Generate lyrics (if not instrumental)
                lyrics = ""
                if not request.instrumental:
                    lyrics_request = {"description": request.described_lyrics}
                    lyrics_response = await self.call_lyrics_service(
                        "generate_lyrics", lyrics_request, correlation_id=correlation_id
                    )
                    lyrics = lyrics_response["lyrics"]
                
                return await self.generate_complete_music_unified(
                    prompt=request.prompt,
                    lyrics=lyrics,
                    description_for_categorization=request.prompt,
                    audio_duration=request.audio_duration,
                    seed=request.seed,
                    guidance_scale=request.guidance_scale,
                    infer_step=request.infer_step,
                    instrumental=request.instrumental,
                    correlation_id=correlation_id
                )
                
            except Exception as e:
                logger.error(f"Failed to generate with described lyrics (unified) [correlation_id: {correlation_id}]: {e}")
                raise


@app.local_entrypoint()
def test_integrated_service():
    """Test integrated service functionality"""
    import asyncio
    
    async def run_test():
        server = IntegratedMusicGenServer()
        server.setup()
        
        # Test service info
        info = await server.service_info()
        print(f"Service info: {info}")
        
        # Test health check
        health = await server.health_check()
        print(f"Health check: {health}")
        
        # Test service availability (will fail without actual services)
        for service_name in ["lyrics", "music", "image"]:
            available = await server.check_service_availability(service_name)
            print(f"Service {service_name} available: {available}")
        
        print("Integration service test completed")
    
    asyncio.run(run_test())


if __name__ == "__main__":
    test_integrated_service()