import modal
import logging
import base64
import os
import time
import uuid
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from shared.models import (
    AudioGenerationBase,
    GenerateMusicResponse,
    GenerationMetadata,
    ServiceConfig,
    GPUType
)
from shared.deployment import music_generation_image, model_volume, hf_volume, music_gen_secrets
from shared.config import settings
from shared.utils import ensure_output_dir, generate_temp_filepath, FileManager

logger = logging.getLogger(__name__)

# Create Modal app with optimized configuration for music generation
app = modal.App("music-generator-core")

# Get service configuration from settings
music_config = ServiceConfig(
    service_name="music-generator-core",
    **settings.get_service_config("music")
)


class MusicGenerationRequest(AudioGenerationBase):
    """Enhanced music generation request with validation"""
    prompt: str = Field(..., min_length=5, max_length=500, description="Music generation prompt")
    lyrics: str = Field(..., min_length=1, max_length=2000, description="Lyrics for the music")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        # Clean and validate prompt
        import re
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if len(cleaned) < 5:
            raise ValueError("Prompt too short after cleaning")
        return cleaned
    
    @field_validator('lyrics')
    @classmethod
    def validate_lyrics(cls, v):
        # Clean and validate lyrics
        import re
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if not cleaned:
            return "[instrumental]"
        return cleaned


class MusicGenerationResponseLocal(BaseModel):
    """Response for music saved to storage"""
    file_path: str = Field(..., description="File path (S3 key or local path)")
    generation_time: float = Field(..., description="Time taken for generation")
    file_size_mb: Optional[float] = Field(default=None, description="File size in MB")
    audio_duration: float = Field(..., description="Duration of generated audio")
    metadata: GenerationMetadata


class MusicGenerationResponseBase64(BaseModel):
    """Response for music returned as base64"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    generation_time: float = Field(..., description="Time taken for generation")
    audio_duration: float = Field(..., description="Duration of generated audio")
    metadata: GenerationMetadata


class SimpleTimeoutManager:
    """Simple timeout manager for operations"""
    def __init__(self):
        self.operations = {}
    
    def check_timeout(self, operation_id: str) -> bool:
        """Check if operation has timed out"""
        if operation_id in self.operations:
            start_time = self.operations[operation_id]
            elapsed = time.time() - start_time
            return elapsed > music_config.max_runtime_seconds
        return False
    
    def start_timeout(self, operation_id: str):
        """Start timeout tracking for operation"""
        self.operations[operation_id] = time.time()
    
    def end_timeout(self, operation_id: str):
        """End timeout tracking for operation"""
        self.operations.pop(operation_id, None)


class SimpleCostMonitor:
    """Simple cost monitor for operations"""
    def __init__(self):
        self.operations = {}
    
    def start_operation(self, operation_id: str, gpu_type: str, service_name: str):
        """Start cost tracking for operation"""
        self.operations[operation_id] = {
            'start_time': time.time(),
            'gpu_type': gpu_type,
            'service_name': service_name
        }
    
    def end_operation(self, operation_id: str):
        """End cost tracking for operation"""
        self.operations.pop(operation_id, None)
    
    def get_operation_cost(self, operation_id: str) -> float:
        """Get estimated cost for operation"""
        if operation_id in self.operations:
            elapsed = time.time() - self.operations[operation_id]['start_time']
            return elapsed * (music_config.cost_per_hour / 3600)
        return 0.0



@app.cls(
    image=music_generation_image,
    gpu=music_config.gpu_type.value,
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=music_config.scaledown_window,
    timeout=music_config.max_runtime_seconds,
    memory=(music_config.memory_gb or 32) * 1024,  # Convert GB to MB, default 32GB
    cpu=8.0  # 8 CPU cores for processing
)
class MusicGenCoreServer:
    # Remove custom __init__ to fix Modal deprecation warning
    # Initialize these in load_model instead
    
    @modal.enter()
    def load_model(self):
        """Load ACE-Step music generation model and initialize utilities"""
        from acestep.pipeline_ace_step import ACEStepPipeline
        
        # Initialize attributes that were previously in __init__
        self.config = music_config
        self._model_loaded = False
        
        # Initialize monitoring utilities
        self.cost_monitor = SimpleCostMonitor()
        self.timeout_manager = SimpleTimeoutManager()
        
        # Initialize FileManager
        self.file_manager = FileManager()
        
        logger.info("Loading ACE-Step music generation model...")
        logger.info(f"Storage mode: {'S3' if settings.use_s3_storage else 'Local'} ({settings.local_storage_dir})")
        
        try:
            start_time = time.time()
            
            # Load the music generation model
            self.music_model = ACEStepPipeline(
                checkpoint_dir=settings.music_model_checkpoint_dir,
                dtype="bfloat16",
                torch_compile=False,
                cpu_offload=False,
                overlapped_decode=False
            )
            
            load_time = time.time() - start_time
            self._model_loaded = True
            logger.info(f"Music generation model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load music generation model: {e}")
            self._model_loaded = False
            raise

    def _generate_operation_metadata(self, operation_id: str, start_time: float, 
                                   end_time: float) -> GenerationMetadata:
        """Generate metadata for the operation"""
        duration = end_time - start_time
        
        return GenerationMetadata(
            generation_time=duration,
            model_info="ACE-Step Pipeline",
            gpu_type=music_config.gpu_type.value,
            estimated_cost=self.cost_monitor.get_operation_cost(operation_id),
            operation_id=operation_id
        )

    def _generate_music_internal(self, request: MusicGenerationRequest, 
                               operation_id: str) -> tuple[str, float]:
        """Internal music generation method"""
        logger.info(f"Generating music with prompt: {request.prompt[:100]}...")
        
        # Handle lyrics
        final_lyrics = "[instrumental]" if request.instrumental else request.lyrics
        logger.info(f"Using lyrics: {final_lyrics[:100]}...")
        
        # Generate temporary output path
        output_dir = ensure_output_dir()
        output_path = generate_temp_filepath(output_dir, ".wav")
        
        # Generate music using ACE-Step pipeline
        self.music_model(
            prompt=request.prompt,
            lyrics=final_lyrics,
            audio_duration=request.audio_duration,
            infer_step=request.inference_steps,
            guidance_scale=request.guidance_scale,
            save_path=output_path,
            manual_seeds=str(request.seed)
        )
        
        return output_path, request.audio_duration

    @modal.fastapi_endpoint(method="POST")
    def generate_music(self, request: MusicGenerationRequest) -> MusicGenerationResponseBase64:
        """Generate music and return base64 encoded audio data (similar to main.py)"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Starting music generation with prompt: {request.prompt[:100]}...")
        
        # Start cost and timeout monitoring
        self.cost_monitor.start_operation(operation_id, music_config.gpu_type.value, music_config.service_name)
        self.timeout_manager.start_timeout(operation_id)
        
        try:
            # Check if model is loaded
            if not hasattr(self, '_model_loaded') or not self._model_loaded:
                raise RuntimeError("Model not loaded. Service temporarily unavailable.")
            
            # Check timeout before starting
            if self.timeout_manager.check_timeout(operation_id):
                raise TimeoutError("Operation timed out before starting")
            
            # Generate music
            output_path, audio_duration = self._generate_music_internal(request, operation_id)
            
            # Check timeout after generation
            if self.timeout_manager.check_timeout(operation_id):
                raise TimeoutError("Operation timed out during generation")
            
            # Read audio file and encode to base64 (similar to main.py)
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Clean up temporary file
            if os.path.exists(output_path):
                os.remove(output_path)
            
            end_time = time.time()
            
            # End monitoring
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            # Generate metadata
            metadata = self._generate_operation_metadata(operation_id, start_time, end_time)
            
            logger.info(f"[{operation_id}] Music generated successfully in {end_time - start_time:.2f}s")
            
            return MusicGenerationResponseBase64(
                audio_data=audio_b64,
                generation_time=end_time - start_time,
                audio_duration=audio_duration,
                metadata=metadata
            )
            
        except Exception as e:
            # End monitoring on error
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            logger.error(f"[{operation_id}] Failed to generate music: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_music_to_storage(self, request: MusicGenerationRequest) -> MusicGenerationResponseLocal:
        """Generate music and save to storage (local or S3)"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Generating music to storage with prompt: {request.prompt[:100]}...")
        
        # Start cost and timeout monitoring
        self.cost_monitor.start_operation(operation_id, music_config.gpu_type.value, f"{music_config.service_name}-storage")
        self.timeout_manager.start_timeout(operation_id)
        
        try:
            # Check if model is loaded
            if not hasattr(self, '_model_loaded') or not self._model_loaded:
                raise RuntimeError("Model not loaded. Service temporarily unavailable.")
            
            # Check timeout before starting
            if self.timeout_manager.check_timeout(operation_id):
                raise TimeoutError("Operation timed out before starting")
            
            # Generate music
            output_path, audio_duration = self._generate_music_internal(request, operation_id)
            
            # Check timeout after generation
            if self.timeout_manager.check_timeout(operation_id):
                raise TimeoutError("Operation timed out during generation")
            
            # Save file using FileManager
            file_key = self.file_manager.save_file(output_path, file_type="audio")
            
            # Get file size
            file_size_mb = None
            if os.path.exists(output_path):
                file_size_bytes = os.path.getsize(output_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
            
            end_time = time.time()
            
            # End monitoring
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            # Generate metadata
            metadata = self._generate_operation_metadata(operation_id, start_time, end_time)
            
            logger.info(f"[{operation_id}] Music generated and saved to storage: {file_key}")
            
            return MusicGenerationResponseLocal(
                file_path=file_key,
                generation_time=end_time - start_time,
                file_size_mb=file_size_mb,
                audio_duration=audio_duration,
                metadata=metadata
            )
            
        except Exception as e:
            # End monitoring on error
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            logger.error(f"[{operation_id}] Failed to generate music to storage: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_demo_music(self) -> MusicGenerationResponseBase64:
        """Generate demo music for quick testing with shorter duration"""
        logger.info("Generating demo music...")
        
        # Create demo request with shorter duration for testing
        demo_request = MusicGenerationRequest(
            prompt="electronic rap, upbeat, 128 bpm",
            lyrics="""[verse]
Waves on the bass, pulsing in the speakers,
Turn the dial up, we chasing six-figure features,
Grinding on the beats, codes in the creases,
Digital hustler, midnight in sneakers.

[chorus]
Electro vibes, hearts beat with the hum,
Urban legends ride, we ain't ever numb,
Circuits sparking live, tapping on the drum,
Living on the edge, never succumb.""",
            audio_duration=60,  # Shorter duration for demo
            inference_steps=30,      # Fewer steps for faster generation
            guidance_scale=15
        )
        
        # Call the generate_music method directly (not as endpoint)
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Start cost and timeout monitoring
        self.cost_monitor.start_operation(operation_id, music_config.gpu_type.value, music_config.service_name)
        self.timeout_manager.start_timeout(operation_id)
        
        try:
            # Check if model is loaded
            if not hasattr(self, '_model_loaded') or not self._model_loaded:
                raise RuntimeError("Model not loaded. Service temporarily unavailable.")
            
            # Generate music
            output_path, audio_duration = self._generate_music_internal(demo_request, operation_id)
            
            # Read audio file and encode to base64
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Clean up temporary file
            if os.path.exists(output_path):
                os.remove(output_path)
            
            end_time = time.time()
            
            # End monitoring
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            # Generate metadata
            metadata = self._generate_operation_metadata(operation_id, start_time, end_time)
            
            logger.info(f"[{operation_id}] Demo music generated successfully in {end_time - start_time:.2f}s")
            
            return MusicGenerationResponseBase64(
                audio_data=audio_b64,
                generation_time=end_time - start_time,
                audio_duration=audio_duration,
                metadata=metadata
            )
            
        except Exception as e:
            # End monitoring on error
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            logger.error(f"[{operation_id}] Failed to generate demo music: {e}")
            raise

    @modal.fastapi_endpoint(method="GET")
    def health_check(self) -> dict:
        """Health check endpoint for service monitoring"""
        return {
            "status": "healthy",
            "service": music_config.service_name,
            "model_loaded": getattr(self, '_model_loaded', False),
            "gpu_type": music_config.gpu_type.value,
            "scaledown_window": music_config.scaledown_window,
            "max_runtime": music_config.max_runtime_seconds,
            "storage_mode": "S3" if settings.use_s3_storage else "Local",
            "storage_path": settings.local_storage_dir if not settings.use_s3_storage else settings.s3_bucket_name,
            "timestamp": time.time()
        }
    
    @modal.fastapi_endpoint(method="GET")
    def serve_file(self, file_key: str):
        """Serve audio file for local storage mode"""
        if settings.use_s3_storage:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400, 
                detail="File serving is only available for local storage mode. Use S3 URLs for S3 storage."
            )
        
        # Simple file serving for local mode
        from fastapi import HTTPException
        from fastapi.responses import FileResponse
        
        file_path = os.path.join(settings.local_storage_dir, file_key)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(file_path)
    
    @modal.fastapi_endpoint(method="GET")
    def download_audio(self, file_key: str) -> dict:
        """Download audio as base64 data (similar to main.py pattern)"""
        from fastapi import HTTPException
        
        # Find the file in the storage directory
        storage_path = settings.local_storage_dir
        file_path = os.path.join(storage_path, file_key)
        
        # If file_key is a full path, use it directly
        if not os.path.exists(file_path) and os.path.exists(file_key):
            file_path = file_key
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_key}")
        
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            file_size = len(audio_bytes)
            
            return {
                "audio_data": audio_b64,
                "file_size": file_size,
                "filename": os.path.basename(file_path),
                "format": "WAV"
            }
            
        except Exception as e:
            logger.error(f"Failed to read audio file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to read audio: {e}")
    
    @modal.fastapi_endpoint(method="GET")
    def get_storage_stats(self) -> dict:
        """Get storage statistics for this service"""
        if settings.use_s3_storage:
            return {"storage_mode": "S3", "bucket": settings.s3_bucket_name}
        
        # Local storage stats
        storage_dir = settings.local_storage_dir
        if not os.path.exists(storage_dir):
            return {"storage_mode": "Local", "directory": storage_dir, "files": 0, "total_size_mb": 0}
        
        files = [f for f in os.listdir(storage_dir) if f.endswith('.wav')]
        total_size = sum(os.path.getsize(os.path.join(storage_dir, f)) for f in files)
        
        return {
            "storage_mode": "Local",
            "directory": storage_dir,
            "files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }


@app.local_entrypoint()
def test_music_generation():
    """Test music generation service with comprehensive endpoint testing (similar to main.py)"""
    import requests
    import time
    
    server = MusicGenCoreServer()
    
    print("=== Music Generation Service Test ===")
    print(f"Configuration: {music_config.service_name} on {music_config.gpu_type.value}")
    print(f"Scaledown window: {music_config.scaledown_window}s")
    print(f"Max runtime: {music_config.max_runtime_seconds}s")
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
            time.sleep(30)  # Music model takes longer to load
        
        # Test music generation with base64 response (similar to main.py)
        print("\n=== Testing Music Generation (Base64) ===")
        request = MusicGenerationRequest(
            prompt="upbeat electronic dance music, 128 bpm, energetic",
            lyrics="[verse]\nDancing through the night\nLights are shining bright\n[chorus]\nFeel the beat tonight\nEverything's alright",
            audio_duration=30,  # Short duration for testing
            inference_steps=20,      # Fewer steps for faster generation
            guidance_scale=15
        )
        
        endpoint_url = server.generate_music.get_web_url()
        print(f"Testing music generation at {endpoint_url}")
        response = requests.post(endpoint_url, json=request.model_dump(), timeout=300)  # 5 min timeout
        response.raise_for_status()
        music_response = MusicGenerationResponseBase64(**response.json())
        
        print(f"✓ Music generation succeeded:")
        print(f"  Audio data length: {len(music_response.audio_data)} characters")
        print(f"  Generation time: {music_response.generation_time:.2f}s")
        print(f"  Audio duration: {music_response.audio_duration}s")
        print(f"  Estimated cost: ${music_response.metadata.estimated_cost:.4f}")
        
        # Save audio to local file (similar to main.py)
        audio_bytes = base64.b64decode(music_response.audio_data)
        output_filename = "generated_music.wav"
        with open(output_filename, "wb") as f:
            f.write(audio_bytes)
        
        file_size = len(audio_bytes) / (1024 * 1024)
        print(f"  Saved to: {output_filename} ({file_size:.2f}MB)")
        
        # Test demo music generation
        print("\n=== Testing Demo Music Generation ===")
        demo_url = server.generate_demo_music.get_web_url()
        demo_response = requests.post(demo_url, timeout=300)
        demo_response.raise_for_status()
        demo_music = MusicGenerationResponseBase64(**demo_response.json())
        
        print(f"✓ Demo music generation succeeded:")
        print(f"  Audio data length: {len(demo_music.audio_data)} characters")
        print(f"  Generation time: {demo_music.generation_time:.2f}s")
        print(f"  Audio duration: {demo_music.audio_duration}s")
        
        # Save demo audio
        demo_audio_bytes = base64.b64decode(demo_music.audio_data)
        demo_filename = "demo_music.wav"
        with open(demo_filename, "wb") as f:
            f.write(demo_audio_bytes)
        
        demo_file_size = len(demo_audio_bytes) / (1024 * 1024)
        print(f"  Saved to: {demo_filename} ({demo_file_size:.2f}MB)")
        
        # Test storage stats
        print("\n=== Testing Storage Stats ===")
        stats_url = server.get_storage_stats.get_web_url()
        stats_response = requests.get(stats_url, timeout=30)
        stats_response.raise_for_status()
        storage_stats = stats_response.json()
        print(f"Storage mode: {storage_stats['storage_mode']}")
        print(f"Files generated: {storage_stats.get('files', 0)}")
        print(f"Total size: {storage_stats.get('total_size_mb', 0)}MB")
        
        print(f"\n=== Test Summary ===")
        print(f"Generated 2 audio files successfully")
        print(f"Local files: {output_filename}, {demo_filename}")
        print(f"Total audio generated: {file_size + demo_file_size:.2f}MB")
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_music_generation()