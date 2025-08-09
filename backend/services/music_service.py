"""
Music Generation Service
Provides music generation using ACE-Step pipeline.
Extracted from main.py to provide dedicated music generation capabilities.
"""

import modal
import logging
import base64
import os
import time
import uuid
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from fastapi import HTTPException

from shared.models import (
    AudioGenerationBase,
    GenerateMusicResponse,
    GenerationMetadata,
    ServiceConfig,
    GPUType
)
from shared.deployment import music_generation_image, model_volume, hf_volume, music_gen_secrets
from shared.config import settings
from shared.base_service import create_service_app
from shared.utils import ensure_output_dir, generate_temp_filepath

logger = logging.getLogger(__name__)

# Get service configuration
music_config = ServiceConfig(
    service_name="music-generator",
    **settings.get_service_config("music")
)

# Create Modal app
app, app_config = create_service_app(
    "music-generator",
    music_config,
    music_generation_image,
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets]
)

# Add CPU cores for music generation
app_config["cpu"] = 8.0


class MusicGenerationRequest(AudioGenerationBase):
    """Music generation request with validation"""
    prompt: str = Field(..., min_length=5, max_length=500, description="Music generation prompt")
    lyrics: str = Field(..., min_length=1, max_length=2000, description="Lyrics for the music")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        import re
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if len(cleaned) < 5:
            raise ValueError("Prompt too short after cleaning")
        return cleaned
    
    @field_validator('lyrics')
    @classmethod
    def validate_lyrics(cls, v):
        import re
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if not cleaned:
            return "[instrumental]"
        return cleaned


class MusicGenerationResponseBase64(BaseModel):
    """Response with base64 encoded audio"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    generation_time: float = Field(..., description="Time taken for generation")
    audio_duration: float = Field(..., description="Duration of generated audio")
    metadata: GenerationMetadata


class MusicGenerationResponseFile(BaseModel):
    """Response with file path"""
    file_path: str = Field(..., description="File path (S3 key or local path)")
    generation_time: float = Field(..., description="Time taken for generation")
    file_size_mb: Optional[float] = Field(default=None, description="File size in MB")
    audio_duration: float = Field(..., description="Duration of generated audio")
    metadata: GenerationMetadata


@app.cls(**app_config)
class MusicGenServer:
    """Music generation server - extracted from main.py ACE-Step functionality"""
    
    @modal.enter()
    def load_model(self):
        """Load ACE-Step music generation model"""
        from acestep.pipeline_ace_step import ACEStepPipeline
        from shared.monitoring import CostMonitor, TimeoutManager
        from shared.storage import FileManager
        
        # Initialize service components (since we can't use __init__)
        self.config = music_config
        self.cost_monitor = CostMonitor()
        self.timeout_manager = TimeoutManager(music_config.max_runtime_seconds)
        self.file_manager = FileManager()
        self._model_loaded = False
        
        logger.info("Loading ACE-Step music generation model...")
        
        try:
            # Load the music generation model (same as main.py)
            self.music_model = ACEStepPipeline(
                checkpoint_dir=settings.music_model_checkpoint_dir,
                dtype="bfloat16",
                torch_compile=False,
                cpu_offload=False,
                overlapped_decode=False
            )
            
            self._model_loaded = True
            logger.info("ACE-Step model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load music generation model: {e}")
            self._model_loaded = False
            raise
    
    def generate_and_upload_to_s3(self, prompt: str, lyrics: str, instrumental: bool,
                                 audio_duration: float, inference_steps: int, 
                                 guidance_scale: float, seed: int) -> str:
        """
        Generate music and upload to S3 - extracted from main.py
        Returns S3 key of uploaded audio file
        """
        import boto3
        
        final_lyrics = "[instrumental]" if instrumental else lyrics
        print(f"Generated lyrics: \n{final_lyrics}")
        print(f"Prompt: \n{prompt}")

        s3_client = boto3.client("s3")
        bucket_name = os.environ["S3_BUCKET_NAME"]

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        # Generate music using ACE-Step (same as main.py)
        self.music_model(
            prompt=prompt,
            lyrics=final_lyrics,
            audio_duration=audio_duration,
            infer_step=inference_steps,
            guidance_scale=guidance_scale,
            save_path=output_path,
            manual_seeds=str(seed)
        )

        audio_s3_key = f"{uuid.uuid4()}.wav"
        s3_client.upload_file(output_path, bucket_name, audio_s3_key)
        os.remove(output_path)
        
        return audio_s3_key
    
    def generate_music_internal(self, request: MusicGenerationRequest, operation_id: str) -> tuple[str, float]:
        """Internal music generation method"""
        self.validate_model_loaded()
        
        if self.check_timeout(operation_id):
            raise TimeoutError(f"Operation {operation_id} timed out before generation")
        
        logger.info(f"[{operation_id}] Generating music: {request.prompt[:100]}...")
        
        # Handle lyrics
        final_lyrics = "[instrumental]" if request.instrumental else request.lyrics
        
        # Generate temporary output path
        output_dir = ensure_output_dir()
        output_path = generate_temp_filepath(output_dir, ".wav")
        
        # Generate music using ACE-Step (same as main.py)
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
    
    # Service management methods
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": self.config.service_name,
            "gpu_type": self.config.gpu_type.value,
            "model_type": "ACE-Step",
            "supported_formats": ["WAV"],
            "max_audio_duration": settings.max_audio_duration,
            "default_audio_duration": settings.default_audio_duration,
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
    
    # API Endpoints
    @modal.fastapi_endpoint(method="POST")
    def generate_music(self, request: MusicGenerationRequest) -> MusicGenerationResponseBase64:
        """Generate music and return as base64 - matches main.py generate endpoint"""
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Music generation request: {request.prompt[:100]}...")
        
        # Start monitoring
        self.start_operation(operation_id, "music_generation")
        
        try:
            # Generate music
            output_path, audio_duration = self.generate_music_internal(request, operation_id)
            
            # Read and encode audio (same as main.py)
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Clean up temporary file
            if os.path.exists(output_path):
                os.remove(output_path)
            
            # End monitoring
            self.end_operation(operation_id, success=True)
            
            # Create metadata
            metadata = self.create_metadata(operation_id, "ACE-Step Pipeline", start_time)
            
            logger.info(f"[{operation_id}] Music generated successfully")
            
            return MusicGenerationResponseBase64(
                audio_data=audio_b64,
                generation_time=time.time() - start_time,
                audio_duration=audio_duration,
                metadata=metadata
            )
            
        except Exception as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Music generation failed: {e}")
            raise HTTPException(status_code=500, detail="Music generation failed")
    
    @modal.fastapi_endpoint(method="POST")
    def generate_demo_music(self) -> MusicGenerationResponseBase64:
        """Generate demo music for testing - matches main.py generate endpoint"""
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Generating demo music...")
        
        # Create demo request (same as main.py hardcoded values)
        demo_request = MusicGenerationRequest(
            prompt="electronic rap",
            lyrics="""[verse]
Waves on the bass, pulsing in the speakers,
Turn the dial up, we chasing six-figure features,
Grinding on the beats, codes in the creases,
Digital hustler, midnight in sneakers.

[chorus]
Electro vibes, hearts beat with the hum,
Urban legends ride, we ain't ever numb,
Circuits sparking live, tapping on the drum,
Living on the edge, never succumb.

[verse]
Synthesizers blaze, city lights a glow,
Rhythm in the haze, moving with the flow,
Swagger on stage, energy to blow,
From the blocks to the booth, you already know.

[bridge]
Night's electric, streets full of dreams,
Bass hits collective, bursting at seams,
Hustle perspective, all in the schemes,
Rise and reflective, ain't no in-betweens.

[verse]
Vibin' with the crew, sync in the wire,
Got the dance moves, fire in the attire,
Rhythm and blues, soul's our supplier,
Run the digital zoo, higher and higher.

[chorus]
Electro vibes, hearts beat with the hum,
Urban legends ride, we ain't ever numb,
Circuits sparking live, tapping on the drum,
Living on the edge, never succumb.""",
            audio_duration=180,
            inference_steps=60,
            guidance_scale=15
        )
        
        # Start monitoring
        self.start_operation(operation_id, "demo_music_generation")
        
        try:
            # Generate music using internal method
            output_path, audio_duration = self.generate_music_internal(demo_request, operation_id)
            
            # Read and encode audio
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Clean up temporary file
            if os.path.exists(output_path):
                os.remove(output_path)
            
            # End monitoring
            self.end_operation(operation_id, success=True)
            
            # Create metadata
            metadata = self.create_metadata(operation_id, "ACE-Step Pipeline (Demo)", start_time)
            
            logger.info(f"[{operation_id}] Demo music generated successfully")
            
            return MusicGenerationResponseBase64(
                audio_data=audio_b64,
                generation_time=time.time() - start_time,
                audio_duration=audio_duration,
                metadata=metadata
            )
            
        except Exception as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Demo music generation failed: {e}")
            raise HTTPException(status_code=500, detail="Demo music generation failed")
    
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


@app.local_entrypoint()
def test_music_service():
    """Test the music service"""
    import requests
    import time
    
    server = MusicGenServer()
    
    print("=== Testing Music Service ===")
    print(f"Service: {music_config.service_name}")
    print(f"GPU: {music_config.gpu_type.value}")
    print("-" * 50)
    
    # Wait for initialization
    print("Waiting for service initialization (model loading may take time)...")
    time.sleep(30)
    
    try:
        # Test health check
        health_url = server.health_check.get_web_url()
        print(f"Health check: {health_url}")
        
        for attempt in range(5):
            try:
                response = requests.get(health_url, timeout=60)
                response.raise_for_status()
                health = response.json()
                print(f"✓ Health check: {health['status']}, Model loaded: {health['model_loaded']}")
                
                if health['model_loaded']:
                    break
                else:
                    print("Waiting for model to load...")
                    time.sleep(30)
            except Exception as e:
                print(f"Health check attempt {attempt + 1} failed: {e}")
                if attempt < 4:
                    time.sleep(15)
        
        # Test demo music generation
        print("\nTesting demo music generation...")
        demo_url = server.generate_demo_music.get_web_url()
        
        start_time = time.time()
        response = requests.post(demo_url, timeout=600)  # 10 min timeout
        duration = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Demo music generated in {duration:.2f}s")
        print(f"  Audio data length: {len(result['audio_data'])} characters")
        print(f"  Audio duration: {result['audio_duration']}s")
        print(f"  Generation time: {result['generation_time']:.2f}s")
        
        # Save demo audio
        audio_bytes = base64.b64decode(result['audio_data'])
        with open("demo_music.wav", "wb") as f:
            f.write(audio_bytes)
        
        file_size = len(audio_bytes) / (1024 * 1024)
        print(f"  Saved to: demo_music.wav ({file_size:.2f}MB)")
        
        print("\n" + "="*50)
        print("Music service test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_music_service()