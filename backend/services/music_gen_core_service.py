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
    ServiceMetadata
)
from shared.modal_config import music_image, model_volume, hf_volume, music_gen_secrets
from shared.config import settings
from shared.utils import (
    ensure_output_dir, 
    generate_temp_filepath, 
    FileManager,
    cost_monitor,
    timeout_manager
)

logger = logging.getLogger(__name__)

# Create Modal app with optimized configuration for music generation
app = modal.App("music-generator-core")


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


@app.cls(
    image=music_image,
    gpu="L40S",  # High-memory GPU for ACE-Step pipeline
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=60,  # 60-second scaledown for cost optimization
    timeout=600,  # 10-minute timeout for music generation
    memory=32768,  # 32GB memory for large models
    cpu=8.0  # 8 CPU cores for processing
)
class MusicGenCoreServer:
    """Standalone music generation server optimized for high-memory GPU instances"""
    
    @modal.enter()
    def load_model(self):
        """Load ACE-Step music generation model and initialize utilities"""
        try:
            from acestep.pipeline_ace_step import ACEStepPipeline
            
            logger.info("Loading ACE-Step music generation model...")
            start_time = time.time()
            
            # Load the music generation model
            self.music_model = ACEStepPipeline(
                checkpoint_dir=settings.music_model_checkpoint_dir,
                dtype="bfloat16",
                torch_compile=False,
                cpu_offload=False,
                overlapped_decode=False
            )
            
            # Initialize file manager for storage handling
            self.file_manager = FileManager(
                use_s3=settings.use_s3_storage,
                local_storage_dir=settings.local_storage_dir
            )
            
            load_time = time.time() - start_time
            logger.info(f"Music generation model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load music generation model: {e}")
            raise

    def _generate_operation_metadata(self, operation_id: str, start_time: float, 
                                   end_time: float) -> GenerationMetadata:
        """Generate metadata for the operation"""
        duration = end_time - start_time
        
        return GenerationMetadata(
            generation_time=duration,
            model_info="ACE-Step Pipeline",
            gpu_type="L40S",
            estimated_cost=cost_monitor.get_operation_cost(operation_id),
            operation_id=operation_id
        )

    def _generate_music_internal(self, request: MusicGenerationRequest, 
                               operation_id: str) -> tuple[str, float]:
        """Internal music generation method"""
        logger.info(f"Generating music with prompt: {request.prompt[:100]}...")
        
        # Handle lyrics
        final_lyrics = "[instrumental]" if request.instrumental else request.lyrics
        
        # Generate temporary output path
        output_dir = ensure_output_dir()
        output_path = generate_temp_filepath(output_dir, ".wav")
        
        # Generate music using ACE-Step pipeline
        self.music_model(
            prompt=request.prompt,
            lyrics=final_lyrics,
            audio_duration=request.audio_duration,
            infer_step=request.infer_step,
            guidance_scale=request.guidance_scale,
            save_path=output_path,
            manual_seeds=str(request.seed)
        )
        
        return output_path, request.audio_duration

    @modal.fastapi_endpoint(method="POST")
    def generate_music(self, request: MusicGenerationRequest) -> MusicGenerationResponseBase64:
        """Generate music and return base64 encoded audio data"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Start cost and timeout monitoring
        cost_monitor.start_operation(operation_id, "L40S", "music-generation")
        timeout_manager.start_timeout(operation_id, 600)  # 10-minute timeout
        
        try:
            # Check timeout before starting
            if timeout_manager.check_timeout(operation_id):
                raise TimeoutError("Operation timed out before starting")
            
            # Generate music
            output_path, audio_duration = self._generate_music_internal(request, operation_id)
            
            # Check timeout after generation
            if timeout_manager.check_timeout(operation_id):
                raise TimeoutError("Operation timed out during generation")
            
            # Read audio file and encode to base64
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Clean up temporary file
            if os.path.exists(output_path):
                os.remove(output_path)
            
            end_time = time.time()
            
            # End monitoring
            cost_monitor.end_operation(operation_id)
            timeout_manager.end_timeout(operation_id)
            
            # Generate metadata
            metadata = self._generate_operation_metadata(operation_id, start_time, end_time)
            
            logger.info(f"Music generated successfully in {end_time - start_time:.2f}s")
            
            return MusicGenerationResponseBase64(
                audio_data=audio_b64,
                generation_time=end_time - start_time,
                audio_duration=audio_duration,
                metadata=metadata
            )
            
        except Exception as e:
            # End monitoring on error
            cost_monitor.end_operation(operation_id)
            timeout_manager.end_timeout(operation_id)
            
            logger.error(f"Failed to generate music: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_music_to_storage(self, request: MusicGenerationRequest) -> MusicGenerationResponseLocal:
        """Generate music and save to storage (local or S3)"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Start cost and timeout monitoring
        cost_monitor.start_operation(operation_id, "L40S", "music-generation-storage")
        timeout_manager.start_timeout(operation_id, 600)  # 10-minute timeout
        
        try:
            # Check timeout before starting
            if timeout_manager.check_timeout(operation_id):
                raise TimeoutError("Operation timed out before starting")
            
            logger.info(f"Generating music to storage with prompt: {request.prompt[:100]}...")
            
            # Generate music
            output_path, audio_duration = self._generate_music_internal(request, operation_id)
            
            # Check timeout after generation
            if timeout_manager.check_timeout(operation_id):
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
            cost_monitor.end_operation(operation_id)
            timeout_manager.end_timeout(operation_id)
            
            # Generate metadata
            metadata = self._generate_operation_metadata(operation_id, start_time, end_time)
            
            logger.info(f"Music generated and saved to storage: {file_key}")
            
            return MusicGenerationResponseLocal(
                file_path=file_key,
                generation_time=end_time - start_time,
                file_size_mb=file_size_mb,
                audio_duration=audio_duration,
                metadata=metadata
            )
            
        except Exception as e:
            # End monitoring on error
            cost_monitor.end_operation(operation_id)
            timeout_manager.end_timeout(operation_id)
            
            logger.error(f"Failed to generate music to storage: {e}")
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
            infer_step=30,      # Fewer steps for faster generation
            guidance_scale=15
        )
        
        return self.generate_music(demo_request)

    @modal.fastapi_endpoint(method="GET")
    def health_check(self) -> dict:
        """Health check endpoint for service monitoring"""
        return {
            "status": "healthy",
            "service": "music-generator-core",
            "gpu_type": "L40S",
            "model_loaded": hasattr(self, 'music_model'),
            "storage_mode": "S3" if self.file_manager.use_s3 else "local"
        }


@app.local_entrypoint()
def test_music_generation():
    """Test music generation service locally"""
    server = MusicGenCoreServer()
    
    # Test music generation with base64 response
    request = MusicGenerationRequest(
        prompt="upbeat electronic dance music, 128 bpm, energetic",
        lyrics="[verse]\nDancing through the night\nLights are shining bright\n[chorus]\nFeel the beat tonight\nEverything's alright",
        audio_duration=30,  # Short duration for testing
        infer_step=20
    )
    
    print("Testing base64 music generation...")
    response = server.generate_music(request)
    print(f"Generated music (base64 length): {len(response.audio_data)}")
    print(f"Generation time: {response.generation_time:.2f}s")
    print(f"Audio duration: {response.audio_duration}s")
    
    print("\nTesting storage music generation...")
    storage_response = server.generate_music_to_storage(request)
    print(f"Generated music saved to: {storage_response.file_path}")
    print(f"Generation time: {storage_response.generation_time:.2f}s")
    print(f"File size: {storage_response.file_size_mb:.2f} MB")
    
    print("\nTesting demo music generation...")
    demo_response = server.generate_demo_music()
    print(f"Demo music generated (base64 length): {len(demo_response.audio_data)}")
    print(f"Generation time: {demo_response.generation_time:.2f}s")


if __name__ == "__main__":
    test_music_generation()