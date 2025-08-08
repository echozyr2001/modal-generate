import modal
import logging
import time
import uuid
import torch
from fastapi import HTTPException
from pydantic import ValidationError
from shared.models import (
    LyricsGenerationRequest, 
    LyricsGenerationResponse,
    PromptGenerationRequest,
    PromptGenerationResponse,
    CategoryGenerationRequest,
    CategoryGenerationResponse,
    ServiceConfig,
    GPUType,
    GenerationMetadata
)
from shared.deployment import llm_image, hf_volume, music_gen_secrets
from shared.config import settings
from shared.monitoring import TimeoutManager, CostMonitor
from prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

app = modal.App("lyrics-generator")

# Get service configuration from settings
lyrics_config = ServiceConfig(
    service_name="lyrics-generator",
    **settings.get_service_config("lyrics")
)


@app.cls(
    image=llm_image,
    gpu=lyrics_config.gpu_type.value if lyrics_config.gpu_type != GPUType.CPU else None,
    volumes={"/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=lyrics_config.scaledown_window,
    timeout=lyrics_config.max_runtime_seconds
)
class LyricsGenServer:
    # No inheritance to avoid Modal deprecation warning
    # All initialization happens in load_model

    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Initialize attributes that were previously in __init__
        self.config = lyrics_config
        self.cost_monitor = CostMonitor()
        self.timeout_manager = TimeoutManager(lyrics_config.max_runtime_seconds)
        self._model_loaded = False
        
        logger.info(f"Loading LLM model: {settings.llm_model_id} on {lyrics_config.gpu_type.value}")
        
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.llm_model_id,
                cache_dir=settings.hf_cache_dir
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with proper device handling
            device = "cuda" if torch.cuda.is_available() and lyrics_config.gpu_type != GPUType.CPU else "cpu"
            
            if device == "cuda":
                # For GPU, use more conservative device mapping to avoid meta device issues
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.llm_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},  # Force all layers to GPU 0
                    cache_dir=settings.hf_cache_dir,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # For CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.llm_model_id,
                    torch_dtype=torch.float32,
                    device_map=None,
                    cache_dir=settings.hf_cache_dir,
                    trust_remote_code=True
                )
                self.model = self.model.to(device)
            
            # Verify model device placement
            try:
                model_device = next(self.model.parameters()).device
                logger.info(f"Model parameters are on device: {model_device}")
                
                # Check for mixed device placement
                devices = set()
                for param in self.model.parameters():
                    devices.add(param.device)
                
                if len(devices) > 1:
                    logger.warning(f"Model has parameters on multiple devices: {devices}")
                else:
                    logger.info(f"All model parameters are on: {list(devices)[0]}")
                    
            except Exception as e:
                logger.warning(f"Could not verify model device placement: {e}")
            
            self._model_loaded = True
            logger.info(f"LLM model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
            raise



    def prompt_qwen(self, question: str, operation_id: str) -> str:
        """通用的LLM推理方法 with timeout protection and error handling"""
        try:
            # Check if model is loaded
            if not hasattr(self, '_model_loaded') or not self._model_loaded:
                raise RuntimeError("Model not loaded")
            
            # Check timeout before starting generation
            if hasattr(self, 'timeout_manager') and self.timeout_manager.check_timeout(operation_id):
                raise TimeoutError(f"Operation {operation_id} timed out before LLM inference")
            
            # Validate input
            if not question or len(question.strip()) == 0:
                raise ValueError("Question cannot be empty")
            
            if len(question) > 5000:  # Reasonable limit for prompt length
                raise ValueError("Question too long for processing")
            
            messages = [{"role": "user", "content": question}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize with proper attention mask
            model_inputs = self.tokenizer(
                [text], 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move inputs to model device with error handling
            try:
                device = next(self.model.parameters()).device
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            except Exception as e:
                logger.error(f"Device placement error during input preparation: {e}")
                # Fallback: try to use cuda:0 if available
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            # Check timeout before generation
            if hasattr(self, 'timeout_manager') and self.timeout_manager.check_timeout(operation_id):
                raise TimeoutError(f"Operation {operation_id} timed out before model generation")
            
            # Generate with error handling
            try:
                with torch.no_grad():  # Save memory
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError("GPU out of memory. Try again later.")
                if "device" in str(e).lower():
                    raise RuntimeError(f"Device placement error: {e}")
                raise RuntimeError(f"Model generation failed: {e}")
            
            # Extract only the new tokens
            input_length = model_inputs['input_ids'].shape[1]
            generated_tokens = generated_ids[0][input_length:]
            
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Validate output
            if not response or len(response.strip()) == 0:
                raise ValueError("Model generated empty response")

            return response.strip()
            
        except TimeoutError:
            raise  # Re-raise timeout errors
        except (ValueError, RuntimeError):
            raise  # Re-raise validation and runtime errors
        except Exception as e:
            logger.error(f"[{operation_id}] LLM inference failed: {e}")
            raise RuntimeError(f"LLM inference failed: {str(e)}")

    @modal.fastapi_endpoint(method="POST")
    def generate_lyrics(self, request: LyricsGenerationRequest) -> LyricsGenerationResponse:
        """生成歌词 with comprehensive error handling"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"[{operation_id}] Generating lyrics for description: {request.description[:100]}...")
            
            # Validate input
            if not request.description or len(request.description.strip()) < 5:
                raise HTTPException(
                    status_code=400, 
                    detail="Description must be at least 5 characters long"
                )
            
            from shared.models.base import validate_text_input
            description = validate_text_input(request.description, min_length=5, max_length=1000)
            
            # Check if model is loaded
            if not self._model_loaded:
                raise HTTPException(
                    status_code=503, 
                    detail="Model not loaded. Service temporarily unavailable."
                )
            
            # Start monitoring
            self.cost_monitor.start_operation(
                operation_id, 
                lyrics_config.gpu_type.value, 
                lyrics_config.service_name
            )
            self.timeout_manager.start_timeout(operation_id, lyrics_config.max_runtime_seconds)
            
            # Check timeout before generation
            if self.timeout_manager.check_timeout(operation_id):
                raise HTTPException(
                    status_code=408, 
                    detail="Request timeout before generation started"
                )
            
            # Generate lyrics
            full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)
            lyrics = self.prompt_qwen(full_prompt, operation_id)
            
            # Validate output
            if not lyrics or len(lyrics.strip()) < 10:
                raise HTTPException(
                    status_code=500, 
                    detail="Generated lyrics are too short or empty"
                )
            
            # End monitoring
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            duration = time.time() - start_time
            logger.info(f"[{operation_id}] Lyrics generated successfully in {duration:.2f}s")
            return LyricsGenerationResponse(lyrics=lyrics.strip())
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            self._cleanup_operation(operation_id)
            raise
        except ValidationError as e:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Validation error: {e}")
            raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
        except TimeoutError:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Operation timed out after {lyrics_config.max_runtime_seconds}s")
            raise HTTPException(
                status_code=408, 
                detail=f"Request timed out after {lyrics_config.max_runtime_seconds} seconds"
            )
        except Exception as e:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Lyrics generation failed: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Internal server error during lyrics generation"
            )

    @modal.fastapi_endpoint(method="POST")
    def generate_prompt(self, request: PromptGenerationRequest) -> PromptGenerationResponse:
        """生成音乐提示词 with comprehensive error handling"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"[{operation_id}] Generating prompt for description: {request.description[:100]}...")
            
            # Validate input
            if not request.description or len(request.description.strip()) < 5:
                raise HTTPException(
                    status_code=400, 
                    detail="Description must be at least 5 characters long"
                )
            
            self._validate_request_size(request.description)
            from shared.models.base import validate_text_input
            description = validate_text_input(request.description, min_length=5, max_length=1000)
            
            # Check if model is loaded
            if not self._model_loaded:
                raise HTTPException(
                    status_code=503, 
                    detail="Model not loaded. Service temporarily unavailable."
                )
            
            # Start monitoring
            self.cost_monitor.start_operation(
                operation_id, 
                lyrics_config.gpu_type.value, 
                lyrics_config.service_name
            )
            self.timeout_manager.start_timeout(operation_id, lyrics_config.max_runtime_seconds)
            
            # Check timeout before generation
            if self.timeout_manager.check_timeout(operation_id):
                raise HTTPException(
                    status_code=408, 
                    detail="Request timeout before generation started"
                )
            
            # Generate prompt
            full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)
            prompt = self.prompt_qwen(full_prompt, operation_id)
            
            # Validate output
            if not prompt or len(prompt.strip()) < 5:
                raise HTTPException(
                    status_code=500, 
                    detail="Generated prompt is too short or empty"
                )
            
            # End monitoring
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            duration = time.time() - start_time
            logger.info(f"[{operation_id}] Prompt generated successfully in {duration:.2f}s")
            return PromptGenerationResponse(prompt=prompt.strip())
            
        except HTTPException:
            self._cleanup_operation(operation_id)
            raise
        except ValidationError as e:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Validation error: {e}")
            raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
        except TimeoutError:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Operation timed out after {lyrics_config.max_runtime_seconds}s")
            raise HTTPException(
                status_code=408, 
                detail=f"Request timed out after {lyrics_config.max_runtime_seconds} seconds"
            )
        except Exception as e:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Prompt generation failed: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Internal server error during prompt generation"
            )

    @modal.fastapi_endpoint(method="GET")
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": lyrics_config.service_name,
            "model_loaded": getattr(self, '_model_loaded', False),
            "gpu_type": lyrics_config.gpu_type.value,
            "scaledown_window": lyrics_config.scaledown_window,
            "max_runtime": lyrics_config.max_runtime_seconds,
            "timestamp": time.time()
        }

    @modal.fastapi_endpoint(method="POST")
    def generate_categories(self, request: CategoryGenerationRequest) -> CategoryGenerationResponse:
        """生成音乐分类 with comprehensive error handling"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"[{operation_id}] Generating categories for description: {request.description[:100]}...")
            
            # Validate input
            if not request.description or len(request.description.strip()) < 5:
                raise HTTPException(
                    status_code=400, 
                    detail="Description must be at least 5 characters long"
                )
            
            self._validate_request_size(request.description)
            from shared.models.base import validate_text_input
            description = validate_text_input(request.description, min_length=5, max_length=1000)
            
            # Check if model is loaded
            if not self._model_loaded:
                raise HTTPException(
                    status_code=503, 
                    detail="Model not loaded. Service temporarily unavailable."
                )
            
            # Start monitoring
            self.cost_monitor.start_operation(
                operation_id, 
                lyrics_config.gpu_type.value, 
                lyrics_config.service_name
            )
            self.timeout_manager.start_timeout(operation_id, lyrics_config.max_runtime_seconds)
            
            # Check timeout before generation
            if self.timeout_manager.check_timeout(operation_id):
                raise HTTPException(
                    status_code=408, 
                    detail="Request timeout before generation started"
                )
            
            # Generate categories
            prompt = (
                f"Based on the following music description, list 3-5 relevant genres or categories "
                f"as a comma-separated list. For example: Pop, Electronic, Sad, 80s. "
                f"Description: '{description}'"
            )
            
            response_text = self.prompt_qwen(prompt, operation_id)
            categories = [cat.strip() for cat in response_text.split(",") if cat.strip()]
            
            # Ensure we have at least some categories
            if not categories:
                logger.warning(f"[{operation_id}] No categories generated, using fallback")
                categories = ["Unknown"]
            
            # Limit categories to reasonable number
            categories = categories[:10]  # Max 10 categories
            
            # End monitoring
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            duration = time.time() - start_time
            logger.info(f"[{operation_id}] Generated {len(categories)} categories in {duration:.2f}s")
            return CategoryGenerationResponse(categories=categories)
            
        except HTTPException:
            self._cleanup_operation(operation_id)
            raise
        except ValidationError as e:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Validation error: {e}")
            raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
        except TimeoutError:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Operation timed out after {lyrics_config.max_runtime_seconds}s")
            raise HTTPException(
                status_code=408, 
                detail=f"Request timed out after {lyrics_config.max_runtime_seconds} seconds"
            )
        except Exception as e:
            self._cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Category generation failed: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Internal server error during category generation"
            )

    # Helper methods for analysis
    def create_metadata(self, operation_id: str, model_info: str, start_time: float) -> GenerationMetadata:
        """Create metadata for responses"""
        from shared.utils import create_metadata
        return create_metadata(
            operation_id=operation_id,
            model_info=model_info,
            start_time=start_time,
            gpu_type=lyrics_config.gpu_type.value,
            cost_per_hour=lyrics_config.cost_per_hour
        )

    def _cleanup_operation(self, operation_id: str):
        """Clean up monitoring for failed operations"""
        try:
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup operation {operation_id}: {e}")
    
    def _validate_request_size(self, text: str):
        """Validate request size to prevent abuse"""
        if len(text) > settings.max_prompt_length:
            raise HTTPException(
                status_code=413,
                detail=f"Request too large: {len(text)} > {settings.max_prompt_length}"
            )


@app.local_entrypoint()
def test_lyrics_generation():
    """Test lyrics generation service with improved error handling"""
    import requests
    import time
    
    server = LyricsGenServer()
    
    print(f"Testing Lyrics Generation Service")
    print(f"Configuration: {lyrics_config.service_name} on {lyrics_config.gpu_type.value}")
    print("-" * 50)
    
    # Wait longer for service to initialize (model loading takes time)
    print("Waiting for service to initialize (model loading may take 30-60 seconds)...")
    time.sleep(15)
    
    # Test health check with retries
    health_url = server.health_check.get_web_url()
    print(f"Health check URL: {health_url}")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Health check attempt {attempt + 1}/{max_retries}...")
            response = requests.get(health_url, timeout=60)  # Increased timeout
            response.raise_for_status()
            health = response.json()
            print(f"✓ Health check successful: {health}")
            
            if not health.get('model_loaded', False):
                print("⚠️  Model not loaded yet, waiting...")
                time.sleep(20)
                continue
            else:
                print("✓ Model is loaded and ready")
                break
                
        except requests.exceptions.Timeout:
            print(f"⚠️  Health check timeout (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            else:
                print("❌ Health check failed after all retries")
                return
        except Exception as e:
            print(f"❌ Health check error: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            else:
                return
    
    # Test basic endpoints
    print("\n" + "="*50)
    print("Testing Service Endpoints")
    print("="*50)
    
    basic_tests = [
        ("generate_lyrics", LyricsGenerationRequest(description="a happy song about sunshine")),
        ("generate_prompt", PromptGenerationRequest(description="upbeat electronic music")),
        ("generate_categories", CategoryGenerationRequest(description="electronic dance music"))
    ]
    
    success_count = 0
    for endpoint_name, request_obj in basic_tests:
        try:
            endpoint = getattr(server, endpoint_name)
            url = endpoint.get_web_url()
            print(f"\nTesting {endpoint_name}...")
            print(f"URL: {url}")
            
            start_time = time.time()
            response = requests.post(url, json=request_obj.model_dump(), timeout=90)
            duration = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            
            print(f"✓ {endpoint_name} succeeded in {duration:.2f}s")
            
            # Show some result details
            if 'lyrics' in result:
                print(f"  Lyrics preview: {result['lyrics'][:100]}...")
            elif 'prompt' in result:
                print(f"  Prompt: {result['prompt']}")
            elif 'categories' in result:
                print(f"  Categories: {result['categories']}")
            
            success_count += 1
            
        except requests.exceptions.Timeout:
            print(f"❌ {endpoint_name} timed out")
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint_name} request failed: {e}")
        except Exception as e:
            print(f"❌ {endpoint_name} failed: {e}")
    
    print(f"\n" + "="*50)
    print(f"Test Summary: {success_count}/{len(basic_tests)} endpoints successful")
    print("="*50)
    
    if success_count == len(basic_tests):
        print("🎉 All tests passed!")
    elif success_count > 0:
        print("⚠️  Some tests passed, some failed")
    else:
        print("❌ All tests failed")


if __name__ == "__main__":
    test_lyrics_generation()