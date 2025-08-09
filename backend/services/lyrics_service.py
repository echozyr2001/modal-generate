"""
Lyrics Generation Service
Provides lyrics, prompt, and category generation using LLM.
Extracted from main.py to provide dedicated text generation capabilities.
"""

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
from shared.base_service import create_service_app
from prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Get service configuration
lyrics_config = ServiceConfig(
    service_name="lyrics-generator",
    **settings.get_service_config("lyrics")
)

# Create Modal app
app, app_config = create_service_app(
    "lyrics-generator",
    lyrics_config,
    llm_image,
    volumes={"/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets]
)


@app.cls(**app_config)
class LyricsGenServer:
    """Lyrics generation server - extracted from main.py LLM functionality"""
    
    @modal.enter()
    def load_model(self):
        """Load LLM model for text generation"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from shared.monitoring import CostMonitor, TimeoutManager
        from shared.storage import FileManager
        
        # Initialize service components (since we can't use __init__)
        self.config = lyrics_config
        self.cost_monitor = CostMonitor()
        self.timeout_manager = TimeoutManager(lyrics_config.max_runtime_seconds)
        self.file_manager = FileManager()
        self._model_loaded = False
        
        logger.info(f"Loading LLM model: {settings.llm_model_id} on {self.config.gpu_type.value}")
        
        try:
            # Load tokenizer with proper pad token handling
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.llm_model_id,
                cache_dir=settings.hf_cache_dir
            )
            
            # Set pad token properly to avoid attention mask warnings
            if self.tokenizer.pad_token is None:
                # Try to use a different token for padding if available
                if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    # Fallback to eos_token but we'll handle attention mask explicitly
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("Using eos_token as pad_token - attention mask will be handled explicitly")
            
            # Load model with proper device handling (same as main.py)
            device = "cuda" if torch.cuda.is_available() and self.config.gpu_type != GPUType.CPU else "cpu"
            
            if device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.llm_model_id,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                    cache_dir=settings.hf_cache_dir,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.llm_model_id,
                    torch_dtype=torch.float32,
                    device_map=None,
                    cache_dir=settings.hf_cache_dir,
                    trust_remote_code=True
                )
                self.model = self.model.to(device)
            
            self._model_loaded = True
            logger.info(f"LLM model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
            raise
    
    def prompt_qwen(self, question: str) -> str:
        """
        LLM inference method - extracted from main.py
        Matches the exact implementation from the original main.py
        """
        messages = [{"role": "user", "content": question}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize with explicit attention mask creation
        model_inputs = self.tokenizer(
            [text], 
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        ).to(self.model.device)

        # Generate with explicit attention mask
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        return response
    
    def _generate_prompt(self, description: str) -> str:
        """
        Generate music prompt - extracted from main.py
        """
        # Insert description into template
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)
        # Run LLM inference and return that
        return self.prompt_qwen(full_prompt)

    def _generate_lyrics(self, description: str) -> str:
        """
        Generate lyrics - extracted from main.py
        """
        # Insert description into template
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)
        # Run LLM inference and return that
        return self.prompt_qwen(full_prompt)

    def _generate_categories(self, description: str) -> List[str]:
        """
        Generate categories - extracted from main.py
        """
        prompt = f"Based on the following music description, list 3-5 relevant genres or categories as a comma-separated list. For example: Pop, Electronic, Sad, 80s. Description: '{description}'"
        
        response_text = self.prompt_qwen(prompt)
        categories = [cat.strip() for cat in response_text.split(",") if cat.strip()]
        return categories
    
    # Service management methods
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": self.config.service_name,
            "gpu_type": self.config.gpu_type.value,
            "model_id": settings.llm_model_id,
            "supported_operations": ["lyrics", "prompt", "categories"],
            "max_text_length": settings.max_prompt_length,
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
    
    # API Endpoints
    @modal.fastapi_endpoint(method="POST")
    def generate_lyrics(self, request: LyricsGenerationRequest) -> LyricsGenerationResponse:
        """Generate lyrics from description"""
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Generating lyrics: {request.description[:100]}...")
        
        # Validate input
        if len(request.description.strip()) < 5:
            raise HTTPException(status_code=400, detail="Description too short")
        
        from shared.models.base import validate_text_input
        description = validate_text_input(request.description, min_length=5, max_length=1000)
        
        # Start monitoring
        self.start_operation(operation_id, "lyrics_generation")
        
        try:
            # Generate lyrics using main.py method
            lyrics = self._generate_lyrics(description)
            
            # End monitoring
            self.end_operation(operation_id, success=True)
            
            logger.info(f"[{operation_id}] Lyrics generated successfully")
            return LyricsGenerationResponse(lyrics=lyrics)
            
        except Exception as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Lyrics generation failed: {e}")
            raise HTTPException(status_code=500, detail="Lyrics generation failed")
    
    @modal.fastapi_endpoint(method="POST")
    def generate_prompt(self, request: PromptGenerationRequest) -> PromptGenerationResponse:
        """Generate music prompt from description"""
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Generating prompt: {request.description[:100]}...")
        
        # Validate input
        if len(request.description.strip()) < 5:
            raise HTTPException(status_code=400, detail="Description too short")
        
        from shared.models.base import validate_text_input
        description = validate_text_input(request.description, min_length=5, max_length=1000)
        
        # Start monitoring
        self.start_operation(operation_id, "prompt_generation")
        
        try:
            # Generate prompt using main.py method
            prompt = self._generate_prompt(description)
            
            # End monitoring
            self.end_operation(operation_id, success=True)
            
            logger.info(f"[{operation_id}] Prompt generated successfully")
            return PromptGenerationResponse(prompt=prompt)
            
        except Exception as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Prompt generation failed: {e}")
            raise HTTPException(status_code=500, detail="Prompt generation failed")
    
    @modal.fastapi_endpoint(method="POST")
    def generate_categories(self, request: CategoryGenerationRequest) -> CategoryGenerationResponse:
        """Generate music categories from description"""
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        logger.info(f"[{operation_id}] Generating categories: {request.description[:100]}...")
        
        # Validate input
        if len(request.description.strip()) < 5:
            raise HTTPException(status_code=400, detail="Description too short")
        
        from shared.models.base import validate_text_input
        description = validate_text_input(request.description, min_length=5, max_length=1000)
        
        # Start monitoring
        self.start_operation(operation_id, "category_generation")
        
        try:
            # Generate categories using main.py method
            categories = self._generate_categories(description)
            
            # Ensure we have categories
            if not categories:
                categories = ["Unknown"]
            
            # Limit to reasonable number
            categories = categories[:10]
            
            # End monitoring
            self.end_operation(operation_id, success=True)
            
            logger.info(f"[{operation_id}] Generated {len(categories)} categories")
            return CategoryGenerationResponse(categories=categories)
            
        except Exception as e:
            self.cleanup_operation(operation_id)
            logger.error(f"[{operation_id}] Category generation failed: {e}")
            raise HTTPException(status_code=500, detail="Category generation failed")
    
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
def test_lyrics_service():
    """Test the lyrics service"""
    import requests
    import time
    
    server = LyricsGenServer()
    
    print("=== Testing Lyrics Service ===")
    print(f"Service: {lyrics_config.service_name}")
    print(f"GPU: {lyrics_config.gpu_type.value}")
    print("-" * 50)
    
    # Wait for initialization
    print("Waiting for service initialization...")
    time.sleep(15)
    
    # Test health check
    try:
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
                    time.sleep(20)
            except Exception as e:
                print(f"Health check attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)
        
        # Test endpoints
        tests = [
            ("lyrics", LyricsGenerationRequest(description="a happy song about sunshine")),
            ("prompt", PromptGenerationRequest(description="upbeat electronic music")),
            ("categories", CategoryGenerationRequest(description="electronic dance music"))
        ]
        
        for test_name, request_obj in tests:
            try:
                endpoint = getattr(server, f"generate_{test_name}")
                url = endpoint.get_web_url()
                print(f"\nTesting {test_name}: {url}")
                
                response = requests.post(url, json=request_obj.model_dump(), timeout=90)
                response.raise_for_status()
                result = response.json()
                
                print(f"✓ {test_name} generation successful")
                if test_name == "lyrics" and "lyrics" in result:
                    print(f"  Preview: {result['lyrics']}")
                elif test_name == "prompt" and "prompt" in result:
                    print(f"  Result: {result['prompt']}")
                elif test_name == "categories" and "categories" in result:
                    print(f"  Categories: {result['categories']}")
                
            except Exception as e:
                print(f"❌ {test_name} test failed: {e}")
        
        print("\n" + "="*50)
        print("Testing completed!")
        
    except Exception as e:
        print(f"Test setup failed: {e}")


if __name__ == "__main__":
    test_lyrics_service()