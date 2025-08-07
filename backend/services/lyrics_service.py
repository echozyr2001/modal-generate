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
    LyricsGenerationRequestEnhanced,
    LyricsGenerationResponseEnhanced,
    PromptGenerationRequestEnhanced,
    PromptGenerationResponseEnhanced,
    CategoryGenerationRequestEnhanced,
    CategoryGenerationResponseEnhanced,
    ServiceConfig,
    GPUType,
    GenerationMetadata
)
from shared.modal_config import llm_image, hf_volume, music_gen_secrets
from shared.config import settings

from shared.utils import TimeoutManager, CostMonitor
from prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

app = modal.App("lyrics-generator")

# Create optimized configuration for lyrics service
lyrics_config = ServiceConfig(
    service_name="lyrics-generator",
    gpu_type=GPUType.T4,  # CPU or T4 for cost optimization
    scaledown_window=30,  # Fast scaledown for text generation
    max_runtime_seconds=60,  # Short timeout for text generation
    max_concurrent_requests=20,  # Higher concurrency for lightweight operations
    cost_per_hour=0.35  # T4 cost
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
            
            description = self.validate_text_input(request.description, max_length=1000)
            
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
            description = self.validate_text_input(request.description, max_length=1000)
            
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
            description = self.validate_text_input(request.description, max_length=1000)
            
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

    # Enhanced endpoints with comprehensive models and metadata
    @modal.fastapi_endpoint(method="POST")
    def generate_lyrics_enhanced(self, request: LyricsGenerationRequestEnhanced) -> LyricsGenerationResponseEnhanced:
        """Enhanced lyrics generation with comprehensive metadata"""
        logger.info(f"Enhanced lyrics generation for: {request.description[:100]}...")
        
        # Start operation monitoring
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Start monitoring
            self.cost_monitor.start_operation(
                operation_id, 
                lyrics_config.gpu_type.value, 
                lyrics_config.service_name
            )
            self.timeout_manager.start_timeout(operation_id, lyrics_config.max_runtime_seconds)
            
            # Build enhanced prompt with style and mood
            style_part = f" in {request.style} style" if request.style else ""
            mood_part = f" with {request.mood} mood" if request.mood else ""
            language_part = f" in {request.language}" if request.language != "english" else ""
            
            enhanced_description = f"{request.description}{style_part}{mood_part}{language_part}"
            full_prompt = LYRICS_GENERATOR_PROMPT.format(description=enhanced_description)
            
            # Generate lyrics
            lyrics = self.prompt_qwen(full_prompt, operation_id)
            
            # Analyze lyrics structure
            word_count = len(lyrics.split())
            structure_tags = self._extract_structure_tags(lyrics)
            
            # End monitoring and create metadata
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            metadata = self.create_metadata(
                operation_id, 
                f"{settings.llm_model_id} (Enhanced)", 
                start_time
            )
            
            logger.info(f"Enhanced lyrics generated: {word_count} words, {len(structure_tags)} structure tags")
            
            return LyricsGenerationResponseEnhanced(
                lyrics=lyrics,
                word_count=word_count,
                structure_tags=structure_tags,
                metadata=metadata
            )
            
        except Exception as e:
            # Cleanup monitoring on error
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            logger.error(f"Enhanced lyrics generation failed: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_prompt_enhanced(self, request: PromptGenerationRequestEnhanced) -> PromptGenerationResponseEnhanced:
        """Enhanced prompt generation with comprehensive metadata"""
        logger.info(f"Enhanced prompt generation for: {request.description[:100]}...")
        
        # Start operation monitoring
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Start monitoring
            self.cost_monitor.start_operation(
                operation_id, 
                lyrics_config.gpu_type.value, 
                lyrics_config.service_name
            )
            self.timeout_manager.start_timeout(operation_id, lyrics_config.max_runtime_seconds)
            
            # Build enhanced prompt with additional context
            context_parts = []
            if request.genre:
                context_parts.append(f"Genre: {request.genre}")
            if request.instruments:
                context_parts.append(f"Instruments: {', '.join(request.instruments)}")
            if request.tempo:
                context_parts.append(f"Tempo: {request.tempo}")
            
            enhanced_description = request.description
            if context_parts:
                enhanced_description += f" ({'; '.join(context_parts)})"
            
            full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=enhanced_description)
            prompt = self.prompt_qwen(full_prompt, operation_id)
            
            # Analyze prompt
            tag_count = len([tag.strip() for tag in prompt.split(',') if tag.strip()])
            detected_genre = self._detect_primary_genre(prompt)
            
            # End monitoring and create metadata
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            metadata = self.create_metadata(
                operation_id, 
                f"{settings.llm_model_id} (Enhanced)", 
                start_time
            )
            
            logger.info(f"Enhanced prompt generated: {tag_count} tags, genre: {detected_genre}")
            
            return PromptGenerationResponseEnhanced(
                prompt=prompt,
                tag_count=tag_count,
                detected_genre=detected_genre,
                metadata=metadata
            )
            
        except Exception as e:
            # Cleanup monitoring on error
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            logger.error(f"Enhanced prompt generation failed: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_categories_enhanced(self, request: CategoryGenerationRequestEnhanced) -> CategoryGenerationResponseEnhanced:
        """Enhanced category generation with comprehensive metadata"""
        logger.info(f"Enhanced category generation for: {request.description[:100]}...")
        
        # Start operation monitoring
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Start monitoring
            self.cost_monitor.start_operation(
                operation_id, 
                lyrics_config.gpu_type.value, 
                lyrics_config.service_name
            )
            self.timeout_manager.start_timeout(operation_id, lyrics_config.max_runtime_seconds)
            
            # Build enhanced prompt
            subgenre_instruction = " Include subgenres where relevant." if request.include_subgenres else " Focus on main genres only."
            
            prompt = (
                f"Based on the following music description, list {request.max_categories} relevant genres or categories "
                f"as a comma-separated list.{subgenre_instruction} "
                f"For example: Pop, Electronic, Sad, 80s. "
                f"Description: '{request.description}'"
            )
            
            response_text = self.prompt_qwen(prompt, operation_id)
            categories = [cat.strip() for cat in response_text.split(",") if cat.strip()]
            
            # Limit to requested number
            categories = categories[:request.max_categories]
            
            # Ensure we have at least some categories
            if not categories:
                categories = ["Unknown"]
            
            # Analyze categories
            primary_genre = self._identify_primary_genre(categories)
            confidence_scores = self._calculate_confidence_scores(categories, request.description)
            
            # End monitoring and create metadata
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            
            metadata = self.create_metadata(
                operation_id, 
                f"{settings.llm_model_id} (Enhanced)", 
                start_time
            )
            
            logger.info(f"Enhanced categories generated: {len(categories)} categories, primary: {primary_genre}")
            
            return CategoryGenerationResponseEnhanced(
                categories=categories,
                primary_genre=primary_genre,
                confidence_scores=confidence_scores,
                metadata=metadata
            )
            
        except Exception as e:
            # Cleanup monitoring on error
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
            logger.error(f"Enhanced category generation failed: {e}")
            raise

    def _extract_structure_tags(self, lyrics: str) -> List[str]:
        """Extract structure tags from lyrics"""
        import re
        tags = re.findall(r'\[([^\]]+)\]', lyrics.lower())
        return list(set(tags))  # Remove duplicates

    def _detect_primary_genre(self, prompt: str) -> Optional[str]:
        """Detect primary genre from prompt tags"""
        common_genres = [
            'pop', 'rock', 'electronic', 'hip hop', 'rap', 'jazz', 'blues', 
            'country', 'folk', 'classical', 'reggae', 'punk', 'metal', 'funk'
        ]
        
        prompt_lower = prompt.lower()
        for genre in common_genres:
            if genre in prompt_lower:
                return genre.title()
        return None

    def _identify_primary_genre(self, categories: List[str]) -> Optional[str]:
        """Identify primary genre from categories list"""
        if not categories:
            return None
        
        # First category is usually the primary genre
        primary = categories[0].lower()
        
        # Map common variations to standard genres
        genre_mapping = {
            'edm': 'Electronic',
            'techno': 'Electronic',
            'house': 'Electronic',
            'hiphop': 'Hip Hop',
            'r&b': 'R&B',
            'rnb': 'R&B'
        }
        
        return genre_mapping.get(primary, categories[0])

    def _calculate_confidence_scores(self, categories: List[str], description: str) -> Dict[str, float]:
        """Calculate confidence scores for categories based on description"""
        scores = {}
        description_lower = description.lower()
        
        for category in categories:
            # Simple confidence based on keyword presence
            category_lower = category.lower()
            if category_lower in description_lower:
                scores[category] = 0.9
            elif any(word in description_lower for word in category_lower.split()):
                scores[category] = 0.7
            else:
                scores[category] = 0.5
        
        return scores

    def _start_monitoring(self, operation_id: str):
        """Safely start monitoring"""
        try:
            if hasattr(self, 'cost_monitor'):
                self.cost_monitor.start_operation(
                    operation_id, 
                    lyrics_config.gpu_type.value, 
                    lyrics_config.service_name
                )
            if hasattr(self, 'timeout_manager'):
                self.timeout_manager.start_timeout(operation_id, lyrics_config.max_runtime_seconds)
        except Exception as e:
            logger.warning(f"Failed to start monitoring for {operation_id}: {e}")
    
    def _end_monitoring(self, operation_id: str):
        """Safely end monitoring"""
        try:
            if hasattr(self, 'cost_monitor'):
                self.cost_monitor.end_operation(operation_id)
            if hasattr(self, 'timeout_manager'):
                self.timeout_manager.end_timeout(operation_id)
        except Exception as e:
            logger.warning(f"Failed to end monitoring for {operation_id}: {e}")

    def _cleanup_operation(self, operation_id: str):
        """Cleanup monitoring for failed operations"""
        self._end_monitoring(operation_id)
    
    def validate_text_input(self, text: str, max_length: int = 1000) -> str:
        """Validate and clean text input"""
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
        
        # Clean text
        cleaned = text.strip()
        
        # Check length
        if len(cleaned) > max_length:
            raise ValueError(f"Text too long. Maximum length: {max_length}")
        
        return cleaned

    def create_metadata(self, operation_id: str, model_info: str, start_time: float) -> GenerationMetadata:
        """Create metadata for enhanced responses"""
        generation_time = time.time() - start_time
        estimated_cost = generation_time * (lyrics_config.cost_per_hour / 3600)  # Convert to per-second cost
        
        return GenerationMetadata(
            operation_id=operation_id,
            model_info=model_info,
            generation_time=generation_time,
            estimated_cost=estimated_cost,
            gpu_type=lyrics_config.gpu_type.value
        )

    def _validate_request_size(self, request_data: str, max_size: int = 10000):
        """Validate request size to prevent abuse"""
        if len(request_data.encode('utf-8')) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"Request too large. Maximum size: {max_size} bytes"
            )


@app.local_entrypoint()
def test_lyrics_generation():
    """测试优化后的歌词生成服务"""
    import requests
    import time
    
    server = LyricsGenServer()
    
    print(f"Testing Lyrics Generation Service")
    print(f"Configuration: {lyrics_config.service_name} on {lyrics_config.gpu_type.value}")
    print(f"Scaledown window: {lyrics_config.scaledown_window}s")
    print(f"Max runtime: {lyrics_config.max_runtime_seconds}s")
    print("-" * 50)
    
    # Wait a bit for the service to fully initialize
    print("Waiting for service to initialize...")
    time.sleep(5)
    
    # 测试健康检查
    try:
        health_url = server.health_check.get_web_url()
        print(f"Health check URL: {health_url}")
        response = requests.get(health_url, timeout=30)
        response.raise_for_status()
        health = response.json()
        print(f"Health check: {health}")
        
        if not health.get('model_loaded', False):
            print("WARNING: Model not loaded yet, waiting...")
            time.sleep(10)
            
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test basic endpoints first
    basic_tests = [
        ("generate_lyrics", LyricsGenerationRequest(description="a happy song about sunshine")),
        ("generate_prompt", PromptGenerationRequest(description="upbeat electronic music")),
        ("generate_categories", CategoryGenerationRequest(description="electronic dance music"))
    ]
    
    for endpoint_name, request_obj in basic_tests:
        try:
            endpoint = getattr(server, endpoint_name)
            url = endpoint.get_web_url()
            print(f"Testing {endpoint_name} at {url}")
            
            response = requests.post(url, json=request_obj.model_dump(), timeout=60)
            response.raise_for_status()
            result = response.json()
            
            print(f"✓ {endpoint_name} succeeded")
            if 'lyrics' in result:
                print(f"  Lyrics preview: {result['lyrics'][:100]}...")
            elif 'prompt' in result:
                print(f"  Prompt: {result['prompt']}")
            elif 'categories' in result:
                print(f"  Categories: {result['categories']}")
            print("-" * 50)
            
        except requests.exceptions.Timeout:
            print(f"✗ {endpoint_name} timed out")
        except requests.exceptions.RequestException as e:
            print(f"✗ {endpoint_name} failed with request error: {e}")
        except Exception as e:
            print(f"✗ {endpoint_name} failed: {e}")
    
    print("Basic testing completed!")
    
    # Only test enhanced endpoints if basic ones work
    print("Testing enhanced endpoints...")
    
    enhanced_tests = [
        ("generate_lyrics_enhanced", LyricsGenerationRequestEnhanced(
            description="a melancholic song about rain",
            style="ballad",
            mood="sad"
        )),
        ("generate_prompt_enhanced", PromptGenerationRequestEnhanced(
            description="upbeat dance music",
            genre="electronic",
            instruments=["synthesizer"]
        )),
        ("generate_categories_enhanced", CategoryGenerationRequestEnhanced(
            description="electronic dance music",
            max_categories=5
        ))
    ]
    
    for endpoint_name, request_obj in enhanced_tests:
        try:
            endpoint = getattr(server, endpoint_name)
            url = endpoint.get_web_url()
            print(f"Testing {endpoint_name}")
            
            response = requests.post(url, json=request_obj.model_dump(), timeout=60)
            response.raise_for_status()
            result = response.json()
            
            print(f"✓ {endpoint_name} succeeded")
            print("-" * 50)
            
        except requests.exceptions.Timeout:
            print(f"✗ {endpoint_name} timed out")
        except Exception as e:
            print(f"✗ {endpoint_name} failed: {e}")
    
    print("All testing completed!")


if __name__ == "__main__":
    test_lyrics_generation()