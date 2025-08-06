import modal
import logging
import time
import uuid
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
from shared.service_base import TextGenerationService
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
class LyricsGenServer(TextGenerationService):
    def __init__(self):
        super().__init__(lyrics_config)
        self.cost_monitor = CostMonitor()
        self.timeout_manager = TimeoutManager(lyrics_config.max_runtime_seconds)

    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading LLM model: {settings.llm_model_id} on {lyrics_config.gpu_type.value}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.llm_model_id,
                cache_dir=settings.hf_cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.llm_model_id,
                torch_dtype="auto",
                device_map="auto",
                cache_dir=settings.hf_cache_dir
            )
            
            self._model_loaded = True
            logger.info(f"LLM model loaded successfully on {lyrics_config.gpu_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, request: Any) -> Dict[str, Any]:
        """Base generate method implementation"""
        # This will be called by specific endpoint methods
        pass

    def prompt_qwen(self, question: str, operation_id: str) -> str:
        """通用的LLM推理方法 with timeout protection and error handling"""
        try:
            # Check timeout before starting generation
            if self.timeout_manager.check_timeout(operation_id):
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
            model_inputs = self.tokenizer(
                [text], return_tensors="pt"
            ).to(self.model.device)

            # Check timeout before generation
            if self.timeout_manager.check_timeout(operation_id):
                raise TimeoutError(f"Operation {operation_id} timed out before model generation")
            
            # Generate with error handling
            try:
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError("GPU out of memory. Try again later.")
                raise RuntimeError(f"Model generation failed: {e}")
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

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
            "model_loaded": self._model_loaded,
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

    def _cleanup_operation(self, operation_id: str):
        """Cleanup monitoring for failed operations"""
        try:
            self.cost_monitor.end_operation(operation_id)
            self.timeout_manager.end_timeout(operation_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup operation {operation_id}: {e}")

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
    server = LyricsGenServer()
    
    print(f"Testing Lyrics Generation Service")
    print(f"Configuration: {lyrics_config.service_name} on {lyrics_config.gpu_type.value}")
    print(f"Scaledown window: {lyrics_config.scaledown_window}s")
    print(f"Max runtime: {lyrics_config.max_runtime_seconds}s")
    print("-" * 50)
    
    # 测试健康检查
    try:
        health = server.health_check()
        print(f"Health check: {health}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # 测试歌词生成
    try:
        lyrics_request = LyricsGenerationRequest(description="a sad song about lost love")
        lyrics_response = server.generate_lyrics(lyrics_request)
        print(f"Generated lyrics:\n{lyrics_response.lyrics}")
        print("-" * 50)
    except Exception as e:
        print(f"Lyrics generation failed: {e}")
    
    # 测试提示词生成
    try:
        prompt_request = PromptGenerationRequest(description="upbeat electronic dance music")
        prompt_response = server.generate_prompt(prompt_request)
        print(f"Generated prompt: {prompt_response.prompt}")
        print("-" * 50)
    except Exception as e:
        print(f"Prompt generation failed: {e}")
    
    # 测试分类生成
    try:
        category_request = CategoryGenerationRequest(description="electronic dance music with heavy bass")
        category_response = server.generate_categories(category_request)
        print(f"Generated categories: {category_response.categories}")
        print("-" * 50)
    except Exception as e:
        print(f"Category generation failed: {e}")
    
    # 测试增强版歌词生成
    try:
        enhanced_lyrics_request = LyricsGenerationRequestEnhanced(
            description="a melancholic song about lost love",
            style="ballad",
            mood="sad",
            language="english"
        )
        enhanced_lyrics_response = server.generate_lyrics_enhanced(enhanced_lyrics_request)
        print(f"Enhanced lyrics (words: {enhanced_lyrics_response.word_count}):")
        print(f"Structure tags: {enhanced_lyrics_response.structure_tags}")
        print(f"Generation time: {enhanced_lyrics_response.metadata.generation_time:.2f}s")
        print(f"Estimated cost: ${enhanced_lyrics_response.metadata.estimated_cost:.4f}")
        print("-" * 50)
    except Exception as e:
        print(f"Enhanced lyrics generation failed: {e}")
    
    # 测试增强版提示词生成
    try:
        enhanced_prompt_request = PromptGenerationRequestEnhanced(
            description="upbeat dance music",
            genre="electronic",
            instruments=["synthesizer", "drums"],
            tempo="128 bpm"
        )
        enhanced_prompt_response = server.generate_prompt_enhanced(enhanced_prompt_request)
        print(f"Enhanced prompt (tags: {enhanced_prompt_response.tag_count}):")
        print(f"Detected genre: {enhanced_prompt_response.detected_genre}")
        print(f"Prompt: {enhanced_prompt_response.prompt}")
        print(f"Generation time: {enhanced_prompt_response.metadata.generation_time:.2f}s")
        print("-" * 50)
    except Exception as e:
        print(f"Enhanced prompt generation failed: {e}")
    
    # 测试增强版分类生成
    try:
        enhanced_category_request = CategoryGenerationRequestEnhanced(
            description="electronic dance music with heavy bass and synthesizers",
            max_categories=7,
            include_subgenres=True
        )
        enhanced_category_response = server.generate_categories_enhanced(enhanced_category_request)
        print(f"Enhanced categories: {enhanced_category_response.categories}")
        print(f"Primary genre: {enhanced_category_response.primary_genre}")
        print(f"Confidence scores: {enhanced_category_response.confidence_scores}")
        print(f"Generation time: {enhanced_category_response.metadata.generation_time:.2f}s")
        print("-" * 50)
    except Exception as e:
        print(f"Enhanced category generation failed: {e}")
    
    print("All testing completed!")


if __name__ == "__main__":
    test_lyrics_generation()