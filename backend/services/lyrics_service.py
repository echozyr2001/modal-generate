import modal
import logging
from shared.models import (
    LyricsGenerationRequest, 
    LyricsGenerationResponse,
    PromptGenerationRequest,
    PromptGenerationResponse,
    CategoryGenerationRequest,
    CategoryGenerationResponse
)
from shared.modal_config import llm_image, hf_volume, music_gen_secrets
from shared.config import settings
from prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT
from typing import List

logger = logging.getLogger(__name__)

app = modal.App("lyrics-generator")


@app.cls(
    image=llm_image,
    gpu=settings.gpu_type,
    volumes={"/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=settings.scaledown_window
)
class LyricsGenServer:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading LLM model: {settings.llm_model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=settings.hf_cache_dir
        )
        
        logger.info("LLM model loaded successfully")

    def prompt_qwen(self, question: str) -> str:
        """通用的LLM推理方法"""
        try:
            messages = [{"role": "user", "content": question}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer(
                [text], return_tensors="pt"
            ).to(self.llm_model.device)

            generated_ids = self.llm_model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return response.strip()
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_lyrics(self, request: LyricsGenerationRequest) -> LyricsGenerationResponse:
        """生成歌词"""
        logger.info(f"Generating lyrics for description: {request.description[:100]}...")
        
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description=request.description)
        lyrics = self.prompt_qwen(full_prompt)
        
        logger.info("Lyrics generated successfully")
        return LyricsGenerationResponse(lyrics=lyrics)

    @modal.fastapi_endpoint(method="POST")
    def generate_prompt(self, request: PromptGenerationRequest) -> PromptGenerationResponse:
        """生成音乐提示词"""
        logger.info(f"Generating prompt for description: {request.description[:100]}...")
        
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=request.description)
        prompt = self.prompt_qwen(full_prompt)
        
        logger.info("Prompt generated successfully")
        return PromptGenerationResponse(prompt=prompt)

    @modal.fastapi_endpoint(method="POST")
    def generate_categories(self, request: CategoryGenerationRequest) -> CategoryGenerationResponse:
        """生成音乐分类"""
        logger.info(f"Generating categories for description: {request.description[:100]}...")
        
        prompt = (
            f"Based on the following music description, list 3-5 relevant genres or categories "
            f"as a comma-separated list. For example: Pop, Electronic, Sad, 80s. "
            f"Description: '{request.description}'"
        )
        
        response_text = self.prompt_qwen(prompt)
        categories = [cat.strip() for cat in response_text.split(",") if cat.strip()]
        
        logger.info(f"Generated categories: {categories}")
        return CategoryGenerationResponse(categories=categories)


@app.local_entrypoint()
def test_lyrics_generation():
    """测试歌词生成服务"""
    import requests
    server = LyricsGenServer()
    
    # 测试歌词生成
    lyrics_request = LyricsGenerationRequest(description="a sad song about lost love")
    lyrics_endpoint_url = server.generate_lyrics.get_web_url()
    
    response = requests.post(lyrics_endpoint_url, json=lyrics_request.model_dump())
    response.raise_for_status()
    lyrics_response = LyricsGenerationResponse(**response.json())
    print(f"Generated lyrics:\n{lyrics_response.lyrics}")
    
    # 测试提示词生成
    prompt_request = PromptGenerationRequest(description="upbeat electronic dance music")
    prompt_endpoint_url = server.generate_prompt.get_web_url()
    
    response = requests.post(prompt_endpoint_url, json=prompt_request.model_dump())
    response.raise_for_status()
    prompt_response = PromptGenerationResponse(**response.json())
    print(f"Generated prompt: {prompt_response.prompt}")
    
    # 测试分类生成
    category_request = CategoryGenerationRequest(description="electronic dance music with heavy bass")
    category_endpoint_url = server.generate_categories.get_web_url()
    
    response = requests.post(category_endpoint_url, json=category_request.model_dump())
    response.raise_for_status()
    category_response = CategoryGenerationResponse(**response.json())
    print(f"Generated categories: {category_response.categories}")


if __name__ == "__main__":
    test_lyrics_generation()