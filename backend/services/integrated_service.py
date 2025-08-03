import modal
import logging
import requests
from typing import Optional
from shared.models import (
    GenerateFromDescriptionRequest,
    GenerateWithCustomLyricsRequest,
    GenerateWithDescribedLyricsRequest,
    GenerateMusicResponseS3,
    LyricsGenerationRequest,
    PromptGenerationRequest,
    CategoryGenerationRequest,
    CoverImageGenerationRequest
)
from shared.modal_config import base_image, music_gen_secrets
from shared.config import settings
from services.music_service import MusicGenerationRequest

logger = logging.getLogger(__name__)

app = modal.App("music-generator-integrated")


@app.cls(
    image=base_image,
    secrets=[music_gen_secrets],
    scaledown_window=settings.scaledown_window
)
class IntegratedMusicGenServer:
    @modal.enter()
    def setup(self):
        """初始化服务URL"""
        # 这些URL在实际部署时需要配置为实际的服务端点
        self.lyrics_service_url = "https://your-lyrics-service-url"
        self.cover_service_url = "https://your-cover-service-url"
        self.music_service_url = "https://your-music-service-url"
        
        logger.info("Integrated service initialized")

    def call_lyrics_service(self, endpoint: str, data: dict) -> dict:
        """调用歌词服务"""
        try:
            url = f"{self.lyrics_service_url}/{endpoint}"
            response = requests.post(url, json=data, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to call lyrics service {endpoint}: {e}")
            raise

    def call_cover_service(self, data: dict) -> dict:
        """调用封面生成服务"""
        try:
            url = f"{self.cover_service_url}/generate_cover_image"
            response = requests.post(url, json=data, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to call cover service: {e}")
            raise

    def call_music_service(self, data: dict) -> dict:
        """调用音乐生成服务"""
        try:
            url = f"{self.music_service_url}/generate_music_to_storage"
            response = requests.post(url, json=data, timeout=600)  # 音乐生成需要更长时间
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to call music service: {e}")
            raise

    def generate_complete_music(
        self,
        prompt: str,
        lyrics: str,
        description_for_categorization: str,
        audio_duration: float = 180.0,
        seed: int = -1,
        guidance_scale: float = 15.0,
        infer_step: int = 60,
        instrumental: bool = False
    ) -> GenerateMusicResponseS3:
        """完整的音乐生成流程"""
        logger.info("Starting complete music generation...")
        
        try:
            # 1. 生成音乐
            music_request = {
                "prompt": prompt,
                "lyrics": lyrics,
                "audio_duration": audio_duration,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "infer_step": infer_step,
                "instrumental": instrumental
            }
            music_response = self.call_music_service(music_request)
            
            # 2. 生成封面图
            cover_request = {
                "prompt": prompt,
                "style": "album cover art"
            }
            cover_response = self.call_cover_service(cover_request)
            
            # 3. 生成分类
            category_request = {"description": description_for_categorization}
            category_response = self.call_lyrics_service("generate_categories", category_request)
            
            logger.info("Complete music generation finished successfully")
            
            return GenerateMusicResponseS3(
                s3_key=music_response["file_path"],
                cover_image_s3_key=cover_response["s3_key"],
                categories=category_response["categories"]
            )
            
        except Exception as e:
            logger.error(f"Failed to generate complete music: {e}")
            raise

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_from_description(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponseS3:
        """基于描述生成完整音乐作品"""
        logger.info(f"Generating from description: {request.full_described_song[:100]}...")
        
        try:
            # 1. 生成提示词
            prompt_request = {"description": request.full_described_song}
            prompt_response = self.call_lyrics_service("generate_prompt", prompt_request)
            prompt = prompt_response["prompt"]
            
            # 2. 生成歌词（如果不是纯音乐）
            lyrics = ""
            if not request.instrumental:
                lyrics_request = {"description": request.full_described_song}
                lyrics_response = self.call_lyrics_service("generate_lyrics", lyrics_request)
                lyrics = lyrics_response["lyrics"]
            
            # 3. 生成完整音乐作品
            return self.generate_complete_music(
                prompt=prompt,
                lyrics=lyrics,
                description_for_categorization=request.full_described_song,
                audio_duration=request.audio_duration,
                seed=request.seed,
                guidance_scale=request.guidance_scale,
                infer_step=request.infer_step,
                instrumental=request.instrumental
            )
            
        except Exception as e:
            logger.error(f"Failed to generate from description: {e}")
            raise

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_lyrics(self, request: GenerateWithCustomLyricsRequest) -> GenerateMusicResponseS3:
        """使用自定义歌词生成音乐"""
        logger.info(f"Generating with custom lyrics, prompt: {request.prompt[:100]}...")
        
        return self.generate_complete_music(
            prompt=request.prompt,
            lyrics=request.lyrics,
            description_for_categorization=request.prompt,
            audio_duration=request.audio_duration,
            seed=request.seed,
            guidance_scale=request.guidance_scale,
            infer_step=request.infer_step,
            instrumental=request.instrumental
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_described_lyrics(self, request: GenerateWithDescribedLyricsRequest) -> GenerateMusicResponseS3:
        """基于歌词描述生成音乐"""
        logger.info(f"Generating with described lyrics: {request.described_lyrics[:100]}...")
        
        try:
            # 生成歌词（如果不是纯音乐）
            lyrics = ""
            if not request.instrumental:
                lyrics_request = {"description": request.described_lyrics}
                lyrics_response = self.call_lyrics_service("generate_lyrics", lyrics_request)
                lyrics = lyrics_response["lyrics"]
            
            return self.generate_complete_music(
                prompt=request.prompt,
                lyrics=lyrics,
                description_for_categorization=request.prompt,
                audio_duration=request.audio_duration,
                seed=request.seed,
                guidance_scale=request.guidance_scale,
                infer_step=request.infer_step,
                instrumental=request.instrumental
            )
            
        except Exception as e:
            logger.error(f"Failed to generate with described lyrics: {e}")
            raise


@app.local_entrypoint()
def test_integrated_service():
    """测试集成服务"""
    server = IntegratedMusicGenServer()
    
    # 测试完整描述生成
    request = GenerateFromDescriptionRequest(
        full_described_song="upbeat electronic dance music with heavy bass",
        audio_duration=30,  # 短一些用于测试
        infer_step=20
    )
    
    try:
        response = server.generate_from_description(request)
        print(f"Generated complete music:")
        print(f"  Audio S3 key: {response.s3_key}")
        print(f"  Cover S3 key: {response.cover_image_s3_key}")
        print(f"  Categories: {response.categories}")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_integrated_service()