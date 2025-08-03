import modal
import logging
import base64
import requests
from shared.models import (
    AudioGenerationBase,
    GenerateMusicResponse
)
from shared.modal_config import music_image, model_volume, hf_volume, music_gen_secrets
from shared.config import settings
from shared.utils import ensure_output_dir, generate_temp_filepath, FileManager

logger = logging.getLogger(__name__)

app = modal.App("music-generator-core")


class MusicGenerationRequest(AudioGenerationBase):
    prompt: str
    lyrics: str


class MusicGenerationResponseLocal(AudioGenerationBase):
    file_path: str


@app.cls(
    image=music_image,
    gpu=settings.gpu_type,
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=settings.scaledown_window
)
class MusicGenCoreServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        
        logger.info("Loading ACE-Step music generation model...")
        
        self.music_model = ACEStepPipeline(
            checkpoint_dir=settings.music_model_checkpoint_dir,
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )
        
        # 初始化文件管理器
        self.file_manager = FileManager(
            use_s3=settings.use_s3_storage,
            local_storage_dir=settings.local_storage_dir
        )
        
        logger.info("Music generation model loaded successfully")

    @modal.fastapi_endpoint(method="POST")
    def generate_music(self, request: MusicGenerationRequest) -> GenerateMusicResponse:
        """生成音乐并返回base64编码的音频数据"""
        logger.info(f"Generating music with prompt: {request.prompt[:100]}...")
        
        try:
            # 处理歌词
            final_lyrics = "[instrumental]" if request.instrumental else request.lyrics
            
            # 生成音乐
            output_dir = ensure_output_dir()
            output_path = generate_temp_filepath(output_dir, ".wav")
            
            self.music_model(
                prompt=request.prompt,
                lyrics=final_lyrics,
                audio_duration=request.audio_duration,
                infer_step=request.infer_step,
                guidance_scale=request.guidance_scale,
                save_path=output_path,
                manual_seeds=str(request.seed)
            )
            
            # 读取音频文件并编码
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            # 清理临时文件
            import os
            os.remove(output_path)
            
            logger.info("Music generated successfully")
            return GenerateMusicResponse(audio_data=audio_b64)
            
        except Exception as e:
            logger.error(f"Failed to generate music: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_music_to_storage(self, request: MusicGenerationRequest) -> MusicGenerationResponseLocal:
        """生成音乐并保存到存储（本地或S3）"""
        logger.info(f"Generating music to storage with prompt: {request.prompt[:100]}...")
        
        try:
            # 处理歌词
            final_lyrics = "[instrumental]" if request.instrumental else request.lyrics
            
            logger.info(f"Generated lyrics: {final_lyrics[:100]}...")
            logger.info(f"Prompt: {request.prompt}")
            
            # 生成音乐
            output_dir = ensure_output_dir()
            output_path = generate_temp_filepath(output_dir, ".wav")
            
            self.music_model(
                prompt=request.prompt,
                lyrics=final_lyrics,
                audio_duration=request.audio_duration,
                infer_step=request.infer_step,
                guidance_scale=request.guidance_scale,
                save_path=output_path,
                manual_seeds=str(request.seed)
            )
            
            # 保存文件（本地或S3）
            file_key = self.file_manager.save_file(output_path)
            
            logger.info(f"Music generated and saved: {file_key}")
            return MusicGenerationResponseLocal(file_path=file_key)
            
        except Exception as e:
            logger.error(f"Failed to generate music to storage: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_demo_music(self) -> GenerateMusicResponse:
        """生成演示音乐（用于测试）"""
        logger.info("Generating demo music...")
        
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
Living on the edge, never succumb.""",
            audio_duration=60,  # 短一些用于演示
            infer_step=30,
            guidance_scale=15
        )
        
        return self.generate_music(demo_request)


@app.local_entrypoint()
def test_music_generation():
    """测试音乐生成服务"""
    server = MusicGenCoreServer()
    
    # 测试音乐生成
    request = MusicGenerationRequest(
        prompt="upbeat electronic dance music, 128 bpm",
        lyrics="[verse]\nDancing through the night\nLights are shining bright\n[chorus]\nFeel the beat tonight\nEverything's alright",
        audio_duration=30,  # 短一些用于测试
        infer_step=20
    )
    
    # 测试返回base64
    response = server.generate_music(request)
    print(f"Generated music (base64 length): {len(response.audio_data)}")
    
    # 测试保存到存储
    storage_response = server.generate_music_to_storage(request)
    print(f"Generated music saved to: {storage_response.file_path}")


if __name__ == "__main__":
    test_music_generation()