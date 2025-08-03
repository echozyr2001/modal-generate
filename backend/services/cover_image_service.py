import modal
import logging
import torch
import requests
from shared.models import CoverImageGenerationRequest, CoverImageGenerationResponse
from shared.modal_config import image_gen_image, hf_volume, music_gen_secrets
from shared.config import settings
from shared.utils import FileManager, ensure_output_dir, generate_temp_filepath

logger = logging.getLogger(__name__)

app = modal.App("cover-image-generator")


@app.cls(
    image=image_gen_image,
    gpu=settings.gpu_type,
    volumes={"/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=settings.scaledown_window
)
class CoverImageGenServer:
    @modal.enter()
    def load_model(self):
        from diffusers import AutoPipelineForText2Image
        
        logger.info(f"Loading image generation model: {settings.image_model_id}")
        
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            settings.image_model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=settings.hf_cache_dir
        )
        self.image_pipe.to("cuda")
        
        # 初始化文件管理器
        self.file_manager = FileManager(
            use_s3=settings.use_s3_storage,
            local_storage_dir=settings.local_storage_dir
        )
        
        logger.info("Image generation model loaded successfully")

    @modal.fastapi_endpoint(method="POST")
    def generate_cover_image(self, request: CoverImageGenerationRequest) -> CoverImageGenerationResponse:
        """生成封面图片"""
        logger.info(f"Generating cover image for prompt: {request.prompt[:100]}...")
        
        try:
            # 构建完整的图像生成提示词
            full_prompt = f"{request.prompt}, {request.style}"
            
            # 生成图像
            image = self.image_pipe(
                prompt=full_prompt,
                num_inference_steps=2,
                guidance_scale=0.0
            ).images[0]
            
            # 保存到临时文件
            output_dir = ensure_output_dir()
            image_path = generate_temp_filepath(output_dir, ".png")
            image.save(image_path)
            
            # 保存文件（本地或S3）
            file_key = self.file_manager.save_file(image_path)
            
            logger.info(f"Cover image generated and saved: {file_key}")
            return CoverImageGenerationResponse(s3_key=file_key)
            
        except Exception as e:
            logger.error(f"Failed to generate cover image: {e}")
            raise

    @modal.fastapi_endpoint(method="POST")
    def generate_cover_image_batch(self, requests: list[CoverImageGenerationRequest]) -> list[CoverImageGenerationResponse]:
        """批量生成封面图片"""
        logger.info(f"Generating {len(requests)} cover images...")
        
        results = []
        for i, request in enumerate(requests):
            try:
                result = self.generate_cover_image(request)
                results.append(result)
                logger.info(f"Batch progress: {i+1}/{len(requests)}")
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                # 可以选择继续处理其他图片或者抛出异常
                raise
        
        return results


@app.local_entrypoint()
def test_cover_generation():
    """测试封面图生成服务"""
    server = CoverImageGenServer()
    
    # 测试单个图片生成
    request = CoverImageGenerationRequest(
        prompt="electronic music, neon lights, futuristic",
        style="album cover art, professional"
    )
    
    endpoint_url = server.generate_cover_image.get_web_url()
    response = requests.post(endpoint_url, json=request.model_dump())
    response.raise_for_status()
    cover_response = CoverImageGenerationResponse(**response.json())
    print(f"Generated cover image: {cover_response.s3_key}")
    
    # 测试批量生成
    batch_requests = [
        CoverImageGenerationRequest(prompt="rock music, guitar", style="album cover art"),
        CoverImageGenerationRequest(prompt="jazz music, saxophone", style="vintage album cover"),
    ]
    
    batch_endpoint_url = server.generate_cover_image_batch.get_web_url()
    response = requests.post(batch_endpoint_url, json=[req.model_dump() for req in batch_requests])
    response.raise_for_status()
    batch_responses = [CoverImageGenerationResponse(**item) for item in response.json()]
    print(f"Generated {len(batch_responses)} cover images")
    for i, response in enumerate(batch_responses):
        print(f"  Image {i+1}: {response.s3_key}")


if __name__ == "__main__":
    test_cover_generation()