import modal
from shared.config import settings

# 基础镜像配置
base_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HOME": settings.hf_cache_dir})
)

# 音乐生成专用镜像
music_image = base_image.run_commands([
    "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
    "cd /tmp/ACE-Step && pip install ."
])

# LLM专用镜像
llm_image = base_image.pip_install("transformers", "torch", "accelerate").add_local_python_source("prompts").add_local_python_source("shared")

# 图像生成专用镜像
image_gen_image = base_image.pip_install("diffusers", "torch")

# 存储卷
model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

# 密钥
music_gen_secrets = modal.Secret.from_name("music-gen-secret")