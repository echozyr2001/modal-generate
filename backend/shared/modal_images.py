"""
Modal image configurations for modular AI services.

This module defines optimized Modal images for different service types:
- Text generation services (lyrics, prompts, categories)
- Music generation services (ACE-Step pipeline)
- Image generation services (SDXL-Turbo)
- Integration services (orchestration)

Each image is optimized for its specific use case with appropriate dependencies
and GPU requirements as specified in requirements 2.1, 2.2, and 2.3.
"""

import modal
from shared.config import settings

# Base image with common dependencies
base_image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "wget")
    .pip_install("pydantic", "pydantic-settings", "fastapi[standard]", "boto3", "tenacity")
    .env({"HF_HOME": settings.hf_cache_dir})
    .add_local_python_source("shared")
)

# Text generation image for lyrics, prompts, and categorization services
# Optimized for CPU/low-cost GPU usage (Requirement 2.1)
text_generation_image = (
    base_image
    .pip_install(
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "accelerate>=0.24.0",
        "tokenizers>=0.15.0"
    )
    .add_local_python_source("prompts")
    .env({
        "TRANSFORMERS_CACHE": "/.cache/huggingface/transformers",
        "HF_DATASETS_CACHE": "/.cache/huggingface/datasets"
    })
)

# Music generation image with ACE-Step pipeline
# Optimized for high-memory GPU instances (Requirement 2.3)
music_generation_image = (
    base_image
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0"
    )
    .run_commands([
        "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
        "cd /tmp/ACE-Step && pip install ."
    ])
    .env({
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0"
    })
)

# Image generation image with SDXL-Turbo
# Optimized for mid-tier GPU instances (Requirement 2.2)
image_generation_image = (
    base_image
    .pip_install(
        "diffusers>=0.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "Pillow>=10.0.0",
        "safetensors>=0.4.0"
    )
    .env({
        "DIFFUSERS_CACHE": "/.cache/huggingface/diffusers",
        "TRANSFORMERS_CACHE": "/.cache/huggingface/transformers"
    })
)

# Integration service image for orchestration
# CPU-only, lightweight for service composition (Requirement 2.1)
integration_service_image = (
    base_image
    .pip_install(
        "httpx>=0.25.0",
        "aiohttp>=3.9.0",
        "asyncio-throttle>=1.0.0",
        "tenacity>=8.2.0"
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root"
    })
)

# Storage volumes for model caching
model_volumes = {
    "ace_step_models": modal.Volume.from_name("ace-step-models", create_if_missing=True),
    "huggingface_cache": modal.Volume.from_name("qwen-hf-cache", create_if_missing=True),
    "diffusers_cache": modal.Volume.from_name("diffusers-cache", create_if_missing=True)
}

# Secrets for external services
secrets = {
    "music_gen": modal.Secret.from_name("music-gen-secret"),
    "aws_credentials": modal.Secret.from_name("aws-credentials", create_if_missing=True)
}