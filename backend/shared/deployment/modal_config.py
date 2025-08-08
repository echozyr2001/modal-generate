"""
Modal deployment configurations for AI services.
Renamed from modal_images.py for better clarity.
"""

import modal
from shared.config import settings

# Base image without local sources (for further building)
base_image_core = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HOME": settings.hf_cache_dir})
)

# Base image with local sources (final)
base_image = (
    base_image_core
    .add_local_python_source("shared")
    .add_local_python_source("prompts")
)

# Text generation image (lyrics, prompts)
llm_image = (
    base_image_core
    .pip_install("transformers", "torch", "accelerate")
    .add_local_python_source("shared")
    .add_local_python_source("prompts")
)

# Music generation image with ACE-Step
music_generation_image = (
    base_image_core
    .run_commands([
        "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
        "cd /tmp/ACE-Step && pip install ."
    ])
    .add_local_python_source("shared")
    .add_local_python_source("prompts")
)

# Image generation image
image_generation_image = (
    base_image_core
    .pip_install("diffusers", "torch", "Pillow")
    .add_local_python_source("shared")
    .add_local_python_source("prompts")
)

# Storage volumes
model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

# Secrets
music_gen_secrets = modal.Secret.from_name("music-gen-secret")