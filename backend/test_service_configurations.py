"""
Test script to validate the modular service configurations.

This script tests the service configuration classes and Modal image setup
to ensure they meet the requirements for GPU optimization and cost control.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

from shared.service_config import (
    ServiceConfigFactory,
    ServiceType,
    GPUType,
    TextGenerationConfig,
    MusicGenerationConfig,
    ImageGenerationConfig,
    IntegrationConfig
)


def test_service_configurations():
    """Test that service configurations meet requirements."""
    print("Testing service configurations...")
    
    # Test default configurations
    default_configs = ServiceConfigFactory.get_default_configs()
    
    # Verify text generation uses CPU/low-cost GPU (Requirement 2.1)
    text_config = default_configs[ServiceType.TEXT_GENERATION]
    assert text_config.gpu_type in [GPUType.CPU_ONLY, GPUType.T4], f"Text service should use CPU or T4, got {text_config.gpu_type}"
    assert text_config.scaledown_window_seconds <= 60, f"Text service should scale down quickly, got {text_config.scaledown_window_seconds}s"
    print("✓ Text generation config meets requirements")
    
    # Verify music generation uses high-memory GPU (Requirement 2.3)
    music_config = default_configs[ServiceType.MUSIC_GENERATION]
    assert music_config.gpu_type in [GPUType.L40S, GPUType.A100], f"Music service should use high-memory GPU, got {music_config.gpu_type}"
    assert music_config.scaledown_window_seconds >= 30, f"Music service should have longer scaledown, got {music_config.scaledown_window_seconds}s"
    print("✓ Music generation config meets requirements")
    
    # Verify image generation uses appropriate GPU (Requirement 2.2)
    image_config = default_configs[ServiceType.IMAGE_GENERATION]
    assert image_config.gpu_type in [GPUType.T4, GPUType.L4, GPUType.L40S], f"Image service should use appropriate GPU, got {image_config.gpu_type}"
    print("✓ Image generation config meets requirements")
    
    # Verify integration service uses CPU only
    integration_config = default_configs[ServiceType.INTEGRATION]
    assert integration_config.gpu_type == GPUType.CPU_ONLY, f"Integration service should be CPU-only, got {integration_config.gpu_type}"
    assert integration_config.scaledown_window_seconds <= 30, f"Integration service should scale down quickly, got {integration_config.scaledown_window_seconds}s"
    print("✓ Integration service config meets requirements")


def test_cost_estimation():
    """Test cost estimation functionality (Requirement 5.4)."""
    print("\nTesting cost estimation...")
    
    # Test cost estimation for different services
    configs = ServiceConfigFactory.get_default_configs()
    
    for service_type, config in configs.items():
        # Test cost calculation
        runtime_minutes = 10
        base_cost = config.cost_estimation.startup_cost_usd
        runtime_cost = config.cost_estimation.cost_per_minute_usd * runtime_minutes
        total_cost = base_cost + runtime_cost
        
        assert total_cost > 0, f"Cost should be positive for {service_type}"
        assert base_cost >= 0, f"Startup cost should be non-negative for {service_type}"
        assert runtime_cost >= 0, f"Runtime cost should be non-negative for {service_type}"
        
        print(f"✓ {service_type.value}: ${total_cost:.4f} for 10 minutes")


def test_resource_limits():
    """Test resource limits for cost control (Requirements 5.1, 5.2)."""
    print("\nTesting resource limits...")
    
    configs = ServiceConfigFactory.get_default_configs()
    
    for service_type, config in configs.items():
        limits = config.resource_limits
        
        # Verify timeout limits
        assert limits.timeout_seconds > 0, f"Timeout should be positive for {service_type}"
        assert limits.max_runtime_seconds > 0, f"Max runtime should be positive for {service_type}"
        assert limits.max_concurrent_requests > 0, f"Max concurrent requests should be positive for {service_type}"
        
        # Verify service-specific limits
        if service_type == ServiceType.TEXT_GENERATION:
            assert limits.timeout_seconds <= 180, f"Text service timeout should be short, got {limits.timeout_seconds}s"
        elif service_type == ServiceType.MUSIC_GENERATION:
            assert limits.timeout_seconds >= 600, f"Music service needs longer timeout, got {limits.timeout_seconds}s"
            assert limits.max_audio_duration <= 600, f"Audio duration should be limited, got {limits.max_audio_duration}s"
        
        print(f"✓ {service_type.value}: timeout={limits.timeout_seconds}s, max_runtime={limits.max_runtime_seconds}s")


def test_gpu_recommendations():
    """Test GPU type recommendations for different services."""
    print("\nTesting GPU recommendations...")
    
    configs = ServiceConfigFactory.get_default_configs()
    
    # Verify each service has appropriate GPU type
    text_config = configs[ServiceType.TEXT_GENERATION]
    music_config = configs[ServiceType.MUSIC_GENERATION] 
    image_config = configs[ServiceType.IMAGE_GENERATION]
    integration_config = configs[ServiceType.INTEGRATION]
    
    # Test GPU assignments match requirements
    assert text_config.gpu_type in [GPUType.CPU_ONLY, GPUType.T4], "Text should use CPU or T4"
    assert music_config.gpu_type in [GPUType.L40S, GPUType.A100], "Music should use high-memory GPU"
    assert image_config.gpu_type in [GPUType.T4, GPUType.L4, GPUType.L40S], "Image should use appropriate GPU"
    assert integration_config.gpu_type == GPUType.CPU_ONLY, "Integration should be CPU-only"
    
    print(f"✓ Text: {text_config.gpu_type.value}")
    print(f"✓ Music: {music_config.gpu_type.value}")
    print(f"✓ Image: {image_config.gpu_type.value}")
    print(f"✓ Integration: {integration_config.gpu_type.value}")


def test_validation_warnings():
    """Test configuration validation (Requirement 2.4)."""
    print("\nTesting configuration validation...")
    
    # Test valid configurations
    valid_config = TextGenerationConfig()
    warnings = ServiceConfigFactory.validate_config(valid_config)
    print(f"✓ Valid text config has {len(warnings)} warnings")
    
    # Test potentially problematic configuration
    problematic_config = MusicGenerationConfig(gpu_type=GPUType.T4)
    warnings = ServiceConfigFactory.validate_config(problematic_config)
    assert len(warnings) > 0, "Should warn about T4 for music generation"
    print(f"✓ Problematic config correctly flagged with {len(warnings)} warnings")


if __name__ == "__main__":
    print("Running modular service configuration tests...\n")
    
    try:
        test_service_configurations()
        test_cost_estimation()
        test_resource_limits()
        test_gpu_recommendations()
        test_validation_warnings()
        
        print("\n🎉 All tests passed! Service configurations meet requirements.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)