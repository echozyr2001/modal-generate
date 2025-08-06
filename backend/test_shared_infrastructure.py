"""
Test script for shared infrastructure and utilities
"""

import os
import sys
import tempfile
import time
import uuid

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock boto3 to avoid dependency issues
class MockS3Client:
    def upload_file(self, *args, **kwargs):
        pass
    
    def generate_presigned_url(self, *args, **kwargs):
        return "https://mock-url.com"
    
    def delete_object(self, *args, **kwargs):
        pass
    
    def head_object(self, *args, **kwargs):
        return {"ContentLength": 1024}

class MockBoto3:
    @staticmethod
    def client(service_name):
        return MockS3Client()

# Mock the boto3 import
sys.modules['boto3'] = MockBoto3()

from shared.utils import CostMonitor, TimeoutManager, create_service_config, get_service_configs
from shared.models import ServiceConfig, GPUType, ResourceLimits, GenerationMetadata


def test_file_manager_local():
    """Test FileManager with local storage"""
    print("Testing FileManager with local storage...")
    
    # Import here after mocking
    from shared.utils import FileManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test local storage
        fm = FileManager(use_s3=False, local_storage_dir=temp_dir)
        
        # Create a test file
        test_content = "Test file content"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(test_content)
            test_file_path = f.name
        
        try:
            # Test save_file
            saved_path = fm.save_file(test_file_path, file_type="temp")
            assert os.path.exists(saved_path), "File should exist after saving"
            
            # Test get_file_url
            url = fm.get_file_url(saved_path)
            assert url == saved_path, "URL should be the file path for local storage"
            
            # Test file_exists
            assert fm.file_exists(saved_path), "File should exist"
            
            # Test get_file_size
            size = fm.get_file_size(saved_path)
            assert size is not None and size > 0, "File size should be positive"
            
            # Test delete_file
            assert fm.delete_file(saved_path), "File deletion should succeed"
            assert not fm.file_exists(saved_path), "File should not exist after deletion"
            
            print("✓ FileManager local storage tests passed")
            
        finally:
            # Cleanup
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)


def test_cost_monitor():
    """Test CostMonitor functionality"""
    print("Testing CostMonitor...")
    
    monitor = CostMonitor()
    operation_id = str(uuid.uuid4())
    
    # Test start_operation
    op_data = monitor.start_operation(operation_id, "L40S", "test_service")
    assert op_data["operation_id"] == operation_id
    assert op_data["gpu_type"] == "L40S"
    assert op_data["service_name"] == "test_service"
    
    # Simulate some work
    time.sleep(0.1)
    
    # Test end_operation
    final_data = monitor.end_operation(operation_id)
    assert "estimated_cost" in final_data
    assert final_data["estimated_cost"] > 0
    
    # Test get_operation_cost
    cost = monitor.get_operation_cost(operation_id)
    assert cost > 0
    
    # Test get_total_cost
    total_cost = monitor.get_total_cost()
    assert total_cost > 0
    
    print("✓ CostMonitor tests passed")


def test_timeout_manager():
    """Test TimeoutManager functionality"""
    print("Testing TimeoutManager...")
    
    manager = TimeoutManager(default_timeout=1)  # 1 second for testing
    operation_id = str(uuid.uuid4())
    
    # Test start_timeout
    timeout_data = manager.start_timeout(operation_id, timeout_seconds=1)
    assert timeout_data["operation_id"] == operation_id
    assert timeout_data["timeout_seconds"] == 1
    
    # Test check_timeout (should not be timed out yet)
    assert not manager.check_timeout(operation_id)
    
    # Test get_remaining_time
    remaining = manager.get_remaining_time(operation_id)
    assert remaining > 0
    
    # Wait for timeout
    time.sleep(1.1)
    
    # Test check_timeout (should be timed out now)
    assert manager.check_timeout(operation_id)
    
    # Test end_timeout
    end_data = manager.end_timeout(operation_id)
    assert end_data is not None
    assert "actual_duration" in end_data
    
    print("✓ TimeoutManager tests passed")


def test_service_config():
    """Test ServiceConfig model"""
    print("Testing ServiceConfig...")
    
    config = ServiceConfig(
        service_name="test_service",
        gpu_type=GPUType.L40S,
        scaledown_window=60,
        max_runtime_seconds=300
    )
    
    assert config.service_name == "test_service"
    assert config.gpu_type == GPUType.L40S
    assert config.cost_per_hour == 1.20  # Should be set based on GPU type
    
    print("✓ ServiceConfig tests passed")


def test_service_base():
    """Test ServiceBase abstract class"""
    print("Testing ServiceBase...")
    
    # Import here after mocking
    from shared.service_base import ServiceBase
    
    # Create a concrete implementation for testing
    class TestService(ServiceBase):
        def load_model(self):
            self._model_loaded = True
        
        def generate(self, request):
            return {"result": "test"}
    
    config = ServiceConfig(
        service_name="test_service",
        gpu_type=GPUType.CPU,
        max_runtime_seconds=60
    )
    
    service = TestService(config)
    
    # Test initialization
    assert service.config.service_name == "test_service"
    assert not service._model_loaded
    
    # Test load_model
    service.load_model()
    assert service._model_loaded
    
    # Test get_service_info
    info = service.get_service_info()
    assert info["service_name"] == "test_service"
    assert info["model_loaded"] == True
    
    # Test health_check
    health = service.health_check()
    assert health["status"] == "healthy"
    
    # Test operation_context
    with service.operation_context("test") as op_id:
        assert isinstance(op_id, str)
        time.sleep(0.1)  # Simulate work
    
    print("✓ ServiceBase tests passed")


def test_config_utilities():
    """Test configuration utility functions"""
    print("Testing configuration utilities...")
    
    # Test create_service_config
    config_dict = {
        "gpu_type": "L40S",
        "scaledown_window": 60,
        "max_runtime_seconds": 300,
        "max_concurrent_requests": 5
    }
    
    config = create_service_config("test_service", config_dict)
    assert config.service_name == "test_service"
    assert config.gpu_type == GPUType.L40S
    
    # Test get_service_configs
    configs = get_service_configs()
    assert "lyrics" in configs
    assert "music" in configs
    assert "image" in configs
    assert "integration" in configs
    
    print("✓ Configuration utilities tests passed")


def run_all_tests():
    """Run all tests"""
    print("Running shared infrastructure tests...\n")
    
    try:
        test_file_manager_local()
        test_cost_monitor()
        test_timeout_manager()
        test_service_config()
        test_service_base()
        test_config_utilities()
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)