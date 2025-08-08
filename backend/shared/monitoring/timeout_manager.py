import time
import logging
from typing import Dict, Optional
from shared.config import settings

logger = logging.getLogger(__name__)


class TimeoutManager:
    """Timeout management for long-running operations"""
    
    def __init__(self, default_timeout: int = None):
        self.default_timeout = default_timeout or settings.default_timeout_seconds
        self.active_operations: Dict[str, Dict[str, float]] = {}
    
    def start_timeout(self, operation_id: str, timeout_seconds: Optional[int] = None, 
                     operation_type: str = "unknown"):
        """Start timeout tracking for an operation"""
        timeout = timeout_seconds or self.default_timeout
        deadline = time.time() + timeout
        
        self.active_operations[operation_id] = {
            "deadline": deadline,
            "timeout_seconds": timeout,
            "start_time": time.time(),
            "operation_type": operation_type
        }
        
        logger.info(f"Started timeout tracking for {operation_id} ({operation_type}): {timeout}s")
    
    def check_timeout(self, operation_id: str) -> bool:
        """Check if an operation has timed out"""
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in timeout tracking")
            return False
        
        current_time = time.time()
        deadline = self.active_operations[operation_id]["deadline"]
        
        if current_time > deadline:
            logger.warning(f"Operation {operation_id} has timed out")
            return True
        
        return False
    
    def get_remaining_time(self, operation_id: str) -> Optional[float]:
        """Get remaining time before timeout"""
        if operation_id not in self.active_operations:
            return None
        
        current_time = time.time()
        deadline = self.active_operations[operation_id]["deadline"]
        remaining = deadline - current_time
        
        return max(0, remaining)
    
    def extend_timeout(self, operation_id: str, additional_seconds: int) -> bool:
        """Extend timeout for an operation"""
        if operation_id not in self.active_operations:
            logger.warning(f"Cannot extend timeout for unknown operation {operation_id}")
            return False
        
        self.active_operations[operation_id]["deadline"] += additional_seconds
        self.active_operations[operation_id]["timeout_seconds"] += additional_seconds
        
        logger.info(f"Extended timeout for {operation_id} by {additional_seconds}s")
        return True
    
    def end_timeout(self, operation_id: str) -> Optional[Dict[str, float]]:
        """End timeout tracking and return operation summary"""
        if operation_id not in self.active_operations:
            return None
        
        operation_data = self.active_operations.pop(operation_id)
        end_time = time.time()
        duration = end_time - operation_data["start_time"]
        
        summary = {
            "duration": duration,
            "timeout_seconds": operation_data["timeout_seconds"],
            "completed_within_timeout": duration < operation_data["timeout_seconds"]
        }
        
        logger.info(f"Completed timeout tracking for {operation_id}: {duration:.2f}s")
        return summary
    
    def get_active_operations(self) -> Dict[str, Dict[str, float]]:
        """Get all active operations with their remaining time"""
        current_time = time.time()
        active_ops = {}
        
        for op_id, data in self.active_operations.items():
            remaining = max(0, data["deadline"] - current_time)
            active_ops[op_id] = {
                "remaining_time": remaining,
                "total_timeout": data["timeout_seconds"],
                "operation_type": data.get("operation_type", "unknown"),
                "elapsed_time": current_time - data["start_time"]
            }
        
        return active_ops
    
    def cleanup_expired_operations(self) -> int:
        """Remove expired operations from tracking"""
        current_time = time.time()
        expired_ops = [
            op_id for op_id, data in self.active_operations.items()
            if current_time > data["deadline"]
        ]
        
        for op_id in expired_ops:
            del self.active_operations[op_id]
            logger.info(f"Cleaned up expired operation {op_id}")
        
        return len(expired_ops)