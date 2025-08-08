import time
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from shared.config import settings

logger = logging.getLogger(__name__)


class CostMonitor:
    """Cost monitoring and tracking for GPU operations"""
    
    def __init__(self, enable_logging: bool = None, log_file: str = None):
        self.enable_logging = enable_logging if enable_logging is not None else settings.enable_cost_monitoring
        self.log_file = log_file or settings.cost_log_file
        self.usage_log: Dict[str, Any] = {}
        
        # Cost rates per hour for different GPU types
        self.cost_rates = {
            "T4": 0.35,
            "L4": 0.60,
            "L40S": 1.20,
            "A100": 2.50,
            "CPU": 0.05
        }
    
    def start_operation(self, operation_id: str, gpu_type: str, service_name: str, 
                       additional_info: Optional[Dict[str, Any]] = None):
        """Start tracking an operation"""
        self.usage_log[operation_id] = {
            "operation_id": operation_id,
            "service_name": service_name,
            "gpu_type": gpu_type,
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "cost_rate_per_hour": self.cost_rates.get(gpu_type, 0.05),
            "additional_info": additional_info or {}
        }
        
        logger.info(f"Started cost tracking for operation {operation_id} on {gpu_type}")
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     additional_info: Optional[Dict[str, Any]] = None):
        """End tracking an operation and calculate cost"""
        if operation_id not in self.usage_log:
            logger.warning(f"Operation {operation_id} not found in usage log")
            return 0.0
        
        data = self.usage_log[operation_id]
        end_time = time.time()
        duration_seconds = end_time - data["start_time"]
        duration_hours = duration_seconds / 3600
        estimated_cost = duration_hours * data["cost_rate_per_hour"]
        
        # Update the log entry
        data.update({
            "end_time": end_time,
            "end_datetime": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "duration_hours": duration_hours,
            "estimated_cost": estimated_cost,
            "success": success,
            "end_additional_info": additional_info or {}
        })
        
        # Log to file if enabled
        if self.enable_logging:
            self._log_to_file(data)
        
        # Check cost alert threshold
        if estimated_cost > settings.cost_alert_threshold:
            logger.warning(f"High cost operation: {operation_id} cost ${estimated_cost:.4f}")
        
        logger.info(f"Completed cost tracking for operation {operation_id}: ${estimated_cost:.4f}")
        return estimated_cost
    
    def get_operation_cost(self, operation_id: str) -> float:
        """Get current estimated cost for an operation"""
        if operation_id not in self.usage_log:
            return 0.0
        
        data = self.usage_log[operation_id]
        if "estimated_cost" in data:
            return data["estimated_cost"]
        
        # Calculate current cost if operation is still running
        current_time = time.time()
        duration_hours = (current_time - data["start_time"]) / 3600
        return duration_hours * data["cost_rate_per_hour"]
    
    def get_total_cost(self, service_name: Optional[str] = None) -> float:
        """Get total cost for all operations or specific service"""
        total = 0.0
        for data in self.usage_log.values():
            if service_name is None or data.get("service_name") == service_name:
                if "estimated_cost" in data:
                    total += data["estimated_cost"]
        return total
    
    def get_operation_summary(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of an operation"""
        return self.usage_log.get(operation_id)
    
    def clear_completed_operations(self):
        """Clear completed operations from memory (but keep in log file)"""
        completed_ops = [
            op_id for op_id, data in self.usage_log.items() 
            if "estimated_cost" in data
        ]
        for op_id in completed_ops:
            del self.usage_log[op_id]
        
        logger.info(f"Cleared {len(completed_ops)} completed operations from memory")
    
    def _log_to_file(self, operation_data: Dict[str, Any]):
        """Log operation data to file"""
        try:
            with open(self.log_file, 'a') as f:
                json.dump(operation_data, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log cost data to file: {e}")
    
    def update_cost_rates(self, new_rates: Dict[str, float]):
        """Update cost rates for GPU types"""
        self.cost_rates.update(new_rates)
        logger.info(f"Updated cost rates: {new_rates}")