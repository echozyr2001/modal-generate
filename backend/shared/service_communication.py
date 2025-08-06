"""
Service communication utilities with timeout handling, retry logic, and circuit breaker patterns
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ServiceCall:
    """Represents a service call with correlation tracking"""
    correlation_id: str
    service_name: str
    endpoint: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    
    @property
    def duration(self) -> float:
        """Get call duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout_seconds: int = 30   # Request timeout


class CircuitBreaker:
    """Circuit breaker implementation for service calls"""
    
    def __init__(self, service_name: str, config: CircuitBreakerConfig):
        self.service_name = service_name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.call_history: List[ServiceCall] = []
    
    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit breaker state"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker for {self.service_name} moved to HALF_OPEN")
                return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self, call: ServiceCall):
        """Record successful call"""
        call.success = True
        call.end_time = time.time()
        self.call_history.append(call)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker for {self.service_name} moved to CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self, call: ServiceCall, error: str):
        """Record failed call"""
        call.success = False
        call.end_time = time.time()
        call.error = error
        self.call_history.append(call)
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker for {self.service_name} moved to OPEN")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker for {self.service_name} moved back to OPEN")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        recent_calls = [call for call in self.call_history if time.time() - call.start_time < 300]  # Last 5 minutes
        success_rate = sum(1 for call in recent_calls if call.success) / max(len(recent_calls), 1)
        
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "recent_success_rate": success_rate,
            "total_calls": len(self.call_history),
            "recent_calls": len(recent_calls)
        }


class ServiceCommunicationManager:
    """Manages service communication with retry logic, circuit breakers, and correlation tracking"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.active_calls: Dict[str, ServiceCall] = {}
        self.correlation_tracking: Dict[str, List[ServiceCall]] = {}
        self.default_timeout = 30
    
    def get_or_create_circuit_breaker(self, service_name: str, 
                                    config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(service_name, config)
        return self.circuit_breakers[service_name]
    
    def generate_correlation_id(self) -> str:
        """Generate unique correlation ID"""
        return str(uuid.uuid4())
    
    def start_call_tracking(self, service_name: str, endpoint: str, 
                          correlation_id: Optional[str] = None) -> ServiceCall:
        """Start tracking a service call"""
        correlation_id = correlation_id or self.generate_correlation_id()
        
        call = ServiceCall(
            correlation_id=correlation_id,
            service_name=service_name,
            endpoint=endpoint,
            start_time=time.time()
        )
        
        self.active_calls[f"{service_name}_{correlation_id}"] = call
        
        if correlation_id not in self.correlation_tracking:
            self.correlation_tracking[correlation_id] = []
        self.correlation_tracking[correlation_id].append(call)
        
        return call
    
    def end_call_tracking(self, call: ServiceCall, success: bool = True, 
                         error: Optional[str] = None, 
                         response_data: Optional[Dict[str, Any]] = None):
        """End tracking a service call"""
        call.end_time = time.time()
        call.success = success
        call.error = error
        call.response_data = response_data
        
        # Remove from active calls
        call_key = f"{call.service_name}_{call.correlation_id}"
        self.active_calls.pop(call_key, None)
        
        # Update circuit breaker
        circuit_breaker = self.get_or_create_circuit_breaker(call.service_name)
        if success:
            circuit_breaker.record_success(call)
        else:
            circuit_breaker.record_failure(call, error or "Unknown error")
    
    @asynccontextmanager
    async def call_context(self, service_name: str, endpoint: str, 
                          correlation_id: Optional[str] = None):
        """Context manager for tracking service calls"""
        call = self.start_call_tracking(service_name, endpoint, correlation_id)
        
        try:
            yield call
            self.end_call_tracking(call, success=True)
        except Exception as e:
            self.end_call_tracking(call, success=False, error=str(e))
            raise
    
    async def call_service_with_retry(self, service_url: str, endpoint: str,
                                    data: Dict[str, Any], service_name: str,
                                    timeout: int = None, max_retries: int = 3,
                                    correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Call service with retry logic and circuit breaker protection"""
        timeout = timeout or self.default_timeout
        circuit_breaker = self.get_or_create_circuit_breaker(service_name)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker for {service_name} is OPEN")
        
        @retry(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
        )
        async def _make_request_with_retry():
            async with self.call_context(service_name, endpoint, correlation_id) as call:
                url = f"{service_url}{endpoint}"
                
                # Add correlation ID to request data
                request_data = {**data, "correlation_id": call.correlation_id}
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    logger.info(f"Calling {service_name}{endpoint} [correlation_id: {call.correlation_id}]")
                    
                    response = await client.post(url, json=request_data)
                    response.raise_for_status()
                    
                    result = response.json()
                    call.response_data = result
                    
                    logger.info(f"Successfully called {service_name}{endpoint} "
                              f"[correlation_id: {call.correlation_id}] in {call.duration:.2f}s")
                    
                    return result
        
        return await _make_request_with_retry()
    
    async def call_service_batch(self, calls: List[Dict[str, Any]], 
                               correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute multiple service calls in parallel"""
        correlation_id = correlation_id or self.generate_correlation_id()
        
        tasks = []
        for call_config in calls:
            task = self.call_service_with_retry(
                service_url=call_config["service_url"],
                endpoint=call_config["endpoint"],
                data=call_config["data"],
                service_name=call_config["service_name"],
                timeout=call_config.get("timeout"),
                max_retries=call_config.get("max_retries", 3),
                correlation_id=correlation_id
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch call {i} failed: {result}")
                processed_results.append({"error": str(result), "success": False})
            else:
                processed_results.append({"data": result, "success": True})
        
        return processed_results
    
    def get_correlation_history(self, correlation_id: str) -> List[ServiceCall]:
        """Get all calls for a correlation ID"""
        return self.correlation_tracking.get(correlation_id, [])
    
    def get_service_stats(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for services"""
        if service_name:
            circuit_breaker = self.circuit_breakers.get(service_name)
            return circuit_breaker.get_stats() if circuit_breaker else {}
        
        return {
            service: breaker.get_stats() 
            for service, breaker in self.circuit_breakers.items()
        }
    
    def cleanup_old_tracking_data(self, max_age_seconds: int = 3600):
        """Clean up old tracking data to prevent memory leaks"""
        current_time = time.time()
        
        # Clean up correlation tracking
        expired_correlations = []
        for correlation_id, calls in self.correlation_tracking.items():
            if all(current_time - call.start_time > max_age_seconds for call in calls):
                expired_correlations.append(correlation_id)
        
        for correlation_id in expired_correlations:
            del self.correlation_tracking[correlation_id]
        
        # Clean up circuit breaker history
        for breaker in self.circuit_breakers.values():
            breaker.call_history = [
                call for call in breaker.call_history 
                if current_time - call.start_time <= max_age_seconds
            ]
        
        logger.info(f"Cleaned up {len(expired_correlations)} expired correlation tracking entries")


# Global instance for shared use
service_comm_manager = ServiceCommunicationManager()


# Convenience functions for common use cases
async def call_service(service_url: str, endpoint: str, data: Dict[str, Any],
                      service_name: str, timeout: int = 30, max_retries: int = 3,
                      correlation_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for single service calls"""
    return await service_comm_manager.call_service_with_retry(
        service_url, endpoint, data, service_name, timeout, max_retries, correlation_id
    )


async def call_services_parallel(calls: List[Dict[str, Any]], 
                               correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function for parallel service calls"""
    return await service_comm_manager.call_service_batch(calls, correlation_id)


def get_circuit_breaker_stats(service_name: Optional[str] = None) -> Dict[str, Any]:
    """Get circuit breaker statistics"""
    return service_comm_manager.get_service_stats(service_name)


def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return service_comm_manager.generate_correlation_id()