"""
Structured Logging Module.

Provides comprehensive logging with structured output for
fraud detection system monitoring and debugging.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Callable

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class FraudLogger:
    """
    Structured logger for fraud detection system.
    
    Provides:
    - JSON formatted logs for production
    - Human-readable format for development
    - Performance metrics tracking
    - Memory usage monitoring
    """
    
    def __init__(
        self,
        name: str = "fraud_detection",
        level: str = "INFO",
        format: str = "json",
        log_file: Optional[str] = None
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name.
            level: Logging level (DEBUG, INFO, WARNING, ERROR).
            format: Output format ('json' or 'text').
            log_file: Path to log file (optional).
        """
        self.name = name
        self.level = level
        self.format = format
        self.log_file = log_file
        
        self._logger = self._setup_logger()
        self._metrics: dict[str, list] = {}
        self._start_times: dict[str, float] = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the underlying logger."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level.upper()))
        
        # Clear existing handlers
        logger.handlers = []
        
        # Create formatter
        if self.format == "json":
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            path = Path(self.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log("error", message, **kwargs)
    
    def _log(self, level: str, message: str, **kwargs) -> None:
        """Internal log method."""
        log_func = getattr(self._logger, level)
        
        if self.format == "json":
            # Include extra fields in JSON format
            extra = {"extra_fields": kwargs} if kwargs else {}
            log_func(message, extra=extra)
        else:
            # Include extra fields in message for text format
            if kwargs:
                extra_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
                message = f"{message} | {extra_str}"
            log_func(message)
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self._start_times[operation] = time.time()
    
    def stop_timer(self, operation: str) -> float:
        """Stop timing and return duration."""
        if operation not in self._start_times:
            return 0.0
        
        duration = time.time() - self._start_times[operation]
        del self._start_times[operation]
        
        # Record metric
        if operation not in self._metrics:
            self._metrics[operation] = []
        self._metrics[operation].append(duration)
        
        return duration
    
    @contextmanager
    def timer(self, operation: str, log_result: bool = True):
        """Context manager for timing operations."""
        self.start_timer(operation)
        try:
            yield
        finally:
            duration = self.stop_timer(operation)
            if log_result:
                self.info(f"{operation} completed", duration_seconds=round(duration, 3))
    
    def log_metrics(self, **metrics) -> None:
        """Log performance metrics."""
        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(value)
        
        self.info("Metrics recorded", **metrics)
    
    def get_metrics_summary(self) -> dict:
        """Get summary statistics for all recorded metrics."""
        summary = {}
        
        for name, values in self._metrics.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1]
                }
        
        return summary
    
    def log_detection_result(
        self,
        num_records: int,
        num_flagged: int,
        num_clusters: int,
        duration: float,
        **extra
    ) -> None:
        """Log detection pipeline results."""
        self.info(
            "Detection completed",
            total_records=num_records,
            flagged_records=num_flagged,
            flagged_ratio=round(num_flagged / max(num_records, 1), 4),
            clusters_found=num_clusters,
            duration_seconds=round(duration, 3),
            **extra
        )
    
    def log_evaluation_result(
        self,
        precision: float,
        recall: float,
        f1_score: float,
        **extra
    ) -> None:
        """Log evaluation metrics."""
        self.info(
            "Evaluation completed",
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1_score, 4),
            **extra
        )


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def get_logger(
    name: str = "fraud_detection",
    level: str = "INFO",
    format: str = "text",
    log_file: Optional[str] = None
) -> FraudLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name.
        level: Logging level.
        format: Output format ('json' or 'text').
        log_file: Path to log file.
        
    Returns:
        Configured FraudLogger instance.
    """
    return FraudLogger(name=name, level=level, format=format, log_file=log_file)


def log_function_call(logger: Optional[FraudLogger] = None):
    """Decorator to log function calls with timing."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger()
            func_name = func.__name__
            
            log.debug(f"Calling {func_name}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
            
            with log.timer(func_name, log_result=True):
                result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator


def get_memory_usage() -> dict:
    """Get current memory usage statistics."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
        }
    except ImportError:
        return {"error": "psutil not installed"}


if __name__ == "__main__":
    # Test logging
    logger = get_logger(format="text", level="DEBUG")
    
    logger.info("Starting fraud detection system")
    logger.debug("Debug message with context", detector="DBSCAN", threshold=0.35)
    
    with logger.timer("test_operation"):
        time.sleep(0.1)
    
    logger.log_detection_result(
        num_records=1000,
        num_flagged=50,
        num_clusters=10,
        duration=2.5
    )
    
    print("\nMetrics summary:")
    print(logger.get_metrics_summary())
