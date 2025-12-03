"""
Advanced logging and monitoring system.
"""

import logging
import logging.config
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import threading
from queue import Queue
import numpy as np
from contextlib import contextmanager

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class LoggingConfig:
    """Configuration for advanced logging."""
    
    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    mlflow_level: str = "INFO"
    
    # File logging
    log_dir: Path = Path("logs")
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # MLflow
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "reservoir_ai"
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "reservoir-ai"
    wandb_entity: Optional[str] = None
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    metrics_update_interval: float = 60.0  # seconds
    
    # Alerting
    enable_alerts: bool = False
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.01,
        "memory_usage": 0.9,
        "inference_time": 1.0,  # seconds
    })
    
    # Custom metrics
    custom_metrics: List[str] = field(default_factory=lambda: [
        "model_performance",
        "training_time",
        "memory_usage",
        "prediction_latency",
    ])


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.history: Dict[str, List] = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def update(self, name: str, value: Union[float, int, dict], timestamp: Optional[float] = None):
        """Update a metric."""
        with self.lock:
            if timestamp is None:
                timestamp = time.time()
            
            # Store current value
            self.metrics[name] = {
                'value': value,
                'timestamp': timestamp,
                'type': type(value).__name__
            }
            
            # Store in history
            if name not in self.history:
                self.history[name] = []
            
            self.history[name].append({
                'timestamp': timestamp,
                'value': value
            })
    
    def increment(self, name: str, amount: float = 1.0):
        """Increment a counter metric."""
        with self.lock:
            current = self.metrics.get(name, {}).get('value', 0)
            self.update(name, current + amount)
    
    def timing(self, name: str, duration: float):
        """Record a timing metric."""
        with self.lock:
            if f"{name}_count" not in self.metrics:
                self.update(f"{name}_count", 0)
                self.update(f"{name}_total", 0.0)
                self.update(f"{name}_avg", 0.0)
                self.update(f"{name}_min", float('inf'))
                self.update(f"{name}_max", 0.0)
            
            # Update statistics
            count = self.metrics[f"{name}_count"]['value'] + 1
            total = self.metrics[f"{name}_total"]['value'] + duration
            
            self.update(f"{name}_count", count)
            self.update(f"{name}_total", total)
            self.update(f"{name}_avg", total / count)
            
            # Update min/max
            current_min = self.metrics[f"{name}_min"]['value']
            current_max = self.metrics[f"{name}_max"]['value']
            
            if duration < current_min:
                self.update(f"{name}_min", duration)
            if duration > current_max:
                self.update(f"{name}_max", duration)
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric value."""
        with self.lock:
            metric = self.metrics.get(name)
            return metric['value'] if metric else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self.lock:
            summary = {
                'timestamp': time.time(),
                'uptime': time.time() - self.start_time,
                'metrics': {},
                'statistics': {}
            }
            
            for name, metric in self.metrics.items():
                summary['metrics'][name] = metric['value']
                
                # Calculate statistics for numeric metrics
                if isinstance(metric['value'], (int, float)):
                    history_values = [h['value'] for h in self.history.get(name, []) 
                                    if isinstance(h['value'], (int, float))]
                    
                    if history_values:
                        summary['statistics'][name] = {
                            'mean': np.mean(history_values),
                            'std': np.std(history_values),
                            'min': np.min(history_values),
                            'max': np.max(history_values),
                            'count': len(history_values)
                        }
            
            return summary
    
    def clear(self):
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()
            self.history.clear()


class PerformanceMonitor:
    """Monitor system and model performance."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize performance metrics."""
        import psutil
        process = psutil.Process()
        
        self.collector.update('system_cpu_percent', 0.0)
        self.collector.update('system_memory_percent', 0.0)
        self.collector.update('process_cpu_percent', 0.0)
        self.collector.update('process_memory_mb', 0.0)
        self.collector.update('process_threads', 0)
        
        # Network
        net_io = psutil.net_io_counters()
        self.collector.update('network_bytes_sent', net_io.bytes_sent)
        self.collector.update('network_bytes_recv', net_io.bytes_recv)
        
        # Disk
        disk_io = psutil.disk_io_counters()
        self.collector.update('disk_read_bytes', disk_io.read_bytes)
        self.collector.update('disk_write_bytes', disk_io.write_bytes)
    
    def start(self, interval: float = 1.0):
        """Start monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.thread.start()
    
    def stop(self):
        """Stop monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        import psutil
        
        while self.running:
            try:
                # System metrics
                self.collector.update('system_cpu_percent', psutil.cpu_percent(interval=None))
                self.collector.update('system_memory_percent', psutil.virtual_memory().percent)
                
                # Process metrics
                process = psutil.Process()
                with process.oneshot():
                    self.collector.update('process_cpu_percent', process.cpu_percent())
                    self.collector.update('process_memory_mb', process.memory_info().rss / 1024 / 1024)
                    self.collector.update('process_threads', process.num_threads())
                
                # Network metrics
                net_io = psutil.net_io_counters()
                self.collector.update('network_bytes_sent_rate', 
                                    (net_io.bytes_sent - self.collector.get_metric('network_bytes_sent')) / interval)
                self.collector.update('network_bytes_sent', net_io.bytes_sent)
                self.collector.update('network_bytes_recv_rate',
                                    (net_io.bytes_recv - self.collector.get_metric('network_bytes_recv')) / interval)
                self.collector.update('network_bytes_recv', net_io.bytes_recv)
                
                # Disk metrics
                disk_io = psutil.disk_io_counters()
                self.collector.update('disk_read_rate',
                                    (disk_io.read_bytes - self.collector.get_metric('disk_read_bytes')) / interval)
                self.collector.update('disk_read_bytes', disk_io.read_bytes)
                self.collector.update('disk_write_rate',
                                    (disk_io.write_bytes - self.collector.get_metric('disk_write_bytes')) / interval)
                self.collector.update('disk_write_bytes', disk_io.write_bytes)
                
                # Python metrics
                import gc
                self.collector.update('python_gc_objects', len(gc.get_objects()))
                self.collector.update('python_gc_collected', gc.get_count()[0])
                
            except Exception as e:
                self.collector.update('monitoring_errors', 
                                    self.collector.get_metric('monitoring_errors') or 0 + 1)
            
            time.sleep(interval)


class AlertManager:
    """Manage alerts based on metric thresholds."""
    
    def __init__(self, collector: MetricsCollector, config: LoggingConfig):
        self.collector = collector
        self.config = config
        self.alerts: List[Dict[str, Any]] = []
        self.alert_handlers: List[callable] = []
        
        # Register default alert handler
        self.register_alert_handler(self._log_alert)
    
    def register_alert_handler(self, handler: callable):
        """Register an alert handler function."""
        self.alert_handlers.append(handler)
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Default alert handler - log to console."""
        logging.getLogger("alerts").warning(
            f"ALERT: {alert['name']} - {alert['message']} "
            f"(value: {alert['value']:.4f}, threshold: {alert['threshold']:.4f})"
        )
    
    def check_thresholds(self):
        """Check all configured thresholds."""
        thresholds = self.config.alert_thresholds
        
        for metric_name, threshold in thresholds.items():
            current_value = self.collector.get_metric(metric_name)
            
            if current_value is not None:
                if metric_name == "error_rate" and current_value > threshold:
                    self._trigger_alert(
                        name="high_error_rate",
                        message=f"Error rate exceeds threshold",
                        value=current_value,
                        threshold=threshold,
                        severity="high"
                    )
                
                elif metric_name == "memory_usage" and current_value > threshold:
                    self._trigger_alert(
                        name="high_memory_usage",
                        message=f"Memory usage exceeds threshold",
                        value=current_value,
                        threshold=threshold,
                        severity="critical"
                    )
                
                elif metric_name == "inference_time" and current_value > threshold:
                    self._trigger_alert(
                        name="slow_inference",
                        message=f"Inference time exceeds threshold",
                        value=current_value,
                        threshold=threshold,
                        severity="medium"
                    )
    
    def _trigger_alert(self, name: str, message: str, value: float, 
                      threshold: float, severity: str):
        """Trigger an alert."""
        alert = {
            'name': name,
            'message': message,
            'value': value,
            'threshold': threshold,
            'severity': severity,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        
        # Call all alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def get_recent_alerts(self, hours: float = 24.0) -> List[Dict[str, Any]]:
        """Get recent alerts within specified hours."""
        cutoff = time.time() - (hours * 3600)
        return [a for a in self.alerts if a['timestamp'] > cutoff]


class MLflowLogger:
    """Logger for MLflow integration."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.active_run = None
        
        if config.use_mlflow and MLFLOW_AVAILABLE:
            self._initialize_mlflow()
    
    def _initialize_mlflow(self):
        """Initialize MLflow connection."""
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(self.config.mlflow_experiment_name)
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Context manager for MLflow run."""
        if not self.config.use_mlflow or not MLFLOW_AVAILABLE:
            yield
            return
        
        try:
            self.active_run = mlflow.start_run(run_name=run_name, tags=tags)
            yield self.active_run
        finally:
            if self.active_run:
                mlflow.end_run()
                self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self.active_run:
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self.active_run:
            mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: Union[str, Path]):
        """Log artifact to MLflow."""
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self.active_run:
            mlflow.log_artifact(str(local_path))
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log model to MLflow."""
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self.active_run:
            mlflow.sklearn.log_model(model, artifact_path)


class WandbLogger:
    """Logger for Weights & Biases integration."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.run = None
        
        if config.use_wandb and WANDB_AVAILABLE:
            self._initialize_wandb()
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases."""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config=self.config.__dict__
        )
        self.run = wandb.run
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log data to Weights & Biases."""
        if self.config.use_wandb and WANDB_AVAILABLE and self.run:
            wandb.log(data, step=step)
    
    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """Watch model for gradients and parameters."""
        if self.config.use_wandb and WANDB_AVAILABLE and self.run:
            wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self):
        """Finish Weights & Biases run."""
        if self.config.use_wandb and WANDB_AVAILABLE and self.run:
            wandb.finish()
            self.run = None


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.queue = Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        self.handlers: List[logging.Handler] = []
    
    def add_handler(self, handler: logging.Handler):
        """Add a handler to process logs."""
        self.handlers.append(handler)
    
    def emit(self, record):
        """Emit a record asynchronously."""
        self.queue.put(record)
    
    def _process_queue(self):
        """Process log records from queue."""
        while True:
            try:
                record = self.queue.get(timeout=1.0)
                for handler in self.handlers:
                    try:
                        handler.emit(record)
                    except Exception:
                        pass
                self.queue.task_done()
            except Exception:
                continue


class ReservoirLogger:
    """Main logging class for Reservoir AI."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Optional[LoggingConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        if self._initialized:
            return
        
        self.config = config or LoggingConfig()
        self.collector = MetricsCollector()
        self.monitor: Optional[PerformanceMonitor] = None
        self.alert_manager: Optional[AlertManager] = None
        self.mlflow_logger: Optional[MLflowLogger] = None
        self.wandb_logger: Optional[WandbLogger] = None
        self.async_handler: Optional[AsyncLogHandler] = None
        
        self._setup_logging()
        self._setup_monitoring()
        self._setup_integrations()
        
        self._initialized = True
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create log directory
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': self.config.log_format,
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'json': {
                    'format': '{"time": "%(asctime)s", "name": "%(name)s", '
                             '"level": "%(levelname)s", "message": "%(message)s"}',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.config.console_level,
                    'formatter': 'detailed',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.config.file_level,
                    'formatter': 'detailed',
                    'filename': str(self.config.log_dir / 'reservoir_ai.log'),
                    'maxBytes': self.config.max_file_size,
                    'backupCount': self.config.backup_count
                },
                'json_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': self.config.file_level,
                    'formatter': 'json',
                    'filename': str(self.config.log_dir / 'reservoir_ai.json'),
                    'maxBytes': self.config.max_file_size,
                    'backupCount': self.config.backup_count
                }
            },
            'loggers': {
                '': {  # Root logger
                    'handlers': ['console', 'file', 'json_file'],
                    'level': 'DEBUG',
                },
                'alerts': {
                    'handlers': ['console', 'file'],
                    'level': 'WARNING',
                    'propagate': False
                },
                'performance': {
                    'handlers': ['file', 'json_file'],
                    'level': 'INFO',
                    'propagate': False
                }
            }
        }
        
        logging.config.dictConfig(logging_config)
        
        # Setup async logging for high-performance scenarios
        self.async_handler = AsyncLogHandler()
        self.async_handler.add_handler(logging.StreamHandler())
        
        # Add async handler to root logger
        logging.getLogger().addHandler(self.async_handler)
    
    def _setup_monitoring(self):
        """Setup performance monitoring."""
        if self.config.enable_performance_monitoring:
            self.monitor = PerformanceMonitor(self.collector)
            self.monitor.start(interval=1.0)
            
            if self.config.enable_alerts:
                self.alert_manager = AlertManager(self.collector, self.config)
    
    def _setup_integrations(self):
        """Setup external integrations."""
        if self.config.use_mlflow:
            self.mlflow_logger = MLflowLogger(self.config)
        
        if self.config.use_wandb:
            self.wandb_logger = WandbLogger(self.config)
    
    def log_training_start(self, model_config: Dict[str, Any], data_info: Dict[str, Any]):
        """Log training start event."""
        logger = logging.getLogger('training')
        logger.info(f"Training started with config: {model_config}")
        logger.info(f"Data info: {data_info}")
        
        # Update metrics
        self.collector.update('training_start_time', time.time())
        self.collector.update('training_config', model_config)
        self.collector.update('training_data_info', data_info)
        
        # Log to MLflow
        if self.mlflow_logger:
            with self.mlflow_logger.start_run(
                run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={"type": "training", "model": model_config.get('model_type', 'unknown')}
            ):
                self.mlflow_logger.log_params(model_config)
                self.mlflow_logger.log_params(data_info)
        
        # Log to Weights & Biases
        if self.wandb_logger:
            self.wandb_logger.log({
                "event": "training_started",
                "timestamp": time.time(),
                **model_config,
                **data_info
            })
    
    def log_training_progress(self, epoch: int, metrics: Dict[str, float], 
                            learning_rate: Optional[float] = None):
        """Log training progress."""
        logger = logging.getLogger('training.progress')
        logger.debug(f"Epoch {epoch}: {metrics}")
        
        # Update metrics
        self.collector.update(f'training_epoch_{epoch}_metrics', metrics)
        if learning_rate:
            self.collector.update(f'training_epoch_{epoch}_lr', learning_rate)
        
        # Log to integrations
        if self.mlflow_logger and self.mlflow_logger.active_run:
            self.mlflow_logger.log_metrics(metrics, step=epoch)
        
        if self.wandb_logger:
            wandb_metrics = {f"train/{k}": v for k, v in metrics.items()}
            if learning_rate:
                wandb_metrics["train/learning_rate"] = learning_rate
            self.wandb_logger.log(wandb_metrics, step=epoch)
    
    def log_training_complete(self, training_stats: Dict[str, Any], 
                            model_path: Optional[Path] = None):
        """Log training completion."""
        logger = logging.getLogger('training')
        logger.info(f"Training completed. Stats: {training_stats}")
        
        # Update metrics
        training_time = time.time() - self.collector.get_metric('training_start_time')
        self.collector.update('training_complete_time', time.time())
        self.collector.update('training_duration', training_time)
        self.collector.update('training_stats', training_stats)
        
        logger.info(f"Training took {training_time:.2f} seconds")
        
        # Log to MLflow
        if self.mlflow_logger and self.mlflow_logger.active_run:
            self.mlflow_logger.log_metrics({
                'training_duration': training_time,
                'final_loss': training_stats.get('final_loss', 0.0),
                'best_metric': training_stats.get('best_metric', 0.0)
            })
            
            if model_path:
                self.mlflow_logger.log_artifact(str(model_path))
        
        # Log to Weights & Biases
        if self.wandb_logger:
            self.wandb_logger.log({
                "event": "training_completed",
                "training_duration": training_time,
                **training_stats
            })
    
    def log_prediction(self, model_id: str, batch_size: int, 
                      inference_time: float, predictions_shape: tuple):
        """Log prediction event."""
        logger = logging.getLogger('prediction')
        logger.debug(f"Model {model_id}: {batch_size} samples in {inference_time:.4f}s")
        
        # Update metrics
        self.collector.timing('inference_time', inference_time)
        self.collector.increment('total_predictions', batch_size)
        self.collector.update('current_batch_size', batch_size)
        self.collector.update('current_inference_time', inference_time)
        
        # Calculate throughput
        throughput = batch_size / inference_time if inference_time > 0 else 0
        self.collector.update('prediction_throughput', throughput)
        
        # Check alerts
        if self.alert_manager:
            self.alert_manager.check_thresholds()
    
    def log_error(self, error_type: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Log error event."""
        logger = logging.getLogger('errors')
        logger.error(f"{error_type}: {message}", extra={'context': context})
        
        # Update metrics
        self.collector.increment('error_count')
        self.collector.increment(f'error_{error_type}_count')
        
        # Store error details
        error_details = {
            'type': error_type,
            'message': message,
            'context': context,
            'timestamp': time.time(),
            'traceback': None
        }
        
        current_errors = self.collector.get_metric('recent_errors') or []
        current_errors.append(error_details)
        
        # Keep only last 100 errors
        if len(current_errors) > 100:
            current_errors = current_errors[-100:]
        
        self.collector.update('recent_errors', current_errors)
        
        # Calculate error rate
        total_predictions = self.collector.get_metric('total_predictions') or 1
        error_rate = self.collector.get_metric('error_count') or 0 / total_predictions
        self.collector.update('error_rate', error_rate)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.collector.get_summary()
    
    def generate_report(self, report_type: str = "daily") -> Path:
        """Generate performance report."""
        report_dir = self.config.log_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'report_type': report_type,
            'performance_metrics': self.collector.get_summary(),
            'recent_alerts': self.alert_manager.get_recent_alerts() if self.alert_manager else [],
            'system_info': self._get_system_info()
        }
        
        report_file = report_dir / f"report_{report_type}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logging.getLogger('reports').info(f"Generated {report_type} report: {report_file}")
        return report_file
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        import psutil
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'disk_usage': {k: v._asdict() for k, v in psutil.disk_usage('/')._asdict().items()}
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.monitor:
            self.monitor.stop()
        
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        # Generate final report
        self.generate_report("session")


# Global logger instance
_logger_instance: Optional[ReservoirLogger] = None


def get_logger(config: Optional[LoggingConfig] = None) -> ReservoirLogger:
    """Get or create global logger instance."""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = ReservoirLogger(config)
    
    return _logger_instance


@contextmanager
def training_session(model_config: Dict[str, Any], data_info: Dict[str, Any]):
    """Context manager for training session logging."""
    logger = get_logger()
    
    try:
        logger.log_training_start(model_config, data_info)
        yield logger
    except Exception as e:
        logger.log_error('training_error', str(e), {'config': model_config})
        raise
    finally:
        pass  # Training completion logged separately


@contextmanager
def prediction_session(model_id: str):
    """Context manager for prediction session logging."""
    logger = get_logger()
    start_time = time.time()
    
    try:
        yield
    except Exception as e:
        logger.log_error('prediction_error', str(e), {'model_id': model_id})
        raise
    finally:
        inference_time = time.time() - start_time
        logger.log_prediction(model_id, 1, inference_time, (1, 1))
