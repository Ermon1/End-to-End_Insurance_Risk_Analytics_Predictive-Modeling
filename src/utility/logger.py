# src/utility/logger.py
import logging
from pathlib import Path
import sys
from typing import Optional, Dict, Any
import json

class MLFormatter(logging.Formatter):
    """Simple formatter with optional metadata"""
    def format(self, record):
        if not hasattr(record, 'ml_metadata'):
            record.ml_metadata = {}
        message = super().format(record)
        if record.ml_metadata:
            return f"{message} | METADATA: {json.dumps(record.ml_metadata)}"
        return message

class MLLogger:
    """Minimal logger for ML pipelines"""
    _instances: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Return singleton logger for a module"""
        name = name or __name__
        if name in cls._instances:
            return cls._instances[name]

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Determine log path
        log_file = cls._get_log_file_path(name)

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(MLFormatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

        cls._instances[name] = logger
        return logger

    @staticmethod
    def _get_log_file_path(name: str) -> Path:
        # src/utility/logger.py -> project root
        project_root = Path(__file__).resolve().parent.parent.parent
        log_dir = project_root / "logs" / (name if name != "__main__" else "")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / ("main.log" if name == "__main__" else "execution.log")
        return log_file

    @staticmethod
    def log_metrics(logger: logging.Logger, metrics: Dict[str, float], step: Optional[int] = None):
        metadata = {'metrics': metrics, 'step': step}
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}", extra={'ml_metadata': metadata})

    @staticmethod
    def log_params(logger: logging.Logger, params: Dict[str, Any]):
        logger.info("HYPERPARAMETERS", extra={'ml_metadata': {'params': params}})
