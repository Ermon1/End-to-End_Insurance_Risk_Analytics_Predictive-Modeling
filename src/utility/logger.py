# src/utility/logger.py
import logging
from pathlib import Path
import sys
from typing import Optional

def get_logger(name: Optional[str] = None, log_name: Optional[str] = None) -> logging.Logger:
    name = name or __name__
    log_name = log_name or name

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs" / log_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "execution.log"

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    return logger
