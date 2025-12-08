# tests/test_custom_exception.py
from src.utility.logger import get_logger
from src.utility.MLexception import MLException

def simple_ml_exception_check():
    logger = get_logger(name="test_exception", log_name="test_exception")
    
    try:
        # Raise a simple exception
        raise ValueError("Original error")
    except Exception as e:
        # Wrap in MLException
        exc = MLException(message="Stage failed", error=e)
        logger.error(str(exc))
        print("MLException works:", str(exc))

if __name__ == "__main__":
    simple_ml_exception_check()
