from src.utility.config_loader import loader
from src.utility.MLexception import MLException

def manual_test_config_loader():
    print("=== Manual test: ConfigLoader ===\n")

    # 1️⃣ Test loading existing config
    try:
        config = loader.load("data.yaml")  # Make sure this exists in configs/
        print("✅ Existing config loaded successfully")
        print("Keys:", list(config.keys()))
    except Exception as e:
        raise MLException("Failed to load existing config", error=e)

    # 2️⃣ Test loading non-existent config


manual_test_config_loader()