import yaml
from pathlib import Path
from src.utility import MLexception
class ConfigLoader:
    """
    Minimal config loader for ML pipelines.
    Resolves configs folder relative to project root.
    """

    def __init__(self):
        # Root is project root: two levels above src/utility/
        self.root = Path(__file__).resolve().parent.parent.parent
        self.configs_dir = self.root / "configs"

        if not self.configs_dir.exists():
            raise FileNotFoundError(f"Configs directory not found at {self.configs_dir}")

    def load(self, filename: str) -> dict:
        """
        Load YAML config by filename from configs folder.
        
        Args:
            filename (str): name of the config file (e.g., "data.yaml")
            
        Returns:
            dict: loaded configuration
        """
        config_path = self.configs_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {filename} not found in {self.configs_dir}")
        
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


loader = ConfigLoader()

# ------------------------------
# Usage example
# ------------------------------
