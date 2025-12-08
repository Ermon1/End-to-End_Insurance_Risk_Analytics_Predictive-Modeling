import pandas as pd
from pathlib import Path
from src.utility.MLexception import MLException

class DataLoader:
    def __init__(self):
        self.root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.root / "data"
        if not self.data_dir.exists():
            raise MLException(f"Data directory not found at {self.data_dir}")

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise MLException(f"CSV file {filename} not found in {self.data_dir}")
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            raise MLException(f"Failed to load CSV {filename}", error=e)

loader = DataLoader()
