# preprocess.py
import pandas as pd
from pathlib import Path

class DataLoader:
    """
    Load and preprocess insurance data.
    
    Input: filename (string, CSV)
    Output: cleaned pandas DataFrame with LossRatio computed
    """
    def __init__(self, data_dir: str = "../../data/raw"):
        self.data_dir = Path(data_dir)

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        path = self.data_dir / filename
        df = pd.read_csv(path, **kwargs)
        return df

def compute_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute LossRatio and handle division issues.
    """
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"]
    df["LossRatio"].replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    return df
