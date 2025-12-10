# src/models/task_4/load_data.py

from src.utility.config_loader import loader as config_loader
from src.utility.data_loader import loader as data_loader

def load_task4_data():
    """
    Load raw insurance data using the project config and data loader.
    Returns:
        df (pd.DataFrame): Raw CSV data as a DataFrame
    """
    # 1️⃣ Load the path from YAML config
    config = config_loader.load('data.yaml')  # read config
    data_path = config['data']['raw_data_path']  # path to raw CSV

    # 2️⃣ Load the CSV into pandas DataFrame
    df = data_loader.load_csv(data_path, sep='|')  # you mentioned separator '|'


    return df
