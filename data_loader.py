import pandas as pd
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processing.log')
    ]
)

def load_data(file_path: str, target_column: str = "Chance_of_Admit") -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Loads dataset from CSV file, preprocesses it, and returns both data and scaler.
    
    Args:
        file_path: Path to the CSV file
        target_column: Name of the target column
        
    Returns:
        Tuple of (DataFrame, fitted StandardScaler)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data loading or processing fails
    """
    try:
        logging.info(f"Loading dataset from: {file_path}")
        
        # Validate file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
            
        # Load
