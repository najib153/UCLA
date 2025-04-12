import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")

try:
    # Load the dataset
    logging.info("Loading the dataset...")
    data = pd.read_csv('Admission.csv')
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error("Error: The specified file 'Admission.csv' was not found.")
    raise
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise

try:
    # Convert the target variable into a categorical variable
    logging.info("Processing target variable...")
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    logging.info("Target variable processed successfully.")
except KeyError:
    logging.error("Error: 'Admit_Chance' column not found in dataset.")
    raise
except Exception as e:
    logging.error(f"Error processing target variable: {e}")
    raise

# Further processing and model implementation would go here with similar logging and error handling
