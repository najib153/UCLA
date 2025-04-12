import pandas as pd
import logging
import joblib
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    try:
        logging.info("Loading the dataset...")
        data = pd.read_csv(file_path)

        # Drop Serial_No if it's not useful for prediction
        data = data.drop(columns=["Serial_No"], errors='ignore')

        # Then fit the scaler
        scaler = StandardScaler()
        scaler.fit(data.drop(columns=["Chance_of_Admit"], errors='ignore'))  # Assuming that's your target

        # Save the scaler
        joblib.dump(scaler, "scaler.pkl")

        logging.info("Dataset loaded and scaler saved successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"Error: The specified file '{file_path}' was not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


def show_data(data, n_rows=5):
    """
    Displays the first few rows of a DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        print("\n" + "=" * 100)
        print(f"Data Shape: {data.shape}")
        print(f"First {n_rows} rows:")
        print(data.head(n_rows))
        print("=" * 100 + "\n")
        print(data.describe())  # Add () to actually call describe()
