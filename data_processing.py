import logging

def process_target_variable(data, target_column):
    try:
        logging.info("Processing target variable...")
        data[target_column] = (data[target_column] >= 0.8).astype(int)
        logging.info("Target variable processed successfully.")
        return data
    except KeyError:
        logging.error(f"Error: '{target_column}' column not found in dataset.")
        raise
    except Exception as e:
        logging.error(f"Error processing target variable: {e}")
        raise

def handle_missing_values(data):
    """Handles missing values by filling them with the median for numerical columns and mode for categorical columns."""
    for column in data.columns:
        if data[column].dtype == 'object':
           data[column].fillna(data[column].mode()[0], inplace=True)
        else:
           data[column].fillna(data[column].median(), inplace=True)
           logging.info("Missing values handled successfully.")
        return data

def encode_categorical_variables(data):
     """Encodes categorical variables using Label Encoding."""
     label_encoders = {}
     for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
        logging.info("Categorical variables encoded successfully.")
     return data, label_encoders

def process_target_variable(data, target_column):
    """Processes the target variable by converting it to a binary classification if necessary."""
    if data[target_column].dtype == 'float64' or data[target_column].dtype == 'int64':
        median_value = data[target_column].median()
        data[target_column] = (data[target_column] >= median_value).astype(int)
    else:
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        logging.info("Target variable processed successfully.")
    return data