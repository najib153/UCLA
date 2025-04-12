import warnings
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from logging_config import configure_logging
from data_loader import load_data
from data_processing import handle_missing_values, encode_categorical_variables, process_target_variable
import joblib




if __name__ == "__main__":
    try:
         configure_logging()
         warnings.filterwarnings("ignore")

         # Load dataset
         file_path = "data/Admission.csv"
         target_column = "Admit_Chance"

         logging.info("Loading dataset...")
         data = load_data(file_path)
    
         # Data Preprocessing
         data = handle_missing_values(data)
         data, label_encoders = encode_categorical_variables(data)
         data = process_target_variable(data, target_column)

         # Feature Selection
         X = data.drop(columns=[target_column])
         y = data[target_column]

         # Feature Scaling
         scaler = StandardScaler()
         X_scaled = scaler.fit_transform(X)

         # Splitting Data
         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

         # Model Training # Logistic Regression model
         logging.info("Training Logistic Regression model...")
         model = LogisticRegression()
         model.fit(X_train, y_train)

         # Model Prediction # Logistic Regression model
         y_pred = model.predict(X_test)
         joblib.dump(scaler, "scaler.pkl")
    
         # Model Evaluation
         accuracy = accuracy_score(y_test, y_pred)
         logging.info(f"Model Accuracy: {accuracy:.4f}")
         print(f"Model Accuracy: {accuracy:.4f}")
         print("Classification Report:\n", classification_report(y_test, y_pred))

         # Model Training MLP Classifier
         logging.info("Training MLP Classifier...")
         mlp = MLPClassifier(hidden_layer_sizes=(3,3), batch_size=70, max_iter=100, random_state=123)
         mlp.fit(X_train, y_train)

         # Model Prediction # Model Prediction
         y_pred = mlp.predict(X_test)
         joblib.dump(mlp, "mlp_model.pkl")

         # Model Evaluation
         accuracy = accuracy_score(y_test, y_pred)
         logging.info(f"Model Accuracy: {accuracy:.4f}")
         print(f"Model Accuracy: {accuracy:.4f}")
         print("Classification Report:\n", classification_report(y_test, y_pred))



    
    
    
    except FileNotFoundError as e:
       logging.error(f"File not found: {e}")
    except Exception as e:
       logging.error(f"An unexpected error occurred: {e}")






