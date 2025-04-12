import warnings
from logging_config import configure_logging
from data_loader import load_data, show_data
from data_processing import process_target_variable, handle_missing_values, encode_categorical_variables, process_target_variable

if __name__ == "__main__":
    configure_logging()
    warnings.filterwarnings("ignore")
   
    filepath = r"data/Admission.xlsx"

    target_column = "Admit_Chance"
    
    data = load_data(filepath)
    data = process_target_variable(data, target_column)
    show_data(data)
    print(data.isna().sum())

    
    # Further processing and model implementation would go here
    #handle_missing_values(data)
    #encode_categorical_variables(data)
    #process_target_variable(data)
