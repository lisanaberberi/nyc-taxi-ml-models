import os
from sklearn.model_selection import train_test_split

from utils import setup_mlflow_tracking
from data_preprocessing import read_dataframe
from model_training import train_models, train_custom_model, predict_trip_duration
from loguru import logger

def main():
    # Set up MLflow tracking
    setup_mlflow_tracking()

    # Define input file path
    #https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-07.parquet (from July to Dec 2024)
    input_file = 'data/green_tripdata_2024-07.parquet'  # Replace with your actual file path

    # Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found!")

    # Read and preprocess data
    df = read_dataframe(input_file)
    logger.info(f"Reading from data:\n{df.head(10)}")


    # Split data into train and validation sets
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

    # Train multiple models
    multiple_models = train_models(train_data, val_data)

    # Train custom random forest and gradient boosting models
    custom_rf_model = train_custom_model(train_data, val_data, model_type='random_forest')
    custom_gb_model = train_custom_model(train_data, val_data, model_type='gradient_boosting')
    custom_gb_model = train_custom_model(train_data, val_data, model_type='decision_tree')
    custom_gb_model = train_custom_model(train_data, val_data, model_type='xgboost')

    # Example prediction (you can modify this part as needed)
    sample_data = val_data.head(10)
    rf_predictions = predict_trip_duration(custom_rf_model, sample_data, feature_engineering=True)
    logger.info(f"Random Forest Predictions:{rf_predictions}")

    gb_predictions = predict_trip_duration(custom_gb_model, sample_data, feature_engineering=True)
    logger.info(f"Gradient Boosting Predictions: {gb_predictions}")

if __name__ == "__main__":
    main()