import os
from sklearn.model_selection import train_test_split

from utils import setup_mlflow_tracking
from data_preprocessing import read_dataframe, engineer_features
from model_training import train_models, train_custom_model, predict_trip_duration
from loguru import logger
from data_preprocessing import load_multiple_dataframes

import mlflow
from datetime import datetime

# Generate run datetime
run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    # Set up MLflow tracking
    setup_mlflow_tracking()

    # Define input file path
    # https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-07.parquet (from July to Dec 2024)
    # Define data path, dataset type, and months to process
    raw_data_path = 'data'
    dataset = 'green'
    months = ['2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12']

    # Load and combine data for all months
    df = load_multiple_dataframes(raw_data_path, dataset, months)
    logger.info(f"Loaded combined data: {df.shape}")

    # Split the raw data into train and test sets before feature engineering
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Raw Train Data preview:\n{train_data.head(10)}")

    # Feature engineering only for custom models
    feature_eng_train_data = engineer_features(train_data)
    feature_eng_test_data = engineer_features(test_data)
    logger.info(f"Engineered Train Data preview:\n{feature_eng_train_data.head(10)}")
    
    # Train models on raw data (without feature engineering)
    multiple_models = train_models(train_data, test_data)
    
    # Train custom models on feature-engineered data
        # Start a single parent run
    with mlflow.start_run(run_name=f"custom_models_{run_datetime}") as parent_run:
        # Loop through each model type and create a nested run
        for model_type in ['random_forest', 'gradient_boosting', 'decision_tree', 'xgboost']:
            train_custom_model(
                train_data=feature_eng_train_data,
                test_data=feature_eng_test_data,
                model_type=model_type
            )

    # Example prediction 
    sample_data = test_data.head(10)

    custom_rf_model = train_custom_model(feature_eng_train_data, feature_eng_test_data, model_type='random_forest')
    rf_predictions = predict_trip_duration(custom_rf_model, sample_data, feature_engineering=True)
    logger.info(f"Random Forest Predictions: {rf_predictions}")

    custom_gb_model = train_custom_model(feature_eng_train_data, feature_eng_test_data, model_type='gradient_boosting')
    gb_predictions = predict_trip_duration(custom_gb_model, sample_data, feature_engineering=True)
    logger.info(f"Gradient Boosting Predictions: {gb_predictions}")

if __name__ == "__main__":
    main()
