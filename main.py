import os
from sklearn.model_selection import train_test_split

from utils import setup_mlflow_tracking
from data_preprocessing import read_dataframe, engineer_features
from model_training import train_models, train_custom_model, predict_trip_duration, engineer_features_improved
from loguru import logger
from data_preprocessing import load_multiple_dataframes, run_data_prep, prepare_dictionaries

import mlflow
from datetime import datetime
import logging


import click
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from data_preprocessing import run_data_prep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Generate run datetime
run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

@click.command()
@click.option(
    "--raw_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to directory containing raw parquet files"
)
@click.option(
    "--dest_path",
    required=True,
    type=click.Path(),
    help="Path to save processed data"
)
@click.option(
    "--dataset",
    default="green",
    type=click.Choice(['green', 'yellow'], case_sensitive=False),
    help="Type of taxi dataset"
)
# @click.option(
#     "--analyze",
#     is_flag=True,
#     help="Generate and display data analysis"
# )


def main(
    raw_data_path: str,
    dest_path: str,
    dataset: str,
    #analyze: bool
) -> None:
    """Main execution pipeline with data analysis options."""
    try:

        #Setup MLflow
        setup_mlflow_tracking()

        # Run data preparation
        logger.info("Starting data preparation...")
        df_train, df_test = run_data_prep(
            raw_data_path=raw_data_path,
            dest_path=dest_path,
            dataset=dataset
        )
        #check for null values
        print("CHECK NULL VALUES")
        df_null=  df_train.isnull().sum().sort_values(ascending=False)
        logger.info(f"NULL VALUES preview:\n{df_null}")


#         # Drop unwanted columns
#         unwanted_columns = ['store_and_fwd_flag', 'RatecodeID', 'congestion_surcharge', 'ehail_fee']
#         train_data = df_train.drop(columns=unwanted_columns, errors='ignore')
#         test_data = df_test.drop(columns=unwanted_columns, errors='ignore')
            # Handle missing values
        logger.info("Handling missing values...")
        for df in [df_train, df_test]:
            df.loc[df['passenger_count'].isnull(), 'passenger_count'] = df['passenger_count'].median()
            df.loc[df['payment_type'].isnull(), 'payment_type'] = df['payment_type'].mode()[0]
            if 'trip_type' in df.columns:
                df.loc[df['trip_type'].isnull(), 'trip_type'] = df['trip_type'].mode()[0]
        
        # Check for remaining null values
        df_null = df_train.isnull().sum().sort_values(ascending=False)
        logger.info(f"Remaining NULL values:\n{df_null}")

        # top_pairs = ['74_75', '74_236', '75_74', '74-166']  # your actual top N

        # # Replace PU_DO outside top pairs with 'other' in train and test
        # train_data['PU_DO'] = train_data['PU_DO'].apply(lambda x: x if x in top_pairs else 'other')
        # test_data['PU_DO'] = test_data['PU_DO'].apply(lambda x: x if x in top_pairs else 'other')

        

        # Train baseline models on raw data (with min feature selection: PU_DO9(cat)+ trip_distance(num))
        multiple_models = train_models(train_data, test_data)
        logger.info(f"Engineered Train Data preview:\n{feature_eng_train_data.head(10)}")

        # Apply enhanced feature engineering
        feature_eng_train_data = engineer_features_improved(df_train)
        feature_eng_test_data = engineer_features_improved(df_test)
        
        # Check for null values after feature engineering
        df_fe_null = feature_eng_train_data.isnull().sum().sort_values(ascending=False)
        logger.info(f"NULL values after feature engineering:\n{df_fe_null}")


        logger.info(f"Train data shape after trip_distance filtering: {feature_eng_train_data.shape}")
        logger.info(f"Test data shape after trip_distance filtering: {feature_eng_test_data.shape}")
        

        # Ends the previous active run
        # mlflow.end_run()

        # Create a dictionary to store the trained custom models
        custom_models = {}
        
        # Train custom models on feature-engineered data
        # Start a single parent run
        with mlflow.start_run(run_name=f"custom_models_{run_datetime}") as parent_run:
            # Loop through each model type and create a nested run
            for model_type in ['random_forest', 'gradient_boosting', 'decision_tree', 'lightgbm']:
                # Train the model and store it in the dictionary
                custom_models[model_type] = train_custom_model(
                    train_data=feature_eng_train_data,
                    test_data=feature_eng_test_data,
                    model_type=model_type
                )
        
        # Example prediction using the already trained models
        sample_data = feature_eng_test_data.head(10)
        
        # Use the stored models 
        rf_predictions = predict_trip_duration(custom_models, 'random_forest', input_data=sample_data, feature_engineering=True)
        logger.info(f"Random Forest Predictions: {rf_predictions}")
        
        gb_predictions = predict_trip_duration(custom_models, 'gradient_boosting', input_data=sample_data, feature_engineering=True)
        logger.info(f"Gradient Boosting Predictions: {gb_predictions}")
        
        # if analyze:
        #     # Perform analysis
        #     logger.info("\n=== Data Analysis ===")
        #     analyze_data(df_train, df_test)
            
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

# # def analyze_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
# #     """Generate and display basic data analysis."""
# #     # Basic info
# #     print(f"\nTraining data shape: {df_train.shape}")
# #     print(f"Test data shape: {df_test.shape}")
    
# #     # Duration analysis
# #     print("\nDuration Statistics:")
# #     print(pd.concat([
# #         df_train['duration'].describe().rename('train'),
# #         df_test['duration'].describe().rename('test')
# #     ], axis=1))
    
# #     # Temporal patterns
# #     if 'pickup_hour' in df_train.columns:
# #         print("\nPickup Hour Distribution:")
# #         print(df_train['pickup_hour'].value_counts().sort_index())
    
# #     # Location patterns
# #     if 'PU_DO' in df_train.columns:
# #         print("\nTop 10 Popular Routes:")
# #         print(df_train['PU_DO'].value_counts().head(10))

if __name__ == '__main__':
    main()