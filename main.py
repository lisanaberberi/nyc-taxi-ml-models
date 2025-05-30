import os
from sklearn.model_selection import train_test_split
from utils import setup_mlflow_tracking
from model_training import (
    train_models, 
    train_custom_model, 
    predict_trip_duration, 
    evaluate_features_fast,  # Updated function name
    optimize_hyperparameters,
    run_optimization_workflow
)
import mlflow
from datetime import datetime
import logging
import click
from pathlib import Path
from typing import Tuple
import pandas as pd
from data_preprocessing import (
    load_multiple_dataframes, 
    run_data_prep, 
    basic_features, 
    feature_eng, 
    read_dataframe
)

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
@click.option(
    "--taxi_zones_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to taxi_zones.shp file (optional)"
)
@click.option(
    "--feature_type", 
    default="feature", 
    type=click.Choice(['feature', 'basic']), 
    help="Type of feature engineering to apply"
)
@click.option(
    "--run_baseline",
    is_flag=True,
    default=False,
    help="Run baseline models training"
)
@click.option(
    "--run_custom",
    is_flag=True,
    default=False,
    help="Run custom models training"
)
@click.option(
    "--run_optimization",
    is_flag=True,
    default=False,
    help="Run hyperparameter optimization"
)
@click.option(
    "--n_trials",
    default=50,
    type=int,
    help="Number of optimization trials for Optuna"
)
@click.option(
    "--evaluate_features",
    is_flag=True,
    default=False,
    help="Run feature evaluation"
)
def main(
    raw_data_path: str,
    dest_path: str,
    dataset: str,
    taxi_zones_path: str,
    feature_type: str,
    run_baseline: bool,
    run_custom: bool,
    run_optimization: bool,
    n_trials: int,
    evaluate_features: bool
) -> None:
    """Main execution pipeline with configurable training options."""
    try:
        # Setup MLflow
        logger.info("Setting up MLflow tracking...")
        setup_mlflow_tracking()
        
        # Run data preparation
        logger.info("Starting data preparation...")
        df_train, df_test = run_data_prep(
            raw_data_path=raw_data_path,
            dest_path=dest_path,
            dataset=dataset,
            taxi_zones_path=taxi_zones_path,
            feature_type=feature_type
        )
        
        # Data quality checks
        logger.info("Performing data quality checks...")
        print("CHECK NULL VALUES")
        df_null = df_train.isnull().sum().sort_values(ascending=False)
        logger.info(f"NULL VALUES preview:\n{df_null}")
        
        
        logger.info(f"Train data shape after cleaning: {df_train.shape}")
        logger.info(f"Test data shape after cleaning: {df_test.shape}")
    
        
        # Train baseline models if requested
        baseline_results = None
        if run_baseline:
            logger.info("Training baseline models...")

            # Apply basic feature engineering
            logger.info("Applying basic feature engineering...")
            df_basic_train, df_cleaned_train= basic_features(df_train)
            df_basic_test, df_cleaned_test= basic_features(df_test)
            baseline_results = train_models(df_basic_train, df_basic_test)
            #baseline_results = train_models(df_cleaned_train, df_cleaned_test)
            logger.info("Baseline models training completed")
        
        # Feature evaluation if requested
        if evaluate_features:
            if feature_type== 'basic':
                logger.info("Evaluating feature importance...")
                df_basic_train, df_cleaned_train= basic_features(df_train)
                df_basic_test, df_cleaned_test= basic_features(df_test)
                eval_features = evaluate_features_fast(df_cleaned_train, df_cleaned_test, 'duration')
            else:
                logger.info("Evaluating feature importance after  feature engineering...")
                df_fe_train = feature_eng(df_train)
                df_fe_test = feature_eng(df_test)
                eval_features = evaluate_features_fast(df_fe_train, df_fe_test, 'duration')
                
            logger.info("Top 10 most important features:")
            for feature, score in eval_features[:10]:
                logger.info(f"  {feature}: {score:.4f}")
        
        # Apply advanced feature engineering
        logger.info("Applying advanced feature engineering...")
        df_fe_train = feature_eng(df_train)
        df_fe_test = feature_eng(df_test)
        
        logger.info(f"Feature engineered train data shape: {df_fe_train.shape}")
        logger.info(f"Feature engineered test data shape: {df_fe_test.shape}")
        
        # Train custom models if requested
        custom_models = {}
        if run_custom:
            logger.info("Training custom models...")
            custom_models = train_custom_model(
                    train_data=df_fe_train,
                    test_data=df_fe_test
                    )
            
            # Define model types to train
            # model_types = ['random_forest', 'gradient_boosting', 'decision_tree', 'lightgbm']
            
            # # Train each model type
            # for model_type in model_types:
            #     logger.info(f"Training {model_type}...")
            #     try:
            #         custom_models[model_type] = train_custom_model(
            #             train_data=df_fe_train,
            #             test_data=df_fe_test,
            #             model_type=model_type
            #         )
            #         logger.info(f"{model_type} training completed successfully")
            #     except Exception as e:
            #         logger.error(f"Error training {model_type}: {str(e)}")
            #         continue
            
        logger.info(f"Custom models training completed. Trained {len(custom_models)} models.")
        
        # Run hyperparameter optimization if requested
        optimization_results = {}
        if run_optimization:
            logger.info("Starting hyperparameter optimization...")
            
            # Define models to optimize (subset for efficiency)
            models_to_optimize = ['random_forest', 'lightgbm']
            
            try:
                optimization_results = run_optimization_workflow(
                    train_data=df_fe_train,
                    test_data=df_fe_test,
                    target='duration',
                    model_types=models_to_optimize,
                    n_trials=n_trials
                )
                
                logger.info("Hyperparameter optimization completed")
                for model_type, result in optimization_results.items():
                    logger.info(f"{model_type} - Best RMSE: {result['best_score']:.4f}")
                    logger.info(f"{model_type} - Best params: {result['best_params']}")
                    
            except Exception as e:
                logger.error(f"Error during optimization: {str(e)}")
        
        # Demonstration: Make predictions using trained models
        if custom_models:
            logger.info("Demonstrating predictions...")
            sample_data = df_fe_test.head(10)
            
            for model_type, model_info in custom_models.items():
                try:
                    predictions = predict_trip_duration(
                        custom_models, 
                        model_type, 
                        input_data=sample_data, 
                        feature_engineering=True
                    )
                    logger.info(f"{model_type} predictions (first 5): {predictions[:5]}")
                except Exception as e:
                    logger.error(f"Error making predictions with {model_type}: {str(e)}")
        
        #Summary report
        logger.info("\n=== PIPELINE SUMMARY ===")
        logger.info(f"Data processed: {df_train.shape[0]} training samples, {df_test.shape[0]} test samples")
        logger.info(f"Baseline models trained: {'Yes' if run_baseline else 'No'}")
        logger.info(f"Custom models trained: {len(custom_models) if custom_models else 0}")
        logger.info(f"Hyperparameter optimization: {'Yes' if run_optimization else 'No'}")
        logger.info(f"Feature evaluation: {'Yes' if evaluate_features else 'No'}")
        
        if optimization_results:
            logger.info("\nBest optimization results:")
            for model_type, result in optimization_results.items():
                logger.info(f"  {model_type}: RMSE = {result['best_score']:.4f}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise



if __name__ == '__main__':
    main()