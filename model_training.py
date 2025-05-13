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
    
    # Create a dictionary to store the trained custom models
    custom_models = {}
    
    # Train custom models on feature-engineered data
    # Start a single parent run
    with mlflow.start_run(run_name=f"custom_models_{run_datetime}") as parent_run:
        # Loop through each model type and create a nested run
        for model_type in ['random_forest', 'gradient_boosting', 'decision_tree', 'xgboost']:
            # Train the model and store it in the dictionary
            custom_models[model_type] = train_custom_model(
                train_data=feature_eng_train_data,
                test_data=feature_eng_test_data,
                model_type=model_type
            )
    
    # Example prediction using the already trained models
    sample_data = test_data.head(10)
    
    # Use the stored models instead of training again
    rf_predictions = predict_trip_duration(custom_models['random_forest'], sample_data, feature_engineering=True)
    logger.info(f"Random Forest Predictions: {rf_predictions}")
    
    gb_predictions = predict_trip_duration(custom_models['gradient_boosting'], sample_data, feature_engineering=True)
    logger.info(f"Gradient Boosting Predictions: {gb_predictions}")

if __name__ == "__main__":
    main()