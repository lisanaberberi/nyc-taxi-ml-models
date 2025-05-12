import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split

from utils import log_model_metrics
from data_preprocessing import engineer_features, prepare_dictionaries


# Generate run datetime
run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

def train_models(train_data, val_data, target='duration'):
    """
    Train multiple models with MLflow tracking
    
    Args:
    - train_data (pd.DataFrame): Training data
    - val_data (pd.DataFrame): Validation data
    - target (str): Target variable name
    
    Returns:
    - Dictionary of trained models and their performance
    """
    # Prepare data
    y_train = train_data[target].values
    y_val = val_data[target].values
    dict_train = prepare_dictionaries(train_data)
    dict_val = prepare_dictionaries(val_data)
    
    # Define models to train
    models = {
        'random_forest': {
            'model': make_pipeline(
                DictVectorizer(),
                RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=10, random_state=42)
            ),
            'params': {
                'max_depth': 20, 
                'n_estimators': 100, 
                'min_samples_leaf': 10
            }
        },
        'xgboost': {
            'model': make_pipeline(
                DictVectorizer(),
                xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            }
        },
        'decision_tree': {
            'model': make_pipeline(
                DictVectorizer(),
                DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
            ),
            'params': {
                'max_depth': 10,
                'min_samples_leaf': 5
            }
        },
        'gradient_boosting': {
            'model': make_pipeline(
                DictVectorizer(),
                GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            }
        }
    }
    
    # Train and log models
    results = {}
    for model_name, model_config in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{run_datetime}"):
            # Fit model
            model = model_config['model']
            model.fit(dict_train, y_train)

            # Log dataset source files or in-memory DataFrames as InputDatasets
            mlflow.log_input(mlflow.data.from_pandas(train_data, source="file://data/green_tripdata_2024-07.parquet"), context="training")
            mlflow.log_input(mlflow.data.from_pandas(val_data, source="file://data/green_tripdata_2024-11.parquet"), context="validation")
        
            
            # Predict and evaluate
            y_pred = model.predict(dict_val)
            
            # Log metrics
            log_model_metrics(
                y_val, 
                y_pred, 
                model_name, 
                params=model_config['params']
            )
            

            # Log model
            mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model")
            # Store results
            results[model_name] = {
                'model': model,
                'y_pred': y_pred
            }
        mlflow.end_run()
    return results

def train_custom_model(train_data, val_data, target='duration', model_type='random_forest'):
    """
    Train a custom model with feature engineering
    
    Args:
    - train_data (pd.DataFrame): Training data
    - val_data (pd.DataFrame): Validation data
    - target (str): Target variable name
    - model_type (str): Type of model to train
    
    Returns:
    - Trained model pipeline
    """
    # Engineer features
    train_data = engineer_features(train_data)
    val_data = engineer_features(val_data)
    
    # Prepare features
    features = [
        'trip_distance', 
        'PULocationID', 
        'DOLocationID', 
        'pickup_hour', 
        'pickup_month', 
        'is_weekend', 
        'PU_DO',
        'time_of_day'
    ]
    
    # Define features
    numerical_features = ['trip_distance', 'pickup_hour', 'pickup_month', 'is_weekend']
    categorical_features = ['PULocationID', 'DOLocationID', 'PU_DO', 'time_of_day']

    # Prepare X and y
    X_train = train_data[numerical_features + categorical_features]
    y_train = train_data[target]
    X_val = val_data[numerical_features + categorical_features]
    y_val = val_data[target]
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Model selection
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    

    # Start MLflow run
    with mlflow.start_run(run_name=f"custom_{model_type}_{run_datetime}"):

        # Log feature engineered datasets
        mlflow.log_input(mlflow.data.from_pandas(train_data, source="feature_eng_training"), context="training_engineered")
        mlflow.log_input(mlflow.data.from_pandas(val_data, source="feature_eng_validation"), context="validation_engineered")
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_val)
        
        # Log metrics
        log_model_metrics(y_val, y_pred, f"custom_{model_type}")
        
        # Log model
        mlflow.sklearn.log_model(pipeline, f"custom_{model_type}_model")
    
    return pipeline

def predict_trip_duration(model, input_data, feature_engineering=False):
    """
    Predict trip duration using a trained model
    
    Args:
    - model: Trained model pipeline
    - input_data: DataFrame with trip features
    - feature_engineering (bool): Whether to apply feature engineering
    
    Returns:
    - Predicted trip durations
    """
    # Apply feature engineering if specified
    if feature_engineering:
        input_data = engineer_features(input_data)
        
        # Select features used during training
        features = [
            'trip_distance', 
            'PULocationID', 
            'DOLocationID', 
            'pickup_hour', 
            'pickup_month', 
            'is_weekend', 
            'PU_DO',
            'time_of_day'
        ]
        input_data = input_data[features]
    else:
        # Prepare input as dictionary
        input_dict = prepare_dictionaries(input_data)
        input_data = input_dict
    
    # Predict
    predictions = model.predict(input_data)
    
    return predictions