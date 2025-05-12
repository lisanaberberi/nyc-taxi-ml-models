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

from mlflow.models import infer_signature

# Generate run datetime
run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

def train_models(train_data, test_data, target='duration'):
    """
    Train multiple models with MLflow tracking using train/test only
    
    Args:
    - train_data (pd.DataFrame): Training data
    - test_data (pd.DataFrame): Test data
    - target (str): Target variable name
    
    Returns:
    - Dictionary of trained models and their performance
    """
    y_train = train_data[target].values
    y_test = test_data[target].values
    dict_train = prepare_dictionaries(train_data)
    dict_test = prepare_dictionaries(test_data)

    
    models = {
        'random_forest': {
            'model': make_pipeline(
                DictVectorizer(),
                RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=10, random_state=42)
            )
        },
        'xgboost': {
            'model': make_pipeline(
                DictVectorizer(),
                xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            )
        },
        'decision_tree': {
            'model': make_pipeline(
                DictVectorizer(),
                DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
            )
        },
        'gradient_boosting': {
            'model': make_pipeline(
                DictVectorizer(),
                GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            )
        }
    }
    
    results = {}
    for model_name, model_config in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{run_datetime}"):
            model = model_config['model']
            model.fit(dict_train, y_train)

            mlflow.log_input(mlflow.data.from_pandas(train_data, source="file:///data/green_tripdata_2024-07.parquet"), context="training")

            # after training your model
            input_example = train_data.iloc[:5]  # or a representative example
            signature = infer_signature(train_data, model.predict(dict_train))

            y_pred = model.predict(dict_test)
            log_model_metrics(y_test, y_pred, model_name)

            mlflow.sklearn.log_model(model, 
                                     artifact_path=f"{model_name}_model",
                                             input_example=input_example,
                                             signature=signature)
            results[model_name] = {'model': model, 'y_pred': y_pred}

        #mlflow.end_run()
    return results

def train_custom_model(train_data, test_data, target='duration', model_type='random_forest'):
    """
    Train a custom model with feature engineering using train/test only
    
    Args:
    - train_data (pd.DataFrame): Training data
    - test_data (pd.DataFrame): Test data
    - target (str): Target variable name
    - model_type (str): Model type
    
    Returns:
    - Trained model pipeline
    """
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)

    # Print DataFrames after feature engineering
    print("Train Data after Feature Engineering:")
    print(train_data.head())

    print("\nTest Data after Feature Engineering:")
    print(test_data.head())


    int_cols = ['pickup_hour', 'pickup_month', 'is_weekend']

    # Convert integer columns to Int64 (nullable integers)
    for col in int_cols:
        train_data[col] = train_data[col].astype('Int64')
        test_data[col] = test_data[col].astype('Int64')


    features = [
        'trip_distance', 'PULocationID', 'DOLocationID',
        'pickup_hour', 'pickup_month', 'is_weekend',
        'PU_DO', 'time_of_day'
    ]

    numerical_features = ['trip_distance', 'pickup_hour', 'pickup_month', 'is_weekend']
    categorical_features = ['PULocationID', 'DOLocationID', 'PU_DO', 'time_of_day']

    X_train = train_data[numerical_features + categorical_features]
    y_train = train_data[target]
    X_test = test_data[numerical_features + categorical_features]
    y_test = test_data[target]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

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

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    with mlflow.start_run(run_name=f"custom_{model_type}_{run_datetime}"):

        # Inference sample
        input_example = X_train.head(5)

        mlflow.log_input(mlflow.data.from_pandas(input_example), context="training_engineered")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        signature = infer_signature(X_train, model.predict(X_train))

        log_model_metrics(y_test, y_pred, f"custom_{model_type}")
        mlflow.sklearn.log_model(pipeline, f"custom_{model_type}_model", 
                                 input_example=input_example,
                                 signature=signature)

    return pipeline

def predict_trip_duration(model, input_data, feature_engineering=False):
    """
    Predict trip duration
    
    Args:
    - model: Trained model pipeline
    - input_data: DataFrame with trip features
    - feature_engineering (bool): Whether to apply feature engineering
    
    Returns:
    - Predictions
    """
    if feature_engineering:
        input_data = engineer_features(input_data)
        features = [
            'trip_distance', 'PULocationID', 'DOLocationID',
            'pickup_hour', 'pickup_month', 'is_weekend',
            'PU_DO', 'time_of_day'
        ]
        input_data = input_data[features]
    else:
        input_data = prepare_dictionaries(input_data)

    return model.predict(input_data)
