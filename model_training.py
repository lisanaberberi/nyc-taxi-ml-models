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

from utils import log_model_parameters

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

     # Start a parent run for all baseline models
    with mlflow.start_run(run_name=f"baseline_models_{run_datetime}") as parent_run:
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
            # Create nested runs within the parent run
            with mlflow.start_run(nested=True, run_name=f"{model_name}_{run_datetime}"):
            #with mlflow.start_run(run_name=f"{model_name}_{run_datetime}"):
                model = model_config['model']
                model.fit(dict_train, y_train)
                
                # Log model parameters
                log_model_parameters(model, 'my_model')

                mlflow.log_input(mlflow.data.from_pandas(train_data, source="file:///data/green_tripdata_2024-07.parquet"), context="training")

                # after training your model

                input_example = train_data.head(5)
                input_example = input_example.where(pd.notnull(input_example), None)  # replace NaN with None


                #input_example = train_data.iloc[:5]  # or a representative example
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


def collapse_top_n(df, col, n=50):
    top = df[col].value_counts().nlargest(n).index
    df = df.copy()
    df[col] = df[col].cat.add_categories('other')
    df[col] = df[col].where(df[col].isin(top), 'other')
    return df

def train_custom_model(train_data, test_data, target='duration', model_type='random_forest'):
    """
    Train a custom model with feature engineering and optimized MLflow logging
    
    Args:
    - train_data (pd.DataFrame): Training data
    - test_data (pd.DataFrame): Test data
    - target (str): Target variable name
    - model_type (str): Model type
    
    Returns:
    - Trained model pipeline
    """

    # Feature engineering
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)
    
    # Prepare features
    numerical_features = ['trip_distance', 'pickup_hour', 'pickup_month', 'is_weekend']
    #categorical_features = ['PULocationID', 'DOLocationID', 'PU_DO', 'time_of_day']
    categorical_features = ['PU_DO', 'time_of_day']
    
    X_train = train_data[numerical_features + categorical_features]
    y_train = train_data[target]
    X_test = test_data[numerical_features + categorical_features]
    y_test = test_data[target]

    # sample_size = 1000
    # train_data_small = train_data.sample(n=sample_size, random_state=42)
    # test_data_small  = test_data.sample(n=sample_size, random_state=42)
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    

    # Model selection
    model_map = {
        'random_forest': RandomForestRegressor(n_estimators=5, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=5, max_depth=5, learning_rate=0.1, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=5, max_depth=6, learning_rate=0.1, random_state=42),
        'decision_tree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    }
    print("Test2")
    
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model_map[model_type])
    ])
    
    # MLflow logging with minimal overhead
    with mlflow.start_run(nested=True, run_name=f"custom_model_{model_type}_{run_datetime}"):
        try:

            print(X_train.shape, X_train.head())
            print(y_train.shape, y_train.head())
            print(X_train.isnull().sum())

            # Collapse rare categories
            for c in ['PULocationID','DOLocationID','PU_DO']:
                train_data = collapse_top_n(train_data, c, n=50)
                test_data  = collapse_top_n(test_data,  c, n=50)

            pipeline.fit(X_train, y_train)
            print("TEST3")
            y_pred = pipeline.predict(X_test)

            # Log model parameters
            #log_model_parameters(model_type, 'my_model')
            log_model_parameters(pipeline.named_steps['regressor'], 'custom_' + model_type)
            
            # Log metrics and metadata
            log_model_metrics(y_test, y_pred, f"custom_{model_type}")
            mlflow.set_tags({
                'model_type': f'custom_{model_type}',
                'feature_engineering': 'applied'
            })

            mlflow.sklearn.log_model(pipeline, f"custom_{model_type}_model")#, 
                        #input_example=train_data,
                        # signature=signature)
            
            # Efficient model logging
            # small_sample = X_train.head(3)
            # mlflow.sklearn.log_model(
            #     pipeline, 
            #     artifact_path=f"custom_{model_type}_model",
            #     input_example=small_sample,
            #     signature=mlflow.models.infer_signature(
            #         small_sample, 
            #         pipeline.predict(small_sample)
            #     )
            #)
        
        except Exception as e:
            mlflow.log_param('error', str(e))
            print(f"Error in custom model {model_type}: {e}")

    return pipeline

def predict_trip_duration(model, input_data, feature_engineering=False):
    """
    Predict trip duration with optional feature engineering
    """
    if feature_engineering:
        input_data = engineer_features(input_data)
        features = [
            'trip_distance', 'PULocationID', 'DOLocationID',
            'pickup_hour', 'pickup_month', 'is_weekend',
            'PU_DO', 'time_of_day'
        ]
        input_data = input_data[features]
    
    return model.predict(input_data)