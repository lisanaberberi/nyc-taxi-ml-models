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
from data_preprocessing import engineer_features, prepare_dictionaries, preprocess_custom_features

from mlflow.models import infer_signature

from utils import log_model_parameters

# Generate run datetime
run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")


def train_models(train_data, test_data, target='duration'):
    """
    Train baseline models with no features using MLflow tracking.
    """
    y_train = train_data[target].values
    y_test = test_data[target].values

    # Baseline: minimal features
    y_train = train_data[target].values
    y_test = test_data[target].values

    # Prepare basic feature dicts
    train_dicts = prepare_dictionaries(train_data)
    test_dicts = prepare_dictionaries(test_data)

    # Vectorize
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)

    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=10, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'decision_tree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }

    with mlflow.start_run(run_name=f"baseline_models_{run_datetime}") as parent_run:
        results = {}

        for model_name, model in models.items():
            run_name = f"baseline_{model_name}_{run_datetime}"
            with mlflow.start_run(nested=True, run_name=run_name):
                pipeline = make_pipeline(dv, model)
                pipeline.fit(train_dicts, y_train)
                preds = pipeline.predict(test_dicts)

                # Log params and metrics
                log_model_parameters(model, f"baseline_{model_name}")
                log_model_metrics(y_test, preds, f"baseline_{model_name}")

                mlflow.log_input(mlflow.data.from_pandas(train_data, source="file:///data/green_tripdata_2024-07.parquet"), context="training")

                signature = infer_signature(X_train, preds[:5])
                input_example = X_train[:5]

                if model_name == 'xgboost':
                    booster = pipeline.named_steps['xgbregressor']
                    mlflow.xgboost.log_model(
                        xgb_model=booster,
                        artifact_path=f"baseline_{model_name}_model",
                        input_example=input_example,
                        signature=signature
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path=f"baseline_{model_name}_model",
                        input_example=input_example,
                        signature=signature
                    )

                results[model_name] = {'model': pipeline, 'y_pred': preds}

        return results


# def collapse_top_n(df, col, n=50):
#     top = df[col].value_counts().nlargest(n).index
#     df = df.copy()
#     df[col] = df[col].cat.add_categories('other')
#     df[col] = df[col].where(df[col].isin(top), 'other')
#     return df

def train_custom_model(train_data, test_data, target='duration', model_type='random_forest'):
    """
    Train a specific type of custom model on engineered features using MLflow tracking.
    """

    # Engineer features first
    train_data_eng = engineer_features(train_data)
    test_data_eng = engineer_features(test_data)

    # Preprocess features into dicts for DictVectorizer
    dict_train = preprocess_custom_features(train_data_eng)
    dict_test = preprocess_custom_features(test_data_eng)

    # Vectorize
    dv = DictVectorizer()
    X_train = dv.fit_transform(dict_train)
    X_test = dv.transform(dict_test)

    # Targets
    y_train = train_data_eng[target].values
    y_test = test_data_eng[target].values

    # Model configs
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=10, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'decision_tree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }

    model = models[model_type]
    run_name = f"{model_type}_custom_model_{run_datetime}"

    results = {}

    with mlflow.start_run(run_name=run_name) as run:
        pipeline = make_pipeline(dv, model)
        pipeline.fit(dict_train, y_train)
        preds = pipeline.predict(dict_test)

        # Log model params & metrics
        log_model_parameters(model, f"{model_type}_custom")
        log_model_metrics(y_test, preds, f"{model_type}_custom")

        # Log training data as input artifact
        mlflow.log_input(mlflow.data.from_pandas(train_data_eng), context="training")

        # Create MLflow model signature
        signature = infer_signature(X_train, preds[:5])
        input_example = X_train[:5]

        if model_type == 'xgboost':
            booster = pipeline.named_steps['xgbregressor']
            mlflow.xgboost.log_model(
                booster, f"{model_type}_custom_model",
                input_example=input_example,
                signature=signature
            )
        else:
            mlflow.sklearn.log_model(
                pipeline, f"{model_type}_custom_model",
                input_example=input_example,
                signature=signature
            )

        results[model_type] = {'model': pipeline, 'y_pred': preds}

    return results



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