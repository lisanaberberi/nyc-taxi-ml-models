import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from utils import log_model_metrics
from data_preprocessing import engineer_features, prepare_dictionaries

from mlflow.models import infer_signature

from utils import log_model_parameters

import matplotlib.pyplot as plt
import os
import seaborn as sns

# Generate run datetime
run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")


def train_models(train_data, test_data, target='duration'):
    """
    Train baseline models with only basic feature using MLflow tracking.
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
        #'xgboost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'lightgbm': LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
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

                # if model_name == 'lightgbm':
                #     booster = pipeline.named_steps['regressor']
                #     mlflow.lightgbm.log_model(
                #         lgbm_model=booster,
                #         artifact_path=f"baseline_{model_name}_model",
                #         input_example=input_example,
                #         signature=signature
                #     )
                # else:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path=f"baseline_{model_name}_model",
                    input_example=input_example,
                    signature=signature
                )

                # Plot the predictions against the ground_truth
                log_prediction_plot(y_test, preds, f"baseline_{model_name}")

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

    # Engineer features
    train_data_eng = engineer_features(train_data)
    test_data_eng = engineer_features(test_data)

    # Targets
    y_train = train_data_eng[target].values
    y_test = test_data_eng[target].values

    # Model configs
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=10, random_state=42),
        'lightgbm': LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'decision_tree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }

    model = models[model_type]
    run_name = f"{model_type}_custom_model_{run_datetime}"

    # Define your feature lists
    categorical_columns = ['PULocationID','PU_DO', 'time_of_day']
    numerical_columns = ['trip_distance', 'pickup_hour', 'pickup_month', 'is_weekend']

    # Prepare tags dictionary
    tags = {f"categorical_{col}": col for col in categorical_columns}
    tags.update({f"numerical_{col}": col for col in numerical_columns})

    results = {}

    with mlflow.start_run(nested=True, run_name=run_name) as run:
        mlflow.set_tags(tags)

        # Column Transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_columns),

                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
            ]
        )

        # Complete pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Train model
        pipeline.fit(train_data_eng, y_train)
        preds = pipeline.predict(test_data_eng)

        # Log model params & metrics
        log_model_parameters(model, f"{model_type}_custom")
        log_model_metrics(y_test, preds, f"{model_type}_custom")

        # Log training data as input artifact
        mlflow.log_input(mlflow.data.from_pandas(train_data_eng), context="training")

        # Create MLflow model signature
        signature = infer_signature(train_data_eng, preds[:5])
        input_example = train_data_eng.head(5)

        # Log model pipeline
        # if model_type == 'lightgbm':
        #     mlflow.lightgbm.log_model(
        #         pipeline.named_steps['model'].booster_,  # access underlying booster
        #         f"{model_type}_custom_model",
        #         input_example=input_example,
        #         signature=signature
        #     )
        #else:
        mlflow.sklearn.log_model(
            pipeline,
            f"{model_type}_custom_model",
            input_example=input_example,
            signature=signature
        )

        #PLot the predictions against the ground_truth
        log_prediction_plot(y_test, preds, f"{model_type}_custom")


        results[model_type] = {'model': pipeline, 'y_pred': preds}

    return results[model_type]


def predict_trip_duration(results, model_type, input_data, feature_engineering=False):
    """
    Predict trip duration with optional feature engineering
    """
    # Retrieve the trained model from the results dictionary
    model = results[model_type]['model']
    
    if feature_engineering:
        input_data = engineer_features(input_data)
        features = [
            'trip_distance', 'PULocationID', 'DOLocationID',
            'pickup_hour', 'pickup_day', 'pickup_month', 'is_weekend',
            'PU_DO', 'time_of_day'
        ]
        input_data = input_data[features]
    
    return model.predict(input_data)

def log_prediction_plot(y_true, y_pred, model_name, sample_size=1000):
    # Filter data for trips between 0-20 minutes
    mask = (np.array(y_true) <= 20) & (np.array(y_true) >= 0)
    y_true_filtered = np.array(y_true)[mask]
    y_pred_filtered = np.array(y_pred)[mask]
    
    # Sample from filtered data if needed
    if len(y_true_filtered) > sample_size:
        indices = np.random.choice(len(y_true_filtered), sample_size, replace=False)
        y_true_plot = y_true_filtered[indices]
        y_pred_plot = y_pred_filtered[indices]
    else:
        y_true_plot = y_true_filtered
        y_pred_plot = y_pred_filtered
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set(style="whitegrid")
    
    # Scatter plot: predicted vs actual
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true_plot, y=y_pred_plot, ax=ax, color="#A0C4FF", edgecolor="w", s=50)
    
    # Set fixed axis limits for 0-20 minute range
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    
    # Plot identity line
    ax.plot([0, 20], [0, 20], 'k--', lw=2)
    
    # Show sample size information
    ax.text(0.05, 0.95, f'n = {len(y_true_filtered)}', 
            transform=ax.transAxes, fontsize=12, va='top', 
            bbox=dict(boxstyle='round', alpha=0.1))
    
    # Labels and title
    ax.set_xlabel("Actual Duration (minutes)")
    ax.set_ylabel("Predicted Duration (minutes)")
    ax.set_title(f"{model_name} — Predicted vs Actual (0-20 min)")
    
    # Save plot
    plot_path = f"{model_name}_0_20_scatter_plot.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    
    # Log figure as artifact to MLflow
    mlflow.log_artifact(plot_path)
    
    # Clean up plot file
    plt.close(fig)
    os.remove(plot_path)