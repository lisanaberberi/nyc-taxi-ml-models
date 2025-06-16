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
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from utils import log_model_metrics
from data_preprocessing import feature_eng

from mlflow.models import infer_signature
from utils import log_model_parameters

import matplotlib.pyplot as plt
import os
import seaborn as sns

import logging
import optuna
from optuna.integration.mlflow import MLflowCallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate run datetime
run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_model_config():
    return {
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        ),
        'lightgbm': LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        'decision_tree': DecisionTreeRegressor(
            max_depth=15,
            min_samples_leaf=10,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
    }


def train_models(train_data, test_data, target='duration'):
    """
    Train baseline models with only basic features using MLflow tracking.
    Optimized for faster execution.
    """
    # Baseline: minimal features
    y_train = train_data[target].values
    y_test = test_data[target].values

    # Columns to drop that cause issues for DictVectorizer (timestamps)
    columns_to_drop = ['lpep_pickup_datetime', 'lpep_dropoff_datetime']

    # Drop target and datetime columns
    train_df = train_data.drop(columns=[target] + columns_to_drop, errors='ignore')
    test_df = test_data.drop(columns=[target] + columns_to_drop, errors='ignore')

    # Fill missing values (example: with 0)
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # Convert to dicts excluding target
    train_dicts = train_df.to_dict(orient='records')
    test_dicts = test_df.to_dict(orient='records')

    print(f"Train dicts length: {len(train_dicts)}")
    print(f"Test dicts length: {len(test_dicts)}")

    if len(train_dicts) == 0 or len(test_dicts) == 0:
        raise ValueError("Training or testing data is empty after processing.")

    # Vectorize once for all models
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)

    models = get_model_config()

    # store the best metric value  of r2 among the models
    best_r2 = -float('inf')
    best_model_name = None

    # Use single parent run instead of nested runs for better performance
    with mlflow.start_run(run_name=f"baseline_models_{run_datetime}") as parent_run:
        results = {}
        
        # Log dataset info once at parent level (smaller sample for efficiency)
        train_data = train_data.drop(columns='duration')
        sample_data = train_data.sample(n=min(1000, len(train_data)), random_state=42)
        mlflow.log_input(
            mlflow.data.from_pandas(sample_data, source="training_sample"), 
            context="training"
        )

        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            # Train model
            pipeline = make_pipeline(dv, model)
            pipeline.fit(train_dicts, y_train)
            preds = pipeline.predict(test_dicts)

            # Log metrics with model prefix
            # metrics = {
            #     f"{model_name}_mae": mean_absolute_error(y_test, preds),
            #     f"{model_name}_mse": mean_squared_error(y_test, preds),
            #     f"{model_name}_rmse": np.sqrt(mean_squared_error(y_test, preds)),
            #     f"{model_name}_r2": r2_score(y_test, preds)
            # }
            # mlflow.log_metrics(metrics)
            log_model_parameters(model, f"baseline_{model_name}")
            log_model_metrics(y_test, preds, f"baseline_{model_name}")

            # Plot the predictions against the ground_truth
            log_prediction_plot(y_test, preds, f"baseline_{model_name}")
            
            # Compute R² to track best model            
            current_r2 = r2_score(y_test, preds)
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_model_name = model_name

           # # Log model parameters with prefix
            # model_params = model.get_params()
            # prefixed_params = {f"{model_name}_{k}": v for k, v in model_params.items()}
            # mlflow.log_params(prefixed_params)

            # Create signature and log model
            signature = infer_signature(X_train, preds[:5])
            
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=f"baseline_{model_name}_model",
                signature=signature
            )

            # Log prediction plot (optimized)
            log_prediction_plot(y_test, preds, f"baseline_{model_name}", sample_size=500)
            
            # Log best R² and best model at the end of training
            mlflow.log_metric("best_r2", best_r2)
            mlflow.set_tag("best_model", best_model_name)


            results[model_name] = {'model': pipeline, 'y_pred': preds}

        return results

def train_custom_model(train_data, test_data, target='duration'):
    """
    Train custom models with engineered features using MLflow tracking.
    Optimized for performance and consistent with baseline train_models style.
    """
    y_train = train_data[target].values
    y_test = test_data[target].values

    #numerical_columns = ['fare_amount', 'total_amount', 'trip_distance']
    numerical_columns = ['fare_amount', 'total_amount',]
    # categorical_columns = ['PULocationID', 'DOLocationID', 
    #                       'time_of_day', 'pickup_day', 'PU_DO']
    categorical_columns = ['PULocationID', 'time_of_day', 'pickup_day']

    columns_taxi_zone = ['pickup_borough', 'drop_zone', 'drop_borough']
    columns_to_keep = numerical_columns + categorical_columns + columns_taxi_zone

    # Filter columns for train/test
    train_df = train_data[columns_to_keep].copy()
    test_df = test_data[columns_to_keep].copy()

    # Preprocessing pipeline (numerical + categorical)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_columns),

            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='unknown')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_columns)
        ],
        remainder='drop'
    )

    models = get_model_config()

    best_r2 = -float('inf')
    best_model_name = None
    results = {}

    with mlflow.start_run(run_name=f"custom_models_{run_datetime}") as parent_run:
        # Log sample of training data (without target)
        sample_data = train_data[columns_to_keep].sample(n=min(1000, len(train_data)), random_state=42)
        mlflow.log_input(mlflow.data.from_pandas(sample_data, source="training_sample"), context="training")

        # Log general tags
        mlflow.set_tags({
            "feature_engineering": "custom",
            "numerical_features": ",".join(numerical_columns),
            "categorical_features": ",".join(categorical_columns),
        })

        for model_name, model in models.items():
            logger.info(f"Training {model_name} with custom features...")

            # Create pipeline with preprocessor and model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Train
            pipeline.fit(train_df, y_train)
            preds = pipeline.predict(test_df)

            # Log params, metrics, plots with prefix
            log_model_parameters(model, f"custom_{model_name}")
            log_model_metrics(y_test, preds, f"custom_{model_name}")
            log_prediction_plot(y_test, preds, f"custom_{model_name}")

            # Check best R²
            current_r2 = r2_score(y_test, preds)
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_model_name = model_name

            # Signature and model logging
            signature = infer_signature(train_df, preds[:5])
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=f"custom_{model_name}_model",
                signature=signature
            )

            results[model_name] = {'model': pipeline, 'y_pred': preds}
            # After training all models, log best model info once 
            mlflow.log_metric("best_r2", best_r2)
            mlflow.set_tag("best_model", best_model_name)

    return results


def predict_trip_duration(results, model_type, input_data, feature_engineering=False):
    model_info = results[model_type]
    
    # Support both dict with 'model' key or direct pipeline
    if isinstance(model_info, dict) and 'model' in model_info:
        model = model_info['model']
    else:
        model = model_info  # assume it's the pipeline itself
    
    if feature_engineering:
        input_data = feature_eng(input_data)
        features = [
            'PULocationID', 'DOLocationID', 'PU_DO_grouped', 
            'time_of_day', 'pickup_day','pickup_month', 'pickup_hour',
            'fare_amount', 'total_amount',  'trip_distance', 'PU_DO'
        ]
        available_features = [f for f in features if f in input_data.columns]
        input_data = input_data[available_features]
    
    return model.predict(input_data)

def log_prediction_plot(y_true, y_pred, model_name, sample_size=10000):
    """
    Optimized prediction plotting function
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate R² on full data first
    r2_full = r2_score(y_true, y_pred)
    
    # Filter data for trips between 0-20 minutes AFTER scoring
    mask = (y_true >= 0) & (y_true <= 20)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Sample from filtered data if needed
    if len(y_true_filtered) > sample_size:
        indices = np.random.choice(len(y_true_filtered), sample_size, replace=False)
        y_true_plot = y_true_filtered[indices]
        y_pred_plot = y_pred_filtered[indices]
    else:
        y_true_plot = y_true_filtered
        y_pred_plot = y_pred_filtered
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.6, s=20, color='#A0C4FF', edgecolors='white', linewidth=0.5)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.plot([0, 20], [0, 20], 'k--', lw=2, alpha=0.8)
    ax.set_xlabel("Actual Duration (min)", fontsize=12)
    ax.set_ylabel("Predicted Duration (min)", fontsize=12)
    ax.set_title(f"{model_name} — Predicted vs Actual", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add R² on full data to plot
    ax.text(0.05, 0.95, f'R² (full) = {r2_full:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save and log
    plot_path = f"{model_name}_prediction_plot.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    plt.close(fig)
    if os.path.exists(plot_path):
        os.remove(plot_path)


def evaluate_features_fast(train_data, test_data, target='duration'):
    """
    Fast feature evaluation using correlation and single model approach
    """
    logger.info("Starting fast feature evaluation...")
    
    # Get all potential features
    all_features = [col for col in train_data.columns 
                   if col != target and col not in ['lpep_pickup_datetime', 'lpep_dropoff_datetime']]
    
    feature_scores = {}
    
    # Calculate correlation for numerical features
    numerical_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
    if target in numerical_features:
        numerical_features.remove(target)
    
    for feature in numerical_features:
        if feature in all_features:
            corr = abs(train_data[feature].corr(train_data[target]))
            feature_scores[feature] = corr
    
    # For categorical features, use simple variance analysis
    categorical_features = [f for f in all_features if f not in numerical_features]
    
    for feature in categorical_features:
        try:
            # Calculate coefficient of variation within groups
            grouped_stats = train_data.groupby(feature)[target].agg(['mean', 'std'])
            cv_mean = grouped_stats['std'].mean() / grouped_stats['mean'].mean()
            feature_scores[feature] = cv_mean if not np.isnan(cv_mean) else 0
        except:
            feature_scores[feature] = 0
    
    # Sort features by importance
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("Feature evaluation completed")
    for feature, score in sorted_features[:10]:
        logger.info(f"{feature}: {score:.4f}")
    
    return sorted_features

