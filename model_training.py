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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from utils import log_model_metrics
from data_preprocessing import engineer_features, prepare_dictionaries

from mlflow.models import infer_signature

from utils import log_model_parameters

import matplotlib.pyplot as plt
import os
import seaborn as sns

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    # train_dicts = prepare_dictionaries(train_data)
    # test_dicts = prepare_dictionaries(test_data)

    train_dicts, train_data = prepare_dictionaries(train_data)
    test_dicts, test_data = prepare_dictionaries(test_data)



    # Vectorize
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    X_test = dv.transform(test_dicts)
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=300,          # More trees to stabilize variance
            max_depth=25,              # Slightly deeper since small-ish dataset
            min_samples_leaf=5,        # Lower to allow a bit more granularity
            max_features='sqrt',       # Classic best practice for RF
            bootstrap=True,
            n_jobs=-1,                 # Parallelize
            random_state=42
        ),
        'lightgbm': LGBMRegressor(
            n_estimators=500,          # Boosting benefits from more iterations
            max_depth=8,               # Slightly deeper trees (as data is small)
            learning_rate=0.05,        # Lower LR with more trees for stability
            min_child_samples=10,      # Less conservative
            subsample=0.9,             # Use more data per boosting round
            colsample_bytree=0.8,
            random_state=42
        ),
        'decision_tree': DecisionTreeRegressor(
            max_depth=18,              # Can afford deeper single trees
            min_samples_leaf=5,        # Finer granularity
            random_state=42
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=300,          # More boosting rounds
            max_depth=6,               # Slightly deeper to reduce bias
            learning_rate=0.05,        # Lower LR for smoother learning
            min_samples_leaf=5,
            subsample=0.9,             # Slightly higher subsampling
            random_state=42
        )
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



def train_custom_model(train_data, test_data, target='duration', model_type='random_forest'):
    """
    Train a specific type of custom model on engineered features using MLflow tracking.
    """

    y_train = train_data[target].values
    y_test = test_data[target].values

    # Model configs
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=100,          # More trees to stabilize variance
            max_depth=20,              # Slightly deeper since small-ish dataset
            min_samples_leaf=10,        # Lower to allow a bit more granularity
            #max_features='sqrt',       # Classic best practice for RF
            bootstrap=True,
            n_jobs=-1,                 # Parallelize
            random_state=42
        ),
        'lightgbm': LGBMRegressor(
            n_estimators=500,          # Boosting benefits from more iterations
            max_depth=8,               # Slightly deeper trees (as data is small)
            learning_rate=0.05,        # Lower LR with more trees for stability
            min_child_samples=10,      # Less conservative
            subsample=0.9,             # Use more data per boosting round
            colsample_bytree=0.8,
            random_state=42
        ),
        'decision_tree': DecisionTreeRegressor(
            max_depth=18,              # Can afford deeper single trees
            min_samples_leaf=5,        # Finer granularity
            random_state=42
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=300,          # More boosting rounds
            max_depth=6,               # Slightly deeper to reduce bias
            learning_rate=0.05,        # Lower LR for smoother learning
            min_samples_leaf=5,
            subsample=0.9,             # Slightly higher subsampling
            random_state=42
        )
    }
    model = models[model_type]
    run_name = f"{model_type}_custom_model_{run_datetime}"

    # Define your feature lists
    #categorical_columns = ['PULocationID', 'DOLocationID', 'time_of_day', 'pickup_day']
    numerical_columns = ['trip_distance', 'pickup_hour']

    categorical_columns = ['PULocationID', 'DOLocationID', 'PU_DO_grouped', 
                           'time_of_day', 'pickup_day']

    # Prepare tags dictionary
    tags = {f"categorical_{col}": col for col in categorical_columns}
    tags.update({f"numerical_{col}": col for col in numerical_columns})

    results = {}

    with mlflow.start_run(nested=True, run_name=run_name) as run:
        mlflow.set_tags(tags)

        # Column Transformer for preprocessing
        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         ('num', Pipeline(steps=[
        #             ('imputer', SimpleImputer(strategy='mean')),
        #             ('scaler', StandardScaler())
        #         ]), numerical_columns),

        #         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        #     ]
        # )

        # Improved preprocessing with better handling of categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Median is more robust to outliers
                    ('scaler', StandardScaler())
                ]), numerical_columns),
                
                # Use OneHotEncoder with handle_unknown='ignore' for robustness
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categoricals
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_columns)
            ],
            remainder='drop'  # Drop columns not specified in transformers
        )

        # Complete pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Train model
        pipeline.fit(train_data, y_train)
        preds = pipeline.predict(test_data)

        # Log model params & metrics
        log_model_parameters(model, f"{model_type}_custom")
        log_model_metrics(y_test, preds, f"{model_type}_custom")

        # Log training data as input artifact
        mlflow.log_input(mlflow.data.from_pandas(train_data), context="training")

        # Create MLflow model signature
        signature = infer_signature(train_data, preds[:5])
        input_example = train_data.head(5)

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
        input_data = engineer_features_improved(input_data)
        features = [
            'PULocationID', 'DOLocationID', 'PU_DO_grouped', 
            'time_of_day', 'pickup_day','pickup_month', 'is_weekend', 'pickup_hour'

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



def engineer_features_improved(df: pd.DataFrame):
    """
    Enhanced feature engineering with more sophisticated features
    """
    df = df.copy()
    
    # Basic temporal features
    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['lpep_pickup_datetime'].dt.day_name()
    df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
    df['is_weekend'] = df['lpep_pickup_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # More sophisticated time features
    df['pickup_dayofweek'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['dropoff_hour'] = df['lpep_dropoff_datetime'].dt.hour
    df['hour_diff'] = (df['dropoff_hour'] - df['pickup_hour']) % 24
    
    # Time of day as both category and cyclical features
    df['time_of_day'] = df['pickup_hour'].apply(get_time_of_day)
    
    # Cyclical encoding of time (hours)
    df['pickup_hour_sin'] = np.sin(2 * np.pi * df['pickup_hour']/24)
    df['pickup_hour_cos'] = np.cos(2 * np.pi * df['pickup_hour']/24)
    
    # Location features
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    
    # Calculate pickup-dropoff frequency
    pu_do_counts = df['PU_DO'].value_counts()
    # Only keep the top N most frequent pairs, replace others with 'other'
    top_n = 50  # Adjust based on your data
    frequent_pairs = pu_do_counts.nlargest(top_n).index
    df['PU_DO_grouped'] = df['PU_DO'].apply(lambda x: x if x in frequent_pairs else 'other')
    
    # Create individual PU and DO frequency features
    pu_counts = df['PULocationID'].value_counts()
    do_counts = df['DOLocationID'].value_counts()
    df['PU_frequency'] = df['PULocationID'].map(pu_counts)
    df['DO_frequency'] = df['DOLocationID'].map(do_counts)

    
    # Convert categorical features to appropriate type
    # categorical_features = ['PULocationID', 'DOLocationID', 'PU_DO_grouped', 
    #                        'time_of_day', 'pickup_day']
    categorical_features = ['PULocationID', 'DOLocationID', 'PU_DO', 
                           'time_of_day', 'pickup_day']
    for feature in categorical_features:
        if feature in df.columns:
            df[feature] = df[feature].astype('category')
    
    return df

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

def evaluate_features(train_data, test_data, target='duration'):
    """
    Evaluate features individually to determine their predictive power
    """
    y_train = train_data[target].values
    y_test = test_data[target].values
    
    # Get all potential features
    all_features = [col for col in train_data.columns 
                   if col != target and col not in ['lpep_pickup_datetime', 'lpep_dropoff_datetime']]
                   
    feature_importance = {}
    
    # Simple model to test each feature
    for feature in all_features:
        try:
            X_train = train_data[[feature]].copy()
            X_test = test_data[[feature]].copy()
            
            # Handle categorical features
            if train_data[feature].dtype.name == 'category' or train_data[feature].dtype == 'object':
                # For categorical features, use OneHotEncoder
                encoder = OneHotEncoder(handle_unknown='ignore')
                X_train_encoded = encoder.fit_transform(X_train)
                X_test_encoded = encoder.transform(X_test)
                
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X_train_encoded, y_train)
                y_pred = model.predict(X_test_encoded)
            else:
                # For numerical features
                # Handle potential NaNs
                imputer = SimpleImputer(strategy='mean')
                X_train_imputed = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test)
                
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X_train_imputed, y_train)
                y_pred = model.predict(X_test_imputed)
            
            # Calculate R² score
            r2 = r2_score(y_test, y_pred)
            feature_importance[feature] = r2
            
        except Exception as e:
            logger.warning(f"Error evaluating feature {feature}: {str(e)}")
            feature_importance[feature] = 0
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_features
