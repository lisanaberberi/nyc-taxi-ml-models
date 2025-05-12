import os
import getpass
import mlflow
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb


def setup_mlflow_tracking(
    tracking_uri="http://localhost:5000", 
    experiment_name="green-taxi-duration-mlmodels-evaluations",
    # MLFLOW_TRACKING_USERNAME=<your-email-address> #if you have setup postgres/mysql as a backend dB
):
    """
    Set up MLflow tracking with configurable server and credentials
    
    Args:
    - tracking_uri (str): MLflow tracking server URI
    - experiment_name (str): Name of the MLflow experiment
    - username (str): Tracking username
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    # Set tracking username and password
    #MLFLOW_TRACKING_PASSWORD = getpass.getpass("Enter MLflow tracking password: ")
    
    #os.environ['MLFLOW_TRACKING_USERNAME'] = username
   # os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    # Set experiment name
    mlflow.set_experiment(experiment_name)

def log_model_metrics(y_true, y_pred, model_name, params=None): 
    """
    Log model metrics to MLflow
    
    Args:
    - y_true (array-like): True target values
    - y_pred (array-like): Predicted target values
    - model_name (str): Name of the model
    - params (dict, optional): Model parameters
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Log parameters if provided
    if params:
        mlflow.log_params(params)
    
    # Log metrics
    mlflow.log_metrics({
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    })
    
    # Print metrics
    print(f"{model_name.upper()} Performance:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape}")


def log_model_parameters(model, model_name):
    """
    Comprehensive parameter logging for different model types
    
    Args:
    - model: Scikit-learn or XGBoost model
    - model_name: Name of the model for context
    """
    # Generic parameters for all models
    try:
        random_state = getattr(model, 'random_state', 'N/A')
        mlflow.log_param(f"{model_name}_random_state", random_state)
    except Exception as e:
        print(f"Could not log random_state for {model_name}: {e}")

    # Model-specific parameter logging
    try:
        if isinstance(model, RandomForestRegressor):
            mlflow.log_params({
                f"{model_name}_n_estimators": model.n_estimators,
                f"{model_name}_max_depth": model.max_depth if model.max_depth is not None else 'None',
                f"{model_name}_min_samples_leaf": model.min_samples_leaf,
                f"{model_name}_max_features": model.max_features if model.max_features is not None else 'None',
                f"{model_name}_bootstrap": model.bootstrap
            })
        
        elif isinstance(model, GradientBoostingRegressor):
            mlflow.log_params({
                f"{model_name}_n_estimators": model.n_estimators,
                f"{model_name}_max_depth": model.max_depth,
                f"{model_name}_learning_rate": model.learning_rate,
                f"{model_name}_min_samples_leaf": model.min_samples_leaf,
                f"{model_name}_max_features": model.max_features if model.max_features is not None else 'None'
            })
        
        elif isinstance(model, xgb.XGBRegressor):
            mlflow.log_params({
                f"{model_name}_n_estimators": model.n_estimators,
                f"{model_name}_max_depth": model.max_depth,
                f"{model_name}_learning_rate": model.learning_rate,
                f"{model_name}_booster": model.booster,
                f"{model_name}_gamma": model.gamma,
                f"{model_name}_subsample": model.subsample,
                f"{model_name}_colsample_bytree": model.colsample_bytree
            })
        
        elif isinstance(model, DecisionTreeRegressor):
            mlflow.log_params({
                f"{model_name}_max_depth": model.max_depth,
                f"{model_name}_min_samples_leaf": model.min_samples_leaf,
                f"{model_name}_min_samples_split": model.min_samples_split,
                f"{model_name}_criterion": model.criterion
            })
        
        else:
            # Log generic model parameters if the specific type is not recognized
            print(f"Generic parameter logging for {model_name}")
            generic_params = {k: v for k, v in model.__dict__.items() if isinstance(v, (int, float, str, bool))}
            mlflow.log_params({f"{model_name}_{k}": v for k, v in generic_params.items()})
    
    except Exception as e:
        print(f"Error logging parameters for {model_name}: {e}")