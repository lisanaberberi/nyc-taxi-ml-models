import os
import getpass
import mlflow
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)

def setup_mlflow_tracking(
    tracking_uri="http://localhost:5000", 
    experiment_name="green-taxi-duration-mlmodels-evaluations",
   # username='lisana.berberi@gmail.com'
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
    rmse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Log parameters if provided
    if params:
        mlflow.log_params(params)
    
    # Log metrics
    mlflow.log_metrics({
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    })
    
    # Print metrics
    print(f"{model_name.upper()} Performance:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape}")