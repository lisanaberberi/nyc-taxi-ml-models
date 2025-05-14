import os
import pickle
import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

def dump_pickle(obj, filename: str):
    """
    Dump an object to a pickle file
    """
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def read_dataframe(filename: str):
    """
    Read and preprocess a single parquet file
    """
    df = pd.read_parquet(filename)
    print("Original dataset info: ", df.info())
    print("Original data prior to transformation: ", df.head(10))
    print("Original data columns: ", list(df.columns))
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(df.head(10))
    return df

def load_multiple_dataframes(raw_data_path: str, dataset: str, months: list):
    """
    Load and concatenate multiple parquet files for a given dataset and months
    
    Args:
    - raw_data_path: Base path to data files
    - dataset: Type of taxi dataset (e.g., 'green')
    - months: List of months to load
    
    Returns:
    - Concatenated DataFrame
    """
    dataframes = []
    for month in months:
        filename = os.path.join(raw_data_path, f"{dataset}_tripdata_{month}.parquet")
        if os.path.exists(filename):
            df = read_dataframe(filename)
            dataframes.append(df)
        else:
            print(f"Warning: File {filename} not found. Skipping.")
    
    if not dataframes:
        raise ValueError("No valid dataframes found. Check file paths and month formats.")
    
    return pd.concat(dataframes, ignore_index=True)

def engineer_features(df: pd.DataFrame):
    """
    Create additional features from the raw data
    
    Args:
    - df (pd.DataFrame): Input dataframe
    
    Returns:
    - pd.DataFrame: Dataframe with engineered features
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Temporal Features
    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['lpep_pickup_datetime'].dt.day_name()
    df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
    df['is_weekend'] = df['lpep_pickup_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Spatial Features
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    
    # Time-based Features
    df['time_of_day'] = pd.cut(
        df['pickup_hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    # Categorical Encoding
    categorical_features = ['PULocationID', 'DOLocationID', 'PU_DO', 'time_of_day']
    for feature in categorical_features:
        df[feature] = df[feature].astype('category')
    
    return df

def prepare_dictionaries(df: pd.DataFrame):
    """
    Prepare feature dictionaries for models
    
    Args:
    - df (pd.DataFrame): Input dataframe
    
    Returns:
    - List of dictionaries with minimial feature
    """
    df = df.copy()

     # Remove the target if present
    df = df.drop(columns=['duration'], errors='ignore')

    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    """
    Preprocess the dataframe using DictVectorizer
    
    Args:
    - df (pd.DataFrame): Input dataframe
    - dv (DictVectorizer): The vectorizer
    - fit_dv (bool): Whether to fit the vectorizer or just transform
    - features (list): Features to include in preprocessing
    
    Returns:
    - Transformed feature matrix
    - The DictVectorizer
    """
    dicts = prepare_dictionaries(df)
    
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv


# def preprocess_custom_features(df):
#     categorical_features = ['PU_DO', 'pickup_day', 'pickup_month' ,'time_of_day']
#     numerical_features = ['trip_distance']
#     all_features = categorical_features + numerical_features
#     dicts = df[all_features].to_dict(orient='records')
#     return dicts


@click.command()
@click.option(
    "--raw_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to directory containing raw parquet files"
)
@click.option(
    "--dest_path",
    required=True,
    type=click.Path(),
    help="Path to save processed data"
)
@click.option(
    "--dataset",
    default="green",
    type=click.Choice(['green', 'yellow'], case_sensitive=False),
    help="Type of taxi dataset"
)
def run_data_prep_cli(raw_data_path: str, dest_path: str, dataset: str) -> None:
    """Click command wrapper for data preparation."""
    run_data_prep(raw_data_path, dest_path, dataset)


def run_data_prep(raw_data_path: str, dest_path: str, dataset: str):
    """
    Prepare data for machine learning model
    
    Loads data for multiple months and preprocesses it
    """
    # Define months to load
    months = [
        '2024-07', 
        '2024-08', 
        '2024-09', 
        '2024-10', 
        '2024-11', 
        '2024-12'
    ]
    
    # Load all data first
    df = load_multiple_dataframes(raw_data_path, dataset, months)
    
    # Split into 80% train and 20% test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Extract the target (duration)
    target = 'duration'
    y_train = df_train[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

    # Print some information about the processed data
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Train size: {len(df_train)} samples ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Test size: {len(df_test)} samples ({len(df_test)/len(df)*100:.1f}%)")
    
    return df_train, df_test

if __name__ == '__main__':
    run_data_prep()