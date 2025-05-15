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

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    elif 21 <= hour <= 23 or 0 <= hour < 5:
        return 'night'
    else:
        return 'unknown'

from pandas.api.types import CategoricalDtype

def engineer_features(df: pd.DataFrame):
    """
    Create additional features from the raw data with defined categorical levels
    
    Args:
    - df (pd.DataFrame): Input dataframe
    
    Returns:
    - pd.DataFrame: Dataframe with engineered features
    """
    df = df.copy()
    
    # Temporal Features
    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['lpep_pickup_datetime'].dt.day_name()
    df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
    df['is_weekend'] = df['lpep_pickup_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Spatial Features
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    
    # Time of Day
    df['time_of_day'] = df['pickup_hour'].apply(get_time_of_day)

        # Categorical Encoding
    categorical_features = ['PULocationID', 'DOLocationID', 'PU_DO', 'time_of_day', 'pickup_day']
    for feature in categorical_features:
        df[feature] = df[feature].astype('category')
    
    # Define categories explicitly
    # time_of_day_type = CategoricalDtype(categories=['night', 'morning', 'afternoon', 'evening'], ordered=True)

    # # Example: known PULocationIDs, DOLocationIDs, or PU_DO combos
    # # For simplicity assuming string types — you might build this dynamically from training data at train-time
    # pu_location_type = CategoricalDtype(categories=[str(i) for i in range(1, 266)], ordered=False)
    # do_location_type = CategoricalDtype(categories=[str(i) for i in range(1, 266)], ordered=False)
    # pu_do_type = CategoricalDtype(categories=[], ordered=False)  # if you know possible combos, list them here

    # # Cast to defined categorical types
    # df['PULocationID'] = df['PULocationID'].astype(str).astype(pu_location_type)
    # df['DOLocationID'] = df['DOLocationID'].astype(str).astype(do_location_type)
    # df['PU_DO'] = df['PU_DO'].astype(pu_do_type)
    # df['time_of_day'] = df['time_of_day'].astype(time_of_day_type)
    
    return df

# def prepare_dictionaries(df: pd.DataFrame):
#     """
#     Prepare feature dictionaries for models
    
#     Args:
#     - df (pd.DataFrame): Input dataframe
    
#     Returns:
#     - List of dictionaries with minimial feature
#     """
#     df = df.copy()

#      # Remove the target if present
#     df = df.drop(columns=['duration'], errors='ignore')

#     df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
#     categorical = ['PU_DO']
#     numerical = ['trip_distance']
#     dicts = df[categorical + numerical].to_dict(orient='records')
#     return dicts

def prepare_dictionaries(df: pd.DataFrame):
    """
    Prepare feature dictionaries for models
    
    Args:
    - df (pd.DataFrame): Input dataframe
    
    Returns:
    - List of dictionaries with minimal features (PU_DO and trip_distance)
    - Dataframe with PU_DO feature added, keeping all other columns
    """
    df = df.copy()

    # Remove the target if present from the dicts, but keep it in the DataFrame
    dict_df = df.drop(columns=['duration'], errors='ignore')

    # Add PU_DO feature to both
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    dict_df['PU_DO'] = df['PU_DO']

    # Prepare minimal feature dicts
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = dict_df[categorical + numerical].to_dict(orient='records')

    return dicts, df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    dicts, _ = prepare_dictionaries(df)  # ignore updated df if you don't need it here
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