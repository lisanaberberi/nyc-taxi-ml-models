import os
import pickle
import click
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

import geopandas as gpd



def dump_pickle(obj, filename):
    """
    Save object as pickle file
    """
    with open(filename, "wb") as f:
        return pickle.dump(obj, f)

def read_dataframe(filename: str):
    """
    Read and preprocess a single parquet file
    """
    df = pd.read_parquet(filename)
    print(f"Reading {filename}...")
    
    # Calculate trip duration
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert location IDs to string for proper merging
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Reading {filename}...")
    print(df.columns)
    
    return df

def read_taxi_zones(shapefile_path):
    """
    Reads taxi zone shapefile and converts it to a pandas DataFrame
    with the necessary columns for merging
    """
    print(f"Reading taxi zones from {shapefile_path}...")
    
    # Read the shapefile using geopandas
    gdf = gpd.read_file(shapefile_path)
    
    # Convert to regular pandas DataFrame with needed columns
    taxizone_df = gdf[['LocationID', 'borough', 'zone', 'Shape_Area']].copy()
    
    # Rename columns to match expected format
    taxizone_df.rename(columns={
        'borough': 'Borough',
        'zone': 'Zone', 
        'Shape_Area': 'Shape_Area'
    }, inplace=True)
    
    # Ensure LocationID is a string to match PULocationID and DOLocationID
    taxizone_df['LocationID'] = taxizone_df['LocationID'].astype(str)
    
    return taxizone_df

def merge_with_zones(df, taxizone_df):
    """
    Merge dataframe with taxi zones information for both pickup and dropoff locations
    """
    print("Merging with taxi zones data...")
    
    # Pickup Location Zone Merge
    df = df.merge(taxizone_df, how='left', left_on='PULocationID', right_on='LocationID')
    df.drop(columns=['LocationID'], axis=1, inplace=True)
    df.rename(columns={
        'Borough': 'pickup_borough', 
        'Zone': 'pickup_zone', 
        'Shape_Area': 'pickup_ShapeArea'
    }, inplace=True)
    
    # Dropoff Location Zone Merge
    df = df.merge(taxizone_df, how='left', left_on='DOLocationID', right_on='LocationID')
    df.drop(columns=['LocationID'], axis=1, inplace=True)
    df.rename(columns={
        'Borough': 'drop_borough', 
        'Zone': 'drop_zone', 
        'Shape_Area': 'dropoff_ShapeArea'
    }, inplace=True)

    print("After merging with taxi_zones ...")
    print(df.columns)
    
    return df

def load_multiple_dataframes(path, dataset, months):
    """
    Load and combine multiple monthly parquet files
    """
    dataframes = []
    
    for month in months:
        file = f"{dataset}_tripdata_{month}.parquet"
        file_path = os.path.join(path, file)
        
        if os.path.exists(file_path):
            df = read_dataframe(file_path)
            dataframes.append(df)
        else:
            print(f"Warning: File {file_path} not found, skipping.")
    
    if not dataframes:
        raise ValueError("No valid parquet files found!")
    
    return pd.concat(dataframes)

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'


def feature_eng(df: pd.DataFrame):
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

def basic_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform minimal baseline feature :
    - Keep essential columns for basic model.
    - Remove problematic columns from the full dataframe.
    - Duration is assumed to be pre-calculated in read_dataframe().
    - Missing basic columns are added with NaN values.

    Args:
        df (pd.DataFrame): Input dataframe with raw data.

    Returns:
        tuple:
            - pd.DataFrame: Dataframe with selected basic features.
            - pd.DataFrame: Cleaned dataframe with problematic columns removed.
    """
    df = df.copy()

    # Location features
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)

    # Define essential columns for the baseline model
    basic_cols = ['PULocationID', 'DOLocationID', 'trip_distance', 'duration']
    #basic_cols = ['PU_DO', 'trip_distance', 'duration']
    #after  merge with taxi_zones
    #basic_cols = ['PULocationID', 'DOLocationID', 'pickup_borough ', 'drop_borough','fare_amount','total_amount','trip_distance', 'duration']

    # Check for missing columns and add them with NaN values
    missing_cols = [col for col in basic_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in basic_features: {missing_cols}")
        for col in missing_cols:
            df[col] = np.nan

    df_basic = df[basic_cols]

    # Clean data - remove problematic columns
    columns_to_drop = [
        'store_and_fwd_flag', 'ehail_fee', 'trip_type', 
        'payment_type', 'passenger_count'
    ]
    # Only drop columns that exist
    available_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=available_to_drop)

    return df_basic, df_cleaned



def preprocess(df, dv, fit_dv=False):
    """
    Preprocess dataframe for model training
    """
    df = df.copy()

    # Drop columns not used as features - check if they exist first
    columns_to_drop = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'duration']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)

    # Convert to dict records
    df_dict = df.to_dict(orient='records')
    
    # Create feature matrix
    if fit_dv:
        X = dv.fit_transform(df_dict)
    else:
        X = dv.transform(df_dict)
    
    return X, dv

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
@click.option(
    "--taxi_zones_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to taxi_zones.shp file (optional)"
)

@click.option("--feature_type", 
    default="feature", 
    type=click.Choice(['feature', 'basic']), 
    help="Type of feature engineering to apply"
)

def run_data_prep_cli(raw_data_path: str, dest_path: str, dataset: str, taxi_zones_path: str, feature_type: str) -> None:
    """Click command wrapper for data preparation."""
    run_data_prep(raw_data_path, dest_path, dataset, taxi_zones_path, feature_type)


def run_data_prep(raw_data_path: str, dest_path: str, dataset: str, taxi_zones_path: str = None, feature_type='feature'):
    """
    Prepare data for machine learning model
    
    Loads data for multiple months and preprocesses it
    """
    import pickle
    
    # Define months to load
    months = [
        # '2024-01', 
        # '2024-02', 
        # '2024-03', 
        # '2024-04', 
        # '2024-05', 
        # '2024-06',
        '2024-07', 
        '2024-08', 
        '2024-09', 
        '2024-10', 
        '2024-11', 
        '2024-12'
    ]
    
    # Load all data first
    df = load_multiple_dataframes(raw_data_path, dataset, months)
    print(f"Loaded data: {len(df)} records")
    print(f"Number of columns from raw data: {len(df.columns)}")
    
    # Add taxi zone information if provided
    if taxi_zones_path:
        taxizone_df = read_taxi_zones(taxi_zones_path)
        df = merge_with_zones(df, taxizone_df)
        print(f"Added taxi zone information. Data shape after merge: {df.shape}")
        print(f"Loaded data after merge: {len(df)} records")
        print(f"Number of columns after merge: {len(df.columns)}")

    
    # Apply selected feature engineering
    if feature_type == 'feature':
        df = feature_eng(df)
    elif feature_type == 'basic':
        df_basic, df_cleaned = basic_features(df)
        # Keep cleaned full dataframe for splitting and training
        df = df_cleaned
    else:
        raise ValueError(f"Unknown feature_type '{feature_type}'. Must be 'feature' or 'basic'.")


    # Split into 80% train and 20% test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Extract the target (duration)
    target = 'duration'
    if target not in df_train.columns:
        print(f"ERROR: Target column '{target}' not found!")
        print(f"Available columns: {df_train.columns.tolist()}")
        raise ValueError(f"Target column '{target}' not found in dataframe")
    
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
    run_data_prep_cli()