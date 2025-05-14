import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import click
from data_preprocessing import load_multiple_dataframes

from data_preprocessing import engineer_features


import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid")
color = '#A0C4FF'
#sns.set_palette("viridis")

#warm_palette = ['#F4A261', '#E76F51', '#E9C46A', '#2A9D8F', '#264653']
#sns.set_palette(warm_palette)

def load_nyc_taxi_zones():
    """
    Load NYC taxi zone shapefile
    If not available, provide instructions on how to download it
    """
    try:
        # Try to load the shapefile
        nyc_zones = gpd.read_file('./data/taxi_zones/taxi_zones.shp')
        return nyc_zones
    except Exception as e:
        print(f"Error loading taxi zones shapefile: {e}")
        print("To load geographic data, please download the NYC Taxi Zones shapefile from:")
        print("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip")
        print("and save it to ./data/taxi_zones/")
        return None


def plot_distribution(df, column, title, xlabel, ylabel='Frequency', bins=30, figsize=(10, 6), xlim=None):
    """Plot distribution of a numerical column with optional x-axis limits"""
    plt.figure(figsize=figsize)
    
    # Plot histogram with KDE
    ax = sns.histplot(data=df, x=column, kde=True, bins=bins)
    
    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim)
    
    # For trip_distance, set a reasonable limit (most NYC trips < 15 miles)
    if column == 'trip_distance':
        plt.xlim(0, 15)  # Limit x-axis to 0-15 miles
    
    # Add basic statistics as text
    # stats_text = (
    #     f"Mean: {df[column].mean():.2f}\n"
    #     f"Median: {df[column].median():.2f}\n"
    #     f"Std Dev: {df[column].std():.2f}\n"
    #     f"Min: {df[column].min():.2f}\n"
    #     f"Max: {df[column].max():.2f}"
    # )
    
    # # Place text in the upper right corner
    # plt.text(0.95, 0.95, stats_text,
    #          transform=ax.transAxes,
    #          verticalalignment='top',
    #          horizontalalignment='right',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"./plots/{column}_distribution.png", dpi=300)
    plt.close()


def plot_time_series(df, time_col, value_col, title, xlabel, ylabel, figsize=(12, 6)):
    """Plot time series data"""
    plt.figure(figsize=figsize)
    
    # Group by time column and get mean value
    time_data = df.groupby(time_col)[value_col].mean().reset_index()
    
    # Plot as column bar chart instead of line
    sns.barplot(data=time_data, x=time_col, y=value_col)
    
    # Set labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"./plots/{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

def plot_heatmap(df, x_col, y_col, values_col, title, figsize=(14, 8), vmax=None):
    """Plot a heatmap with improved background and controlled color scale"""
    # Create figure with white background
    plt.figure(figsize=figsize, facecolor='white')
    
    # Create pivot table
    pivot_data = df.pivot_table(index=y_col, columns=x_col, values=values_col, aggfunc='mean')
    
    # For trip distance values, set a reasonable scale
    if 'distance' in values_col.lower():
        # Set vmax to a reasonable maximum for taxi trips
        vmax = 10  # Most NYC taxi trips are under 10 miles
    
    # Plot heatmap with white background
    ax = sns.heatmap(
        pivot_data, 
        cmap='viridis', 
        annot=False, 
        fmt='.2f', 
        cbar_kws={'label': values_col},
        linewidths=0.5,
        linecolor='white',
        vmin=0,  # Start at 0
        vmax=vmax,  # Use provided vmax or None for auto-scaling
        robust=True  # Use robust color scaling
    )
    
    # Set the figure and axes background to white
    ax.set_facecolor('white')
    
    # Set labels
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Save the plot with white background
    plt.savefig(f"./plots/{title.lower().replace(' ', '_')}.png", dpi=300, facecolor='white')
    plt.close()

def plot_geo_heatmap(df, nyc_zones, column='trip_count', title='Trip Density'):
    """Plot geographic heatmap of taxi trips with zone names"""
    if nyc_zones is None:
        print("Skipping geographic visualization - no shapefile available")
        return
    
    # Create a copy of the zones
    gdf = nyc_zones.copy()
    
    # Count trips by zone
    if 'PULocationID' in df.columns:
        location_counts = df['PULocationID'].value_counts().reset_index()
        location_counts.columns = ['LocationID', column]
        
        # Convert to numeric if necessary
        location_counts['LocationID'] = pd.to_numeric(location_counts['LocationID'], errors='coerce')
        
        # Merge with GeoDataFrame
        gdf = gdf.merge(location_counts, left_on='LocationID', right_on='LocationID', how='left')
        gdf[column] = gdf[column].fillna(0)
        
        # Create plot with white background
        fig, ax = plt.subplots(1, 1, figsize=(15, 15), facecolor='white')
        ax.set_facecolor('white')
        
        # Plot zones colored by trip count
        gdf.plot(column=column, 
                 cmap='viridis', 
                 linewidth=0.5, 
                 ax=ax, 
                 edgecolor='0.8',
                 legend=True,
                 legend_kwds={
                     'shrink': 0.8,  # Adjust the size of the colorbar
                     'label': column,
                     'orientation': 'vertical'
                    })
                    
        # Add background map
        ctx.add_basemap(ax, crs=gdf.crs.to_string())
        
        # Add title
        ax.set_title(title, fontsize=16)
        
        # Add zone names for top zones
        # Find top zones by trip count
        # top_n = 15  # Adjust this number as needed
        # if gdf[column].max() > 0:  # Only if we have valid counts
        #     top_zones = gdf.sort_values(by=column, ascending=False).head(top_n)
            
        #     # Check if 'zone' column exists in the data
        #     label_column = 'zone' if 'zone' in gdf.columns else 'Zone'
        #     if label_column not in gdf.columns:
        #         label_column = 'borough' if 'borough' in gdf.columns else 'Borough'
        #         if label_column not in gdf.columns:
        #             print("No zone or borough name column found in shapefile")
        #             label_column = None
            
        #     # Add labels for top zones if we have a label column
        #     if label_column is not None:
        #         # Get centroid for each zone polygon
        #         top_zones['centroid'] = top_zones.geometry.centroid
                
        #         # Add text labels for top zones
        #         for idx, row in top_zones.iterrows():
        #             plt.text(
        #                 row.centroid.x, 
        #                 row.centroid.y,
        #                 row[label_column],
        #                 fontsize=8,
        #                 ha='center',
        #                 color='black',
        #                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
        #             )
        
        # Remove axis
        ax.set_axis_off()
        
        # Save the plot
        plt.savefig(f"./plots/geo_{column}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        print("PULocationID column not found in dataframe")

def plot_daily_avg_trips_by_vendor(df):
    """Plot average number of trips per day by VendorID"""
    plt.figure(figsize=(12, 6))

    df = df.copy()

    # Create a 'pickup_date' column
    df['pickup_date'] = df['lpep_pickup_datetime'].dt.date

    # Group by date and VendorID and count trips
    daily_counts = df.groupby(['pickup_date', 'VendorID']).size().reset_index(name='trip_count')

    # Then, compute average trips per day for each vendor
    avg_daily_trips = daily_counts.groupby('VendorID')['trip_count'].mean().reset_index()

    # Plot as barplot
    sns.barplot(data=avg_daily_trips, x='VendorID', y='trip_count', color=color, width=0.6)

    plt.title('Average Number of Trips per Day by VendorID', fontsize=16)
    plt.xlabel('VendorID', fontsize=12)
    plt.ylabel('Average Number of Trips per Day', fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig("./plots/average_daily_trips_by_vendorid.png", dpi=300)
    plt.close()

def plot_top_location_pairs_avg_trips(df, top_n=20, title='Top Location Pairs by Average Daily Trips', figsize=(14, 8)):
    """
    Plot a barplot of the top N pickup-dropoff location pairs by average number of trips per day
    using a viridis color gradient based on the trip count values
    """
    
    df = df.copy()
    if 'PULocationID' not in df.columns or 'DOLocationID' not in df.columns:
        print("Dataframe must contain 'PULocationID' and 'DOLocationID' columns.")
        return
        
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['pickup_date'] = df['lpep_pickup_datetime'].dt.date
    
    # Number of trips per day per pair
    daily_counts = df.groupby(['pickup_date', 'PULocationID', 'DOLocationID']).size().reset_index(name='trip_count')
    
    # Average number of trips per pair per day
    avg_counts = daily_counts.groupby(['PULocationID', 'DOLocationID'])['trip_count'].mean().reset_index()
    
    # Select top N pairs
    top_pairs = avg_counts.sort_values(by='trip_count', ascending=False).head(top_n)
    
    # Create a pair label
    top_pairs['pair'] = top_pairs['PULocationID'].astype(str) + ' ➝ ' + top_pairs['DOLocationID'].astype(str)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create a viridis colormap
    viridis = cm.get_cmap('viridis', top_n)
    
    # Sort data for proper color mapping (highest value gets darkest color)
    top_pairs = top_pairs.sort_values(by='trip_count')
    
    # Plot bars manually with custom colors from viridis
    for i, (_, row) in enumerate(top_pairs.iterrows()):
        color_val = i / (len(top_pairs) - 1) if len(top_pairs) > 1 else 0.5
        bar_color = viridis(color_val)
        ax.barh(row['pair'], row['trip_count'], color=bar_color, height=0.6)
    
    # Add grid lines behind the bars
    ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    
    # Set background color
    ax.set_facecolor('white')
    
    # Add titles and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Avg Trips per Day', fontsize=14)
    plt.ylabel('Pickup ➝ Drop-off', fontsize=14)
    
    # Create colorbar to show the gradient mapping
    sm = plt.cm.ScalarMappable(cmap=viridis)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Trip Count Magnitude', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f"./plots/top_location_pairs_avg_daily_trips.png", dpi=300, facecolor='white')
    plt.close()



def run_eda(df):
    """Run full EDA on dataframe with adjusted distance scales"""
    # Create plots directory if it doesn't exist
    os.makedirs("./plots", exist_ok=True)
    
    print("Starting Exploratory Data Analysis...")
    
    # Data overview
    print("\nData Overview:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumns:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Add datetime features if not already present
    if 'pickup_hour' not in df.columns:
        print("\nEngineering datetime features...")
        df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['lpep_pickup_datetime'].dt.day_name()
        df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
        df['pickup_dayofweek'] = df['lpep_pickup_datetime'].dt.dayofweek
        
    # 1. Distribution of trip distances (with reasonable limit)
    print("\nPlotting trip distance distribution...")
    plot_distribution(df, 'trip_distance', 'Distribution of Trip Distances', 'Trip Distance (miles)')
    
    # 2. Distribution of trip durations
    print("\nPlotting trip duration distribution...")
    plot_distribution(df, 'duration', 'Distribution of Trip Durations', 'Trip Duration (minutes)')
    
    # 3. Hourly taxi demand - COLUMN BAR CHART
    print("\nPlotting hourly taxi demand...")
    hourly_counts = df.groupby('pickup_hour').size().reset_index(name='trip_count')
    plt.figure(figsize=(10, 4))
    sns.barplot(x='pickup_hour', y='trip_count', data=hourly_counts, width=0.6)
    plt.title('Hourly Taxi Demand', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=10)
    plt.ylabel('Number of Trips', fontsize=10)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig("./plots/hourly_demand.png", dpi=300)
    plt.close()
        
    # 4. Weekly taxi demand - COLUMN BAR CHART
    print("\nPlotting weekly taxi demand...")
    # Create day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Get day counts
    day_counts = df['pickup_day'].value_counts().reset_index()
    day_counts.columns = ['day', 'count']
    # Reorder
    day_counts['day'] = pd.Categorical(day_counts['day'], categories=day_order, ordered=True)
    day_counts = day_counts.sort_values('day')

    # Create figure and axes with specific style
    plt.figure(figsize=(10, 4))

    # Set the style to match the reference image
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Custom color palette similar to the reference image (blue shades)
    blue_colors = ['#4472C4', '#5B9BD5', '#8FAADC', '#B4C7E7', '#D9E1F2']
    bar_color = '#B4C7E7'  # Medium blue from reference image
    ##4472C4
    # Create barplot with custom style
    bars = plt.bar(day_counts['day'], day_counts['count'], width=0.6, color=bar_color)

    # Add error bars (if you have error data)
    # For demonstration, using 5% of the count as error
    # errors = day_counts['count'] * 0.05
    # plt.errorbar(x=day_counts['day'], y=day_counts['count'], yerr=errors, fmt='none', ecolor='black', capsize=3)

    # Add title and labels with specific styling
    plt.title('Weekly Taxi Demand', fontsize=14, fontweight='bold')
    plt.xlabel('Day of Week', fontsize=10)
    plt.ylabel('Number of Trips', fontsize=10)
    plt.xticks(rotation=45)

    # Add a subtle grid like in the reference image
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    # Ensure the x-axis labels are properly placed
    plt.tight_layout()

    # Save the figure
    plt.savefig("./plots/weekly_demand.png", dpi=300)
    plt.close()
        
    #5. Monthly taxi demand - COLUMN BAR CHART--only February month then skip this
    # Remove Jan if needed
    df = df[df['lpep_pickup_datetime'].dt.month != 1]

    # Then, group and sort
    month_counts = df.groupby('pickup_month').size().reset_index(name='trip_count')
    month_counts = month_counts.sort_values('pickup_month')

    # Map month numbers to names
    month_labels = month_counts['pickup_month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%b'))

    # Plot
    plt.figure(figsize=(10, 4))
    sns.barplot(x='pickup_month', y='trip_count', data=month_counts, width=0.6)
    plt.title('Monthly Taxi Demand', fontsize=14)
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('Number of Trips', fontsize=10)

    # Correctly set xticks
    plt.xticks(ticks=range(len(month_labels)), labels=month_labels)

    plt.tight_layout()
    plt.savefig("./plots/monthly_demand.png", dpi=300)
    plt.close()
    
    # 5.1  Average daily trips by VendorID
    print("\nPlotting average daily trips by VendorID...")
    plot_daily_avg_trips_by_vendor(df)

    # 5.2  Average daily trips by combininng pickup/dropoff
    print("\nPlotting top location pairs by average daily trips...")
    plot_top_location_pairs_avg_trips(df)

    # 6. Trip distance vs. duration scatter plot (with reasonable limits)
    print("\nPlotting trip distance vs. duration...")
    plt.figure(figsize=(10, 8))
    # Sample to avoid overcrowding the plot
    sample_size = min(10000, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    sns.scatterplot(x='trip_distance', y='duration', data=sample_df, alpha=0.5)
    plt.title('Trip Distance vs. Duration', fontsize=14)
    plt.xlabel('Trip Distance (miles)', fontsize=10)
    plt.ylabel('Trip Duration (minutes)', fontsize=10)
    # Set reasonable limits
    plt.xlim(0, 15)  # Most trips under 15 miles
    plt.ylim(0, 60)  # Most trips under 60 minutes
    plt.tight_layout()
    plt.savefig("./plots/distance_vs_duration.png", dpi=300)
    plt.close()
    
    # 7. Hourly trip distance heatmap (with reasonable scale)
    print("\nPlotting hourly trip distance heatmap...")
    if 'pickup_day' in df.columns:
        day_hour_distance = df.pivot_table(
            index='pickup_day', 
            columns='pickup_hour', 
            values='trip_distance', 
            aggfunc='mean'
        )
        
        # Reorder days
        day_hour_distance = day_hour_distance.reindex(day_order)
        
        # Create figure with white background
        plt.figure(figsize=(12, 8), facecolor='white')
        
        # Plot heatmap with adjusted scale for distance
        sns.heatmap(
            day_hour_distance, 
            cmap='viridis', 
            annot=False, 
            fmt='.2f', 
            cbar_kws={'label': 'Avg Trip Distance (miles)'},
            vmin=0,
            vmax=10,  # Set maximum to reasonable value for NYC taxi trips
            linewidths=0.5,
            linecolor='white'
        )
        
        plt.title('Average Trip Distance by Day and Hour', fontsize=16)
        plt.tight_layout()
        plt.savefig("./plots/day_hour_distance_heatmap.png", dpi=300, facecolor='white')
        plt.close()
    
    # 8. Geographic visualization
    print("\nPlotting geographic visualization...")
    nyc_zones = load_nyc_taxi_zones()
    if nyc_zones is not None:
        plot_geo_heatmap(df, nyc_zones, 'trip_count', 'Taxi Pickup Density')
    
    print("\nEDA complete! Plots saved to ./plots/ directory.")

@click.command()
@click.option(
    "--raw_data_path",
    required=True,
    help="Location where the NYC taxi trip data was saved"
)
@click.option(
    "--dataset",
    default="green",
    help="Type of taxi dataset (green/yellow)"
)
@click.option(
    "--months",
    default="2024-07",
    help="Comma-separated list of months to analyze (format: YYYY-MM)"
)
@click.option(
    "--sample_size",
    default=100000,
    type=int,
    help="Number of rows to sample for analysis (use -1 for all data)"
)
def main(raw_data_path, dataset, months, sample_size):
    """Run exploratory data analysis on NYC taxi trip data"""
    # Parse months
    month_list = [m.strip() for m in months.split(',')]
    
    print(f"Loading data for months: {month_list}")
    
    # Load data
    df = load_multiple_dataframes(raw_data_path, dataset, month_list)
    
    # Sample if needed
    if sample_size > 0 and sample_size < len(df):
        print(f"Sampling {sample_size} rows for analysis")
        df = df.sample(sample_size, random_state=42)
    
    # Run EDA
    run_eda(df)

    #RUN EDA after FE
    df_fe = engineer_features(df)
    run_eda(df_fe)

if __name__ == '__main__':
    main()