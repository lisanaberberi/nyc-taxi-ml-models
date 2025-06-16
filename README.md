#  NYC Green Taxi Trip Duration Prediction

This project builds and tracks machine learning models to predict taxi trip durations using New York City's green taxi trip records. It includes feature engineering, model training with various regressors, and MLflow tracking for experiment management.


##  Project Structure
``` 
├── data_preprocessing.py # Data cleaning & feature engineering utilities
├── model_training.py # Train baseline & feature-engineered models
├── utils.py # Utility functions (metrics & parameter logging)
├── plots/ # Generated EDA plots and feature insights
├── data/ # Data directory (green taxi parquet files)
├── exploratory_data_analysis/ # Code to analyse the dataframe
└── README.md # Project documentation
``` 


###  Model Training

Two workflows:

1. **Baseline Models** (minimal features)
   - Random Forest
   - LightGBM
   - Decision Tree
   - Gradient Boosting  
   → Trained on original features like pickup/dropoff location and trip distance.

2. **Custom Feature-Engineered Models**
   - Same models, trained on engineered features (set a combination of them):
     - `PU_DO`, `pickup_day`, `time_of_day`, `is_weekend`,`trip_distance`
   - Includes custom MLflow run tags for categorical/numerical columns.


### 📈 Experiment Tracking with MLflow
- Logs:
  - Model parameters
  - Performance metrics (MSE, MAE, R²)
  - Input data references
  - Model artifacts with inferred input/output signature  
- Supports nested MLflow runs for clean experiment grouping.


## 📊 Example Visualizations

Plots in `./plots/` include:
- **Distribution of Trip Durations**
- **Trip Counts**
- **Feature Distributions**

## 🛠️ Usage

Run the main script with the following command:

```bash
#First install requirements 
pip install -r requirements.txt

#first load all the data from your input dir
python data_preprocessing.py   --raw_data_path=<input_data_path> --dest_path=<destination_path>    --dataset=green   --taxi_zones_path=data/taxi_zones/taxi_zones.shp

#then run  main.py (the baseline models only)
python main.py  --raw_data_path=data/ --dest_path=data/output/    --dataset=green   --taxi_zones_path=data/taxi_zones/taxi_zones.shp  --run_baseline

#then run  main.py (custom models with feature_eng only)
python main.py  --raw_data_path=data/ --dest_path=data/output/    --dataset=green   --taxi_zones_path=data/taxi_zones/taxi_zones.shp  --feature_type=feature --run_custom 

#run  main.py (to check the results of the most important features against our target=duration)
python main.py  --raw_data_path=data/ --dest_path=data/output/    --dataset=green   --taxi_zones_path=data/taxi_zones/taxi_zones.shp  --feature_type=feature  --evaluate_features