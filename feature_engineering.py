import os
import glob
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

def run_feature_engineering(input_dir="./processed", output_dir="./training_data", exclude_sensors=[]):
    print("Starting tsfresh feature engineering...")
    os.makedirs(output_dir, exist_ok=True)

    # This will extract only 10 basic, which is ideal for our small dataset
    extraction_settings = MinimalFCParameters()

    extracted_dataframes = []

    # extract features
    master_files = glob.glob(os.path.join(input_dir, "master_*.csv"))
    for file_path in master_files:
    
        sensor_name = os.path.basename(file_path).replace(".csv", "").replace("master_", "")
        if sensor_name in exclude_sensors: 
            continue # skip if this sensor is in the exclude list
        print(f"Extracting features for: {sensor_name}")
        
        df = pd.read_csv(file_path)
        
        features = extract_features(
            df, 
            column_id="Ride_ID", 
            column_sort="timestamp",
            default_fc_parameters=extraction_settings,
            n_jobs=0 
        )
        extracted_dataframes.append(features)

    # merge
    print("\nMerging data...")
    X_master = pd.concat(extracted_dataframes, axis=1)

    # clean up missing values
    impute(X_master)

    # skip feature selection and let xgboost handle it
    X_selected = X_master 

    print(f"Total features kept for training: {X_selected.shape[1]}")

    # save the data and the feature list
    final_dataset_path = os.path.join(output_dir, "X_selected_features.csv")
    X_selected.to_csv(final_dataset_path)
    
    feature_list_path = os.path.join(output_dir, "selected_feature_names.txt")
    with open(feature_list_path, "w") as f:
        for feature_name in X_selected.columns:
            f.write(f"{feature_name}\n")

    print(f"\nSuccess, final dataset saved to: {final_dataset_path}")

if __name__ == "__main__":
    exclude_sensors = ["optical_temperature_sensor", "accelerometer", "gyroscope", "magnetometer", "barometer", "photoplethysmograph", "temperature_sensor"]
    run_feature_engineering(exclude_sensors=exclude_sensors)