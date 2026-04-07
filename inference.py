import os
import glob
import pandas as pd
from xgboost import XGBRegressor
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

new_ride_folder = "./OpenWearable_Recording_2026-03-26T101935.303786_3"
model_path = "./models/model.json"
feature_list_path = "./training_data/selected_feature_names.txt"

def predict_score(new_ride_folder="./OpenWearable_Recording_2026-03-26T101935.303786_3", model_path="./models/model.json", feature_list_path="./training_data/selected_feature_names.txt"):
    print(f"Processing new ride: {os.path.basename(new_ride_folder)}")

    with open(feature_list_path, "r") as f:
        required_features = [line.strip() for line in f.readlines() if line.strip()]
    
    csv_files = glob.glob(os.path.join(new_ride_folder, "*.csv"))
    live_ride_id = "LIVE_RIDE_001" 
    sensor_groups = {}
    
    # 1. Group files by sensor type
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        if file_name.startswith("._") or "ACCELEROMETER_1" in file_name:
            continue 
            
        sensor_name = file_name.split('_')[-1].replace('.csv', '')
        
        try:
            df = pd.read_csv(csv_file)
            df.insert(0, 'Ride_ID', live_ride_id)
            # Rename columns dynamically
            new_cols = {col: f"{sensor_name}_{col}" for col in df.columns if col not in ['Ride_ID', 'timestamp']}
            df.rename(columns=new_cols, inplace=True)
            
            if sensor_name not in sensor_groups:
                sensor_groups[sensor_name] = []
            sensor_groups[sensor_name].append(df)
        except Exception:
            continue

    if not sensor_groups:
        print("Error: Could not process sensor files.")
        return

    # 2. Extract features per sensor group
    live_features_list = []
    for sensor_name, dfs in sensor_groups.items():
        merged_df = pd.concat(dfs, ignore_index=True)
        features = extract_features(
            merged_df, 
            column_id="Ride_ID", 
            column_sort="timestamp",
            default_fc_parameters=MinimalFCParameters(),
            n_jobs=0,
            disable_progressbar=True 
        )
        live_features_list.append(features)
        
    # Merge all extractions and clean up
    live_features = pd.concat(live_features_list, axis=1)
    live_features = live_features.loc[:, ~live_features.columns.duplicated()].copy()
    impute(live_features)

    # 3. Align with expected model features and cast to float
    aligned_dict = {}
    for feature in required_features:
        if feature in live_features.columns:
            extracted_col = live_features[feature]
            aligned_dict[feature] = extracted_col.iloc[0, 0] if isinstance(extracted_col, pd.DataFrame) else extracted_col.iloc[0]
        else:
            aligned_dict[feature] = 0.0 

    live_features_aligned = pd.DataFrame([aligned_dict], index=[live_ride_id])
    live_features_aligned = live_features_aligned[required_features].astype(float)

    # 4. Predict
    model = XGBRegressor()
    model.load_model(model_path)
    prediction = model.predict(live_features_aligned)[0]

    print(f"Calculated distraction score: {prediction:.2f} / 4.0")

if __name__ == "__main__":
    predict_score()