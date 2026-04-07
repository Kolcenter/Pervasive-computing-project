import os
import pandas as pd
from xgboost import XGBRegressor

def diagnose_training_data(features_path="./training_data/X_selected_features.csv",
                          labels_path="./processed/y_labels.csv",
                          model_path="./models/model.json",
                          feature_list_path="./training_data/selected_feature_names.txt"):
    print("Loading data and model for diagnostics...\n")
    
    # 1. Load the exact features used during training
    X = pd.read_csv(features_path, index_col=0)
    X.index.name = "Ride_ID"
    
    # 2. Load the true labels
    y_df = pd.read_csv(labels_path).set_index("Ride_ID")
    
    # Combine to ensure they are aligned perfectly
    combined = X.join(y_df, how="inner")
    X_clean = combined.drop(columns=["Distraction_Score"])
    y_true = combined["Distraction_Score"]
    
    # 3. Load the expected feature order to guarantee alignment
    with open(feature_list_path, "r") as f:
        required_features = [line.strip() for line in f.readlines() if line.strip()]
        
    # Reorder the columns to match exactly what the model expects
    X_aligned = X_clean[required_features]
    
    # 4. Load the trained model
    model = XGBRegressor()
    model.load_model(model_path)
    
    # 5. Make predictions on the entire training set
    predictions = model.predict(X_aligned)
    
    # 6. Print the results ride by ride
    print(f"{'Ride ID (Shortened)':<25} | {'True Score':<10} | {'Predicted':<10} | {'Error':<10}")
    print("-" * 65)
    
    total_error = 0
    for i in range(len(y_true)):
        ride_id = y_true.index[i].replace("OpenWearable_Recording_", "") # Shorten for display
        actual = y_true.iloc[i]
        pred = predictions[i]
        error = abs(actual - pred)
        total_error += error
        
        print(f"{ride_id:<25} | {actual:<10.1f} | {pred:<10.2f} | {error:<10.2f}")
        
    print("-" * 65)
    avg_mae = total_error / len(y_true)
    print(f"Average MAE across this dataset: {avg_mae:.2f}")

if __name__ == "__main__":
    diagnose_training_data()