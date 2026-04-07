import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_evaluate(features_path="./training_data/X_selected_features.csv",
                       labels_path="./processed/y_labels.csv",
                       custom_labels=False,
                       custom_labels_path="./processed/y_labels_custom.csv",
                       model_output_path="./models/model.json"):
    
    # 1. Load Data
    labels_path = custom_labels_path if custom_labels else labels_path
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Error: Could not find the data files.")
        return
        
    # Use column 0 (the first column) as the index, since the header is blank
    X = pd.read_csv(features_path, index_col=0)
    X.index.name = "Ride_ID"  # Manually restore the name so the files link up perfectly
    y_df = pd.read_csv(labels_path).set_index("Ride_ID")
    
    # Combine and clean
    combined = X.join(y_df, how="inner")
    X_clean = combined.drop(columns=["Distraction_Score"])
    y_clean = combined["Distraction_Score"]
    
    print(f"Dataset ready: {X_clean.shape[0]} rides, {X_clean.shape[1]} features per ride.")

    # 2. Setup XGBoost (Hyperparameters strictly tuned for tiny datasets)
    model = XGBRegressor(
        n_estimators=50,       
        learning_rate=0.05,     
        max_depth=2,            # Extremely shallow trees so it doesn't memorize the data
        subsample=0.8,
        random_state=42
    )

    # 3. Evaluate using Leave-One-Out Cross-Validation
    print("Evaluating model using Leave-One-Out Cross-Validation...")
    loo = LeaveOneOut()
    
    y_true_all = []
    y_pred_all = []
    
    # This loop trains 12 separate models to test accuracy fairly
    for train_index, test_index in loo.split(X_clean):
        X_train, X_test = X_clean.iloc[train_index], X_clean.iloc[test_index]
        y_train, y_test = y_clean.iloc[train_index], y_clean.iloc[test_index]
        
        # Train on 11 rides
        model.fit(X_train, y_train)
        
        # Predict the 1 hidden ride
        prediction = model.predict(X_test)[0]
        
        y_true_all.append(y_test.values[0])
        y_pred_all.append(prediction)

    # 4. Calculate Final True Accuracy
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE REPORT")
    print("="*40)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("="*40)
    print(f"The model's guesses are off by an average of {mae:.2f} points on the 0-4 scale.")

    # 5. Train the FINAL Production Model
    print("\nTraining final production model on 100% of the data...")
    # Now that we know the true accuracy, we use all 12 rides to make the best possible model for the app
    model.fit(X_clean, y_clean)
    
    # 6. Save the Model
    model.save_model(model_output_path)
    print(f"Trained model saved to: {model_output_path}")
if __name__ == "__main__":
    train_and_evaluate()