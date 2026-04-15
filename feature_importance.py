import os
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


def analyze_feature_importance(model_path="./models/model.json", feature_list_path="./training_data/selected_feature_names.txt", output_csv_path="./models/feature_importance.csv"):

    if not os.path.exists(model_path) or not os.path.exists(feature_list_path):
        print("Error: Missing model or feature list artifacts.")
        return

    # 1. Load the exact feature names the model expects
    with open(feature_list_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Load the trained XGBoost model
    model = XGBRegressor()
    model.load_model(model_path)

    # 3. Extract the importance scores
    # XGBoost returns an array of numbers between 0.0 and 1.0
    importances = model.feature_importances_

    # 4. Combine into a Pandas DataFrame for easy sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort from highest impact to lowest impact
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # 5. Print the Top 15 features
    print("\nTop 15 Most Important Features for Predicting Distraction:")
    print("-" * 70)
    print(f"{'Rank':<5} | {'Feature Name':<45} | {'Importance Score'}")
    print("-" * 70)
    
    for i in range(min(15, len(importance_df))):
        feat = importance_df.loc[i, 'Feature']
        score = importance_df.loc[i, 'Importance']
        
        # Stop printing if the remaining features have zero impact
        if score == 0:
            break
            
        print(f"{i+1:<5} | {feat:<45} | {score:.4f}")

    print("-" * 70)

    # 6. Save the full list to a CSV for your report
    importance_df.to_csv(output_csv_path, index=False)
    print(f"\nFull importance ranking saved to: {output_csv_path}")

def plot_feature_importance():

    df = pd.read_csv("models/feature_importance.csv")
    
    # top 10
    top_features = df.head(10).sort_values(by="Importance", ascending=True)
    
    # clean up the feature names for the graph
    clean_names = []
    for name in top_features['Feature']:
        parts = name.split('__')
        sensor_part = parts[0].replace('_', ' ').title()
        stat_part = parts[1].replace('_', ' ').title()
        clean_names.append(f"{sensor_part}: {stat_part}")
        
    plt.figure(figsize=(10, 6))
    plt.barh(clean_names, top_features['Importance'], color='steelblue', edgecolor='black')
    
    plt.xlabel('Relative feature importance', fontsize=12, fontweight='bold')
    plt.title('Top 10 most predictive features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig('models/feature_importance_plot.png', dpi=300)
    print("Graph saved as feature_importance_plot.png")


if __name__ == "__main__":
    analyze_feature_importance()
    plot_feature_importance()