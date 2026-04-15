import os
import matplotlib.pyplot as plt
import xgboost as xgb

model_path = "./models/model.json"
output_image_path = "./models/trees_graphs"

def visualize_model_tree(j=1, visualize=False):
    
    if not os.path.exists(model_path):
        print("Model not found")
        return

    os.makedirs(output_image_path, exist_ok=True)

    # Load the trained model
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    for i in range(j):
        # create a fresh figure for each tree and close it after saving so it isn't displayed
        fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
        xgb.plot_tree(model, tree_idx=i, ax=ax, rankdir='TB')
        ax.set_title(f"XGBoost Decision Tree #{i} (max_depth=2)", fontsize=16, fontweight='bold', pad=20)
        fig.tight_layout()

        out_file = os.path.join(output_image_path, f"tree_{i}.png")
        fig.savefig(out_file, bbox_inches='tight')
        if not visualize:
            plt.close(fig)
    print(f"Saved {j} tree visualizations to: {output_image_path}")

if __name__ == "__main__":
    visualize_model_tree(visualize=True)