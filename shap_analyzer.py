import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
print("--- SHAP Interaction Analyzer for LightGBM ---")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Must match the files used/created by 'LightGBM_Alpha_Model.py'
DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv') 
MODEL_DIR = os.path.join(ROOT_DIR, 'lgbm_model')
MODEL_INPUT_FILE = os.path.join(MODEL_DIR, 'lgbm_alpha_model.joblib')

# --- 2. DEFINE OUTPUT FILES ---
# We will create three artifacts in the model directory
OUTPUT_PLOT_FILE = os.path.join(MODEL_DIR, 'lgbm_shap_interaction_summary.png')
OUTPUT_TEXT_FILE = os.path.join(MODEL_DIR, 'lgbm_shap_interaction_ranking.txt')
OUTPUT_DEP_DIR = os.path.join(MODEL_DIR, 'shap_dependence_plots')

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DEP_DIR, exist_ok=True)

# Number of interactions to analyze
TOP_K_INTERACTIONS = 20
TOP_K_PLOTS = 5

def load_data_for_analysis(data_path):
    """
    Loads and preprocesses data *exactly* as the training script did.
    We will use the 'test' split for an unbiased analysis.
    """
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure 'tsp_features.csv' exists.")
        return None

    # --- Define Features (X) ---
    # This logic is copied directly from LightGBM_Alpha_Model.py
    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'split', 'distribution_type'
    ]
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)

    # --- Identify Categorical Features ---
    categorical_features = ['dimension', 'grid_size']
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    # --- Get the Test Set for Analysis ---
    # Analyzing the test set shows how the model behaves on unseen data
    test_mask = (df['split'] == 'test')
    X_analysis = X[test_mask].copy()
    
    print(f"Loaded {len(X_analysis)} test samples for analysis.")
    return X_analysis

def get_ranked_interactions(shap_interaction_values, feature_names):
    """
    Calculates the mean absolute SHAP value for all off-diagonal
    interaction pairs and returns a ranked DataFrame.
    """
    # 1. Calculate mean absolute interaction values
    # Shape goes from (N, K, K) -> (K, K)
    mean_abs_interactions = np.abs(shap_interaction_values).mean(0)
    
    interactions = []
    
    # 2. Extract unique (i, j) pairs where i < j
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            interactions.append({
                'feature_1': feature_names[i],
                'feature_2': feature_names[j],
                'interaction_strength': mean_abs_interactions[i, j]
            })
            
    # 3. Create and sort the DataFrame
    interactions_df = pd.DataFrame(interactions)
    interactions_df = interactions_df.sort_values(by='interaction_strength', ascending=False)
    
    return interactions_df.reset_index(drop=True)

def main():
    # --- 1. Load Data ---
    X_analysis = load_data_for_analysis(DATA_FILE)
    if X_analysis is None:
        return

    # --- 2. Load Model ---
    print(f"Loading trained model from {MODEL_INPUT_FILE}...")
    try:
        model = joblib.load(MODEL_INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_INPUT_FILE}")
        print("Please run 'LightGBM_Alpha_Model.py' first to train the model.")
        return

    # --- 3. Compute SHAP Interaction Values ---
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    print(f"Computing SHAP interaction values for {len(X_analysis)} samples...")
    print("This may take several minutes...")
    # This computes the full (N, K, K) interaction matrix
    shap_interaction_values = explainer.shap_interaction_values(X_analysis)
    print("SHAP computation complete.")

    # --- 4. Save Main Summary Plot ---
    print(f"Saving main interaction summary plot to {OUTPUT_PLOT_FILE}...")
    # This plot shows the top 20 features, ranked by their
    # main effect + interaction effects.
    shap.summary_plot(
        shap_interaction_values, 
        X_analysis, 
        max_display=TOP_K_INTERACTIONS,
        show=False
    )
    plt.savefig(OUTPUT_PLOT_FILE, bbox_inches='tight', dpi=150)
    plt.close()
    print("...Summary plot saved.")

    # --- 5. Get and Save Ranked Text List ---
    ranked_interactions_df = get_ranked_interactions(
        shap_interaction_values, 
        X_analysis.columns
    )
    
    print(f"Saving top {TOP_K_INTERACTIONS} interactions to {OUTPUT_TEXT_FILE}...")
    with open(OUTPUT_TEXT_FILE, 'w') as f:
        f.write(f"Top {TOP_K_INTERACTIONS} Feature Interactions for LightGBM Alpha Model\n")
        f.write("Ranked by mean absolute SHAP interaction value\n")
        f.write("----------------------------------------------------------\n\n")
        f.write(
            ranked_interactions_df.head(TOP_K_INTERACTIONS).to_string()
        )
    print("...Interaction ranking saved.")

    # --- 6. Save Top K Dependence Plots ---
    print(f"Saving top {TOP_K_PLOTS} dependence plots to {OUTPUT_DEP_DIR}...")
    
    for _, row in ranked_interactions_df.head(TOP_K_PLOTS).iterrows():
        f1 = row['feature_1']
        f2 = row['feature_2']
        
        plot_filename = os.path.join(OUTPUT_DEP_DIR, f'1_dep_plot_{f1}_x_{f2}.png')
        
        # Create the plot: (feature_1, feature_2)
        # This shows the effect of f1, colored by f2
        try:
            shap.dependence_plot(
                (f1, f2),
                shap_interaction_values,
                X_analysis,
                show=False
            )
            plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
            plt.close()
            
            # Create the inverse plot: (feature_2, feature_1)
            # This shows the effect of f2, colored by f1
            plot_filename_inv = os.path.join(OUTPUT_DEP_DIR, f'2_dep_plot_{f2}_x_{f1}.png')
            shap.dependence_plot(
                (f2, f1),
                shap_interaction_values,
                X_analysis,
                show=False
            )
            plt.savefig(plot_filename_inv, bbox_inches='tight', dpi=100)
            plt.close()
            
        except Exception as e:
            print(f"  ...Could not plot {f1} x {f2}. Error: {e}")
            # Close any open plots to avoid memory leaks
            plt.close()

    print("...Dependence plots saved.")
    print("\n✅ Analysis complete.")
    print(f"Check the '{MODEL_DIR}' folder for results.")

if __name__ == "__main__":
    main()