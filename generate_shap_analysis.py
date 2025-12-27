"""
Generate SHAP analysis visualizations for manuscript
Addresses Reviewer 1 request for model interpretability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import yaml
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Loading trained model and data...")

# Find the most recent model file
saved_models_dir = Path('saved_models')
model_files = list(saved_models_dir.glob('best_model_*.pkl'))

if not model_files:
    print("ERROR: No model files found in saved_models/. Please run training first.")
    exit(1)

# Get the most recent model file
model_path = max(model_files, key=lambda p: p.stat().st_mtime)
print(f"Using model: {model_path.name}")

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load feature names
timestamp = model_path.stem.replace('best_model_', '')
feature_names_path = saved_models_dir / f'feature_names_{timestamp}.pkl'

if feature_names_path.exists():
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
else:
    feature_names = None

# Load and preprocess data
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing import DFUNextAppointmentPreprocessor

# Load configuration
CSV_PATH = config['data']['csv_path']
N_PATIENT_CLUSTERS = config['training']['n_patient_clusters']
AUGMENTATION_TYPE = config['training']['augmentation_type']

print("Preprocessing data...")
preprocessor = DFUNextAppointmentPreprocessor(CSV_PATH)
df = preprocessor.initial_cleaning()
df = preprocessor.convert_categorical_to_numeric()
df = preprocessor.create_temporal_features()

# Create augmented dataset
df_processed, patient_cluster_map, kmeans_model = preprocessor.create_next_appointment_dataset_with_augmentation(
    n_patient_clusters=N_PATIENT_CLUSTERS,
    augmentation_type=AUGMENTATION_TYPE
)

# Prepare features
target_col = 'Next_Healing_Phase'
exclude_cols = ['Patient#', 'DFU#', 'Appt#', target_col, 'ID']
feature_cols = [col for col in df_processed.columns if col not in exclude_cols]

# Filter to numeric columns only
numeric_feature_cols = []
for col in feature_cols:
    if df_processed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        numeric_feature_cols.append(col)

# Use the feature names that match the model
if feature_names:
    # Ensure we use the same features the model was trained on
    available_features = [f for f in feature_names if f in df_processed.columns]
    X_train = df_processed[available_features].copy()
else:
    # Fallback to all numeric features
    X_train = df_processed[numeric_feature_cols].copy()
    feature_names = numeric_feature_cols

y_train = df_processed[target_col].values

print(f"Data shape: {X_train.shape}")
print(f"Number of features: {len(feature_names)}")

# Handle missing values  - use the same imputer from training
imputer_path = saved_models_dir / f'imputer_{timestamp}.pkl'
if imputer_path.exists():
    print("Loading imputer from training...")
    with open(imputer_path, 'rb') as f:
        imputer = pickle.load(f)
    X_train_imputed = imputer.transform(X_train)
    X_train = pd.DataFrame(X_train_imputed, columns=feature_names)
else:
    # Fill NaN values with median
    X_train = X_train.fillna(X_train.median())

# Load scaler
scaler_path = saved_models_dir / f'scaler_{timestamp}.pkl'
if scaler_path.exists():
    print("Loading scaler from training...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_train_scaled = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
else:
    print("Warning: Scaler not found, using unscaled data")

# Convert to numpy array if needed
if hasattr(X_train, 'values'):
    X_train_array = X_train.values
else:
    X_train_array = X_train

# Limit to 200 samples for computational efficiency
if X_train_array.shape[0] > 200:
    np.random.seed(42)
    sample_indices = np.random.choice(X_train_array.shape[0], 200, replace=False)
    X_sample = X_train_array[sample_indices]
else:
    X_sample = X_train_array

print(f"\nCalculating SHAP values (using {X_sample.shape[0]} samples)...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Handle multi-class output
if isinstance(shap_values, list):
    # Multi-class: shap_values is a list of arrays, one per class
    n_classes = len(shap_values)
    class_names = ['Unfavorable', 'Acceptable', 'Favorable'][:n_classes]
    print(f"Multi-class model detected: {n_classes} classes")
    shap_values_for_importance = shap_values[0]
elif len(shap_values.shape) == 3:
    # TreeExplainer with multi-class returns 3D array (samples, features, classes)
    n_classes = shap_values.shape[2]
    class_names = ['Unfavorable', 'Acceptable', 'Favorable'][:n_classes]
    print(f"Multi-class model detected: {n_classes} classes (3D array)")
    # Split the 3D array into list of 2D arrays per class
    shap_values = [shap_values[:, :, i] for i in range(n_classes)]
    shap_values_for_importance = shap_values[0]
else:
    # Binary or single output
    shap_values = [shap_values]
    class_names = ['Prediction']
    shap_values_for_importance = shap_values[0]

print(f"SHAP values for importance shape: {shap_values_for_importance.shape}")

print("\nGenerating visualizations...")

# Create output directory
output_dir = Path('figures/shap_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. SHAP Summary Plot (Global Feature Importance) - Multi-class
# ============================================================================
print("1. Creating SHAP summary plot (global importance)...")

if len(shap_values) > 1:
    # Multi-class: show all classes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        shap.summary_plot(
            shap_values[idx],
            X_sample,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=15
        )
        plt.sca(ax)
        plt.title(f'{class_name} Transitions', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_multiclass.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'shap_summary_multiclass.pdf', bbox_inches='tight')
    print(f"   Saved: {output_dir / 'shap_summary_multiclass.png'}")
    plt.close()

# Single summary plot (using first class or combined)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values[0],
    X_sample,
    feature_names=feature_names,
    plot_type="dot",
    show=False,
    max_display=20
)
plt.title('SHAP Feature Importance (Transition Prediction)', fontsize=16, fontweight='bold', pad=20)
# Increase axis label and tick font sizes
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel(ax.get_xlabel(), fontsize=14)
ax.set_ylabel(ax.get_ylabel(), fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / 'shap_summary_global.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'shap_summary_global.pdf', bbox_inches='tight')
print(f"   Saved: {output_dir / 'shap_summary_global.png'}")
plt.close()

# ============================================================================
# 2. SHAP Bar Plot (Mean Absolute SHAP Values)
# ============================================================================
print("2. Creating SHAP bar plot (mean importance)...")

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values[0],
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.title('Mean Absolute SHAP Values (Feature Importance)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'shap_bar_plot.pdf', bbox_inches='tight')
print(f"   Saved: {output_dir / 'shap_bar_plot.png'}")
plt.close()

# ============================================================================
# 3. SHAP Dependence Plots (Top 3 Features)
# ============================================================================
print("3. Creating SHAP dependence plots (top 3 features)...")

# Calculate mean absolute SHAP values to find top features
mean_abs_shap = np.abs(shap_values_for_importance).mean(axis=0)
top_features_idx_array = np.argsort(mean_abs_shap)[-3:][::-1]

# Convert feature_names to numpy array for easy indexing
feature_names_array = np.array(feature_names if isinstance(feature_names, list) else list(feature_names))
top_features = feature_names_array[top_features_idx_array].tolist()
top_features_idx = top_features_idx_array.tolist()

print(f"   Top 3 features: {top_features}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (feature_idx, feature_name) in enumerate(zip(top_features_idx, top_features)):
    plt.sca(axes[idx])
    shap.dependence_plot(
        feature_idx,  # Already converted to int
        shap_values_for_importance,
        X_sample,
        feature_names=feature_names,
        show=False,
        ax=axes[idx]
    )
    axes[idx].set_title(f'SHAP Dependence: {feature_name}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'shap_dependence_top3.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'shap_dependence_top3.pdf', bbox_inches='tight')
print(f"   Saved: {output_dir / 'shap_dependence_top3.png'}")
plt.close()

# ============================================================================
# 4. SHAP Force Plot (Example Predictions)
# ============================================================================
print("4. Creating SHAP force plots (example predictions)...")

# Get predictions for the sample
predictions = model.predict(X_sample)

# Find examples of each class
examples = {}
for class_idx, class_name in enumerate(class_names):
    mask = predictions == class_idx
    if mask.sum() > 0:
        # Get first example of this class
        example_idx = np.where(mask)[0][0]
        examples[class_name] = example_idx

# Create force plots for each example
for class_name, example_idx in examples.items():
    print(f"   Creating force plot for {class_name} example...")

    # Force plot returns matplotlib figure
    shap.force_plot(
        explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        shap_values[0][example_idx, :],
        X_sample[example_idx, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )

    plt.title(f'SHAP Force Plot: {class_name} Transition Example', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()

    safe_name = class_name.lower().replace(' ', '_')
    plt.savefig(output_dir / f'shap_force_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'shap_force_{safe_name}.pdf', bbox_inches='tight')
    print(f"   Saved: {output_dir / f'shap_force_{safe_name}.png'}")
    plt.close()

# ============================================================================
# 5. Feature Importance Table
# ============================================================================
print("5. Creating feature importance table...")

# Calculate mean absolute SHAP values for each feature
feature_importance = pd.DataFrame({
    'Feature': feature_names_array.tolist(),
    'Mean_Abs_SHAP': mean_abs_shap,
    'Rank': range(1, len(feature_names_array) + 1)
}).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

feature_importance['Rank'] = range(1, len(feature_importance) + 1)

# Save top 20 features
top_20 = feature_importance.head(20)
top_20.to_csv(output_dir / 'shap_feature_importance_top20.csv', index=False)
print(f"   Saved: {output_dir / 'shap_feature_importance_top20.csv'}")

# Save full table
feature_importance.to_csv(output_dir / 'shap_feature_importance_full.csv', index=False)
print(f"   Saved: {output_dir / 'shap_feature_importance_full.csv'}")

# Print top 10
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES (by mean absolute SHAP value)")
print("="*70)
print(top_20.head(10).to_string(index=False))
print("="*70)

print("\n✅ SHAP analysis complete!")
print(f"\nAll visualizations saved to: {output_dir.absolute()}")
print("\nGenerated files:")
print("  - shap_summary_global.png/pdf (main figure for manuscript)")
print("  - shap_summary_multiclass.png/pdf (all 3 classes)")
print("  - shap_bar_plot.png/pdf (mean importance)")
print("  - shap_dependence_top3.png/pdf (top 3 features)")
print("  - shap_force_*.png/pdf (example predictions)")
print("  - shap_feature_importance_*.csv (importance tables)")
