"""
Main script for training DFU healing prediction models.

This script provides a simplified interface for training models with proper
cross-validation and model saving.
"""

import sys
import os
from pathlib import Path
import yaml
import pickle
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    DFUNextAppointmentPreprocessor,
    FlexibleResampler,
    filter_features_for_model
)
from src.models import train_fold
from src.config.constants import OPTIMIZED_FEATURES, SELECTED_FEATURES, SELECTED_BASE_FEATURES


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_detailed_results(all_fold_results, timestamp, save_dir):
    """
    Save detailed training results including confusion matrices for all strategies.

    Parameters
    ----------
    all_fold_results : list
        List of dictionaries containing results from each fold
    timestamp : str
        Timestamp string for file naming
    save_dir : Path
        Directory to save results
    """
    from sklearn.metrics import confusion_matrix

    results_dir = save_dir / "training_results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / f"training_results_{timestamp}.txt"

    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DFU HEALING TRAJECTORY PREDICTION - DETAILED TRAINING RESULTS\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Configuration summary
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of folds: {len(all_fold_results)}\n")
        f.write(f"Model type: ExtraTrees (optimized hyperparameters)\n")
        f.write(f"Evidence-based thresholds: I→I >21 days, P→P >28 days\n")
        f.write(f"References: Sheehan et al. 2003, Guo & DiPietro 2010\n")
        f.write("\n")

        # Per-fold detailed results
        f.write("="*80 + "\n")
        f.write("PER-FOLD DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")

        for idx, result in enumerate(all_fold_results, 1):
            f.write(f"\nFOLD {idx}\n")
            f.write("-" * 80 + "\n")

            # Phase-based metrics
            f.write("\n1. PHASE-BASED METRICS (Direct Phase Prediction)\n")
            f.write(f"   Accuracy:          {result['accuracy']:.4f}\n")
            f.write(f"   Balanced Accuracy: {result['balanced_accuracy']:.4f}\n")
            f.write(f"   F1-Weighted:       {result['f1_weighted']:.4f}\n")
            f.write(f"   F1-Macro:          {result['f1_macro']:.4f}\n")
            f.write(f"   Combined Score:    {result['combined_score']:.4f}\n")

            f.write("\n   Confusion Matrix (Phase-based):\n")
            cm = result['confusion_matrix']
            f.write("   Predicted:    I    P    R\n")
            f.write(f"   True I:    [{cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}]\n")
            if cm.shape[0] > 1:
                f.write(f"   True P:    [{cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}]\n")
            if cm.shape[0] > 2:
                f.write(f"   True R:    [{cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}]\n")

            f.write(f"\n   Per-class F1 scores: {result['f1_per_class']}\n")

            # Transition-based metrics (basic)
            f.write("\n2. TRANSITION-BASED METRICS (Basic Classification)\n")
            f.write(f"   Accuracy:          {result['transition_accuracy']:.4f}\n")
            f.write(f"   Balanced Accuracy: {result['transition_balanced_accuracy']:.4f}\n")
            f.write(f"   F1-Weighted:       {result['transition_f1_weighted']:.4f}\n")
            f.write(f"   F1-Macro:          {result['transition_f1_macro']:.4f}\n")
            f.write(f"   Combined Score:    {result['transition_combined_score']:.4f}\n")

            # Calculate confusion matrix for basic transitions
            from src.models.evaluation import convert_to_transition_labels
            y_true_trans_basic = convert_to_transition_labels(
                result['y_true'],
                [0] * len(result['y_true'])  # Placeholder, will be computed properly
            )
            y_pred_trans_basic = convert_to_transition_labels(
                result['y_pred'],
                [0] * len(result['y_pred'])
            )

            f.write(f"\n   Per-class F1 scores: {result['transition_f1_per_class']}\n")
            f.write("   Classes: 0=Unfavorable, 1=Acceptable, 2=Favorable\n")

            # Transition-based metrics (chronicity-aware)
            f.write("\n3. TRANSITION-BASED METRICS (Chronicity-Aware - Evidence-Based)\n")
            f.write(f"   Accuracy:          {result['transition_chrono_accuracy']:.4f}\n")
            f.write(f"   Balanced Accuracy: {result['transition_chrono_balanced_accuracy']:.4f}\n")
            f.write(f"   F1-Weighted:       {result['transition_chrono_f1_weighted']:.4f}\n")
            f.write(f"   F1-Macro:          {result['transition_chrono_f1_macro']:.4f}\n")
            f.write(f"   Combined Score:    {result['transition_chrono_combined_score']:.4f}\n")
            f.write(f"\n   Reclassified:      {result['transition_chrono_reclassified_count']} samples ")
            f.write(f"({result['transition_chrono_reclassified_pct']:.1f}%)\n")

            f.write(f"\n   Per-class F1 scores: {result['transition_chrono_f1_per_class']}\n")
            f.write("   Classes: 0=Unfavorable, 1=Acceptable, 2=Favorable\n")

            f.write("\n" + "-" * 80 + "\n")

        # Cross-validation summary
        f.write("\n" + "="*80 + "\n")
        f.write("CROSS-VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Phase-based summary
        accuracies = [r['accuracy'] for r in all_fold_results]
        balanced_accs = [r['balanced_accuracy'] for r in all_fold_results]
        f1_macros = [r['f1_macro'] for r in all_fold_results]

        f.write("1. PHASE-BASED METRICS\n")
        f.write(f"   Mean Accuracy:          {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"   Mean Balanced Accuracy: {np.mean(balanced_accs):.4f} ± {np.std(balanced_accs):.4f}\n")
        f.write(f"   Mean F1-Macro:          {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}\n")
        f.write("\n")

        # Transition-based (basic) summary
        trans_accs = [r['transition_accuracy'] for r in all_fold_results]
        trans_bal_accs = [r['transition_balanced_accuracy'] for r in all_fold_results]
        trans_f1s = [r['transition_f1_macro'] for r in all_fold_results]

        f.write("2. TRANSITION-BASED METRICS (Basic)\n")
        f.write(f"   Mean Accuracy:          {np.mean(trans_accs):.4f} ± {np.std(trans_accs):.4f}\n")
        f.write(f"   Mean Balanced Accuracy: {np.mean(trans_bal_accs):.4f} ± {np.std(trans_bal_accs):.4f}\n")
        f.write(f"   Mean F1-Macro:          {np.mean(trans_f1s):.4f} ± {np.std(trans_f1s):.4f}\n")
        f.write("\n")

        # Transition-based (chronicity) summary
        chrono_accs = [r['transition_chrono_accuracy'] for r in all_fold_results]
        chrono_bal_accs = [r['transition_chrono_balanced_accuracy'] for r in all_fold_results]
        chrono_f1s = [r['transition_chrono_f1_macro'] for r in all_fold_results]
        chrono_reclassified = [r['transition_chrono_reclassified_pct'] for r in all_fold_results]

        f.write("3. TRANSITION-BASED METRICS (Chronicity-Aware - Evidence-Based)\n")
        f.write(f"   Mean Accuracy:          {np.mean(chrono_accs):.4f} ± {np.std(chrono_accs):.4f}\n")
        f.write(f"   Mean Balanced Accuracy: {np.mean(chrono_bal_accs):.4f} ± {np.std(chrono_bal_accs):.4f}\n")
        f.write(f"   Mean F1-Macro:          {np.mean(chrono_f1s):.4f} ± {np.std(chrono_f1s):.4f}\n")
        f.write(f"   Mean Reclassified:      {np.mean(chrono_reclassified):.1f}% ± {np.std(chrono_reclassified):.1f}%\n")
        f.write("\n")

        # Aggregated confusion matrices across all folds
        f.write("="*80 + "\n")
        f.write("AGGREGATED CONFUSION MATRICES (All Folds Combined)\n")
        f.write("="*80 + "\n\n")

        # Concatenate all predictions from all folds
        all_y_true = np.concatenate([r['y_true'] for r in all_fold_results])
        all_y_pred = np.concatenate([r['y_pred'] for r in all_fold_results])

        # 1. Phase-based confusion matrix
        cm_phase = confusion_matrix(all_y_true, all_y_pred)

        f.write("1. PHASE-BASED CONFUSION MATRIX\n")
        f.write("   Predicted:       I      P      R    Total\n")
        f.write("   " + "-" * 45 + "\n")
        phase_names = ['I (Inflam)', 'P (Prolif)', 'R (Remodel)']
        for i in range(cm_phase.shape[0]):
            row_total = np.sum(cm_phase[i])
            f.write(f"   True {phase_names[i]:12s} ")
            for j in range(max(3, cm_phase.shape[1])):
                if j < cm_phase.shape[1]:
                    f.write(f"{cm_phase[i][j]:5d}  ")
                else:
                    f.write("    0  ")
            f.write(f"{row_total:5d}\n")

        col_totals = np.sum(cm_phase, axis=0)
        f.write("   " + "-" * 45 + "\n")
        f.write(f"   Total:             ")
        for j in range(max(3, len(col_totals))):
            if j < len(col_totals):
                f.write(f"{col_totals[j]:5d}  ")
            else:
                f.write("    0  ")
        f.write(f"{np.sum(cm_phase):5d}\n")

        # Calculate per-class metrics for phase-based
        f.write("\n   Per-class Performance:\n")
        phase_full_names = ['Inflammatory', 'Proliferative', 'Remodeling']
        for i in range(cm_phase.shape[0]):
            precision = cm_phase[i][i] / max(col_totals[i], 1) if i < len(col_totals) else 0
            recall = cm_phase[i][i] / max(np.sum(cm_phase[i]), 1)
            f.write(f"   {phase_full_names[i]:15s}: Precision={precision:.3f}, Recall={recall:.3f}\n")

        # 2. Basic transition-based - show F1 scores since we can't easily reconstruct
        f.write("\n\n2. TRANSITION-BASED METRICS (Basic) - Per-Class F1 Scores\n")
        f.write("   Classes: 0=Unfavorable, 1=Acceptable, 2=Favorable\n")
        all_trans_f1 = np.array([r['transition_f1_per_class'] for r in all_fold_results])
        mean_trans_f1 = np.mean(all_trans_f1, axis=0)

        f.write("   Mean F1 scores across all folds:\n")
        trans_class_names = ['Unfavorable', 'Acceptable', 'Favorable']
        for i, class_name in enumerate(trans_class_names):
            if i < len(mean_trans_f1):
                f.write(f"     {class_name:12s}: {mean_trans_f1[i]:.4f}\n")

        # 3. Chronicity-aware transition-based
        f.write("\n3. TRANSITION-BASED METRICS (Chronicity-Aware) - Per-Class F1 Scores\n")
        f.write("   Classes: 0=Unfavorable, 1=Acceptable, 2=Favorable\n")
        all_chrono_f1 = np.array([r['transition_chrono_f1_per_class'] for r in all_fold_results])
        mean_chrono_f1 = np.mean(all_chrono_f1, axis=0)

        f.write("   Mean F1 scores across all folds:\n")
        for i, class_name in enumerate(trans_class_names):
            if i < len(mean_chrono_f1):
                f.write(f"     {class_name:12s}: {mean_chrono_f1[i]:.4f}\n")

        # Show improvement in class-specific performance
        f.write("\n   Improvement (Chronicity-Aware vs Basic):\n")
        for i, class_name in enumerate(trans_class_names):
            if i < len(mean_chrono_f1) and i < len(mean_trans_f1):
                improvement = ((mean_chrono_f1[i] - mean_trans_f1[i]) / max(mean_trans_f1[i], 0.001)) * 100
                f.write(f"     {class_name:12s}: {improvement:+.2f}%\n")

        f.write("\n")

        # Performance comparison
        f.write("="*80 + "\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n\n")

        f.write("Metric                    Phase-based    Basic-Trans    Chrono-Trans   Change\n")
        f.write("-" * 80 + "\n")

        phase_ba = np.mean(balanced_accs)
        basic_ba = np.mean(trans_bal_accs)
        chrono_ba = np.mean(chrono_bal_accs)

        f.write(f"Balanced Accuracy         {phase_ba:6.4f}        {basic_ba:6.4f}        {chrono_ba:6.4f}     ")
        change = ((chrono_ba - basic_ba) / basic_ba) * 100
        f.write(f"{change:+.2f}%\n")

        phase_acc = np.mean(accuracies)
        basic_acc = np.mean(trans_accs)
        chrono_acc = np.mean(chrono_accs)

        f.write(f"Accuracy                  {phase_acc:6.4f}        {basic_acc:6.4f}        {chrono_acc:6.4f}     ")
        change = ((chrono_acc - basic_acc) / basic_acc) * 100
        f.write(f"{change:+.2f}%\n")

        phase_f1 = np.mean(f1_macros)
        basic_f1 = np.mean(trans_f1s)
        chrono_f1 = np.mean(chrono_f1s)

        f.write(f"F1-Macro                  {phase_f1:6.4f}        {basic_f1:6.4f}        {chrono_f1:6.4f}     ")
        change = ((chrono_f1 - basic_f1) / basic_f1) * 100
        f.write(f"{change:+.2f}%\n")

        f.write("\n")
        f.write("KEY FINDINGS:\n")
        f.write(f"- Evidence-based chronicity thresholds reclassified {np.mean(chrono_reclassified):.1f}% of samples\n")
        f.write(f"- Accuracy improved by {((chrono_acc - basic_acc) / basic_acc * 100):+.2f}%\n")
        f.write(f"- Balanced accuracy changed by {((chrono_ba - basic_ba) / basic_ba * 100):+.2f}%\n")
        f.write(f"- Clinical thresholds: I→I >21 days (chronic inflammation)\n")
        f.write(f"                       P→P >28 days (stalled proliferation)\n")
        f.write("\n")

        # References
        f.write("="*80 + "\n")
        f.write("CLINICAL EVIDENCE REFERENCES\n")
        f.write("="*80 + "\n\n")
        f.write("1. Sheehan et al. (2003). Diabetes Care 26(6):1879-1882\n")
        f.write("   - 50% wound area reduction at 4 weeks predicts healing (91% NPV)\n\n")
        f.write("2. Guo & DiPietro (2010). J Dental Research 89(3):219-229\n")
        f.write("   - Normal inflammatory phase: 2-7 days\n")
        f.write("   - DFU inflammatory phase: weeks to months\n\n")
        f.write("3. SVS Guidelines (2016). J Vascular Surgery 63:3S-21S\n")
        f.write("   - Grade 1B recommendation for 4-week assessment\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"\n  Detailed results saved to: {results_file}")
    return results_file


def main():
    """Main training pipeline."""
    print("="*70)
    print("DFU MODEL TRAINING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Load configuration
    config = load_config()

    # Extract configuration
    CSV_PATH = config['data']['csv_path']
    SAVE_DIR = Path(config['data']['save_dir'])
    SAVE_DIR.mkdir(exist_ok=True)

    training_config = config['training']
    USE_CROSS_VALIDATION = training_config['use_cross_validation']
    N_FOLDS = training_config['n_folds']
    TRAIN_SPLIT_RATIO = training_config['train_split_ratio']
    N_PATIENT_CLUSTERS = training_config['n_patient_clusters']
    RESAMPLING_STRATEGY = training_config['resampling_strategy']
    AUGMENTATION_TYPE = training_config['augmentation_type']

    print(f"\nConfiguration:")
    print(f"  Data path: {CSV_PATH}")
    print(f"  Training mode: {'Cross-Validation' if USE_CROSS_VALIDATION else 'Single Split'}")
    if USE_CROSS_VALIDATION:
        print(f"  Cross-validation folds: {N_FOLDS}")
    print(f"  Patient clusters: {N_PATIENT_CLUSTERS}")
    print(f"  Resampling: {RESAMPLING_STRATEGY}")
    print(f"  Augmentation type: {AUGMENTATION_TYPE}")

    # Initialize preprocessor
    print("\n1. Data preprocessing pipeline...")
    preprocessor = DFUNextAppointmentPreprocessor(CSV_PATH)

    # Preprocessing steps
    df = preprocessor.initial_cleaning()
    df = preprocessor.convert_categorical_to_numeric()
    df = preprocessor.create_temporal_features()

    # Create augmented dataset
    print("\n2. Creating augmented dataset...")
    df_processed, patient_cluster_map, kmeans_model = preprocessor.create_next_appointment_dataset_with_augmentation(
        n_patient_clusters=N_PATIENT_CLUSTERS,
        augmentation_type=AUGMENTATION_TYPE
    )

    # Prepare features
    print("\n3. Preparing features...")
    unique_patients = df_processed['Patient#'].unique()
    n_patients = len(unique_patients)
    print(f"  Total patients: {n_patients}")

    target_col = 'Next_Healing_Phase'
    exclude_cols = ['Patient#', 'DFU#', 'Appt#', target_col, 'ID']
    feature_cols = [col for col in df_processed.columns if col not in exclude_cols]

    # Filter to numeric columns
    numeric_feature_cols = []
    for col in feature_cols:
        if df_processed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_feature_cols.append(col)

    # Use optimized features if available (from hyperparameter optimization)
    available_features = []
    for feat in OPTIMIZED_FEATURES:
        if feat in df_processed.columns:
            available_features.append(feat)

    if len(available_features) > 0:
        feature_cols = available_features
        print(f"  Using {len(feature_cols)} optimized features (from Oct 2025 optimization)")
    else:
        # Fallback to legacy features
        available_features = []
        for feat in SELECTED_FEATURES:
            if feat in df_processed.columns:
                available_features.append(feat)

        if len(available_features) > 0:
            feature_cols = available_features
            print(f"  Using {len(feature_cols)} legacy selected features")
        else:
            feature_cols = filter_features_for_model(df_processed, numeric_feature_cols)
            print(f"  Using {len(feature_cols)} filtered features")

    # Initialize resampler
    resampler = FlexibleResampler(strategy=RESAMPLING_STRATEGY)

    # Extract model parameters from config
    model_type = config['model']['type']
    if model_type == 'ExtraTrees':
        model_params = config['model']['extratrees']
    else:
        model_params = config['model']['gradientboosting']

    print(f"\nModel Configuration:")
    print(f"  Type: {model_type}")
    print(f"  Parameters: {model_params}")

    # Initialize best model variables
    best_model = None
    best_scaler = None
    best_imputer = None
    best_val_patients = None
    best_fold_results = None

    if USE_CROSS_VALIDATION:
        print("\n4. Starting cross-validation...")

        all_fold_results = []
        all_fold_models = []
        all_fold_scalers = []
        all_fold_imputers = []
        all_fold_val_patients = []

        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(unique_patients)):
            train_patients = unique_patients[train_idx]
            val_patients = unique_patients[val_idx]

            fold_results, model, scaler, imputer, val_patients_list = train_fold(
                fold, train_patients, val_patients, df_processed,
                preprocessor,
                feature_cols, target_col, resampler,
                AUGMENTATION_TYPE,
                patient_cluster_map,
                model_type=model_type,
                model_params=model_params
            )

            all_fold_results.append(fold_results)
            all_fold_models.append(model)
            all_fold_scalers.append(scaler)
            all_fold_imputers.append(imputer)
            all_fold_val_patients.append(val_patients_list)

        # Select best model
        print("\n" + "="*70)
        print("SELECTING BEST MODEL")
        print("="*70)

        best_fold_idx = np.argmax([r['combined_score'] for r in all_fold_results])
        best_fold_results = all_fold_results[best_fold_idx]
        best_model = all_fold_models[best_fold_idx]
        best_scaler = all_fold_scalers[best_fold_idx]
        best_imputer = all_fold_imputers[best_fold_idx]
        best_val_patients = all_fold_val_patients[best_fold_idx]

        print(f"\nBest model from Fold {best_fold_idx + 1}:")
        print(f"  Accuracy: {best_fold_results['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {best_fold_results['balanced_accuracy']:.4f}")
        print(f"  F1-Macro: {best_fold_results['f1_macro']:.4f}")

        # Print overall statistics
        accuracies = [r['accuracy'] for r in all_fold_results]
        balanced_accs = [r['balanced_accuracy'] for r in all_fold_results]
        f1_macros = [r['f1_macro'] for r in all_fold_results]

        # Transition-based statistics (basic)
        transition_accuracies = [r['transition_accuracy'] for r in all_fold_results]
        transition_balanced_accs = [r['transition_balanced_accuracy'] for r in all_fold_results]
        transition_f1_macros = [r['transition_f1_macro'] for r in all_fold_results]

        # Transition-based statistics (chronicity-aware)
        chrono_accuracies = [r['transition_chrono_accuracy'] for r in all_fold_results]
        chrono_balanced_accs = [r['transition_chrono_balanced_accuracy'] for r in all_fold_results]
        chrono_f1_macros = [r['transition_chrono_f1_macro'] for r in all_fold_results]
        chrono_reclassified = [r['transition_chrono_reclassified_pct'] for r in all_fold_results]

        print(f"\nCross-Validation Summary (Phase-based):")
        print(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Mean Balanced Accuracy: {np.mean(balanced_accs):.4f} ± {np.std(balanced_accs):.4f}")
        print(f"  Mean F1-Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")

        print(f"\nCross-Validation Summary (Transition-based - Basic):")
        print(f"  Mean Accuracy: {np.mean(transition_accuracies):.4f} ± {np.std(transition_accuracies):.4f}")
        print(f"  Mean Balanced Accuracy: {np.mean(transition_balanced_accs):.4f} ± {np.std(transition_balanced_accs):.4f}")
        print(f"  Mean F1-Macro: {np.mean(transition_f1_macros):.4f} ± {np.std(transition_f1_macros):.4f}")

        print(f"\nCross-Validation Summary (Transition-based - Chronicity-Aware):")
        print(f"  Mean Accuracy: {np.mean(chrono_accuracies):.4f} ± {np.std(chrono_accuracies):.4f}")
        print(f"  Mean Balanced Accuracy: {np.mean(chrono_balanced_accs):.4f} ± {np.std(chrono_balanced_accs):.4f}")
        print(f"  Mean F1-Macro: {np.mean(chrono_f1_macros):.4f} ± {np.std(chrono_f1_macros):.4f}")
        print(f"  Mean Reclassified: {np.mean(chrono_reclassified):.1f}% ± {np.std(chrono_reclassified):.1f}%")
        print(f"  Evidence-based thresholds: I→I >21d, P→P >28d (Sheehan 2003, Guo 2010)")

    # Save model and components
    print("\n" + "="*70)
    print("SAVING MODEL AND COMPONENTS")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = SAVE_DIR / f"best_model_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  Model saved to: {model_path}")

    # Save scaler
    scaler_path = SAVE_DIR / f"scaler_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(best_scaler, f)
    print(f"  Scaler saved to: {scaler_path}")

    # Save imputer
    imputer_path = SAVE_DIR / f"imputer_{timestamp}.pkl"
    with open(imputer_path, 'wb') as f:
        pickle.dump(best_imputer, f)
    print(f"  Imputer saved to: {imputer_path}")

    # Save preprocessing parameters
    preprocessing_params = {
        'features_to_remove': preprocessor.features_to_remove,
        'string_to_numeric_mappings': preprocessor.string_to_numeric_mappings,
        'ordinal_mappings': preprocessor.ordinal_mappings,
        'categorical_columns': preprocessor.categorical_columns,
        'integer_columns': preprocessor.integer_columns,
        'numerical_columns': preprocessor.numerical_columns,
        'label_encoders': {col: {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
                          for col, le in preprocessor.label_encoders.items()},
        'target_col': preprocessor.target_col,
        'ordinal_mapping': preprocessor.ordinal_mapping,
        'reverse_mapping': preprocessor.reverse_mapping,
        'n_patient_clusters': N_PATIENT_CLUSTERS,
        'selected_base_features': SELECTED_BASE_FEATURES,
        'augmentation_type': AUGMENTATION_TYPE
    }

    preprocessing_path = SAVE_DIR / f"preprocessing_params_{timestamp}.pkl"
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing_params, f)
    print(f"  Preprocessing parameters saved to: {preprocessing_path}")

    # Save feature names
    feature_names_path = SAVE_DIR / f"feature_names_{timestamp}.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"  Feature names saved: {feature_names_path}")

    # Save detailed results report
    if USE_CROSS_VALIDATION:
        save_detailed_results(all_fold_results, timestamp, SAVE_DIR)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Model Performance:")
    print(f"  Accuracy: {best_fold_results['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_fold_results.get('balanced_accuracy', 0):.4f}")
    print(f"  F1-Macro: {best_fold_results.get('f1_macro', 0):.4f}")
    print(f"\nAll files saved with timestamp: {timestamp}")

    return timestamp


if __name__ == "__main__":
    timestamp = main()
    print(f"\n✓ Training completed successfully. Timestamp: {timestamp}")
