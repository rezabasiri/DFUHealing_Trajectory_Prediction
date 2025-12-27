"""
Model creation and training functions for DFU healing prediction.

This module contains functions for creating and training machine learning models.
"""

import time
import numpy as np
import warnings

# Suppress numpy warnings during KNN imputation
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)

from .evaluation import (analyze_transitions, calculate_transition_metrics,
                         calculate_transition_metrics_chronicity_aware)


def train_fold(fold, train_patients, val_patients, df_processed_full, preprocessor,
               feature_cols, target_col, resampler, augmentation_type, patient_cluster_map,
               model_type='ExtraTrees', model_params=None):
    """
    Train a single fold with proper augmentation separation.

    Parameters
    ----------
    fold : int
        Fold number
    train_patients : array-like
        Patient IDs for training
    val_patients : array-like
        Patient IDs for validation
    df_processed_full : pd.DataFrame
        Full processed dataset
    preprocessor : DFUNextAppointmentPreprocessor
        Preprocessor instance
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    resampler : FlexibleResampler
        Resampler instance
    augmentation_type : str
        Augmentation strategy
    patient_cluster_map : dict
        Patient cluster mappings
    model_type : str, default='ExtraTrees'
        Type of model to train
    model_params : dict, optional
        Model hyperparameters

    Returns
    -------
    fold_results : dict
        Dictionary containing fold metrics
    model : sklearn estimator
        Trained model
    scaler : StandardScaler
        Fitted scaler
    imputer : KNNImputer
        Fitted imputer
    val_patients_list : array-like
        Validation patient IDs
    """
    print(f"\n  Fold {fold + 1}")
    print("  " + "-" * 50)

    # Create validation data WITHOUT augmentation (only sequential)
    print("  Creating validation data (sequential only)...")
    val_samples = []

    # Get original non-augmented data for validation patients
    for patient in val_patients:
        patient_data = preprocessor.df[preprocessor.df['Patient#'] == patient]
        for (pat, dfu), group in patient_data.groupby(['Patient#', 'DFU#']):
            group = group.sort_values('Appt#').reset_index(drop=True)
            if len(group) < 2:
                continue

            patient_cluster = patient_cluster_map.get(patient, 'Unknown')

            # Only create sequential samples for validation
            for target_idx in range(1, len(group)):
                history_indices = tuple(range(target_idx))
                sample = preprocessor._create_sample_from_appointments(
                    group, history_indices, target_idx, patient_cluster, patient_cluster_map
                )
                if sample is not None:
                    val_samples.append(sample)

    import pandas as pd
    val_df = pd.DataFrame(val_samples)

    # Extract previous phases and onset days from the preprocessed samples
    # The preprocessor creates 'Previous_Phase' column in each sample
    if 'Previous_Phase' in val_df.columns:
        val_prev_phases = val_df['Previous_Phase'].values
    else:
        # Fallback: use Initial_Phase or default to 1
        print("    Warning: 'Previous_Phase' not found, using fallback")
        val_prev_phases = val_df.get('Initial_Phase', pd.Series([1] * len(val_df))).values

    # Extract onset days for first appointments (used when prev_phase=-1)
    if 'Onset (Days)' in val_df.columns:
        val_onset_days = val_df['Onset (Days)'].values
    elif 'Onset (Days)' in feature_cols:
        val_onset_days = val_df['Onset (Days)'].values
    else:
        print("    Warning: 'Onset (Days)' not found, using default value of 90 days")
        val_onset_days = np.full(len(val_df), 90.0)

    # Extract days to next appointment for chronicity-aware metrics
    # This is used to determine if stagnation is prolonged for subsequent appointments
    if 'Days_To_Next_Appt' in val_df.columns:
        val_days_to_next = val_df['Days_To_Next_Appt'].values
    else:
        print("    Warning: 'Days_To_Next_Appt' not found, using default value of 14 days")
        val_days_to_next = np.full(len(val_df), 14.0)
    # Handle NaN values
    val_days_to_next = np.nan_to_num(val_days_to_next, nan=14.0)

    # Now get augmented training data
    print("  Creating training data (with augmentation)...")
    train_df = df_processed_full[df_processed_full['Patient#'].isin(train_patients)].copy()

    print(f"  Training samples: {len(train_df)} (augmented), Validation samples: {len(val_df)} (sequential only)")

    # Prepare features
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values.astype(int)
    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values.astype(int)

    # Replace infinite values with NaN before imputation
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    # Clip extreme values to prevent overflow in KNN distance calculations
    # Use percentile-based clipping for each feature
    for col in feature_cols:
        if X_train[col].notna().sum() > 0:
            lower_bound = X_train[col].quantile(0.001)
            upper_bound = X_train[col].quantile(0.999)
            X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
            X_val[col] = X_val[col].clip(lower=lower_bound, upper=upper_bound)

    # Handle missing values
    print("  Handling missing values...")
    imputer = KNNImputer(n_neighbors=min(5, len(X_train) - 1))
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    X_train = pd.DataFrame(X_train_imputed, columns=feature_cols, index=X_train.index)
    X_val = pd.DataFrame(X_val_imputed, columns=feature_cols, index=X_val.index)

    # Scale features
    print("  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Apply resampling
    print("  Applying resampling...")
    X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)

    # Train model
    print("  Training model...")

    # Default model parameters if not provided
    if model_params is None:
        if model_type == 'ExtraTrees':
            model_params = {
                'n_estimators': 437,
                'max_depth': 39,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': None,
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            }
        else:  # GradientBoosting
            model_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'learning_rate': 0.01,
                'min_samples_split': 22,
                'min_samples_leaf': 15,
                'subsample': 0.953,
                'max_features': 'log2',
                'random_state': 42
            }
    else:
        # Ensure random_state and n_jobs are set
        model_params = model_params.copy()
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
        if model_type == 'ExtraTrees' and 'n_jobs' not in model_params:
            model_params['n_jobs'] = -1

    # Create model
    if model_type == 'ExtraTrees':
        model = ExtraTreesClassifier(**model_params)
    else:
        model = GradientBoostingClassifier(**model_params)

    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    training_time = time.time() - start_time

    # Store feature names
    model.feature_names_ = feature_cols

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)

    # Calculate phase-based metrics
    accuracy = accuracy_score(y_val, y_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
    transition_metrics = analyze_transitions(y_val, y_pred)

    combined_score = (balanced_acc + f1_macro) / 2

    # Calculate transition-based metrics (basic)
    transition_based_metrics = calculate_transition_metrics(y_val, y_pred, val_prev_phases)

    # Calculate chronicity-aware transition metrics (evidence-based)
    transition_chronicity_metrics = calculate_transition_metrics_chronicity_aware(
        y_val, y_pred, val_prev_phases, val_days_to_next, val_onset_days
    )

    # Display all metric sets
    print(f"\n    Phase-based Metrics:")
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Balanced Accuracy: {balanced_acc:.4f}")
    print(f"      F1-Macro: {f1_macro:.4f}")
    print(f"      Combined Score: {combined_score:.4f}")

    print(f"\n    Transition-based Metrics (Basic):")
    print(f"      Accuracy: {transition_based_metrics['accuracy']:.4f}")
    print(f"      Balanced Accuracy: {transition_based_metrics['balanced_accuracy']:.4f}")
    print(f"      F1-Macro: {transition_based_metrics['f1_macro']:.4f}")
    print(f"      Combined Score: {transition_based_metrics['combined_score']:.4f}")

    print(f"\n    Transition-based Metrics (Chronicity-Aware - Evidence-Based):")
    print(f"      Accuracy: {transition_chronicity_metrics['accuracy']:.4f}")
    print(f"      Balanced Accuracy: {transition_chronicity_metrics['balanced_accuracy']:.4f}")
    print(f"      F1-Macro: {transition_chronicity_metrics['f1_macro']:.4f}")
    print(f"      Combined Score: {transition_chronicity_metrics['combined_score']:.4f}")
    print(f"      Reclassified: {transition_chronicity_metrics['reclassified_count']} samples ({transition_chronicity_metrics['reclassified_pct']:.1f}%)")

    # Show class distribution
    class_dist = transition_chronicity_metrics['class_distribution']
    class_names = {0: 'Unfavorable', 1: 'Acceptable', 2: 'Favorable'}
    print(f"      Class distribution: ", end='')
    for cls in sorted(class_dist.keys()):
        print(f"{class_names.get(cls, cls)}={class_dist[cls]} ", end='')
    print()

    fold_results = {
        'fold': fold,
        # Phase-based metrics
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'combined_score': combined_score,
        'confusion_matrix': confusion_matrix(y_val, y_pred),
        'training_time': training_time,
        'feature_names': feature_cols,
        'f1_per_class': f1_per_class,
        'transition_metrics': transition_metrics,
        # Transition-based metrics (basic)
        'transition_accuracy': transition_based_metrics['accuracy'],
        'transition_balanced_accuracy': transition_based_metrics['balanced_accuracy'],
        'transition_f1_weighted': transition_based_metrics['f1_weighted'],
        'transition_f1_macro': transition_based_metrics['f1_macro'],
        'transition_combined_score': transition_based_metrics['combined_score'],
        'transition_f1_per_class': transition_based_metrics['f1_per_class'],
        # Transition-based metrics (chronicity-aware)
        'transition_chrono_accuracy': transition_chronicity_metrics['accuracy'],
        'transition_chrono_balanced_accuracy': transition_chronicity_metrics['balanced_accuracy'],
        'transition_chrono_f1_weighted': transition_chronicity_metrics['f1_weighted'],
        'transition_chrono_f1_macro': transition_chronicity_metrics['f1_macro'],
        'transition_chrono_combined_score': transition_chronicity_metrics['combined_score'],
        'transition_chrono_f1_per_class': transition_chronicity_metrics['f1_per_class'],
        'transition_chrono_reclassified_count': transition_chronicity_metrics['reclassified_count'],
        'transition_chrono_reclassified_pct': transition_chronicity_metrics['reclassified_pct'],
        # Raw predictions
        'y_true': y_val,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    return fold_results, model, scaler, imputer, val_patients
