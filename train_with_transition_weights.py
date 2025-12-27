"""
Training Script with Transition-Aware Weighting

This script trains the DFU healing prediction model using sample weights
that account for transition outcomes (Favorable/Acceptable/Unfavorable)
rather than just phase classes (I/P/R).

This addresses the class imbalance in transitions, particularly improving
prediction of rare but clinically important Favorable transitions (I→P, P→R).

Supports both basic and chronicity-aware transition classification:
- Basic: Simple direction-based classification
- Chronicity-Aware: Uses Days_To_Next_Appt with configurable thresholds

Also supports probability calibration using isotonic regression or Platt scaling
to improve the reliability of predicted probabilities.

All parameters can be configured via config/config.yaml. Command-line arguments
override config file values.

Usage:
    # Use config.yaml defaults (recommended)
    python train_with_transition_weights.py

    # Override specific parameters
    python train_with_transition_weights.py --favorable-boost 2.0 --i-threshold 30

    # Disable calibration
    python train_with_transition_weights.py --no-calibration

Configuration (config/config.yaml):
    transition_weighting:
      weight_method: "favorable_boost"
      favorable_boost: 0.5
      unfavorable_boost: 0.1
      chronicity_aware: true
      i_threshold: 42
      p_threshold: 14
      r_threshold: null

    feature_selection:
      use_feature_selection: false
      importance_threshold: 0.95

    calibration:
      apply_calibration: true
      method: "isotonic"  # or "sigmoid"
      use_per_class_calibration: true
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import yaml
import warnings

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.preprocessing import (
    DFUNextAppointmentPreprocessor,
    FlexibleResampler,
    compute_transition_weights,
    compute_transition_weights_chronicity_aware,
    compute_transition_from_phases,
    compute_transition_labels_chronicity_aware,
    compute_focal_weights
)
from src.models.evaluation import (
    convert_to_transition_labels,
    convert_to_transition_labels_chronicity_aware,
    calculate_transition_metrics
)
from src.config.constants import OPTIMIZED_FEATURES


# ============================================================================
# Per-Class Calibration Functions
# ============================================================================

def calibrate_per_class(y_true, y_pred_proba, n_bins=10):
    """
    Create per-class calibration mappings using isotonic regression.

    This creates separate calibration curves for each class, allowing
    different calibration characteristics per transition type.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    n_bins : int
        Number of bins for calibration curve

    Returns
    -------
    List
        List of interpolation functions, one per class
    """
    n_classes = y_pred_proba.shape[1]
    calibration_maps = []

    for class_idx in range(n_classes):
        y_binary = (y_true == class_idx).astype(int)
        y_prob = y_pred_proba[:, class_idx]

        try:
            # Get calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob, n_bins=n_bins, strategy='quantile'
            )

            # Add endpoints to ensure full range coverage
            mean_predicted_value = np.concatenate([[0], mean_predicted_value, [1]])
            fraction_of_positives = np.concatenate([[0], fraction_of_positives, [1]])

            # Create interpolation function
            calibration_map = interp1d(
                mean_predicted_value,
                fraction_of_positives,
                kind='linear',
                bounds_error=False,
                fill_value=(0, 1)
            )
        except ValueError:
            # Fallback to identity mapping if calibration fails
            calibration_map = lambda x: x

        calibration_maps.append(calibration_map)

    return calibration_maps


def apply_calibration_maps(y_pred_proba, calibration_maps):
    """
    Apply per-class calibration mappings to predicted probabilities.

    Parameters
    ----------
    y_pred_proba : np.ndarray
        Original predicted probabilities (n_samples, n_classes)
    calibration_maps : List
        List of calibration functions from calibrate_per_class()

    Returns
    -------
    np.ndarray
        Calibrated probabilities (normalized to sum to 1)
    """
    calibrated_proba = np.zeros_like(y_pred_proba)

    for class_idx, calibration_map in enumerate(calibration_maps):
        calibrated_proba[:, class_idx] = calibration_map(y_pred_proba[:, class_idx])

    # Clip to valid probability range [0, 1] before normalization
    calibrated_proba = np.clip(calibrated_proba, 0, 1)

    # Renormalize to ensure probabilities sum to 1
    row_sums = calibrated_proba.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    calibrated_proba = calibrated_proba / row_sums

    # Final clip to ensure valid probabilities
    calibrated_proba = np.clip(calibrated_proba, 0, 1)

    return calibrated_proba


def temperature_scale_probabilities(y_pred_proba, temperature):
    """
    Apply temperature scaling to predicted probabilities.

    Temperature scaling is a simple post-hoc calibration method that
    divides logits by a learned temperature parameter before softmax.

    Parameters
    ----------
    y_pred_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    temperature : float
        Temperature parameter (>1 = softer, <1 = sharper)

    Returns
    -------
    np.ndarray
        Temperature-scaled probabilities
    """
    # Convert probabilities to logits (inverse softmax)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    logits = np.log(y_pred_proba)

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Convert back to probabilities (softmax)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    scaled_proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return scaled_proba


def optimize_temperature(y_true, y_pred_proba, n_iter=50):
    """
    Find optimal temperature for calibration using grid search.

    Optimizes temperature to minimize negative log-likelihood (NLL).

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    n_iter : int
        Number of iterations for optimization

    Returns
    -------
    float
        Optimal temperature
    """
    from scipy.optimize import minimize_scalar

    def nll_loss(temperature):
        """Negative log-likelihood loss for temperature scaling."""
        if temperature <= 0:
            return float('inf')
        scaled_proba = temperature_scale_probabilities(y_pred_proba, temperature)
        # Compute NLL
        eps = 1e-10
        log_probs = np.log(np.clip(scaled_proba[np.arange(len(y_true)), y_true], eps, 1))
        return -np.mean(log_probs)

    # Grid search + refinement
    best_temp = 1.0
    best_loss = nll_loss(1.0)

    # Coarse grid search
    for temp in np.linspace(0.1, 5.0, n_iter):
        loss = nll_loss(temp)
        if loss < best_loss:
            best_loss = loss
            best_temp = temp

    # Fine-tune with scipy minimize
    try:
        result = minimize_scalar(nll_loss, bounds=(0.1, 5.0), method='bounded')
        if result.fun < best_loss:
            best_temp = result.x
    except:
        pass

    return best_temp


def train_with_focal_loss(X_train, y_train, sample_weights, model_params,
                          gamma=2.0, alpha=None, n_iterations=2):
    """
    Train model with focal loss weighting using iterative refinement.

    Focal loss down-weights easy examples and focuses on hard examples,
    which can improve calibration and minority class performance.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    sample_weights : np.ndarray
        Base sample weights (e.g., from transition weighting)
    model_params : dict
        Model hyperparameters
    gamma : float
        Focal loss focusing parameter (0 = no focal, 2 = standard)
    alpha : np.ndarray or None
        Per-class weights for focal loss
    n_iterations : int
        Number of refinement iterations

    Returns
    -------
    model : ExtraTreesClassifier
        Trained model with focal loss weighting
    final_weights : np.ndarray
        Final combined weights used for training
    """
    n_classes = len(np.unique(y_train))

    if alpha is None:
        # Use balanced class weights as alpha
        class_counts = np.bincount(y_train, minlength=n_classes)
        alpha = len(y_train) / (n_classes * class_counts + 1e-10)
        alpha = alpha / alpha.sum() * n_classes  # Normalize

    # Initial model training
    model = ExtraTreesClassifier(
        n_estimators=model_params.get('n_estimators', 800),
        max_depth=model_params.get('max_depth', 100),
        min_samples_split=model_params.get('min_samples_split', 10),
        min_samples_leaf=model_params.get('min_samples_leaf', 10),
        max_features=model_params.get('max_features', 'sqrt'),
        bootstrap=model_params.get('bootstrap', True),
        class_weight=model_params.get('class_weight'),
        random_state=42,
        n_jobs=-1
    )

    current_weights = sample_weights.copy()

    for iteration in range(n_iterations):
        # Train model with current weights
        model.fit(X_train, y_train, sample_weight=current_weights)

        if iteration < n_iterations - 1:
            # Get predictions for focal weight calculation
            y_pred_proba = model.predict_proba(X_train)

            # Compute focal weights
            focal_weights = compute_focal_weights(y_pred_proba, y_train, gamma=gamma, alpha=alpha)

            # Combine with base sample weights
            # Normalize focal weights to have mean 1
            focal_weights = focal_weights / (focal_weights.mean() + 1e-10)

            # Combine: base weights * focal weights
            current_weights = sample_weights * focal_weights

            # Normalize combined weights
            current_weights = current_weights / (current_weights.mean() + 1e-10)

    return model, current_weights


def load_config():
    """Load configuration from config.yaml."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments with defaults from config.yaml."""
    # Load config first to get defaults
    config = load_config()
    tw_config = config.get('transition_weighting', {})
    fs_config = config.get('feature_selection', {})
    training_config = config.get('training', {})

    parser = argparse.ArgumentParser(
        description='Train with transition-aware weighting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Transition weighting arguments (defaults from config.yaml)
    parser.add_argument('--method', type=str,
                        default=tw_config.get('weight_method', 'favorable_boost'),
                        choices=['balanced', 'favorable_boost', 'clinical'],
                        help='Weighting method')
    parser.add_argument('--favorable-boost', type=float,
                        default=tw_config.get('favorable_boost', 0.5),
                        help='Boost factor for favorable transitions')
    parser.add_argument('--unfavorable-boost', type=float,
                        default=tw_config.get('unfavorable_boost', 0.1),
                        help='Boost factor for unfavorable transitions')
    parser.add_argument('--resampling', type=str,
                        default=training_config.get('resampling_strategy', 'none'),
                        choices=['none', 'oversample', 'undersample', 'combined', 'smote'],
                        help='Resampling strategy to combine with weighting')

    # Chronicity-aware arguments (defaults from config.yaml)
    parser.add_argument('--chronicity-aware', action='store_true',
                        default=tw_config.get('chronicity_aware', True),
                        help='Use chronicity-aware transition classification with Days_To_Next_Appt')
    parser.add_argument('--no-chronicity-aware', action='store_false', dest='chronicity_aware',
                        help='Disable chronicity-aware transition classification')
    parser.add_argument('--i-threshold', type=float,
                        default=tw_config.get('i_threshold', 42),
                        help='Days threshold for I→I to become Unfavorable')
    parser.add_argument('--p-threshold', type=float,
                        default=tw_config.get('p_threshold', 14),
                        help='Days threshold for P→P to become Unfavorable')
    parser.add_argument('--r-threshold', type=float,
                        default=tw_config.get('r_threshold', None),
                        help='Days threshold for R→R to become Unfavorable (None = never)')

    # Feature selection arguments (defaults from config.yaml)
    parser.add_argument('--use-feature-selection', action='store_true',
                        default=fs_config.get('use_feature_selection', False),
                        help='Apply importance-based feature selection')
    parser.add_argument('--no-feature-selection', action='store_false', dest='use_feature_selection',
                        help='Disable feature selection (use all OPTIMIZED_FEATURES)')
    parser.add_argument('--importance-threshold', type=float,
                        default=fs_config.get('importance_threshold', 0.95),
                        help='Cumulative importance threshold (0.95 = top features explaining 95%% importance)')

    # Calibration arguments (defaults from config.yaml)
    cal_config = config.get('calibration', {})
    parser.add_argument('--apply-calibration', action='store_true',
                        default=cal_config.get('apply_calibration', True),
                        help='Apply probability calibration to the model')
    parser.add_argument('--no-calibration', action='store_false', dest='apply_calibration',
                        help='Disable probability calibration')
    parser.add_argument('--calibration-method', type=str,
                        default=cal_config.get('method', 'isotonic'),
                        choices=['isotonic', 'sigmoid'],
                        help='Calibration method: isotonic (non-parametric) or sigmoid (Platt scaling)')
    parser.add_argument('--use-per-class-calibration', action='store_true',
                        default=cal_config.get('use_per_class_calibration', True),
                        help='Apply separate calibration curves per transition type')
    parser.add_argument('--no-per-class-calibration', action='store_false', dest='use_per_class_calibration',
                        help='Disable per-class calibration')

    # Temperature scaling arguments
    parser.add_argument('--use-temperature-scaling', action='store_true',
                        default=cal_config.get('use_temperature_scaling', False),
                        help='Apply temperature scaling for calibration')
    parser.add_argument('--no-temperature-scaling', action='store_false', dest='use_temperature_scaling',
                        help='Disable temperature scaling')

    # Calibration data split arguments
    parser.add_argument('--calibration-split', type=float,
                        default=cal_config.get('calibration_split', 0.0),
                        help='Fraction of training data to hold out for calibration (0.0=use CV, 0.3=30%% held out)')
    parser.add_argument('--use-holdout-calibration', action='store_true',
                        default=cal_config.get('use_holdout_calibration', False),
                        help='Use held-out calibration set instead of CV (enables --calibration-split)')

    # Focal loss arguments
    focal_config = config.get('focal_loss', {})
    parser.add_argument('--use-focal-loss', action='store_true',
                        default=focal_config.get('enabled', False),
                        help='Use focal loss weighting to focus on hard examples')
    parser.add_argument('--no-focal-loss', action='store_false', dest='use_focal_loss',
                        help='Disable focal loss weighting')
    parser.add_argument('--focal-gamma', type=float,
                        default=focal_config.get('gamma', 2.0),
                        help='Focal loss gamma parameter (0=no focus, 2=standard)')
    parser.add_argument('--focal-iterations', type=int,
                        default=focal_config.get('n_iterations', 2),
                        help='Number of focal loss refinement iterations')

    return parser.parse_args(), config


def select_features_by_importance(X_train, y_train, feature_names, importance_threshold=0.5):
    """
    Select features based on cumulative importance threshold.

    Parameters
    ----------
    X_train : array-like
        Training features (already imputed and scaled)
    y_train : array-like
        Training labels
    feature_names : list
        Names of all features
    importance_threshold : float
        Cumulative importance threshold (e.g., 0.5 means select features
        that explain 50% of total importance)

    Returns
    -------
    selected_features : list
        Names of selected features
    importance_df : DataFrame
        Feature importance rankings
    """
    # Train a quick ExtraTrees model to get feature importance
    quick_model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    quick_model.fit(X_train, y_train)

    # Get feature importances
    importances = quick_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    total_importance = importance_df['importance'].sum()
    importance_df['cumulative_importance_pct'] = importance_df['cumulative_importance'] / total_importance

    # Add features until we reach the threshold
    selected_features = []
    for _, row in importance_df.iterrows():
        selected_features.append(row['feature'])
        if row['cumulative_importance_pct'] >= importance_threshold:
            break

    return selected_features, importance_df


def train_fold_with_weights(fold, train_patients, val_patients, df_processed,
                            preprocessor, feature_cols, target_col, resampler,
                            augmentation_type, patient_cluster_map, model_params,
                            weight_method='favorable_boost', favorable_boost=0.5,
                            unfavorable_boost=0.1, chronicity_aware=True,
                            i_threshold=42, p_threshold=14, r_threshold=None,
                            apply_calibration=True, calibration_method='isotonic',
                            use_per_class_calibration=True, use_temperature_scaling=False,
                            use_focal_loss=False, focal_gamma=2.0, focal_iterations=2,
                            use_holdout_calibration=False, calibration_split=0.30):
    """
    Train a single fold with transition-aware sample weights.

    Parameters
    ----------
    chronicity_aware : bool
        If True, use chronicity-aware transition classification with Days_To_Next_Appt
    i_threshold : float
        Days threshold for I→I to become Unfavorable
    p_threshold : float
        Days threshold for P→P to become Unfavorable
    r_threshold : float or None
        Days threshold for R→R to become Unfavorable (None = never)
    apply_calibration : bool
        If True, apply probability calibration using CalibratedClassifierCV
    calibration_method : str
        Calibration method: 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling)
    use_per_class_calibration : bool
        If True, apply separate calibration curves per transition type
    use_temperature_scaling : bool
        If True, apply temperature scaling after other calibration methods
    use_focal_loss : bool
        If True, use focal loss weighting to focus on hard examples
    focal_gamma : float
        Focal loss gamma parameter (0 = no focusing, 2 = standard)
    focal_iterations : int
        Number of iterations for focal loss refinement
    use_holdout_calibration : bool
        If True, use held-out calibration set instead of internal CV
    calibration_split : float
        Fraction of training data to hold out for calibration (default 0.30 = 30%)
    """
    print(f"\n  Fold {fold + 1}:")

    # Get training data (augmented)
    train_mask = df_processed['Patient#'].isin(train_patients)
    train_df = df_processed[train_mask].copy()

    # Get validation data (sequential only)
    val_samples = []
    for patient in val_patients:
        patient_data = preprocessor.df[preprocessor.df['Patient#'] == patient]
        for (pat, dfu), group in patient_data.groupby(['Patient#', 'DFU#']):
            group = group.sort_values('Appt#').reset_index(drop=True)
            if len(group) < 2:
                continue

            patient_cluster = patient_cluster_map.get(patient, 'Unknown')

            for target_idx in range(1, len(group)):
                history_indices = tuple(range(target_idx))
                sample = preprocessor._create_sample_from_appointments(
                    group, history_indices, target_idx, patient_cluster, patient_cluster_map
                )
                if sample is not None:
                    val_samples.append(sample)

    val_df = pd.DataFrame(val_samples)

    # Prepare features
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values.astype(int)

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values.astype(int)

    # Get previous phases for training data
    if 'Previous_Phase' in train_df.columns:
        train_prev_phases = train_df['Previous_Phase'].values
    elif 'Initial_Phase' in train_df.columns:
        train_prev_phases = train_df['Initial_Phase'].values
    else:
        train_prev_phases = np.ones(len(train_df))

    # Get previous phases for validation
    if 'Previous_Phase' in val_df.columns:
        val_prev_phases = val_df['Previous_Phase'].values
    elif 'Initial_Phase' in val_df.columns:
        val_prev_phases = val_df['Initial_Phase'].values
    else:
        val_prev_phases = np.ones(len(val_df))

    # Get Days_To_Next_Appt for chronicity-aware weighting
    if chronicity_aware:
        if 'Days_To_Next_Appt' in train_df.columns:
            train_days_to_next = train_df['Days_To_Next_Appt'].values
        else:
            print(f"    WARNING: Days_To_Next_Appt not found, using default 14 days")
            train_days_to_next = np.full(len(train_df), 14.0)

        if 'Days_To_Next_Appt' in val_df.columns:
            val_days_to_next = val_df['Days_To_Next_Appt'].values
        else:
            val_days_to_next = np.full(len(val_df), 14.0)

        # Replace NaN with median
        train_days_to_next = np.nan_to_num(train_days_to_next, nan=np.nanmedian(train_days_to_next))
        val_days_to_next = np.nan_to_num(val_days_to_next, nan=np.nanmedian(val_days_to_next))

    print(f"    Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # Compute transition-aware weights for training
    if chronicity_aware:
        print(f"    Computing CHRONICITY-AWARE transition weights (method={weight_method})...")
        print(f"    Thresholds: I→I>{i_threshold}d, P→P>{p_threshold}d" +
              (f", R→R>{r_threshold}d" if r_threshold else ""))
        sample_weights, train_transitions = compute_transition_weights_chronicity_aware(
            y_train, train_prev_phases, train_days_to_next,
            method=weight_method,
            favorable_boost=favorable_boost, unfavorable_boost=unfavorable_boost,
            inflammatory_threshold=i_threshold,
            proliferative_threshold=p_threshold,
            remodeling_threshold=r_threshold
        )
    else:
        print(f"    Computing BASIC transition weights (method={weight_method})...")
        sample_weights, train_transitions = compute_transition_weights(
            y_train, train_prev_phases, method=weight_method,
            favorable_boost=favorable_boost, unfavorable_boost=unfavorable_boost
        )

    # Handle missing values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    # Apply resampling if specified (after scaling)
    if resampler.strategy != 'none':
        print(f"    Applying {resampler.strategy} resampling...")

        # Create a combined label that encodes both prev_phase and target_phase
        # This ensures SMOTE generates samples that preserve the transition relationship
        # Combined label = prev_phase * 3 + target_phase (gives 9 unique combinations)
        combined_labels = train_prev_phases.astype(int) * 3 + y_train.astype(int)

        # Resample based on combined labels to balance transitions, not just phases
        X_train_resampled, combined_resampled = resampler.fit_resample(X_train_scaled, combined_labels)

        # Decode the combined labels back to prev_phases and y_train
        prev_phases_resampled = combined_resampled // 3
        y_train_resampled = combined_resampled % 3

        # Now we can compute proper transition weights for the resampled data
        if chronicity_aware:
            # For resampled data, use median days_to_next since we can't preserve original values
            # This is a limitation, but better than ignoring transitions entirely
            median_days = np.nanmedian(train_days_to_next)
            days_resampled = np.full(len(y_train_resampled), median_days)

            sample_weights_resampled, _ = compute_transition_weights_chronicity_aware(
                y_train_resampled, prev_phases_resampled, days_resampled,
                method=weight_method,
                favorable_boost=favorable_boost, unfavorable_boost=unfavorable_boost,
                inflammatory_threshold=i_threshold,
                proliferative_threshold=p_threshold,
                remodeling_threshold=r_threshold
            )
        else:
            sample_weights_resampled, _ = compute_transition_weights(
                y_train_resampled, prev_phases_resampled, method=weight_method,
                favorable_boost=favorable_boost, unfavorable_boost=unfavorable_boost
            )

        print(f"    Resampled: {len(X_train_resampled)} samples")
        print(f"    Resampled weights - min: {sample_weights_resampled.min():.2f}, "
              f"max: {sample_weights_resampled.max():.2f}, mean: {sample_weights_resampled.mean():.2f}")
    else:
        X_train_resampled = X_train_scaled
        y_train_resampled = y_train
        sample_weights_resampled = sample_weights

    # Train model with sample weights (optionally with focal loss)
    if use_focal_loss:
        print(f"    Training with FOCAL LOSS (gamma={focal_gamma}, iterations={focal_iterations})...")
        model, final_weights = train_with_focal_loss(
            X_train_resampled, y_train_resampled, sample_weights_resampled,
            model_params, gamma=focal_gamma, alpha=None, n_iterations=focal_iterations
        )
        print(f"    Focal loss training complete. Final weight stats - "
              f"min: {final_weights.min():.2f}, max: {final_weights.max():.2f}, "
              f"mean: {final_weights.mean():.2f}")
    else:
        model = ExtraTreesClassifier(
            n_estimators=model_params.get('n_estimators', 800),
            max_depth=model_params.get('max_depth', 100),
            min_samples_split=model_params.get('min_samples_split', 10),
            min_samples_leaf=model_params.get('min_samples_leaf', 10),
            max_features=model_params.get('max_features', 'sqrt'),
            bootstrap=model_params.get('bootstrap', True),
            class_weight=model_params.get('class_weight'),
            random_state=42,
            n_jobs=-1
        )
        # Always fit the base model (needed for saving and comparison)
        model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights_resampled)

    # Apply probability calibration if enabled
    if apply_calibration:
        if use_holdout_calibration and calibration_split > 0:
            # Use held-out calibration set (more calibration data = better calibration)
            print(f"    Applying {calibration_method} calibration with {calibration_split*100:.0f}% held-out set...")

            # Split training data: (1-calibration_split) for model, calibration_split for calibration
            X_model, X_calib, y_model, y_calib, w_model, w_calib = train_test_split(
                X_train_resampled, y_train_resampled, sample_weights_resampled,
                test_size=calibration_split, random_state=42, stratify=y_train_resampled
            )
            print(f"    Split: {len(X_model)} model samples, {len(X_calib)} calibration samples")

            # Train base model on the model subset
            base_model_for_calibration = ExtraTreesClassifier(
                n_estimators=model_params.get('n_estimators', 800),
                max_depth=model_params.get('max_depth', 100),
                min_samples_split=model_params.get('min_samples_split', 10),
                min_samples_leaf=model_params.get('min_samples_leaf', 10),
                max_features=model_params.get('max_features', 'sqrt'),
                bootstrap=model_params.get('bootstrap', True),
                class_weight=model_params.get('class_weight'),
                random_state=42,
                n_jobs=-1
            )
            base_model_for_calibration.fit(X_model, y_model, sample_weight=w_model)

            # Use cv='prefit' since model is already trained
            calibrated_model = CalibratedClassifierCV(
                estimator=base_model_for_calibration,
                method=calibration_method,
                cv='prefit'  # Use prefit since model is already trained
            )
            # Fit calibration on the held-out calibration set
            calibrated_model.fit(X_calib, y_calib)
            model_for_prediction = calibrated_model
        else:
            # Use internal 3-fold CV for calibration (original behavior)
            print(f"    Applying {calibration_method} calibration with 3-fold CV...")
            # Create a NEW unfitted model for CalibratedClassifierCV
            # cv=3 does internal cross-validation on training data to learn calibration
            # This avoids the data leakage of fitting calibration on validation data
            base_model_for_calibration = ExtraTreesClassifier(
                n_estimators=model_params.get('n_estimators', 800),
                max_depth=model_params.get('max_depth', 100),
                min_samples_split=model_params.get('min_samples_split', 10),
                min_samples_leaf=model_params.get('min_samples_leaf', 10),
                max_features=model_params.get('max_features', 'sqrt'),
                bootstrap=model_params.get('bootstrap', True),
                class_weight=model_params.get('class_weight'),
                random_state=42,
                n_jobs=-1
            )
            calibrated_model = CalibratedClassifierCV(
                estimator=base_model_for_calibration,
                method=calibration_method,
                cv=3  # Internal 3-fold CV for proper calibration
            )
            # Fit on training data - CalibratedClassifierCV handles CV internally
            calibrated_model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights_resampled)
            model_for_prediction = calibrated_model
    else:
        model_for_prediction = model
        calibrated_model = None

    # Predict using either calibrated or uncalibrated model
    y_pred = model_for_prediction.predict(X_val_scaled)
    y_pred_proba = model_for_prediction.predict_proba(X_val_scaled)

    # Apply per-class calibration if enabled
    if use_per_class_calibration and len(y_train_resampled) > 100:
        print(f"    Applying per-class calibration...")
        try:
            # Create calibration maps from training predictions
            y_train_proba = model_for_prediction.predict_proba(X_train_resampled)
            calibration_maps = calibrate_per_class(y_train_resampled, y_train_proba)
            y_pred_proba = apply_calibration_maps(y_pred_proba, calibration_maps)
        except Exception as e:
            print(f"    Warning: Per-class calibration failed: {e}")

    # Apply temperature scaling if enabled
    learned_temperature = None
    if use_temperature_scaling:
        print(f"    Applying temperature scaling...")
        try:
            # Get training predictions for temperature optimization
            y_train_proba = model_for_prediction.predict_proba(X_train_resampled)
            # Optimize temperature on training data
            learned_temperature = optimize_temperature(y_train_resampled, y_train_proba)
            print(f"    Learned temperature: {learned_temperature:.4f}")
            # Apply temperature scaling to validation predictions
            y_pred_proba = temperature_scale_probabilities(y_pred_proba, learned_temperature)
        except Exception as e:
            print(f"    Warning: Temperature scaling failed: {e}")

    # Phase-based metrics
    phase_accuracy = accuracy_score(y_val, y_pred)
    phase_balanced_acc = balanced_accuracy_score(y_val, y_pred)
    phase_f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

    # Transition-based metrics (basic for comparison)
    y_true_trans_basic = convert_to_transition_labels(y_val, val_prev_phases)
    y_pred_trans_basic = convert_to_transition_labels(y_pred, val_prev_phases)

    trans_accuracy_basic = accuracy_score(y_true_trans_basic, y_pred_trans_basic)
    trans_balanced_acc_basic = balanced_accuracy_score(y_true_trans_basic, y_pred_trans_basic)
    trans_f1_macro_basic = f1_score(y_true_trans_basic, y_pred_trans_basic, average='macro', zero_division=0)
    trans_f1_per_class_basic = f1_score(y_true_trans_basic, y_pred_trans_basic, average=None, zero_division=0)
    trans_precision_basic = precision_score(y_true_trans_basic, y_pred_trans_basic, average='macro', zero_division=0)
    trans_recall_basic = recall_score(y_true_trans_basic, y_pred_trans_basic, average='macro', zero_division=0)

    # Chronicity-aware transition metrics (if applicable)
    if chronicity_aware:
        y_true_trans_chrono = compute_transition_labels_chronicity_aware(
            y_val, val_prev_phases, val_days_to_next,
            i_threshold, p_threshold, r_threshold
        )
        y_pred_trans_chrono = compute_transition_labels_chronicity_aware(
            y_pred, val_prev_phases, val_days_to_next,
            i_threshold, p_threshold, r_threshold
        )

        trans_accuracy_chrono = accuracy_score(y_true_trans_chrono, y_pred_trans_chrono)
        trans_balanced_acc_chrono = balanced_accuracy_score(y_true_trans_chrono, y_pred_trans_chrono)
        trans_f1_macro_chrono = f1_score(y_true_trans_chrono, y_pred_trans_chrono, average='macro', zero_division=0)
        trans_f1_per_class_chrono = f1_score(y_true_trans_chrono, y_pred_trans_chrono, average=None, zero_division=0)
        trans_precision_chrono = precision_score(y_true_trans_chrono, y_pred_trans_chrono, average='macro', zero_division=0)
        trans_recall_chrono = recall_score(y_true_trans_chrono, y_pred_trans_chrono, average='macro', zero_division=0)

    print(f"    Phase - Balanced Acc: {phase_balanced_acc:.4f}, F1-Macro: {phase_f1_macro:.4f}")
    print(f"    Trans (Basic) - Balanced Acc: {trans_balanced_acc_basic:.4f}, F1-Macro: {trans_f1_macro_basic:.4f}")
    print(f"    Trans (Basic) F1 - Unfav: {trans_f1_per_class_basic[0]:.3f}, "
          f"Accept: {trans_f1_per_class_basic[1]:.3f}, Favor: {trans_f1_per_class_basic[2]:.3f}")

    if chronicity_aware:
        print(f"    Trans (Chrono) - Balanced Acc: {trans_balanced_acc_chrono:.4f}, F1-Macro: {trans_f1_macro_chrono:.4f}")
        print(f"    Trans (Chrono) F1 - Unfav: {trans_f1_per_class_chrono[0]:.3f}, "
              f"Accept: {trans_f1_per_class_chrono[1]:.3f}, Favor: {trans_f1_per_class_chrono[2]:.3f}")

    result = {
        'phase_accuracy': phase_accuracy,
        'phase_balanced_accuracy': phase_balanced_acc,
        'phase_f1_macro': phase_f1_macro,
        # Basic transition metrics
        'trans_accuracy': trans_accuracy_basic,
        'trans_balanced_accuracy': trans_balanced_acc_basic,
        'trans_f1_macro': trans_f1_macro_basic,
        'trans_f1_per_class': trans_f1_per_class_basic,
        'trans_precision': trans_precision_basic,
        'trans_recall': trans_recall_basic,
        'y_true': y_val,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'prev_phases': val_prev_phases,
        'learned_temperature': learned_temperature
    }

    # Add chronicity-aware metrics if applicable
    if chronicity_aware:
        result['trans_chrono_accuracy'] = trans_accuracy_chrono
        result['trans_chrono_balanced_accuracy'] = trans_balanced_acc_chrono
        result['trans_chrono_f1_macro'] = trans_f1_macro_chrono
        result['trans_chrono_f1_per_class'] = trans_f1_per_class_chrono
        result['trans_chrono_precision'] = trans_precision_chrono
        result['trans_chrono_recall'] = trans_recall_chrono
        result['days_to_next_appt'] = val_days_to_next

    # Return calibrated model if available, otherwise return base model
    final_model = calibrated_model if calibrated_model is not None else model
    return result, final_model, scaler, imputer, model


def main():
    args, config = parse_args()

    print("="*70)
    print("DFU MODEL TRAINING WITH TRANSITION-AWARE WEIGHTING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    CSV_PATH = config['data']['csv_path']
    SAVE_DIR = Path(config['data']['save_dir'])
    SAVE_DIR.mkdir(exist_ok=True)

    N_FOLDS = config['training']['n_folds']
    N_PATIENT_CLUSTERS = config['training']['n_patient_clusters']
    AUGMENTATION_TYPE = config['training']['augmentation_type']

    model_params = config['model']['extratrees']

    print(f"\nConfiguration (from config.yaml + command-line overrides):")
    print(f"  Weighting method: {args.method}")
    print(f"  Favorable boost: {args.favorable_boost}")
    print(f"  Unfavorable boost: {args.unfavorable_boost}")
    print(f"  Resampling: {args.resampling}")
    print(f"  Augmentation: {AUGMENTATION_TYPE}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Chronicity-aware: {args.chronicity_aware}")
    if args.chronicity_aware:
        print(f"    I→I threshold: {args.i_threshold} days")
        print(f"    P→P threshold: {args.p_threshold} days")
        print(f"    R→R threshold: {args.r_threshold if args.r_threshold else 'None (always acceptable)'}")
    print(f"  Feature selection: {args.use_feature_selection}")
    if args.use_feature_selection:
        print(f"    Importance threshold: {args.importance_threshold}")
    print(f"  Probability calibration: {args.apply_calibration}")
    if args.apply_calibration:
        print(f"    Calibration method: {args.calibration_method}")
        if args.use_holdout_calibration:
            print(f"    Using held-out calibration: {args.calibration_split*100:.0f}% of training data")
        else:
            print(f"    Using internal 3-fold CV for calibration")
    print(f"  Per-class calibration: {args.use_per_class_calibration}")
    print(f"  Temperature scaling: {args.use_temperature_scaling}")
    print(f"  Focal loss: {args.use_focal_loss}")
    if args.use_focal_loss:
        print(f"    Focal gamma: {args.focal_gamma}")
        print(f"    Focal iterations: {args.focal_iterations}")

    # Initialize preprocessor
    print("\n1. Preprocessing data...")
    preprocessor = DFUNextAppointmentPreprocessor(CSV_PATH)
    df = preprocessor.initial_cleaning()
    df = preprocessor.convert_categorical_to_numeric()
    df = preprocessor.create_temporal_features()

    df_processed, patient_cluster_map, _ = preprocessor.create_next_appointment_dataset_with_augmentation(
        n_patient_clusters=N_PATIENT_CLUSTERS,
        augmentation_type=AUGMENTATION_TYPE
    )

    # Get initial feature pool
    available_features = [f for f in OPTIMIZED_FEATURES if f in df_processed.columns]
    if len(available_features) == 0:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['Patient#', 'DFU#', 'Appt#', 'Next_Healing_Phase', 'ID']
        available_features = [c for c in numeric_cols if c not in exclude]

    target_col = 'Next_Healing_Phase'

    # Apply feature selection if enabled
    if args.use_feature_selection:
        print(f"\n2. Applying feature selection (threshold={args.importance_threshold})...")

        # Prepare data for feature selection
        X_full = df_processed[available_features].copy()
        y_full = df_processed[target_col].values.astype(int)

        # Handle missing values for feature selection
        X_full = X_full.replace([np.inf, -np.inf], np.nan)
        imputer_fs = SimpleImputer(strategy='median')
        X_full_imputed = imputer_fs.fit_transform(X_full)

        # Select features by importance
        feature_cols, importance_df = select_features_by_importance(
            X_full_imputed, y_full, available_features,
            importance_threshold=args.importance_threshold
        )

        print(f"  Selected {len(feature_cols)} features (from {len(available_features)} available)")
        print(f"  Top 10 features by importance:")
        for i, row in importance_df.head(10).iterrows():
            marker = "*" if row['feature'] in feature_cols else " "
            print(f"    {marker} {row['feature']}: {row['importance']:.4f} ({row['cumulative_importance_pct']*100:.1f}% cumulative)")
    else:
        feature_cols = available_features
        importance_df = None
        print(f"\n2. Using all {len(feature_cols)} OPTIMIZED_FEATURES (feature selection disabled)")

    # Initialize resampler
    resampler = FlexibleResampler(strategy=args.resampling)

    # Cross-validation
    print("\n3. Running cross-validation...")
    unique_patients = df_processed['Patient#'].unique()
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_results = []
    all_models = []
    all_base_models = []  # Store uncalibrated models for comparison
    all_scalers = []
    all_imputers = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(unique_patients)):
        train_patients = unique_patients[train_idx]
        val_patients = unique_patients[val_idx]

        result, model, scaler, imputer, base_model = train_fold_with_weights(
            fold, train_patients, val_patients, df_processed,
            preprocessor, feature_cols, target_col, resampler,
            AUGMENTATION_TYPE, patient_cluster_map, model_params,
            weight_method=args.method,
            favorable_boost=args.favorable_boost,
            unfavorable_boost=args.unfavorable_boost,
            chronicity_aware=args.chronicity_aware,
            i_threshold=args.i_threshold,
            p_threshold=args.p_threshold,
            r_threshold=args.r_threshold,
            apply_calibration=args.apply_calibration,
            calibration_method=args.calibration_method,
            use_per_class_calibration=args.use_per_class_calibration,
            use_temperature_scaling=args.use_temperature_scaling,
            use_focal_loss=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
            focal_iterations=args.focal_iterations,
            use_holdout_calibration=args.use_holdout_calibration,
            calibration_split=args.calibration_split
        )

        all_results.append(result)
        all_models.append(model)
        all_base_models.append(base_model)
        all_scalers.append(scaler)
        all_imputers.append(imputer)

    # Aggregate results
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)

    # Phase-based
    phase_bal_accs = [r['phase_balanced_accuracy'] for r in all_results]
    phase_f1s = [r['phase_f1_macro'] for r in all_results]

    print(f"\nPhase-based Metrics:")
    print(f"  Balanced Accuracy: {np.mean(phase_bal_accs):.4f} ± {np.std(phase_bal_accs):.4f}")
    print(f"  F1-Macro:          {np.mean(phase_f1s):.4f} ± {np.std(phase_f1s):.4f}")

    # Transition-based
    trans_bal_accs = [r['trans_balanced_accuracy'] for r in all_results]
    trans_f1s = [r['trans_f1_macro'] for r in all_results]
    trans_precisions = [r['trans_precision'] for r in all_results]
    trans_recalls = [r['trans_recall'] for r in all_results]

    print(f"\nTransition-based Metrics (Basic):")
    print(f"  Balanced Accuracy: {np.mean(trans_bal_accs):.4f} ± {np.std(trans_bal_accs):.4f}")
    print(f"  F1-Macro:          {np.mean(trans_f1s):.4f} ± {np.std(trans_f1s):.4f}")
    print(f"  Precision (Macro): {np.mean(trans_precisions):.4f} ± {np.std(trans_precisions):.4f}")
    print(f"  Recall (Macro):    {np.mean(trans_recalls):.4f} ± {np.std(trans_recalls):.4f}")

    # Per-class F1 (Basic)
    all_f1_per_class = np.array([r['trans_f1_per_class'] for r in all_results])
    mean_f1_per_class = np.mean(all_f1_per_class, axis=0)
    std_f1_per_class = np.std(all_f1_per_class, axis=0)

    print(f"\nPer-class F1 Scores (Basic):")
    print(f"  Unfavorable: {mean_f1_per_class[0]:.4f} ± {std_f1_per_class[0]:.4f}")
    print(f"  Acceptable:  {mean_f1_per_class[1]:.4f} ± {std_f1_per_class[1]:.4f}")
    print(f"  Favorable:   {mean_f1_per_class[2]:.4f} ± {std_f1_per_class[2]:.4f}")

    # Chronicity-aware transition metrics (if applicable)
    if args.chronicity_aware:
        chrono_bal_accs = [r['trans_chrono_balanced_accuracy'] for r in all_results]
        chrono_f1s = [r['trans_chrono_f1_macro'] for r in all_results]
        chrono_precisions = [r['trans_chrono_precision'] for r in all_results]
        chrono_recalls = [r['trans_chrono_recall'] for r in all_results]

        print(f"\nTransition-based Metrics (Chronicity-Aware):")
        print(f"  Thresholds: I→I>{args.i_threshold}d, P→P>{args.p_threshold}d")
        print(f"  Balanced Accuracy: {np.mean(chrono_bal_accs):.4f} ± {np.std(chrono_bal_accs):.4f}")
        print(f"  F1-Macro:          {np.mean(chrono_f1s):.4f} ± {np.std(chrono_f1s):.4f}")
        print(f"  Precision (Macro): {np.mean(chrono_precisions):.4f} ± {np.std(chrono_precisions):.4f}")
        print(f"  Recall (Macro):    {np.mean(chrono_recalls):.4f} ± {np.std(chrono_recalls):.4f}")

        # Per-class F1 (Chronicity-aware)
        all_chrono_f1_per_class = np.array([r['trans_chrono_f1_per_class'] for r in all_results])
        mean_chrono_f1_per_class = np.mean(all_chrono_f1_per_class, axis=0)
        std_chrono_f1_per_class = np.std(all_chrono_f1_per_class, axis=0)

        print(f"\nPer-class F1 Scores (Chronicity-Aware):")
        print(f"  Unfavorable: {mean_chrono_f1_per_class[0]:.4f} ± {std_chrono_f1_per_class[0]:.4f}")
        print(f"  Acceptable:  {mean_chrono_f1_per_class[1]:.4f} ± {std_chrono_f1_per_class[1]:.4f}")
        print(f"  Favorable:   {mean_chrono_f1_per_class[2]:.4f} ± {std_chrono_f1_per_class[2]:.4f}")

    # Save best model
    print("\n" + "="*70)
    print("SAVING BEST MODEL")
    print("="*70)

    best_idx = np.argmax([r['trans_f1_macro'] for r in all_results])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the best model (calibrated if calibration was applied)
    model_path = SAVE_DIR / f"transition_weighted_model_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(all_models[best_idx], f)
    print(f"  Model saved: {model_path}")

    # Also save the base (uncalibrated) model for reference
    base_model_path = SAVE_DIR / f"transition_weighted_model_base_{timestamp}.pkl"
    with open(base_model_path, 'wb') as f:
        pickle.dump(all_base_models[best_idx], f)
    if args.apply_calibration:
        print(f"  Base (uncalibrated) model saved: {base_model_path}")

    scaler_path = SAVE_DIR / f"scaler_tw_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(all_scalers[best_idx], f)

    imputer_path = SAVE_DIR / f"imputer_tw_{timestamp}.pkl"
    with open(imputer_path, 'wb') as f:
        pickle.dump(all_imputers[best_idx], f)

    feature_path = SAVE_DIR / f"feature_names_tw_{timestamp}.pkl"
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_cols, f)

    # Save preprocessing parameters for calibration analysis
    preprocessing_params = {
        'n_patient_clusters': N_PATIENT_CLUSTERS,
        'augmentation_type': AUGMENTATION_TYPE,
        'target_col': 'Next_Healing_Phase',
    }
    preprocessing_path = SAVE_DIR / f"preprocessing_params_tw_{timestamp}.pkl"
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing_params, f)

    # Save training configuration metadata
    # Collect learned temperatures across folds
    learned_temperatures = [r.get('learned_temperature') for r in all_results if r.get('learned_temperature') is not None]
    avg_temperature = np.mean(learned_temperatures) if learned_temperatures else None

    training_config = {
        'method': args.method,
        'favorable_boost': args.favorable_boost,
        'unfavorable_boost': args.unfavorable_boost,
        'resampling': args.resampling,
        'augmentation': AUGMENTATION_TYPE,
        'chronicity_aware': args.chronicity_aware,
        'i_threshold': args.i_threshold if args.chronicity_aware else None,
        'p_threshold': args.p_threshold if args.chronicity_aware else None,
        'r_threshold': args.r_threshold if args.chronicity_aware else None,
        'use_feature_selection': args.use_feature_selection,
        'importance_threshold': args.importance_threshold if args.use_feature_selection else None,
        'n_features_used': len(feature_cols),
        'n_folds': N_FOLDS,
        'timestamp': timestamp,
        'model_params': model_params,
        'apply_calibration': args.apply_calibration,
        'calibration_method': args.calibration_method if args.apply_calibration else None,
        'use_per_class_calibration': args.use_per_class_calibration,
        'use_temperature_scaling': args.use_temperature_scaling,
        'learned_temperature': avg_temperature,
        'use_focal_loss': args.use_focal_loss,
        'focal_gamma': args.focal_gamma if args.use_focal_loss else None,
        'focal_iterations': args.focal_iterations if args.use_focal_loss else None,
        'use_holdout_calibration': args.use_holdout_calibration,
        'calibration_split': args.calibration_split if args.use_holdout_calibration else None,
    }
    config_path = SAVE_DIR / f"training_config_tw_{timestamp}.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(training_config, f)

    # Save latest model metadata as JSON for easy access
    import json
    latest_metadata = {
        'timestamp': timestamp,
        'model_path': str(model_path),
        'base_model_path': str(base_model_path),
        'scaler_path': str(scaler_path),
        'imputer_path': str(imputer_path),
        'feature_path': str(feature_path),
        'preprocessing_path': str(preprocessing_path),
        'config_path': str(config_path),
        'chronicity_aware': args.chronicity_aware,
        'method': args.method,
        'use_feature_selection': args.use_feature_selection,
        'n_features_used': len(feature_cols),
        'apply_calibration': args.apply_calibration,
        'calibration_method': args.calibration_method if args.apply_calibration else None,
        'use_per_class_calibration': args.use_per_class_calibration,
        'use_temperature_scaling': args.use_temperature_scaling,
        'learned_temperature': float(avg_temperature) if avg_temperature is not None else None,
        'use_focal_loss': args.use_focal_loss,
        'focal_gamma': args.focal_gamma if args.use_focal_loss else None,
        'use_holdout_calibration': args.use_holdout_calibration,
        'calibration_split': args.calibration_split if args.use_holdout_calibration else None,
    }
    latest_path = SAVE_DIR / "latest_tw_model_metadata.json"
    with open(latest_path, 'w') as f:
        json.dump(latest_metadata, f, indent=2)
    print(f"  Metadata saved: {latest_path}")

    # Save results summary
    results_summary = {
        'method': args.method,
        'favorable_boost': args.favorable_boost,
        'unfavorable_boost': args.unfavorable_boost,
        'resampling': args.resampling,
        'augmentation': AUGMENTATION_TYPE,
        'chronicity_aware': args.chronicity_aware,
        'i_threshold': args.i_threshold if args.chronicity_aware else None,
        'p_threshold': args.p_threshold if args.chronicity_aware else None,
        'r_threshold': args.r_threshold if args.chronicity_aware else None,
        'use_feature_selection': args.use_feature_selection,
        'importance_threshold': args.importance_threshold if args.use_feature_selection else None,
        'n_features_used': len(feature_cols),
        'model_params': model_params,
        'apply_calibration': args.apply_calibration,
        'calibration_method': args.calibration_method if args.apply_calibration else None,
        'use_per_class_calibration': args.use_per_class_calibration,
        'use_temperature_scaling': args.use_temperature_scaling,
        'learned_temperature': float(avg_temperature) if avg_temperature is not None else None,
        'use_focal_loss': args.use_focal_loss,
        'focal_gamma': args.focal_gamma if args.use_focal_loss else None,
        'focal_iterations': args.focal_iterations if args.use_focal_loss else None,
        'phase_balanced_accuracy': f"{np.mean(phase_bal_accs):.4f} ± {np.std(phase_bal_accs):.4f}",
        'phase_f1_macro': f"{np.mean(phase_f1s):.4f} ± {np.std(phase_f1s):.4f}",
        'trans_basic_balanced_accuracy': f"{np.mean(trans_bal_accs):.4f} ± {np.std(trans_bal_accs):.4f}",
        'trans_basic_f1_macro': f"{np.mean(trans_f1s):.4f} ± {np.std(trans_f1s):.4f}",
        'trans_basic_f1_unfavorable': f"{mean_f1_per_class[0]:.4f} ± {std_f1_per_class[0]:.4f}",
        'trans_basic_f1_acceptable': f"{mean_f1_per_class[1]:.4f} ± {std_f1_per_class[1]:.4f}",
        'trans_basic_f1_favorable': f"{mean_f1_per_class[2]:.4f} ± {std_f1_per_class[2]:.4f}",
    }

    # Add chronicity-aware metrics to summary
    if args.chronicity_aware:
        results_summary['trans_chrono_balanced_accuracy'] = f"{np.mean(chrono_bal_accs):.4f} ± {np.std(chrono_bal_accs):.4f}"
        results_summary['trans_chrono_f1_macro'] = f"{np.mean(chrono_f1s):.4f} ± {np.std(chrono_f1s):.4f}"
        results_summary['trans_chrono_f1_unfavorable'] = f"{mean_chrono_f1_per_class[0]:.4f} ± {std_chrono_f1_per_class[0]:.4f}"
        results_summary['trans_chrono_f1_acceptable'] = f"{mean_chrono_f1_per_class[1]:.4f} ± {std_chrono_f1_per_class[1]:.4f}"
        results_summary['trans_chrono_f1_favorable'] = f"{mean_chrono_f1_per_class[2]:.4f} ± {std_chrono_f1_per_class[2]:.4f}"

    summary_path = SAVE_DIR / f"training_summary_tw_{timestamp}.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)

    print(f"\n✅ Training complete!")
    print(f"   Results saved to: {SAVE_DIR}")

    return results_summary


if __name__ == "__main__":
    main()
