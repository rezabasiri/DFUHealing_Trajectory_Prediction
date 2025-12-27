#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Search with Dynamic Feature Selection
(Direct Transition Label Training)

DIRECT TRANSITION TRAINING:
- Trains DIRECTLY on transition labels (Unfavorable/Acceptable/Favorable)
- No conversion from phase labels needed - cleaner calibration
- Uses fixed thresholds: I=21 days, P=42 days

KEY FEATURES:
- Dynamic feature selection (importance_threshold or top_n methods)
- Direct transition label training (no phase→transition conversion)
- Onset value correction (median by phase at first appointment)
- Temperature scaling for post-hoc calibration
- Focal loss training for hard examples
- Cumulative phase duration for chronicity checks
- Per-class calibration with isotonic regression

FIXED THRESHOLDS (not searched):
- i_threshold = 21 days (inflammatory phase chronicity)
- p_threshold = 42 days (proliferative phase chronicity, 6 weeks)

Usage:
    python optuna_search_feature_selection.py --n-trials 500 --optimize-for combined_score
"""

import sys
import os
import argparse
import warnings
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from collections import Counter
import numpy as np
import pandas as pd
import yaml

# Optuna imports
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.special import softmax

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.preprocessing import (
    DFUNextAppointmentPreprocessor,
    FlexibleResampler,
    compute_transition_weights_chronicity_aware,
    compute_transition_labels_chronicity_aware,
    compute_focal_weights
)
from src.config.constants import (
    CLINICAL_ESSENTIAL_FEATURES,
    TREATMENT_CONSISTENCY_NUMERIC,
    FEATURES_TO_REMOVE,  # Columns already dropped by preprocessor
)


# ============================================================================
# Essential Features (always included in selection)
# ============================================================================

# Features that should ALWAYS be included regardless of feature selection
ESSENTIAL_FEATURES = CLINICAL_ESSENTIAL_FEATURES + ['Onset (Days)']

# Columns to EXCLUDE from feature pool (minimal list - let feature selection decide the rest)
# Note: FEATURES_TO_REMOVE are already dropped by preprocessor, included here for safety
EXCLUDE_COLUMNS = list(set(
    FEATURES_TO_REMOVE + [  # Already dropped: 'Healing Phase', 'Phase Confidence (%)', 'Type of Pain2', 'Type of Pain_Grouped2', 'ID'
        # Identifiers
        'Patient#', 'DFU#', 'Appt#',
        # Target-related (would cause leakage)
        'Next_Healing_Phase', 'Healing_Phase',  # Engineered target columns
        'Healing Phase Abs',  # Used for onset/transition calc, NOT as feature
        'Previous_Phase',  # DATA LEAKAGE: directly used in transition label computation
        # Free-text columns (user-specified exclusions)
        'Type of Pain', 'Type of Pain Grouped',
        'Dressing',  # Use 'Dressing Grouped' instead (ordinal encoded)
    ]
))


# ============================================================================
# FIXED THRESHOLDS FOR TRANSITION CLASSIFICATION
# ============================================================================
# These are clinically-determined thresholds for chronicity classification.
# They use CUMULATIVE phase duration (total time in phase across consecutive
# same-phase appointments), not just days between two appointments.

I_THRESHOLD = 21   # Days in Inflammatory phase before I→I becomes Unfavorable
P_THRESHOLD = 42   # Days in Proliferative phase before P→P becomes Unfavorable (6 weeks)
R_THRESHOLD = None # Remodeling threshold (None = R→R is always Acceptable)


# ============================================================================
# Onset Value Correction Functions
# ============================================================================

def compute_median_onset_by_first_phase(df: pd.DataFrame, preprocessor) -> Dict[int, float]:
    """Compute median onset values grouped by the phase at the first appointment (Appt 0).

    Uses 'Healing Phase Abs' column (I/P/R) which is kept in preprocessor.df.
    Note: 'Healing Phase Abs' is NOT used as a feature (explicitly in EXCLUDE_FROM_FEATURES
    and NON_NUMERIC_COLUMNS in constants.py, filtered out via EXCLUDE_COLUMNS).
    """
    raw_df = preprocessor.df
    first_appts = []

    # Phase string to int mapping (I=0, P=1, R=2)
    def parse_phase(phase_val):
        if pd.isna(phase_val):
            return np.nan
        if isinstance(phase_val, (int, float)):
            return int(phase_val)
        phase_str = str(phase_val).strip().upper()
        if phase_str.startswith('I'):
            return 0
        elif phase_str.startswith('P'):
            return 1
        elif phase_str.startswith('R'):
            return 2
        return np.nan

    for (patient, dfu), group in raw_df.groupby(['Patient#', 'DFU#']):
        group_sorted = group.sort_values('Appt#')
        first_row = group_sorted.iloc[0]
        onset = first_row.get('Onset (Days)', np.nan)
        if pd.isna(onset):
            onset = first_row.get('Days_Since_Onset', np.nan)
        # Use 'Healing Phase Abs' (kept in preprocessor.df, contains I/P/R strings)
        phase = first_row.get('Healing Phase Abs', np.nan)
        phase = parse_phase(phase)
        if not pd.isna(phase) and not pd.isna(onset):
            first_appts.append({'phase_at_appt0': int(phase), 'onset_days': float(onset)})

    if not first_appts:
        print("  Warning: No first appointments found for onset median computation")
        return {0: 90.0, 1: 90.0, 2: 90.0}

    first_df = pd.DataFrame(first_appts)
    median_onset_by_phase = {}
    for phase in [0, 1, 2]:
        phase_onsets = first_df[first_df['phase_at_appt0'] == phase]['onset_days']
        if len(phase_onsets) > 0:
            median_onset_by_phase[phase] = float(phase_onsets.median())
        else:
            median_onset_by_phase[phase] = float(first_df['onset_days'].median())
    return median_onset_by_phase


def get_phase_at_first_appt_map(preprocessor) -> Dict[Tuple[Any, Any], int]:
    """Create a mapping from (Patient#, DFU#) to the phase at their first appointment.

    Uses 'Healing Phase Abs' column (I/P/R) which is kept in preprocessor.df.
    """
    raw_df = preprocessor.df
    phase_map = {}

    # Phase string to int mapping (I=0, P=1, R=2)
    def parse_phase(phase_val):
        if pd.isna(phase_val):
            return np.nan
        if isinstance(phase_val, (int, float)):
            return int(phase_val)
        phase_str = str(phase_val).strip().upper()
        if phase_str.startswith('I'):
            return 0
        elif phase_str.startswith('P'):
            return 1
        elif phase_str.startswith('R'):
            return 2
        return np.nan

    for (patient, dfu), group in raw_df.groupby(['Patient#', 'DFU#']):
        group_sorted = group.sort_values('Appt#')
        first_row = group_sorted.iloc[0]
        # Use 'Healing Phase Abs' (kept in preprocessor.df, contains I/P/R strings)
        phase = first_row.get('Healing Phase Abs', np.nan)
        phase = parse_phase(phase)
        if not pd.isna(phase):
            phase_map[(patient, dfu)] = int(phase)
    return phase_map


def replace_onset_with_median(df: pd.DataFrame, preprocessor,
                               median_onset_by_phase: Dict[int, float],
                               phase_at_first_appt_map: Dict[Tuple[Any, Any], int],
                               silent: bool = False,
                               outlier_std: float = 2.0) -> pd.DataFrame:
    """Replace missing and outlier onset values with phase-based medians.

    Only replaces:
    1. Missing (NaN) onset values
    2. Outliers outside `outlier_std` standard deviations from the mean

    Args:
        df: DataFrame with onset values
        preprocessor: Preprocessor object
        median_onset_by_phase: Dict mapping phase (0,1,2) to median onset
        phase_at_first_appt_map: Dict mapping (Patient#, DFU#) to first phase
        silent: If True, suppress output
        outlier_std: Number of std deviations for outlier detection (default 2.0)
    """
    df = df.copy()
    onset_col = 'Onset (Days)' if 'Onset (Days)' in df.columns else 'Days_Since_Onset'
    if onset_col not in df.columns:
        return df

    original_onset = df[onset_col].copy()
    new_onset = original_onset.copy()

    # Compute outlier bounds (using valid values only)
    valid_onsets = original_onset.dropna()
    if len(valid_onsets) > 0:
        onset_mean = valid_onsets.mean()
        onset_std = valid_onsets.std()
        lower_bound = onset_mean - outlier_std * onset_std
        upper_bound = onset_mean + outlier_std * onset_std
    else:
        lower_bound, upper_bound = -np.inf, np.inf

    n_missing = 0
    n_outliers = 0

    for idx in df.index:
        patient = df.loc[idx, 'Patient#']
        dfu = df.loc[idx, 'DFU#']
        phase_at_first = phase_at_first_appt_map.get((patient, dfu), None)
        current_onset = original_onset.loc[idx]

        if phase_at_first is not None and phase_at_first in median_onset_by_phase:
            median_val = median_onset_by_phase[phase_at_first]

            # Replace if missing
            if pd.isna(current_onset):
                new_onset.loc[idx] = median_val
                n_missing += 1
            # Replace if outlier (outside 2 std)
            elif current_onset < lower_bound or current_onset > upper_bound:
                new_onset.loc[idx] = median_val
                n_outliers += 1

    df[onset_col] = new_onset

    if not silent:
        print(f"  Onset correction: {n_missing} missing + {n_outliers} outliers replaced "
              f"(bounds: {lower_bound:.0f}-{upper_bound:.0f} days)")
    return df


# ============================================================================
# Utility: Suppress verbose preprocessing output
# ============================================================================

@contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress stdout."""
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


# ============================================================================
# Calibration Functions
# ============================================================================

def train_with_calibration(X_train: np.ndarray, y_train: np.ndarray,
                           sample_weights: np.ndarray, config: Dict,
                           random_state: int = 42, n_jobs: int = -1,
                           calibration_method: str = 'isotonic') -> CalibratedClassifierCV:
    """Train model with proper calibration using a held-out calibration set."""
    X_train_base, X_cal, y_train_base, y_cal, weights_base, _ = train_test_split(
        X_train, y_train, sample_weights,
        test_size=0.2, random_state=random_state, stratify=y_train
    )

    base_model = ExtraTreesClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        bootstrap=config['bootstrap'],
        class_weight=config['class_weight'],
        criterion=config.get('criterion', 'gini'),
        ccp_alpha=config.get('ccp_alpha', 0.0),
        min_weight_fraction_leaf=config.get('min_weight_fraction_leaf', 0.0),
        max_samples=config.get('max_samples') if config['bootstrap'] else None,
        random_state=random_state,
        n_jobs=n_jobs
    )
    base_model.fit(X_train_base, y_train_base, sample_weight=weights_base)

    calibrated_model = CalibratedClassifierCV(
        base_model, method=calibration_method, cv='prefit'
    )
    calibrated_model.fit(X_cal, y_cal)

    return calibrated_model


# ============================================================================
# Temperature Scaling Functions
# ============================================================================

def apply_temperature_scaling(y_pred_proba: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to predicted probabilities."""
    if temperature <= 0:
        temperature = 1.0
    eps = 1e-10
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    logits = np.log(y_pred_proba)
    scaled_logits = logits / temperature
    scaled_proba = softmax(scaled_logits, axis=1)
    return scaled_proba


def optimize_temperature_nll(y_true: np.ndarray, y_pred_proba: np.ndarray,
                              temp_range: Tuple[float, float] = (0.1, 5.0)) -> float:
    """Find optimal temperature using negative log-likelihood minimization."""
    def nll_loss(temp):
        scaled_proba = apply_temperature_scaling(y_pred_proba, temp)
        eps = 1e-10
        scaled_proba = np.clip(scaled_proba, eps, 1 - eps)
        n_samples = len(y_true)
        nll = 0.0
        for i in range(n_samples):
            nll -= np.log(scaled_proba[i, int(y_true[i])])
        return nll / n_samples

    try:
        result = minimize_scalar(nll_loss, bounds=temp_range, method='bounded')
        return result.x if result.success else 1.0
    except Exception:
        return 1.0


def optimize_temperature_ece(y_true: np.ndarray, y_pred_proba: np.ndarray,
                              temp_range: Tuple[float, float] = (0.1, 5.0),
                              n_bins: int = 10) -> float:
    """Find optimal temperature using ECE minimization."""
    def ece_loss(temp):
        scaled_proba = apply_temperature_scaling(y_pred_proba, temp)
        n_classes = scaled_proba.shape[1]
        ece_scores = []
        for class_idx in range(n_classes):
            y_binary = (y_true == class_idx).astype(int)
            y_prob = scaled_proba[:, class_idx]
            if y_binary.sum() == 0:
                continue
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, y_prob, n_bins=n_bins, strategy='uniform'
                )
                ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                ece_scores.append(ece)
            except ValueError:
                pass
        return np.mean(ece_scores) if ece_scores else 1.0

    try:
        result = minimize_scalar(ece_loss, bounds=temp_range, method='bounded')
        return result.x if result.success else 1.0
    except Exception:
        return 1.0


# ============================================================================
# Focal Loss Training
# ============================================================================

def train_with_focal_loss(X_train: np.ndarray, y_train: np.ndarray,
                          sample_weights: np.ndarray, config: Dict,
                          random_state: int = 42, n_jobs: int = -1,
                          focal_gamma: float = 2.0, focal_iterations: int = 2) -> ExtraTreesClassifier:
    """Train ExtraTreesClassifier with iterative focal loss weighting."""
    n_classes = len(np.unique(y_train))
    class_counts = np.bincount(y_train, minlength=n_classes)
    alpha = len(y_train) / (n_classes * class_counts + 1e-10)
    alpha = alpha / alpha.sum() * n_classes
    current_weights = sample_weights.copy()

    for iteration in range(focal_iterations):
        model = ExtraTreesClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'],
            bootstrap=config['bootstrap'],
            class_weight=config['class_weight'],
            criterion=config.get('criterion', 'gini'),
            ccp_alpha=config.get('ccp_alpha', 0.0),
            min_weight_fraction_leaf=config.get('min_weight_fraction_leaf', 0.0),
            max_samples=config.get('max_samples') if config['bootstrap'] else None,
            random_state=random_state,
            n_jobs=n_jobs
        )
        model.fit(X_train, y_train, sample_weight=current_weights)

        if iteration < focal_iterations - 1:
            y_pred_proba = model.predict_proba(X_train)
            if y_pred_proba.shape[1] < n_classes:
                full_proba = np.zeros((len(y_pred_proba), n_classes))
                for i, cls in enumerate(model.classes_):
                    full_proba[:, int(cls)] = y_pred_proba[:, i]
                y_pred_proba = full_proba
            focal_weights = compute_focal_weights(y_pred_proba, y_train, gamma=focal_gamma, alpha=alpha)
            focal_weights = focal_weights / (focal_weights.mean() + 1e-10)
            current_weights = sample_weights * focal_weights
            current_weights = current_weights / (current_weights.mean() + 1e-10)

    return model


def train_with_focal_and_calibration(X_train: np.ndarray, y_train: np.ndarray,
                                      sample_weights: np.ndarray, config: Dict,
                                      random_state: int = 42, n_jobs: int = -1,
                                      focal_gamma: float = 2.0, focal_iterations: int = 2,
                                      calibration_method: str = 'isotonic') -> CalibratedClassifierCV:
    """Train model with focal loss AND held-out calibration."""
    X_train_base, X_cal, y_train_base, y_cal, weights_base, _ = train_test_split(
        X_train, y_train, sample_weights,
        test_size=0.2, random_state=random_state, stratify=y_train
    )
    base_model = train_with_focal_loss(
        X_train_base, y_train_base, weights_base, config,
        random_state=random_state, n_jobs=n_jobs,
        focal_gamma=focal_gamma, focal_iterations=focal_iterations
    )
    calibrated_model = CalibratedClassifierCV(base_model, method=calibration_method, cv='prefit')
    calibrated_model.fit(X_cal, y_cal)
    return calibrated_model


def calibrate_per_class(y_true: np.ndarray, y_pred_proba: np.ndarray,
                        n_bins: int = 10) -> List:
    """Create per-class calibration mappings using isotonic regression."""
    n_classes = y_pred_proba.shape[1]
    calibration_maps = []

    for class_idx in range(n_classes):
        y_binary = (y_true == class_idx).astype(int)
        y_prob = y_pred_proba[:, class_idx]

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob, n_bins=n_bins, strategy='quantile'
            )
            mean_predicted_value = np.concatenate([[0], mean_predicted_value, [1]])
            fraction_of_positives = np.concatenate([[0], fraction_of_positives, [1]])

            calibration_map = interp1d(
                mean_predicted_value, fraction_of_positives,
                kind='linear', bounds_error=False, fill_value=(0, 1)
            )
        except ValueError:
            calibration_map = lambda x: x

        calibration_maps.append(calibration_map)

    return calibration_maps


def apply_calibration_maps(y_pred_proba: np.ndarray, calibration_maps: List) -> np.ndarray:
    """Apply per-class calibration mappings to predicted probabilities."""
    calibrated_proba = np.zeros_like(y_pred_proba)

    for class_idx, calibration_map in enumerate(calibration_maps):
        calibrated_proba[:, class_idx] = calibration_map(y_pred_proba[:, class_idx])

    calibrated_proba = np.clip(calibrated_proba, 0, 1)
    row_sums = np.maximum(calibrated_proba.sum(axis=1, keepdims=True), 1e-10)
    calibrated_proba = calibrated_proba / row_sums
    calibrated_proba = np.clip(calibrated_proba, 0, 1)

    return calibrated_proba


# ============================================================================
# Feature Selection Functions
# ============================================================================

def select_features_by_importance(X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   importance_threshold: float = 0.5,
                                   essential_features: List[str] = None,
                                   random_state: int = 42) -> Tuple[List[str], Dict]:
    """
    Select features based on cumulative importance threshold.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    feature_names : List[str]
        Names of all features
    importance_threshold : float
        Cumulative importance threshold (e.g., 0.5 = features explaining 50% of importance)
    essential_features : List[str]
        Features to always include
    random_state : int
        Random seed

    Returns
    -------
    selected_features : List[str]
        Names of selected features
    importance_info : Dict
        Feature importance information
    """
    if essential_features is None:
        essential_features = []

    # Train quick model for importance
    quick_model = ExtraTreesClassifier(
        n_estimators=100, max_depth=20, random_state=random_state, n_jobs=-1
    )
    quick_model.fit(X, y)

    importances = quick_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Calculate cumulative importance
    total_importance = importance_df['importance'].sum()
    importance_df['cumulative_pct'] = importance_df['importance'].cumsum() / total_importance

    # Select features up to threshold
    selected_features = []
    for _, row in importance_df.iterrows():
        selected_features.append(row['feature'])
        if row['cumulative_pct'] >= importance_threshold:
            break

    # Ensure essential features are included
    for feat in essential_features:
        if feat in feature_names and feat not in selected_features:
            selected_features.append(feat)

    importance_info = {
        'n_selected': len(selected_features),
        'importance_threshold': importance_threshold,
        'top_features': importance_df.head(20).to_dict('records')
    }

    return selected_features, importance_info


def select_features_by_top_n(X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             n_top: int = 50,
                             essential_features: List[str] = None,
                             random_state: int = 42) -> Tuple[List[str], Dict]:
    """
    Select top N features by importance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    feature_names : List[str]
        Names of all features
    n_top : int
        Number of top features to select
    essential_features : List[str]
        Features to always include
    random_state : int
        Random seed

    Returns
    -------
    selected_features : List[str]
        Names of selected features
    importance_info : Dict
        Feature importance information
    """
    if essential_features is None:
        essential_features = []

    # Train quick model for importance
    quick_model = ExtraTreesClassifier(
        n_estimators=100, max_depth=20, random_state=random_state, n_jobs=-1
    )
    quick_model.fit(X, y)

    importances = quick_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Select top N
    selected_features = importance_df.head(n_top)['feature'].tolist()

    # Ensure essential features are included
    for feat in essential_features:
        if feat in feature_names and feat not in selected_features:
            selected_features.append(feat)

    importance_info = {
        'n_selected': len(selected_features),
        'n_top': n_top,
        'top_features': importance_df.head(20).to_dict('records')
    }

    return selected_features, importance_info


# ============================================================================
# Metric Calculation Functions (copied from optuna_search.py)
# ============================================================================

def calculate_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray,
                      n_classes: int = 3) -> Dict[str, float]:
    """Calculate ROC-AUC scores (macro and per-class)."""
    results = {}
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_pred_proba.shape[1] < n_classes:
            full_proba = np.zeros((len(y_pred_proba), n_classes))
            for i in range(y_pred_proba.shape[1]):
                full_proba[:, i] = y_pred_proba[:, i]
            y_pred_proba = full_proba

        results['roc_auc_macro'] = roc_auc_score(
            y_true_bin, y_pred_proba, average='macro', multi_class='ovr'
        )
        for i in range(n_classes):
            if y_true_bin[:, i].sum() > 0:
                results[f'roc_auc_class_{i}'] = roc_auc_score(
                    y_true_bin[:, i], y_pred_proba[:, i]
                )
            else:
                results[f'roc_auc_class_{i}'] = np.nan
    except Exception:
        results['roc_auc_macro'] = np.nan
        for i in range(n_classes):
            results[f'roc_auc_class_{i}'] = np.nan
    return results


def calculate_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   n_classes: int = 3) -> Dict[str, float]:
    """Calculate calibration metrics: Brier Score, ECE, MCE."""
    metrics = {'brier_scores': [], 'ece_scores': [], 'mce_scores': [], 'mean_pred_probs': []}

    for class_idx in range(n_classes):
        y_binary = (y_true == class_idx).astype(int)
        y_prob = y_pred_proba[:, class_idx] if y_pred_proba.shape[1] > class_idx else np.zeros(len(y_true))
        if y_binary.sum() == 0:
            continue

        brier = brier_score_loss(y_binary, y_prob)
        metrics['brier_scores'].append(brier)
        metrics['mean_pred_probs'].append(np.mean(y_prob))

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob, n_bins=10, strategy='uniform'
            )
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        except ValueError:
            ece, mce = np.nan, np.nan

        metrics['ece_scores'].append(ece)
        metrics['mce_scores'].append(mce)

    return {
        'brier_score': np.nanmean(metrics['brier_scores']),
        'ece': np.nanmean(metrics['ece_scores']),
        'mce': np.nanmean(metrics['mce_scores']),
        'mean_predicted_probability': np.nanmean(metrics['mean_pred_probs'])
    }


def convert_phase_proba_to_transition_proba(
    y_pred_proba: np.ndarray, prev_phases: np.ndarray,
    days_to_next_appt: np.ndarray, i_threshold: float, p_threshold: float
) -> np.ndarray:
    """Convert phase probabilities to transition probabilities."""
    n_samples = len(prev_phases)
    transition_proba = np.zeros((n_samples, 3))
    y_pred_proba = np.clip(y_pred_proba, 0, 1)

    for i in range(n_samples):
        prev = int(prev_phases[i])
        days = float(days_to_next_appt[i]) if not np.isnan(days_to_next_appt[i]) else 14.0
        p_I, p_P, p_R = y_pred_proba[i]

        if prev < 0:
            p_favorable = p_P + p_R
            if days > i_threshold:
                p_unfavorable, p_acceptable = p_I, 0.0
            else:
                p_unfavorable, p_acceptable = 0.0, p_I
            transition_proba[i] = [p_unfavorable, p_acceptable, p_favorable]
            continue

        if prev == 0:
            p_favorable = p_P
            p_regression = p_R
            if days > i_threshold:
                p_unfavorable, p_acceptable = p_I + p_regression, 0.0
            else:
                p_unfavorable, p_acceptable = p_regression, p_I
        elif prev == 1:
            p_favorable = p_R
            p_regression = p_I
            if days > p_threshold:
                p_unfavorable, p_acceptable = p_P + p_regression, 0.0
            else:
                p_unfavorable, p_acceptable = p_regression, p_P
        else:
            p_favorable = 0.0
            p_regression = p_I + p_P
            p_unfavorable, p_acceptable = p_regression, p_R

        transition_proba[i] = [p_unfavorable, p_acceptable, p_favorable]

    transition_proba = np.clip(transition_proba, 0, 1)
    row_sums = np.maximum(transition_proba.sum(axis=1, keepdims=True), 1e-10)
    transition_proba = transition_proba / row_sums
    transition_proba = np.clip(transition_proba, 0, 1)

    return transition_proba


def calculate_all_metrics(
    y_true_phases: np.ndarray, y_pred_phases: np.ndarray, y_pred_proba: np.ndarray,
    prev_phases: np.ndarray, days_to_next_appt: np.ndarray,
    i_threshold: float, p_threshold: float, r_threshold: Optional[float]
) -> Dict[str, float]:
    """Calculate all evaluation metrics for a configuration."""
    y_true_trans = compute_transition_labels_chronicity_aware(
        y_true_phases, prev_phases, days_to_next_appt, i_threshold, p_threshold, r_threshold
    )
    y_pred_trans = compute_transition_labels_chronicity_aware(
        y_pred_phases, prev_phases, days_to_next_appt, i_threshold, p_threshold, r_threshold
    )
    y_pred_trans_proba = convert_phase_proba_to_transition_proba(
        y_pred_proba, prev_phases, days_to_next_appt, i_threshold, p_threshold
    )

    metrics = {}
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_trans, y_pred_trans)
    metrics['accuracy'] = accuracy_score(y_true_trans, y_pred_trans)
    metrics['f1_macro'] = f1_score(y_true_trans, y_pred_trans, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true_trans, y_pred_trans, average='weighted', zero_division=0)

    f1_per_class = f1_score(y_true_trans, y_pred_trans, average=None, zero_division=0)
    metrics['f1_unfavorable'] = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
    metrics['f1_acceptable'] = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    metrics['f1_favorable'] = f1_per_class[2] if len(f1_per_class) > 2 else 0.0
    metrics['f1_balanced'] = np.mean([metrics['f1_unfavorable'], metrics['f1_acceptable'], metrics['f1_favorable']])

    metrics['precision_macro'] = precision_score(y_true_trans, y_pred_trans, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true_trans, y_pred_trans, average='macro', zero_division=0)

    roc_metrics = calculate_roc_auc(y_true_trans, y_pred_trans_proba, n_classes=3)
    metrics.update(roc_metrics)

    calib_metrics = calculate_calibration_metrics(y_true_trans, y_pred_trans_proba, n_classes=3)
    metrics.update(calib_metrics)

    metrics['phase_balanced_accuracy'] = balanced_accuracy_score(y_true_phases, y_pred_phases)
    metrics['phase_f1_macro'] = f1_score(y_true_phases, y_pred_phases, average='macro', zero_division=0)

    return metrics


# Combined score functions (copied from optuna_search.py)
def _compute_weighted_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Helper function to compute weighted score from metrics."""
    score, total_weight = 0.0, 0.0
    for metric, weight in weights.items():
        if metric.endswith('_inv'):
            base_metric = metric[:-4]
            if base_metric in metrics and not np.isnan(metrics[base_metric]):
                score += weight * (1.0 - metrics[base_metric])
                total_weight += weight
        else:
            if metric in metrics and not np.isnan(metrics[metric]):
                score += weight * metrics[metric]
                total_weight += weight
    return score / total_weight if total_weight > 0 else 0.0


def calculate_combined_score(metrics: Dict[str, float]) -> float:
    """Calculate combined score from multiple metrics (balanced approach)."""
    weights = {
        'balanced_accuracy': 0.15, 'f1_macro': 0.15, 'f1_balanced': 0.10,
        'roc_auc_macro': 0.15, 'brier_score_inv': 0.15, 'ece_inv': 0.10,
        'mce_inv': 0.10, 'f1_unfavorable': 0.05, 'f1_favorable': 0.05,
    }
    return _compute_weighted_score(metrics, weights)


def calculate_combined_score_imbalanced(metrics: Dict[str, float]) -> float:
    """Combined score optimized for imbalanced datasets."""
    weights = {
        'f1_unfavorable': 0.25, 'f1_favorable': 0.20, 'f1_acceptable': 0.10,
        'balanced_accuracy': 0.15, 'recall_macro': 0.15, 'roc_auc_macro': 0.10,
        'brier_score_inv': 0.05,
    }
    return _compute_weighted_score(metrics, weights)


def calculate_combined_score_imbalanced_v2(metrics: Dict[str, float]) -> float:
    """
    Combined score for imbalanced datasets - variant 2.
    Uses geometric mean of per-class F1s to penalize poor performance on any class.
    """
    # Geometric mean of F1 scores (penalizes if any class has low F1)
    f1_scores = [
        max(0.001, metrics.get('f1_unfavorable', 0.001)),
        max(0.001, metrics.get('f1_acceptable', 0.001)),
        max(0.001, metrics.get('f1_favorable', 0.001))
    ]
    f1_geometric_mean = np.power(np.prod(f1_scores), 1/3)

    weights = {
        'balanced_accuracy': 0.20,
        'roc_auc_macro': 0.15,
        'recall_macro': 0.15,
        'brier_score_inv': 0.10,
    }
    base_score = _compute_weighted_score(metrics, weights)

    # Combine: 60% geometric mean of F1s + 40% other metrics
    return 0.60 * f1_geometric_mean + 0.40 * base_score


def calculate_combined_score_clinical(metrics: Dict[str, float]) -> float:
    """
    Combined score with clinical focus.
    Prioritizes detecting unfavorable outcomes (patient safety) and calibration
    (reliable probability estimates for clinical decision making).
    """
    weights = {
        'f1_unfavorable': 0.30,      # Critical: don't miss deteriorating wounds
        'recall_macro': 0.15,        # Ensure we catch all transition types
        'roc_auc_macro': 0.15,       # Overall discrimination
        'brier_score_inv': 0.15,     # Well-calibrated probabilities
        'ece_inv': 0.10,             # Calibration error
        'balanced_accuracy': 0.10,
        'f1_favorable': 0.05,        # Also important but less critical
    }
    return _compute_weighted_score(metrics, weights)


def calculate_combined_score_calibration(metrics: Dict[str, float]) -> float:
    """
    Combined score focused on probability calibration.
    Best for when you need reliable probability estimates (e.g., risk stratification).
    """
    weights = {
        'brier_score_inv': 0.25,     # Primary calibration metric
        'ece_inv': 0.20,             # Expected calibration error
        'mce_inv': 0.15,             # Maximum calibration error
        'roc_auc_macro': 0.15,       # Still need good discrimination
        'balanced_accuracy': 0.10,
        'f1_macro': 0.10,
        'f1_balanced': 0.05,
    }
    return _compute_weighted_score(metrics, weights)


def calculate_combined_score_discrimination(metrics: Dict[str, float]) -> float:
    """Combined score focused on discrimination/classification ability."""
    weights = {
        'roc_auc_macro': 0.25, 'balanced_accuracy': 0.20, 'f1_macro': 0.20,
        'precision_macro': 0.15, 'recall_macro': 0.15, 'f1_balanced': 0.05,
    }
    return _compute_weighted_score(metrics, weights)


def calculate_combined_score_f1_focused(metrics: Dict[str, float]) -> float:
    """
    Combined score heavily focused on F1 scores.
    Balances precision and recall across all classes.
    """
    weights = {
        'f1_macro': 0.25,
        'f1_balanced': 0.20,
        'f1_unfavorable': 0.15,
        'f1_acceptable': 0.15,
        'f1_favorable': 0.15,
        'balanced_accuracy': 0.10,
    }
    return _compute_weighted_score(metrics, weights)


def calculate_combined_score_calibration_minority(metrics: Dict[str, float]) -> float:
    """
    Combined score focused on calibration and minority class F1 scores.
    ECE, MCE, f1_acceptable, f1_unfavorable, f1_favorable.
    """
    weights = {
        'ece_inv': 0.15,
        'mce_inv': 0.35,
        'f1_acceptable': 0.20,
        'f1_unfavorable': 0.10,
        'f1_favorable': 0.20,
    }
    return _compute_weighted_score(metrics, weights)


# ============================================================================
# Direct Transition Label Functions
# ============================================================================

def compute_transition_labels_for_df(df: pd.DataFrame,
                                      skip_first_appt_chronicity: bool = False) -> np.ndarray:
    """
    Compute transition labels for a dataframe using fixed thresholds and cumulative duration.
    Uses global I_THRESHOLD, P_THRESHOLD, R_THRESHOLD constants.
    """
    next_phases = df['Next_Healing_Phase'].values.astype(int)
    prev_phases = df.get('Previous_Phase', df.get('Initial_Phase', pd.Series(-1, index=df.index))).values

    days_to_next = df.get('Days_To_Next_Appt', pd.Series(14.0, index=df.index)).values
    onset_days = df.get('Onset (Days)', pd.Series(dtype=float)).values

    if 'Cumulative_Phase_Duration' in df.columns:
        cumulative_raw = df['Cumulative_Phase_Duration'].values
        cumulative_duration = np.where(np.isnan(cumulative_raw), days_to_next, cumulative_raw)
    else:
        cumulative_duration = days_to_next.copy()

    labels = compute_transition_labels_chronicity_aware(
        next_phases, prev_phases, days_to_next,
        inflammatory_threshold=I_THRESHOLD,
        proliferative_threshold=P_THRESHOLD,
        remodeling_threshold=R_THRESHOLD,
        onset_days=onset_days,
        cumulative_phase_duration=cumulative_duration,
        skip_first_appt_chronicity=skip_first_appt_chronicity
    )
    return labels


def calculate_all_metrics_direct(
    y_true_trans: np.ndarray, y_pred_trans: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """Calculate all evaluation metrics for direct transition prediction."""
    metrics = {}

    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_trans, y_pred_trans)
    metrics['accuracy'] = accuracy_score(y_true_trans, y_pred_trans)
    metrics['f1_macro'] = f1_score(y_true_trans, y_pred_trans, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true_trans, y_pred_trans, average='weighted', zero_division=0)

    f1_per_class = f1_score(y_true_trans, y_pred_trans, average=None, zero_division=0)
    metrics['f1_unfavorable'] = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
    metrics['f1_acceptable'] = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    metrics['f1_favorable'] = f1_per_class[2] if len(f1_per_class) > 2 else 0.0
    metrics['f1_balanced'] = np.mean([metrics['f1_unfavorable'], metrics['f1_acceptable'], metrics['f1_favorable']])

    metrics['precision_macro'] = precision_score(y_true_trans, y_pred_trans, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true_trans, y_pred_trans, average='macro', zero_division=0)

    roc_metrics = calculate_roc_auc(y_true_trans, y_pred_proba, n_classes=3)
    metrics.update(roc_metrics)

    calib_metrics = calculate_calibration_metrics(y_true_trans, y_pred_proba, n_classes=3)
    metrics.update(calib_metrics)

    return metrics


# ============================================================================
# Training Function with Dynamic Feature Selection (Direct Transition Training)
# ============================================================================

def train_and_evaluate_config(
    config: Dict, preprocessor: DFUNextAppointmentPreprocessor,
    patient_cluster_map: Dict, df_processed: pd.DataFrame,
    all_feature_cols: List[str], n_folds: int = 3, random_state: int = 42, n_jobs: int = -1,
    median_onset_by_phase: Optional[Dict[int, float]] = None,
    phase_at_first_appt_map: Optional[Dict[Tuple[Any, Any], int]] = None
) -> Optional[Tuple[Dict[str, float], List[str]]]:
    """
    Train and evaluate a single configuration with cross-validation.
    DIRECT TRANSITION TRAINING: Trains on transition labels, not phase labels.
    Returns both metrics and the list of selected features.
    """
    unique_patients = df_processed['Patient#'].unique()
    resampler = FlexibleResampler(strategy=config['resampling'])
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Get config parameters
    skip_first_chron = config.get('skip_first_appt_chronicity', False)
    use_focal_loss = config.get('use_focal_loss', False)
    focal_gamma = config.get('focal_gamma', 2.0)
    focal_iterations = config.get('focal_iterations', 2)
    calibration_method = config.get('calibration_method', 'isotonic')

    all_y_true_trans, all_y_pred_trans, all_y_pred_proba = [], [], []
    selected_features = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(unique_patients)):
        train_patients = unique_patients[train_idx]
        val_patients = unique_patients[val_idx]

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

        if len(val_samples) == 0:
            continue

        val_df = pd.DataFrame(val_samples)

        # Apply onset correction if maps provided
        if median_onset_by_phase is not None and phase_at_first_appt_map is not None:
            train_df = replace_onset_with_median(train_df, preprocessor, median_onset_by_phase,
                                                   phase_at_first_appt_map, silent=True)
            val_df = replace_onset_with_median(val_df, preprocessor, median_onset_by_phase,
                                                 phase_at_first_appt_map, silent=True)

        # Compute TRANSITION labels directly (not phase labels)
        y_train_trans = compute_transition_labels_for_df(train_df, skip_first_appt_chronicity=skip_first_chron)
        y_val_trans = compute_transition_labels_for_df(val_df, skip_first_appt_chronicity=skip_first_chron)

        # Feature selection (done in first fold only for consistency)
        if fold == 0 and config.get('use_feature_selection', False):
            X_for_selection = train_df[all_feature_cols].copy()
            X_for_selection = X_for_selection.replace([np.inf, -np.inf], np.nan)
            imputer_sel = SimpleImputer(strategy='median')
            X_for_selection_imputed = imputer_sel.fit_transform(X_for_selection)
            available_essential = [f for f in ESSENTIAL_FEATURES if f in all_feature_cols]

            method = config.get('feature_selection_method', 'importance_threshold')
            if method == 'importance_threshold':
                threshold = config.get('importance_threshold', 0.5)
                selected_features, _ = select_features_by_importance(
                    X_for_selection_imputed, y_train_trans, all_feature_cols,
                    importance_threshold=threshold, essential_features=available_essential,
                    random_state=random_state
                )
            else:
                n_top = config.get('n_top_features', 50)
                selected_features, _ = select_features_by_top_n(
                    X_for_selection_imputed, y_train_trans, all_feature_cols,
                    n_top=n_top, essential_features=available_essential,
                    random_state=random_state
                )
        elif fold == 0:
            selected_features = all_feature_cols

        feature_cols = selected_features
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()

        # Handle missing values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_val = X_val.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        # Sample weights based on transition class frequency with boost factors
        n_classes = 3
        class_counts = np.bincount(y_train_trans, minlength=n_classes)
        class_weights_dict = {i: len(y_train_trans) / (n_classes * class_counts[i] + 1e-10) for i in range(n_classes)}

        # Apply favorable/unfavorable/acceptable boost
        favorable_boost = config.get('favorable_boost', 1.0)
        unfavorable_boost = config.get('unfavorable_boost', 1.0)
        acceptable_boost = config.get('acceptable_boost', 1.0)

        sample_weights = np.ones(len(y_train_trans))
        for i, label in enumerate(y_train_trans):
            base_weight = class_weights_dict.get(label, 1.0)
            if label == 0:  # Unfavorable
                sample_weights[i] = base_weight * unfavorable_boost
            elif label == 2:  # Favorable
                sample_weights[i] = base_weight * favorable_boost
            else:  # Acceptable
                sample_weights[i] = base_weight * acceptable_boost
        sample_weights = sample_weights / sample_weights.mean()

        # Apply resampling on transition labels
        if config['resampling'] != 'none':
            try:
                with suppress_stdout():
                    X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train_trans)
                # Recompute sample weights after resampling
                class_counts_res = np.bincount(y_train_resampled, minlength=n_classes)
                class_weights_dict = {i: len(y_train_resampled) / (n_classes * class_counts_res[i] + 1e-10) for i in range(n_classes)}
                sample_weights = np.array([
                    class_weights_dict.get(label, 1.0) *
                    (unfavorable_boost if label == 0 else favorable_boost if label == 2 else acceptable_boost)
                    for label in y_train_resampled
                ])
                sample_weights = sample_weights / sample_weights.mean()
            except Exception:
                X_train_resampled, y_train_resampled = X_train_scaled, y_train_trans
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train_trans

        # Train model
        use_calibration = config.get('use_calibration', True)
        if len(np.unique(y_train_resampled)) < 2:
            continue

        try:
            if use_focal_loss and use_calibration:
                model = train_with_focal_and_calibration(
                    X_train_resampled, y_train_resampled, sample_weights, config,
                    random_state=random_state, n_jobs=n_jobs,
                    focal_gamma=focal_gamma, focal_iterations=focal_iterations,
                    calibration_method=calibration_method
                )
            elif use_focal_loss:
                model = train_with_focal_loss(
                    X_train_resampled, y_train_resampled, sample_weights, config,
                    random_state=random_state, n_jobs=n_jobs,
                    focal_gamma=focal_gamma, focal_iterations=focal_iterations
                )
            elif use_calibration:
                model = train_with_calibration(
                    X_train_resampled, y_train_resampled, sample_weights, config,
                    random_state=random_state, n_jobs=n_jobs,
                    calibration_method=calibration_method
                )
            else:
                model = ExtraTreesClassifier(
                    n_estimators=config['n_estimators'], max_depth=config['max_depth'],
                    min_samples_split=config['min_samples_split'], min_samples_leaf=config['min_samples_leaf'],
                    max_features=config['max_features'], bootstrap=config['bootstrap'],
                    class_weight=config['class_weight'], criterion=config.get('criterion', 'gini'),
                    ccp_alpha=config.get('ccp_alpha', 0.0), min_weight_fraction_leaf=config.get('min_weight_fraction_leaf', 0.0),
                    max_samples=config.get('max_samples') if config['bootstrap'] else None,
                    random_state=random_state, n_jobs=n_jobs
                )
                model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)
        except Exception:
            model = ExtraTreesClassifier(
                n_estimators=config['n_estimators'], max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'], min_samples_leaf=config['min_samples_leaf'],
                max_features=config['max_features'], bootstrap=config['bootstrap'],
                class_weight=config['class_weight'], criterion=config.get('criterion', 'gini'),
                ccp_alpha=config.get('ccp_alpha', 0.0), min_weight_fraction_leaf=config.get('min_weight_fraction_leaf', 0.0),
                max_samples=config.get('max_samples') if config['bootstrap'] else None,
                random_state=random_state, n_jobs=n_jobs
            )
            model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)

        # Predict
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)

        if y_pred_proba.shape[1] < 3:
            full_proba = np.zeros((len(y_pred_proba), 3))
            for i, cls in enumerate(model.classes_):
                full_proba[:, int(cls)] = y_pred_proba[:, i]
            y_pred_proba = full_proba

        # Apply per-class calibration
        use_per_class_calibration = config.get('use_per_class_calibration', False)
        if use_per_class_calibration:
            try:
                train_proba_for_cal = model.predict_proba(X_train_scaled)
                if train_proba_for_cal.shape[1] < 3:
                    full_proba_cal = np.zeros((len(train_proba_for_cal), 3))
                    for i, cls in enumerate(model.classes_):
                        full_proba_cal[:, int(cls)] = train_proba_for_cal[:, i]
                    train_proba_for_cal = full_proba_cal
                calibration_maps = calibrate_per_class(y_train_trans, train_proba_for_cal, n_bins=10)
                y_pred_proba = apply_calibration_maps(y_pred_proba, calibration_maps)
                y_pred = np.argmax(y_pred_proba, axis=1)
            except Exception:
                pass

        # Apply temperature scaling (FIX: optimize on TRAINING data to avoid data leakage)
        use_temperature_scaling = config.get('use_temperature_scaling', False)
        if use_temperature_scaling:
            # Get training predictions for temperature optimization
            train_proba_for_temp = model.predict_proba(X_train_scaled)
            if train_proba_for_temp.shape[1] < 3:
                full_proba_temp = np.zeros((len(train_proba_for_temp), 3))
                for i, cls in enumerate(model.classes_):
                    full_proba_temp[:, int(cls)] = train_proba_for_temp[:, i]
                train_proba_for_temp = full_proba_temp

            # Optimize temperature on TRAINING data (no leakage)
            temperature_method = config.get('temperature_method', 'nll')
            if temperature_method == 'nll':
                opt_temp = optimize_temperature_nll(y_train_trans, train_proba_for_temp)
            else:
                opt_temp = optimize_temperature_ece(y_train_trans, train_proba_for_temp)
            # Apply learned temperature to VALIDATION predictions
            y_pred_proba = apply_temperature_scaling(y_pred_proba, opt_temp)
            y_pred = np.argmax(y_pred_proba, axis=1)

        all_y_true_trans.append(y_val_trans)
        all_y_pred_trans.append(y_pred)
        all_y_pred_proba.append(y_pred_proba)

    if len(all_y_true_trans) == 0:
        return None, []

    # Concatenate all folds
    y_true_trans = np.concatenate(all_y_true_trans)
    y_pred_trans = np.concatenate(all_y_pred_trans)
    y_pred_proba = np.concatenate(all_y_pred_proba)

    # Calculate metrics using direct transition metrics function
    metrics = calculate_all_metrics_direct(y_true_trans, y_pred_trans, y_pred_proba)

    # Calculate all combined scores
    metrics['combined_score'] = calculate_combined_score(metrics)
    metrics['combined_score_imbalanced'] = calculate_combined_score_imbalanced(metrics)
    metrics['combined_score_imbalanced_v2'] = calculate_combined_score_imbalanced_v2(metrics)
    metrics['combined_score_clinical'] = calculate_combined_score_clinical(metrics)
    metrics['combined_score_calibration'] = calculate_combined_score_calibration(metrics)
    metrics['combined_score_discrimination'] = calculate_combined_score_discrimination(metrics)
    metrics['combined_score_f1_focused'] = calculate_combined_score_f1_focused(metrics)
    metrics['combined_score_calibration_minority'] = calculate_combined_score_calibration_minority(metrics)

    # Add feature count to metrics
    metrics['n_features_used'] = len(selected_features) if selected_features else len(all_feature_cols)

    return metrics, selected_features if selected_features else all_feature_cols


# ============================================================================
# Optuna Objective Function with Feature Selection
# ============================================================================

class OptunaObjectiveFeatureSelection:
    """Optuna objective class with dynamic feature selection and direct transition training."""

    MINIMIZE_METRICS = {'brier_score', 'ece', 'mce'}

    def __init__(self, preprocessor, augmentation_datasets, n_folds, n_jobs,
                 optimize_for: str = 'combined_score', ensemble_seeds: list = None,
                 no_temp_scaling: bool = False):
        self.preprocessor = preprocessor
        self.augmentation_datasets = augmentation_datasets
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.trial_count = 0
        self.optimize_for = optimize_for
        # Multi-seed ensemble (default to single seed=42 for backward compatibility)
        self.ensemble_seeds = ensemble_seeds if ensemble_seeds else [42]
        # Track features selected across trials
        self.feature_selections = []
        # Override temperature scaling from CLI
        self.no_temp_scaling = no_temp_scaling

        # Compute onset correction maps once upfront (using 'Healing Phase Abs' column)
        print("Computing onset value corrections...")
        self.median_onset_by_phase = compute_median_onset_by_first_phase(
            augmentation_datasets['none']['df_processed'], preprocessor
        )
        self.phase_at_first_appt_map = get_phase_at_first_appt_map(preprocessor)

        if len(self.ensemble_seeds) > 1:
            print(f"Multi-seed ensemble enabled: {len(self.ensemble_seeds)} seeds per trial")

    def __call__(self, trial: optuna.Trial) -> float:
        self.trial_count += 1

        # Sample hyperparameters
        config = {
            # ExtraTrees parameters
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150, 200, 300, 400, 500, 600, 800]),
            'max_depth': trial.suggest_categorical('max_depth', [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, None]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 3, 5, 7, 10, 15, 20]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 3, 5, 7, 10]),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'ccp_alpha': trial.suggest_categorical('ccp_alpha', [0.0, 0.001, 0.005, 0.01]),
            'min_weight_fraction_leaf': trial.suggest_categorical('min_weight_fraction_leaf', [0.0, 0.01, 0.02, 0.05]),
            'max_samples': trial.suggest_categorical('max_samples', [None, 0.5, 0.7, 0.8, 0.9]),

            # Transition weighting (boost factors for each class)
            'favorable_boost': trial.suggest_categorical('favorable_boost', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            'unfavorable_boost': trial.suggest_categorical('unfavorable_boost', [0.5, 1.0, 1.5, 2.0, 3.0]),
            'acceptable_boost': trial.suggest_categorical('acceptable_boost', [0.5, 1.0, 1.5, 2.0, 3.0]),

            # Resampling
            'resampling': trial.suggest_categorical('resampling', ['none', 'smote', 'oversample', 'combined', 'undersample']),

            # Augmentation
            'augmentation': trial.suggest_categorical('augmentation', ['none', 'safe_sequential']),

            # Feature selection
            'use_feature_selection': trial.suggest_categorical('use_feature_selection', [True, False]),
            'feature_selection_method': trial.suggest_categorical('feature_selection_method', ['importance_threshold', 'top_n']),
            'importance_threshold': trial.suggest_categorical('importance_threshold', [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
            'n_top_features': trial.suggest_categorical('n_top_features', [10, 20, 30, 40, 50, 60, 70, 80, 100]),

            # First appointment chronicity handling
            'skip_first_appt_chronicity': trial.suggest_categorical('skip_first_appt_chronicity', [True, False]),

            # Calibration options
            'use_calibration': trial.suggest_categorical('use_calibration', [True, False]),
            'calibration_method': trial.suggest_categorical('calibration_method', ['isotonic', 'sigmoid']),
            'use_per_class_calibration': trial.suggest_categorical('use_per_class_calibration', [True, False]),

            # Temperature scaling
            'use_temperature_scaling': trial.suggest_categorical('use_temperature_scaling', [True, False]),
            'temperature_method': trial.suggest_categorical('temperature_method', ['nll', 'ece']),
        }

        # Override temperature scaling if --no-temp-scaling CLI flag was set
        if self.no_temp_scaling:
            config['use_temperature_scaling'] = False

        # Continue config with focal loss training
        config.update({
            # Focal loss training
            'use_focal_loss': trial.suggest_categorical('use_focal_loss', [True, False]),
            'focal_gamma': trial.suggest_categorical('focal_gamma', [0.5, 1.0, 2.0, 3.0, 5.0]),
            'focal_iterations': trial.suggest_categorical('focal_iterations', [1, 2, 3]),
        })

        # Get dataset for this augmentation type
        aug_type = config['augmentation']
        aug_data = self.augmentation_datasets[aug_type]

        try:
            # Run with multiple seeds for robust averaging
            all_seed_metrics = []
            selected_features = None

            for seed in self.ensemble_seeds:
                result = train_and_evaluate_config(
                    config, self.preprocessor,
                    aug_data['patient_cluster_map'],
                    aug_data['df_processed'],
                    aug_data['all_feature_cols'],
                    n_folds=self.n_folds,
                    random_state=seed,
                    n_jobs=self.n_jobs,
                    median_onset_by_phase=self.median_onset_by_phase,
                    phase_at_first_appt_map=self.phase_at_first_appt_map
                )

                if result is not None:
                    seed_metrics, seed_features = result
                    if seed_metrics is not None:
                        all_seed_metrics.append(seed_metrics)
                        if selected_features is None:
                            selected_features = seed_features

            if not all_seed_metrics:
                return 0.0

            # Average metrics across seeds
            if len(all_seed_metrics) > 1:
                metrics = {}
                for key in all_seed_metrics[0].keys():
                    values = [m[key] for m in all_seed_metrics if m.get(key) is not None and not np.isnan(m.get(key, np.nan))]
                    if values:
                        metrics[key] = np.mean(values)
                        # Store std for key metrics
                        if key in ['ece', 'mce', 'f1_macro']:
                            trial.set_user_attr(f'{key}_std', float(np.std(values)))
            else:
                metrics = all_seed_metrics[0]

            # Store ALL metrics as user attributes
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    trial.set_user_attr(key, float(value))

            # Store config for reference
            for key, value in config.items():
                trial.set_user_attr(f'config_{key}', value)

            # Store selected features (n_features_used is already in metrics)
            trial.set_user_attr('selected_features', selected_features)
            trial.set_user_attr('n_seeds', len(self.ensemble_seeds))

            # Track feature selection across trials
            self.feature_selections.append({
                'trial_number': self.trial_count,
                'features': selected_features,
                'n_features': len(selected_features) if selected_features else 0,
                'score': metrics.get(self.optimize_for, 0.0)
            })

            return metrics.get(self.optimize_for, 0.0)

        except Exception as e:
            print(f"  Trial {self.trial_count} failed: {str(e)[:50]}...")
            return float('inf') if self.optimize_for in self.MINIMIZE_METRICS else 0.0


# ============================================================================
# Results Processing with Feature Analysis
# ============================================================================

def process_study_results(study: optuna.Study, output_dir: Path, study_name: str,
                          objective: OptunaObjectiveFeatureSelection = None, silent: bool = False):
    """Process Optuna study results with feature analysis."""

    metrics_to_track = [
        # Combined scores (different weighting strategies)
        'combined_score',                    # Balanced approach
        'combined_score_imbalanced',         # Weighted toward minority classes
        'combined_score_imbalanced_v2',      # Geometric mean of F1s
        'combined_score_clinical',           # Prioritizes unfavorable detection
        'combined_score_calibration',        # Focused on probability calibration
        'combined_score_discrimination',     # Focused on classification ability
        'combined_score_f1_focused',         # Heavily weighted toward F1 scores
        'combined_score_calibration_minority',  # ECE/MCE + f1_acceptable/f1_unfavorable
        # Individual metrics
        'balanced_accuracy', 'f1_macro', 'f1_balanced',
        'f1_unfavorable', 'f1_acceptable', 'f1_favorable',
        'roc_auc_macro', 'brier_score', 'ece', 'mce',
        'mean_predicted_probability', 'precision_macro', 'recall_macro',
        'n_features_used'
    ]

    minimize_metrics = {'brier_score', 'ece', 'mce'}

    # Build all_results list
    all_results = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        result = {}
        for key, value in trial.user_attrs.items():
            if key.startswith('config_'):
                result[key[7:]] = value
            elif key == 'selected_features':
                # Store first 20 feature names for CSV (n_features_used is in metrics)
                result['selected_features'] = ','.join(value[:20]) if value else ''
            else:
                result[key] = value

        if result:
            all_results.append(result)

    # Find best for each metric
    best_results = {}
    for metric in metrics_to_track:
        best_value = float('inf') if metric in minimize_metrics else float('-inf')
        best_trial = None

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            if metric not in trial.user_attrs:
                continue

            value = trial.user_attrs[metric]
            if np.isnan(value):
                continue

            is_better = (value < best_value) if metric in minimize_metrics else (value > best_value)
            if is_better:
                best_value = value
                best_trial = trial

        if best_trial is not None:
            config = {k[7:]: v for k, v in best_trial.user_attrs.items() if k.startswith('config_')}
            all_metrics = {k: v for k, v in best_trial.user_attrs.items()
                          if not k.startswith('config_') and k != 'selected_features'}
            selected_feats = best_trial.user_attrs.get('selected_features', [])

            best_results[metric] = {
                'value': float(best_value),
                'config': config,
                'all_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                               for k, v in all_metrics.items()},
                'selected_features': selected_feats
            }
        else:
            best_results[metric] = {'value': None, 'config': None, 'all_metrics': None, 'selected_features': None}

    # Save best_results.json
    best_results_path = output_dir / f"{study_name}_best_results.json"
    # Convert selected_features lists to JSON-serializable format
    best_results_serializable = {}
    for metric, data in best_results.items():
        if data['selected_features'] is not None:
            data_copy = data.copy()
            data_copy['selected_features'] = list(data['selected_features'])
            best_results_serializable[metric] = data_copy
        else:
            best_results_serializable[metric] = data

    with open(best_results_path, 'w') as f:
        json.dump(best_results_serializable, f, indent=2, default=str)
    if not silent:
        print(f"Best results saved to: {best_results_path}")

    # Save all_results.csv
    if all_results:
        all_results_path = output_dir / f"{study_name}_all_results.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(all_results_path, index=False)
        if not silent:
            print(f"All results saved to: {all_results_path}")

    # Feature frequency analysis
    if objective and objective.feature_selections:
        analyze_feature_frequency(objective.feature_selections, study, output_dir, study_name, silent)

    # Generate final report
    report_path = output_dir / f"{study_name}_final_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OPTUNA HYPERPARAMETER SEARCH WITH FEATURE SELECTION - FINAL REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total trials completed: {len(all_results)}\n")
        f.write(f"Best combined_score: {study.best_value:.4f}\n\n")

        # Best features analysis
        best_combined = best_results.get('combined_score')
        if best_combined and best_combined['selected_features']:
            f.write("="*80 + "\n")
            f.write("BEST CONFIGURATION FEATURES\n")
            f.write("="*80 + "\n\n")
            f.write(f"Number of features: {len(best_combined['selected_features'])}\n")
            f.write(f"Features:\n")
            for feat in best_combined['selected_features']:
                f.write(f"  - {feat}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("BEST CONFIGURATIONS BY METRIC\n")
        f.write("="*80 + "\n\n")

        for metric in metrics_to_track:
            best = best_results.get(metric)
            if best and best['config']:
                f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
                f.write("-"*60 + "\n")
                f.write(f"  Best Value: {best['value']:.4f}\n")
                f.write(f"  N Features: {len(best.get('selected_features', []))}\n")
                f.write(f"  Configuration:\n")
                for k, v in best['config'].items():
                    f.write(f"    {k}: {v}\n")

    if not silent:
        print(f"Final report saved to: {report_path}")

    return best_results, all_results


# ============================================================================
# Periodic Saving Callback
# ============================================================================

class PeriodicSaveCallback:
    """Callback to save results periodically during optimization."""

    def __init__(self, output_dir: Path, study_name: str, save_every: int = 50,
                 objective: 'OptunaObjectiveFeatureSelection' = None):
        self.output_dir = output_dir
        self.study_name = study_name
        self.save_every = save_every
        self.last_save_count = 0
        self.objective = objective

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Called after each trial completes."""
        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])

        # Save every N trials
        if completed_trials > 0 and completed_trials % self.save_every == 0:
            if completed_trials > self.last_save_count:
                self.last_save_count = completed_trials
                self._save_results(study, completed_trials)

    def _save_results(self, study: optuna.Study, n_completed: int):
        """Save current results to JSON and CSV."""
        print(f"\n    [Checkpoint: Saving results at {n_completed} trials...]")

        try:
            # Use the same processing logic but in silent mode
            best_results, all_results = process_study_results(
                study, self.output_dir, self.study_name,
                objective=self.objective, silent=True
            )
            print(f"    [Checkpoint saved: {n_completed} trials, "
                  f"best_score={study.best_value:.4f}]")
        except Exception as e:
            print(f"    [Checkpoint save failed: {str(e)[:50]}]")


class EarlyStoppingCallback:
    """
    Callback to implement early stopping based on improvement rate.
    Stops if improvement over the last N trials is less than threshold percentage.
    """

    def __init__(self, min_trials: int = 250, patience: int = 100,
                 improvement_threshold: float = 0.001, minimize: bool = False):
        self.min_trials = min_trials
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.minimize = minimize
        self.best_value_at_checkpoint = None
        self.trials_since_checkpoint = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])

        if completed_trials < self.min_trials:
            return

        current_best = study.best_value

        if self.best_value_at_checkpoint is None:
            self.best_value_at_checkpoint = current_best
            self.trials_since_checkpoint = 0
            return

        self.trials_since_checkpoint += 1

        if self.trials_since_checkpoint >= self.patience:
            if self.minimize:
                if self.best_value_at_checkpoint != 0:
                    improvement = (self.best_value_at_checkpoint - current_best) / abs(self.best_value_at_checkpoint)
                else:
                    improvement = 0.0
            else:
                if self.best_value_at_checkpoint != 0:
                    improvement = (current_best - self.best_value_at_checkpoint) / abs(self.best_value_at_checkpoint)
                else:
                    improvement = 0.0

            if improvement < self.improvement_threshold:
                print(f"\n    [EARLY STOPPING: Improvement {improvement*100:.3f}% < {self.improvement_threshold*100:.1f}% "
                      f"over last {self.patience} trials after {completed_trials} total trials]")
                print(f"    [Best value at checkpoint: {self.best_value_at_checkpoint:.6f}, "
                      f"Current best: {current_best:.6f}]")
                study.stop()
            else:
                print(f"\n    [Progress check: {improvement*100:.3f}% improvement over last {self.patience} trials, continuing...]")
                self.best_value_at_checkpoint = current_best
                self.trials_since_checkpoint = 0


def analyze_feature_frequency(feature_selections: List[Dict], study: optuna.Study,
                               output_dir: Path, study_name: str, silent: bool = False):
    """Analyze which features appear most frequently in top trials."""

    # Sort trials by score (descending)
    sorted_selections = sorted(feature_selections, key=lambda x: x['score'], reverse=True)

    # Analyze top 10%, 25%, 50% trials
    n_trials = len(sorted_selections)
    percentiles = [
        ('top_10pct', int(n_trials * 0.1) or 1),
        ('top_25pct', int(n_trials * 0.25) or 1),
        ('top_50pct', int(n_trials * 0.5) or 1),
        ('all', n_trials)
    ]

    analysis = {}
    for name, n_top in percentiles:
        top_trials = sorted_selections[:n_top]
        feature_counter = Counter()

        for trial in top_trials:
            for feat in trial['features']:
                feature_counter[feat] += 1

        # Normalize by number of trials
        feature_freq = {feat: count / n_top for feat, count in feature_counter.most_common()}

        analysis[name] = {
            'n_trials': n_top,
            'feature_frequency': feature_freq,
            'features_in_all_trials': [feat for feat, freq in feature_freq.items() if freq == 1.0],
            'features_in_90pct': [feat for feat, freq in feature_freq.items() if freq >= 0.9],
            'avg_n_features': np.mean([t['n_features'] for t in top_trials])
        }

    # Save analysis
    analysis_path = output_dir / f"{study_name}_feature_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    if not silent:
        print(f"Feature analysis saved to: {analysis_path}")

        # Print summary
        print("\n" + "="*60)
        print("FEATURE FREQUENCY ANALYSIS (Top 10% trials)")
        print("="*60)

        top_10_analysis = analysis.get('top_10pct', {})
        print(f"Average features used: {top_10_analysis.get('avg_n_features', 0):.1f}")

        print("\nFeatures in ALL top 10% trials:")
        for feat in top_10_analysis.get('features_in_all_trials', [])[:10]:
            print(f"  - {feat}")

        print("\nFeatures in 90%+ of top 10% trials:")
        for feat in top_10_analysis.get('features_in_90pct', [])[:15]:
            if feat not in top_10_analysis.get('features_in_all_trials', []):
                freq = top_10_analysis.get('feature_frequency', {}).get(feat, 0)
                print(f"  - {feat} ({freq*100:.0f}%)")


# ============================================================================
# Helper Functions for Combined Results
# ============================================================================

def _find_best_across_studies(all_studies: Dict[str, optuna.Study], all_results: List[Dict]) -> Dict:
    """Find the best configuration for each metric across all studies."""
    metrics_to_track = [
        'combined_score', 'combined_score_imbalanced', 'combined_score_imbalanced_v2',
        'combined_score_clinical', 'combined_score_calibration', 'combined_score_discrimination',
        'combined_score_f1_focused', 'combined_score_calibration_minority',
        'balanced_accuracy', 'f1_macro', 'f1_balanced',
        'f1_unfavorable', 'f1_acceptable', 'f1_favorable',
        'roc_auc_macro', 'brier_score', 'ece', 'mce',
        'mean_predicted_probability', 'precision_macro', 'recall_macro', 'n_features_used'
    ]
    minimize_metrics = {'brier_score', 'ece', 'mce'}

    best_results = {}
    for metric in metrics_to_track:
        best_value = float('inf') if metric in minimize_metrics else float('-inf')
        best_result = None

        for result in all_results:
            if metric not in result:
                continue
            value = result[metric]
            if isinstance(value, (int, float)) and not np.isnan(value):
                is_better = (value < best_value) if metric in minimize_metrics else (value > best_value)
                if is_better:
                    best_value = value
                    best_result = result

        if best_result is not None:
            config = {k: v for k, v in best_result.items()
                     if k not in metrics_to_track and k != 'optimized_for'}
            all_metrics = {k: v for k, v in best_result.items()
                         if k in metrics_to_track}
            best_results[metric] = {
                'value': float(best_value),
                'config': config,
                'all_metrics': all_metrics,
                'found_in_study': best_result.get('optimized_for', 'unknown')
            }
        else:
            best_results[metric] = {'value': None, 'config': None, 'all_metrics': None, 'found_in_study': None}

    return best_results


def _print_combined_summary(all_studies: Dict[str, optuna.Study], best_results: Dict, optimize_for: List[str]):
    """Print combined summary from all studies."""
    print("\n" + "="*80)
    print("COMBINED SEARCH COMPLETE - SUMMARY")
    print("="*80)

    total_trials = sum(len(s.trials) for s in all_studies.values())
    print(f"\nTotal trials across all studies: {total_trials}")

    print("\nBest values per optimization target:")
    for target in optimize_for:
        study = all_studies.get(target)
        if study:
            print(f"  {target}: {study.best_value:.4f}")

    key_metrics = ['combined_score', 'combined_score_imbalanced', 'balanced_accuracy',
                   'f1_macro', 'roc_auc_macro', 'brier_score', 'n_features_used']

    print("\nBest values for key metrics (across ALL studies):")
    for metric in key_metrics:
        best = best_results.get(metric)
        if best and best['value'] is not None:
            found_in = best.get('found_in_study', 'unknown')
            print(f"  {metric}: {best['value']:.4f} (from {found_in} study)")

    print("\n" + "="*80)
    print("RECOMMENDED: Review the combined_best_results.json file")
    print("             Each metric shows which study found its best config")
    print("="*80)


# ============================================================================
# Main Function
# ============================================================================

DEFAULT_OPTIMIZE_FOR = ['combined_score']
MINIMIZE_METRICS = {'brier_score', 'ece', 'mce'}


def run_optuna_feature_selection_search(
    n_trials: int = 500,
    n_jobs: int = -1,
    resume: bool = False,
    study_name: Optional[str] = None,
    output_dir: str = 'grid_search_results',
    timeout: Optional[int] = None,
    save_every: int = 50,
    optimize_for: Optional[List[str]] = None,
    early_stop_min_trials: int = 250,
    early_stop_patience: int = 100,
    early_stop_threshold: float = 0.001,
    n_folds: int = 3,
    ensemble_seeds: List[int] = None,
    no_temp_scaling: bool = False
):
    """Run Optuna hyperparameter search with dynamic feature selection."""

    if ensemble_seeds is None:
        ensemble_seeds = [42]  # Default single seed

    if optimize_for is None:
        optimize_for = DEFAULT_OPTIMIZE_FOR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if study_name is None:
        study_name = f"optuna_feature_selection_{timestamp}"

    print("="*80)
    print("OPTUNA HYPERPARAMETER SEARCH WITH DYNAMIC FEATURE SELECTION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nStudy name: {study_name}")
    print(f"Optimization targets: {optimize_for}")
    print(f"Trials per target: {n_trials}")
    print(f"Parallel jobs for training: {n_jobs}")
    print(f"Resume mode: {resume}")
    print(f"Save checkpoint every: {save_every} trials")
    if timeout:
        print(f"Timeout per study: {timeout} seconds")
    if no_temp_scaling:
        print(f"Temperature scaling: DISABLED (--no-temp-scaling)")

    # Load config
    print("\n[1/4] Loading configuration...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    CSV_PATH = config['data']['csv_path']
    N_PATIENT_CLUSTERS = config['training']['n_patient_clusters']

    # Initialize preprocessor
    print("\n[2/4] Initializing preprocessor...")
    preprocessor = DFUNextAppointmentPreprocessor(CSV_PATH)
    df = preprocessor.initial_cleaning()
    gc.collect()
    df = preprocessor.convert_categorical_to_numeric()
    gc.collect()
    df = preprocessor.create_temporal_features()
    gc.collect()
    print("  Preprocessor ready.")

    # Pre-compute datasets for each augmentation type
    print("\n[3/4] Preparing datasets with ALL features...")
    augmentation_datasets = {}

    for aug_type in ['none', 'safe_sequential']:
        print(f"  Creating dataset for augmentation: {aug_type}")
        df_processed, patient_cluster_map, _ = preprocessor.create_next_appointment_dataset_with_augmentation(
            n_patient_clusters=N_PATIENT_CLUSTERS,
            augmentation_type=aug_type
        )

        # Get ALL available numeric features (not just OPTIMIZED_FEATURES)
        # IMPORTANT: Only select columns that are actually numeric in the DataFrame
        # This prevents "Cannot use median strategy with non-numeric data" errors
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        all_feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLUMNS]

        # Also check for treatment consistency features (should be numeric)
        for feat in TREATMENT_CONSISTENCY_NUMERIC:
            if feat in df_processed.columns and feat not in all_feature_cols:
                # Verify the column is actually numeric before adding
                if np.issubdtype(df_processed[feat].dtype, np.number):
                    all_feature_cols.append(feat)

        # Remove duplicates while preserving order
        seen = set()
        all_feature_cols = [x for x in all_feature_cols if not (x in seen or seen.add(x))]

        # Final validation: ensure all selected columns are actually numeric
        valid_feature_cols = []
        for col in all_feature_cols:
            if col in df_processed.columns:
                if np.issubdtype(df_processed[col].dtype, np.number):
                    valid_feature_cols.append(col)
                else:
                    print(f"    WARNING: Excluding non-numeric column '{col}' (dtype={df_processed[col].dtype})")

        all_feature_cols = valid_feature_cols
        print(f"    Available numeric features: {len(all_feature_cols)}")

        augmentation_datasets[aug_type] = {
            'df_processed': df_processed,
            'patient_cluster_map': patient_cluster_map,
            'all_feature_cols': all_feature_cols
        }

    # Run optimization
    print("\n[4/4] Starting Optuna search with dynamic feature selection...")
    all_studies = {}
    all_combined_results = []
    all_objectives = {}

    for idx, opt_target in enumerate(optimize_for):
        print("\n" + "="*80)
        print(f"OPTIMIZATION TARGET {idx+1}/{len(optimize_for)}: {opt_target}")
        print("="*80)

        target_study_name = f"{study_name}_{opt_target}"
        storage_path = output_path / f"{target_study_name}.db"
        storage = f"sqlite:///{storage_path}"

        direction = 'minimize' if opt_target in MINIMIZE_METRICS else 'maximize'
        sampler = TPESampler(seed=42 + idx, multivariate=True)

        if resume and storage_path.exists():
            print(f"  Resuming from existing study: {storage_path}")
            study = optuna.load_study(study_name=target_study_name, storage=storage, sampler=sampler)
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"  Already completed: {completed} trials")
        else:
            study = optuna.create_study(
                study_name=target_study_name,
                storage=storage,
                direction=direction,
                sampler=sampler,
                load_if_exists=resume
            )

        # Create objective with feature tracking
        objective = OptunaObjectiveFeatureSelection(
            preprocessor=preprocessor,
            augmentation_datasets=augmentation_datasets,
            n_folds=n_folds,
            n_jobs=n_jobs,
            optimize_for=opt_target,
            ensemble_seeds=ensemble_seeds,
            no_temp_scaling=no_temp_scaling
        )
        all_objectives[opt_target] = objective

        print(f"\n  Running {n_trials} trials optimizing for {opt_target} ({direction})...")
        print("-"*80)

        # Create periodic save callback
        save_callback = PeriodicSaveCallback(
            output_dir=output_path,
            study_name=target_study_name,
            save_every=save_every,
            objective=objective
        )

        # Create early stopping callback
        early_stop_callback = EarlyStoppingCallback(
            min_trials=early_stop_min_trials,
            patience=early_stop_patience,
            improvement_threshold=early_stop_threshold,
            minimize=(opt_target in MINIMIZE_METRICS)
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True,
            callbacks=[save_callback, early_stop_callback]
        )

        all_studies[opt_target] = study

        # Collect results
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {'optimized_for': opt_target}
                for key, value in trial.user_attrs.items():
                    if key.startswith('config_'):
                        result[key[7:]] = value
                    elif key == 'selected_features':
                        # Skip storing full feature list in combined results
                        # (n_features_used is already in metrics)
                        pass
                    else:
                        result[key] = value
                all_combined_results.append(result)

        # Save intermediate results with feature analysis
        process_study_results(study, output_path, target_study_name, objective=objective, silent=False)

    # Combine and save overall results
    print("\n" + "="*80)
    print("COMBINING RESULTS FROM ALL STUDIES")
    print("="*80)

    # Save combined CSV with all trials from all studies
    combined_csv_path = output_path / f"{study_name}_combined_all_results.csv"
    if all_combined_results:
        df_combined = pd.DataFrame(all_combined_results)
        df_combined.to_csv(combined_csv_path, index=False)
        print(f"Combined results saved to: {combined_csv_path}")

    # Create combined best results (find best for each metric across all studies)
    combined_best_results = _find_best_across_studies(all_studies, all_combined_results)
    combined_best_path = output_path / f"{study_name}_combined_best_results.json"
    with open(combined_best_path, 'w') as f:
        json.dump(combined_best_results, f, indent=2, default=str)
    print(f"Combined best results saved to: {combined_best_path}")

    # Print final summary
    _print_combined_summary(all_studies, combined_best_results, optimize_for)

    return all_studies, combined_best_results, all_combined_results


def main():
    parser = argparse.ArgumentParser(
        description='Optuna Search with Dynamic Feature Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script extends optuna_search.py with dynamic feature selection:
- Uses ALL available features instead of curated OPTIMIZED_FEATURES
- Feature selection method is a searchable hyperparameter
- Tracks which features were selected in each trial
- Generates feature frequency analysis across top trials

Example:
  python optuna_search_feature_selection.py --n-trials 500 --optimize-for combined_score
        """
    )
    parser.add_argument('--n-trials', type=int, default=500,
                        help='Number of trials PER optimization target (default: 500)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for model training (default: -1)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing study')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Base name for the Optuna studies')
    parser.add_argument('--output-dir', type=str, default='grid_search_results',
                        help='Output directory for results')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Optional timeout in seconds per study')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N trials (default: 50)')
    parser.add_argument('--optimize-for', type=str, default=None,
                        help='Comma-separated list of metrics to optimize. Default: combined_score')
    parser.add_argument('--early-stop-min-trials', type=int, default=250,
                        help='Minimum trials before early stopping can trigger (default: 250)')
    parser.add_argument('--early-stop-patience', type=int, default=100,
                        help='Trials to wait without significant improvement (default: 100)')
    parser.add_argument('--early-stop-threshold', type=float, default=0.001,
                        help='Minimum relative improvement required (default: 0.001 = 0.1%%)')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--n-seeds', type=int, default=1,
                        help='Number of seeds per trial for robust averaging (default: 1, use 7 for robust results)')
    parser.add_argument('--n-folds', type=int, default=3,
                        help='Number of cross-validation folds (default: 3)')
    parser.add_argument('--no-temp-scaling', action='store_true',
                        help='Disable temperature scaling (can reduce ECE variance)')

    args = parser.parse_args()

    # Define seeds for ensemble
    ENSEMBLE_SEEDS = [42, 123, 456, 789, 1234, 5678, 9012]
    args.ensemble_seeds = ENSEMBLE_SEEDS[:args.n_seeds]

    if args.optimize_for:
        optimize_for = [m.strip() for m in args.optimize_for.split(',')]
    else:
        optimize_for = None

    # Set early stopping parameters (disable if --no-early-stop is set)
    early_stop_min_trials = args.n_trials + 1 if args.no_early_stop else args.early_stop_min_trials
    early_stop_patience = args.early_stop_patience
    early_stop_threshold = args.early_stop_threshold

    run_optuna_feature_selection_search(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        resume=args.resume,
        study_name=args.study_name,
        output_dir=args.output_dir,
        timeout=args.timeout,
        save_every=args.save_every,
        optimize_for=optimize_for,
        early_stop_min_trials=early_stop_min_trials,
        early_stop_patience=early_stop_patience,
        early_stop_threshold=early_stop_threshold,
        n_folds=args.n_folds,
        ensemble_seeds=args.ensemble_seeds,
        no_temp_scaling=args.no_temp_scaling
    )


if __name__ == "__main__":
    main()
