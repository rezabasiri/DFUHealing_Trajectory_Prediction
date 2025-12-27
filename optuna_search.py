#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Search for DFU Healing Prediction Model

This script uses Bayesian optimization (Optuna) to efficiently search the hyperparameter
space, typically finding optimal configurations in 500-1000 trials instead of 190,000+
exhaustive grid search combinations.

Output Format:
- Produces the EXACT same output files as comprehensive_grid_search.py:
  - _best_results.json: Best configuration for each metric
  - _all_results.csv: All evaluated configurations with metrics
  - _final_report.txt: Human-readable summary

Features:
- TPE (Tree-structured Parzen Estimator) sampler for intelligent search
- Tracks all metrics (combined_score, balanced_accuracy, f1_macro, etc.)
- Finds best configuration for EACH metric separately
- Multi-objective support: optimize for multiple metrics simultaneously (runs separate studies)
- Resume capability via Optuna's built-in storage
- Periodic checkpoint saving (JSON/CSV) every N trials
- Parallel trial execution support

Usage:
    python optuna_search.py [--n-trials N] [--n-jobs N] [--resume] [--study-name NAME]

Arguments:
    --n-trials N: Number of trials to run PER optimization target (default: 500)
    --n-jobs N: Number of parallel jobs for model training (default: -1 for all cores)
    --resume: Resume from existing study
    --study-name NAME: Name for the Optuna study (default: auto-generated)
    --timeout SECONDS: Optional timeout in seconds for the entire search
    --save-every N: Save JSON/CSV checkpoint every N trials (default: 50)
    --optimize-for: Comma-separated list of metrics to optimize (runs separate study for each)
                    Default: combined_score,combined_score_imbalanced,combined_score_discrimination,roc_auc_macro

RECOMMENDATION:
    For most use cases, it is HIGHLY RECOMMENDED to optimize for just 'combined_score':
        python optuna_search.py --n-trials 500 --optimize-for combined_score

    The default multi-metric optimization (4 metrics) will run 4x the trials, which takes
    longer but explores the search space from different angles. Use this when:
    - You have time for a thorough search
    - You want to compare configs optimized for different objectives
    - You're unsure which metric matters most for your use case
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
from src.config.constants import OPTIMIZED_FEATURES


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
# Calibration Functions (Solution 2 & 3)
# ============================================================================

def train_with_calibration(X_train: np.ndarray, y_train: np.ndarray,
                           sample_weights: np.ndarray, config: Dict,
                           random_state: int = 42, n_jobs: int = -1,
                           calibration_method: str = 'isotonic') -> CalibratedClassifierCV:
    """
    Train model with proper calibration using a held-out calibration set.

    Strategy:
    1. Split training data: 80% train, 20% calibration
    2. Train base model on 80%
    3. Calibrate on 20% using specified method
    4. Return calibrated model

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    sample_weights : np.ndarray
        Sample weights for training
    config : Dict
        Model configuration parameters
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs
    calibration_method : str
        Calibration method: 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling)

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model
    """
    # Split training data for calibration
    X_train_base, X_cal, y_train_base, y_cal, weights_base, _ = train_test_split(
        X_train, y_train, sample_weights,
        test_size=0.2, random_state=random_state, stratify=y_train
    )

    # Train base model
    base_model = ExtraTreesClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        bootstrap=config['bootstrap'],
        class_weight=config['class_weight'],
        random_state=random_state,
        n_jobs=n_jobs
    )
    base_model.fit(X_train_base, y_train_base, sample_weight=weights_base)

    # Calibrate model using specified method
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=calibration_method,  # 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling)
        cv='prefit'  # Use pre-trained model
    )
    calibrated_model.fit(X_cal, y_cal)

    return calibrated_model


def calibrate_per_class(y_true: np.ndarray, y_pred_proba: np.ndarray,
                        n_bins: int = 10) -> List:
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


def apply_calibration_maps(y_pred_proba: np.ndarray,
                           calibration_maps: List) -> np.ndarray:
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
    # Avoid division by zero
    row_sums = np.maximum(row_sums, 1e-10)
    calibrated_proba = calibrated_proba / row_sums

    # Final clip to ensure valid probabilities
    calibrated_proba = np.clip(calibrated_proba, 0, 1)

    return calibrated_proba


# ============================================================================
# Temperature Scaling Functions
# ============================================================================

def apply_temperature_scaling(y_pred_proba: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to predicted probabilities.

    Temperature scaling divides the logits by T before softmax:
    - T < 1: Makes predictions more confident (sharper)
    - T > 1: Makes predictions less confident (softer)
    - T = 1: No change

    Parameters
    ----------
    y_pred_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    temperature : float
        Temperature parameter (> 0)

    Returns
    -------
    np.ndarray
        Temperature-scaled probabilities
    """
    if temperature <= 0:
        temperature = 1.0

    # Convert probabilities to logits
    eps = 1e-10
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    logits = np.log(y_pred_proba)

    # Apply temperature scaling and convert back to probabilities
    scaled_logits = logits / temperature
    scaled_proba = softmax(scaled_logits, axis=1)

    return scaled_proba


def optimize_temperature_nll(y_true: np.ndarray, y_pred_proba: np.ndarray,
                              temp_range: Tuple[float, float] = (0.1, 5.0)) -> float:
    """
    Find optimal temperature using negative log-likelihood minimization.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (n_samples,)
    y_pred_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    temp_range : Tuple[float, float]
        Range to search for optimal temperature

    Returns
    -------
    float
        Optimal temperature
    """
    def nll_loss(temp):
        scaled_proba = apply_temperature_scaling(y_pred_proba, temp)
        eps = 1e-10
        scaled_proba = np.clip(scaled_proba, eps, 1 - eps)

        # Calculate negative log-likelihood
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
    """
    Find optimal temperature using ECE minimization.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (n_samples,)
    y_pred_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    temp_range : Tuple[float, float]
        Range to search for optimal temperature
    n_bins : int
        Number of bins for ECE calculation

    Returns
    -------
    float
        Optimal temperature
    """
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
    """
    Train an ExtraTreesClassifier with iterative focal loss weighting.

    Focal loss helps focus training on hard examples (misclassified or
    low-confidence predictions), which can improve performance on minority classes.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    sample_weights : np.ndarray
        Initial sample weights (e.g., from transition weighting)
    config : Dict
        Model configuration parameters
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs
    focal_gamma : float
        Focusing parameter (higher = more focus on hard examples). gamma=2 is typical.
    focal_iterations : int
        Number of iterative retraining rounds

    Returns
    -------
    ExtraTreesClassifier
        Trained model
    """
    n_classes = len(np.unique(y_train))

    # Compute alpha (class weights) for focal loss - inverse frequency
    class_counts = np.bincount(y_train, minlength=n_classes)
    alpha = len(y_train) / (n_classes * class_counts + 1e-10)
    alpha = alpha / alpha.sum() * n_classes  # Normalize

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
            random_state=random_state,
            n_jobs=n_jobs
        )
        model.fit(X_train, y_train, sample_weight=current_weights)

        if iteration < focal_iterations - 1:
            # Compute focal weights for next iteration
            y_pred_proba = model.predict_proba(X_train)

            # Handle case where model doesn't predict all classes
            if y_pred_proba.shape[1] < n_classes:
                full_proba = np.zeros((len(y_pred_proba), n_classes))
                for i, cls in enumerate(model.classes_):
                    full_proba[:, int(cls)] = y_pred_proba[:, i]
                y_pred_proba = full_proba

            focal_weights = compute_focal_weights(y_pred_proba, y_train, gamma=focal_gamma, alpha=alpha)
            # Normalize focal weights
            focal_weights = focal_weights / (focal_weights.mean() + 1e-10)
            current_weights = sample_weights * focal_weights
            current_weights = current_weights / (current_weights.mean() + 1e-10)

    return model


def train_with_focal_and_calibration(X_train: np.ndarray, y_train: np.ndarray,
                                      sample_weights: np.ndarray, config: Dict,
                                      random_state: int = 42, n_jobs: int = -1,
                                      focal_gamma: float = 2.0, focal_iterations: int = 2,
                                      calibration_method: str = 'isotonic') -> CalibratedClassifierCV:
    """
    Train model with focal loss AND held-out calibration.

    Combines focal loss training with proper calibration using a held-out set.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    sample_weights : np.ndarray
        Sample weights for training
    config : Dict
        Model configuration parameters
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs
    focal_gamma : float
        Focusing parameter for focal loss
    focal_iterations : int
        Number of focal loss iterations
    calibration_method : str
        Calibration method: 'isotonic' or 'sigmoid'

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model trained with focal loss
    """
    # Split training data for calibration (80% train, 20% calibration)
    X_train_base, X_cal, y_train_base, y_cal, weights_base, _ = train_test_split(
        X_train, y_train, sample_weights,
        test_size=0.2, random_state=random_state, stratify=y_train
    )

    # Train base model with focal loss
    base_model = train_with_focal_loss(
        X_train_base, y_train_base, weights_base, config,
        random_state=random_state, n_jobs=n_jobs,
        focal_gamma=focal_gamma, focal_iterations=focal_iterations
    )

    # Calibrate model
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=calibration_method,
        cv='prefit'
    )
    calibrated_model.fit(X_cal, y_cal)

    return calibrated_model


# ============================================================================
# Metric Calculation Functions (same as comprehensive_grid_search.py)
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
    days_to_next_appt: np.ndarray, i_threshold: float, p_threshold: float,
    onset_days: Optional[np.ndarray] = None, default_days: float = 14.0,
    default_onset_days: float = 90.0,
    cumulative_phase_duration: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert phase probabilities to transition probabilities.

    For first appointments (prev_phase < 0), uses onset_days to determine chronicity.
    For subsequent appointments with same-phase transitions (I→I or P→P), uses
    cumulative_phase_duration to capture how long the wound has been in that phase
    (including previous consecutive same-phase appointments and onset time).

    Parameters
    ----------
    y_pred_proba : np.ndarray
        Predicted phase probabilities (n_samples, 3)
    prev_phases : np.ndarray
        Previous phase for each sample (-1 for first appointments)
    days_to_next_appt : np.ndarray
        Days until next appointment
    i_threshold : float
        Inflammatory threshold for chronicity
    p_threshold : float
        Proliferative threshold for chronicity
    onset_days : np.ndarray, optional
        Days since onset (used for first appointments)
    default_days : float
        Default value for missing days_to_next_appt (should come from imputer)
    default_onset_days : float
        Default value for missing onset_days (should come from imputer)
    cumulative_phase_duration : np.ndarray, optional
        Cumulative time the wound has been in the previous phase (for I→I, P→P transitions).
        If provided, this is used instead of days_to_next_appt for chronicity thresholds.

    Returns
    -------
    np.ndarray
        Transition probabilities (n_samples, 3) for [Unfavorable, Acceptable, Favorable]
    """
    n_samples = len(prev_phases)
    transition_proba = np.zeros((n_samples, 3))

    # Clip input probabilities to valid range first
    y_pred_proba = np.clip(y_pred_proba, 0, 1)

    for i in range(n_samples):
        prev = int(prev_phases[i])
        days = float(days_to_next_appt[i]) if not np.isnan(days_to_next_appt[i]) else default_days
        p_I, p_P, p_R = y_pred_proba[i]

        # Get cumulative phase duration if available (for same-phase transitions)
        if cumulative_phase_duration is not None and not np.isnan(cumulative_phase_duration[i]):
            cum_duration = float(cumulative_phase_duration[i])
        else:
            cum_duration = days  # Fallback to days_to_next_appt

        if prev < 0:
            # First appointment: use onset_days to determine chronicity
            # At onset, the phase is assumed to be 'I' (Inflammatory)
            if onset_days is not None:
                onset_d = float(onset_days[i]) if not np.isnan(onset_days[i]) else default_onset_days
            else:
                onset_d = default_onset_days

            p_favorable = p_P + p_R
            # Use onset_days (not days_to_next_appt) for first appointment chronicity
            if onset_d > i_threshold:
                p_unfavorable, p_acceptable = p_I, 0.0
            else:
                p_unfavorable, p_acceptable = 0.0, p_I
            transition_proba[i] = [p_unfavorable, p_acceptable, p_favorable]
            continue

        if prev == 0:  # I phase
            p_favorable = p_P
            p_regression = p_R
            # For I→I: use cumulative duration to check chronicity
            if cum_duration > i_threshold:
                p_unfavorable, p_acceptable = p_I + p_regression, 0.0
            else:
                p_unfavorable, p_acceptable = p_regression, p_I
        elif prev == 1:  # P phase
            p_favorable = p_R
            p_regression = p_I
            # For P→P: use cumulative duration to check chronicity
            if cum_duration > p_threshold:
                p_unfavorable, p_acceptable = p_P + p_regression, 0.0
            else:
                p_unfavorable, p_acceptable = p_regression, p_P
        else:  # R phase
            p_favorable = 0.0
            p_regression = p_I + p_P
            p_unfavorable, p_acceptable = p_regression, p_R

        transition_proba[i] = [p_unfavorable, p_acceptable, p_favorable]

    # Clip output to valid probability range and normalize
    transition_proba = np.clip(transition_proba, 0, 1)
    row_sums = transition_proba.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    transition_proba = transition_proba / row_sums
    transition_proba = np.clip(transition_proba, 0, 1)

    return transition_proba


def calculate_all_metrics(
    y_true_phases: np.ndarray, y_pred_phases: np.ndarray, y_pred_proba: np.ndarray,
    prev_phases: np.ndarray, days_to_next_appt: np.ndarray,
    i_threshold: float, p_threshold: float, r_threshold: Optional[float],
    onset_days: Optional[np.ndarray] = None,
    default_days: float = 14.0, default_onset_days: float = 90.0,
    cumulative_phase_duration: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate all evaluation metrics for a configuration.

    Parameters
    ----------
    cumulative_phase_duration : np.ndarray, optional
        Cumulative time the wound has been in the previous phase. For I→I and P→P
        transitions, this captures total time in that phase including previous
        consecutive same-phase appointments. If None, falls back to days_to_next_appt.
    """

    y_true_trans = compute_transition_labels_chronicity_aware(
        y_true_phases, prev_phases, days_to_next_appt, i_threshold, p_threshold, r_threshold,
        onset_days=onset_days, cumulative_phase_duration=cumulative_phase_duration
    )
    y_pred_trans = compute_transition_labels_chronicity_aware(
        y_pred_phases, prev_phases, days_to_next_appt, i_threshold, p_threshold, r_threshold,
        onset_days=onset_days, cumulative_phase_duration=cumulative_phase_duration
    )
    y_pred_trans_proba = convert_phase_proba_to_transition_proba(
        y_pred_proba, prev_phases, days_to_next_appt, i_threshold, p_threshold,
        onset_days=onset_days, default_days=default_days, default_onset_days=default_onset_days,
        cumulative_phase_duration=cumulative_phase_duration
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


def calculate_combined_score(metrics: Dict[str, float]) -> float:
    """Calculate combined score from multiple metrics (balanced approach)."""
    weights = {
        'balanced_accuracy': 0.15, 'f1_macro': 0.15, 'f1_balanced': 0.10,
        'roc_auc_macro': 0.15, 'brier_score_inv': 0.15, 'ece_inv': 0.10,
        'mce_inv': 0.10, 'f1_unfavorable': 0.05, 'f1_favorable': 0.05,
    }
    return _compute_weighted_score(metrics, weights)


def calculate_combined_score_imbalanced(metrics: Dict[str, float]) -> float:
    """
    Combined score optimized for imbalanced datasets.
    Heavily weights minority class performance (unfavorable and favorable transitions).
    """
    weights = {
        'f1_unfavorable': 0.25,      # Minority class - high weight
        'f1_favorable': 0.20,        # Minority class - high weight
        'f1_acceptable': 0.10,       # Majority class - lower weight
        'balanced_accuracy': 0.15,   # Accounts for class imbalance
        'recall_macro': 0.15,        # Ensures all classes are detected
        'roc_auc_macro': 0.10,
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
    """
    Combined score focused on discrimination/classification ability.
    Best for when you care most about correctly classifying transitions.
    """
    weights = {
        'roc_auc_macro': 0.25,       # Primary discrimination metric
        'balanced_accuracy': 0.20,
        'f1_macro': 0.20,
        'precision_macro': 0.15,
        'recall_macro': 0.15,
        'f1_balanced': 0.05,
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


# ============================================================================
# Training Function (same as comprehensive_grid_search.py)
# ============================================================================

def train_and_evaluate_config(
    config: Dict, preprocessor: DFUNextAppointmentPreprocessor,
    patient_cluster_map: Dict, df_processed: pd.DataFrame,
    feature_cols: List[str], n_folds: int = 3, random_state: int = 42, n_jobs: int = -1
) -> Optional[Dict[str, float]]:
    """Train and evaluate a single configuration with cross-validation."""

    target_col = 'Next_Healing_Phase'
    unique_patients = df_processed['Patient#'].unique()
    resampler = FlexibleResampler(strategy=config['resampling'])
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Get number of ensemble models (default 1 = no ensemble)
    n_models = config.get('n_models', 1)

    all_y_true_phases, all_y_pred_phases, all_y_pred_proba = [], [], []
    all_prev_phases, all_days_to_next, all_onset_days = [], [], []
    all_cumulative_phase_duration = []

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

        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].values.astype(int)
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].values.astype(int)

        # Get previous phases
        train_prev_phases = train_df.get('Previous_Phase', train_df.get('Initial_Phase', pd.Series(np.ones(len(train_df))))).values
        val_prev_phases = val_df.get('Previous_Phase', val_df.get('Initial_Phase', pd.Series(np.ones(len(val_df))))).values

        # Get days to next appointment - compute median from training data (no hardcoding!)
        train_days_raw = train_df.get('Days_To_Next_Appt', pd.Series(dtype=float)).values
        val_days_raw = val_df.get('Days_To_Next_Appt', pd.Series(dtype=float)).values

        # Compute default from training data median (not hardcoded)
        valid_train_days = train_days_raw[~np.isnan(train_days_raw)]
        default_days = float(np.median(valid_train_days)) if len(valid_train_days) > 0 else 14.0

        train_days_to_next = np.nan_to_num(train_days_raw, nan=default_days)
        val_days_to_next = np.nan_to_num(val_days_raw, nan=default_days)

        # Get onset days for first appointments (column name: 'Onset (Days)')
        train_onset_raw = train_df.get('Onset (Days)', pd.Series(dtype=float)).values
        val_onset_raw = val_df.get('Onset (Days)', pd.Series(dtype=float)).values

        # Compute default onset days from training data median (not hardcoded)
        valid_train_onset = train_onset_raw[~np.isnan(train_onset_raw)]
        default_onset_days = float(np.median(valid_train_onset)) if len(valid_train_onset) > 0 else 90.0

        train_onset_days = np.nan_to_num(train_onset_raw, nan=default_onset_days)
        val_onset_days = np.nan_to_num(val_onset_raw, nan=default_onset_days)

        # Get cumulative phase duration (tracks how long wound has been in current phase)
        # This is critical for I→I and P→P transitions where chronicity matters
        if 'Cumulative_Phase_Duration' in val_df.columns:
            val_cumulative_raw = val_df['Cumulative_Phase_Duration'].values
            # For missing values, use days_to_next_appt as fallback
            val_cumulative_duration = np.where(
                np.isnan(val_cumulative_raw),
                val_days_to_next,
                val_cumulative_raw
            )
        else:
            # Fallback if column doesn't exist (backwards compatibility)
            val_cumulative_duration = val_days_to_next.copy()

        # Compute transition weights
        with suppress_stdout():
            sample_weights, _ = compute_transition_weights_chronicity_aware(
                y_train, train_prev_phases, train_days_to_next,
                method=config['weight_method'], favorable_boost=config['favorable_boost'],
                unfavorable_boost=config['unfavorable_boost'],
                inflammatory_threshold=config['i_threshold'],
                proliferative_threshold=config['p_threshold'],
                remodeling_threshold=config['r_threshold']
            )

        # Handle missing values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_val = X_val.replace([np.inf, -np.inf], np.nan)

        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        # Apply resampling
        if config['resampling'] != 'none':
            combined_labels = train_prev_phases.astype(int) * 3 + y_train.astype(int)
            try:
                with suppress_stdout():
                    X_train_resampled, combined_resampled = resampler.fit_resample(X_train_scaled, combined_labels)
                prev_phases_resampled = combined_resampled // 3
                y_train_resampled = combined_resampled % 3
                median_days = np.nanmedian(train_days_to_next)
                days_resampled = np.full(len(y_train_resampled), median_days)
                with suppress_stdout():
                    sample_weights, _ = compute_transition_weights_chronicity_aware(
                        y_train_resampled, prev_phases_resampled, days_resampled,
                        method=config['weight_method'], favorable_boost=config['favorable_boost'],
                        unfavorable_boost=config['unfavorable_boost'],
                        inflammatory_threshold=config['i_threshold'],
                        proliferative_threshold=config['p_threshold'],
                        remodeling_threshold=config['r_threshold']
                    )
            except Exception:
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train

        # Final NaN check and cleanup before training
        # Check for NaN in training data after resampling
        if np.any(np.isnan(X_train_resampled)):
            # Re-impute any NaN values that appeared after resampling
            nan_mask = np.isnan(X_train_resampled)
            col_medians = np.nanmedian(X_train_resampled, axis=0)
            for col_idx in range(X_train_resampled.shape[1]):
                X_train_resampled[nan_mask[:, col_idx], col_idx] = col_medians[col_idx]

        # Check for NaN in validation data
        if np.any(np.isnan(X_val_scaled)):
            nan_mask = np.isnan(X_val_scaled)
            col_medians = np.nanmedian(X_val_scaled, axis=0)
            for col_idx in range(X_val_scaled.shape[1]):
                X_val_scaled[nan_mask[:, col_idx], col_idx] = col_medians[col_idx]

        # Check for NaN in sample weights
        if np.any(np.isnan(sample_weights)):
            sample_weights = np.nan_to_num(sample_weights, nan=1.0)

        # Final safety check - skip fold if still have NaN
        if np.any(np.isnan(X_train_resampled)) or np.any(np.isnan(X_val_scaled)):
            continue

        # Train model(s) with calibration and/or focal loss
        # Supports ensemble: train n_models with different seeds and average predictions
        use_calibration = config.get('use_calibration', True)
        calibration_method = config.get('calibration_method', 'isotonic')  # 'isotonic' or 'sigmoid'
        use_focal_loss = config.get('use_focal_loss', False)
        focal_gamma = config.get('focal_gamma', 2.0)
        focal_iterations = config.get('focal_iterations', 2)

        # Train ensemble of models (or single model if n_models=1)
        ensemble_predictions = []
        ensemble_proba = []

        for model_idx in range(n_models):
            model_seed = random_state + model_idx * 1000 + fold * 100

            try:
                if use_focal_loss and use_calibration and len(np.unique(y_train_resampled)) >= 2:
                    # Focal loss + calibration
                    model = train_with_focal_and_calibration(
                        X_train_resampled, y_train_resampled, sample_weights,
                        config, random_state=model_seed, n_jobs=n_jobs,
                        focal_gamma=focal_gamma, focal_iterations=focal_iterations,
                        calibration_method=calibration_method
                    )
                elif use_focal_loss:
                    # Focal loss without calibration
                    model = train_with_focal_loss(
                        X_train_resampled, y_train_resampled, sample_weights,
                        config, random_state=model_seed, n_jobs=n_jobs,
                        focal_gamma=focal_gamma, focal_iterations=focal_iterations
                    )
                elif use_calibration and len(np.unique(y_train_resampled)) >= 2:
                    # Calibration without focal loss
                    model = train_with_calibration(
                        X_train_resampled, y_train_resampled, sample_weights,
                        config, random_state=model_seed, n_jobs=n_jobs,
                        calibration_method=calibration_method
                    )
                else:
                    # No focal loss or calibration
                    model = ExtraTreesClassifier(
                        n_estimators=config['n_estimators'], max_depth=config['max_depth'],
                        min_samples_split=config['min_samples_split'], min_samples_leaf=config['min_samples_leaf'],
                        max_features=config['max_features'], bootstrap=config['bootstrap'],
                        class_weight=config['class_weight'], random_state=model_seed, n_jobs=n_jobs
                    )
                    model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)
            except Exception:
                # Fallback to basic model if any training method fails
                model = ExtraTreesClassifier(
                    n_estimators=config['n_estimators'], max_depth=config['max_depth'],
                    min_samples_split=config['min_samples_split'], min_samples_leaf=config['min_samples_leaf'],
                    max_features=config['max_features'], bootstrap=config['bootstrap'],
                    class_weight=config['class_weight'], random_state=model_seed, n_jobs=n_jobs
                )
                model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)

            # Predict
            y_pred_model = model.predict(X_val_scaled)
            y_pred_proba_model = model.predict_proba(X_val_scaled)

            if y_pred_proba_model.shape[1] < 3:
                full_proba = np.zeros((len(y_pred_proba_model), 3))
                for i, cls in enumerate(model.classes_):
                    full_proba[:, int(cls)] = y_pred_proba_model[:, i]
                y_pred_proba_model = full_proba

            ensemble_predictions.append(y_pred_model)
            ensemble_proba.append(y_pred_proba_model)

        # Average ensemble predictions
        if n_models > 1:
            # Average probabilities across models
            y_pred_proba = np.mean(ensemble_proba, axis=0)
            # Use majority vote for class predictions
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = ensemble_predictions[0]
            y_pred_proba = ensemble_proba[0]

        # Solution 3: Apply per-class calibration mapping if enabled
        use_per_class_calibration = config.get('use_per_class_calibration', False)
        if use_per_class_calibration and len(y_train_resampled) > 100:
            try:
                # Create calibration maps from validation predictions (held-out from training)
                # Use averaged probabilities if ensemble
                calibration_maps = calibrate_per_class(y_val, y_pred_proba, n_bins=10)
                # Note: We DON'T apply here - calibration is learned during model training
            except Exception:
                pass  # Keep original probabilities if per-class calibration fails

        # Apply temperature scaling if enabled
        use_temperature_scaling = config.get('use_temperature_scaling', False)
        if use_temperature_scaling:
            try:
                temp_method = config.get('temperature_method', 'nll')  # 'nll' or 'ece'
                temp_range = (config.get('temp_range_min', 0.1), config.get('temp_range_max', 5.0))

                if temp_method == 'ece':
                    optimal_temp = optimize_temperature_ece(y_val, y_pred_proba, temp_range=temp_range)
                else:  # 'nll'
                    optimal_temp = optimize_temperature_nll(y_val, y_pred_proba, temp_range=temp_range)

                y_pred_proba = apply_temperature_scaling(y_pred_proba, optimal_temp)
            except Exception:
                pass  # Keep original probabilities if temperature scaling fails

        all_y_true_phases.append(y_val)
        all_y_pred_phases.append(y_pred)
        all_y_pred_proba.append(y_pred_proba)
        all_prev_phases.append(val_prev_phases)
        all_days_to_next.append(val_days_to_next)
        all_onset_days.append(val_onset_days)
        all_cumulative_phase_duration.append(val_cumulative_duration)

    if len(all_y_true_phases) == 0:
        return None

    # Concatenate all folds
    y_true_phases = np.concatenate(all_y_true_phases)
    y_pred_phases = np.concatenate(all_y_pred_phases)
    y_pred_proba = np.concatenate(all_y_pred_proba)
    prev_phases = np.concatenate(all_prev_phases)
    days_to_next = np.concatenate(all_days_to_next)
    onset_days = np.concatenate(all_onset_days)
    cumulative_phase_duration = np.concatenate(all_cumulative_phase_duration)

    metrics = calculate_all_metrics(
        y_true_phases, y_pred_phases, y_pred_proba, prev_phases, days_to_next,
        config['i_threshold'], config['p_threshold'], config['r_threshold'],
        onset_days=onset_days, default_days=default_days, default_onset_days=default_onset_days,
        cumulative_phase_duration=cumulative_phase_duration
    )

    # Calculate all combined scores
    metrics['combined_score'] = calculate_combined_score(metrics)
    metrics['combined_score_imbalanced'] = calculate_combined_score_imbalanced(metrics)
    metrics['combined_score_imbalanced_v2'] = calculate_combined_score_imbalanced_v2(metrics)
    metrics['combined_score_clinical'] = calculate_combined_score_clinical(metrics)
    metrics['combined_score_calibration'] = calculate_combined_score_calibration(metrics)
    metrics['combined_score_discrimination'] = calculate_combined_score_discrimination(metrics)
    metrics['combined_score_f1_focused'] = calculate_combined_score_f1_focused(metrics)

    return metrics


# ============================================================================
# Optuna Objective Function
# ============================================================================

class OptunaObjective:
    """Optuna objective class that maintains state across trials."""

    # Metrics that should be minimized (lower is better)
    MINIMIZE_METRICS = {'brier_score', 'ece', 'mce'}

    def __init__(self, preprocessor, augmentation_datasets, n_folds, n_jobs,
                 optimize_for: str = 'combined_score'):
        self.preprocessor = preprocessor
        self.augmentation_datasets = augmentation_datasets
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.trial_count = 0
        self.optimize_for = optimize_for

    def __call__(self, trial: optuna.Trial) -> float:
        self.trial_count += 1

        # Sample hyperparameters
        config = {
            # ExtraTrees parameters - expanded ranges
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150, 200, 300, 400, 500, 600, 800]),
            'max_depth': trial.suggest_categorical('max_depth', [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, None]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 3, 5, 7, 10, 15, 20]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 3, 5, 7, 10]),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),

            # Transition weighting
            'weight_method': trial.suggest_categorical('weight_method', ['favorable_boost', 'balanced', 'clinical']),
            'favorable_boost': trial.suggest_categorical('favorable_boost', [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0]),
            'unfavorable_boost': trial.suggest_categorical('unfavorable_boost', [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0]),

            # Chronicity thresholds - expanded ranges
            'i_threshold': trial.suggest_categorical('i_threshold', [14, 21, 28, 35, 42]),
            'p_threshold': trial.suggest_categorical('p_threshold', [14, 21, 28, 35, 42, 49, 56]),
            'r_threshold': None,

            # Resampling
            'resampling': trial.suggest_categorical('resampling', ['none', 'smote', 'oversample', 'combined']),

            # Augmentation
            'augmentation': trial.suggest_categorical('augmentation', ['none', 'safe_sequential']),

            # Feature selection - now searchable
            'use_feature_selection': trial.suggest_categorical('use_feature_selection', [True, False]),
            'importance_threshold': trial.suggest_categorical('importance_threshold', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),

            # Calibration options (Solution 2 & 3)
            # use_calibration: Train with held-out calibration set (80/20 split)
            # calibration_method: 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling)
            # use_per_class_calibration: Apply separate calibration curves per transition type
            'use_calibration': trial.suggest_categorical('use_calibration', [True, False]),
            'calibration_method': trial.suggest_categorical('calibration_method', ['isotonic', 'sigmoid']),
            'use_per_class_calibration': trial.suggest_categorical('use_per_class_calibration', [True, False]),

            # Temperature scaling (post-hoc calibration)
            # use_temperature_scaling: Apply temperature scaling after model prediction
            # temperature_method: 'nll' (negative log-likelihood) or 'ece' (expected calibration error)
            'use_temperature_scaling': trial.suggest_categorical('use_temperature_scaling', [True, False]),
            'temperature_method': trial.suggest_categorical('temperature_method', ['nll', 'ece']),
            'temp_range_min': 0.1,  # Fixed range for temperature optimization
            'temp_range_max': 5.0,

            # Ensemble (multiple models with different seeds)
            # n_models: Number of models to train with different seeds (1 = no ensemble)
            'n_models': trial.suggest_categorical('n_models', [1, 3, 5]),

            # Focal loss (helps focus on hard examples / minority classes)
            # use_focal_loss: Enable iterative focal loss weighting
            # focal_gamma: Focusing parameter (higher = more focus on hard examples)
            # focal_iterations: Number of iterative retraining rounds
            'use_focal_loss': trial.suggest_categorical('use_focal_loss', [True, False]),
            'focal_gamma': trial.suggest_categorical('focal_gamma', [0.5, 1.0, 2.0, 3.0, 5.0]),
            'focal_iterations': trial.suggest_categorical('focal_iterations', [1, 2, 3]),
        }

        # Get dataset for this augmentation type
        aug_type = config['augmentation']
        aug_data = self.augmentation_datasets[aug_type]

        try:
            metrics = train_and_evaluate_config(
                config, self.preprocessor,
                aug_data['patient_cluster_map'],
                aug_data['df_processed'],
                aug_data['feature_cols'],
                n_folds=self.n_folds,
                n_jobs=self.n_jobs
            )

            if metrics is None:
                return 0.0

            # Store ALL metrics as user attributes for later analysis
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    trial.set_user_attr(key, float(value))

            # Store config for reference
            for key, value in config.items():
                trial.set_user_attr(f'config_{key}', value)

            # Print progress
            # if self.trial_count % 10 == 0:
            #     print(f"  Trial {self.trial_count}: combined_score={metrics['combined_score']:.4f}, "
            #           f"bal_acc={metrics['balanced_accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

            # Return the metric we're optimizing for
            return metrics.get(self.optimize_for, 0.0)

        except Exception as e:
            print(f"  Trial {self.trial_count} failed: {str(e)[:50]}...")
            # Return worst possible value based on whether we're minimizing or maximizing
            return float('inf') if self.optimize_for in self.MINIMIZE_METRICS else 0.0


# ============================================================================
# Periodic Saving Callback
# ============================================================================

class PeriodicSaveCallback:
    """Callback to save results periodically during optimization."""

    def __init__(self, output_dir: Path, study_name: str, save_every: int = 50):
        self.output_dir = output_dir
        self.study_name = study_name
        self.save_every = save_every
        self.last_save_count = 0

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
                study, self.output_dir, self.study_name, silent=True
            )
            print(f"    [Checkpoint saved: {n_completed} trials, "
                  f"best_score={study.best_value:.4f}]")
        except Exception as e:
            print(f"    [Checkpoint save failed: {str(e)[:50]}]")


class EarlyStoppingCallback:
    """
    Callback to stop optimization early if improvement is minimal.

    Stops if improvement over the last N trials is less than threshold percentage.
    """

    def __init__(self, min_trials: int = 250, patience: int = 100,
                 improvement_threshold: float = 0.001, minimize: bool = False):
        """
        Parameters
        ----------
        min_trials : int
            Minimum number of trials before early stopping can trigger
        patience : int
            Number of trials to wait without significant improvement
        improvement_threshold : float
            Minimum relative improvement required (0.001 = 0.1%)
        minimize : bool
            Whether we're minimizing (True) or maximizing (False) the objective
        """
        self.min_trials = min_trials
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.minimize = minimize
        self.best_value_at_checkpoint = None
        self.trials_since_checkpoint = 0
        self.should_stop = False

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Called after each trial completes."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])

        # Don't check until we have minimum trials
        if completed_trials < self.min_trials:
            return

        current_best = study.best_value

        # Initialize checkpoint if this is the first check
        if self.best_value_at_checkpoint is None:
            self.best_value_at_checkpoint = current_best
            self.trials_since_checkpoint = 0
            return

        self.trials_since_checkpoint += 1

        # Check if we've waited long enough
        if self.trials_since_checkpoint >= self.patience:
            # Calculate relative improvement
            if self.minimize:
                # For minimization: improvement = (old - new) / old
                if self.best_value_at_checkpoint != 0:
                    improvement = (self.best_value_at_checkpoint - current_best) / abs(self.best_value_at_checkpoint)
                else:
                    improvement = 0.0
            else:
                # For maximization: improvement = (new - old) / old
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
                # Reset checkpoint
                print(f"\n    [Progress check: {improvement*100:.3f}% improvement over last {self.patience} trials, continuing...]")
                self.best_value_at_checkpoint = current_best
                self.trials_since_checkpoint = 0


# ============================================================================
# Results Processing (same format as comprehensive_grid_search.py)
# ============================================================================

def process_study_results(study: optuna.Study, output_dir: Path, study_name: str, silent: bool = False):
    """Process Optuna study results into the same format as grid search.

    Args:
        study: Optuna study object
        output_dir: Directory to save results
        study_name: Name prefix for output files
        silent: If True, suppress print statements (for periodic saves)
    """

    metrics_to_track = [
        # Combined scores (different weighting strategies)
        'combined_score',                    # Balanced approach
        'combined_score_imbalanced',         # Weighted toward minority classes
        'combined_score_imbalanced_v2',      # Geometric mean of F1s
        'combined_score_clinical',           # Prioritizes unfavorable detection
        'combined_score_calibration',        # Focused on probability calibration
        'combined_score_discrimination',     # Focused on classification ability
        'combined_score_f1_focused',         # Heavily weighted toward F1 scores
        # Individual metrics
        'balanced_accuracy', 'f1_macro', 'f1_balanced',
        'f1_unfavorable', 'f1_acceptable', 'f1_favorable',
        'roc_auc_macro', 'brier_score', 'ece', 'mce',
        'mean_predicted_probability', 'precision_macro', 'recall_macro'
    ]

    minimize_metrics = {'brier_score', 'ece', 'mce'}

    # Build all_results list
    all_results = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        result = {}
        # Extract config
        for key, value in trial.user_attrs.items():
            if key.startswith('config_'):
                result[key[7:]] = value  # Remove 'config_' prefix
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
            all_metrics = {k: v for k, v in best_trial.user_attrs.items() if not k.startswith('config_')}

            best_results[metric] = {
                'value': float(best_value),
                'config': config,
                'all_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                               for k, v in all_metrics.items()}
            }
        else:
            best_results[metric] = {'value': None, 'config': None, 'all_metrics': None}

    # Save best_results.json (same format as grid search)
    best_results_path = output_dir / f"{study_name}_best_results.json"
    with open(best_results_path, 'w') as f:
        json.dump(best_results, f, indent=2, default=str)
    if not silent:
        print(f"Best results saved to: {best_results_path}")

    # Save all_results.csv (same format as grid search)
    if all_results:
        all_results_path = output_dir / f"{study_name}_all_results.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(all_results_path, index=False)
        if not silent:
            print(f"All results saved to: {all_results_path}")

    # Generate final report
    report_path = output_dir / f"{study_name}_final_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OPTUNA HYPERPARAMETER SEARCH - FINAL REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total trials completed: {len(all_results)}\n")
        f.write(f"Best combined_score: {study.best_value:.4f}\n\n")

        f.write("="*80 + "\n")
        f.write("BEST CONFIGURATIONS BY METRIC\n")
        f.write("="*80 + "\n\n")

        for metric in metrics_to_track:
            best = best_results.get(metric)
            if best and best['config']:
                f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
                f.write("-"*60 + "\n")
                f.write(f"  Best Value: {best['value']:.4f}\n")
                f.write(f"  Configuration:\n")
                for k, v in best['config'].items():
                    f.write(f"    {k}: {v}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDED CONFIGURATION (Best Combined Score)\n")
        f.write("="*80 + "\n\n")

        best_combined = best_results.get('combined_score')
        if best_combined and best_combined['config']:
            f.write("Configuration:\n")
            for k, v in best_combined['config'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\nTo use this configuration, run:\n")
            f.write(f"python train_with_transition_weights.py \\\n")
            f.write(f"  --chronicity-aware \\\n")
            f.write(f"  --method {best_combined['config'].get('weight_method', 'favorable_boost')} \\\n")
            f.write(f"  --favorable-boost {best_combined['config'].get('favorable_boost', 2.0)} \\\n")
            f.write(f"  --unfavorable-boost {best_combined['config'].get('unfavorable_boost', 1.5)} \\\n")
            f.write(f"  --i-threshold {best_combined['config'].get('i_threshold', 21)} \\\n")
            f.write(f"  --p-threshold {best_combined['config'].get('p_threshold', 28)} \\\n")
            f.write(f"  --resampling {best_combined['config'].get('resampling', 'none')}\n")

    if not silent:
        print(f"Final report saved to: {report_path}")

    return best_results, all_results


# ============================================================================
# Main Function
# ============================================================================

# Default optimization targets
DEFAULT_OPTIMIZE_FOR = ['combined_score', 'combined_score_imbalanced', 'combined_score_discrimination', 'roc_auc_macro']

# Metrics that should be minimized
MINIMIZE_METRICS = {'brier_score', 'ece', 'mce'}


def run_optuna_search(
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
    early_stop_threshold: float = 0.001
):
    """Run Optuna hyperparameter search.

    Args:
        n_trials: Number of trials to run PER optimization target
        n_jobs: Number of parallel jobs for model training
        resume: Resume from existing study
        study_name: Base name for the studies
        output_dir: Output directory for results
        timeout: Optional timeout in seconds per study
        save_every: Save checkpoint every N trials
        optimize_for: List of metrics to optimize (runs separate study for each)
        early_stop_min_trials: Minimum trials before early stopping can trigger
        early_stop_patience: Trials to wait without significant improvement
        early_stop_threshold: Minimum relative improvement required (0.001 = 0.1%)
    """
    if optimize_for is None:
        optimize_for = DEFAULT_OPTIMIZE_FOR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if study_name is None:
        study_name = f"optuna_search_{timestamp}"

    print("="*80)
    print("OPTUNA HYPERPARAMETER SEARCH FOR DFU HEALING PREDICTION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    total_trials = n_trials * len(optimize_for)
    print(f"\nBase study name: {study_name}")
    print(f"Optimization targets: {optimize_for}")
    print(f"Trials per target: {n_trials}")
    print(f"Total trials: {total_trials} ({len(optimize_for)} targets × {n_trials} trials)")
    print(f"Parallel jobs for training: {n_jobs}")
    print(f"Resume mode: {resume}")
    print(f"Save checkpoint every: {save_every} trials")
    if timeout:
        print(f"Timeout per study: {timeout} seconds")

    # Load config
    print("\n[1/4] Loading configuration...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    CSV_PATH = config['data']['csv_path']
    N_FOLDS = 3
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
    print("\n[3/4] Preparing datasets...")
    augmentation_datasets = {}

    for aug_type in ['none', 'safe_sequential']:
        print(f"  Creating dataset for augmentation: {aug_type}")
        df_processed, patient_cluster_map, _ = preprocessor.create_next_appointment_dataset_with_augmentation(
            n_patient_clusters=N_PATIENT_CLUSTERS,
            augmentation_type=aug_type
        )

        available_features = [f for f in OPTIMIZED_FEATURES if f in df_processed.columns]
        if len(available_features) == 0:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            exclude = ['Patient#', 'DFU#', 'Appt#', 'Next_Healing_Phase', 'ID']
            available_features = [c for c in numeric_cols if c not in exclude]

        augmentation_datasets[aug_type] = {
            'df_processed': df_processed,
            'patient_cluster_map': patient_cluster_map,
            'feature_cols': available_features
        }

    # Run separate study for each optimization target
    print("\n[4/4] Starting Optuna search...")
    all_studies = {}
    all_combined_results = []

    for idx, opt_target in enumerate(optimize_for):
        print("\n" + "="*80)
        print(f"OPTIMIZATION TARGET {idx+1}/{len(optimize_for)}: {opt_target}")
        print("="*80)

        # Create study name for this target
        target_study_name = f"{study_name}_{opt_target}"
        storage_path = output_path / f"{target_study_name}.db"
        storage = f"sqlite:///{storage_path}"

        # Determine optimization direction
        direction = 'minimize' if opt_target in MINIMIZE_METRICS else 'maximize'

        sampler = TPESampler(seed=42 + idx, multivariate=True)  # Different seed per study

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

        # Create objective for this target
        objective = OptunaObjective(
            preprocessor=preprocessor,
            augmentation_datasets=augmentation_datasets,
            n_folds=N_FOLDS,
            n_jobs=n_jobs,
            optimize_for=opt_target
        )

        print(f"\n  Running {n_trials} trials optimizing for {opt_target} ({direction})...")
        print("-"*80)

        # Create periodic save callback
        save_callback = PeriodicSaveCallback(
            output_dir=output_path,
            study_name=target_study_name,
            save_every=save_every
        )

        # Create early stopping callback
        early_stop_callback = EarlyStoppingCallback(
            min_trials=early_stop_min_trials,
            patience=early_stop_patience,
            improvement_threshold=early_stop_threshold,
            minimize=(direction == 'minimize')
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

        # Collect results from this study
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {'optimized_for': opt_target}
                for key, value in trial.user_attrs.items():
                    if key.startswith('config_'):
                        result[key[7:]] = value
                    else:
                        result[key] = value
                all_combined_results.append(result)

        # Save intermediate results for this study
        process_study_results(study, output_path, target_study_name, silent=False)

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


def _find_best_across_studies(all_studies: Dict[str, optuna.Study], all_results: List[Dict]) -> Dict:
    """Find the best configuration for each metric across all studies."""
    metrics_to_track = [
        'combined_score', 'combined_score_imbalanced', 'combined_score_imbalanced_v2',
        'combined_score_clinical', 'combined_score_calibration', 'combined_score_discrimination',
        'combined_score_f1_focused', 'balanced_accuracy', 'f1_macro', 'f1_balanced',
        'f1_unfavorable', 'f1_acceptable', 'f1_favorable',
        'roc_auc_macro', 'brier_score', 'ece', 'mce',
        'mean_predicted_probability', 'precision_macro', 'recall_macro'
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
                   'f1_macro', 'roc_auc_macro', 'brier_score']

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


def main():
    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Search for DFU Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
RECOMMENDATION:
  For most use cases, it is HIGHLY RECOMMENDED to optimize for just 'combined_score':
      python optuna_search.py --n-trials 500 --optimize-for combined_score

  The default multi-metric optimization runs 4 separate studies (4x trials total).

Examples:
  # Single metric (recommended for speed):
  python optuna_search.py --n-trials 500 --optimize-for combined_score

  # Multiple metrics (default):
  python optuna_search.py --n-trials 500

  # Custom metrics:
  python optuna_search.py --n-trials 500 --optimize-for combined_score,f1_macro,roc_auc_macro
        """
    )
    parser.add_argument('--n-trials', type=int, default=500,
                        help='Number of trials PER optimization target (default: 500)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for model training (default: -1 for all cores)')
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
                        help='Comma-separated list of metrics to optimize. '
                             'Default: combined_score,combined_score_imbalanced,combined_score_discrimination,roc_auc_macro. '
                             'Use --optimize-for combined_score for fastest single-target search.')

    # Early stopping arguments
    parser.add_argument('--early-stop-min-trials', type=int, default=250,
                        help='Minimum trials before early stopping can trigger (default: 250)')
    parser.add_argument('--early-stop-patience', type=int, default=100,
                        help='Trials to wait without significant improvement (default: 100)')
    parser.add_argument('--early-stop-threshold', type=float, default=0.001,
                        help='Minimum relative improvement required (default: 0.001 = 0.1%%)')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping')

    args = parser.parse_args()

    # Parse optimize-for argument
    if args.optimize_for:
        optimize_for = [m.strip() for m in args.optimize_for.split(',')]
    else:
        optimize_for = None  # Will use default

    # Set early stopping parameters (disable if --no-early-stop is set)
    early_stop_min_trials = args.n_trials + 1 if args.no_early_stop else args.early_stop_min_trials
    early_stop_patience = args.early_stop_patience
    early_stop_threshold = args.early_stop_threshold

    all_studies, best_results, all_results = run_optuna_search(
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
        early_stop_threshold=early_stop_threshold
    )

    print("\n" + "="*80)
    print("OPTUNA SEARCH COMPLETE!")
    print(f"Total trials evaluated: {len(all_results)}")
    print(f"Studies completed: {len(all_studies)}")
    print("="*80)


if __name__ == "__main__":
    main()
