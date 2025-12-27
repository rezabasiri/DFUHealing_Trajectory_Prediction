"""
Ensemble Calibration Training Script (Aligned with Optuna Search)

This script trains an ensemble of models matching the optuna_search_feature_selection.py
process for reproducible metrics. It trains DIRECTLY on transition labels.

Key features:
- Direct transition label training (no phase→transition conversion needed)
- Dynamic feature selection (top_n method)
- Onset value correction for missing/outlier values
- Resampling support (smote, combined, oversample)
- Temperature scaling for post-hoc calibration
- Multi-seed ensemble for robust metrics
- Load configs from grid search JSON result files

Usage:
    # Use config.yaml (default)
    python train_ensemble_calibration.py --n-seeds 2 --n-folds 3

    # Load config from JSON file
    python train_ensemble_calibration.py --config-file grid_search_results/Dec10_results.json --config-name combined_score

    # Multiple configs from one file
    python train_ensemble_calibration.py -f results.json -c combined_score -c combined_score_clinical

    # Multiple files (uses all configs from each)
    python train_ensemble_calibration.py -f results1.json -f results2.json
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import yaml
import warnings
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, brier_score_loss
)
from scipy.optimize import minimize_scalar
from scipy.special import softmax

warnings.filterwarnings('ignore')

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
    FEATURES_TO_REMOVE,
)
from src.visualization import (
    generate_calibration_analysis,
    generate_shap_figures,
    SHAP_AVAILABLE,
    plot_feature_categories,
    plot_roc_curves,
)

# ============================================================================
# FIXED THRESHOLDS (matching search script)
# ============================================================================
I_THRESHOLD = 21   # Days in Inflammatory phase before I→I becomes Unfavorable
P_THRESHOLD = 42   # Days in Proliferative phase before P→P becomes Unfavorable
R_THRESHOLD = None # R→R is always Acceptable

# Essential features (always included)
ESSENTIAL_FEATURES = CLINICAL_ESSENTIAL_FEATURES + ['Onset (Days)']

# Columns to EXCLUDE from features (matching search script)
EXCLUDE_COLUMNS = list(set(
    FEATURES_TO_REMOVE + [
        'Patient#', 'DFU#', 'Appt#',
        'Next_Healing_Phase', 'Healing_Phase',
        'Healing Phase Abs',
        'Previous_Phase',  # DATA LEAKAGE
        'Type of Pain', 'Type of Pain Grouped',
        'Dressing',
    ]
))


# ============================================================================
# Onset Value Correction (matching search script)
# ============================================================================

def compute_median_onset_by_first_phase(df: pd.DataFrame, preprocessor) -> Dict[int, float]:
    """Compute median onset values grouped by phase at first appointment."""
    raw_df = preprocessor.df
    first_appts = []

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
        phase = first_row.get('Healing Phase Abs', np.nan)
        phase = parse_phase(phase)
        if not pd.isna(phase) and not pd.isna(onset):
            first_appts.append({'phase_at_appt0': int(phase), 'onset_days': float(onset)})

    if not first_appts:
        return {0: 90.0, 1: 90.0, 2: 90.0}

    first_appts_df = pd.DataFrame(first_appts)
    median_onset = first_appts_df.groupby('phase_at_appt0')['onset_days'].median().to_dict()

    for phase in [0, 1, 2]:
        if phase not in median_onset:
            median_onset[phase] = first_appts_df['onset_days'].median()

    return median_onset


def get_phase_at_first_appt_map(preprocessor) -> Dict[Tuple[int, int], int]:
    """Get phase at first appointment for each patient/DFU."""
    raw_df = preprocessor.df
    phase_map = {}

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
        phase = first_row.get('Healing Phase Abs', np.nan)
        phase = parse_phase(phase)
        if not pd.isna(phase):
            phase_map[(patient, dfu)] = int(phase)

    return phase_map


def replace_onset_with_median(df, preprocessor, median_onset_by_phase, phase_at_first_appt_map,
                               silent=False, outlier_std=2.0):
    """Replace missing/outlier onset values with median by phase."""
    onset_col = 'Onset (Days)'
    if onset_col not in df.columns:
        return df

    original_onset = df[onset_col].copy()
    new_onset = original_onset.copy()

    valid_onsets = original_onset.dropna()
    if len(valid_onsets) == 0:
        return df

    onset_mean = valid_onsets.mean()
    onset_std = valid_onsets.std()
    lower_bound = onset_mean - outlier_std * onset_std
    upper_bound = onset_mean + outlier_std * onset_std

    n_missing = 0
    n_outliers = 0

    for idx in df.index:
        patient = df.loc[idx, 'Patient#']
        dfu = df.loc[idx, 'DFU#']
        phase_at_first = phase_at_first_appt_map.get((patient, dfu), None)
        current_onset = original_onset.loc[idx]

        if phase_at_first is not None and phase_at_first in median_onset_by_phase:
            median_val = median_onset_by_phase[phase_at_first]

            if pd.isna(current_onset):
                new_onset.loc[idx] = median_val
                n_missing += 1
            elif current_onset < lower_bound or current_onset > upper_bound:
                new_onset.loc[idx] = median_val
                n_outliers += 1

    df[onset_col] = new_onset

    if not silent:
        print(f"  Onset correction: {n_missing} missing + {n_outliers} outliers replaced")
    return df


# ============================================================================
# Feature Selection (matching search script)
# ============================================================================

def select_features_by_top_n(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                             n_top: int = 30, essential_features: List[str] = None,
                             random_state: int = 42) -> Tuple[List[str], Dict]:
    """Select top N features by importance."""
    if essential_features is None:
        essential_features = []

    quick_model = ExtraTreesClassifier(
        n_estimators=100, max_depth=20, random_state=random_state, n_jobs=-1
    )
    quick_model.fit(X, y)

    importances = quick_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    selected_features = importance_df.head(n_top)['feature'].tolist()

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
# Temperature Scaling (matching search script)
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
# Calibration Metrics (matching search script)
# ============================================================================

def calculate_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   n_classes: int = 3) -> Dict[str, float]:
    """Calculate calibration metrics: Brier Score, ECE, MCE."""
    metrics = {'brier_scores': [], 'ece_scores': [], 'mce_scores': []}

    for class_idx in range(n_classes):
        y_binary = (y_true == class_idx).astype(int)
        y_prob = y_pred_proba[:, class_idx] if y_pred_proba.shape[1] > class_idx else np.zeros(len(y_true))
        if y_binary.sum() == 0:
            continue

        brier = brier_score_loss(y_binary, y_prob)
        metrics['brier_scores'].append(brier)

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
    }


def calculate_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray,
                      n_classes: int = 3) -> Dict[str, float]:
    """Calculate ROC-AUC scores."""
    results = {}
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
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


# ============================================================================
# Training with Calibration (matching search script)
# ============================================================================

def train_with_calibration(X_train: np.ndarray, y_train: np.ndarray,
                           sample_weights: np.ndarray, config: Dict,
                           random_state: int = 42, n_jobs: int = -1,
                           calibration_method: str = 'sigmoid') -> CalibratedClassifierCV:
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
# Config Loading from JSON
# ============================================================================

def load_configs_from_json(file_path: str, config_names: Optional[List[str]] = None,
                          include_selected_features: bool = False) -> List[Tuple[str, Dict, Optional[List[str]]]]:
    """
    Load configuration(s) from a grid search results JSON file.

    Parameters
    ----------
    file_path : str
        Path to JSON file with grid search results
    config_names : list of str, optional
        Specific config names to load. If None, loads all configs.
    include_selected_features : bool
        If True, also return selected_features from JSON (for --use-json-features)

    Returns
    -------
    list of (name, config_dict, selected_features) tuples
        selected_features is None if include_selected_features=False or not in JSON
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    configs = []
    file_stem = Path(file_path).stem  # For naming outputs

    if config_names:
        # Load specific configs
        for name in config_names:
            if name in data:
                cfg = data[name].get('config', {})
                selected_feats = data[name].get('selected_features') if include_selected_features else None
                configs.append((f"{file_stem}__{name}", cfg, selected_feats))
            else:
                print(f"  WARNING: Config '{name}' not found in {file_path}")
                print(f"  Available: {list(data.keys())}")
    else:
        # Load all configs
        for name, entry in data.items():
            if isinstance(entry, dict) and 'config' in entry:
                selected_feats = entry.get('selected_features') if include_selected_features else None
                configs.append((f"{file_stem}__{name}", entry['config'], selected_feats))

    return configs


def json_config_to_training_params(cfg: Dict) -> Dict:
    """
    Convert JSON config format to training parameters.

    Parameters
    ----------
    cfg : dict
        Config dict from JSON file (the 'config' sub-dict)

    Returns
    -------
    dict with keys: model_params, training_params, calibration_params
    """
    # Model parameters for ExtraTreesClassifier
    model_params = {
        'n_estimators': cfg.get('n_estimators', 800),
        'max_depth': cfg.get('max_depth', 50),
        'min_samples_split': cfg.get('min_samples_split', 5),
        'min_samples_leaf': cfg.get('min_samples_leaf', 1),
        'max_features': cfg.get('max_features', 1.0),
        'bootstrap': cfg.get('bootstrap', True),
        'class_weight': cfg.get('class_weight'),
        'criterion': cfg.get('criterion', 'log_loss'),
        'ccp_alpha': cfg.get('ccp_alpha', 0.0),
        'min_weight_fraction_leaf': cfg.get('min_weight_fraction_leaf', 0.0),
        'max_samples': cfg.get('max_samples'),
    }

    # Training parameters
    training_params = {
        'resampling': cfg.get('resampling', 'combined'),
        'augmentation': cfg.get('augmentation', 'safe_sequential'),
        'skip_first_appt_chronicity': cfg.get('skip_first_appt_chronicity', True),
        'n_seeds': cfg.get('n_seeds', 2),
    }

    # Transition weighting
    training_params['favorable_boost'] = cfg.get('favorable_boost', 1.0)
    training_params['unfavorable_boost'] = cfg.get('unfavorable_boost', 1.0)
    training_params['acceptable_boost'] = cfg.get('acceptable_boost', 1.0)

    # Feature selection
    training_params['use_feature_selection'] = cfg.get('use_feature_selection', True)
    training_params['n_top_features'] = cfg.get('n_top_features', 30)

    # Calibration parameters
    calibration_params = {
        'calibration_method': cfg.get('calibration_method', 'sigmoid'),
        'use_temperature_scaling': cfg.get('use_temperature_scaling', False),
        'temperature_method': cfg.get('temperature_method', 'ece'),
    }

    return {
        'model_params': model_params,
        'training_params': training_params,
        'calibration_params': calibration_params,
    }


# ============================================================================
# Main Training Function
# ============================================================================

def run_training(
    config_name: str,
    model_params: Dict,
    training_params: Dict,
    calibration_params: Dict,
    n_folds: int,
    n_seeds: int,
    base_seed: int,
    output_dir: Optional[Path],
    skip_figures: bool,
    preprocessor,
    df_processed: pd.DataFrame,
    median_onset_by_phase: Dict,
    phase_at_first_appt_map: Dict,
    all_feature_cols: List[str],
    seed_selection: str = 'average',
    json_selected_features: Optional[List[str]] = None,
) -> Dict:
    """
    Run training with a specific configuration.

    Parameters
    ----------
    seed_selection : str
        How to combine seeds: 'average' (average probabilities) or 'best_ece' (pick best by ECE)
    json_selected_features : list of str, optional
        Pre-selected features from JSON config. If provided, skips feature selection.

    Returns dict with final metrics.
    """
    # Extract params
    RESAMPLING = training_params.get('resampling', 'combined')
    SKIP_FIRST_APPT_CHRONICITY = training_params.get('skip_first_appt_chronicity', True)
    n_top_features = training_params.get('n_top_features', 30)
    favorable_boost = training_params.get('favorable_boost', 1.0)
    unfavorable_boost = training_params.get('unfavorable_boost', 1.0)

    calibration_method = calibration_params.get('calibration_method', 'sigmoid')
    use_temperature_scaling = calibration_params.get('use_temperature_scaling', False)

    print("\n" + "=" * 70)
    print(f"TRAINING CONFIG: {config_name}")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Seeds: {n_seeds}, Folds: {n_folds}, Seed selection: {seed_selection}")
    print(f"  Resampling: {RESAMPLING}")
    print(f"  Feature selection: top_{n_top_features}")
    print(f"  Calibration: {calibration_method}, temp_scaling={use_temperature_scaling}")
    print(f"  Thresholds: I>{I_THRESHOLD}d, P>{P_THRESHOLD}d")
    print(f"  Boosts: unfav={unfavorable_boost}, fav={favorable_boost}")

    # Get auxiliary columns
    prev_phase_col = 'Previous_Phase' if 'Previous_Phase' in df_processed.columns else 'Initial_Phase'

    # Cross-validation
    print(f"\nRunning {n_folds}-fold CV with {n_seeds}-seed ensemble...")

    unique_patients = df_processed['Patient#'].unique()
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)

    all_y_true = []
    all_y_pred_proba = []
    all_metrics = []

    # Store last fold data for SHAP analysis
    last_fold_data = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(unique_patients)):
        print(f"\n  Fold {fold + 1}/{n_folds}:")
        train_patients = unique_patients[train_idx]
        val_patients = unique_patients[val_idx]

        # Split data
        train_mask = df_processed['Patient#'].isin(train_patients)
        val_mask = df_processed['Patient#'].isin(val_patients)

        train_df = df_processed[train_mask].copy()
        val_df = df_processed[val_mask].copy()

        # Get auxiliary data
        train_prev_phases = train_df[prev_phase_col].values
        val_prev_phases = val_df[prev_phase_col].values
        train_days = train_df['Days_To_Next_Appt'].fillna(14).values
        val_days = val_df['Days_To_Next_Appt'].fillna(14).values

        # Get cumulative phase duration
        train_cumul_dur = train_df.get('Cumulative_Phase_Duration', pd.Series(np.zeros(len(train_df)))).values
        val_cumul_dur = val_df.get('Cumulative_Phase_Duration', pd.Series(np.zeros(len(val_df)))).values

        # Compute transition labels (DIRECT training)
        y_train = compute_transition_labels_chronicity_aware(
            train_df['Next_Healing_Phase'].values,
            train_prev_phases, train_days,
            inflammatory_threshold=I_THRESHOLD,
            proliferative_threshold=P_THRESHOLD,
            remodeling_threshold=R_THRESHOLD,
            cumulative_phase_duration=train_cumul_dur,
            skip_first_appt_chronicity=SKIP_FIRST_APPT_CHRONICITY
        )
        y_val = compute_transition_labels_chronicity_aware(
            val_df['Next_Healing_Phase'].values,
            val_prev_phases, val_days,
            inflammatory_threshold=I_THRESHOLD,
            proliferative_threshold=P_THRESHOLD,
            remodeling_threshold=R_THRESHOLD,
            cumulative_phase_duration=val_cumul_dur,
            skip_first_appt_chronicity=SKIP_FIRST_APPT_CHRONICITY
        )

        X_train = train_df[all_feature_cols].values
        X_val = val_df[all_feature_cols].values

        # Impute before feature selection (needed for tree model)
        imputer_fs = SimpleImputer(strategy='median')
        X_train_imputed_fs = imputer_fs.fit_transform(X_train)
        X_val_imputed_fs = imputer_fs.transform(X_val)

        # Feature selection: use JSON features if provided, otherwise compute
        if json_selected_features is not None:
            # Use pre-selected features from JSON config
            selected_features = [f for f in json_selected_features if f in all_feature_cols]
            if fold == 0:
                print(f"    Using {len(selected_features)} features from JSON config")
        else:
            # Feature selection on training data
            selected_features, _ = select_features_by_top_n(
                X_train_imputed_fs, y_train, all_feature_cols,
                n_top=n_top_features, essential_features=ESSENTIAL_FEATURES,
                random_state=base_seed
            )
        selected_indices = [all_feature_cols.index(f) for f in selected_features if f in all_feature_cols]

        X_train_selected = X_train_imputed_fs[:, selected_indices]
        X_val_selected = X_val_imputed_fs[:, selected_indices]

        if json_selected_features is None:
            print(f"    Selected {len(selected_features)} features")

        # Scale (already imputed above)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)

        # Compute sample weights
        sample_weights, _ = compute_transition_weights_chronicity_aware(
            y_train, train_prev_phases, train_days,
            method='favorable_boost',
            favorable_boost=favorable_boost,
            unfavorable_boost=unfavorable_boost,
            inflammatory_threshold=I_THRESHOLD,
            proliferative_threshold=P_THRESHOLD,
            remodeling_threshold=R_THRESHOLD
        )

        # Apply resampling
        if RESAMPLING != 'none':
            resampler = FlexibleResampler(strategy=RESAMPLING, random_state=base_seed)
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)
            weights_resampled = np.ones(len(y_train_resampled))
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
            weights_resampled = sample_weights

        print(f"    Train: {len(X_train_resampled)}, Val: {len(X_val_scaled)}")

        # Multi-seed ensemble
        seed_probas = []
        seed_models = []
        seed_eces = []
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx * 1000

            model = train_with_calibration(
                X_train_resampled, y_train_resampled, weights_resampled,
                model_params, random_state=seed, calibration_method=calibration_method
            )

            proba = model.predict_proba(X_val_scaled)
            if proba.shape[1] < 3:
                full_proba = np.zeros((len(proba), 3))
                for i, cls in enumerate(model.classes_):
                    full_proba[:, int(cls)] = proba[:, i]
                proba = full_proba

            seed_probas.append(proba)
            seed_models.append(model)

            # Compute ECE for this seed (for best_ece selection)
            if seed_selection == 'best_ece':
                seed_metrics = calculate_calibration_metrics(y_val, proba)
                seed_eces.append(seed_metrics['ece'])

        # Combine seeds based on selection method
        if seed_selection == 'best_ece' and n_seeds > 1:
            best_seed_idx = np.argmin(seed_eces)
            y_pred_proba = seed_probas[best_seed_idx]
            model = seed_models[best_seed_idx]  # Keep best model for SHAP
            # Show ECE range across seeds
            print(f"    Seed ECEs: min={min(seed_eces):.4f}, max={max(seed_eces):.4f}, range={max(seed_eces)-min(seed_eces):.4f}")
            print(f"    Selected seed {best_seed_idx + 1}/{n_seeds} (ECE: {seed_eces[best_seed_idx]:.4f})")
        else:
            # Average across seeds
            y_pred_proba = np.mean(seed_probas, axis=0)
            model = seed_models[-1]  # Use last model for SHAP

        # Temperature scaling on TRAINING predictions
        if use_temperature_scaling:
            train_probas = []
            for seed_idx in range(n_seeds):
                seed = base_seed + seed_idx * 1000
                model = train_with_calibration(
                    X_train_resampled, y_train_resampled, weights_resampled,
                    model_params, random_state=seed, calibration_method=calibration_method
                )
                train_proba = model.predict_proba(X_train_scaled[:min(500, len(X_train_scaled))])
                if train_proba.shape[1] < 3:
                    full_proba = np.zeros((len(train_proba), 3))
                    for i, cls in enumerate(model.classes_):
                        full_proba[:, int(cls)] = train_proba[:, i]
                    train_proba = full_proba
                train_probas.append(train_proba)

            train_avg_proba = np.mean(train_probas, axis=0)
            train_y_subset = y_train[:min(500, len(y_train))]
            temperature = optimize_temperature_ece(train_y_subset, train_avg_proba)
            y_pred_proba = apply_temperature_scaling(y_pred_proba, temperature)
            print(f"    Temperature: {temperature:.4f}")

        # Store for aggregation
        all_y_true.extend(y_val)
        all_y_pred_proba.append(y_pred_proba)

        # Fold metrics
        y_pred = np.argmax(y_pred_proba, axis=1)
        fold_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
            'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
            'f1_per_class': f1_score(y_val, y_pred, average=None, zero_division=0),
        }
        fold_metrics.update(calculate_calibration_metrics(y_val, y_pred_proba))
        fold_metrics.update(calculate_roc_auc(y_val, y_pred_proba))
        all_metrics.append(fold_metrics)

        print(f"    F1-macro: {fold_metrics['f1_macro']:.4f}, ECE: {fold_metrics['ece']:.4f}")

        # Store last fold data for SHAP analysis
        if fold == n_folds - 1:
            last_fold_data = {
                'model': model,
                'X_train': X_train_resampled,
                'X_val': X_val_scaled,
                'y_train': y_train_resampled,
                'y_val': y_val,
                'selected_features': selected_features,
            }

    # Aggregate results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)

    all_y_true = np.array(all_y_true)
    all_y_pred_proba = np.vstack(all_y_pred_proba)
    all_y_pred = np.argmax(all_y_pred_proba, axis=1)

    # Final metrics on aggregated predictions
    final_metrics = {
        'config_name': config_name,
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'balanced_accuracy': balanced_accuracy_score(all_y_true, all_y_pred),
        'f1_macro': f1_score(all_y_true, all_y_pred, average='macro', zero_division=0),
        'f1_unfavorable': f1_score(all_y_true, all_y_pred, average=None, zero_division=0)[0],
        'f1_acceptable': f1_score(all_y_true, all_y_pred, average=None, zero_division=0)[1],
        'f1_favorable': f1_score(all_y_true, all_y_pred, average=None, zero_division=0)[2],
    }
    final_metrics.update(calculate_calibration_metrics(all_y_true, all_y_pred_proba))
    final_metrics.update(calculate_roc_auc(all_y_true, all_y_pred_proba))

    f1_macro_std = np.std([m['f1_macro'] for m in all_metrics])
    ece_std = np.std([m['ece'] for m in all_metrics])

    print(f"\nClassification Metrics:")
    print(f"  F1-Macro:          {final_metrics['f1_macro']:.4f} +/- {f1_macro_std:.4f}")
    print(f"  F1-Unfavorable:    {final_metrics['f1_unfavorable']:.4f}")
    print(f"  F1-Acceptable:     {final_metrics['f1_acceptable']:.4f}")
    print(f"  F1-Favorable:      {final_metrics['f1_favorable']:.4f}")

    print(f"\nCalibration Metrics:")
    print(f"  ECE:               {final_metrics['ece']:.4f} +/- {ece_std:.4f}")
    print(f"  MCE:               {final_metrics['mce']:.4f}")
    print(f"  Brier Score:       {final_metrics['brier_score']:.4f}")

    print(f"\nROC-AUC Macro:       {final_metrics['roc_auc_macro']:.4f}")

    # Generate calibration analysis figures and CSV files
    if not skip_figures and output_dir:
        config_output_dir = output_dir / config_name
        print(f"\nGenerating figures in: {config_output_dir}")

        analysis_results = generate_calibration_analysis(
            y_true=all_y_true,
            y_pred_proba=all_y_pred_proba,
            output_dir=config_output_dir,
            method_name=config_name.replace('__', ': ')
        )

        # Generate ROC curves figure (Fig. 3)
        print("\n7. Creating ROC curves figure...")
        roc_results = plot_roc_curves(
            y_true=all_y_true,
            y_pred_proba=all_y_pred_proba,
            output_path=config_output_dir / 'phase_roc_curve',
        )
        print(f"    ROC-AUC: Unfav={roc_results['roc_auc_unfavorable']:.2f}, "
              f"Accept={roc_results['roc_auc_acceptable']:.2f}, "
              f"Fav={roc_results['roc_auc_favorable']:.2f}, "
              f"Avg={roc_results['roc_auc_average']:.2f}")

        # Generate feature categories figure (Fig. 1)
        if last_fold_data and 'selected_features' in last_fold_data:
            print("\n8. Creating feature categories figure...")
            feature_counts = plot_feature_categories(
                feature_names=last_fold_data['selected_features'],
                output_path=config_output_dir / 'feature_categories_distribution',
            )
            print(f"    Categories: " + ", ".join([f"{k}={v}" for k, v in feature_counts.items()]))

        # Generate SHAP figures
        if last_fold_data and SHAP_AVAILABLE:
            print("\nGenerating SHAP analysis figures...")
            # Combine train and val data for more comprehensive SHAP analysis
            X_combined = np.vstack([last_fold_data['X_train'], last_fold_data['X_val']])
            shap_figures = generate_shap_figures(
                model=last_fold_data['model'],
                X_data=X_combined,
                feature_names=last_fold_data['selected_features'],
                output_dir=config_output_dir,
                class_names=['Unfavorable', 'Acceptable', 'Favorable'],
                max_samples=500
            )
            if shap_figures:
                print(f"  Generated {len(shap_figures)} SHAP figures")

        # Save summary JSON
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'timestamp': timestamp_str,
            'config_name': config_name,
            'n_seeds': n_seeds,
            'n_folds': n_folds,
            'n_samples': len(all_y_true),
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in final_metrics.items() if k != 'config_name'},
            'model_params': model_params,
            'training_params': training_params,
            'calibration_params': calibration_params,
        }
        with open(config_output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {config_output_dir / 'summary.json'}")

    return final_metrics


def main():
    # Load default config
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train ensemble with calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config.yaml (default)
  python train_ensemble_calibration.py --n-seeds 2 --n-folds 3

  # Load config from JSON file
  python train_ensemble_calibration.py -f grid_search_results/Dec10_results.json -c combined_score

  # Multiple configs from one file
  python train_ensemble_calibration.py -f results.json -c combined_score -c combined_score_clinical

  # All configs from file
  python train_ensemble_calibration.py -f results.json --all-configs
        """
    )
    parser.add_argument('--n-seeds', type=int, default=None,
                        help='Number of seeds for ensemble (overrides config)')
    parser.add_argument('--n-folds', type=int, default=yaml_config.get('training', {}).get('n_folds', 3),
                        help='Number of cross-validation folds (default: 3)')
    parser.add_argument('--base-seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for figures and CSV files')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Skip generating figures and CSV files')
    parser.add_argument('--config-file', '-f', type=str, action='append', default=[],
                        help='JSON file(s) with grid search results (can specify multiple)')
    parser.add_argument('--config-name', '-c', type=str, action='append', default=[],
                        help='Config name(s) to use from JSON file (can specify multiple)')
    parser.add_argument('--all-configs', action='store_true',
                        help='Run all configs from specified JSON file(s)')
    parser.add_argument('--list-configs', action='store_true',
                        help='List available configs in JSON file(s) and exit')
    parser.add_argument('--seed-selection', type=str, default='average',
                        choices=['average', 'best_ece'],
                        help='How to combine seeds: average probabilities or pick best by ECE (default: average)')
    parser.add_argument('--no-temp-scaling', action='store_true',
                        help='Disable temperature scaling (can reduce ECE variance)')
    parser.add_argument('--use-json-features', action='store_true',
                        help='Use selected_features from JSON instead of re-running feature selection')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ENSEMBLE CALIBRATION TRAINING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Handle --list-configs
    if args.list_configs:
        if not args.config_file:
            print("ERROR: --list-configs requires --config-file")
            return
        for file_path in args.config_file:
            print(f"\n{file_path}:")
            with open(file_path, 'r') as f:
                data = json.load(f)
            for name in data.keys():
                if isinstance(data[name], dict) and 'config' in data[name]:
                    value = data[name].get('value', 'N/A')
                    print(f"  - {name} (score: {value:.4f})" if isinstance(value, float) else f"  - {name}")
        return

    # Build list of configs to run
    configs_to_run = []

    if args.config_file:
        # Load from JSON file(s)
        for file_path in args.config_file:
            print(f"\nLoading configs from: {file_path}")
            config_names = args.config_name if args.config_name and not args.all_configs else None
            loaded = load_configs_from_json(file_path, config_names,
                                           include_selected_features=args.use_json_features)
            configs_to_run.extend(loaded)
            print(f"  Loaded {len(loaded)} config(s)")
            if args.use_json_features:
                for name, cfg, feats in loaded:
                    if feats:
                        print(f"    {name}: {len(feats)} features from JSON")

        if not configs_to_run:
            print("ERROR: No configs loaded. Check file path and config names.")
            return
    else:
        # Use config.yaml
        print("\nUsing config.yaml (no --config-file specified)")

        # Extract config values
        data_config = yaml_config.get('data', {})
        training_config = yaml_config.get('training', {})
        model_config = yaml_config.get('model', {}).get('extratrees', {})
        cal_config = yaml_config.get('calibration', {})
        tw_config = yaml_config.get('transition_weighting', {})
        fs_config = yaml_config.get('feature_selection', {})

        model_params = {
            'n_estimators': model_config.get('n_estimators', 800),
            'max_depth': model_config.get('max_depth', 20),
            'min_samples_split': model_config.get('min_samples_split', 15),
            'min_samples_leaf': model_config.get('min_samples_leaf', 1),
            'max_features': model_config.get('max_features', 1.0),
            'bootstrap': model_config.get('bootstrap', True),
            'class_weight': model_config.get('class_weight', 'balanced'),
            'criterion': model_config.get('criterion', 'log_loss'),
            'ccp_alpha': model_config.get('ccp_alpha', 0.005),
            'min_weight_fraction_leaf': model_config.get('min_weight_fraction_leaf', 0.0),
            'max_samples': model_config.get('max_samples'),
        }

        training_params = {
            'resampling': training_config.get('resampling_strategy', 'combined'),
            'augmentation': training_config.get('augmentation_type', 'safe_sequential'),
            'skip_first_appt_chronicity': training_config.get('skip_first_appt_chronicity', True),
            'favorable_boost': tw_config.get('favorable_boost', 1.0),
            'unfavorable_boost': tw_config.get('unfavorable_boost', 1.5),
            'acceptable_boost': tw_config.get('acceptable_boost', 1.0),
            'use_feature_selection': fs_config.get('use_feature_selection', True),
            'n_top_features': fs_config.get('n_top_features', 30),
            'n_seeds': training_config.get('n_seeds', 2),
        }

        calibration_params = {
            'calibration_method': cal_config.get('method', 'sigmoid'),
            'use_temperature_scaling': cal_config.get('use_temperature_scaling', True),
            'temperature_method': cal_config.get('temperature_method', 'ece'),
        }

        configs_to_run.append(('config_yaml', {
            'model_params': model_params,
            'training_params': training_params,
            'calibration_params': calibration_params,
        }, None))  # No JSON features for yaml config

    # Data preprocessing (done once, shared across configs)
    print("\nPreprocessing data...")
    data_config = yaml_config.get('data', {})
    CSV_PATH = Path(data_config.get('csv_path', 'data/DataMaster_Processed_V12_WithMissing.csv'))

    preprocessor = DFUNextAppointmentPreprocessor(CSV_PATH)
    preprocessor.initial_cleaning()
    preprocessor.convert_categorical_to_numeric()
    preprocessor.create_temporal_features()

    N_PATIENT_CLUSTERS = yaml_config.get('training', {}).get('n_patient_clusters', 2)
    AUGMENTATION_TYPE = yaml_config.get('training', {}).get('augmentation_type', 'safe_sequential')

    df_processed, patient_cluster_map, _ = preprocessor.create_next_appointment_dataset_with_augmentation(
        n_patient_clusters=N_PATIENT_CLUSTERS,
        augmentation_type=AUGMENTATION_TYPE
    )

    # Compute onset correction maps
    print("  Computing onset correction maps...")
    median_onset_by_phase = compute_median_onset_by_first_phase(df_processed, preprocessor)
    phase_at_first_appt_map = get_phase_at_first_appt_map(preprocessor)
    df_processed = replace_onset_with_median(
        df_processed, preprocessor, median_onset_by_phase, phase_at_first_appt_map
    )

    # Get all numeric features
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    all_feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLUMNS]
    print(f"  Available features: {len(all_feature_cols)}")

    # Set up output directory
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('figures') / f'ensemble_calibration_{timestamp_str}'

    # Run training for each config
    all_results = []
    print(f"\nRunning {len(configs_to_run)} configuration(s)...")

    for config_name, config_dict, json_features in configs_to_run:
        # Parse config if from JSON
        if 'model_params' in config_dict:
            params = config_dict
        else:
            params = json_config_to_training_params(config_dict)

        # Override n_seeds from CLI if specified
        n_seeds = args.n_seeds if args.n_seeds else params['training_params'].get('n_seeds', 2)

        # Override temperature scaling from CLI if specified
        calibration_params = params['calibration_params'].copy()
        if args.no_temp_scaling:
            calibration_params['use_temperature_scaling'] = False

        result = run_training(
            config_name=config_name,
            model_params=params['model_params'],
            training_params=params['training_params'],
            calibration_params=calibration_params,
            n_folds=args.n_folds,
            n_seeds=n_seeds,
            base_seed=args.base_seed,
            output_dir=output_dir,
            skip_figures=args.skip_figures,
            preprocessor=preprocessor,
            df_processed=df_processed,
            median_onset_by_phase=median_onset_by_phase,
            phase_at_first_appt_map=phase_at_first_appt_map,
            all_feature_cols=all_feature_cols,
            seed_selection=args.seed_selection,
            json_selected_features=json_features if args.use_json_features else None,
        )
        all_results.append(result)

    # Summary across all configs
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY ACROSS ALL CONFIGS")
        print("=" * 70)
        print(f"\n{'Config':<40} {'ECE':>8} {'F1-Macro':>10} {'F1-Unfav':>10} {'ROC-AUC':>10}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['config_name']:<40} {r['ece']:>8.4f} {r['f1_macro']:>10.4f} {r['f1_unfavorable']:>10.4f} {r['roc_auc_macro']:>10.4f}")

        # Save comparison CSV
        if not args.skip_figures:
            comparison_df = pd.DataFrame(all_results)
            comparison_path = output_dir / 'comparison.csv'
            comparison_df.to_csv(comparison_path, index=False)
            print(f"\nSaved comparison: {comparison_path}")

    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
