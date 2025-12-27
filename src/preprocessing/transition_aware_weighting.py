"""
Transition-Aware Sample Weighting for DFU Healing Prediction

This module provides functions to compute sample weights that account for
the transition outcome (Favorable/Acceptable/Unfavorable) rather than just
the phase class (I/P/R).

The goal is to improve prediction of under-represented transitions,
particularly Favorable transitions (I→P, P→R) which are clinically important
but rare in the dataset.

Supports both basic and chronicity-aware transition classification:
- Basic: Simple direction-based classification
- Chronicity-Aware: Uses time thresholds to reclassify stagnation as unfavorable
"""

import numpy as np
from collections import Counter


# ============================================================================
# Default Chronicity Thresholds (configurable)
# ============================================================================

# Default thresholds based on clinical evidence
# These can be overridden by passing custom values to the functions
DEFAULT_THRESHOLDS = {
    'inflammatory_days': 21,    # I→I becomes Unfavorable after this many days
    'proliferative_days': 28,   # P→P becomes Unfavorable after this many days
    'remodeling_days': None,    # R→R is always Acceptable (no threshold)
}


def compute_transition_from_phases(prev_phase, next_phase):
    """
    Classify a phase transition as Favorable, Acceptable, or Unfavorable.

    For first appointments (prev_phase=-1), we assume the wound started at
    Inflammatory phase:
    - If current phase is P or R → Favorable (progression from assumed I)
    - If current phase is I → Acceptable (no time-based thresholding in basic mode)

    Parameters
    ----------
    prev_phase : int
        Previous phase (0=I, 1=P, 2=R, -1=unknown/first appointment)
    next_phase : int
        Next phase (0=I, 1=P, 2=R)

    Returns
    -------
    transition : int
        0=Unfavorable, 1=Acceptable, 2=Favorable
    """
    # Handle first appointments (prev_phase=-1)
    # Assume wound started at Inflammatory phase
    if prev_phase < 0:
        if next_phase == 1 or next_phase == 2:
            # Progressed to P or R from assumed I start → Favorable
            return 2  # Favorable
        else:
            # Still in I phase (or fallback) - Acceptable in basic mode (no time thresholds)
            return 1  # Acceptable

    # Favorable: progression (I→P, P→R)
    if (prev_phase == 0 and next_phase == 1) or (prev_phase == 1 and next_phase == 2):
        return 2  # Favorable

    # Acceptable: stagnation (I→I, P→P, R→R)
    elif prev_phase == next_phase:
        return 1  # Acceptable

    # Unfavorable: regression
    else:
        return 0  # Unfavorable


def compute_transition_from_phases_chronicity_aware(
    prev_phase, next_phase, days_to_next_appt,
    inflammatory_threshold=21, proliferative_threshold=28, remodeling_threshold=None,
    onset_days=None, cumulative_phase_duration=None,
    skip_first_appt_chronicity=False
):
    """
    Classify a phase transition using chronicity-aware thresholds.

    IMPORTANT: For same-phase transitions (I→I, P→P), this uses the
    CUMULATIVE duration in that phase, not just days between appointments.
    This correctly identifies chronic wounds that have been stuck in a phase
    for a long time across multiple appointments.

    For first appointments (prev_phase=-1), we assume the wound started at
    Inflammatory phase and use onset_days/cumulative_phase_duration to determine:
    - If current phase is P or R → Favorable (progression from I)
    - If current phase is I and duration > threshold → Unfavorable (chronic)
    - If current phase is I and duration <= threshold → Acceptable (early wound)

    Parameters
    ----------
    prev_phase : int
        Previous phase (0=I, 1=P, 2=R, -1=unknown/first appointment)
    next_phase : int
        Next phase (0=I, 1=P, 2=R)
    days_to_next_appt : float
        Days between current and next appointment (used as fallback)
    inflammatory_threshold : float
        Days threshold for I→I to become Unfavorable (default: 21)
    proliferative_threshold : float
        Days threshold for P→P to become Unfavorable (default: 28)
    remodeling_threshold : float or None
        Days threshold for R→R to become Unfavorable (default: None = never)
    onset_days : float, optional
        Days since wound onset (used for first appointments when prev_phase=-1)
    cumulative_phase_duration : float, optional
        CUMULATIVE time wound has been in prev_phase (from streak start or onset).
        This is the CORRECT value to use for I→I and P→P thresholds.
        If not provided, falls back to days_to_next_appt (legacy behavior).
    skip_first_appt_chronicity : bool, optional
        If True, skip chronicity check for first appointments (prev_phase=-1).
        I→I at first appointment will be Acceptable regardless of onset_days.
        This reduces class imbalance since onset values are often unreliable.
        Default: False (apply chronicity check).

    Returns
    -------
    transition : int
        0=Unfavorable, 1=Acceptable, 2=Favorable
    """
    # Handle first appointments (prev_phase=-1)
    # Assume wound started at Inflammatory phase
    if prev_phase < 0:
        if next_phase == 1 or next_phase == 2:
            # Progressed to P or R from assumed I start → Favorable
            return 2  # Favorable
        elif next_phase == 0:
            # Still in I phase at first appointment
            if skip_first_appt_chronicity:
                # Skip chronicity check - first appointments are always Acceptable
                # This is useful because onset values are often unreliable
                return 1  # Acceptable
            else:
                # Apply chronicity check using cumulative_phase_duration/onset_days
                duration = cumulative_phase_duration if cumulative_phase_duration is not None else onset_days
                if duration is not None and duration > inflammatory_threshold:
                    return 0  # Unfavorable (chronic inflammation at presentation)
                else:
                    return 1  # Acceptable (early wound or unknown onset)
        else:
            return 1  # Acceptable (fallback)

    # Favorable: progression (I→P, P→R) - always favorable regardless of time
    if (prev_phase == 0 and next_phase == 1) or (prev_phase == 1 and next_phase == 2):
        return 2  # Favorable

    # Stagnation: apply chronicity thresholds using CUMULATIVE duration
    elif prev_phase == next_phase:
        # Use cumulative_phase_duration if available, otherwise fall back to days_to_next_appt
        duration = cumulative_phase_duration if cumulative_phase_duration is not None else days_to_next_appt

        # I→I stagnation
        if prev_phase == 0:
            if inflammatory_threshold is not None and duration > inflammatory_threshold:
                return 0  # Unfavorable (chronic inflammation)
            else:
                return 1  # Acceptable (early wound, still healing)

        # P→P stagnation
        elif prev_phase == 1:
            if proliferative_threshold is not None and duration > proliferative_threshold:
                return 0  # Unfavorable (stalled proliferation)
            else:
                return 1  # Acceptable (still progressing)

        # R→R maintenance
        elif prev_phase == 2:
            if remodeling_threshold is not None and duration > remodeling_threshold:
                return 0  # Unfavorable (if threshold is set)
            else:
                return 1  # Acceptable (remodeling takes 12-24 months normally)

    # Unfavorable: regression (P→I, R→I, R→P)
    else:
        return 0  # Unfavorable


def compute_transition_labels_chronicity_aware(
    y_true, prev_phases, days_to_next_appt,
    inflammatory_threshold=21, proliferative_threshold=28, remodeling_threshold=None,
    onset_days=None, cumulative_phase_duration=None,
    skip_first_appt_chronicity=False
):
    """
    Convert phase labels to transition labels using chronicity-aware thresholds.

    NAMING CONVENTION:
    - prev_phases = phase at CURRENT appointment N (called "previous" relative to N+1)
    - y_true = phase at NEXT appointment N+1 (the future phase we predict)
    - Transition: prev_phases (N) → y_true (N+1)

    IMPORTANT: If cumulative_phase_duration is provided, it will be used for
    I→I and P→P transitions instead of days_to_next_appt. This correctly
    identifies chronic wounds that have been stuck in a phase across multiple
    consecutive appointments.

    Parameters
    ----------
    y_true : array-like
        Next phase labels at appointment N+1 (0=I, 1=P, 2=R)
    prev_phases : array-like
        Current phase at appointment N for each sample (named "prev" relative to N+1)
    days_to_next_appt : array-like
        Days between appointments for each sample (fallback if cumulative not available)
    inflammatory_threshold : float
        Days threshold for I→I to become Unfavorable
    proliferative_threshold : float
        Days threshold for P→P to become Unfavorable
    remodeling_threshold : float or None
        Days threshold for R→R to become Unfavorable
    onset_days : array-like, optional
        Days since wound onset for each sample (used for first appointments)
    cumulative_phase_duration : array-like, optional
        CUMULATIVE time wound has been in current phase for each sample.
        This is the correct value for I→I and P→P threshold comparison.
    skip_first_appt_chronicity : bool, optional
        If True, skip chronicity check for first appointments (prev_phase=-1).
        I→I at first appointment will be Acceptable regardless of onset_days.
        This reduces class imbalance since onset values are often unreliable.
        Default: False (apply chronicity check).

    Returns
    -------
    transition_labels : np.ndarray
        Transition labels (0=Unfavorable, 1=Acceptable, 2=Favorable)
    """
    if onset_days is None:
        onset_days = [None] * len(y_true)
    if cumulative_phase_duration is None:
        cumulative_phase_duration = [None] * len(y_true)

    transition_labels = np.array([
        compute_transition_from_phases_chronicity_aware(
            prev, next_ph, days,
            inflammatory_threshold, proliferative_threshold, remodeling_threshold,
            onset, cumul_dur, skip_first_appt_chronicity
        )
        for prev, next_ph, days, onset, cumul_dur in zip(
            prev_phases, y_true, days_to_next_appt, onset_days, cumulative_phase_duration
        )
    ])
    return transition_labels


def compute_transition_weights_chronicity_aware(
    y_true, prev_phases, days_to_next_appt,
    method='balanced', favorable_boost=2.0, unfavorable_boost=1.5,
    inflammatory_threshold=21, proliferative_threshold=28, remodeling_threshold=None
):
    """
    Compute sample weights based on chronicity-aware transition outcomes.

    This weights samples using transition labels that account for time-based
    reclassification of stagnation as unfavorable.

    Parameters
    ----------
    y_true : array-like
        True next phase labels (0=I, 1=P, 2=R)
    prev_phases : array-like
        Previous phase for each sample
    days_to_next_appt : array-like
        Days between appointments for each sample
    method : str
        Weighting method: 'balanced', 'favorable_boost', 'clinical'
    favorable_boost : float
        Multiplier for favorable transition weights
    unfavorable_boost : float
        Multiplier for unfavorable transition weights
    inflammatory_threshold : float
        Days threshold for I→I to become Unfavorable
    proliferative_threshold : float
        Days threshold for P→P to become Unfavorable
    remodeling_threshold : float or None
        Days threshold for R→R to become Unfavorable

    Returns
    -------
    weights : np.ndarray
        Sample weights for training
    transition_labels : np.ndarray
        Chronicity-aware transition labels
    """
    n_samples = len(y_true)

    # Compute chronicity-aware transition labels
    transition_labels = compute_transition_labels_chronicity_aware(
        y_true, prev_phases, days_to_next_appt,
        inflammatory_threshold, proliferative_threshold, remodeling_threshold
    )

    # Also compute basic labels for comparison
    basic_labels = np.array([
        compute_transition_from_phases(prev, next_ph)
        for prev, next_ph in zip(prev_phases, y_true)
    ])

    # Count how many were reclassified
    reclassified = np.sum(transition_labels != basic_labels)
    reclassified_pct = (reclassified / n_samples) * 100

    # Count transitions
    transition_counts = Counter(transition_labels)
    total = sum(transition_counts.values())

    print(f"  Chronicity-aware transition distribution:")
    print(f"    Thresholds: I→I>{inflammatory_threshold}d, P→P>{proliferative_threshold}d" +
          (f", R→R>{remodeling_threshold}d" if remodeling_threshold else ", R→R=always acceptable"))
    print(f"    Unfavorable: {transition_counts.get(0, 0)} ({transition_counts.get(0, 0)/total*100:.1f}%)")
    print(f"    Acceptable:  {transition_counts.get(1, 0)} ({transition_counts.get(1, 0)/total*100:.1f}%)")
    print(f"    Favorable:   {transition_counts.get(2, 0)} ({transition_counts.get(2, 0)/total*100:.1f}%)")
    print(f"    Reclassified from basic: {reclassified} ({reclassified_pct:.1f}%)")

    if method == 'balanced':
        weights = np.zeros(n_samples)
        for trans_class in [0, 1, 2]:
            count = transition_counts.get(trans_class, 1)
            class_weight = total / (3 * count) if count > 0 else 1.0
            weights[transition_labels == trans_class] = class_weight

    elif method == 'favorable_boost':
        # Use direct weight multipliers instead of stacking on balanced weights
        # This gives more intuitive control: favorable_boost=2 means Favorable
        # gets 2x the weight of Acceptable (baseline=1)
        weights = np.ones(n_samples)
        weights[transition_labels == 0] = unfavorable_boost  # Unfavorable
        weights[transition_labels == 1] = 1.0                 # Acceptable (baseline)
        weights[transition_labels == 2] = favorable_boost     # Favorable

    elif method == 'clinical':
        clinical_weights = {
            0: 2.0,   # Unfavorable
            1: 1.0,   # Acceptable
            2: 3.0    # Favorable
        }
        weights = np.array([clinical_weights[t] for t in transition_labels])

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to mean=1
    weights = weights / weights.mean()

    print(f"  Weight statistics:")
    print(f"    Unfavorable mean weight: {weights[transition_labels == 0].mean():.3f}")
    print(f"    Acceptable mean weight:  {weights[transition_labels == 1].mean():.3f}")
    print(f"    Favorable mean weight:   {weights[transition_labels == 2].mean():.3f}")

    return weights, transition_labels


def compute_transition_weights(y_true, prev_phases, method='balanced',
                                favorable_boost=2.0, unfavorable_boost=1.5):
    """
    Compute sample weights based on transition outcomes.

    This weights samples to balance transition classes, not just phase classes.
    Particularly useful for boosting Favorable transitions which are rare but
    clinically important.

    Parameters
    ----------
    y_true : array-like
        True next phase labels (0=I, 1=P, 2=R)
    prev_phases : array-like
        Previous phase for each sample
    method : str
        Weighting method:
        - 'balanced': Inverse frequency weighting for transitions
        - 'favorable_boost': Extra weight on favorable transitions
        - 'clinical': Weights based on clinical importance
    favorable_boost : float
        Multiplier for favorable transition weights (used in 'favorable_boost' method)
    unfavorable_boost : float
        Multiplier for unfavorable transition weights (used in 'favorable_boost' method)

    Returns
    -------
    weights : np.ndarray
        Sample weights for training
    transition_labels : np.ndarray
        Transition labels for each sample (0=Unfavorable, 1=Acceptable, 2=Favorable)
    """
    n_samples = len(y_true)
    transition_labels = np.array([
        compute_transition_from_phases(prev, next_ph)
        for prev, next_ph in zip(prev_phases, y_true)
    ])

    # Count transitions
    transition_counts = Counter(transition_labels)
    total = sum(transition_counts.values())

    print(f"  Transition distribution:")
    print(f"    Unfavorable: {transition_counts.get(0, 0)} ({transition_counts.get(0, 0)/total*100:.1f}%)")
    print(f"    Acceptable:  {transition_counts.get(1, 0)} ({transition_counts.get(1, 0)/total*100:.1f}%)")
    print(f"    Favorable:   {transition_counts.get(2, 0)} ({transition_counts.get(2, 0)/total*100:.1f}%)")

    if method == 'balanced':
        # Inverse frequency weighting
        weights = np.zeros(n_samples)
        for trans_class in [0, 1, 2]:
            count = transition_counts.get(trans_class, 1)
            class_weight = total / (3 * count) if count > 0 else 1.0
            weights[transition_labels == trans_class] = class_weight

    elif method == 'favorable_boost':
        # Use direct weight multipliers instead of stacking on balanced weights
        # This gives more intuitive control: favorable_boost=2 means Favorable
        # gets 2x the weight of Acceptable (baseline=1)
        weights = np.ones(n_samples)
        weights[transition_labels == 0] = unfavorable_boost  # Unfavorable
        weights[transition_labels == 1] = 1.0                 # Acceptable (baseline)
        weights[transition_labels == 2] = favorable_boost     # Favorable

    elif method == 'clinical':
        # Clinical importance weighting
        # Favorable transitions are most important (early intervention opportunity)
        # Unfavorable transitions need detection (prevent deterioration)
        # Acceptable is baseline
        clinical_weights = {
            0: 2.0,   # Unfavorable - important to detect
            1: 1.0,   # Acceptable - baseline
            2: 3.0    # Favorable - most important, rarest
        }
        weights = np.array([clinical_weights[t] for t in transition_labels])

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to mean=1
    weights = weights / weights.mean()

    print(f"  Weight statistics:")
    print(f"    Unfavorable mean weight: {weights[transition_labels == 0].mean():.3f}")
    print(f"    Acceptable mean weight:  {weights[transition_labels == 1].mean():.3f}")
    print(f"    Favorable mean weight:   {weights[transition_labels == 2].mean():.3f}")

    return weights, transition_labels


def compute_phase_weights_from_transitions(y_true, prev_phases, method='balanced'):
    """
    Compute phase-level class weights that account for transition outcomes.

    This is useful when using sklearn's class_weight parameter, which expects
    weights per class, not per sample.

    Parameters
    ----------
    y_true : array-like
        True next phase labels
    prev_phases : array-like
        Previous phase for each sample
    method : str
        Weighting method (see compute_transition_weights)

    Returns
    -------
    class_weight : dict
        Dictionary mapping phase class to weight
    """
    # Get sample weights
    sample_weights, transition_labels = compute_transition_weights(
        y_true, prev_phases, method=method
    )

    # Compute average weight per phase class
    class_weights = {}
    for phase_class in [0, 1, 2]:
        mask = y_true == phase_class
        if mask.sum() > 0:
            class_weights[phase_class] = sample_weights[mask].mean()
        else:
            class_weights[phase_class] = 1.0

    return class_weights


def create_transition_aware_resampler(X, y, prev_phases, target_ratio=None):
    """
    Create oversampled dataset that balances transition outcomes.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target phase labels
    prev_phases : array-like
        Previous phase for each sample
    target_ratio : dict, optional
        Target ratio for each transition class.
        Default balances to equal counts.

    Returns
    -------
    X_resampled : np.ndarray
        Resampled feature matrix
    y_resampled : np.ndarray
        Resampled target labels
    prev_phases_resampled : np.ndarray
        Resampled previous phases
    """
    from imblearn.over_sampling import RandomOverSampler

    # Compute transition labels
    transition_labels = np.array([
        compute_transition_from_phases(prev, next_ph)
        for prev, next_ph in zip(prev_phases, y)
    ])

    # Create combined label for resampling (prev_phase * 10 + transition)
    # This ensures we maintain the relationship between prev_phase and transition
    combined_labels = prev_phases * 10 + transition_labels

    # Oversample based on combined labels
    ros = RandomOverSampler(random_state=42)

    # Combine X with prev_phases for resampling
    X_combined = np.column_stack([X, prev_phases])

    X_combined_resampled, y_resampled = ros.fit_resample(X_combined, y)

    # Separate back
    X_resampled = X_combined_resampled[:, :-1]
    prev_phases_resampled = X_combined_resampled[:, -1].astype(int)

    return X_resampled, y_resampled, prev_phases_resampled


def get_transition_stratified_folds(X, y, prev_phases, n_splits=5, random_state=42):
    """
    Create cross-validation folds stratified by transition outcome.

    This ensures each fold has similar distribution of transition outcomes,
    not just phase outcomes.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target phase labels
    prev_phases : array-like
        Previous phase for each sample
    n_splits : int
        Number of folds
    random_state : int
        Random seed

    Returns
    -------
    folds : list of tuples
        List of (train_indices, val_indices) for each fold
    """
    from sklearn.model_selection import StratifiedKFold

    # Compute transition labels
    transition_labels = np.array([
        compute_transition_from_phases(prev, next_ph)
        for prev, next_ph in zip(prev_phases, y)
    ])

    # Stratify by transition
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = list(skf.split(X, transition_labels))

    return folds


# ============================================================================
# Focal Loss Implementation for Transition-Aware Training
# ============================================================================

def compute_focal_weights(y_pred_proba, y_true, gamma=2.0, alpha=None):
    """
    Compute focal loss weights to down-weight easy examples.

    Focal loss helps focus training on hard examples (misclassified or
    low-confidence predictions), which can improve performance on
    minority classes.

    Parameters
    ----------
    y_pred_proba : array, shape (n_samples, n_classes)
        Predicted probabilities
    y_true : array
        True labels
    gamma : float
        Focusing parameter. Higher values = more focus on hard examples.
        gamma=0 is equivalent to cross-entropy.
        gamma=2 is common default.
    alpha : array or None
        Class weights. If None, uses uniform weights.

    Returns
    -------
    focal_weights : np.ndarray
        Per-sample weights for focal loss
    """
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]

    if alpha is None:
        alpha = np.ones(n_classes)

    # Get predicted probability for true class
    pt = y_pred_proba[np.arange(n_samples), y_true]

    # Focal weight: (1 - pt)^gamma
    focal_weights = (1 - pt) ** gamma

    # Apply class weights
    class_weights = np.array([alpha[y] for y in y_true])
    focal_weights *= class_weights

    return focal_weights
