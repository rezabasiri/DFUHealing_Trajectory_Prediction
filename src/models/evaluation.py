"""
Model evaluation functions for DFU healing prediction.

This module contains functions for analyzing and evaluating model predictions.
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def analyze_transitions(y_true, y_pred):
    """
    Analyze specific phase transitions.

    Parameters
    ----------
    y_true : array-like
        True phase labels
    y_pred : array-like
        Predicted phase labels

    Returns
    -------
    transition_metrics : dict
        Dictionary containing transition-specific metrics
    """
    transitions = {}

    # Create transition matrix
    for true_phase, pred_phase in zip(y_true, y_pred):
        key = f"{true_phase}→{pred_phase}"
        transitions[key] = transitions.get(key, 0) + 1

    # Calculate specific transition accuracies
    transition_metrics = {
        'I→P': transitions.get('0→1', 0) / max(sum(1 for y in y_true if y == 0), 1),
        'I→I': transitions.get('0→0', 0) / max(sum(1 for y in y_true if y == 0), 1),
        'P→P': transitions.get('1→1', 0) / max(sum(1 for y in y_true if y == 1), 1),
        'P→R': transitions.get('1→2', 0) / max(sum(1 for y in y_true if y == 1), 1),
        'R→R': transitions.get('2→2', 0) / max(sum(1 for y in y_true if y == 2), 1),
    }

    return transition_metrics


def convert_to_json_serializable(obj):
    """
    Convert numpy types to native Python types for JSON serialization.

    Parameters
    ----------
    obj : any
        Object to convert

    Returns
    -------
    converted : any
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def classify_transition(prev_phase, next_phase):
    """
    Classify a phase transition as Favorable, Acceptable, or Unfavorable.

    For first appointments (prev_phase=-1), we assume the wound started at
    Inflammatory phase:
    - If current phase is P or R → Favorable (progression from assumed I)
    - If current phase is I → Acceptable (no time-based thresholding in basic mode)

    Transition definitions:
    - Favorable: Progression in healing (I→P, P→R)
    - Acceptable: Maintaining current phase (I→I, P→P, R→R)
    - Unfavorable: Regression in healing (P→I, R→I, R→P)

    Parameters
    ----------
    prev_phase : int
        Previous phase (0=I, 1=P, 2=R, -1=unknown/first appointment)
    next_phase : int
        Next phase (0=I, 1=P, 2=R)

    Returns
    -------
    transition_class : int
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

    # Favorable transitions: moving forward in healing
    if (prev_phase == 0 and next_phase == 1) or (prev_phase == 1 and next_phase == 2):
        return 2  # Favorable

    # Acceptable transitions: maintaining current phase
    elif prev_phase == next_phase:
        return 1  # Acceptable

    # Unfavorable transitions: regression
    else:
        return 0  # Unfavorable


def convert_to_transition_labels(phases, prev_phases):
    """
    Convert phase labels to transition labels.

    Parameters
    ----------
    phases : array-like
        Phase labels (0=I, 1=P, 2=R)
    prev_phases : array-like
        Previous phase labels

    Returns
    -------
    transitions : np.ndarray
        Transition labels (0=Unfavorable, 1=Acceptable, 2=Favorable)
    """
    transitions = []
    for prev, next_phase in zip(prev_phases, phases):
        transitions.append(classify_transition(prev, next_phase))
    return np.array(transitions)


def calculate_transition_metrics(y_true_phases, y_pred_phases, prev_phases):
    """
    Calculate transition-based metrics from phase predictions.

    Parameters
    ----------
    y_true_phases : array-like
        True phase labels
    y_pred_phases : array-like
        Predicted phase labels
    prev_phases : array-like
        Previous phase labels

    Returns
    -------
    metrics : dict
        Dictionary with transition-based metrics
    """
    # Convert phases to transitions
    y_true_transitions = convert_to_transition_labels(y_true_phases, prev_phases)
    y_pred_transitions = convert_to_transition_labels(y_pred_phases, prev_phases)

    # Calculate metrics
    accuracy = accuracy_score(y_true_transitions, y_pred_transitions)
    balanced_acc = balanced_accuracy_score(y_true_transitions, y_pred_transitions)
    f1_weighted = f1_score(y_true_transitions, y_pred_transitions, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true_transitions, y_pred_transitions, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true_transitions, y_pred_transitions, average=None, zero_division=0)

    combined_score = (balanced_acc + f1_macro) / 2

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'combined_score': combined_score
    }


# ============================================================================
# Evidence-Based Time-Aware Transition Classification
# ============================================================================

def classify_transition_chronicity_aware(prev_phase, next_phase, days_to_next_appt, onset_days=None):
    """
    Classify a phase transition using evidence-based chronicity thresholds.

    For first appointments (prev_phase=-1), we assume the wound started at
    Inflammatory phase and use onset_days to determine the transition:
    - If current phase is P or R → Favorable (progression from I)
    - If current phase is I and onset_days > 21 → Unfavorable (chronic)
    - If current phase is I and onset_days <= 21 → Acceptable (early wound)

    For subsequent appointments, use days_to_next_appt to determine if stagnation
    is chronic (i.e., whether the wound stayed in the same phase too long).

    Evidence-based thresholds:
    - Inflammatory resolution: 21 days (3 weeks) - Guo & DiPietro 2010
    - Proliferative primary: 28 days (4 weeks) - Sheehan et al. 2003 (91% NPV)
    - Proliferative secondary: 56 days (8 weeks) - Warriner et al. 2011 (82% NPV)
    - Remodeling: No specific threshold (takes 12-24 months)

    Parameters
    ----------
    prev_phase : int
        Previous phase (0=I, 1=P, 2=R, -1=unknown/first appointment)
    next_phase : int
        Next phase (0=I, 1=P, 2=R)
    days_to_next_appt : float
        Days between current and next appointment (used for chronicity thresholds)
    onset_days : float, optional
        Days since wound onset (used only for first appointments when prev_phase=-1)

    Returns
    -------
    transition_class : int
        0=Unfavorable, 1=Acceptable, 2=Favorable

    References
    ----------
    - Sheehan et al. 2003: <50% PAR at 4 weeks → 9% vs 58% healing
    - Warriner et al. 2011: <90% PAR at 8 weeks → 19% vs 51% healing
    - Guo & DiPietro 2010: Normal inflammation 2-7 days, DFU weeks-months
    - SVS Guidelines 2016: Grade 1B recommendation for 4-week assessment
    """
    # Handle first appointments (prev_phase=-1)
    # Assume wound started at Inflammatory phase, use onset_days for chronicity
    if prev_phase < 0:
        if next_phase == 1 or next_phase == 2:
            # Progressed to P or R from assumed I start → Favorable
            return 2  # Favorable
        elif next_phase == 0:
            # Still in I phase at first appointment - use onset_days
            if onset_days is not None and onset_days > 21:
                return 0  # Unfavorable (chronic inflammation at presentation)
            else:
                return 1  # Acceptable (early wound or unknown onset)
        else:
            return 1  # Acceptable (fallback)

    # Favorable: Progression (always favorable regardless of chronicity)
    if (prev_phase == 0 and next_phase == 1) or (prev_phase == 1 and next_phase == 2):
        return 2  # Favorable

    # Stagnation: Apply evidence-based chronicity thresholds using days_to_next_appt
    elif prev_phase == next_phase:
        # Inflammatory phase stagnation (I→I)
        if prev_phase == 0:
            if days_to_next_appt > 21:  # >3 weeks between appointments
                return 0  # Unfavorable (chronic inflammation)
            else:
                return 1  # Acceptable (short interval)

        # Proliferative phase stagnation (P→P)
        elif prev_phase == 1:
            if days_to_next_appt > 28:  # >4 weeks between appointments
                return 0  # Unfavorable (stalled proliferation)
            else:
                return 1  # Acceptable (short interval)

        # Remodeling phase maintenance (R→R)
        elif prev_phase == 2:
            # No specific threshold; remodeling takes 12-24 months
            return 1  # Acceptable

    # Regression: Always unfavorable
    else:
        return 0  # Unfavorable


def convert_to_transition_labels_chronicity_aware(phases, prev_phases, days_to_next_appt, onset_days=None):
    """
    Convert phase labels to transition labels using chronicity information.

    Parameters
    ----------
    phases : array-like
        Phase labels (0=I, 1=P, 2=R)
    prev_phases : array-like
        Previous phase labels
    days_to_next_appt : array-like
        Days between appointments for each sample (used for chronicity thresholds)
    onset_days : array-like, optional
        Days since wound onset for each sample (used for first appointments)

    Returns
    -------
    transitions : np.ndarray
        Transition labels (0=Unfavorable, 1=Acceptable, 2=Favorable)
    """
    if onset_days is None:
        onset_days = [None] * len(phases)

    transitions = []
    for prev, next_phase, days, onset in zip(prev_phases, phases, days_to_next_appt, onset_days):
        transitions.append(classify_transition_chronicity_aware(prev, next_phase, days, onset))
    return np.array(transitions)


def calculate_transition_metrics_chronicity_aware(y_true_phases, y_pred_phases,
                                                   prev_phases, days_to_next_appt,
                                                   onset_days=None):
    """
    Calculate transition-based metrics using evidence-based chronicity thresholds.

    This implementation applies clinical evidence from DFU healing literature:
    - I→I with >21 days between appts → Unfavorable (chronic inflammation)
    - P→P with >28 days between appts → Unfavorable (stalled proliferation)
    - R→R → Acceptable (remodeling is prolonged by nature)
    - First appointments use onset_days for chronicity determination

    Parameters
    ----------
    y_true_phases : array-like
        True phase labels
    y_pred_phases : array-like
        Predicted phase labels
    prev_phases : array-like
        Previous phase labels
    days_to_next_appt : array-like
        Days between appointments for each sample
    onset_days : array-like, optional
        Days since wound onset for each sample (used for first appointments)

    Returns
    -------
    metrics : dict
        Dictionary with transition-based metrics including reclassification stats
    """
    # Convert phases to transitions using chronicity-aware classification
    y_true_transitions = convert_to_transition_labels_chronicity_aware(
        y_true_phases, prev_phases, days_to_next_appt, onset_days
    )
    y_pred_transitions = convert_to_transition_labels_chronicity_aware(
        y_pred_phases, prev_phases, days_to_next_appt, onset_days
    )

    # Calculate metrics
    accuracy = accuracy_score(y_true_transitions, y_pred_transitions)
    balanced_acc = balanced_accuracy_score(y_true_transitions, y_pred_transitions)
    f1_weighted = f1_score(y_true_transitions, y_pred_transitions, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true_transitions, y_pred_transitions, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true_transitions, y_pred_transitions, average=None, zero_division=0)

    combined_score = (balanced_acc + f1_macro) / 2

    # Calculate reclassification statistics (compare to basic classification)
    basic_transitions = convert_to_transition_labels(y_true_phases, prev_phases)
    reclassified_count = np.sum(y_true_transitions != basic_transitions)
    reclassified_pct = (reclassified_count / len(y_true_transitions)) * 100

    # Count class distribution
    unique, counts = np.unique(y_true_transitions, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'combined_score': combined_score,
        'reclassified_count': int(reclassified_count),
        'reclassified_pct': float(reclassified_pct),
        'class_distribution': class_distribution
    }
