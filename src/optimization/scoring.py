"""
Clinical scoring functions for DFU healing optimization.

This module contains functions for calculating clinical scores and
classifying healing outcomes.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def classify_outcome(current_phase, next_phase):
    """
    Classify transition outcome as Favorable, Acceptable, or Unfavorable.

    Parameters
    ----------
    current_phase : int
        Current healing phase (0=I, 1=P, 2=R)
    next_phase : int
        Next healing phase (0=I, 1=P, 2=R)

    Returns
    -------
    category : str
        Outcome category
    transition : str
        Transition description
    """
    # Favorable outcomes (healing progression)
    if (current_phase == 0 and next_phase == 1) or \
       (current_phase == 0 and next_phase == 2) or \
       (current_phase == 1 and next_phase == 2) or \
       (current_phase == 2 and next_phase == 2):
        return 'Favorable', f'{current_phase}→{next_phase}'

    # Acceptable outcomes (stable proliferation)
    elif current_phase == 1 and next_phase == 1:
        return 'Acceptable', f'{current_phase}→{next_phase}'

    # Unfavorable outcomes (regression or prolonged inflammation)
    else:
        return 'Unfavorable', f'{current_phase}→{next_phase}'


def convert_phases_to_transitions(y_current, y_next):
    """
    Convert phase predictions to transition categories.

    Parameters
    ----------
    y_current : array-like
        Current phase labels
    y_next : array-like
        Next phase labels

    Returns
    -------
    transition_labels : np.ndarray
        Transition labels as integers (Favorable=0, Acceptable=1, Unfavorable=2)
    """
    transition_labels = []
    for curr, next_phase in zip(y_current, y_next):
        category, _ = classify_outcome(curr, next_phase)
        if category == 'Favorable':
            transition_labels.append(0)
        elif category == 'Acceptable':
            transition_labels.append(1)
        else:  # Unfavorable
            transition_labels.append(2)
    return np.array(transition_labels)


def calculate_clinical_score_v2(y_true, y_pred, return_details=False):
    """
    Enhanced clinical score focusing on Favorable/Acceptable/Unfavorable outcomes.

    Parameters
    ----------
    y_true : array-like
        True phase labels
    y_pred : array-like
        Predicted phase labels
    return_details : bool, default=False
        If True, return detailed metrics

    Returns
    -------
    adjusted_score : float
        Adjusted clinical score
    details : dict, optional
        Detailed metrics (only if return_details=True)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Error weights matrix
    error_weights = np.array([
        [0.0, 1.0, 2.5],  # True I
        [2.0, 0.0, 1.5],  # True P
        [3.0, 2.0, 0.0]   # True R
    ])

    weighted_errors = cm * error_weights
    total_weighted_error = weighted_errors.sum()
    max_possible_error = len(y_true) * 3.0

    clinical_score = 1 - (total_weighted_error / max_possible_error)

    # Count outcome types
    favorable_count = 0
    acceptable_count = 0
    unfavorable_count = 0

    for i in range(len(y_true)):
        outcome, _ = classify_outcome(y_true[i], y_pred[i])
        if outcome == 'Favorable':
            favorable_count += 1
        elif outcome == 'Acceptable':
            acceptable_count += 1
        else:
            unfavorable_count += 1

    favorable_rate = favorable_count / len(y_true)
    unfavorable_rate = unfavorable_count / len(y_true)

    # Adjust score based on outcome distribution
    adjusted_score = clinical_score * (1 + 0.2 * favorable_rate - 0.3 * unfavorable_rate)

    if return_details:
        details = {
            'confusion_matrix': cm,
            'clinical_score': clinical_score,
            'adjusted_score': adjusted_score,
            'favorable_rate': favorable_rate,
            'acceptable_rate': acceptable_count / len(y_true),
            'unfavorable_rate': unfavorable_rate
        }
        return adjusted_score, details

    return adjusted_score
