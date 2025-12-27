"""
Preprocessing module for DFU Healing Trajectory Prediction.

This module contains all preprocessing components including data cleaning,
feature engineering, and resampling strategies.
"""

from .preprocessor import DFUNextAppointmentPreprocessor, classify_transition
from .resampler import FlexibleResampler
from .feature_engineering import create_temporal_features, create_patient_clusters
from .utils import filter_features_for_model
from .transition_aware_weighting import (
    compute_transition_weights,
    compute_transition_weights_chronicity_aware,
    compute_phase_weights_from_transitions,
    create_transition_aware_resampler,
    get_transition_stratified_folds,
    compute_transition_from_phases,
    compute_transition_from_phases_chronicity_aware,
    compute_transition_labels_chronicity_aware,
    compute_focal_weights,
    DEFAULT_THRESHOLDS
)

__all__ = [
    'DFUNextAppointmentPreprocessor',
    'FlexibleResampler',
    'create_temporal_features',
    'create_patient_clusters',
    'filter_features_for_model',
    'classify_transition',
    'compute_transition_weights',
    'compute_transition_weights_chronicity_aware',
    'compute_phase_weights_from_transitions',
    'create_transition_aware_resampler',
    'get_transition_stratified_folds',
    'compute_transition_from_phases',
    'compute_transition_from_phases_chronicity_aware',
    'compute_transition_labels_chronicity_aware',
    'compute_focal_weights',
    'DEFAULT_THRESHOLDS'
]
