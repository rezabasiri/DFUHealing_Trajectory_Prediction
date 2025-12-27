"""
Feature selection functions for hyperparameter optimization.

This module contains functions for computing feature importance and
selecting features based on various criteria.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

from ..config.constants import CLINICAL_ESSENTIAL_FEATURES


def compute_feature_importance(X_train, y_train, algorithm='GradientBoosting'):
    """
    Compute feature importance scores using the specified algorithm.

    Parameters
    ----------
    X_train : array-like
        Training data
    y_train : array-like
        Training labels
    algorithm : str, default='GradientBoosting'
        Algorithm to use for importance calculation

    Returns
    -------
    feature_importances : np.ndarray
        Feature importance scores
    """
    if algorithm == 'GradientBoosting':
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
    else:  # ExtraTrees
        model = ExtraTreesClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train, y_train)
    return model.feature_importances_


def select_features(feature_importances, feature_names, params,
                   feature_selection_method='importance_threshold',
                   min_features=10):
    """
    Select features based on importance and parameters.

    Parameters
    ----------
    feature_importances : np.ndarray
        Feature importance scores
    feature_names : list
        List of feature names
    params : dict
        Parameters dictionary containing selection criteria
    feature_selection_method : str, default='importance_threshold'
        Method for feature selection ('top_n' or 'importance_threshold')
    min_features : int, default=10
        Minimum number of features to keep

    Returns
    -------
    selected_features : list
        List of selected feature names
    selected_indices : list
        List of selected feature indices
    importance_df : pd.DataFrame
        DataFrame with feature importance information
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    # Always include clinical essential features
    selected_features = []
    for feat in CLINICAL_ESSENTIAL_FEATURES:
        if feat in feature_names:
            selected_features.append(feat)

    if feature_selection_method == 'top_n':
        n_to_select = params.get('n_features_to_select', len(feature_names))
        top_features = importance_df.head(n_to_select)['feature'].tolist()

        for feat in top_features:
            if feat not in selected_features:
                selected_features.append(feat)

        # Ensure minimum number of features
        if len(selected_features) < min_features:
            for feat in importance_df['feature'].tolist():
                if feat not in selected_features:
                    selected_features.append(feat)
                if len(selected_features) >= min_features:
                    break

    elif feature_selection_method == 'importance_threshold':
        threshold = params.get('importance_threshold', 0.8)

        # Calculate cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

        # Select features up to threshold
        threshold_features = importance_df[
            importance_df['cumulative_importance'] <= threshold
        ]['feature'].tolist()

        for feat in threshold_features:
            if feat not in selected_features:
                selected_features.append(feat)

        # Ensure minimum number of features
        if len(selected_features) < min_features:
            for feat in importance_df['feature'].tolist():
                if feat not in selected_features:
                    selected_features.append(feat)
                if len(selected_features) >= min_features:
                    break

    else:
        # No feature selection, use all features
        selected_features = list(feature_names)

    # Get indices of selected features
    selected_indices = [feature_names.index(feat) for feat in selected_features]

    return selected_features, selected_indices, importance_df
