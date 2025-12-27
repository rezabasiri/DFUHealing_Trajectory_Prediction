"""
Utility functions for preprocessing operations.
"""

from ..config.constants import SELECTED_BASE_FEATURES, ENGINEERED_FEATURES_TO_KEEP


def filter_features_for_model(df, feature_cols):
    """
    Filter features to only include the selected base features and their engineered versions.
    Exact implementation from RiskPredict_V12.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names to filter

    Returns
    -------
    filtered_cols : list
        List of filtered feature column names
    """
    filtered_cols = []

    for col in feature_cols:
        # Check if it's a base feature
        if col in SELECTED_BASE_FEATURES:
            filtered_cols.append(col)
        # Check if it's an engineered feature from a selected base feature
        else:
            # Check for consistency features
            for base_feat in SELECTED_BASE_FEATURES:
                if col.startswith(base_feat) and '_Consistency' in col:
                    filtered_cols.append(col)
                    break

            # Special engineered features to keep
            if col in ENGINEERED_FEATURES_TO_KEEP:
                filtered_cols.append(col)

    return filtered_cols
