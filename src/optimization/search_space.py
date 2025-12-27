"""
Hyperparameter search space definitions for model optimization.

This module defines the search spaces for different algorithms.
"""

from skopt.space import Real, Integer, Categorical


def get_search_space(algorithm, include_feature_selection=True, total_features=100,
                     feature_selection_method='importance_threshold',
                     min_features=10, max_features_to_select=100):
    """
    Get search space for the algorithm, optionally including feature selection.

    Parameters
    ----------
    algorithm : str
        Algorithm name ('GradientBoosting' or 'ExtraTrees')
    include_feature_selection : bool, default=True
        Whether to include feature selection parameters
    total_features : int, default=100
        Total number of available features
    feature_selection_method : str, default='importance_threshold'
        Feature selection method ('top_n' or 'importance_threshold')
    min_features : int, default=10
        Minimum number of features to select
    max_features_to_select : int, default=100
        Maximum number of features to select (for 'top_n' method)

    Returns
    -------
    space : list
        List of search space dimensions
    """
    base_space = {
        'GradientBoosting': [
            Integer(500, 800, name='n_estimators'),
            Integer(14, 15, name='max_depth'),
            Real(0.01, 0.015, name='learning_rate', prior='log-uniform'),
            Integer(3, 5, name='min_samples_split'),
            Integer(1, 3, name='min_samples_leaf'),
            Real(0.9, 1.0, name='subsample'),
            Categorical(['sqrt', 'log2'], name='max_features')
        ],
        'ExtraTrees': [
            Integer(50, 700, name='n_estimators'),
            Integer(10, 100, name='max_depth'),
            Integer(2, 60, name='min_samples_split'),
            Integer(1, 20, name='min_samples_leaf'),
            Categorical(['sqrt', 'log2', 0.3, 0.5, 0.7, None], name='max_features'),
            Categorical([True, False], name='bootstrap'),
            Categorical(['balanced', None], name='class_weight')
        ]
    }

    space = base_space[algorithm].copy()

    # Add feature selection parameters if enabled
    if include_feature_selection:
        if feature_selection_method == 'top_n':
            max_features = min(max_features_to_select, total_features)
            space.append(Integer(min_features, max_features, name='n_features_to_select'))
        elif feature_selection_method == 'importance_threshold':
            space.append(Real(0.5, 0.95, name='importance_threshold'))

    return space


def create_model_from_params(algorithm, params):
    """
    Create a model instance based on algorithm type and parameters.

    Parameters
    ----------
    algorithm : str
        Algorithm name ('GradientBoosting' or 'ExtraTrees')
    params : dict
        Model hyperparameters

    Returns
    -------
    model : sklearn estimator
        Model instance
    """
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

    # Remove feature selection parameters
    model_params = params.copy()
    model_params.pop('n_features_to_select', None)
    model_params.pop('importance_threshold', None)

    if algorithm == 'GradientBoosting':
        return GradientBoostingClassifier(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            learning_rate=model_params['learning_rate'],
            min_samples_split=model_params['min_samples_split'],
            min_samples_leaf=model_params['min_samples_leaf'],
            subsample=model_params['subsample'],
            max_features=model_params['max_features'],
            random_state=42
        )
    elif algorithm == 'ExtraTrees':
        return ExtraTreesClassifier(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            min_samples_split=model_params['min_samples_split'],
            min_samples_leaf=model_params['min_samples_leaf'],
            max_features=model_params['max_features'],
            bootstrap=model_params['bootstrap'],
            class_weight=model_params['class_weight'],
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
