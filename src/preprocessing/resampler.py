"""
Flexible resampling strategies for handling class imbalance in DFU data.

This module provides various resampling strategies to address class imbalance issues
in the training data while preserving the original logic.
"""

import numpy as np
import warnings

# Suppress specific sklearn deprecation warnings from imblearn
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class FlexibleResampler:
    """
    Flexible resampling with multiple strategies for handling class imbalance.

    Supports:
    - 'none': No resampling
    - 'smote': SMOTE (Synthetic Minority Over-sampling Technique)
    - 'oversample': Random oversampling
    - 'undersample': Random undersampling
    - 'combined': Combination of oversampling and undersampling
    """

    def __init__(self, strategy='combined', random_state=42):
        """
        Initialize the resampler.

        Parameters
        ----------
        strategy : str, default='combined'
            Resampling strategy to use
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        Apply the selected resampling strategy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        X_resampled : array-like
            Resampled training data
        y_resampled : array-like
            Resampled target values
        """

        if self.strategy == 'none':
            return X, y

        elif self.strategy == 'smote':
            unique, counts = np.unique(y, return_counts=True)
            min_class_count = np.min(counts)

            if min_class_count > 5:
                k_neighbors = min(5, min_class_count - 1)
                smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X, y)
            else:
                print(f"    Min class has only {min_class_count} samples, using random oversampling")
                ros = RandomOverSampler(random_state=self.random_state)
                X_resampled, y_resampled = ros.fit_resample(X, y)

        elif self.strategy == 'oversample':
            ros = RandomOverSampler(random_state=self.random_state)
            X_resampled, y_resampled = ros.fit_resample(X, y)

        elif self.strategy == 'undersample':
            rus = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_resampled = rus.fit_resample(X, y)

        elif self.strategy == 'combined':
            unique, counts = np.unique(y, return_counts=True)
            class_dict = dict(zip(unique, counts))

            target_count = int(np.median(counts))

            minority_classes = {cls: cnt for cls, cnt in class_dict.items() if cnt < target_count}
            majority_classes = {cls: cnt for cls, cnt in class_dict.items() if cnt > target_count}

            X_resampled, y_resampled = X.copy(), y.copy()

            # Oversample minority classes
            if minority_classes:
                oversample_strategy = {cls: target_count for cls in minority_classes.keys()}
                for cls, cnt in class_dict.items():
                    if cls not in minority_classes:
                        oversample_strategy[cls] = cnt

                ros = RandomOverSampler(sampling_strategy=oversample_strategy, random_state=self.random_state)
                X_resampled, y_resampled = ros.fit_resample(X_resampled, y_resampled)

            # Undersample majority classes
            if majority_classes:
                undersample_strategy = {cls: target_count for cls in majority_classes.keys()}
                for cls in unique:
                    if cls not in majority_classes:
                        undersample_strategy[cls] = target_count

                rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=self.random_state)
                X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

        else:
            raise ValueError(f"Unknown resampling strategy: {self.strategy}")

        return X_resampled, y_resampled
