"""
Comprehensive preprocessor for DFU next appointment prediction.

This module contains the main preprocessing class that handles data cleaning,
categorical to numeric conversion, and dataset creation with augmentation.
All preprocessing steps are preserved exactly as in the original implementation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

from ..config.constants import (
    TARGET_COLUMN, ORDINAL_MAPPING, REVERSE_MAPPING,
    FEATURES_TO_REMOVE, STRING_TO_NUMERIC_MAPPINGS, ORDINAL_MAPPINGS,
    CATEGORICAL_COLUMNS, INTEGER_COLUMNS, NUMERICAL_COLUMNS,
    TREATMENT_CONSISTENCY_FEATURES, OFFLOADING_FEATURES
)
from .feature_engineering import create_temporal_features, create_patient_clusters


def classify_transition(current_phase, next_phase):
    """
    Classify the transition between phases.

    Parameters
    ----------
    current_phase : int
        Current healing phase (0=I, 1=P, 2=R)
    next_phase : int
        Next healing phase (0=I, 1=P, 2=R)

    Returns
    -------
    category : str
        Transition category ('Favorable', 'Acceptable', 'Unfavorable')
    transition_label : str
        Detailed transition label
    """
    # Favorable Outcomes
    favorable_transitions = {
        (0, 1): 'Improving: I→P',
        (0, 2): 'Improving: I→R',
        (1, 2): 'Improving: P→R',
        (2, 2): 'Stable-Good: R→R'
    }

    # Acceptable Outcomes
    acceptable_transitions = {
        (1, 1): 'Stable-Acceptable: P→P'
    }

    # Unfavorable Outcomes
    unfavorable_transitions = {
        (1, 0): 'Worsening: P→I',
        (2, 0): 'Worsening: R→I',
        (2, 1): 'Worsening: R→P',
        (0, 0): 'Stable-Poor: I→I'
    }

    key = (current_phase, next_phase)
    if key in favorable_transitions:
        return 'Favorable', favorable_transitions[key]
    elif key in acceptable_transitions:
        return 'Acceptable', acceptable_transitions[key]
    elif key in unfavorable_transitions:
        return 'Unfavorable', unfavorable_transitions[key]
    else:
        return 'Unknown', f"{current_phase}→{next_phase}"


class DFUNextAppointmentPreprocessor:
    """
    Comprehensive preprocessor for next appointment prediction.

    This class handles all preprocessing steps including:
    - Data cleaning
    - Categorical to numeric conversion
    - Temporal feature engineering
    - Patient clustering
    - Data augmentation with various strategies
    """

    def __init__(self, csv_path):
        """
        Initialize the preprocessor.

        Parameters
        ----------
        csv_path : str
            Path to the CSV data file
        """
        self.df = pd.read_csv(csv_path)
        self.target_col = TARGET_COLUMN
        self.ordinal_mapping = ORDINAL_MAPPING
        self.reverse_mapping = REVERSE_MAPPING
        self.label_encoders = {}

        # Configuration from constants
        self.features_to_remove = FEATURES_TO_REMOVE
        self.string_to_numeric_mappings = STRING_TO_NUMERIC_MAPPINGS
        self.ordinal_mappings = ORDINAL_MAPPINGS
        self.categorical_columns = CATEGORICAL_COLUMNS
        self.integer_columns = INTEGER_COLUMNS
        self.numerical_columns = NUMERICAL_COLUMNS

    def initial_cleaning(self):
        """
        Perform initial data cleaning.

        Returns
        -------
        df : pd.DataFrame
            Cleaned dataframe
        """
        print("  Performing initial data cleaning...")

        # Drop specified columns
        cols_to_drop = [col for col in self.features_to_remove if col in self.df.columns]
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)

        # Sort by patient, DFU, and appointment number
        self.df = self.df.sort_values(['Patient#', 'DFU#', 'Appt#'])

        print(f"    Data shape after cleaning: {self.df.shape}")
        return self.df

    def convert_categorical_to_numeric(self):
        """
        Convert categorical columns to numeric.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with converted columns
        """
        print("  Converting categorical columns to numeric...")

        # Handle columns with specific binary/ordinal mappings
        for col, mapping in self.string_to_numeric_mappings.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping).fillna(0).astype(int)

        # Handle ordinal columns
        for col, mapping in self.ordinal_mappings.items():
            if col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].str.lower()
                    self.df[col] = self.df[col].map(mapping).fillna(
                        list(mapping.values())[len(mapping)//2]
                    ).astype(int)

        # Handle remaining categorical columns with label encoding
        for col in self.categorical_columns:
            if col in self.df.columns and self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna('missing')
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le

        # Convert float columns that should be integers
        for col in self.integer_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)

        # Encode target variable
        if self.target_col in self.df.columns:
            self.df[self.target_col] = self.df[self.target_col].str.strip().map(self.ordinal_mapping)
            self.df[self.target_col] = self.df[self.target_col].fillna(1).astype(int)

        return self.df

    def create_temporal_features(self):
        """
        Create temporal features using the feature engineering module.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with temporal features
        """
        print("  Creating temporal features...")
        self.df = create_temporal_features(self.df)
        return self.df

    def create_patient_clusters(self, df, n_clusters=2):
        """
        Create patient clusters using the feature engineering module.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        n_clusters : int, default=2
            Number of clusters to create

        Returns
        -------
        patient_cluster_map : dict
            Mapping from patient ID to cluster label
        kmeans_model : KMeans
            Fitted KMeans model
        """
        return create_patient_clusters(df, self.target_col, n_clusters)

    def create_next_appointment_dataset_with_augmentation(self, n_patient_clusters=2, augmentation_type='all_combinations'):
        """
        Create dataset with different augmentation strategies.

        Parameters
        ----------
        n_patient_clusters : int, default=2
            Number of patient clusters
        augmentation_type : str, default='all_combinations'
            Augmentation strategy:
            - 'none': No augmentation, only sequential samples
            - 'safe_sequential': Only direct sequential augmentation
            - 'all_combinations': All possible combinations (most aggressive)

        Returns
        -------
        result_df : pd.DataFrame
            Processed and augmented dataset
        patient_cluster_map : dict
            Patient cluster assignments
        kmeans_model : KMeans
            Fitted clustering model
        """
        print(f"  Creating dataset with augmentation type: {augmentation_type}")
        processed_samples = []

        patient_cluster_map, kmeans_model = self.create_patient_clusters(self.df, n_clusters=n_patient_clusters)

        total_original = 0
        total_augmented = 0

        for (patient, dfu), group in self.df.groupby(['Patient#', 'DFU#']):
            group = group.sort_values('Appt#').reset_index(drop=True)

            if len(group) < 2:
                continue

            patient_cluster = patient_cluster_map.get(patient, 'Unknown')

            if augmentation_type == 'none':
                # No augmentation - only sequential samples
                for target_idx in range(1, len(group)):
                    history_indices = tuple(range(target_idx))
                    sample = self._create_sample_from_appointments(
                        group, history_indices, target_idx, patient_cluster, patient_cluster_map
                    )
                    if sample is not None:
                        processed_samples.append(sample)
                        total_original += 1

            elif augmentation_type == 'safe_sequential':
                # Safe sequential augmentation - only use contiguous sequences
                for target_idx in range(1, len(group)):
                    for start_idx in range(target_idx):
                        history_indices = tuple(range(start_idx, target_idx))
                        sample = self._create_sample_from_appointments(
                            group, history_indices, target_idx, patient_cluster, patient_cluster_map
                        )
                        if sample is not None:
                            processed_samples.append(sample)
                            if start_idx == 0:
                                total_original += 1
                            else:
                                total_augmented += 1

            elif augmentation_type == 'all_combinations':
                # All combinations augmentation (most aggressive)
                for target_idx in range(1, len(group)):
                    target_appt = group.iloc[target_idx]
                    historical_indices = list(range(target_idx))

                    for subset_size in range(1, len(historical_indices) + 1):
                        for hist_combo in combinations(historical_indices, subset_size):
                            # CRITICAL FIX: Include recent appointment for continuity
                            if (target_idx - 1) not in hist_combo and subset_size < len(historical_indices):
                                extended_combo = hist_combo + (target_idx - 1,)
                                if len(extended_combo) <= len(historical_indices):
                                    hist_combo = tuple(sorted(extended_combo))

                            sample = self._create_sample_from_appointments(
                                group, hist_combo, target_idx, patient_cluster, patient_cluster_map
                            )

                            if sample is not None:
                                processed_samples.append(sample)

                                if len(hist_combo) == target_idx:
                                    total_original += 1
                                else:
                                    total_augmented += 1
            else:
                raise ValueError(f"Unknown augmentation type: {augmentation_type}")

        result_df = pd.DataFrame(processed_samples)

        # Remove target column if it exists in features
        if self.target_col in result_df.columns:
            result_df = result_df.drop(columns=[self.target_col])

        print(f"    Created {len(result_df)} total samples")
        if augmentation_type != 'none':
            print(f"      Original sequential: {total_original}")
            print(f"      Augmented: {total_augmented}")
            print(f"      Augmentation factor: {len(result_df) / max(total_original, 1):.2f}x")

        return result_df, patient_cluster_map, kmeans_model

    def _create_sample_from_appointments(self, group, history_indices, target_idx, patient_cluster, patient_cluster_map=None):
        """
        Create a single training sample from selected appointments.

        Parameters
        ----------
        group : pd.DataFrame
            Group of appointments for a single DFU
        history_indices : tuple
            Indices of historical appointments to use
        target_idx : int
            Index of target appointment
        patient_cluster : str
            Patient cluster label
        patient_cluster_map : dict, optional
            Full patient cluster mapping

        Returns
        -------
        sample : dict or None
            Dictionary containing the sample features and target
        """
        current_idx = max(history_indices)
        current_appt = group.iloc[current_idx].to_dict()
        next_appt = group.iloc[target_idx]

        # Add patient cluster features
        current_appt['Patient_Cluster'] = patient_cluster
        current_appt['Patient_Cluster_Fast_Healer'] = int(patient_cluster == 'Fast_Healer')
        current_appt['Patient_Cluster_Slow_Healer'] = int(patient_cluster == 'Slow_Healer')

        if patient_cluster == 'Moderate_Healer':
            current_appt['Patient_Cluster_Moderate_Healer'] = 1
        elif patient_cluster_map and 'Moderate_Healer' in patient_cluster_map.values():
            current_appt['Patient_Cluster_Moderate_Healer'] = 0

        # CRITICAL: Days to next appointment
        current_appt['Days_To_Next_Appt'] = next_appt['Appt Days'] - current_appt['Appt Days']

        # Add History_Length and History_Completeness
        current_appt['History_Length'] = len(history_indices)
        current_appt['History_Completeness'] = len(history_indices) / target_idx

        # Add comprehensive historical features
        if len(history_indices) > 1:
            history = group.iloc[list(history_indices)]

            # Process selected features
            if 'Exudate Amount (None:0,Minor,Medium,Severe:3)' in history.columns:
                col = 'Exudate Amount (None:0,Minor,Medium,Severe:3)'
                valid_values = history[col].dropna()
                if len(valid_values) > 0:
                    current_appt[f'{col}_History_Mean'] = valid_values.mean()
                    if pd.notna(current_appt[col]):
                        consistency = (history[col] == current_appt[col]).mean()
                        current_appt[f'{col}_Consistency'] = consistency

            if 'Pain Level' in history.columns:
                col = 'Pain Level'
                valid_values = history[col].dropna()
                if len(valid_values) > 0:
                    if len(valid_values) > 1:
                        recent_trend = current_appt[col] - valid_values.mean() if pd.notna(current_appt[col]) else 0
                        current_appt[f'{col}_Trend_Direction'] = np.sign(recent_trend)
                    else:
                        current_appt[f'{col}_Trend_Direction'] = 0

            # Treatment consistency features
            for col in TREATMENT_CONSISTENCY_FEATURES:
                if col in history.columns and col in current_appt:
                    current_treatment = current_appt[col]
                    if pd.notna(current_treatment):
                        consistency = (history[col] == current_treatment).mean()
                        current_appt[f'{col}_Consistency'] = consistency

            # Healing phase progression features
            if self.target_col in history.columns:
                phase_history = history[self.target_col].values
                # NOTE: "Previous_Phase" = phase at CURRENT appointment N
                # Named "previous" relative to Next_Healing_Phase at appointment N+1
                # Transition computed: Previous_Phase (N) → Next_Healing_Phase (N+1)
                # EXCLUDED from training features to prevent data leakage
                current_appt['Previous_Phase'] = phase_history[-1]
                current_appt['Initial_Phase'] = phase_history[0]

                for phase in [0, 1, 2]:
                    current_appt[f'History_Phase_{phase}_Count'] = (phase_history == phase).sum()
                    current_appt[f'History_Phase_{phase}_Proportion'] = (phase_history == phase).mean()

                # Transition counts
                if len(phase_history) > 1:
                    favorable_count = 0
                    acceptable_count = 0
                    unfavorable_count = 0

                    for j in range(1, len(phase_history)):
                        category, _ = classify_transition(phase_history[j-1], phase_history[j])
                        if category == 'Favorable':
                            favorable_count += 1
                        elif category == 'Acceptable':
                            acceptable_count += 1
                        elif category == 'Unfavorable':
                            unfavorable_count += 1

                    current_appt['History_Favorable_Transitions'] = favorable_count
                    current_appt['History_Acceptable_Transitions'] = acceptable_count
                    current_appt['History_Unfavorable_Transitions'] = unfavorable_count

                    current_appt['Phase_Improvements_Count'] = sum((phase_history[j] > phase_history[j-1])
                                                                  for j in range(1, len(phase_history)))
                    current_appt['Phase_Regressions_Count'] = sum((phase_history[j] < phase_history[j-1])
                                                                 for j in range(1, len(phase_history)))

                    recent_phases = phase_history[-min(3, len(phase_history)):]
                    if len(recent_phases) > 1:
                        healing_momentum = np.mean(np.diff(recent_phases))
                    else:
                        healing_momentum = 0
                    current_appt['Healing_Momentum'] = healing_momentum

                # CRITICAL: Compute Cumulative_Phase_Duration
                # This tracks how long the wound has been in the PREVIOUS phase
                # (the phase before the transition we're predicting)
                prev_phase = phase_history[-1]
                appt_days_history = history['Appt Days'].values

                # Find when the current phase streak started by looking backward
                streak_start_idx = len(phase_history) - 1
                for j in range(len(phase_history) - 2, -1, -1):
                    if phase_history[j] == prev_phase:
                        streak_start_idx = j
                    else:
                        break  # Phase changed, streak started at j+1

                # Get days at the next appointment (target)
                target_appt_days = next_appt['Appt Days']

                if streak_start_idx == 0:
                    # Phase streak started at first appointment
                    # For I phase: this means wound was in I since onset
                    # Use Onset (Days) from the TARGET appointment for total duration
                    onset_at_target = next_appt.get('Onset (Days)', None)
                    if prev_phase == 0 and pd.notna(onset_at_target):
                        # Wound has been in I since onset - use onset_days at target
                        cumulative_duration = onset_at_target
                    else:
                        # For P or R that started at first appt, use time from first appt to target
                        cumulative_duration = target_appt_days - appt_days_history[0]
                else:
                    # Phase streak started mid-sequence
                    # Calculate time from streak start to target appointment
                    streak_start_appt_days = appt_days_history[streak_start_idx]
                    cumulative_duration = target_appt_days - streak_start_appt_days

                current_appt['Cumulative_Phase_Duration'] = cumulative_duration

            current_appt['Appointments_So_Far'] = len(history_indices)

            # Calculate appointment intervals
            appt_days = history['Appt Days'].values
            if len(appt_days) > 1:
                intervals = np.diff(sorted(appt_days))
                if len(intervals) > 0:
                    current_appt['Avg_Days_Between_Appts'] = intervals.mean()
                    current_appt['Std_Days_Between_Appts'] = intervals.std() if len(intervals) > 1 else 0
                else:
                    current_appt['Avg_Days_Between_Appts'] = 0
                    current_appt['Std_Days_Between_Appts'] = 0
            else:
                current_appt['Avg_Days_Between_Appts'] = 0
                current_appt['Std_Days_Between_Appts'] = 0

            # Treatment intensity score
            treatment_intensity = 0
            for feat in OFFLOADING_FEATURES:
                if feat in current_appt:
                    treatment_intensity += current_appt.get(feat, 0)

            if 'Dressing Grouped' in current_appt:
                treatment_intensity += current_appt['Dressing Grouped']

            current_appt['Treatment_Intensity_Score'] = treatment_intensity

            if 'Previous_Phase' in current_appt:
                current_appt['Previous_Phase_X_Treatment'] = (
                    current_appt['Previous_Phase'] * current_appt['Treatment_Intensity_Score']
                )

        else:
            # First appointment - set defaults
            current_appt['Previous_Phase'] = -1
            current_appt['Initial_Phase'] = current_appt.get(self.target_col, -1)
            current_appt['Appointments_So_Far'] = 1
            current_appt['Treatment_Intensity_Score'] = 0
            current_appt['Healing_Momentum'] = 0
            current_appt['Previous_Phase_X_Treatment'] = 0
            current_appt['History_Favorable_Transitions'] = 0
            current_appt['History_Acceptable_Transitions'] = 0
            current_appt['History_Unfavorable_Transitions'] = 0

            # For first appointment, cumulative phase duration is onset_days at target
            # (assuming wound started at I phase at onset)
            onset_at_target = next_appt.get('Onset (Days)', None)
            if pd.notna(onset_at_target):
                current_appt['Cumulative_Phase_Duration'] = onset_at_target
            else:
                # If onset_days not available, use days_to_next_appt as fallback
                current_appt['Cumulative_Phase_Duration'] = next_appt['Appt Days'] - current_appt['Appt Days']

        # Set TARGET: phase at appointment N+1 (future, what we predict)
        # Transition label computed from: Previous_Phase (N) → Next_Healing_Phase (N+1)
        current_appt['Next_Healing_Phase'] = next_appt[self.target_col]

        if pd.notna(current_appt['Next_Healing_Phase']):
            current_appt['Current_Dressing'] = current_appt.get('Dressing', 0)
            current_appt['Current_Dressing_Grouped'] = current_appt.get('Dressing Grouped', 0)

            current_offloading = 0
            for feat in OFFLOADING_FEATURES:
                if feat in current_appt:
                    current_offloading += current_appt.get(feat, 0)

            current_appt['Current_Offloading_Score'] = current_offloading

            return current_appt

        return None
