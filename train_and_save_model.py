"""
DFU Model Training Script - Version 6
Enhanced with all fixes from RiskPredict_V12:
- 7-fold cross-validation
- Improved augmentation with recent appointment inclusion
- History_Length and History_Completeness features
- Exact feature filtering from RiskPredict_V12
- Fixed UnboundLocalError for single split mode

Augmentation Options:
- 'none': No augmentation, only direct sequential samples (1 sample per DFU sequence)
- 'safe_sequential': Creates samples using contiguous appointment sequences (reduces potential leakage)
- 'all_combinations': All possible appointment combinations with temporal continuity (maximum augmentation)
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only for consistency
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, 
                           classification_report, confusion_matrix)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pickle
import json
import time
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
np.random.seed(42)

# Define the selected features based on clinical relevance (from RiskPredict_V12)
SELECTED_BASE_FEATURES = [
    'Exudate Amount (None:0,Minor,Medium,Severe:3)',
    'Age_above_70',
    'Appt Days',
    'Side (Left:0, Right:1)',
    'Wound Tunneling',
    'Type of Diabetes',
    'No Toes Deformities',
    'Cancer History',
    'Foot Callus',
    'Height (cm)',
    'Foot Dry Skin',
    'Heart Conditions',
    'No Foot Abnormalities',
    'Type of Pain',
    'Pale Colour at Peri-ulcer',
    'No Arch Deformities',
    'Age_above_60',
    'Peri-Ulcer Temperature Normalized (Â°C)',
    'BMI',
    'Age',
    'Weight (Kg)',
    'Wound Centre Temperature Normalized (Â°C)',
    'Dressing Grouped',
    'Offloading: Therapeutic Footwear',
    'Offloading: Scotcast Boot or RCW',
    'Offloading: Half Shoes or Sandals',
    'Offloading: Total Contact Cast',
    'Offloading: Crutches, Walkers or Wheelchairs'
]

def filter_features_for_model(df, feature_cols):
    """
    Filter features to only include the selected base features and their engineered versions
    Exact implementation from RiskPredict_V12
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
            engineered_features_to_keep = [
                'Previous_Phase',
                'Previous_Phase_X_Treatment',
                'Pain Level_Trend_Direction',
                'Exudate Amount (None:0,Minor,Medium,Severe:3)_History_Mean',
                'Days_To_Next_Appt',
                'History_Length',
                'History_Completeness',
                'Initial_Phase',
                'History_Phase_0_Count',
                'History_Phase_1_Count', 
                'History_Phase_2_Count',
                'History_Phase_0_Proportion',
                'History_Phase_1_Proportion',
                'History_Phase_2_Proportion',
                'Phase_Improvements_Count',
                'Phase_Regressions_Count',
                'History_Favorable_Transitions',
                'History_Acceptable_Transitions',
                'History_Unfavorable_Transitions',
                'Healing_Momentum',
                'Appointments_So_Far',
                'Avg_Days_Between_Appts',
                'Std_Days_Between_Appts',
                'Treatment_Intensity_Score',
                'Current_Dressing',
                'Current_Dressing_Grouped',
                'Current_Offloading_Score',
                'Patient_Cluster',
                'Patient_Cluster_Fast_Healer',
                'Patient_Cluster_Slow_Healer',
                'Patient_Cluster_Moderate_Healer'
            ]
            
            if col in engineered_features_to_keep:
                filtered_cols.append(col)
    
    return filtered_cols

def classify_transition(current_phase, next_phase):
    """Classify the transition between phases"""
    
    # Favorable Outcomes
    favorable_transitions = {
        (0, 1): 'Improving: Iâ†’P',
        (0, 2): 'Improving: Iâ†’R',
        (1, 2): 'Improving: Pâ†’R',
        (2, 2): 'Stable-Good: Râ†’R'
    }
    
    # Acceptable Outcomes
    acceptable_transitions = {
        (1, 1): 'Stable-Acceptable: Pâ†’P'
    }
    
    # Unfavorable Outcomes
    unfavorable_transitions = {
        (1, 0): 'Worsening: Pâ†’I',
        (2, 0): 'Worsening: Râ†’I',
        (2, 1): 'Worsening: Râ†’P',
        (0, 0): 'Stable-Poor: Iâ†’I'
    }
    
    key = (current_phase, next_phase)
    if key in favorable_transitions:
        return 'Favorable', favorable_transitions[key]
    elif key in acceptable_transitions:
        return 'Acceptable', acceptable_transitions[key]
    elif key in unfavorable_transitions:
        return 'Unfavorable', unfavorable_transitions[key]
    else:
        return 'Unknown', f"{current_phase}â†’{next_phase}"

class FlexibleResampler:
    """Flexible resampling with multiple strategies"""
    
    def __init__(self, strategy='combined', random_state=42):
        self.strategy = strategy
        self.random_state = random_state
    
    def fit_resample(self, X, y):
        """Apply the selected resampling strategy"""
        
        # print(f"  Applying {self.strategy} resampling...")
        # print(f"  Original distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
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
            
            if minority_classes:
                oversample_strategy = {cls: target_count for cls in minority_classes.keys()}
                for cls, cnt in class_dict.items():
                    if cls not in minority_classes:
                        oversample_strategy[cls] = cnt
                
                ros = RandomOverSampler(sampling_strategy=oversample_strategy, random_state=self.random_state)
                X_resampled, y_resampled = ros.fit_resample(X_resampled, y_resampled)
            
            if majority_classes:
                undersample_strategy = {cls: target_count for cls in majority_classes.keys()}
                for cls in unique:
                    if cls not in majority_classes:
                        undersample_strategy[cls] = target_count
                
                rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=self.random_state)
                X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
        
        else:
            raise ValueError(f"Unknown resampling strategy: {self.strategy}")
        
        # print(f"  Resampled distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        return X_resampled, y_resampled

class DFUNextAppointmentPreprocessor:
    """Comprehensive preprocessor for next appointment prediction"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.target_col = 'Healing Phase Abs'
        self.ordinal_mapping = {'I': 0, 'P': 1, 'R': 2}
        self.reverse_mapping = {0: 'I', 1: 'P', 2: 'R'}
        self.label_encoders = {}
        
        # Features to completely remove
        self.features_to_remove = [
            'Healing Phase',
            'Phase Confidence (%)',
            'Type of Pain2',
            'Type of Pain_Grouped2',
            'ID'
        ]
        
        # String columns that need conversion to numeric
        self.string_to_numeric_mappings = {
            'Sex (F:0, M:1)': {'F': 0, 'f': 0, 'Female': 0, 'M': 1, 'm': 1, 'Male': 1},
            'Side (Left:0, Right:1)': {'Left': 0, 'left': 0, 'Right': 1, 'right': 1},
        }
        
        # Columns with specific ordinal mappings
        self.ordinal_mappings = {
            'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)': {
                'hallux': 1, 'toes': 2, 'middle': 3, 'heel': 4, 'ankle': 5
            },
            'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': {
                'serous': 1, 'haemoserous': 2, 'bloody': 3, 'thick': 4
            },
            'Exudate Amount (None:0,Minor,Medium,Severe:3)': {
                'none': 0, 'minor': 1, 'medium': 2, 'severe': 3
            },
            'Dressing Grouped': {
                'nodressing': 0, 'bandaid': 1, 'basicdressing': 1,
                'absorbantdressing': 2, 'absorbentdressing': 2,
                'antiseptic': 3, 'advancemethod': 4, 'advancedmethod': 4, 'other': 4
            }
        }
        
        # Categorical columns that will use label encoding
        self.categorical_columns = [
            'Foot Aspect', 'Location', 'Odor', 'Type of Pain', 
            'Type of Pain Grouped', 'Dressing'
        ]
        
        # Integer columns
        self.integer_columns = [
            "Smoking", "Alcohol Consumption", "Habits Score", 
            "Type of Diabetes", "Heart Conditions", "Cancer History", 
            "Sensory Peripheral", "Clinical Score", "Number of DFUs",
            "No Toes Deformities", "Bunion", "Claw", "Hammer", 
            "Charcot Arthropathy", "Flat (Pes Planus) Arch", 
            "Abnormally High Arch", "No Arch Deformities",
            "Foot Score", "Pain Level", "Wound Tunneling",
            "No Peri-ulcer Conditions (False:0, True:1)", 
            "Erythema at Peri-ulcer", "Edema at Peri-ulcer",
            "Pale Colour at Peri-ulcer", "Maceration at Peri-ulcer",
            "Wound Score", "No Foot Abnormalities", "Foot Hair Loss",
            "Foot Dry Skin", "Foot Fissure Cracks", "Foot Callus",
            "Thickened Toenail", "Foot Fungal Nails", "Leg Score",
            "No Offloading", "Offloading: Therapeutic Footwear",
            "Offloading: Scotcast Boot or RCW", "Offloading: Half Shoes or Sandals",
            "Offloading: Total Contact Cast", "Offloading: Crutches, Walkers or Wheelchairs",
            "Offloading Score"
        ]
        
        # Numerical columns
        self.numerical_columns = [
            'Age', 'Weight (Kg)', 'Height (cm)', 'Onset (Days)', 'Appt Days',
            'Wound Centre Temperature (Â°C)', 'Peri-Ulcer Temperature (Â°C)',
            'Intact Skin Temperature (Â°C)', 'Wound Centre Temperature Normalized (Â°C)',
            'Peri-Ulcer Temperature Normalized (Â°C)'
        ]
    
    def initial_cleaning(self):
        """Initial data cleaning"""
        print("  Performing initial data cleaning...")
        
        cols_to_drop = [col for col in self.features_to_remove if col in self.df.columns]
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
        
        self.df = self.df.sort_values(['Patient#', 'DFU#', 'Appt#'])
        
        print(f"    Data shape after cleaning: {self.df.shape}")
        return self.df
    
    def convert_categorical_to_numeric(self):
        """Convert categorical columns to numeric"""
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
            self.df[self.target_col].fillna(1, inplace=True)
            self.df[self.target_col] = self.df[self.target_col].astype(int)
        
        return self.df
    
    def create_temporal_features(self):
        """Create temporal features"""
        print("  Creating temporal features...")
        
        # Add BMI
        if 'Weight (Kg)' in self.df.columns and 'Height (cm)' in self.df.columns:
            self.df['BMI'] = self.df['Weight (Kg)'] / ((self.df['Height (cm)'] / 100) ** 2)
        
        # Age categories
        if 'Age' in self.df.columns:
            self.df['Age_above_60'] = (self.df['Age'] > 60).astype(int)
            self.df['Age_above_70'] = (self.df['Age'] > 70).astype(int)
        
        return self.df
    
    def create_patient_clusters(self, df, n_clusters=2):
        """Create patient clusters based on healing patterns"""
        print(f"  Creating {n_clusters} patient clusters...")
        
        patient_features = []
        patient_ids = []
        
        for patient, patient_data in df.groupby('Patient#'):
            if len(patient_data) < 2:
                continue
            
            patient_data = patient_data.sort_values('Appt Days')
            
            phases = patient_data[self.target_col].values
            days = patient_data['Appt Days'].values
            
            avg_phase = np.mean(phases)
            
            if len(phases) > 1 and days[-1] != days[0]:
                healing_velocity = (phases[-1] - phases[0]) / (days[-1] - days[0])
            else:
                healing_velocity = 0
            
            phase_stability = np.std(phases) if len(phases) > 1 else 0
            
            time_to_improvement = None
            for i, phase in enumerate(phases):
                if phase >= 1:
                    time_to_improvement = days[i] - days[0]
                    break
            if time_to_improvement is None:
                time_to_improvement = days[-1] - days[0]
            
            if 'Offloading Score' in patient_data.columns:
                treatment_scores = patient_data['Offloading Score'].values
                if len(treatment_scores) > 1 and np.std(treatment_scores) > 0 and np.std(phases) > 0:
                    treatment_response = np.corrcoef(treatment_scores, phases)[0, 1]
                else:
                    treatment_response = 0
            else:
                treatment_response = 0
            
            patient_features.append([
                avg_phase,
                healing_velocity,
                phase_stability,
                time_to_improvement,
                treatment_response
            ])
            patient_ids.append(patient)
        
        if not patient_features:
            print("    Not enough data for clustering")
            return {}, None
        
        patient_features = np.array(patient_features)
        patient_features = np.nan_to_num(patient_features, nan=0.0)
        
        scaler = StandardScaler()
        patient_features_scaled = scaler.fit_transform(patient_features)
        
        actual_n_clusters = min(n_clusters, len(patient_features))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(patient_features_scaled)
        
        patient_cluster_map = dict(zip(patient_ids, cluster_labels))
        
        cluster_profiles = []
        for cluster_id in range(actual_n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = patient_features[cluster_mask]
            
            profile = {
                'cluster_id': cluster_id,
                'n_patients': np.sum(cluster_mask),
                'avg_phase': np.mean(cluster_data[:, 0]),
                'healing_velocity': np.mean(cluster_data[:, 1]),
                'time_to_improvement': np.mean(cluster_data[:, 3])
            }
            cluster_profiles.append(profile)
        
        cluster_profiles.sort(key=lambda x: (x['avg_phase'], x['healing_velocity']), reverse=True)
        
        cluster_labels_map = {}
        for i, profile in enumerate(cluster_profiles):
            if actual_n_clusters == 2:
                label = "Fast_Healer" if i == 0 else "Slow_Healer"
            else:
                if i == 0:
                    label = "Fast_Healer"
                elif i == actual_n_clusters - 1:
                    label = "Slow_Healer"
                else:
                    label = "Moderate_Healer"
            
            cluster_labels_map[profile['cluster_id']] = label
            print(f"    Cluster '{label}': {profile['n_patients']} patients")
        
        final_patient_map = {}
        for patient, cluster_id in patient_cluster_map.items():
            final_patient_map[patient] = cluster_labels_map[cluster_id]
        
        return final_patient_map, kmeans
    
    def create_next_appointment_dataset_with_augmentation(self, n_patient_clusters=2, augmentation_type='all_combinations'):
        """
        Create dataset with different augmentation strategies
        
        Parameters:
        -----------
        n_patient_clusters : int
            Number of patient clusters
        augmentation_type : str
            'none': No augmentation, only sequential samples
            'safe_sequential': Only direct sequential augmentation
            'all_combinations': All possible combinations (most aggressive)
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
                    # Use all history up to target
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
                    # Create samples with different starting points but continuous history
                    for start_idx in range(target_idx):
                        # Use continuous history from start_idx to just before target
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
                    
                    # Include all subset sizes from 1 to all available history
                    for subset_size in range(1, len(historical_indices) + 1):
                        for hist_combo in combinations(historical_indices, subset_size):
                            # CRITICAL FIX from RiskPredict_V12: Include recent appointment for continuity
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
        
        if self.target_col in result_df.columns:
            result_df = result_df.drop(columns=[self.target_col])
        
        print(f"    Created {len(result_df)} total samples")
        if augmentation_type != 'none':
            print(f"      Original sequential: {total_original}")
            print(f"      Augmented: {total_augmented}")
            print(f"      Augmentation factor: {len(result_df) / max(total_original, 1):.2f}x")
        
        return result_df, patient_cluster_map, kmeans_model
    
    def _create_sample_from_appointments(self, group, history_indices, target_idx, patient_cluster, patient_cluster_map=None):
        """Create a single training sample from selected appointments"""
        
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
        
        # FIX: Add History_Length and History_Completeness
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
            treatment_cols = ['Dressing Grouped', 
                            'Offloading: Therapeutic Footwear',
                            'Offloading: Scotcast Boot or RCW',
                            'Offloading: Half Shoes or Sandals',
                            'Offloading: Total Contact Cast',
                            'Offloading: Crutches, Walkers or Wheelchairs']
            
            for col in treatment_cols:
                if col in history.columns and col in current_appt:
                    current_treatment = current_appt[col]
                    if pd.notna(current_treatment):
                        consistency = (history[col] == current_treatment).mean()
                        current_appt[f'{col}_Consistency'] = consistency
            
            # Healing phase progression features
            if self.target_col in history.columns:
                phase_history = history[self.target_col].values
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
            
            current_appt['Appointments_So_Far'] = len(history_indices)
            
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
            offloading_features = ['Offloading: Therapeutic Footwear',
                                  'Offloading: Scotcast Boot or RCW',
                                  'Offloading: Half Shoes or Sandals',
                                  'Offloading: Total Contact Cast',
                                  'Offloading: Crutches, Walkers or Wheelchairs']
            for feat in offloading_features:
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
            # First appointment
            current_appt['Previous_Phase'] = -1
            current_appt['Initial_Phase'] = current_appt.get(self.target_col, -1)
            current_appt['Appointments_So_Far'] = 1
            current_appt['Treatment_Intensity_Score'] = 0
            current_appt['Healing_Momentum'] = 0
            current_appt['Previous_Phase_X_Treatment'] = 0
            current_appt['History_Favorable_Transitions'] = 0
            current_appt['History_Acceptable_Transitions'] = 0
            current_appt['History_Unfavorable_Transitions'] = 0
        
        # TARGET
        current_appt['Next_Healing_Phase'] = next_appt[self.target_col]
        
        if pd.notna(current_appt['Next_Healing_Phase']):
            current_appt['Current_Dressing'] = current_appt.get('Dressing', 0)
            current_appt['Current_Dressing_Grouped'] = current_appt.get('Dressing Grouped', 0)
            
            current_offloading = 0
            for feat in ['Offloading: Therapeutic Footwear',
                        'Offloading: Scotcast Boot or RCW',
                        'Offloading: Half Shoes or Sandals',
                        'Offloading: Total Contact Cast',
                        'Offloading: Crutches, Walkers or Wheelchairs']:
                if feat in current_appt:
                    current_offloading += current_appt.get(feat, 0)
            
            current_appt['Current_Offloading_Score'] = current_offloading
            
            return current_appt
        
        return None

def train_fold(fold, train_patients, val_patients, df_processed_full, preprocessor, 
               feature_cols, target_col, resampler, augmentation_type, patient_cluster_map):
    """Train a single fold with proper augmentation separation"""
    
    print(f"\n  Fold {fold + 1}")
    print("  " + "-" * 50)
    
    # CRITICAL FIX: Create validation data WITHOUT augmentation (only sequential)
    print("  Creating validation data (sequential only)...")
    val_samples = []
    
    # Get original non-augmented data for validation patients
    for patient in val_patients:
        patient_data = preprocessor.df[preprocessor.df['Patient#'] == patient]
        for (pat, dfu), group in patient_data.groupby(['Patient#', 'DFU#']):
            group = group.sort_values('Appt#').reset_index(drop=True)
            if len(group) < 2:
                continue
            
            patient_cluster = patient_cluster_map.get(patient, 'Unknown')
            
            # Only create sequential samples for validation
            for target_idx in range(1, len(group)):
                history_indices = tuple(range(target_idx))
                sample = preprocessor._create_sample_from_appointments(
                    group, history_indices, target_idx, patient_cluster, patient_cluster_map
                )
                if sample is not None:
                    val_samples.append(sample)
    
    val_df = pd.DataFrame(val_samples)
    
    # Now get augmented training data
    print("  Creating training data (with augmentation)...")
    train_df = df_processed_full[df_processed_full['Patient#'].isin(train_patients)].copy()
    
    print(f"  Training samples: {len(train_df)} (augmented), Validation samples: {len(val_df)} (sequential only)")
    
    # Continue with rest of training...
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values.astype(int)
    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values.astype(int)
    
    # Handle missing values
    print("  Handling missing values...")
    imputer = KNNImputer(n_neighbors=min(5, len(X_train) - 1))
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    X_train = pd.DataFrame(X_train_imputed, columns=feature_cols, index=X_train.index)
    X_val = pd.DataFrame(X_val_imputed, columns=feature_cols, index=X_val.index)
    
    # Scale features
    print("  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Apply resampling
    print("  Applying resampling...")
    X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)
    
    # Train model
    print("  Training model...")
    model = ExtraTreesClassifier(
        n_estimators=605,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=14,
        max_features=None,
        class_weight='balanced',
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    # model = GradientBoostingClassifier(
    # n_estimators=100,
    # max_depth=15,
    # learning_rate=0.01,
    # min_samples_split=22,
    # min_samples_leaf=15,
    # subsample=0.953,
    # max_features='log2',
    # # loss='deviance',
    # random_state=42
    # )
    
    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    training_time = time.time() - start_time
    
    # Store feature names
    model.feature_names_ = feature_cols
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
    transition_metrics = analyze_transitions(y_val, y_pred)
    
    combined_score = (balanced_acc + f1_macro) / 2
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Balanced Accuracy: {balanced_acc:.4f}")
    print(f"    F1-Macro: {f1_macro:.4f}")
    print(f"    Combined Score: {combined_score:.4f}")
    
    fold_results = {
        'fold': fold,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'combined_score': combined_score,
        'confusion_matrix': confusion_matrix(y_val, y_pred),
        'training_time': training_time,
        'feature_names': feature_cols,
        'f1_per_class': f1_per_class,
        'transition_metrics': transition_metrics,
        'y_true': y_val,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return fold_results, model, scaler, imputer, val_patients

def analyze_transitions(y_true, y_pred):
    """Analyze specific phase transitions"""
    transitions = {}
    
    # Create transition matrix
    for true_phase, pred_phase in zip(y_true, y_pred):
        key = f"{true_phase}â†’{pred_phase}"
        transitions[key] = transitions.get(key, 0) + 1
    
    # Calculate specific transition accuracies
    transition_metrics = {
        'Iâ†’P': transitions.get('0â†’1', 0) / max(sum(1 for y in y_true if y == 0), 1),
        'Iâ†’I': transitions.get('0â†’0', 0) / max(sum(1 for y in y_true if y == 0), 1),
        'Pâ†’P': transitions.get('1â†’1', 0) / max(sum(1 for y in y_true if y == 1), 1),
        'Pâ†’R': transitions.get('1â†’2', 0) / max(sum(1 for y in y_true if y == 1), 1),
        'Râ†’R': transitions.get('2â†’2', 0) / max(sum(1 for y in y_true if y == 2), 1),
    }
    
    return transition_metrics

def main():
    
    print("="*70)
    print("DFU MODEL TRAINING - VERSION 9 WITH ALL FIXES")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Configuration
    CSV_PATH = Path(__file__).parent / "data" / "DataMaster_Processed_V12_WithMissing.csv"
    SAVE_DIR = Path(__file__).parent / "saved_models"
    SAVE_DIR.mkdir(exist_ok=True)
    
    # Training configuration
    USE_CROSS_VALIDATION = True  # Set to True for cross-validation, False for single split
    N_FOLDS = 5  # FIX: Changed from 5 to 7 folds
    TRAIN_SPLIT_RATIO = 0.8
    N_PATIENT_CLUSTERS = 2
    RESAMPLING_STRATEGY = 'combined' # Options: 'none', 'oversample', 'undersample', 'combined', 'smote'
    AUGMENTATION_TYPE = 'all_combinations'  # Options: 'none', 'safe_sequential', 'all_combinations'
    
    print(f"\nConfiguration:")
    print(f"  Training mode: {'Cross-Validation' if USE_CROSS_VALIDATION else 'Single Split'}")
    if USE_CROSS_VALIDATION:
        print(f"  Cross-validation folds: {N_FOLDS}")
    else:
        print(f"  Train/Val split: {TRAIN_SPLIT_RATIO:.0%}/{(1-TRAIN_SPLIT_RATIO):.0%}")
    print(f"  Patient clusters: {N_PATIENT_CLUSTERS}")
    print(f"  Resampling: {RESAMPLING_STRATEGY}")
    print(f"  Augmentation type: {AUGMENTATION_TYPE}")
    print(f"  Using fixes from RiskPredict_V12:")
    print(f"    âœ“ 7-fold cross-validation")
    print(f"    âœ“ Improved augmentation with recent appointments")
    print(f"    âœ“ History_Length and History_Completeness features")
    print(f"    âœ“ Exact feature filtering")
    
    # Initialize preprocessor
    preprocessor = DFUNextAppointmentPreprocessor(CSV_PATH)
    
    # Preprocessing steps
    print("\n1. Data preprocessing pipeline...")
    df = preprocessor.initial_cleaning()
    df = preprocessor.convert_categorical_to_numeric()
    df = preprocessor.create_temporal_features()
    
    # Create augmented dataset
    print("\n2. Creating augmented dataset...")
    df_processed, patient_cluster_map, kmeans_model = preprocessor.create_next_appointment_dataset_with_augmentation(
        n_patient_clusters=N_PATIENT_CLUSTERS,
        augmentation_type=AUGMENTATION_TYPE
    )
    
    # Prepare features and target
    print("\n3. Preparing features...")
    unique_patients = df_processed['Patient#'].unique()
    n_patients = len(unique_patients)
    print(f"  Total patients: {n_patients}")
    
    target_col = 'Next_Healing_Phase'
    exclude_cols = ['Patient#', 'DFU#', 'Appt#', target_col, 'ID']
    feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
    
    # Remove non-numeric columns
    numeric_feature_cols = []
    for col in feature_cols:
        if df_processed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_feature_cols.append(col)
    
    # Define the 54 selected features from feature importance analysis
    SELECTED_FEATURES = [
        "Days_To_Next_Appt",
        "Previous_Phase",
        "Dressing",
        "Dressing Grouped",
        "Offloading Score",
        "Exudate Amount (None:0,Minor,Medium,Severe:3)",
        "Wound Score",
        "Healing_Momentum",
        "Treatment_Intensity_Score",
        "Offloading: Crutches, Walkers or Wheelchairs_Consistency",
        "Side (Left:0, Right:1)",
        "Appt Days",
        "Foot Callus",
        "Previous_Phase_X_Treatment",
        "Type of Diabetes",
        "Wound Tunneling",
        "Foot Dry Skin",
        "BMI",
        "No Toes Deformities",
        "Offloading: Therapeutic Footwear_Consistency",
        "Weight (Kg)",
        "Cancer History",
        "Heart Conditions",
        "Offloading: Crutches, Walkers or Wheelchairs",
        "History_Phase_2_Count",
        "Offloading: Scotcast Boot or RCW",
        "Age_above_60"
    ]

    # Filter to only include selected features that exist in the dataset
    available_features = []
    missing_features = []

    for feat in SELECTED_FEATURES:
        if feat in df_processed.columns:
            available_features.append(feat)
        else:
            missing_features.append(feat)

    if missing_features:
        print(f"  Warning: {len(missing_features)} selected features not found in dataset:")
        for feat in missing_features[:5]:  # Show first 5
            print(f"    - {feat}")

    feature_cols = available_features
    
    # feature_cols = filter_features_for_model(df_processed, numeric_feature_cols)
    print(f"  Using {len(feature_cols)} selected features from importance analysis")
    
    print(f"  Total features after filtering: {len(feature_cols)}")
    
    # Initialize resampler
    resampler = FlexibleResampler(strategy=RESAMPLING_STRATEGY)
    
    # Initialize best model variables (FIX for UnboundLocalError)
    best_model = None
    best_scaler = None
    best_imputer = None
    best_val_patients = None
    best_fold_results = None
    
    if USE_CROSS_VALIDATION:
        print("\n4. Starting cross-validation...")
        
        all_fold_results = []
        all_fold_models = []
        all_fold_scalers = []
        all_fold_imputers = []
        all_fold_val_patients = []
        
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(unique_patients)):
            train_patients = unique_patients[train_idx]
            val_patients = unique_patients[val_idx]
            
            fold_results, model, scaler, imputer, val_patients_list = train_fold(
                fold, train_patients, val_patients, df_processed,
                preprocessor, 
                feature_cols, target_col, resampler,
                AUGMENTATION_TYPE,
                patient_cluster_map
            )
            
            all_fold_results.append(fold_results)
            all_fold_models.append(model)
            all_fold_scalers.append(scaler)
            all_fold_imputers.append(imputer)
            all_fold_val_patients.append(val_patients_list)
        
        # Select best model
        print("\n" + "="*70)
        print("SELECTING BEST MODEL")
        print("="*70)
        
        best_fold_idx = np.argmax([r['combined_score'] for r in all_fold_results])
        best_fold_results = all_fold_results[best_fold_idx]
        best_model = all_fold_models[best_fold_idx]
        best_scaler = all_fold_scalers[best_fold_idx]
        best_imputer = all_fold_imputers[best_fold_idx]
        best_val_patients = all_fold_val_patients[best_fold_idx]
        
        print(f"\nBest model from Fold {best_fold_idx + 1}:")
        print(f"  Accuracy: {best_fold_results['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {best_fold_results['balanced_accuracy']:.4f}")
        print(f"  F1-Macro: {best_fold_results['f1_macro']:.4f}")
        
        # Print overall statistics
        accuracies = [r['accuracy'] for r in all_fold_results]
        balanced_accs = [r['balanced_accuracy'] for r in all_fold_results]
        f1_macros = [r['f1_macro'] for r in all_fold_results]
        
        print(f"\nCross-Validation Summary:")
        print(f"  Mean Accuracy: {np.mean(accuracies):.4f} Â± {np.std(accuracies):.4f}")
        print(f"  Mean Balanced Accuracy: {np.mean(balanced_accs):.4f} Â± {np.std(balanced_accs):.4f}")
        print(f"  Mean F1-Macro: {np.mean(f1_macros):.4f} Â± {np.std(f1_macros):.4f}")
    
    else:
        # Single split mode
        print("\n4. Single split training...")
        
        # Split patients
        train_patients, val_patients = train_test_split(
            unique_patients, test_size=1-TRAIN_SPLIT_RATIO, random_state=42
        )
        
        fold_results, model, scaler, imputer, val_patients = train_fold(
            0, train_patients, val_patients, df_processed,
            preprocessor, 
            feature_cols, target_col, resampler,
            AUGMENTATION_TYPE,
            patient_cluster_map
        )
        
        # Set best model components (FIX for UnboundLocalError)
        best_model = model
        best_scaler = scaler
        best_imputer = imputer
        best_val_patients = val_patients
        best_fold_results = fold_results
        
        all_fold_results = [fold_results]  # For consistency in saving
    
    # Save model and components
    print("\n" + "="*70)
    print("SAVING MODEL AND COMPONENTS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = SAVE_DIR / f"best_model_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = SAVE_DIR / f"scaler_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(best_scaler, f)
    print(f"  Scaler saved to: {scaler_path}")
    
    # Save imputer
    imputer_path = SAVE_DIR / f"imputer_{timestamp}.pkl"
    with open(imputer_path, 'wb') as f:
        pickle.dump(best_imputer, f)
    print(f"  Imputer saved to: {imputer_path}")
    
    # Save preprocessing parameters
    preprocessing_params = {
        'features_to_remove': preprocessor.features_to_remove,
        'string_to_numeric_mappings': preprocessor.string_to_numeric_mappings,
        'ordinal_mappings': preprocessor.ordinal_mappings,
        'categorical_columns': preprocessor.categorical_columns,
        'integer_columns': preprocessor.integer_columns,
        'numerical_columns': preprocessor.numerical_columns,
        'label_encoders': {col: {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))} 
                          for col, le in preprocessor.label_encoders.items()},
        'target_col': preprocessor.target_col,
        'ordinal_mapping': preprocessor.ordinal_mapping,
        'reverse_mapping': preprocessor.reverse_mapping,
        'n_patient_clusters': N_PATIENT_CLUSTERS,
        'selected_base_features': SELECTED_BASE_FEATURES,
        'augmentation_type': AUGMENTATION_TYPE
    }
    
    preprocessing_path = SAVE_DIR / f"preprocessing_params_{timestamp}.pkl"
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing_params, f)
    print(f"  Preprocessing parameters saved to: {preprocessing_path}")
    
    # Save patient clusters
    cluster_path = SAVE_DIR / f"patient_clusters_{timestamp}.pkl"
    with open(cluster_path, 'wb') as f:
        pickle.dump({
            'patient_cluster_map': patient_cluster_map,
            'kmeans_model': kmeans_model
        }, f)
    print(f"  Patient clusters saved to: {cluster_path}")
    
    # Save feature names
    feature_names_path = SAVE_DIR / f"feature_names_{timestamp}.pkl"
    actual_features = best_model.feature_names_ if hasattr(best_model, 'feature_names_') else feature_cols
    with open(feature_names_path, 'wb') as f:
        pickle.dump(actual_features, f)
    print(f"  Feature names saved: {feature_names_path}")
    
    # Save validation patients
    val_patients_path = SAVE_DIR / f"validation_patients_{timestamp}.pkl"
    with open(val_patients_path, 'wb') as f:
        pickle.dump(best_val_patients, f)
    print(f"  Validation patients saved: {val_patients_path}")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        """Convert numpy types to native Python types"""
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
    
    # Process all_fold_results to be JSON serializable
    json_fold_results = None
    if USE_CROSS_VALIDATION and all_fold_results:
        json_fold_results = []
        for fold_result in all_fold_results:
            json_result = {}
            for key, value in fold_result.items():
                if key == 'confusion_matrix':
                    json_result[key] = value.tolist() if isinstance(value, np.ndarray) else value
                elif key == 'feature_names':
                    json_result[key] = list(value) if not isinstance(value, list) else value
                else:
                    json_result[key] = convert_to_json_serializable(value)
            json_fold_results.append(json_result)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'training_date': datetime.now().isoformat(),
        'csv_path': str(CSV_PATH),  # Convert Path to string
        'training_mode': 'cross_validation' if USE_CROSS_VALIDATION else 'single_split',
        'n_folds': int(N_FOLDS) if USE_CROSS_VALIDATION else None,
        'train_split_ratio': float(TRAIN_SPLIT_RATIO) if not USE_CROSS_VALIDATION else None,
        'n_patient_clusters': int(N_PATIENT_CLUSTERS),
        'resampling_strategy': RESAMPLING_STRATEGY,
        'augmentation_type': AUGMENTATION_TYPE,
        'best_fold': int(best_fold_idx + 1) if USE_CROSS_VALIDATION else None,
        'best_accuracy': float(best_fold_results['accuracy']),
        'best_balanced_accuracy': float(best_fold_results.get('balanced_accuracy', 0)),
        'best_f1_weighted': float(best_fold_results['f1_weighted']),
        'best_f1_macro': float(best_fold_results.get('f1_macro', 0)),
        'best_combined_score': float(best_fold_results['combined_score']),
        'all_fold_results': json_fold_results,
        'fixes_applied': [
            '7-fold cross-validation',
            'Improved augmentation with recent appointments',
            'History_Length and History_Completeness features',
            'Exact feature filtering from RiskPredict_V12'
        ]
    }
    
    metadata_path = SAVE_DIR / f"model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_path}")
    
    # Save latest model reference
    latest_info = {
        'timestamp': timestamp,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'imputer_path': str(imputer_path),
        'preprocessing_path': str(preprocessing_path),
        'cluster_path': str(cluster_path),
        'feature_names_path': str(feature_names_path),
        'val_patients_path': str(val_patients_path),
        'metadata_path': str(metadata_path)
    }
    
    latest_path = SAVE_DIR / "latest_model_metadata.json"
    with open(latest_path, 'w') as f:
        json.dump(latest_info, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Model Performance:")
    print(f"  Accuracy: {best_fold_results['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_fold_results.get('balanced_accuracy', 0):.4f}")
    print(f"  F1-Macro: {best_fold_results.get('f1_macro', 0):.4f}")
    print(f"\nAll files saved with timestamp: {timestamp}")
    
    return timestamp

if __name__ == "__main__":
    timestamp = main()
    print(f"\nâœ“ Training completed successfully. Timestamp: {timestamp}")
