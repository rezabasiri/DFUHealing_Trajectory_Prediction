"""
Feature engineering module for DFU healing trajectory prediction.

This module contains functions for creating temporal features and patient clusters.
All preprocessing logic is preserved exactly as in the original implementation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from ..config.constants import TARGET_COLUMN


def create_temporal_features(df):
    """
    Create temporal features from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    df : pd.DataFrame
        Dataframe with added temporal features
    """
    # Add BMI
    if 'Weight (Kg)' in df.columns and 'Height (cm)' in df.columns:
        df['BMI'] = df['Weight (Kg)'] / ((df['Height (cm)'] / 100) ** 2)

    # Age categories
    if 'Age' in df.columns:
        df['Age_above_60'] = (df['Age'] > 60).astype(int)
        df['Age_above_70'] = (df['Age'] > 70).astype(int)

    return df


def create_patient_clusters(df, target_col=TARGET_COLUMN, n_clusters=2):
    """
    Create patient clusters based on healing patterns.

    This function groups patients by their healing velocity, phase stability,
    time to improvement, and treatment response.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with patient data
    target_col : str
        Name of the target column (healing phase)
    n_clusters : int, default=2
        Number of clusters to create

    Returns
    -------
    patient_cluster_map : dict
        Mapping from patient ID to cluster label
    kmeans_model : KMeans
        Fitted KMeans clustering model
    """
    print(f"  Creating {n_clusters} patient clusters...")

    patient_features = []
    patient_ids = []

    for patient, patient_data in df.groupby('Patient#'):
        if len(patient_data) < 2:
            continue

        patient_data = patient_data.sort_values('Appt Days')

        phases = patient_data[target_col].values
        days = patient_data['Appt Days'].values

        # Average phase
        avg_phase = np.mean(phases)

        # Healing velocity
        if len(phases) > 1 and days[-1] != days[0]:
            healing_velocity = (phases[-1] - phases[0]) / (days[-1] - days[0])
        else:
            healing_velocity = 0

        # Phase stability
        phase_stability = np.std(phases) if len(phases) > 1 else 0

        # Time to improvement
        time_to_improvement = None
        for i, phase in enumerate(phases):
            if phase >= 1:
                time_to_improvement = days[i] - days[0]
                break
        if time_to_improvement is None:
            time_to_improvement = days[-1] - days[0]

        # Treatment response
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

    # Scale features
    scaler = StandardScaler()
    patient_features_scaled = scaler.fit_transform(patient_features)

    # Perform clustering
    actual_n_clusters = min(n_clusters, len(patient_features))
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(patient_features_scaled)

    patient_cluster_map = dict(zip(patient_ids, cluster_labels))

    # Analyze cluster profiles
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

    # Sort clusters by healing performance
    cluster_profiles.sort(key=lambda x: (x['avg_phase'], x['healing_velocity']), reverse=True)

    # Assign cluster labels
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

    # Map patient IDs to cluster labels
    final_patient_map = {}
    for patient, cluster_id in patient_cluster_map.items():
        final_patient_map[patient] = cluster_labels_map[cluster_id]

    return final_patient_map, kmeans
