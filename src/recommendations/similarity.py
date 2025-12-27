"""
Hierarchical similarity matching for treatment recommendations.

This module provides similarity-based matching for finding similar successful
cases to guide treatment recommendations.
"""

import numpy as np
import pandas as pd


class HierarchicalSimilarityMatcher:
    """
    Hierarchical similarity matcher for treatment recommendations.
    """

    def __init__(self, successful_cases):
        """
        Initialize the similarity matcher.

        Parameters
        ----------
        successful_cases : pd.DataFrame
            Database of successful treatment cases
        """
        self.successful_cases = successful_cases

        # Hierarchical feature tiers for offloading
        self.offloading_tiers = {
            'critical': {
                'features': ['Current_Phase', 'Charcot Arthropathy', 'Deformity_Severity', 'Location_Risk'],
                'weight': 3.0
            },
            'important': {
                'features': ['Wound Score', 'Age', 'Weight (Kg)', 'Onset_Category', 'Location_Numeric'],
                'weight': 2.0
            },
            'refining': {
                'features': ['Clinical Score', 'Mobility_Risk', 'Foot Aspect', 'Side_Numeric'],
                'weight': 1.0
            }
        }

        # Hierarchical feature tiers for dressing
        self.dressing_tiers = {
            'critical': {
                'features': ['Current_Phase', 'Onset_Category', 'Location_Heel', 'Previous_Failure'],
                'weight': 3.0
            },
            'important': {
                'features': ['Wound Score', 'Exudate Amount (None:0,Minor,Medium,Severe:3)',
                           'Moisture_Management_Need', 'Infection_Risk'],
                'weight': 2.0
            },
            'refining': {
                'features': ['Clinical Score', 'Maceration at Peri-ulcer', 'Wound Tunneling'],
                'weight': 1.0
            }
        }

    def hierarchical_similarity_search(self, patient_data, treatment_type='offloading', n_neighbors=15):
        """
        Find similar cases using hierarchical tier-based matching.

        Parameters
        ----------
        patient_data : pd.DataFrame
            Patient data (single row)
        treatment_type : str, default='offloading'
            Type of treatment ('offloading' or 'dressing')
        n_neighbors : int, default=15
            Number of similar cases to find

        Returns
        -------
        similar_cases : pd.DataFrame
            Similar cases
        similarity_scores : np.ndarray
            Similarity scores
        """
        # Select appropriate tiers
        tiers = self.offloading_tiers if treatment_type == 'offloading' else self.dressing_tiers

        # Start with all successful cases
        candidates = self.successful_cases.copy()

        # Calculate similarity scores
        similarity_scores = []

        for idx, candidate in candidates.iterrows():
            score = 0
            max_score = 0

            for tier_name, tier_info in tiers.items():
                weight = tier_info['weight']

                for feature in tier_info['features']:
                    if feature in patient_data.columns and feature in candidates.columns:
                        max_score += weight

                        patient_val = patient_data[feature].iloc[0] if isinstance(patient_data[feature], pd.Series) else patient_data[feature].iloc[0]
                        candidate_val = candidate[feature]

                        # Calculate similarity based on feature type
                        if pd.notna(patient_val) and pd.notna(candidate_val):
                            try:
                                # Categorical features - exact match
                                if feature in ['Current_Phase', 'Onset_Category', 'Location_Heel', 'Previous_Failure']:
                                    if patient_val == candidate_val:
                                        score += weight
                                # Numeric features - distance-based
                                else:
                                    patient_num = float(patient_val)
                                    candidate_num = float(candidate_val)

                                    # Get range for normalization
                                    feature_vals = candidates[feature].dropna()
                                    if len(feature_vals) > 0:
                                        feat_max = feature_vals.max()
                                        feat_min = feature_vals.min()
                                        feat_range = feat_max - feat_min

                                        if feat_range > 0:
                                            diff = abs(patient_num - candidate_num)
                                            similarity = 1 - (diff / feat_range)
                                            score += weight * max(0, similarity)
                                        elif patient_num == candidate_num:
                                            score += weight
                            except:
                                continue

            final_score = score / max_score if max_score > 0 else 0
            similarity_scores.append((idx, final_score))

        # Sort by score and get top N
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [s[0] for s in similarity_scores[:n_neighbors]]
        top_scores = [s[1] for s in similarity_scores[:n_neighbors]]

        similar_cases = candidates.loc[top_indices]

        return similar_cases, np.array(top_scores)

    def recommend_offloading(self, patient_data):
        """
        Recommend offloading using similarity matching.

        Parameters
        ----------
        patient_data : pd.DataFrame
            Patient data (single row)

        Returns
        -------
        recommendation : dict
            Recommendation dictionary
        """
        similar_cases, similarities = self.hierarchical_similarity_search(patient_data, 'offloading')

        # Vote on offloading score
        offloading_votes = {}
        for idx, (case_idx, case) in enumerate(similar_cases.iterrows()):
            score = case.get('Offloading Score', 1.0)
            weight = similarities[idx]

            if score not in offloading_votes:
                offloading_votes[score] = 0
            offloading_votes[score] += weight

        # Get highest voted score
        if offloading_votes:
            best_score = max(offloading_votes.keys(), key=offloading_votes.get)
            confidence = offloading_votes[best_score] / similarities.sum() if similarities.sum() > 0 else 0.5
        else:
            best_score = 1.0
            confidence = 0.8

        # Map score to name
        offloading_names = {
            0: 'No Offloading',
            1: 'Therapeutic Footwear',
            1.0: 'Therapeutic Footwear',
            2: 'Boot/RCW',
            2.0: 'Boot/RCW',
            3: 'Total Contact Cast',
            3.0: 'Total Contact Cast'
        }

        return {
            'recommendation': offloading_names.get(best_score, 'Therapeutic Footwear'),
            'score': best_score,
            'confidence': confidence,
            'rationale': 'Based on similar successful cases',
            'method': 'similarity-based'
        }

    def recommend_dressing(self, patient_data, onset_days):
        """
        Recommend dressing using similarity matching.

        Parameters
        ----------
        patient_data : pd.DataFrame
            Patient data (single row)
        onset_days : float
            Days since wound onset

        Returns
        -------
        recommendation : dict
            Recommendation dictionary
        """
        similar_cases, similarities = self.hierarchical_similarity_search(patient_data, 'dressing')

        # Vote on dressing
        dressing_votes = {}
        for idx, (case_idx, case) in enumerate(similar_cases.iterrows()):
            dressing = case.get('Dressing', 'Betadine')
            weight = similarities[idx]

            if dressing not in dressing_votes:
                dressing_votes[dressing] = 0
            dressing_votes[dressing] += weight

        # Get highest voted dressing
        if dressing_votes:
            best_dressing = max(dressing_votes.keys(), key=dressing_votes.get)
            confidence = dressing_votes[best_dressing] / similarities.sum() if similarities.sum() > 0 else 0.5
        else:
            best_dressing = 'Betadine'
            confidence = 0.8

        # Default to Betadine for standard cases
        if confidence < 0.6 and onset_days < 180:
            return {
                'recommendation': 'Betadine',
                'confidence': 0.90,
                'rationale': 'Standard protocol for acute/subacute wounds',
                'method': 'default-protocol'
            }

        return {
            'recommendation': best_dressing,
            'confidence': confidence,
            'rationale': 'Based on similar successful cases',
            'method': 'similarity-based'
        }
