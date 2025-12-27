"""
DFU Treatment Recommendation System.

This module provides treatment recommendations based on hierarchical similarity
matching and clinical decision rules. All logic is preserved from the original
implementation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

from .similarity import HierarchicalSimilarityMatcher


class DFUTreatmentRecommendationSystem:
    """
    Complete treatment recommendation system for diabetic foot ulcers.
    Based on hierarchical similarity matching and clinical decision rules.
    """

    def __init__(self, acute_threshold=90, subacute_threshold=180,
                 chronic_threshold=365, standard_offloading=1.0,
                 standard_dressing='Betadine'):
        """
        Initialize the treatment recommendation system.

        Parameters
        ----------
        acute_threshold : int, default=90
            Threshold for acute wounds (days)
        subacute_threshold : int, default=180
            Threshold for subacute wounds (days)
        chronic_threshold : int, default=365
            Threshold for chronic wounds (days)
        standard_offloading : float, default=1.0
            Standard offloading score
        standard_dressing : str, default='Betadine'
            Standard dressing type
        """
        self.standard_offloading = standard_offloading
        self.standard_dressing = standard_dressing

        # Chronicity thresholds (in days)
        self.acute_threshold = acute_threshold
        self.subacute_threshold = subacute_threshold
        self.chronic_threshold = chronic_threshold

        # Initialize components
        self.successful_cases = None
        self.scaler = MinMaxScaler()
        self.similarity_matcher = None

    def prepare_data(self, df):
        """
        Prepare data with all necessary transformations and features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        df : pd.DataFrame
            Prepared dataframe with features
        """
        # Convert categorical columns to numeric
        if 'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)' in df.columns:
            location_map = {
                'Hallux': 1, '1': 1, 1: 1,
                'Toes': 2, '2': 2, 2: 2,
                'Middle': 3, '3': 3, 3: 3,
                'Heel': 4, '4': 4, 4: 4,
                'Ankle': 5, '5': 5, 5: 5
            }
            df['Location_Numeric'] = df['Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)'].map(
                lambda x: location_map.get(x, 3)
            )
        else:
            df['Location_Numeric'] = 3

        # Side conversion
        if 'Side (Left:0, Right:1)' in df.columns:
            side_map = {
                'Left': 0, 'left': 0, 'L': 0, '0': 0, 0: 0,
                'Right': 1, 'right': 1, 'R': 1, '1': 1, 1: 1
            }
            df['Side_Numeric'] = df['Side (Left:0, Right:1)'].map(
                lambda x: side_map.get(x, 0.5)
            )
        else:
            df['Side_Numeric'] = 0.5

        # Create chronicity categories
        if 'Onset (Days)' in df.columns:
            df['Onset_Category'] = pd.cut(
                df['Onset (Days)'].fillna(180),
                bins=[0, self.acute_threshold, self.subacute_threshold,
                      self.chronic_threshold, 99999],
                labels=['Acute', 'Subacute', 'Chronic', 'Very_Chronic'],
                include_lowest=True
            )
            df['Onset_Category_Numeric'] = df['Onset_Category'].cat.codes
        else:
            df['Onset_Category'] = 'Subacute'
            df['Onset_Category_Numeric'] = 1

        # Location-specific risk (heel/ankle are high risk)
        df['Location_Risk'] = 0
        if 'Location' in df.columns:
            heel_ankle_terms = ['heel', 'ankle', 'achilles', 'calcaneus']
            df['Location_Risk'] = df['Location'].str.lower().apply(
                lambda x: 1 if any(term in str(x) for term in heel_ankle_terms) else 0
            )

        df['Location_Heel'] = df['Location_Risk']

        # Composite features
        df = self._create_composite_features(df)

        # Treatment history indicators
        df['Previous_Failure'] = 0

        # Phase encoding
        phase_map = {'I': 0, 'P': 1, 'R': 2}
        if 'Healing Phase' in df.columns:
            df['Current_Phase'] = df['Healing Phase'].str.extract(r'([IPR])', expand=False)
            df['Current_Phase_Numeric'] = df['Current_Phase'].map(phase_map).fillna(1)
        elif 'Healing Phase Abs' in df.columns:
            df['Current_Phase'] = df['Healing Phase Abs']
            df['Current_Phase_Numeric'] = df['Current_Phase'].map(phase_map).fillna(1)
        else:
            df['Current_Phase'] = 'P'
            df['Current_Phase_Numeric'] = 1

        return df

    def _create_composite_features(self, df):
        """Create clinically meaningful composite features."""
        # Deformity severity
        df['Deformity_Severity'] = (
            df.get('Bunion', 0).fillna(0) * 1 +
            df.get('Claw', 0).fillna(0) * 2 +
            df.get('Hammer', 0).fillna(0) * 2 +
            df.get('Charcot Arthropathy', 0).fillna(0) * 5 +
            df.get('Flat (Pes Planus) Arch', 0).fillna(0) * 1 +
            df.get('Abnormally High Arch', 0).fillna(0) * 1
        )

        # Mobility risk factors
        df['Mobility_Risk'] = (
            (df.get('Age', 0) > 70).astype(int) +
            (df.get('Weight (Kg)', 0) > 90).astype(int) +
            df.get('Sensory Peripheral', 0).fillna(0) +
            (df.get('Age', 0) > 80).astype(int)
        )

        # Moisture management need
        df['Moisture_Management_Need'] = (
            df.get('Exudate Amount (None:0,Minor,Medium,Severe:3)', 0).fillna(0) +
            df.get('Maceration at Peri-ulcer', 0).fillna(0) * 2 +
            df.get('Edema at Peri-ulcer', 0).fillna(0) +
            df.get('Wound Tunneling', 0).fillna(0) * 2
        )

        # Infection risk
        if 'Odor' in df.columns:
            df['Odor_Binary'] = df['Odor'].astype(str).str.lower().isin(['yes', 'present', 'true', '1']).astype(int)
        else:
            df['Odor_Binary'] = 0

        df['Infection_Risk'] = (
            df.get('Odor_Binary', 0) * 2 +
            df.get('Erythema at Peri-ulcer', 0).fillna(0) +
            df.get('Pale Colour at Peri-ulcer', 0).fillna(0)
        )

        return df

    def build_similarity_database(self, df):
        """
        Build database of successful cases for similarity matching.

        Parameters
        ----------
        df : pd.DataFrame
            Training data

        Returns
        -------
        self : DFUTreatmentRecommendationSystem
            Self reference for method chaining
        """
        print("\n" + "="*70)
        print("BUILDING SIMILARITY DATABASE")
        print("="*70)

        # Identify multiple appointment cases only
        appt_counts = df.groupby(['Patient#', 'DFU#'])['Appt#'].nunique()
        multiple_appt_mask = df.set_index(['Patient#', 'DFU#']).index.isin(
            appt_counts[appt_counts > 1].index
        )

        # Filter to successful cases
        success_indicators = (
            (df['Current_Phase'].isin(['P', 'R'])) |
            (df['Healing Phase'].str.contains(r'\+', na=False, regex=True))
        )

        successful_cases = df[multiple_appt_mask & success_indicators].copy()

        print(f"  Total records: {len(df)}")
        print(f"  Multiple appointment records: {multiple_appt_mask.sum()}")
        print(f"  Successful cases identified: {len(successful_cases)}")

        self.successful_cases = successful_cases

        # Initialize similarity matcher
        self.similarity_matcher = HierarchicalSimilarityMatcher(successful_cases)

        return self

    def recommend_offloading(self, patient_data):
        """
        Recommend offloading with clinical decision rules and similarity matching.

        Parameters
        ----------
        patient_data : pd.DataFrame
            Patient data (single row)

        Returns
        -------
        recommendation : dict
            Recommendation dictionary
        """
        patient_data = self.prepare_data(patient_data)

        # Rule-based decisions first
        if 'Charcot Arthropathy' in patient_data.columns and patient_data['Charcot Arthropathy'].iloc[0] == 1:
            return {
                'recommendation': 'Total Contact Cast',
                'score': 3.0,
                'confidence': 0.90,
                'rationale': 'Charcot arthropathy requires maximum offloading',
                'method': 'rule-based'
            }

        if 'Deformity_Severity' in patient_data.columns and patient_data['Deformity_Severity'].iloc[0] > 3:
            return {
                'recommendation': 'Boot/RCW',
                'score': 2.0,
                'confidence': 0.85,
                'rationale': 'High deformity severity requires enhanced offloading',
                'method': 'rule-based'
            }

        # Use similarity matching
        return self.similarity_matcher.recommend_offloading(patient_data)

    def recommend_dressing(self, patient_data):
        """
        Recommend dressing based on chronicity and clinical patterns.

        Parameters
        ----------
        patient_data : pd.DataFrame
            Patient data (single row)

        Returns
        -------
        recommendation : dict
            Recommendation dictionary
        """
        patient_data = self.prepare_data(patient_data)

        onset_days = patient_data['Onset (Days)'].iloc[0] if 'Onset (Days)' in patient_data.columns else 180
        wound_score = patient_data['Wound Score'].iloc[0] if 'Wound Score' in patient_data.columns else 0.5
        location_heel = patient_data['Location_Heel'].iloc[0] if 'Location_Heel' in patient_data.columns else 0

        # Clinical decision rules
        if location_heel == 1 and wound_score > 0.8:
            return {
                'recommendation': 'vac',
                'confidence': 0.85,
                'rationale': 'Heel location with significant wound score - VAC indicated',
                'method': 'rule-based'
            }

        # Chronicity-based decisions
        if onset_days > 365:
            if wound_score < 0.3:
                return {
                    'recommendation': 'polysporin',
                    'confidence': 0.80,
                    'rationale': 'Very chronic (>1 year) mild wound - simple antimicrobial',
                    'method': 'rule-based'
                }
            else:
                return {
                    'recommendation': 'Idosorb',
                    'confidence': 0.85,
                    'rationale': 'Chronic wound (>1 year) - enhanced iodine absorption needed',
                    'method': 'rule-based'
                }

        # Use similarity matching
        return self.similarity_matcher.recommend_dressing(patient_data, onset_days)

    def detect_red_flags(self, patient_data):
        """
        Detect red flags requiring special attention.

        Parameters
        ----------
        patient_data : pd.DataFrame
            Patient data (single row)

        Returns
        -------
        flags : list
            List of detected red flags
        """
        flags = []

        if 'Charcot Arthropathy' in patient_data.columns:
            if patient_data['Charcot Arthropathy'].iloc[0] == 1:
                flags.append('CHARCOT')

        if 'Deformity_Severity' in patient_data.columns:
            if patient_data['Deformity_Severity'].iloc[0] > 3:
                flags.append('HIGH_DEFORMITY')

        # Multiple risk factors
        risk_count = 0
        if 'Mobility_Risk' in patient_data.columns and patient_data['Mobility_Risk'].iloc[0] > 2:
            risk_count += 1
        if 'Age' in patient_data.columns and patient_data['Age'].iloc[0] > 80:
            risk_count += 1
        if 'Weight (Kg)' in patient_data.columns and patient_data['Weight (Kg)'].iloc[0] > 100:
            risk_count += 1
        if 'Wound Score' in patient_data.columns and patient_data['Wound Score'].iloc[0] > 2.5:
            risk_count += 1

        if risk_count >= 3:
            flags.append('MULTIPLE_RISKS')

        # Chronicity
        if 'Onset (Days)' in patient_data.columns:
            if patient_data['Onset (Days)'].iloc[0] > 365:
                flags.append('VERY_CHRONIC')

        return flags

    def evaluate_recommendations(self, test_df):
        """
        Evaluate recommendation system on test data.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test dataset

        Returns
        -------
        results : dict
            Evaluation results
        """
        print("\n" + "="*70)
        print("RECOMMENDATION SYSTEM EVALUATION")
        print("="*70)

        test_df = self.prepare_data(test_df)

        # Evaluate offloading
        print("\n1. OFFLOADING RECOMMENDATIONS:")
        print("-"*40)

        offloading_results = []
        for idx, row in test_df.iterrows():
            patient = pd.DataFrame([row])
            rec = self.recommend_offloading(patient)

            offloading_results.append({
                'actual': row.get('Offloading Score', 1.0),
                'predicted': rec['score'],
                'confidence': rec['confidence'],
                'method': rec['method']
            })

        off_df = pd.DataFrame(offloading_results)

        exact_match = (off_df['actual'] == off_df['predicted']).mean()
        close_match = (abs(off_df['actual'] - off_df['predicted']) <= 1).mean()
        avg_confidence = off_df['confidence'].mean()

        print(f"  Total recommendations: {len(off_df)}")
        print(f"  Exact match rate: {exact_match:.1%}")
        print(f"  Within-1 match rate: {close_match:.1%}")
        print(f"  Average confidence: {avg_confidence:.3f}")

        # Evaluate dressing
        print("\n2. DRESSING RECOMMENDATIONS:")
        print("-"*40)

        dressing_results = []
        for idx, row in test_df.iterrows():
            patient = pd.DataFrame([row])
            rec = self.recommend_dressing(patient)

            dressing_results.append({
                'actual': row.get('Dressing', 'Betadine'),
                'predicted': rec['recommendation'],
                'confidence': rec['confidence'],
                'method': rec['method'],
                'chronicity': row.get('Onset_Category', 'Unknown')
            })

        dress_df = pd.DataFrame(dressing_results)

        match_rate = (dress_df['actual'] == dress_df['predicted']).mean()
        avg_confidence = dress_df['confidence'].mean()

        print(f"  Total recommendations: {len(dress_df)}")
        print(f"  Match rate: {match_rate:.1%}")
        print(f"  Average confidence: {avg_confidence:.3f}")

        # Red flags
        print("\n3. RED FLAG DETECTION:")
        print("-"*40)

        red_flag_count = 0
        for idx, row in test_df.iterrows():
            patient = pd.DataFrame([row])
            flags = self.detect_red_flags(patient)
            if flags:
                red_flag_count += 1

        print(f"  Cases with red flags: {red_flag_count}/{len(test_df)} ({red_flag_count/len(test_df)*100:.1f}%)")

        return {
            'offloading_exact_match': exact_match,
            'offloading_close_match': close_match,
            'dressing_match': match_rate,
            'offloading_confidence': off_df['confidence'].mean(),
            'dressing_confidence': dress_df['confidence'].mean()
        }
