"""
Configuration constants and mappings for DFU Healing Trajectory Prediction.

This module contains all constant values, feature lists, and mappings used across the project.
All preprocessing steps remain unchanged to preserve the unique preprocessing logic.
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Selected base features based on clinical relevance (from RiskPredict_V12)
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

# Optimized features from hyperparameter optimization (42 features)
# These were selected through Bayesian optimization with transition-based validation
# achieving 0.6939 balanced accuracy (October 2025 optimization run)
OPTIMIZED_FEATURES = [
    "Days_To_Next_Appt",
    "History_Phase_0_Proportion",
    "Initial_Phase",
    "Exudate Amount (None:0,Minor,Medium,Severe:3)",
    "Appt Days",
    "Side (Left:0, Right:1)",
    "Type of Diabetes",
    "History_Phase_2_Count",
    "No Toes Deformities",
    "History_Phase_1_Proportion",
    "Onset (Days)",
    "Cancer History",
    "History_Phase_0_Count",
    "Patient_Cluster_Slow_Healer",
    "Previous_Phase_X_Treatment",
    "Weight (Kg)",
    "History_Phase_2_Proportion",
    "Foot Callus",
    "Clinical Score",
    "Number of DFUs",
    "Age_above_70",
    "Foot Score",
    "Height (cm)",
    "Peri-Ulcer Temperature (Â°C)",
    "Location",
    "Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)",
    "Wound Centre Temperature Normalized (Â°C)",
    "Wound Score",
    "No Peri-ulcer Conditions (False:0, True:1)",
    "Offloading: Therapeutic Footwear_Consistency",
    "BMI",
    "Patient_Cluster_Fast_Healer",
    "Wound Centre Temperature (Â°C)",
    "History_Phase_1_Count",
    "Exudate Amount (None:0,Minor,Medium,Severe:3)_History_Mean",
    "Peri-Ulcer Temperature Normalized (Â°C)",
    "Claw",
    "Current_Offloading_Score",
    "Edema at Peri-ulcer",
    "Leg Score",
    # Additional features that might be useful
    "Previous_Phase",
    "Healing_Momentum"
]

# Legacy selected features (kept for reference)
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

# Engineered features to keep
ENGINEERED_FEATURES_TO_KEEP = [
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

# Features to completely remove
FEATURES_TO_REMOVE = [
    'Healing Phase',
    'Phase Confidence (%)',
    'Type of Pain2',
    'Type of Pain_Grouped2',
    'ID'
]

# String columns that need conversion to numeric
STRING_TO_NUMERIC_MAPPINGS = {
    'Sex (F:0, M:1)': {'F': 0, 'f': 0, 'Female': 0, 'M': 1, 'm': 1, 'Male': 1},
    'Side (Left:0, Right:1)': {'Left': 0, 'left': 0, 'Right': 1, 'right': 1},
}

# Columns with specific ordinal mappings
ORDINAL_MAPPINGS = {
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
CATEGORICAL_COLUMNS = [
    'Foot Aspect', 'Location', 'Odor', 'Type of Pain',
    'Type of Pain Grouped', 'Dressing'
]

# Integer columns
INTEGER_COLUMNS = [
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
NUMERICAL_COLUMNS = [
    'Age', 'Weight (Kg)', 'Height (cm)', 'Onset (Days)', 'Appt Days',
    'Wound Centre Temperature (Â°C)', 'Peri-Ulcer Temperature (Â°C)',
    'Intact Skin Temperature (Â°C)', 'Wound Centre Temperature Normalized (Â°C)',
    'Peri-Ulcer Temperature Normalized (Â°C)'
]

# Target column and phase mappings
TARGET_COLUMN = 'Healing Phase Abs'
ORDINAL_MAPPING = {'I': 0, 'P': 1, 'R': 2}
REVERSE_MAPPING = {0: 'I', 1: 'P', 2: 'R'}

# Phase transition classifications
FAVORABLE_TRANSITIONS = {
    (0, 1): 'Improving: I→P',
    (0, 2): 'Improving: I→R',
    (1, 2): 'Improving: P→R',
    (2, 2): 'Stable-Good: R→R'
}

ACCEPTABLE_TRANSITIONS = {
    (1, 1): 'Stable-Acceptable: P→P'
}

UNFAVORABLE_TRANSITIONS = {
    (1, 0): 'Worsening: P→I',
    (2, 0): 'Worsening: R→I',
    (2, 1): 'Worsening: R→P',
    (0, 0): 'Stable-Poor: I→I'
}

# Offloading features
OFFLOADING_FEATURES = [
    'Offloading: Therapeutic Footwear',
    'Offloading: Scotcast Boot or RCW',
    'Offloading: Half Shoes or Sandals',
    'Offloading: Total Contact Cast',
    'Offloading: Crutches, Walkers or Wheelchairs'
]

# Treatment consistency features
TREATMENT_CONSISTENCY_FEATURES = [
    'Dressing Grouped',
    'Offloading: Therapeutic Footwear',
    'Offloading: Scotcast Boot or RCW',
    'Offloading: Half Shoes or Sandals',
    'Offloading: Total Contact Cast',
    'Offloading: Crutches, Walkers or Wheelchairs'
]

# Clinical essential features for optimization
CLINICAL_ESSENTIAL_FEATURES = [
    'Days_To_Next_Appt',
    'Current_Phase_Numeric',
]

# All original features from DataMaster_Processed_V12_WithMissing.csv
# (excluding target, ID, and removed features)
ALL_ORIGINAL_FEATURES = [
    'Age',
    'Sex (F:0, M:1)',
    'Weight (Kg)',
    'Height (cm)',
    'Smoking',
    'Alcohol Consumption',
    'Habits Score',
    'Type of Diabetes',
    'Heart Conditions',
    'Cancer History',
    'Sensory Peripheral',
    'Clinical Score',
    'Number of DFUs',
    'Side (Left:0, Right:1)',
    'Foot Aspect',
    'Location',
    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)',
    'Onset (Days)',
    'No Toes Deformities',
    'Bunion',
    'Claw',
    'Hammer',
    'Charcot Arthropathy',
    'Flat (Pes Planus) Arch',
    'Abnormally High Arch',
    'No Arch Deformities',
    'Foot Score',
    'Appt Days',
    'Wound Centre Temperature (°C)',
    'Peri-Ulcer Temperature (°C)',
    'Intact Skin Temperature (°C)',
    'Wound Centre Temperature Normalized (°C)',
    'Peri-Ulcer Temperature Normalized (°C)',
    'Pain Level',
    'Type of Pain',
    'Type of Pain Grouped',
    'Wound Tunneling',
    'Exudate Amount (None:0,Minor,Medium,Severe:3)',
    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)',
    'Odor',
    'No Peri-ulcer Conditions (False:0, True:1)',
    'Erythema at Peri-ulcer',
    'Edema at Peri-ulcer',
    'Pale Colour at Peri-ulcer',
    'Maceration at Peri-ulcer',
    'Wound Score',
    'Dressing',
    'Dressing Grouped',
    'No Foot Abnormalities',
    'Foot Hair Loss',
    'Foot Dry Skin',
    'Foot Fissure Cracks',
    'Foot Callus',
    'Thickened Toenail',
    'Foot Fungal Nails',
    'Leg Score',
    'No Offloading',
    'Offloading: Therapeutic Footwear',
    'Offloading: Scotcast Boot or RCW',
    'Offloading: Half Shoes or Sandals',
    'Offloading: Total Contact Cast',
    'Offloading: Crutches, Walkers or Wheelchairs',
    'Offloading Score',
]

# All engineered features created during preprocessing
ALL_ENGINEERED_FEATURES = [
    # Temporal/History features
    'Days_To_Next_Appt',
    'Previous_Phase',
    'Previous_Phase_X_Treatment',
    'Initial_Phase',
    'History_Length',
    'History_Completeness',
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
    # Treatment features
    'Treatment_Intensity_Score',
    'Current_Dressing',
    'Current_Dressing_Grouped',
    'Current_Offloading_Score',
    # Patient clustering
    'Patient_Cluster',
    'Patient_Cluster_Fast_Healer',
    'Patient_Cluster_Slow_Healer',
    'Patient_Cluster_Moderate_Healer',
    # Trend features
    'Pain Level_Trend_Direction',
    'Exudate Amount (None:0,Minor,Medium,Severe:3)_History_Mean',
    # Treatment consistency features (auto-generated)
    'Dressing Grouped_Consistency',
    'Offloading: Therapeutic Footwear_Consistency',
    'Offloading: Scotcast Boot or RCW_Consistency',
    'Offloading: Half Shoes or Sandals_Consistency',
    'Offloading: Total Contact Cast_Consistency',
    'Offloading: Crutches, Walkers or Wheelchairs_Consistency',
    # Derived age features
    'Age_above_60',
    'Age_above_70',
    # BMI (derived from weight/height)
    'BMI',
]

# Combined list of all features (original + engineered)
# Use this when you want to search over all possible features
ALL_FEATURES = ALL_ORIGINAL_FEATURES + ALL_ENGINEERED_FEATURES

# ============================================================================
# Feature Type Classifications for Data Processing
# ============================================================================

# Non-numeric columns that need to be excluded or converted before imputation
# These columns contain string/object data types that cannot be used with median imputation
NON_NUMERIC_COLUMNS = [
    'ID',
    'Sex (F:0, M:1)',  # Contains 'F', 'M' strings
    'Side (Left:0, Right:1)',  # Contains 'Left', 'Right' strings
    'Foot Aspect',  # Contains 'Plantar', 'Medial', 'Dorsal', 'Lateral'
    'Location',  # Free text descriptions
    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)',  # Contains 'Hallux', 'toes', etc.
    'Type of Pain',  # Free text or categorical
    'Type of Pain Grouped',  # Categorical: 'NoPain', 'GeneralAches', etc.
    'Type of Pain2',  # Same as Type of Pain (mostly null)
    'Type of Pain_Grouped2',  # Same as Type of Pain Grouped (mostly null)
    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)',  # Contains 'Serous', 'Bloody', etc.
    'Odor',  # Contains 'NoOdor', 'Unpleasant'
    'Dressing',  # Free text dressing names
    'Dressing Grouped',  # Categorical: 'AbsorbentDressing', 'Antiseptic', etc.
    'Healing Phase',  # Contains 'P-', 'P+', 'P', 'I+', etc.
    'Healing Phase Abs',  # Contains 'I', 'P', 'R'
    # Engineered features that might be non-numeric
    'Patient_Cluster',  # Categorical cluster names
    'Current_Dressing',  # Might inherit string type from Dressing
    'Current_Dressing_Grouped',  # Might inherit string type from Dressing Grouped
]

# Columns that should be excluded from feature selection entirely
# (identifiers, targets, or columns used for other purposes)
EXCLUDE_FROM_FEATURES = [
    'ID',
    'Patient#',
    'DFU#',
    'Appt#',
    'Healing Phase',
    'Healing Phase Abs',
    'Phase Confidence (%)',
    'Next_Healing_Phase',  # Target column
    'Type of Pain2',  # Mostly null (716/890)
    'Type of Pain_Grouped2',  # Mostly null (716/890)
]

# Columns that can be safely converted to numeric using ordinal mappings
# These are encoded in constants.py ORDINAL_MAPPINGS or STRING_TO_NUMERIC_MAPPINGS
CONVERTIBLE_TO_NUMERIC = [
    'Sex (F:0, M:1)',  # F=0, M=1
    'Side (Left:0, Right:1)',  # Left=0, Right=1
    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)',  # hallux=1, toes=2, middle=3, heel=4, ankle=5
    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)',  # serous=1, haemoserous=2, bloody=3, thick=4
    'Dressing Grouped',  # nodressing=0, bandaid=1, etc.
]

# Columns that need label encoding (arbitrary numeric mapping)
# These have no natural ordering
LABEL_ENCODE_COLUMNS = [
    'Foot Aspect',  # Plantar, Medial, Dorsal, Lateral
    'Location',  # Free text - many unique values
    'Type of Pain',  # Free text - many unique values
    'Type of Pain Grouped',  # NoPain, GeneralAches, ChronicPain, etc.
    'Odor',  # NoOdor, Unpleasant
    'Dressing',  # Many unique dressing names
]

# Binary columns (0/1 values stored as float due to missing data)
BINARY_COLUMNS = [
    'Smoking',
    'Alcohol Consumption',
    'Heart Conditions',
    'Cancer History',
    'Sensory Peripheral',
    'No Toes Deformities',
    'Bunion',
    'Claw',
    'Hammer',
    'Charcot Arthropathy',
    'Flat (Pes Planus) Arch',
    'Abnormally High Arch',
    'No Arch Deformities',
    'Wound Tunneling',
    'No Peri-ulcer Conditions (False:0, True:1)',
    'Erythema at Peri-ulcer',
    'Edema at Peri-ulcer',
    'Pale Colour at Peri-ulcer',
    'Maceration at Peri-ulcer',
    'No Foot Abnormalities',
    'Foot Hair Loss',
    'Foot Dry Skin',
    'Foot Fissure Cracks',
    'Foot Callus',
    'Thickened Toenail',
    'Foot Fungal Nails',
    'No Offloading',
    'Offloading: Therapeutic Footwear',
    'Offloading: Scotcast Boot or RCW',
    'Offloading: Half Shoes or Sandals',
    'Offloading: Total Contact Cast',
    'Offloading: Crutches, Walkers or Wheelchairs',
    # Engineered binary features
    'Age_above_60',
    'Age_above_70',
    'Patient_Cluster_Fast_Healer',
    'Patient_Cluster_Slow_Healer',
    'Patient_Cluster_Moderate_Healer',
]

# Continuous numerical columns (can use median imputation)
CONTINUOUS_COLUMNS = [
    'Age',
    'Weight (Kg)',
    'Height (cm)',
    'BMI',
    'Onset (Days)',
    'Appt Days',
    'Wound Centre Temperature (°C)',
    'Peri-Ulcer Temperature (°C)',
    'Intact Skin Temperature (°C)',
    'Wound Centre Temperature Normalized (°C)',
    'Peri-Ulcer Temperature Normalized (°C)',
    # Engineered continuous features
    'Days_To_Next_Appt',
    'Avg_Days_Between_Appts',
    'Std_Days_Between_Appts',
    'Healing_Momentum',
    'Treatment_Intensity_Score',
    'History_Phase_0_Proportion',
    'History_Phase_1_Proportion',
    'History_Phase_2_Proportion',
    'Exudate Amount (None:0,Minor,Medium,Severe:3)_History_Mean',
]

# Ordinal/score columns (integer values with natural ordering)
ORDINAL_SCORE_COLUMNS = [
    'Type of Diabetes',  # 0, 1, 2
    'Habits Score',
    'Clinical Score',
    'Number of DFUs',
    'Foot Score',
    'Pain Level',  # 0-10 scale
    'Exudate Amount (None:0,Minor,Medium,Severe:3)',  # 0-3
    'Wound Score',
    'Leg Score',
    'Offloading Score',
    # Engineered ordinal features
    'Previous_Phase',  # 0, 1, 2
    'Initial_Phase',  # 0, 1, 2
    'History_Length',
    'History_Completeness',
    'History_Phase_0_Count',
    'History_Phase_1_Count',
    'History_Phase_2_Count',
    'Phase_Improvements_Count',
    'Phase_Regressions_Count',
    'History_Favorable_Transitions',
    'History_Acceptable_Transitions',
    'History_Unfavorable_Transitions',
    'Appointments_So_Far',
    'Current_Offloading_Score',
    'Previous_Phase_X_Treatment',
    'Pain Level_Trend_Direction',
]

# All numeric-safe columns that can be used with median imputation
# This combines BINARY_COLUMNS, CONTINUOUS_COLUMNS, and ORDINAL_SCORE_COLUMNS
NUMERIC_SAFE_COLUMNS = BINARY_COLUMNS + CONTINUOUS_COLUMNS + ORDINAL_SCORE_COLUMNS

# Treatment consistency features (auto-generated, all numeric)
TREATMENT_CONSISTENCY_NUMERIC = [
    'Dressing Grouped_Consistency',
    'Offloading: Therapeutic Footwear_Consistency',
    'Offloading: Scotcast Boot or RCW_Consistency',
    'Offloading: Half Shoes or Sandals_Consistency',
    'Offloading: Total Contact Cast_Consistency',
    'Offloading: Crutches, Walkers or Wheelchairs_Consistency',
]

# ============================================================================
# Feature Name Mapping for Reader-Friendly Display
# ============================================================================
# Based on Table 1 "Selected Features by Category and Importance Ranking" in paper
# Maps technical/code feature names to clinician-friendly display names

FEATURE_DISPLAY_NAMES = {
    # Temporal features (Rank 1, 5, 11)
    'Days_To_Next_Appt': 'Days to Next Appointment',
    'days_to_next_appt': 'Days to Next Appointment',
    'Cumulative_Treatment_Days': 'Cumulative Days in Treatment',
    'Total_Treatment_Days': 'Cumulative Days in Treatment',
    'Days_Since_Onset': 'Wound Duration Since Onset',
    'Onset (Days)': 'Wound Duration Since Onset',
    'Cumulative_Phase_Duration': 'Cumulative Phase Duration',
    'Days_In_Current_Phase': 'Days in Current Phase',
    'Appt Days': 'Appointment Interval',

    # Historical phase proportions (Rank 2, 10, 17)
    # Phases: 0=Inflammatory (I), 1=Proliferative (P), 2=Remodeling (R)
    'History_Phase_0_Proportion': 'Historical Inflammatory Phase Proportion',
    'History_Phase_1_Proportion': 'Historical Proliferative Phase Proportion',
    'History_Phase_2_Proportion': 'Historical Remodeling Phase Proportion',

    # Historical phase counts (Rank 8, 13)
    'History_Phase_0_Count': 'Historical Inflammatory Phase Count',
    'History_Phase_1_Count': 'Historical Proliferative Phase Count',
    'History_Phase_2_Count': 'Historical Remodeling Phase Count',

    # Healing phase features (Rank 3)
    'Initial_Phase': 'Initial Healing Phase',
    'Healing_Phase': 'Current Healing Phase',
    'Previous_Phase': 'Previous Healing Phase',
    'Next_Healing_Phase': 'Next Healing Phase',

    # Wound characteristics (Rank 4, 18, 24, 27, 29)
    'Exudate Amount (None:0,Minor,Medium,Severe:3)': 'Exudate Amount',
    'Exudate_Amount': 'Exudate Amount',
    'Exudate': 'Exudate Amount',
    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': 'Exudate Appearance',
    'Exudate_Appearance': 'Exudate Appearance',
    'Foot Callus': 'Foot Callus Presence',
    'Callus': 'Foot Callus Presence',
    'Peri-Ulcer Temperature (Â°C)': 'Peri-Ulcer Temperature',
    'Peri-Ulcer Temperature (°C)': 'Peri-Ulcer Temperature',
    'Peri-Ulcer Temperature Normalized (Â°C)': 'Normalized Peri-Ulcer Temperature',
    'Peri-Ulcer Temperature Normalized (°C)': 'Normalized Peri-Ulcer Temperature',
    'Peri_Ulcer_Temp': 'Peri-Ulcer Temperature',
    'Wound Centre Temperature (°C)': 'Wound Center Temperature',
    'Wound Centre Temperature Normalized (Â°C)': 'Normalized Wound Center Temperature',
    'Wound Centre Temperature Normalized (°C)': 'Normalized Wound Center Temperature',
    'Wound_Center_Temp_Norm': 'Normalized Wound Center Temperature',
    'Intact Skin Temperature (°C)': 'Intact Skin Temperature',
    'No Peri-ulcer Conditions (False:0, True:1)': 'Absence of Peri-Ulcer Conditions',
    'Peri_Ulcer_Conditions': 'Absence of Peri-Ulcer Conditions',
    'Wound_Area': 'Wound Area',
    'Wound_Depth': 'Wound Depth',
    'Wound Score': 'Wound Severity Score',

    # Anatomical features (Rank 6, 9, 22, 25, 26)
    'Side (Left:0, Right:1)': 'Affected Foot Side',
    'Affected_Side': 'Affected Foot Side',
    'Side': 'Affected Foot Side',
    'No Toes Deformities': 'Absence of Toe Deformities',
    'Toe_Deformities': 'Absence of Toe Deformities',
    'Foot Score': 'Foot Deformity Score',
    'Foot_Deformity_Score': 'Foot Deformity Score',
    'Location': 'Anatomical Location',
    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)': 'Location Risk Category',
    'Location_Risk': 'Location Risk Category',

    # Patient factors (Rank 7, 12, 16, 20, 21, 23)
    'Type of Diabetes': 'Diabetes Type',
    'Diabetes_Type': 'Diabetes Type',
    'Cancer History': 'Cancer History',
    'Cancer_History': 'Cancer History',
    'Weight (Kg)': 'Body Weight',
    'Weight': 'Body Weight',
    'Number of DFUs': 'Number of Concurrent DFUs',
    'Concurrent_DFUs': 'Number of Concurrent DFUs',
    'Age_above_70': 'Age Above 70 Years',
    'Age_Above_70': 'Age Above 70 Years',
    'Age': 'Patient Age',
    'Height (cm)': 'Body Height',
    'Height': 'Body Height',
    'BMI': 'Body Mass Index',
    'Sex (F:0, M:1)': 'Patient Gender',
    'Gender': 'Patient Gender',

    # Engineered features (Rank 14, 15)
    'Patient_Cluster_Slow_Healer': 'Slow Healer Phenotype Assignment',
    'Slow_Healer': 'Slow Healer Phenotype Assignment',
    'Patient_Cluster_Fast_Healer': 'Fast Healer Phenotype Assignment',
    'Patient_Cluster': 'Patient Healing Phenotype',
    'Previous_Phase_X_Treatment': 'Phase-Adjusted Treatment Effect',
    'Prev_Phase_Treatment': 'Phase-Adjusted Treatment Effect',
    'Prev_Phase_Treatment_Interaction': 'Phase-Adjusted Treatment Effect',
    'Exudate Amount (None:0,Minor,Medium,Severe:3)_Consistency': 'Exudate Amount Consistency',

    # Composite scores (Rank 19, 28)
    'Clinical Score': 'Composite Clinical Score',
    'Clinical_Score': 'Composite Clinical Score',
    'Wound_Severity': 'Wound Severity Score',

    # Treatment features (Rank 30)
    'Offloading: Therapeutic Footwear_Consistency': 'Therapeutic Footwear Consistency',
    'Footwear_Consistency': 'Therapeutic Footwear Consistency',
    'Offloading_Type': 'Offloading Type',
    'Dressing_Type': 'Dressing Type',
    'Dressing Grouped': 'Dressing Category',
    'Current_Offloading_Score': 'Current Offloading Score',
    'Treatment_Intensity_Score': 'Treatment Intensity Score',

    # Comorbidities
    'Sensory Peripheral': 'Peripheral Neuropathy',
    'Peripheral_Neuropathy': 'Peripheral Neuropathy',
    'Peripheral_Vascular_Disease': 'Peripheral Vascular Disease',
    'Heart Conditions': 'Cardiac Disease',
    'Cardiac_Disease': 'Cardiac Disease',
    'Renal_Disease': 'Renal Disease',
    'Hypertension': 'Hypertension',

    # Appointment features
    'Appt#': 'Appointment Number',
    'Appointments_So_Far': 'Appointments to Date',
    'Total_Appointments': 'Appointments to Date',
    'Avg_Days_Between_Appts': 'Mean Appointment Interval',
    'Appt_Interval_Mean': 'Mean Appointment Interval',
    'Std_Days_Between_Appts': 'Appointment Interval Variability',
    'Appt_Interval_Std': 'Appointment Interval Variability',

    # History/trend features
    'Healing_Momentum': 'Healing Momentum',
    'History_Length': 'Treatment History Length',
    'History_Completeness': 'History Completeness',
    'Phase_Improvements_Count': 'Phase Improvement Count',
    'Phase_Regressions_Count': 'Phase Regression Count',
    'History_Favorable_Transitions': 'Historical Favorable Transitions',
    'History_Acceptable_Transitions': 'Historical Acceptable Transitions',
    'History_Unfavorable_Transitions': 'Historical Unfavorable Transitions',
    'Exudate Amount (None:0,Minor,Medium,Severe:3)_History_Mean': 'Historical Mean Exudate',

    # Other anatomical/wound features
    'Foot Aspect': 'Foot Aspect',
    'Wound Tunneling': 'Wound Tunneling',
    'Pain Level': 'Pain Level',
    'Edema at Peri-ulcer': 'Peri-Ulcer Edema',
    'Erythema at Peri-ulcer': 'Peri-Ulcer Erythema',
    'Leg Score': 'Leg Score',
    'Claw': 'Claw Toe Deformity',
    'Bunion': 'Bunion',
    'Hammer': 'Hammer Toe',
}
