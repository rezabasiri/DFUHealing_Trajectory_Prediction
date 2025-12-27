# Temporal Machine Learning Framework for Diabetic Foot Ulcer Healing Trajectory Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the implementation of a temporal machine learning framework for predicting diabetic foot ulcer (DFU) healing trajectories at the next clinical appointment and recommending evidence-based treatment protocols. The system transforms DFU management from reactive assessment to proactive prediction, enabling clinicians to optimize treatments before adverse trajectories manifest.

**Publication:** "Temporal Machine Learning Framework for Diabetic Foot Ulcer Healing Trajectory Prediction" - *BioMedical Engineering OnLine* (Under Review)

## Clinical Significance

Diabetic foot ulcers affect approximately 6.3% of the global diabetic population and serve as the gateway to 85% of all diabetes-related lower extremity amputations. Despite advances in wound care, only 50% of DFUs achieve healing within one year. This framework addresses critical clinical gaps by:

- **Enabling proactive intervention:** Predicting healing phase transitions at the next appointment (rather than final outcomes) allows timely treatment modification
- **Accessible deployment:** Using only routinely collected clinical metadata without requiring specialized wound imaging infrastructure
- **Treatment guidance:** Providing evidence-based offloading and dressing recommendations aligned with clinical decision-making

## Key Features

### Prediction Framework
- **Transition-based classification:** Predicts three clinically meaningful categories (Favorable, Acceptable, Unfavorable) rather than direct phase prediction
- **Temporal feature engineering:** Novel approach normalizing clinical measurements by inter-appointment intervals to address irregular follow-up schedules
- **Patient phenotyping:** Unsupervised clustering identifies fast-healer and slow-healer phenotypes for personalized trajectory forecasting
- **Performance:** 78.0% ± 4.0% accuracy with balanced category performance (weighted F1: 0.76)

### Treatment Recommendation System
- **Hierarchical similarity matching:** Combines case-based reasoning with clinical decision rules
- **Chronicity-stratified recommendations:** Adapts recommendations based on wound age and treatment resistance
- **Offloading recommendations:** 88.7% within-category agreement with clinical practice
- **Dressing recommendations:** Chronicity-dependent performance appropriately reflecting clinical reality (acute wounds: 83.7%, very chronic wounds: 5.6%)

## Repository Structure

```
DFUHealing_Trajectory_Prediction/
├── optuna_search_feature_selection.py  # PRIMARY: Hyperparameter search
├── train_ensemble_calibration.py       # PRIMARY: Train & generate figures
├── src/                                # Source code modules
│   ├── config/                         # Configuration and constants
│   │   └── constants.py                # All constants, mappings, feature lists
│   ├── preprocessing/                  # Data preprocessing modules
│   │   ├── preprocessor.py             # Main DFU preprocessor class
│   │   ├── resampler.py                # Flexible resampling strategies
│   │   ├── transition_aware_weighting.py  # Transition label computation
│   │   └── utils.py                    # Preprocessing utilities
│   ├── models/                         # Model training and evaluation
│   │   ├── classifier.py               # Model creation and training
│   │   └── evaluation.py               # Metrics and analysis
│   ├── optimization/                   # Hyperparameter optimization
│   │   ├── scoring.py                  # Clinical scoring functions
│   │   └── feature_selection.py        # Feature selection logic
│   ├── visualization/                  # Plotting and analysis outputs
│   │   └── calibration_plots.py        # Calibration figure generation
│   └── recommendations/                # Treatment recommendation system
│       ├── treatment_system.py         # Main recommendation system
│       └── similarity.py               # Hierarchical similarity matching
├── config/                             # Configuration files
│   └── config.yaml                     # Default training configuration
├── grid_search_results/                # Optuna search results (JSON)
├── figures/                            # Generated figures and metrics
├── data/                               # Data directory
│   └── DataMaster_Processed_V12_WithMissing.csv
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```


## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/rezabasiri/DFUHealing_Trajectory_Prediction.git
cd DFUHealing_Trajectory_Prediction

# Create virtual environment (recommended)
python -m venv DFUPred
source DFUPred/bin/activate  # On Windows: DFUPred\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core libraries:
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `imbalanced-learn>=0.9.0` - Resampling strategies
- `scikit-optimize>=0.9.0` - Bayesian optimization
- `shap>=0.40.0` - Feature importance analysis
- `matplotlib>=3.4.0` - Visualization
- `seaborn>=0.11.0` - Statistical visualization

## Usage

### Main Workflow (Two Scripts)

The workflow consists of two main scripts:

| Script | Purpose |
|--------|---------|
| `optuna_search_feature_selection.py` | Find optimal hyperparameters using Bayesian optimization |
| `train_ensemble_calibration.py` | Train model with best config, generate metrics & figures |

---

### Step 1: Hyperparameter Search

Use `optuna_search_feature_selection.py` to find the best configuration:

```bash
# Basic search (500 trials, optimizing for combined_score)
python optuna_search_feature_selection.py --n-trials 500 --optimize-for combined_score

# Quick test run (50 trials)
python optuna_search_feature_selection.py --n-trials 50 --optimize-for combined_score

# Optimize for calibration (lower ECE/MCE)
python optuna_search_feature_selection.py --n-trials 500 --optimize-for combined_score_calibration

# Optimize for minority class performance
python optuna_search_feature_selection.py --n-trials 500 --optimize-for combined_score_calibration_minority

# Multi-target optimization (runs separate studies for each)
python optuna_search_feature_selection.py --n-trials 500 --optimize-for combined_score combined_score_calibration f1_macro

# Use more parallel jobs (faster on multi-core systems)
python optuna_search_feature_selection.py --n-trials 500 --n-jobs 8 --optimize-for combined_score

# Resume an interrupted search
python optuna_search_feature_selection.py --n-trials 500 --resume --study-name my_study
```

**Output:** Results saved to `grid_search_results/` as JSON files containing best configurations for each optimization target.

---

### Step 2: Train with Best Configuration

Use `train_ensemble_calibration.py` to train with a winning configuration and generate publication-ready figures:

```bash
# List available configs in a results file
python train_ensemble_calibration.py -f grid_search_results/Dec10_results.json --list-configs

# Train with a specific config
python train_ensemble_calibration.py -f grid_search_results/Dec10_results.json -c combined_score

# Train with multiple configs (comparison run)
python train_ensemble_calibration.py -f grid_search_results/Dec10_results.json -c combined_score -c combined_score_calibration -c f1_macro

# Run all configs from a file
python train_ensemble_calibration.py -f grid_search_results/Dec10_results.json --all-configs

# Multiple files at once
python train_ensemble_calibration.py -f results1.json -f results2.json --all-configs

# Custom output directory
python train_ensemble_calibration.py -f results.json -c combined_score -o my_figures/

# Skip figure generation (metrics only)
python train_ensemble_calibration.py -f results.json -c combined_score --skip-figures

# Increase ensemble seeds for more robust metrics
python train_ensemble_calibration.py -f results.json -c combined_score --n-seeds 5

# Use 5-fold cross-validation instead of default 3-fold
python train_ensemble_calibration.py -f results.json -c combined_score --n-folds 5

# Use default config.yaml (no JSON file)
python train_ensemble_calibration.py --n-seeds 2 --n-folds 3
```

**Output per config:**
- `calibration_curves.png/pdf` - Per-class calibration reliability diagrams
- `calibration_combined.png/pdf` - All classes overlaid
- `probability_distributions.png/pdf` - Prediction confidence histograms
- `confusion_matrix.png/pdf` - Normalized confusion matrix
- `calibration_metrics.csv` - ECE, MCE, Brier score per class
- `calibration_curve_data.csv` - Bin-by-bin calibration data
- `summary.json` - Full configuration and metrics

---

### Complete Workflow Example

```bash
# 1. Run hyperparameter search
python optuna_search_feature_selection.py --n-trials 500 --optimize-for combined_score combined_score_calibration

# 2. Check what configs were found
python train_ensemble_calibration.py -f grid_search_results/optuna_search_*_combined_best_results.json --list-configs

# 3. Train and generate figures for best configs
python train_ensemble_calibration.py \
    -f grid_search_results/optuna_search_*_combined_best_results.json \
    -c combined_score \
    -c combined_score_calibration \
    --n-seeds 3 \
    -o figures/final_results/

# 4. Results are in figures/final_results/
ls figures/final_results/
```

---

### Key Configuration Files

| File | Purpose |
|------|---------|
| `config/config.yaml` | Default training parameters |
| `grid_search_results/*.json` | Saved Optuna search results |
| `notes.md` | Paper documentation (metrics, clinical recommendations) |
| `TRACKER.md` | Change log for development |

---

### Legacy Scripts (Reference Only)

These scripts are preserved for reference but the workflow above is recommended:

```bash
python train_with_transition_weights.py  # Reference training code
python optuna_search.py                  # Legacy phase-based search
```

### Treatment Recommendations

```python
from dfu_treatment_system import DFUTreatmentRecommendationSystem
import pandas as pd

# Initialize system
system = DFUTreatmentRecommendationSystem()

# Load and prepare data
df = pd.read_csv("your_data.csv")
df = system.prepare_data(df)

# Build similarity database
train_df, test_df = train_test_split(df, test_size=0.3)
system.build_similarity_database(train_df)

# Get recommendations for a patient
patient_data = test_df.iloc[[0]]
offloading_rec = system.recommend_offloading(patient_data)
dressing_rec = system.recommend_dressing(patient_data)

print(f"Offloading: {offloading_rec['recommendation']} (confidence: {offloading_rec['confidence']:.2f})")
print(f"Dressing: {dressing_rec['recommendation']} (confidence: {dressing_rec['confidence']:.2f})")
```

## Dataset

This work utilizes the Zivot dataset from a specialized wound care center in Alberta, Canada. The dataset comprises:

- **Patients:** 268 unique patients
- **Wounds:** 329 distinct DFUs
- **Appointments:** 889 clinical visits
- **Features:** 72 raw clinical features (56 numerical, 16 categorical)
- **Follow-up:** Mean 3.3 ± 3.2 appointments per patient

**Data availability:** The dataset cannot be publicly shared due to privacy regulations and institutional agreements. Researchers interested in collaboration may contact the corresponding author.

**Ethical approval:** 
- Conjoint Health Research Ethics Board, University of Calgary (#21-1052)
- Research Ethics Board, University Health Network (#21-5352)

## Methodology

### Model Architecture

**Classifier:** Extra Trees (Extremely Randomized Trees)
- **Estimators:** 437 trees
- **Maximum depth:** 39
- **Feature selection:** SHAP-based importance threshold (42 features from 102)
- **Dimensionality reduction:** 58.8%

### Temporal Feature Engineering

Novel approaches addressing irregular appointment scheduling:
- **Days to next appointment normalization:** Critical temporal feature (top predictor)
- **Historical phase aggregations:** Proportions and counts of inflammatory, proliferative, remodeling phases
- **Healing momentum:** Weighted recent trajectory dynamics
- **Treatment consistency scores:** Longitudinal intervention adherence patterns

### Validation Strategy

- **Cross-validation:** 5-fold patient-level stratified
- **Data augmentation:** Safe sequential appointment combinations preserving temporal continuity
- **Performance metrics:** Balanced accuracy, weighted F1, confusion matrices, ROC curves

## Results Summary

### Prediction Performance

| Metric | Value |
|--------|-------|
| Transition Accuracy | 78.0% ± 4.0% |
| Weighted F1 Score | 76.0% ± 2.0% |
| Macro F1 Score | 69.7% ± 2.8% |

### Category-Specific Performance

| Transition Category | Precision | Recall | F1 Score |
|---------------------|-----------|--------|----------|
| Favorable | 0.77 | 0.73 | 0.75 |
| Acceptable | 0.73 | 0.71 | 0.72 |
| Unfavorable | 0.73 | 0.89 | 0.84 |

### Treatment Recommendations

| Component | Performance |
|-----------|-------------|
| Offloading (within-category) | 88.7% agreement |
| Dressing (acute wounds) | 83.7% match rate |
| Dressing (very chronic wounds) | 5.6% match rate* |

*Low match rate for very chronic wounds appropriately reflects clinical reality that treatment-resistant cases require individualized experimentation rather than algorithmic protocols.

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{basiri2025temporal,
  title={Temporal Machine Learning Framework for Diabetic Foot Ulcer Healing Trajectory Prediction},
  author={Basiri, Reza and Saleh, Asem and Khan, Shehroz S. and Popovic, Milos R.},
  journal={BioMedical Engineering OnLine},
  year={2025},
  note={Under Review}
}
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Corresponding Author:** Reza Basiri  
Email: reza.basiri@mail.utoronto.ca

**Principal Investigator:** Dr. Milos R. Popovic  
KITE Research Institute, Toronto Rehabilitation Institute  
University Health Network, Toronto, Canada

## Acknowledgments

We thank:
- Clinical team at Zivot Limb Preservation Centre for dataset facilitation
- University Health Network for computational resources
- Reviewers and collaborators for valuable feedback

## Related Publications

1. Basiri, R., et al. (2025). "Accessible healing phase classification of diabetic foot ulcer" *[Computers in Biology and Medicine]*

2. Basiri, R., et al. (2024). "Protocol for metadata and image collection at diabetic foot ulcer clinics: enabling research in wound analytics and deep learning." *[BioMedical Engineering OnLine]*

## Funding

No external funding was received for this research.

## Disclaimer

This software is provided for research purposes only and is not intended for clinical use without appropriate validation, regulatory approval, and clinical oversight. Always consult qualified healthcare professionals for medical decisions.

---

**Last Updated:** December 2024  
**Version:** 1.0.0  
**Status:** Under peer review
