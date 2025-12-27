"""
SHAP visualization helpers for DFU Healing prediction.
Generates manuscript-quality SHAP figures for model interpretability.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import feature display names from constants
from src.config.constants import FEATURE_DISPLAY_NAMES

# Alias for backward compatibility
FEATURE_NAME_MAPPING = FEATURE_DISPLAY_NAMES


def get_readable_feature_name(technical_name: str) -> str:
    """
    Convert technical feature name to reader-friendly clinical name.

    Parameters
    ----------
    technical_name : str
        The technical/code feature name

    Returns
    -------
    str
        Reader-friendly name for medical professionals
    """
    # Check direct mapping
    if technical_name in FEATURE_NAME_MAPPING:
        return FEATURE_NAME_MAPPING[technical_name]

    # Handle pattern-based conversions
    name = technical_name

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Handle common patterns
    if 'Phase' in name and 'Proportion' in name:
        # e.g., "History Phase 0 Proportion" -> "Historical Phase 0 (%)"
        name = name.replace('Proportion', '(%)')
        if 'History' in name:
            name = name.replace('History', 'Historical')

    # Capitalize appropriately
    words = name.split()
    capitalized = []
    for word in words:
        if word.lower() in ['of', 'in', 'to', 'the', 'and', 'or']:
            capitalized.append(word.lower())
        else:
            capitalized.append(word.capitalize())

    return ' '.join(capitalized)


def convert_feature_names(feature_names: List[str]) -> List[str]:
    """
    Convert list of technical feature names to reader-friendly names.

    Parameters
    ----------
    feature_names : list
        List of technical feature names

    Returns
    -------
    list
        List of reader-friendly names
    """
    return [get_readable_feature_name(name) for name in feature_names]


def generate_shap_figures(
    model,
    X_data: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    class_names: List[str] = ['Unfavorable', 'Acceptable', 'Favorable'],
    max_samples: int = 200
) -> Dict[str, str]:
    """
    Generate SHAP analysis figures for model interpretability.
    Style matches generate_shap_analysis.py for manuscript quality.

    Parameters
    ----------
    model : fitted model
        The trained model (base estimator from CalibratedClassifierCV)
    X_data : np.ndarray
        Feature data for SHAP analysis
    feature_names : list
        Names of features
    output_dir : Path
        Directory to save figures
    class_names : list
        Names of classes for labeling
    max_samples : int
        Maximum samples to use for SHAP (for speed)

    Returns
    -------
    dict with paths to generated figures
    """
    if not SHAP_AVAILABLE:
        print("  SHAP not available, skipping SHAP figures")
        return {}

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style to match generate_shap_analysis.py
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subsample if needed
    np.random.seed(42)
    if len(X_data) > max_samples:
        indices = np.random.choice(len(X_data), max_samples, replace=False)
        X_sample = X_data[indices]
    else:
        X_sample = X_data

    print(f"  Computing SHAP values for {len(X_sample)} samples...")

    # Convert feature names to reader-friendly format
    readable_names = convert_feature_names(feature_names)

    # Get the base estimator from CalibratedClassifierCV
    if hasattr(model, 'calibrated_classifiers_'):
        # CalibratedClassifierCV - get base estimator
        base_model = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, 'estimators_'):
        # Direct ensemble model
        base_model = model
    else:
        base_model = model

    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"  SHAP TreeExplainer failed: {e}")
        return {}

    # Handle multi-class output
    if isinstance(shap_values, list):
        # Multi-class: shap_values is a list of arrays, one per class
        n_classes = len(shap_values)
        print(f"  Multi-class model detected: {n_classes} classes")
        shap_values_for_plots = shap_values[0]  # Use first class (Unfavorable)
    elif len(shap_values.shape) == 3:
        # TreeExplainer with multi-class returns 3D array (samples, features, classes)
        n_classes = shap_values.shape[2]
        print(f"  Multi-class model detected: {n_classes} classes (3D array)")
        shap_values = [shap_values[:, :, i] for i in range(n_classes)]
        shap_values_for_plots = shap_values[0]
    else:
        shap_values = [shap_values]
        shap_values_for_plots = shap_values[0]

    figures = {}

    # ========================================================================
    # 1. SHAP Summary Plot (Global Feature Importance) - Dot plot
    # ========================================================================
    print("  Creating SHAP summary plot (global importance)...")
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_for_plots,
            X_sample,
            feature_names=readable_names,
            plot_type="dot",
            show=False,
            max_display=20
        )
        plt.tight_layout()

        # Save PNG and PDF
        fig_path_png = output_dir / 'shap_summary_global.png'
        fig_path_pdf = output_dir / 'shap_summary_global.pdf'
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        plt.close()
        figures['summary_global_png'] = str(fig_path_png)
        figures['summary_global_pdf'] = str(fig_path_pdf)
        print(f"  Saved: {fig_path_png}")
        print(f"  Saved: {fig_path_pdf}")
    except Exception as e:
        print(f"  Failed to create summary plot: {e}")

    # ========================================================================
    # 2. SHAP Bar Plot (Mean Absolute SHAP Values)
    # ========================================================================
    print("  Creating SHAP bar plot (mean importance)...")
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_for_plots,
            X_sample,
            feature_names=readable_names,
            plot_type="bar",
            show=False,
            max_display=20
        )
        plt.title('Mean Absolute SHAP Values (Feature Importance)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        # Save PNG and PDF
        fig_path_png = output_dir / 'shap_bar_plot.png'
        fig_path_pdf = output_dir / 'shap_bar_plot.pdf'
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        plt.close()
        figures['bar_plot_png'] = str(fig_path_png)
        figures['bar_plot_pdf'] = str(fig_path_pdf)
        print(f"  Saved: {fig_path_png}")
        print(f"  Saved: {fig_path_pdf}")
    except Exception as e:
        print(f"  Failed to create bar plot: {e}")

    # ========================================================================
    # 3. Feature Importance CSV
    # ========================================================================
    print("  Creating feature importance table...")
    try:
        mean_abs_shap = np.abs(shap_values_for_plots).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': readable_names,
            'Mean_Abs_SHAP': mean_abs_shap,
        }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
        feature_importance['Rank'] = range(1, len(feature_importance) + 1)

        # Save top 20
        csv_path = output_dir / 'shap_feature_importance_top20.csv'
        feature_importance.head(20).to_csv(csv_path, index=False)
        figures['importance_csv'] = str(csv_path)
        print(f"  Saved: {csv_path}")

        # Print top 10
        print("\n  TOP 10 MOST IMPORTANT FEATURES:")
        print("  " + "-" * 50)
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['Rank']:2d}. {row['Feature']:<40} {row['Mean_Abs_SHAP']:.4f}")
    except Exception as e:
        print(f"  Failed to create importance table: {e}")

    return figures
