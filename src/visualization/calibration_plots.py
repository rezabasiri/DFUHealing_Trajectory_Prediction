"""
Calibration visualization and analysis helpers.

This module provides functions for generating calibration analysis outputs
including figures (calibration curves, probability distributions, confusion matrix)
and CSV files (metrics, curve data).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss, log_loss, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Class names for transition labels
CLASS_NAMES = ['Unfavorable', 'Acceptable', 'Favorable']
CLASS_COLORS = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green


def calculate_per_class_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> List[Dict]:
    """
    Calculate calibration metrics for each class.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0, 1, 2 for Unfavorable, Acceptable, Favorable)
    y_pred_proba : np.ndarray
        Predicted probabilities (n_samples, 3)
    n_bins : int
        Number of bins for calibration curve

    Returns
    -------
    List[Dict]
        List of metrics dictionaries for each class
    """
    n_classes = 3
    calibration_metrics = []

    for class_idx in range(n_classes):
        y_binary = (y_true == class_idx).astype(int)
        y_prob = y_pred_proba[:, class_idx] if y_pred_proba.shape[1] > class_idx else np.zeros(len(y_true))

        # Brier score
        brier = brier_score_loss(y_binary, y_prob)

        # Calibration curve for ECE/MCE
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob, n_bins=n_bins, strategy='uniform'
            )
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        except ValueError:
            ece = np.nan
            mce = np.nan

        calibration_metrics.append({
            'Class': CLASS_NAMES[class_idx],
            'Brier_Score': brier,
            'ECE': ece,
            'MCE': mce,
            'Mean_Prob': y_prob.mean(),
            'Std_Prob': y_prob.std(),
            'Support': int(y_binary.sum())
        })

    return calibration_metrics


def save_calibration_metrics_csv(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    additional_metrics: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Save calibration metrics to CSV file.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    output_path : Path
        Path to save CSV file
    additional_metrics : Dict, optional
        Additional metrics to include

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics
    """
    metrics = calculate_per_class_calibration_metrics(y_true, y_pred_proba)

    # Add overall metrics
    try:
        ll = log_loss(y_true, y_pred_proba, labels=[0, 1, 2])
    except ValueError:
        ll = np.nan

    metrics_df = pd.DataFrame(metrics)

    # Add summary row
    summary = {
        'Class': 'Mean',
        'Brier_Score': metrics_df['Brier_Score'].mean(),
        'ECE': metrics_df['ECE'].mean(),
        'MCE': metrics_df['MCE'].mean(),
        'Mean_Prob': np.nan,
        'Std_Prob': np.nan,
        'Support': int(len(y_true))
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([summary])], ignore_index=True)

    # Add log loss as a separate column in the Mean row
    metrics_df['Log_Loss'] = np.nan
    metrics_df.loc[metrics_df['Class'] == 'Mean', 'Log_Loss'] = ll

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return metrics_df


def save_calibration_curve_data_csv(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Save per-bin calibration curve data to CSV.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    output_path : Path
        Path to save CSV file
    n_bins : int
        Number of bins

    Returns
    -------
    pd.DataFrame
        DataFrame with curve data
    """
    curve_data_rows = []

    for class_idx in range(3):
        y_binary = (y_true == class_idx).astype(int)
        y_prob = y_pred_proba[:, class_idx] if y_pred_proba.shape[1] > class_idx else np.zeros(len(y_true))

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob, n_bins=n_bins, strategy='uniform'
            )
            for bin_idx, (mean_pred, frac_pos) in enumerate(zip(mean_predicted_value, fraction_of_positives)):
                curve_data_rows.append({
                    'Class': CLASS_NAMES[class_idx],
                    'Bin': bin_idx + 1,
                    'Mean_Predicted_Probability': mean_pred,
                    'Fraction_of_Positives': frac_pos,
                    'Calibration_Error': frac_pos - mean_pred
                })
        except ValueError:
            pass

    curve_df = pd.DataFrame(curve_data_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return curve_df


def plot_calibration_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (18, 5)
) -> None:
    """
    Create 3-panel calibration curves (one per class).

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    output_path : Path
        Path to save figure (without extension)
    n_bins : int
        Number of bins
    figsize : tuple
        Figure size
    """
    metrics = calculate_per_class_calibration_metrics(y_true, y_pred_proba, n_bins)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for class_idx, ax in enumerate(axes):
        y_binary = (y_true == class_idx).astype(int)
        y_prob = y_pred_proba[:, class_idx] if y_pred_proba.shape[1] > class_idx else np.zeros(len(y_true))

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob, n_bins=n_bins, strategy='uniform'
            )

            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
            ax.plot(
                mean_predicted_value, fraction_of_positives,
                's-', linewidth=2, markersize=8, color=CLASS_COLORS[class_idx],
                label=f'{CLASS_NAMES[class_idx]} (Brier: {metrics[class_idx]["Brier_Score"]:.3f})'
            )
        except ValueError:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

        ax.set_xlabel('Mean Predicted Probability', fontsize=16)
        ax.set_ylabel('Fraction of Positives', fontsize=16)
        ax.set_title(f'{CLASS_NAMES[class_idx]} Transitions', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path) + '.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}.png")
    plt.close()


def plot_combined_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    title: str = "Calibration Curves",
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create combined calibration plot (all classes on one figure).

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    output_path : Path
        Path to save figure (without extension)
    title : str
        Plot title
    n_bins : int
        Number of bins
    figsize : tuple
        Figure size
    """
    metrics = calculate_per_class_calibration_metrics(y_true, y_pred_proba, n_bins)

    plt.figure(figsize=figsize)

    for class_idx in range(3):
        y_binary = (y_true == class_idx).astype(int)
        y_prob = y_pred_proba[:, class_idx] if y_pred_proba.shape[1] > class_idx else np.zeros(len(y_true))

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob, n_bins=n_bins, strategy='uniform'
            )
            plt.plot(
                mean_predicted_value, fraction_of_positives,
                's-', linewidth=2.5, markersize=10, color=CLASS_COLORS[class_idx],
                label=f'{CLASS_NAMES[class_idx]} (Brier: {metrics[class_idx]["Brier_Score"]:.3f})'
            )
        except ValueError:
            pass

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2.5, label='Perfect Calibration', alpha=0.7)
    plt.xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path) + '.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}.png")
    plt.close()


def plot_probability_distributions(
    y_pred_proba: np.ndarray,
    output_path: Path,
    figsize: Tuple[int, int] = (18, 5)
) -> None:
    """
    Create 3-panel probability distribution histograms.

    Parameters
    ----------
    y_pred_proba : np.ndarray
        Predicted probabilities
    output_path : Path
        Path to save figure (without extension)
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for class_idx, ax in enumerate(axes):
        y_prob = y_pred_proba[:, class_idx] if y_pred_proba.shape[1] > class_idx else np.zeros(len(y_pred_proba))
        ax.hist(y_prob, bins=30, alpha=0.7, color=CLASS_COLORS[class_idx], edgecolor='black')
        ax.axvline(y_prob.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y_prob.mean():.3f}')
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{CLASS_NAMES[class_idx]} Probability Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path) + '.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}.png")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    output_path : Path
        Path to save figure (without extension)
    normalize : bool
        Whether to normalize the confusion matrix
    figsize : tuple
        Figure size
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=figsize)

    # Create annotation text with both count and percentage
    if normalize:
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'
    else:
        annot = cm

    sns.heatmap(
        cm_display,
        annot=annot if normalize else True,
        fmt='' if normalize else 'd',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path) + '.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}.png")
    plt.close()


# Feature category mapping for selected features (6 categories)
FEATURE_CATEGORIES = {
    # Treatment Response features
    'Previous_Phase_X_Treatment': 'Treatment Response',
    'Phase-Adjusted Treatment Effect': 'Treatment Response',
    'Treatment_Intensity_Score': 'Treatment Response',
    'Healing_Momentum': 'Treatment Response',
    'Phase_Improvements_Count': 'Treatment Response',

    # Historical Pattern features
    'History_Acceptable_Transitions': 'Historical Pattern',
    'History_Phase_1_Count': 'Historical Pattern',
    'History_Phase_1_Proportion': 'Historical Pattern',
    'History_Phase_0_Proportion': 'Historical Pattern',
    'History_Phase_0_Count': 'Historical Pattern',
    'History_Favorable_Transitions': 'Historical Pattern',
    'History_Length': 'Historical Pattern',
    'History_Completeness': 'Historical Pattern',

    # Patient Phenotype features
    'Patient_Cluster_Slow_Healer': 'Patient Phenotype',
    'Patient_Cluster_Fast_Healer': 'Patient Phenotype',

    # Temporal features
    'Appointments_So_Far': 'Temporal',
    'Appt Days': 'Temporal',
    'Days_To_Next_Appt': 'Temporal',
    'Cumulative_Phase_Duration': 'Temporal',
    'Onset (Days)': 'Temporal',
    'Std_Days_Between_Appts': 'Temporal',

    # Wound Assessment features
    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': 'Wound Assessment',
    'Exudate Amount (None:0,Minor,Medium,Severe:3)': 'Wound Assessment',
    'Exudate Amount (None:0,Minor,Medium,Severe:3)_Consistency': 'Wound Assessment',
    'Exudate Amount (None:0,Minor,Medium,Severe:3)_History_Mean': 'Wound Assessment',
    'Wound Score': 'Wound Assessment',

    # Temperature features
    'Peri-Ulcer Temperature (°C)': 'Temperature',
    'Intact Skin Temperature (°C)': 'Temperature',
    'Peri-Ulcer Temperature Normalized (°C)': 'Temperature',
    'Wound Centre Temperature (°C)': 'Temperature',
    'Wound Centre Temperature Normalized (°C)': 'Temperature',
}


def categorize_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize a list of features into clinical categories.

    Parameters
    ----------
    feature_names : list of str
        List of feature names to categorize

    Returns
    -------
    dict
        Dictionary mapping category names to lists of features
    """
    categories = {
        'Treatment Response': [],
        'Historical Pattern': [],
        'Patient Phenotype': [],
        'Temporal': [],
        'Wound Assessment': [],
        'Temperature': []
    }

    for feat in feature_names:
        # Check exact match first
        if feat in FEATURE_CATEGORIES:
            cat = FEATURE_CATEGORIES[feat]
            categories[cat].append(feat)
        else:
            # Try to infer category from feature name patterns
            feat_lower = feat.lower()
            if any(kw in feat_lower for kw in ['treatment', 'intensity', 'momentum', 'improvement']):
                categories['Treatment Response'].append(feat)
            elif any(kw in feat_lower for kw in ['history', 'transition', 'completeness']):
                categories['Historical Pattern'].append(feat)
            elif any(kw in feat_lower for kw in ['cluster', 'phenotype', 'healer']):
                categories['Patient Phenotype'].append(feat)
            elif any(kw in feat_lower for kw in ['days', 'appt', 'onset', 'duration', 'time', 'appointment']):
                categories['Temporal'].append(feat)
            elif any(kw in feat_lower for kw in ['temp', 'temperature']):
                categories['Temperature'].append(feat)
            elif any(kw in feat_lower for kw in ['exudate', 'wound', 'ulcer', 'skin', 'score']):
                categories['Wound Assessment'].append(feat)
            else:
                # Default to Historical Pattern for unknown features
                categories['Historical Pattern'].append(feat)

    return categories


def plot_feature_categories(
    feature_names: List[str],
    output_path: Path,
    figsize: Tuple[int, int] = (10, 8),
) -> Dict[str, int]:
    """
    Create a pie chart showing the distribution of features by clinical category.

    Parameters
    ----------
    feature_names : list of str
        List of selected feature names
    output_path : Path
        Path to save figure (without extension)
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    dict
        Dictionary with category counts
    """
    # Categorize features
    categories = categorize_features(feature_names)

    # Count features per category
    counts = {cat: len(feats) for cat, feats in categories.items()}
    total = sum(counts.values())

    # Filter out empty categories
    counts = {k: v for k, v in counts.items() if v > 0}

    # Colors for 6 categories
    colors = {
        'Treatment Response': '#3498db',   # Blue
        'Historical Pattern': '#9b59b6',   # Purple
        'Patient Phenotype': '#1abc9c',    # Teal
        'Temporal': '#2ecc71',             # Green
        'Wound Assessment': '#e74c3c',     # Red
        'Temperature': '#f39c12',          # Orange
    }

    # Create pie chart
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(counts.keys())
    sizes = list(counts.values())
    chart_colors = [colors.get(cat, '#95a5a6') for cat in labels]

    # Create pie with percentages
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # We'll add custom legend
        autopct=lambda pct: f'{pct:.0f}%\n({int(pct/100.*total)})',
        colors=chart_colors,
        startangle=90,
        explode=[0.02] * len(sizes),  # Slight separation
        textprops={'fontsize': 16, 'fontweight': 'bold'}
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Add legend without feature counts (cleaner look)
    legend_labels = [f'{cat}' for cat in labels]
    ax.legend(
        wedges, legend_labels,
        title="Feature Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=16,
        title_fontsize=16
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path) + '.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}.png")
    plt.close()

    return counts


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 8),
) -> Dict[str, float]:
    """
    Create ROC curves for each class and the macro average.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0, 1, 2 for Unfavorable, Acceptable, Favorable)
    y_pred_proba : np.ndarray
        Predicted probabilities (n_samples, 3)
    output_path : Path
        Path to save figure (without extension)
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    dict
        Dictionary with ROC-AUC values for each class and average
    """
    n_classes = 3

    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)

    # Plot each class
    class_labels = ['Unfavorable', 'Acceptable', 'Favorable']
    line_styles = ['-', '-', '-']

    for i in range(n_classes):
        ax.plot(
            fpr[i], tpr[i],
            color=CLASS_COLORS[i],
            linewidth=2.5,
            linestyle=line_styles[i],
            label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})'
        )

    # Plot macro-average (renamed to just "Average")
    ax.plot(
        fpr["macro"], tpr["macro"],
        color='#2c3e50',  # Dark gray
        linewidth=3,
        linestyle='--',
        label=f'Average (AUC = {roc_auc["macro"]:.2f})'
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Make the plot square
    ax.set_aspect('equal')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path) + '.pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}.png")
    plt.close()

    # Return AUC values
    return {
        'roc_auc_unfavorable': roc_auc[0],
        'roc_auc_acceptable': roc_auc[1],
        'roc_auc_favorable': roc_auc[2],
        'roc_auc_average': roc_auc["macro"]
    }


def generate_calibration_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: Path,
    method_name: str = "ensemble",
    generate_pdf: bool = True
) -> Dict:
    """
    Generate complete calibration analysis with all figures and CSV files.

    Parameters
    ----------
    y_true : np.ndarray
        True transition labels (0=Unfavorable, 1=Acceptable, 2=Favorable)
    y_pred_proba : np.ndarray
        Predicted probabilities (n_samples, 3)
    output_dir : Path
        Directory to save outputs
    method_name : str
        Name for the analysis (used in titles and filenames)
    generate_pdf : bool
        Whether to also generate PDF versions of figures

    Returns
    -------
    Dict
        Dictionary with all computed metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"\n{'='*70}")
    print(f"CALIBRATION ANALYSIS: {method_name.upper()}")
    print(f"{'='*70}")

    # 1. Save calibration metrics CSV
    print("\n1. Saving calibration metrics...")
    metrics_df = save_calibration_metrics_csv(
        y_true, y_pred_proba,
        output_dir / 'calibration_metrics.csv'
    )

    # 2. Save calibration curve data CSV
    print("\n2. Saving calibration curve data...")
    curve_df = save_calibration_curve_data_csv(
        y_true, y_pred_proba,
        output_dir / 'calibration_curve_data.csv'
    )

    # 3. Create 3-panel calibration curves
    print("\n3. Creating calibration curves (3-panel)...")
    plot_calibration_curves(
        y_true, y_pred_proba,
        output_dir / 'calibration_curves'
    )

    # 4. Create combined calibration plot
    print("\n4. Creating combined calibration plot...")
    plot_combined_calibration(
        y_true, y_pred_proba,
        output_dir / 'calibration_combined',
        title=f'Calibration Curves: {method_name.replace("_", " ").title()}'
    )

    # 5. Create probability distribution histograms
    print("\n5. Creating probability distribution histograms...")
    plot_probability_distributions(
        y_pred_proba,
        output_dir / 'probability_distributions'
    )

    # 6. Create confusion matrix
    print("\n6. Creating confusion matrix...")
    plot_confusion_matrix(
        y_true, y_pred,
        output_dir / 'confusion_matrix'
    )

    # Calculate overall metrics
    try:
        ll = log_loss(y_true, y_pred_proba, labels=[0, 1, 2])
    except ValueError:
        ll = np.nan

    per_class_metrics = calculate_per_class_calibration_metrics(y_true, y_pred_proba)

    # Classification metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Print summary
    print(f"\n{'='*70}")
    print(f"CALIBRATION METRICS SUMMARY - {method_name.upper()}")
    print(f"{'='*70}")
    print(metrics_df.to_string(index=False))

    print(f"\n\nClassification Performance:")
    print(f"  F1-Macro: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: F1={f1_per_class[i]:.3f}, P={precision_per_class[i]:.3f}, R={recall_per_class[i]:.3f}")

    print(f"\nOverall Log Loss: {ll:.4f}")
    print(f"{'='*70}")

    return {
        'calibration_metrics': per_class_metrics,
        'log_loss': ll,
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_per_class': f1_per_class.tolist(),
        'mean_ece': np.nanmean([m['ECE'] for m in per_class_metrics]),
        'mean_mce': np.nanmean([m['MCE'] for m in per_class_metrics]),
        'mean_brier': np.nanmean([m['Brier_Score'] for m in per_class_metrics]),
    }
