"""
Metrics and visualization utilities for model evaluation.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import seaborn as sns


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        class_names: List of class names
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(targets, predictions)
    metrics['precision'] = precision_score(targets, predictions, average=average, zero_division=0)
    metrics['recall'] = recall_score(targets, predictions, average=average, zero_division=0)
    metrics['f1'] = f1_score(targets, predictions, average=average, zero_division=0)
    
    # Per-class metrics
    if class_names is not None and len(class_names) > 2:  # Multi-class
        precision = precision_score(targets, predictions, average=None, zero_division=0)
        recall = recall_score(targets, predictions, average=None, zero_division=0)
        f1 = f1_score(targets, predictions, average=None, zero_division=0)
        
        for i, name in enumerate(class_names):
            metrics[f'precision_{name}'] = precision[i]
            metrics[f'recall_{name}'] = recall[i]
            metrics[f'f1_{name}'] = f1[i]
    
    # Classification report
    if class_names is not None:
        report = classification_report(
            targets, predictions, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Flatten the report
        for k1, v1 in report.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    metrics[f'report_{k1}_{k2}'] = v2
            else:
                metrics[f'report_{k1}'] = v1
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        cmap: Color map
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Add labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_scores: Predicted probabilities for each class
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    n_classes = len(class_names)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot micro-average ROC curve
    ax.plot(
        fpr["micro"], tpr["micro"],
        label=f'micro-average (AUC = {roc_auc["micro"]:0.2f})',
        color='deeppink',
        linestyle=':',
        linewidth=4
    )
    
    # Plot ROC curve for each class
    colors = sns.color_palette('hsv', n_classes)
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i], tpr[i],
            color=color,
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})'
        )
    
    # Plot random guess line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot precision-recall curves for multi-class classification.
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_scores: Predicted probabilities for each class
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute precision-recall curve and average precision for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    n_classes = len(class_names)
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_scores[:, i])
    
    # Compute micro-average precision-recall curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true.ravel(), y_scores.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true, y_scores, average="micro"
    )
    
    # Plot all precision-recall curves
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot micro-average precision-recall curve
    ax.plot(
        recall["micro"], precision["micro"],
        label=f'micro-average (AP = {average_precision["micro"]:0.2f})',
        color='deeppink',
        linestyle=':',
        linewidth=4
    )
    
    # Plot precision-recall curve for each class
    colors = sns.color_palette('hsv', n_classes)
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            recall[i], precision[i],
            color=color,
            lw=2,
            label=f'{class_names[i]} (AP = {average_precision[i]:0.2f})'
        )
    
    # Set plot properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='best')
    
    return fig


def plot_training_curves(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    metric_name: str = 'loss',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        metric_name: Name of the metric to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get metric values
    train_values = train_metrics.get(metric_name, [])
    val_values = val_metrics.get(metric_name, [])
    
    # Plot curves
    epochs = range(1, len(train_values) + 1)
    ax.plot(epochs, train_values, 'b-', label=f'Training {metric_name}')
    
    if val_values:
        ax.plot(epochs, val_values, 'r-', label=f'Validation {metric_name}')
    
    # Set plot properties
    ax.set_title(f'Training and Validation {metric_name.capitalize()}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name.capitalize())
    ax.legend()
    ax.grid(True)
    
    return fig


def save_metrics(
    metrics: Dict[str, Any],
    filepath: Union[str, Path]
) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the metrics file
    """
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)): 
            return None
        return obj
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save metrics to file
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2, default=convert_numpy)


def load_metrics(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        filepath: Path to the metrics file
        
    Returns:
        Dictionary of metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)
