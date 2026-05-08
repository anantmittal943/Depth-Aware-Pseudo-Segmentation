"""
Visualization utilities for predictions and analysis.
Generate plots and visualizations for model results.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from PIL import Image

import config


def setup_logging() -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class ColorMap:
    """Generate consistent color maps for class visualization."""
    
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        """Initialize color map."""
        self.num_classes = num_classes
        self.cmap = plt.cm.get_cmap('tab20', num_classes)
    
    def get_color(self, class_id: int) -> tuple:
        """Get RGBA color for class."""
        return self.cmap(class_id / (self.num_classes - 1))
    
    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert class indices to RGB image."""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id in range(self.num_classes):
            color = self.cmap(class_id / (self.num_classes - 1))
            rgb = (np.array(color[:3]) * 255).astype(np.uint8)
            colored[mask == class_id] = rgb
        return colored


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    ground_truth: Optional[torch.Tensor] = None,
    depth_pred: Optional[torch.Tensor] = None,
    depth_true: Optional[torch.Tensor] = None,
    class_names: Dict[int, str] = config.CLASS_NAMES,
    save_path: Optional[Path] = None,
) -> None:
    """
    Create comprehensive visualization of predictions.
    
    Args:
        images: Input images (B, 3, H, W)
        predictions: Predicted masks (B, H, W)
        ground_truth: Ground truth masks (B, H, W)
        depth_pred: Predicted depth (B, H, W) or (B, 1, H, W)
        depth_true: Ground truth depth (B, H, W)
        class_names: Class name mapping
        save_path: Path to save visualization
    """
    logger = setup_logging()
    
    # Convert to numpy
    images_np = images.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    # Handle batch
    batch_size = images_np.shape[0]
    
    # Create figure
    num_cols = 3
    if ground_truth is not None:
        num_cols += 1
    if depth_pred is not None:
        num_cols += 1
    if depth_true is not None:
        num_cols += 1
    
    fig, axes = plt.subplots(
        batch_size,
        num_cols,
        figsize=(4*num_cols, 4*batch_size)
    )
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    color_mapper = ColorMap(len(class_names))
    
    for idx in range(batch_size):
        col = 0
        
        # Original image
        img = images_np[idx].transpose(1, 2, 0)
        if img.max() > 1.0:
            img = img / 255.0
        axes[idx, col].imshow(img)
        axes[idx, col].set_title('Image')
        axes[idx, col].axis('off')
        col += 1
        
        # Predicted mask
        pred_colored = color_mapper.colorize_mask(predictions_np[idx])
        axes[idx, col].imshow(pred_colored)
        axes[idx, col].set_title('Predicted')
        axes[idx, col].axis('off')
        col += 1
        
        # Ground truth mask
        if ground_truth is not None:
            gt_np = ground_truth.cpu().numpy()[idx]
            gt_colored = color_mapper.colorize_mask(gt_np)
            axes[idx, col].imshow(gt_colored)
            axes[idx, col].set_title('Ground Truth')
            axes[idx, col].axis('off')
            col += 1
        
        # Predicted depth
        if depth_pred is not None:
            depth_p = depth_pred.cpu().numpy()[idx]
            if depth_p.ndim == 3:
                depth_p = depth_p[0]
            axes[idx, col].imshow(depth_p, cmap='jet')
            axes[idx, col].set_title('Predicted Depth')
            axes[idx, col].axis('off')
            col += 1
        
        # Ground truth depth
        if depth_true is not None:
            depth_t = depth_true.cpu().numpy()[idx]
            axes[idx, col].imshow(depth_t, cmap='jet')
            axes[idx, col].set_title('Ground Truth Depth')
            axes[idx, col].axis('off')
            col += 1
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_position_analysis(
    position_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot analysis of metrics by object position.
    
    Args:
        position_metrics: Metrics grouped by position
        save_path: Path to save plot
    """
    logger = setup_logging()
    
    positions = list(position_metrics.keys())
    metrics_to_plot = ['iou', 'dice', 'accuracy']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 4))
    
    for ax_idx, metric in enumerate(metrics_to_plot):
        values = [position_metrics[pos][metric] for pos in positions]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        
        bars = axes[ax_idx].bar(positions, values, color=colors)
        axes[ax_idx].set_ylim([0, 1.0])
        axes[ax_idx].set_ylabel(metric.upper())
        axes[ax_idx].set_title(f'{metric.upper()} by Position')
        axes[ax_idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[ax_idx].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom',
                fontweight='bold'
            )
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Position analysis saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_class_performance(
    class_metrics: Dict[str, float],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot per-class IoU performance.
    
    Args:
        class_metrics: Per-class metrics
        save_path: Path to save plot
    """
    logger = setup_logging()
    
    classes = list(class_metrics.keys())
    values = list(class_metrics.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71' if v > 0.75 else '#f39c12' if v > 0.60 else '#e74c3c' 
              for v in values]
    
    bars = ax.barh(classes, values, color=colors)
    ax.set_xlim([0, 1.0])
    ax.set_xlabel('IoU')
    ax.set_title('Per-Class IoU Performance')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(
            width, bar.get_y() + bar.get_height()/2.,
            f'{value:.3f}',
            ha='left', va='center',
            fontweight='bold',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Class performance saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: Dict,
    metrics: list = ['loss', 'iou', 'dice'],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        metrics: List of metrics to plot
        save_path: Path to save plot
    """
    logger = setup_logging()
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax_idx, metric in enumerate(metrics):
        if 'train' in history and metric in history['train']:
            train_values = history['train'][metric]
            epochs = range(len(train_values))
            axes[ax_idx].plot(epochs, train_values, marker='o', label='Train')
        
        if 'val' in history and metric in history['val']:
            val_values = history['val'][metric]
            epochs = range(len(val_values))
            axes[ax_idx].plot(epochs, val_values, marker='s', label='Val')
        
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel(metric.upper())
        axes[ax_idx].set_title(f'{metric.upper()} over Time')
        axes[ax_idx].legend()
        axes[ax_idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
        plt.close()
    else:
        plt.show()
