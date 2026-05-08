"""
Metrics module for semantic segmentation evaluation.
Implements IoU, Dice, Accuracy, Depth MAE, and position-based metrics.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import config


def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = config.NUM_CLASSES,
    ignore_index: int = -100,
) -> float:
    """
    Calculate Intersection over Union (IoU) metric.
    
    Args:
        predictions: Predicted class indices (B, H, W)
        targets: Ground truth class indices (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore in calculation
        
    Returns:
        Mean IoU across all classes
    """
    # Ensure tensors are on CPU and detached
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    iou_per_class = []
    
    for class_idx in range(num_classes):
        pred_mask = predictions == class_idx
        target_mask = targets == class_idx
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            # If there are no pixels of this class, IoU is undefined (set to 1.0)
            iou_per_class.append(1.0 if intersection == 0 else 0.0)
        else:
            iou = intersection / union
            iou_per_class.append(iou)
    
    return np.mean(iou_per_class)


def calculate_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = config.NUM_CLASSES,
) -> float:
    """
    Calculate Dice coefficient (F1 score) metric.
    
    Args:
        predictions: Predicted class indices (B, H, W)
        targets: Ground truth class indices (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Mean Dice coefficient across all classes
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    dice_per_class = []
    
    for class_idx in range(num_classes):
        pred_mask = predictions == class_idx
        target_mask = targets == class_idx
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()
        
        denominator = pred_sum + target_sum
        
        if denominator == 0:
            dice_per_class.append(1.0 if intersection == 0 else 0.0)
        else:
            dice = 2 * intersection / denominator
            dice_per_class.append(dice)
    
    return np.mean(dice_per_class)


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate pixel-wise accuracy.
    
    Args:
        predictions: Predicted class indices (B, H, W)
        targets: Ground truth class indices (B, H, W)
        
    Returns:
        Accuracy as fraction of correct pixels
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    correct = (predictions == targets).sum()
    total = targets.size
    
    return correct / total if total > 0 else 0.0


def calculate_depth_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate Mean Absolute Error for depth estimation.
    
    Args:
        predictions: Predicted depth maps (B, 1, H, W) or (B, H, W)
        targets: Ground truth depth maps (B, H, W)
        
    Returns:
        Mean Absolute Error
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # Ensure same shape
    if len(predictions.shape) == 4 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(1)
    
    mae = np.abs(predictions - targets).mean()
    return float(mae)


def calculate_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = config.NUM_CLASSES,
) -> np.ndarray:
    """
    Calculate confusion matrix for multi-class classification.
    
    Args:
        predictions: Predicted class indices (B, H, W)
        targets: Ground truth class indices (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    predictions = predictions.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for pred, target in zip(predictions, targets):
        confusion_matrix[target, pred] += 1
    
    return confusion_matrix


def calculate_metrics_by_position(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = config.NUM_CLASSES,
    position_bins: Dict[str, Tuple[float, float]] = config.POSITION_BINS,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate segmentation metrics grouped by object position in image.
    
    Objects lower in the image (higher y-coordinate) are considered "close",
    those in the middle are "middle", and those at the top are "distant".
    
    Args:
        predictions: Predicted class indices (B, H, W)
        targets: Ground truth class indices (B, H, W)
        num_classes: Number of classes
        position_bins: Dictionary mapping position names to (min_y, max_y) fractions
        
    Returns:
        Dictionary of metrics for each position bin
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    height = targets.shape[-2]
    position_metrics = {}
    
    for position_name, (min_frac, max_frac) in position_bins.items():
        min_y = int(min_frac * height)
        max_y = int(max_frac * height)
        
        # Extract region
        pred_region = predictions[:, min_y:max_y, :]
        target_region = targets[:, min_y:max_y, :]
        
        if target_region.size == 0:
            continue
        
        # Calculate metrics for this region
        iou = calculate_iou(
            torch.from_numpy(pred_region),
            torch.from_numpy(target_region),
            num_classes
        )
        
        dice = calculate_dice(
            torch.from_numpy(pred_region),
            torch.from_numpy(target_region),
            num_classes
        )
        
        accuracy = calculate_accuracy(
            torch.from_numpy(pred_region),
            torch.from_numpy(target_region)
        )
        
        position_metrics[position_name] = {
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'pixel_count': target_region.size,
        }
    
    return position_metrics


def calculate_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = config.NUM_CLASSES,
    class_names: Dict[int, str] = config.CLASS_NAMES,
) -> Dict[str, float]:
    """
    Calculate per-class IoU scores.
    
    Args:
        predictions: Predicted class indices (B, H, W)
        targets: Ground truth class indices (B, H, W)
        num_classes: Number of classes
        class_names: Dictionary mapping class indices to names
        
    Returns:
        Dictionary of per-class IoU scores
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    class_iou = {}
    
    for class_idx in range(num_classes):
        pred_mask = predictions == class_idx
        target_mask = targets == class_idx
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        class_name = class_names.get(class_idx, f"class_{class_idx}")
        class_iou[class_name] = iou
    
    return class_iou
