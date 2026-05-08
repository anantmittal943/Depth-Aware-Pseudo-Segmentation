"""
Test/Inference script for Depth-Aware Pseudo-Segmentation model.
Measures inference speed, calculates metrics, and generates visualizations.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

import config
from dataset import DepthAwareSegmentationDataset
from model import create_model
from metrics import (
    calculate_iou,
    calculate_dice,
    calculate_accuracy,
    calculate_depth_mae,
    calculate_confusion_matrix,
    calculate_metrics_by_position,
    calculate_class_metrics,
)


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(
        log_dir / f"test_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# INFERENCE
# ============================================================================
class Inferencer:
    """Handles model inference with timing and metric calculation."""
    
    def __init__(
        self,
        model_path: Path,
        device: str = config.DEVICE,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize inferencer.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to use (cuda or cpu)
            logger: Logger instance
        """
        self.model_path = model_path
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Load model
        self.model = create_model(
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            freeze_backbone=False,
        )
        
        if not self.model_path.exists():
            self.logger.warning(f"Model path not found: {self.model_path}")
        else:
            checkpoint = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded model from {self.model_path}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Timing statistics
        self.inference_times = []
    
    def infer_single(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image tensor (3, H, W) or (1, 3, H, W)
            
        Returns:
            Tuple of (class_logits, depth, inference_time_ms)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Measure inference time using perf_counter for precision
        start_time = time.perf_counter()
        
        with torch.no_grad():
            class_logits, depth = self.model(image)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        self.inference_times.append(elapsed_time)
        
        return class_logits, depth, elapsed_time
    
    def infer_batch(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run inference on a batch of images.
        
        Args:
            images: Batch of input images (B, 3, H, W)
            
        Returns:
            Tuple of (class_logits, depth, total_inference_time_ms)
        """
        images = images.to(self.device)
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            class_logits, depth = self.model(images)
        
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Record per-image time
        batch_size = images.shape[0]
        per_image_time = elapsed_time / batch_size
        self.inference_times.extend([per_image_time] * batch_size)
        
        return class_logits, depth, elapsed_time
    
    def get_timing_stats(self) -> Dict[str, float]:
        """
        Get inference timing statistics.
        
        Returns:
            Dictionary with timing stats (mean, min, max, std)
        """
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        # Check against target
        max_time = config.MAX_INFERENCE_TIME_MS
        exceeded = (times > max_time).sum()
        
        return {
            'mean_ms': float(np.mean(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'std_ms': float(np.std(times)),
            'total_inferences': len(times),
            'exceeded_target': int(exceeded),
            'target_ms': max_time,
        }


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    depth: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    class_names: Dict[int, str] = config.CLASS_NAMES,
) -> np.ndarray:
    """
    Create visualization of prediction results.
    
    Args:
        image: Original image (H, W, 3)
        prediction: Predicted class map (H, W)
        depth: Predicted depth map (H, W)
        ground_truth: Optional ground truth mask (H, W)
        class_names: Dictionary of class names
        
    Returns:
        Visualization image
    """
    # Create color map for classes
    num_classes = len(class_names)
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    # Normalize image
    if image.max() > 1.0:
        image = image / 255.0
    
    # Create figure with subplots
    num_plots = 4 if ground_truth is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
    
    if num_plots == 3:
        axes = [axes] if not hasattr(axes, '__len__') else axes
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted segmentation
    pred_colored = cmap(prediction / (num_classes - 1))[:, :, :3]
    axes[1].imshow(pred_colored)
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')
    
    # Depth map
    axes[2].imshow(depth, cmap='jet')
    axes[2].set_title('Depth Estimation')
    axes[2].colorbar(axes[2].images[0])
    axes[2].axis('off')
    
    # Ground truth if available
    if ground_truth is not None:
        gt_colored = cmap(ground_truth / (num_classes - 1))[:, :, :3]
        axes[3].imshow(gt_colored)
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return image_array


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Dict[int, str] = config.CLASS_NAMES,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: Dictionary of class names
        save_path: Optional path to save the plot
    """
    class_list = [class_names.get(i, f"class_{i}") for i in range(len(confusion_matrix))]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_list,
        yticklabels=class_list,
        ax=ax,
        cbar_kws={'label': 'Count'},
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def test_dataset(
    test_dataset: DepthAwareSegmentationDataset,
    inferencer: Inferencer,
    device: str = config.DEVICE,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Run inference on entire test dataset.
    
    Args:
        test_dataset: Test dataset instance
        inferencer: Inferencer instance
        device: Device to use
        logger: Logger instance
        
    Returns:
        Dictionary of aggregated metrics
    """
    logger = logger or logging.getLogger(__name__)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Aggregate metrics
    all_metrics = {
        'iou': 0.0,
        'dice': 0.0,
        'accuracy': 0.0,
        'depth_mae': 0.0,
    }
    position_metrics = {pos: {metric: [] for metric in ['iou', 'dice', 'accuracy']}
                       for pos in config.POSITION_BINS.keys()}
    all_preds = []
    all_targets = []
    all_depths_pred = []
    all_depths_true = []
    
    logger.info(f"Running inference on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            depths = batch['depth'].to(device)
            
            # Inference
            class_logits, depth_pred = inferencer.infer_batch(images)
            
            # Get predictions
            preds = torch.argmax(class_logits, dim=1)
            
            # Calculate metrics
            iou = calculate_iou(preds, masks, config.NUM_CLASSES)
            dice = calculate_dice(preds, masks, config.NUM_CLASSES)
            accuracy = calculate_accuracy(preds, masks)
            depth_mae = calculate_depth_mae(depth_pred, depths)
            
            all_metrics['iou'] += iou
            all_metrics['dice'] += dice
            all_metrics['accuracy'] += accuracy
            all_metrics['depth_mae'] += depth_mae
            
            # Position-based metrics
            if config.EVALUATE_BY_POSITION:
                pos_metrics = calculate_metrics_by_position(preds, masks)
                for pos_name, metrics in pos_metrics.items():
                    position_metrics[pos_name]['iou'].append(metrics['iou'])
                    position_metrics[pos_name]['dice'].append(metrics['dice'])
                    position_metrics[pos_name]['accuracy'].append(metrics['accuracy'])
            
            # Store predictions for confusion matrix
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            all_depths_pred.append(depth_pred.cpu().numpy())
            all_depths_true.append(depths.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * config.TEST_BATCH_SIZE} samples")
    
    # Average metrics
    num_batches = len(test_loader)
    for key in all_metrics:
        all_metrics[key] /= num_batches
    
    # Average position metrics
    position_results = {}
    for pos_name, metrics in position_metrics.items():
        position_results[pos_name] = {
            k: np.mean(v) if v else 0.0 for k, v in metrics.items()
        }
    
    # Get inference timing
    timing_stats = inferencer.get_timing_stats()
    
    # Confusion matrix
    confusion_matrix = calculate_confusion_matrix(
        torch.from_numpy(np.array(all_preds).reshape(-1)),
        torch.from_numpy(np.array(all_targets).reshape(-1)),
        config.NUM_CLASSES
    )
    
    results = {
        'global_metrics': all_metrics,
        'position_metrics': position_results,
        'timing': timing_stats,
        'confusion_matrix': confusion_matrix,
    }
    
    return results


def print_results(results: Dict, logger: logging.Logger) -> None:
    """
    Print evaluation results in formatted manner.
    
    Args:
        results: Dictionary of evaluation results
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("GLOBAL METRICS")
    logger.info("=" * 80)
    
    for metric, value in results['global_metrics'].items():
        logger.info(f"{metric.upper():15s}: {value:.4f}")
    
    if config.EVALUATE_BY_POSITION:
        logger.info("\n" + "=" * 80)
        logger.info("POSITION-BASED METRICS")
        logger.info("=" * 80)
        
        for position, metrics in results['position_metrics'].items():
            logger.info(f"\n{position.upper()}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric.upper():15s}: {value:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE TIMING")
    logger.info("=" * 80)
    
    for key, value in results['timing'].items():
        if isinstance(value, float):
            logger.info(f"{key:20s}: {value:.4f} ms")
        else:
            logger.info(f"{key:20s}: {value}")
    
    # Check if inference speed meets target
    mean_time = results['timing'].get('mean_ms', float('inf'))
    target_time = config.MAX_INFERENCE_TIME_MS
    
    if mean_time <= target_time:
        logger.info(
            f"✓ Inference speed target MET: "
            f"{mean_time:.2f}ms <= {target_time}ms"
        )
    else:
        logger.warning(
            f"✗ Inference speed target NOT MET: "
            f"{mean_time:.2f}ms > {target_time}ms"
        )


# ============================================================================
# MAIN
# ============================================================================
def main(args):
    """Main test function."""
    # Setup
    config.set_seed(config.SEED)
    logger = setup_logging(config.LOGS_DIR)
    
    logger.info("Starting inference and evaluation...")
    
    # Create test dataset
    test_dataset = DepthAwareSegmentationDataset(
        image_dir=Path(args.test_images),
        mask_dir=Path(args.test_masks),
        split="test",
        augment=False,
    )
    
    # Create inferencer
    inferencer = Inferencer(
        model_path=Path(args.model_path),
        device=config.DEVICE,
        logger=logger,
    )
    
    # Run inference
    results = test_dataset(
        test_dataset=test_dataset,
        inferencer=inferencer,
        device=config.DEVICE,
        logger=logger,
    )
    
    # Print results
    print_results(results, logger)
    
    # Save confusion matrix
    if config.SAVE_VISUALIZATION:
        confusion_matrix_path = config.VISUALIZATION_OUTPUT_DIR / "confusion_matrix.png"
        plot_confusion_matrix(
            results['confusion_matrix'],
            class_names=config.CLASS_NAMES,
            save_path=confusion_matrix_path,
        )
        logger.info(f"Confusion matrix saved to {confusion_matrix_path}")
    
    logger.info("Testing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Depth-Aware Pseudo-Segmentation model"
    )
    
    parser.add_argument(
        "--test-images",
        type=str,
        default="data/test/images",
        help="Path to test images directory"
    )
    parser.add_argument(
        "--test-masks",
        type=str,
        default="data/test/masks",
        help="Path to test masks directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pth",
        help="Path to saved model checkpoint"
    )
    
    args = parser.parse_args()
    main(args)
