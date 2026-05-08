"""
Training script for Depth-Aware Pseudo-Segmentation model.
Implements complete training loop with logging, checkpointing, and validation.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import config
from dataset import create_dataloaders
from model import create_model
from metrics import calculate_iou, calculate_dice, calculate_accuracy, calculate_depth_mae


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(
        log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(
        self,
        alpha: float = config.FOCAL_ALPHA,
        gamma: float = config.FOCAL_GAMMA,
        ignore_index: int = -100,
    ) -> None:
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in range (0,1) to balance positive vs negative examples
            gamma: Exponent of the modulating factor to balance easy vs hard examples
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            
        Returns:
            Focal loss value
        """
        ce_loss = self.ce(predictions, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DepthLoss(nn.Module):
    """Loss for depth estimation (MAE)."""
    
    def __init__(self) -> None:
        """Initialize depth loss."""
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate depth MAE loss.
        
        Args:
            predictions: Predicted depth maps (B, 1, H, W)
            targets: Ground truth depth maps (B, H, W)
            
        Returns:
            MAE loss value
        """
        # Ensure same shape
        if predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        
        mae_loss = torch.abs(predictions - targets).mean()
        return mae_loss


class Trainer:
    """Training manager with checkpoint and metrics tracking."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: str = config.DEVICE,
        num_epochs: int = config.NUM_EPOCHS,
        learning_rate: float = config.LEARNING_RATE,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            device: Device to use (cuda or cpu)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            logger: Logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.logger = logger or logging.getLogger(__name__)
        
        # Loss functions
        self.class_loss_fn = FocalLoss() if config.USE_FOCAL_LOSS else nn.CrossEntropyLoss()
        self.depth_loss_fn = DepthLoss()
        
        # Optimizer
        if config.OPTIMIZER.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=config.WEIGHT_DECAY,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY,
            )
        
        # Scheduler
        if config.SCHEDULER.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
            )
        elif config.SCHEDULER.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=num_epochs // 3,
                gamma=0.1,
            )
        else:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.9,
            )
        
        # Automatic Mixed Precision
        self.use_amp = config.USE_AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.best_iou = 0.0
        self.best_epoch = 0
        self.metrics_history = {
            'train': {},
            'val': {},
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of metrics for this epoch
        """
        self.model.train()
        metrics = {
            'loss': 0.0,
            'class_loss': 0.0,
            'depth_loss': 0.0,
            'iou': 0.0,
            'dice': 0.0,
        }
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            depths = batch['depth'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    class_logits, depth_pred = self.model(images)
                    class_loss = self.class_loss_fn(class_logits, masks)
                    depth_loss = self.depth_loss_fn(depth_pred, depths)
                    total_loss = (
                        config.CLASS_LOSS_WEIGHT * class_loss +
                        config.DEPTH_LOSS_WEIGHT * depth_loss
                    )
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                class_logits, depth_pred = self.model(images)
                class_loss = self.class_loss_fn(class_logits, masks)
                depth_loss = self.depth_loss_fn(depth_pred, depths)
                total_loss = (
                    config.CLASS_LOSS_WEIGHT * class_loss +
                    config.DEPTH_LOSS_WEIGHT * depth_loss
                )
                total_loss.backward()
                self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                metrics['loss'] += total_loss.item()
                metrics['class_loss'] += class_loss.item()
                metrics['depth_loss'] += depth_loss.item()
                
                # Calculate segmentation metrics
                preds = torch.argmax(class_logits, dim=1)
                iou = calculate_iou(preds, masks, config.NUM_CLASSES)
                dice = calculate_dice(preds, masks, config.NUM_CLASSES)
                
                metrics['iou'] += iou
                metrics['dice'] += dice
            
            # Log batch progress
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                avg_loss = metrics['loss'] / (batch_idx + 1)
                avg_iou = metrics['iou'] / (batch_idx + 1)
                self.logger.info(
                    f"Epoch [{epoch+1}/{self.num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}"
                )
        
        # Average metrics over all batches
        num_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= num_batches
        
        epoch_time = time.time() - start_time
        self.logger.info(
            f"Epoch [{epoch+1}/{self.num_epochs}] Training completed in {epoch_time:.2f}s"
        )
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        metrics = {
            'loss': 0.0,
            'iou': 0.0,
            'dice': 0.0,
            'accuracy': 0.0,
            'depth_mae': 0.0,
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                depths = batch['depth'].to(self.device)
                
                # Forward pass
                class_logits, depth_pred = self.model(images)
                
                # Calculate loss
                class_loss = self.class_loss_fn(class_logits, masks)
                depth_loss = self.depth_loss_fn(depth_pred, depths)
                total_loss = (
                    config.CLASS_LOSS_WEIGHT * class_loss +
                    config.DEPTH_LOSS_WEIGHT * depth_loss
                )
                
                metrics['loss'] += total_loss.item()
                
                # Calculate metrics
                preds = torch.argmax(class_logits, dim=1)
                iou = calculate_iou(preds, masks, config.NUM_CLASSES)
                dice = calculate_dice(preds, masks, config.NUM_CLASSES)
                accuracy = calculate_accuracy(preds, masks)
                depth_mae = calculate_depth_mae(depth_pred, depths)
                
                metrics['iou'] += iou
                metrics['dice'] += dice
                metrics['accuracy'] += accuracy
                metrics['depth_mae'] += depth_mae
        
        # Average metrics
        num_batches = len(self.val_loader)
        for key in metrics:
            metrics[key] /= num_batches
        
        self.logger.info(
            f"Epoch [{epoch+1}/{self.num_epochs}] Validation - "
            f"Loss: {metrics['loss']:.4f} | IoU: {metrics['iou']:.4f} | "
            f"Dice: {metrics['dice']:.4f} | Accuracy: {metrics['accuracy']:.4f}"
        )
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'best_epoch': self.best_epoch,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, config.CHECKPOINT_PATH)
        self.logger.info(f"Checkpoint saved to {config.CHECKPOINT_PATH}")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, config.BEST_MODEL_PATH)
            self.logger.info(f"Best model saved to {config.BEST_MODEL_PATH}")
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Total epochs: {self.num_epochs}")
        
        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.metrics_history['train'][epoch] = train_metrics
            
            # Validate
            if (epoch + 1) % config.VAL_INTERVAL == 0 and self.val_loader is not None:
                val_metrics = self.validate(epoch)
                self.metrics_history['val'][epoch] = val_metrics
                
                # Check if best model
                if val_metrics['iou'] > self.best_iou:
                    self.best_iou = val_metrics['iou']
                    self.best_epoch = epoch
                    self.save_checkpoint(is_best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                self.save_checkpoint(is_best=False)
            
            # Step scheduler
            self.scheduler.step()
        
        self.logger.info(f"Training completed. Best IoU: {self.best_iou:.4f} at epoch {self.best_epoch}")


# ============================================================================
# MAIN
# ============================================================================
def main(args):
    """Main training function."""
    # Setup
    config.set_seed(config.SEED)
    logger = setup_logging(config.LOGS_DIR)
    logger.info(f"Configuration: {vars(config)}")
    
    # Create model
    model = create_model(
        num_classes=config.NUM_CLASSES,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_image_dir=Path(args.train_images),
        train_mask_dir=Path(args.train_masks),
        val_image_dir=Path(args.val_images) if args.val_images else None,
        val_mask_dir=Path(args.val_masks) if args.val_masks else None,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        logger=logger,
    )
    
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Depth-Aware Pseudo-Segmentation model"
    )
    
    parser.add_argument(
        "--train-images",
        type=str,
        default="data/train/images",
        help="Path to training images directory"
    )
    parser.add_argument(
        "--train-masks",
        type=str,
        default="data/train/masks",
        help="Path to training masks directory"
    )
    parser.add_argument(
        "--val-images",
        type=str,
        default="data/val/images",
        help="Path to validation images directory"
    )
    parser.add_argument(
        "--val-masks",
        type=str,
        default="data/val/masks",
        help="Path to validation masks directory"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained backbone"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=False,
        help="Freeze backbone layers"
    )
    
    args = parser.parse_args()
    main(args)
