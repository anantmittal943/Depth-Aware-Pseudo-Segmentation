"""
Dataset module for Depth-Aware Pseudo-Segmentation.
Handles data loading, augmentation, and preprocessing with robust error handling.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

import config

logger = logging.getLogger(__name__)


class DepthAwareSegmentationDataset(Dataset):
    """
    PyTorch Dataset for semantic segmentation with pseudo-depth estimation.
    
    Handles:
    - Image and mask loading with corruption handling
    - Data augmentation with albumentations
    - Normalization using ImageNet statistics
    - Depth estimation based on vertical position (spatial prior)
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        split: str = "train",
        augment: bool = True,
        image_size: Tuple[int, int] = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            image_dir: Path to directory containing images
            mask_dir: Path to directory containing segmentation masks
            split: Data split - 'train', 'val', or 'test'
            augment: Whether to apply augmentation
            image_size: Target image size (height, width)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.split = split
        self.augment = augment and (split == "train")
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = sorted(
            [f for f in self.image_dir.glob("**/*.jpg") 
             if f.is_file()] +
            [f for f in self.image_dir.glob("**/*.png")
             if f.is_file()] +
            [f for f in self.image_dir.glob("**/*.jpeg")
             if f.is_file()]
        )
        
        if not self.image_files:
            logger.warning(f"No images found in {self.image_dir}")
            self.image_files = []
        
        logger.info(f"Initialized {split} dataset with {len(self.image_files)} images")
        
        # Set up augmentation pipeline
        self.transform = self._get_transforms()

    def _get_transforms(self) -> A.Compose:
        """Create augmentation pipeline based on split."""
        if self.augment and config.ENABLE_AUGMENTATION:
            aug_config = config.AUGMENTATION_CONFIG
            return A.Compose([
                A.Resize(
                    height=self.image_size[0],
                    width=self.image_size[1],
                    p=1.0
                ),
                A.HorizontalFlip(
                    p=aug_config.get("horizontal_flip", 0.5)
                ),
                A.VerticalFlip(
                    p=aug_config.get("vertical_flip", 0.0)
                ),
                A.Rotate(
                    limit=aug_config.get("rotation", 15),
                    p=0.5
                ),
                A.RandomScale(
                    scale_limit=aug_config.get("scale", [0.85, 1.15]),
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=aug_config.get("brightness", 0.2),
                    contrast_limit=aug_config.get("contrast", 0.2),
                    p=0.5
                ),
                A.GaussBlur(
                    blur_limit=aug_config.get("blur_limit", 3),
                    p=0.3
                ),
                A.Normalize(
                    mean=config.MEAN,
                    std=config.STD,
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=False
            ))
        else:
            return A.Compose([
                A.Resize(
                    height=self.image_size[0],
                    width=self.image_size[1],
                    p=1.0
                ),
                A.Normalize(
                    mean=config.MEAN,
                    std=config.STD,
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ])

    def _safe_load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Safely load image with error handling.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array or None if loading fails
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def _safe_load_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        """
        Safely load segmentation mask with error handling.
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            Mask array or None if loading fails
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                return None
            return mask
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {e}")
            return None

    def _compute_pseudo_depth(
        self,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute pseudo-depth estimation based on vertical position.
        
        Spatial prior: Objects lower in image (higher y-coordinates) are closer.
        This creates a depth map where y-position influences depth values.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Pseudo-depth map (normalized to [0, 1])
        """
        height, width = mask.shape[:2]
        
        # Create depth map based on vertical position
        # Lower in image (higher y) = closer (higher depth value)
        depth = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            # Normalize vertical position: 0 at top, 1 at bottom
            vertical_position = y / (height - 1) if height > 1 else 0.0
            # Depth increases towards bottom (closer objects)
            depth[y, :] = vertical_position
        
        # Adjust depth based on which objects are present
        # If mask has non-background pixels, refine the depth
        for class_id in range(1, config.NUM_CLASSES):
            class_mask = (mask == class_id).astype(np.float32)
            if class_mask.sum() > 0:
                # Get average y-position of this class
                y_coords, x_coords = np.where(class_mask > 0)
                if len(y_coords) > 0:
                    mean_y = np.mean(y_coords) / (height - 1) if height > 1 else 0.0
                    # Refine depth for this class based on its position
                    depth = np.where(
                        class_mask > 0.5,
                        mean_y,
                        depth
                    )
        
        # Normalize to [0, 1]
        depth = np.clip(depth, 0.0, 1.0)
        return depth

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'image', 'mask', and 'depth' tensors
        """
        image_path = self.image_files[idx]
        mask_path = self.mask_dir / image_path.relative_to(self.image_dir).with_suffix('.png')
        
        # Load image and mask
        image = self._safe_load_image(image_path)
        mask = self._safe_load_mask(mask_path)
        
        # Handle corrupted files
        if image is None or mask is None:
            logger.warning(f"Using fallback for corrupted files at index {idx}")
            # Return a valid sample with fallback
            image = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 128
            mask = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Apply augmentation/normalization
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Manual normalization if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        # Compute pseudo-depth map
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        depth = self._compute_pseudo_depth(mask_np)
        depth = torch.from_numpy(depth).float()
        
        # Ensure mask is long tensor
        if mask.dtype != torch.long:
            mask = mask.long()
        
        return {
            'image': image,
            'mask': mask,
            'depth': depth,
        }


def create_dataloaders(
    train_image_dir: Path,
    train_mask_dir: Path,
    val_image_dir: Optional[Path] = None,
    val_mask_dir: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_image_dir: Training images directory
        train_mask_dir: Training masks directory
        val_image_dir: Validation images directory (optional)
        val_mask_dir: Validation masks directory (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = DepthAwareSegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        split="train",
        augment=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    
    val_loader = None
    if val_image_dir and val_mask_dir:
        val_dataset = DepthAwareSegmentationDataset(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            split="val",
            augment=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    return train_loader, val_loader
