"""
Utility script for data preparation and dummy dataset generation.
Useful for testing the pipeline without a real dataset.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def create_dummy_dataset(
    output_dir: Path,
    num_train: int = 100,
    num_val: int = 20,
    num_test: int = 20,
    image_size: tuple = (512, 512),
    num_classes: int = 10,
    seed: int = 42,
) -> None:
    """
    Create a dummy dataset for testing the pipeline.
    
    Args:
        output_dir: Root directory for dataset
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
        image_size: Image size (height, width)
        num_classes: Number of segmentation classes
        seed: Random seed for reproducibility
    """
    logger = setup_logging()
    np.random.seed(seed)
    
    logger.info(f"Creating dummy dataset in {output_dir}")
    
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test,
    }
    
    for split_name, num_samples in splits.items():
        logger.info(f"\nGenerating {split_name} split with {num_samples} samples...")
        
        # Create directories
        split_dir = output_dir / split_name
        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        for i in tqdm(range(num_samples), desc=f"{split_name.capitalize()} samples"):
            # Generate random image
            image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            
            # Generate segmentation mask with some structure
            mask = np.zeros(image_size, dtype=np.uint8)
            
            # Create a few random regions with different classes
            num_regions = np.random.randint(3, 8)
            for _ in range(num_regions):
                # Random position and size
                y1 = np.random.randint(0, image_size[0] - 50)
                x1 = np.random.randint(0, image_size[1] - 50)
                y2 = y1 + np.random.randint(50, 150)
                x2 = x1 + np.random.randint(50, 150)
                
                # Random class
                class_id = np.random.randint(1, num_classes)
                mask[y1:y2, x1:x2] = class_id
            
            # Save image
            image_path = image_dir / f"{i:06d}.jpg"
            Image.fromarray(image).save(image_path)
            
            # Save mask
            mask_path = mask_dir / f"{i:06d}.png"
            Image.fromarray(mask).save(mask_path)
    
    logger.info(f"\n✓ Dummy dataset created successfully in {output_dir}")
    logger.info(f"  - Train: {num_train} samples")
    logger.info(f"  - Val: {num_val} samples")
    logger.info(f"  - Test: {num_test} samples")
    logger.info(f"  - Image size: {image_size}")
    logger.info(f"  - Classes: {num_classes}")


def verify_dataset(data_dir: Path) -> None:
    """
    Verify dataset structure and files.
    
    Args:
        data_dir: Root data directory
    """
    logger = setup_logging()
    
    logger.info(f"Verifying dataset in {data_dir}")
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        
        if not image_dir.exists() or not mask_dir.exists():
            logger.warning(f"  ✗ {split} split missing")
            continue
        
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        mask_files = list(mask_dir.glob("*.png"))
        
        logger.info(
            f"  ✓ {split} split: {len(image_files)} images, {len(mask_files)} masks"
        )
        
        if len(image_files) != len(mask_files):
            logger.warning(
                f"    ⚠ Mismatch: {len(image_files)} images vs {len(mask_files)} masks"
            )


def main(args):
    """Main function."""
    output_dir = Path(args.output_dir)
    
    if args.command == "generate":
        create_dummy_dataset(
            output_dir=output_dir,
            num_train=args.train_samples,
            num_val=args.val_samples,
            num_test=args.test_samples,
            image_size=(args.image_size, args.image_size),
            num_classes=args.num_classes,
            seed=args.seed,
        )
    
    elif args.command == "verify":
        verify_dataset(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data preparation utilities"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate dummy dataset"
    )
    gen_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for dataset"
    )
    gen_parser.add_argument(
        "--train-samples",
        type=int,
        default=100,
        help="Number of training samples"
    )
    gen_parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
        help="Number of validation samples"
    )
    gen_parser.add_argument(
        "--test-samples",
        type=int,
        default=20,
        help="Number of test samples"
    )
    gen_parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size (square)"
    )
    gen_parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes"
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Verify command
    ver_parser = subparsers.add_parser(
        "verify",
        help="Verify dataset structure"
    )
    ver_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Data directory to verify"
    )
    
    args = parser.parse_args()
    
    if args.command:
        main(args)
    else:
        parser.print_help()
