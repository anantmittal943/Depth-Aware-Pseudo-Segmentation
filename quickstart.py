"""
Quick Start Guide - Example script showing complete pipeline usage.
Run this to understand the full workflow.
"""

import sys
from pathlib import Path

def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Show complete pipeline example."""
    
    print_section("DEPTH-AWARE PSEUDO-SEGMENTATION - QUICK START")
    
    # Step 1: Environment Setup
    print_section("Step 1: Environment Setup")
    print("""
    1. Create and activate virtual environment:
       python -m venv venv
       venv\\Scripts\\activate  # Windows
       source venv/bin/activate  # macOS/Linux
    
    2. Install dependencies:
       pip install -r requirements.txt
    """)
    
    # Step 2: Prepare Dataset
    print_section("Step 2: Prepare Dataset")
    print("""
    Option A: Generate dummy dataset for testing:
       python prepare_data.py generate --train-samples 100 --val-samples 20 --test-samples 20
    
    Option B: Use your own dataset:
       Place your data in: data/train/images, data/train/masks, etc.
       See README.md for dataset format requirements
    
    Verify dataset structure:
       python prepare_data.py verify
    """)
    
    # Step 3: Configuration
    print_section("Step 3: Configuration (Optional)")
    print("""
    Edit config.py to customize:
    - Model architecture (BACKBONE, DECODER_CHANNELS)
    - Training parameters (BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS)
    - Loss weights (CLASS_LOSS_WEIGHT, DEPTH_LOSS_WEIGHT)
    - Hardware settings (USE_AMP, NUM_WORKERS)
    
    Default configuration is already optimized for most use cases.
    """)
    
    # Step 4: Training
    print_section("Step 4: Training")
    print("""
    Basic training:
       python train.py \\
           --train-images data/train/images \\
           --train-masks data/train/masks \\
           --val-images data/val/images \\
           --val-masks data/val/masks \\
           --pretrained
    
    With custom options:
       python train.py \\
           --train-images data/train/images \\
           --train-masks data/train/masks \\
           --val-images data/val/images \\
           --val-masks data/val/masks \\
           --freeze-backbone False
    
    Output:
    - models/best_model.pth - Best checkpoint
    - models/checkpoint_latest.pth - Latest checkpoint
    - logs/training_*.log - Training log
    """)
    
    # Step 5: Testing
    print_section("Step 5: Testing & Evaluation")
    print("""
    Run inference on test set:
       python test.py \\
           --test-images data/test/images \\
           --test-masks data/test/masks \\
           --model-path models/best_model.pth
    
    Output includes:
    - Global metrics (IoU, Dice, Accuracy, Depth MAE)
    - Position-based metrics (close/middle/distant objects)
    - Per-class IoU scores
    - Confusion matrix
    - Inference timing statistics
    - Visualization: results/visualizations/confusion_matrix.png
    """)
    
    # Step 6: Interpretation
    print_section("Step 6: Interpret Results")
    print("""
    Check the test output for:
    
    1. Inference Speed:
       ✓ Inference speed target MET: 25.34ms <= 50ms
       If not met, apply optimizations from README.md
    
    2. Position-Based Performance:
       CLOSE:    IoU: 0.85  (lower objects - should be best)
       MIDDLE:   IoU: 0.82
       DISTANT:  IoU: 0.78  (upper objects - improves with spatial prior)
       
       The gap shows how well spatial prior helps distant objects.
    
    3. Per-Class Performance:
       Check which classes need improvement:
       background: 0.92
       person:     0.78
       car:        0.85
       ...
    
    4. Confusion Matrix:
       Saved to: results/visualizations/confusion_matrix.png
       Shows which classes are confused with each other.
    """)
    
    # Step 7: Next Steps
    print_section("Step 7: Next Steps")
    print("""
    If performance needs improvement:
    
    1. Poor inference speed (> 50ms):
       - Use smaller backbone: BACKBONE = "resnet18"
       - Enable batch inference: BATCH_INFERENCE = True
       - Reduce image size: IMAGE_SIZE = 256
    
    2. Low accuracy on distant objects:
       - Increase DEPTH_LOSS_WEIGHT: 0.5 -> 1.0
       - Add more training data from distant scenes
       - Increase NUM_EPOCHS: 100 -> 150
    
    3. Class imbalance issues:
       - Adjust loss weights per class
       - Use FOCAL_LOSS = True (enabled by default)
       - Increase USE_FOCAL_LOSS: FOCAL_GAMMA
    
    4. General performance boost:
       - Train longer: NUM_EPOCHS = 150+
       - Use better learning rate scheduler
       - Augment dataset with more variations
       - Use stronger backbone: resnet101
    """)
    
    # File Structure
    print_section("File Structure & Quick Reference")
    print("""
    config.py          - All hyperparameters and settings
    dataset.py         - Dataset loading with augmentation
    model.py           - Multi-head U-Net architecture
    train.py           - Training loop with logging
    test.py            - Inference and evaluation
    metrics.py         - Metric calculations
    prepare_data.py    - Data preparation utilities
    README.md          - Full documentation
    requirements.txt   - Python dependencies
    
    data/               - Dataset directory
    models/             - Saved model checkpoints
    logs/               - Training and test logs
    results/            - Predictions and visualizations
    """)
    
    # Key Commands
    print_section("Key Commands Summary")
    print("""
    # Generate dummy data
    python prepare_data.py generate
    
    # Train model
    python train.py
    
    # Test model
    python test.py
    
    # View training log
    tail -f logs/training_*.log
    
    # Check GPU status
    python -c "import torch; print(torch.cuda.is_available())"
    """)
    
    print_section("Good Luck!")
    print("""
    For detailed information, see:
    - README.md for full documentation
    - config.py for parameter descriptions
    - Log files for detailed metrics and errors
    
    Questions? Check the Troubleshooting section in README.md
    """)


if __name__ == "__main__":
    main()
