"""
Configuration module for Depth-Aware Pseudo-Segmentation project.
Centralizes all hyperparameters, file paths, and hardware settings.
"""

from pathlib import Path
from typing import Dict, List
import torch

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
NUM_CLASSES: int = 10
IMAGE_HEIGHT: int = 512
IMAGE_WIDTH: int = 512
MEAN: List[float] = [0.485, 0.456, 0.406]
STD: List[float] = [0.229, 0.224, 0.225]

# Class names and indices for 10-class semantic segmentation
CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "person",
    2: "car",
    3: "dog",
    4: "cat",
    5: "bicycle",
    6: "tree",
    7: "building",
    8: "sky",
    9: "road",
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 100
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
MOMENTUM: float = 0.9
NUM_WORKERS: int = 4

# Optimizer config
OPTIMIZER: str = "adam"  # "adam" or "sgd"
SCHEDULER: str = "cosine"  # "cosine", "step", or "exponential"
WARMUP_EPOCHS: int = 5

# ============================================================================
# LOSS CONFIGURATION
# ============================================================================
# Class segmentation loss
CLASS_LOSS_WEIGHT: float = 1.0
DEPTH_LOSS_WEIGHT: float = 0.5

# Focal loss parameters (for handling class imbalance)
USE_FOCAL_LOSS: bool = True
FOCAL_ALPHA: float = 0.25
FOCAL_GAMMA: float = 2.0

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NAME: str = "depth_aware_unet"
BACKBONE: str = "resnet50"  # "resnet18", "resnet50", "resnet101"
PRETRAINED: bool = True
FREEZE_BACKBONE: bool = False  # Set to True to freeze backbone layers

# Encoder decoder architecture
ENCODER_DEPTH: int = 5
DECODER_CHANNELS: List[int] = [256, 128, 64, 32, 16]

# Multi-head output configuration
DEPTH_ESTIMATION_HEAD: bool = True
DEPTH_RANGE: tuple = (0.0, 1.0)  # Normalized depth output range

# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================
ENABLE_AUGMENTATION: bool = True
AUGMENTATION_CONFIG: Dict = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.0,
    "rotation": 15,
    "scale": [0.85, 1.15],
    "brightness": 0.2,
    "contrast": 0.2,
    "blur_limit": 3,
    "elastic_transform": 0.1,
}

# ============================================================================
# VALIDATION & CHECKPOINT
# ============================================================================
VAL_INTERVAL: int = 5  # Validate every N epochs
SAVE_INTERVAL: int = 5  # Save checkpoint every N epochs
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.pth"
CHECKPOINT_PATH: Path = MODELS_DIR / "checkpoint_latest.pth"

# ============================================================================
# EVALUATION METRICS
# ============================================================================
METRICS: List[str] = ["iou", "dice", "accuracy", "depth_mae"]
IOU_THRESHOLD: float = 0.5

# Position-based evaluation
EVALUATE_BY_POSITION: bool = True
POSITION_BINS: Dict[str, tuple] = {
    "close": (0.0, 0.33),      # Lower third of image
    "middle": (0.33, 0.67),     # Middle third
    "distant": (0.67, 1.0),     # Upper third
}

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP: bool = torch.cuda.is_available()  # Automatic Mixed Precision
NUM_GPUS: int = torch.cuda.device_count()
DISTRIBUTED_TRAINING: bool = NUM_GPUS > 1

# Inference configuration
INFERENCE_DEVICE: str = DEVICE
MAX_INFERENCE_TIME_MS: float = 50.0  # Target: < 50ms per image
BATCH_INFERENCE: bool = True

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================
LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_INTERVAL: int = 100  # Log metrics every N iterations
DEBUG_MODE: bool = False
SEED: int = 42

# ============================================================================
# RANDOM SEED REPRODUCIBILITY
# ============================================================================
def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility across all libraries."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
TEST_BATCH_SIZE: int = 32
SAVE_PREDICTIONS: bool = True
PREDICTION_OUTPUT_DIR: Path = RESULTS_DIR / "predictions"
PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization
VISUALIZE_PREDICTIONS: bool = True
SAVE_VISUALIZATION: bool = True
VISUALIZATION_OUTPUT_DIR: Path = RESULTS_DIR / "visualizations"
VISUALIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
