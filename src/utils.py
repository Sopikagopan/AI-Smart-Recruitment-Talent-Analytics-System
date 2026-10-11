from pathlib import Path
import random

import numpy as np


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# Create a directory if it does not exist
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
