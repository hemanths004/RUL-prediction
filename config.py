"""
Configuration and constants for Turbofan RUL prediction.
CNN+BiLSTM+3DAttn + True MoE(4) + MC Dropout
"""
import os
import random
import numpy as np
import torch

# ─── Column names for C-MAPSS data ───────────────────────────────────────────
col_names = (
    ['engine_id', 'cycle']
    + [f'op_{i}' for i in range(1, 4)]
    + [f's{i}' for i in range(1, 22)]
)

ALL_SENSORS = [f's{i}' for i in range(1, 22)]
OP_SETTINGS = ['op_1', 'op_2', 'op_3']
N_CONDITIONS = {1: 1, 2: 6, 3: 1, 4: 6}

# ─── Per-dataset hyperparameters ─────────────────────────────────────────────
RUL_MAX     = {1: 130, 2: 125, 3: 125, 4: 150}
WINDOW_SIZE = {1: 30,  2: 60,  3: 30,  4: 60}

VAR_THRESHOLD = 0.01
BATCH_SIZE    = 256
MC_T          = 50
N_EXPERTS     = 4

TRAIN_CFG = {
    1: {'epochs': 300, 'lr': 5e-4, 'wd': 5e-3, 'dropout': 0.35, 'patience': 25},
    2: {'epochs': 400, 'lr': 2e-4, 'wd': 5e-3, 'dropout': 0.35, 'patience': 30},
    3: {'epochs': 300, 'lr': 5e-4, 'wd': 5e-3, 'dropout': 0.4,  'patience': 25},
    4: {'epochs': 450, 'lr': 1e-4, 'wd': 5e-3, 'dropout': 0.35, 'patience': 35},
}

BASE = os.environ.get('CMAPSS_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

# ─── Random seed and device ──────────────────────────────────────────────────
SEED = 42


def setup_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Module-level device (set once on import)
DEVICE = get_device()
