"""
Scaler Manager for Turbofan RUL Prediction.
Caches and provides condition scalers, KMeans operating condition models,
op_settings scalers, and handcrafted feature scalers for FD001-FD004.
"""
import os
import sys
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from config import (
    BASE, OP_SETTINGS, ALL_SENSORS, N_CONDITIONS,
    WINDOW_SIZE, RUL_MAX, SEED, setup_seed
)
from data_loading import load_cmapss, select_sensors_single, select_sensors_multi
from data_processing import add_piecewise_rul, fit_condition_scalers, normalize_by_condition
from dataset import CMAPSSDataset

SCALERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scalers')
SCALERS_CACHE_FILE = os.path.join(SCALERS_DIR, 'all_scalers.pkl')


def compute_and_cache_scalers(data_dir=None, force=False):
    """
    Fit all preprocessing scalers on training datasets and persist to disk.
    This ensures identical deterministic preprocessing without needing to re-fit
    at server runtime.
    """
    if data_dir is None:
        data_dir = BASE
        
    os.makedirs(SCALERS_DIR, exist_ok=True)
    if os.path.exists(SCALERS_CACHE_FILE) and not force:
        print(f"[ScalerManager] Loading pre-cached scalers from {SCALERS_CACHE_FILE}")
        with open(SCALERS_CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print(f"[ScalerManager] Computing scalers from training data in {data_dir}...")
    setup_seed(SEED)
    
    scalers_data = {}
    for fd_id in range(1, 5):
        print(f"  Fitting scalers for FD00{fd_id}...")
        train, test, rul = load_cmapss(data_dir, fd_id)
        n_cond = N_CONDITIONS[fd_id]
        W = WINDOW_SIZE[fd_id]
        RM = RUL_MAX[fd_id]
        
        # Sensor selection
        if n_cond == 1:
            selected, _ = select_sensors_single(train, ALL_SENSORS)
        else:
            selected, _, train = select_sensors_multi(train, ALL_SENSORS, n_cond)
            
        train = add_piecewise_rul(train, RM)
        
        # Condition scalers
        cs, km, op_scaler = fit_condition_scalers(train, selected, OP_SETTINGS, n_cond)
        train_norm = normalize_by_condition(train, selected, OP_SETTINGS, cs, km, op_scaler)
        
        # Handcrafted feature scaler
        train_ds = CMAPSSDataset(train_norm, selected, W, RM, is_train=True)
        hc_scaler = train_ds.hc_scaler
        hc_dim = train_ds.hc_feats.shape[1]
        
        scalers_data[fd_id] = {
            'selected_sensors': selected,
            'cond_scalers': cs,
            'km': km,
            'op_scaler': op_scaler,
            'hc_scaler': hc_scaler,
            'hc_dim': hc_dim,
            'n_cond': n_cond,
            'window_size': W,
            'rul_max': RM,
        }
        print(f"  ✓ FD00{fd_id}: {len(selected)} sensors, {hc_dim} HC feats")

    with open(SCALERS_CACHE_FILE, 'wb') as f:
        pickle.dump(scalers_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"[ScalerManager] Saved all scalers to {SCALERS_CACHE_FILE}")
    return scalers_data


_SCALERS_CACHE = None

def get_all_scalers():
    global _SCALERS_CACHE
    if _SCALERS_CACHE is None:
        _SCALERS_CACHE = compute_and_cache_scalers()
    return _SCALERS_CACHE


def get_scaler_for_dataset(fd_id):
    scalers = get_all_scalers()
    return scalers.get(fd_id)


if __name__ == '__main__':
    compute_and_cache_scalers(force=True)
