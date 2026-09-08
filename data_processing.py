"""
Preprocessing: RUL computation, condition-based normalization.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from config import OP_SETTINGS, N_CONDITIONS, RUL_MAX


def add_piecewise_rul(df, rul_max):
    """Add piecewise-linear RUL column to training data."""
    mc = df.groupby('engine_id')['cycle'].max().reset_index()
    mc.columns = ['engine_id', 'max_cycle']
    df = df.merge(mc, on='engine_id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=rul_max)
    df.drop('max_cycle', axis=1, inplace=True)
    return df


def fit_condition_scalers(train_df, sensors, op_settings, n_conditions):
    """
    Fit per-condition MinMaxScalers on TRAINING data only.
    Also fits and saves the op_settings scaler used for KMeans assignment.

    Returns:
        cond_scalers: dict of {condition_id: MinMaxScaler}
        km: KMeans model (or None for single condition)
        op_scaler: MinMaxScaler for op_settings (or None for single condition)
    """
    if n_conditions == 1:
        sc = MinMaxScaler()
        sc.fit(train_df[sensors])
        return {0: sc}, None, None

    op_scaler = MinMaxScaler()
    op_norm = op_scaler.fit_transform(train_df[op_settings].values)

    km = KMeans(n_clusters=n_conditions, random_state=42, n_init=10)
    km.fit(op_norm)
    labels = km.predict(op_norm)

    cond_scalers = {}
    for cond in range(n_conditions):
        mask = labels == cond
        sc = MinMaxScaler()
        sc.fit(train_df.loc[mask, sensors])
        cond_scalers[cond] = sc

    return cond_scalers, km, op_scaler


def normalize_by_condition(df, sensors, op_settings, cond_scalers, km, op_scaler):
    """
    Normalize sensor values per operating condition.
    Uses the op_scaler fitted on TRAINING data to assign conditions consistently.
    """
    df = df.copy()
    for s in sensors:
        df[s] = df[s].astype(np.float64)
    if km is None:  # single condition
        df[sensors] = cond_scalers[0].transform(df[sensors])
        return df

    op_norm = op_scaler.transform(df[op_settings].values)
    cond_labels = km.predict(op_norm)
    df['_cond'] = cond_labels
    for cond in cond_scalers:
        mask = df['_cond'] == cond
        if mask.any():
            df.loc[mask, sensors] = cond_scalers[cond].transform(df.loc[mask, sensors])
    df.drop('_cond', axis=1, inplace=True)
    return df


def preprocess_all(datasets, selected_sensors):
    """
    Apply preprocessing to all datasets:
    - Add piecewise RUL to training data
    - Build RUL for test data
    - Fit condition scalers on training data
    - Normalize both train and test consistently

    Args:
        datasets: dict from load_all_datasets()
        selected_sensors: dict of selected sensor names per fd_id

    Returns:
        processed: dict with keys 1-4, each containing 'train' and 'test' DataFrames
        cond_scalers: dict of condition scalers per fd_id
        cond_km: dict of KMeans models per fd_id
        op_scalers: dict of op_settings scalers per fd_id
    """
    processed = {}
    cond_scalers_dict = {}
    cond_km_dict = {}
    op_scalers_dict = {}

    for fd_id in range(1, 5):
        train   = datasets[fd_id]['train'].copy()
        test    = datasets[fd_id]['test'].copy()
        rul     = datasets[fd_id]['rul'].copy()
        sensors = selected_sensors[fd_id]
        n_cond  = N_CONDITIONS[fd_id]

        # Add piecewise RUL to training data
        train = add_piecewise_rul(train, RUL_MAX[fd_id])

        # Build RUL for test data
        test_rul_list = []
        for eng_id, grp in test.groupby('engine_id'):
            n_cycles = len(grp)
            true_rul = int(rul.iloc[eng_id - 1]['RUL'])
            test_rul_list.extend(range(true_rul + n_cycles - 1, true_rul - 1, -1))
        test['RUL'] = np.clip(test_rul_list, 0, RUL_MAX[fd_id])

        # Fit condition scalers on TRAIN, then apply consistently to TRAIN and TEST
        cs, km, op_scaler = fit_condition_scalers(train, sensors, OP_SETTINGS, n_cond)

        train = normalize_by_condition(train, sensors, OP_SETTINGS, cs, km, op_scaler)
        test  = normalize_by_condition(test,  sensors, OP_SETTINGS, cs, km, op_scaler)

        cond_scalers_dict[fd_id] = cs
        cond_km_dict[fd_id]      = km
        op_scalers_dict[fd_id]   = op_scaler
        processed[fd_id]         = {'train': train, 'test': test}

        # Verify condition distribution
        if km is not None:
            op_norm_tr = op_scaler.transform(train[OP_SETTINGS].values)
            op_norm_te = op_scaler.transform(test[OP_SETTINGS].values)
            tr_counts  = np.bincount(km.predict(op_norm_tr), minlength=n_cond)
            te_counts  = np.bincount(km.predict(op_norm_te), minlength=n_cond)
            print(f'FD00{fd_id} condition distribution:')
            print(f'  Train: {(tr_counts / tr_counts.sum() * 100).round(1)}%')
            print(f'  Test:  {(te_counts / te_counts.sum() * 100).round(1)}%')
        else:
            print(f'FD00{fd_id} preprocessed (single condition)')

        print(f'  train:{train.shape}, test:{test.shape}')

    return processed, cond_scalers_dict, cond_km_dict, op_scalers_dict
