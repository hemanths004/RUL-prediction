"""
Load C-MAPSS raw data and select informative sensors.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from config import col_names, ALL_SENSORS, OP_SETTINGS, N_CONDITIONS, BASE


def load_cmapss(base, fd_id):
    """Load train, test, and RUL data for a given FD dataset."""
    kw = dict(sep=r'\s+', header=None, names=col_names)
    train = pd.read_csv(f'{base}/train_FD00{fd_id}.txt', **kw)
    test  = pd.read_csv(f'{base}/test_FD00{fd_id}.txt',  **kw)
    rul   = pd.read_csv(f'{base}/RUL_FD00{fd_id}.txt', sep=r'\s+', header=None, names=['RUL'])
    return train, test, rul


def select_sensors_single(df, sensors, threshold=0.01):
    """Select sensors with variance above threshold (single operating condition)."""
    variances = {}
    for s in sensors:
        vals = df[s].values.astype(np.float32)
        rng  = vals.max() - vals.min()
        normed = (vals - vals.min()) / rng if rng > 0 else np.zeros_like(vals)
        variances[s] = float(normed.var())
    return [s for s, v in variances.items() if v >= threshold], variances


def select_sensors_multi(df, sensors, n_clusters, threshold=0.01):
    """Select sensors with variance above threshold (multiple operating conditions)."""
    op_norm = MinMaxScaler().fit_transform(df[OP_SETTINGS].values)
    df = df.copy()
    df['condition'] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(op_norm)
    df_norm = df.copy()
    for s in sensors:
        df_norm[s] = df_norm[s].astype(np.float32)
    for cond in range(n_clusters):
        mask = df['condition'] == cond
        for s in sensors:
            vals = df.loc[mask, s].values.astype(np.float32)
            rng  = vals.max() - vals.min()
            df_norm.loc[mask, s] = (vals - vals.min()) / rng if rng > 0 else np.zeros_like(vals)
    variances = {s: float(df_norm[s].var()) for s in sensors}
    return [s for s, v in variances.items() if v >= threshold], variances, df


def load_all_datasets(base_dir=None):
    """
    Load all four FD datasets and select appropriate sensors.

    Returns:
        datasets: dict with keys 1-4, each containing 'train', 'test', 'rul'
        selected_sensors: dict with keys 1-4, each containing list of selected sensor names
    """
    if base_dir is None:
        base_dir = BASE

    datasets = {}
    selected_sensors = {}

    for fd_id in range(1, 5):
        train, test, rul = load_cmapss(base_dir, fd_id)
        datasets[fd_id] = {'train': train, 'test': test, 'rul': rul}
        n_cond = N_CONDITIONS[fd_id]

        if n_cond == 1:
            selected, _ = select_sensors_single(train, ALL_SENSORS)
        else:
            selected, _, train_wc = select_sensors_multi(train, ALL_SENSORS, n_cond)
            datasets[fd_id]['train'] = train_wc

        selected_sensors[fd_id] = selected
        print(f'FD00{fd_id}: {len(selected)} sensors selected')

    return datasets, selected_sensors


if __name__ == '__main__':
    datasets, selected_sensors = load_all_datasets()
    for fd_id in range(1, 5):
        print(f'FD00{fd_id}: sensors = {selected_sensors[fd_id]}')
