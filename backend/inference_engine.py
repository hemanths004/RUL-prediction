"""
Inference Engine for Turbofan Engine RUL Prediction.
Performs real-time model inference using trained PyTorch checkpoints,
condition-aware preprocessing, and MC Dropout uncertainty estimation.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    col_names, ALL_SENSORS, OP_SETTINGS, N_CONDITIONS,
    WINDOW_SIZE, RUL_MAX, TRAIN_CFG, N_EXPERTS
)
from data_loading import load_cmapss
from dataset import compute_window_hc
from model import TurbofanRULModel
from backend.scaler_manager import get_all_scalers

def get_inference_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS fallback check
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # test small tensor
            t = torch.zeros(1, device='mps')
            return torch.device('mps')
        except Exception:
            pass
    return torch.device('cpu')

DEVICE = get_inference_device()


class RULInferenceEngine:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.join(PROJECT_ROOT, 'data')
        self.device = DEVICE
        print(f"[InferenceEngine] Initialized on device: {self.device}")

        # Load scalers from cache
        self.scalers = get_all_scalers()

        # Cache datasets (raw test and train)
        self.datasets = {}
        self._load_datasets()

        # Load PyTorch models
        self.models = {}
        self._load_models()

        # Prediction cache: (fd_id, engine_id, cycle) -> prediction dict
        self._prediction_cache = {}
        # Fleet summary cache: (fd_id, dataset_type) -> fleet summary dict
        self._fleet_cache = {}

    def _load_datasets(self):
        """Preload test data and RUL ground truth for fast access."""
        for fd_id in range(1, 5):
            train, test, rul = load_cmapss(self.data_dir, fd_id)
            self.datasets[fd_id] = {
                'train': train,
                'test': test,
                'rul': rul
            }

    def _load_models(self):
        """Load trained PyTorch checkpoints for FD001-FD004."""
        models_dir = PROJECT_ROOT
        for fd_id in range(1, 5):
            scaler_info = self.scalers[fd_id]
            cfg = TRAIN_CFG[fd_id]
            n_sensors = len(scaler_info['selected_sensors'])
            n_hc = scaler_info['hc_dim']

            model = TurbofanRULModel(
                n_sensors=n_sensors,
                n_hc_features=n_hc,
                dropout=cfg['dropout'],
                n_experts=N_EXPERTS
            ).to(self.device)

            model_path = os.path.join(models_dir, f'model_FD00{fd_id}.pt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            self.models[fd_id] = model
            print(f"[InferenceEngine] Loaded model FD00{fd_id} from {model_path}")

    def get_dataset_info(self, fd_id):
        """Return dataset properties and window configuration."""
        fd_id = int(fd_id)
        if fd_id not in self.scalers:
            raise ValueError(f"Invalid dataset FD00{fd_id}")
        info = self.scalers[fd_id]
        test_df = self.datasets[fd_id]['test']
        engine_ids = sorted(test_df['engine_id'].unique().tolist())
        return {
            "dataset": f"FD00{fd_id}",
            "fd_id": fd_id,
            "window_size": info['window_size'],
            "rul_max": info['rul_max'],
            "n_conditions": info['n_cond'],
            "n_sensors": len(info['selected_sensors']),
            "selected_sensors": info['selected_sensors'],
            "total_engines": len(engine_ids),
            "engine_ids": engine_ids,
        }

    def get_engine_data(self, fd_id, engine_id, dataset_type='test'):
        """Return all sensor and setting rows for a given engine."""
        fd_id = int(fd_id)
        engine_id = int(engine_id)
        df = self.datasets[fd_id][dataset_type]
        eng_df = df[df['engine_id'] == engine_id].sort_values('cycle').reset_index(drop=True)
        if eng_df.empty:
            raise ValueError(f"Engine {engine_id} not found in FD00{fd_id} ({dataset_type})")

        # Compute cluster condition for each cycle if multi-condition
        scaler_info = self.scalers[fd_id]
        km = scaler_info['km']
        op_scaler = scaler_info['op_scaler']

        if km is not None and op_scaler is not None:
            op_norm = op_scaler.transform(eng_df[OP_SETTINGS].values)
            eng_df['condition_cluster'] = km.predict(op_norm).astype(int)
        else:
            eng_df['condition_cluster'] = 0

        # Ground truth RUL if available
        if dataset_type == 'test':
            true_final_rul = int(self.datasets[fd_id]['rul'].iloc[engine_id - 1]['RUL'])
            max_c = eng_df['cycle'].max()
            eng_df['ground_truth_rul'] = (true_final_rul + (max_c - eng_df['cycle'])).clip(0, scaler_info['rul_max'])
        else:
            eng_df['ground_truth_rul'] = None

        return eng_df

    def predict_window(self, fd_id, window_df, current_cycle=None, mc_samples=50):
        """
        Run actual PyTorch inference on a single sequential sensor window.
        """
        fd_id = int(fd_id)
        scaler_info = self.scalers[fd_id]
        W = scaler_info['window_size']
        RM = scaler_info['rul_max']
        sensors = scaler_info['selected_sensors']
        cs = scaler_info['cond_scalers']
        km = scaler_info['km']
        op_scaler = scaler_info['op_scaler']
        hc_scaler = scaler_info['hc_scaler']
        model = self.models[fd_id]

        if len(window_df) < W:
            return {
                "status": "insufficient_history",
                "min_required_cycles": W,
                "available_cycles": len(window_df),
                "message": f"Insufficient history for RUL prediction. Minimum {W} cycles required.",
                "predicted_rul": None,
                "uncertainty": None,
                "confidence_interval": None,
                "health_status": "INSUFFICIENT_DATA"
            }

        # Take last W rows
        window = window_df.iloc[-W:].copy()
        for s in sensors:
            window[s] = window[s].astype(np.float64)

        # Condition-aware normalization
        normed_win = np.zeros((W, len(sensors)), dtype=np.float32)
        if km is None:
            normed_win = cs[0].transform(window[sensors]).astype(np.float32)
            detected_cond = 0
        else:
            op_norm = op_scaler.transform(window[OP_SETTINGS].values)
            cond_labels = km.predict(op_norm)
            detected_cond = int(cond_labels[-1])  # Current cycle condition
            for cond in cs:
                mask = cond_labels == cond
                if mask.any():
                    normed_win[mask] = cs[cond].transform(window.loc[mask, sensors]).astype(np.float32)

        # Handcrafted statistical features
        raw_hc = compute_window_hc(normed_win)
        normed_hc = np.clip(hc_scaler.transform([raw_hc]), 0, 1).astype(np.float32)

        # Form Tensors
        x_seq = torch.tensor(normed_win[np.newaxis, ...], dtype=torch.float32).to(self.device)
        x_hc = torch.tensor(normed_hc, dtype=torch.float32).to(self.device)

        # MC Dropout Inference (model.train() activates dropout)
        model.train()
        with torch.no_grad():
            preds = torch.stack(
                [model(x_seq, x_hc, training=False).squeeze() for _ in range(mc_samples)]
            ) * RM

        preds_np = preds.cpu().numpy()
        mean_rul = float(np.mean(preds_np))
        std_rul = float(np.std(preds_np))
        lo_ci = float(np.quantile(preds_np, 0.05))
        hi_ci = float(np.quantile(preds_np, 0.95))

        cur_c = int(current_cycle if current_cycle is not None else window['cycle'].iloc[-1])
        est_fail_cycle = int(round(cur_c + mean_rul))

        # Health status determination
        # Configurable thresholds: Critical < 25, Warning < 55, Healthy >= 55
        if mean_rul < 25 or (mean_rul < 35 and std_rul > 10):
            health_status = "CRITICAL"
        elif mean_rul < 55:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"

        # Dynamic Alerts
        alerts = []
        if mean_rul < 25:
            alerts.append({
                "type": "LOW_RUL_CRITICAL",
                "severity": "critical",
                "title": "CRITICAL LOW RUL ALERT",
                "message": f"Predicted RUL ({mean_rul:.1f} cycles) is below critical limit. Failure estimated at cycle {est_fail_cycle}."
            })
        elif mean_rul < 45:
            alerts.append({
                "type": "LOW_RUL_WARNING",
                "severity": "warning",
                "title": "LOW RUL WARNING",
                "message": f"Engine entered warning threshold ({mean_rul:.1f} cycles remaining)."
            })

        if std_rul >= 8.0:
            alerts.append({
                "type": "HIGH_UNCERTAINTY",
                "severity": "warning",
                "title": "HIGH UNCERTAINTY ALERT",
                "message": f"Prediction uncertainty is elevated (±{std_rul:.1f} cycles). Confidence interval spans [{lo_ci:.1f}, {hi_ci:.1f}]."
            })

        # Decision support
        if health_status == "CRITICAL":
            maintenance_rec = {
                "action": "Immediate Inspection & Overhaul",
                "priority": "HIGH",
                "recommended_window": f"Within 0 - {max(1, int(mean_rul * 0.5))} cycles",
                "status": "Urgent Action Required"
            }
        elif health_status == "WARNING":
            maintenance_rec = {
                "action": "Schedule Preventative Maintenance",
                "priority": "MEDIUM",
                "recommended_window": f"Within {max(1, int(mean_rul * 0.4))} - {int(mean_rul * 0.8)} cycles",
                "status": "Maintenance Advisory"
            }
        else:
            maintenance_rec = {
                "action": "Normal Operating Parameters",
                "priority": "LOW",
                "recommended_window": f"Next routine check (~{int(mean_rul * 0.75)} cycles)",
                "status": "Operational"
            }

        return {
            "status": "success",
            "dataset": f"FD00{fd_id}",
            "current_cycle": cur_c,
            "estimated_failure_cycle": est_fail_cycle,
            "predicted_rul": round(mean_rul, 1),
            "uncertainty": round(std_rul, 1),
            "confidence_interval": {
                "lower": round(lo_ci, 1),
                "upper": round(hi_ci, 1)
            },
            "health_status": health_status,
            "operating_condition_cluster": detected_cond,
            "mc_samples": mc_samples,
            "alerts": alerts,
            "maintenance_recommendation": maintenance_rec
        }

    def predict_engine_cycle(self, fd_id, engine_id, cycle, dataset_type='test', mc_samples=50):
        """Predict RUL for an engine up to the specified cycle."""
        cache_key = (fd_id, engine_id, cycle, dataset_type)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        eng_df = self.get_engine_data(fd_id, engine_id, dataset_type)
        sub_df = eng_df[eng_df['cycle'] <= cycle]
        if sub_df.empty:
            raise ValueError(f"No sensor data for cycle {cycle}")

        result = self.predict_window(fd_id, sub_df, current_cycle=cycle, mc_samples=mc_samples)
        result['engine_id'] = engine_id

        # Attach ground truth RUL if test set
        if dataset_type == 'test' and result['status'] == 'success':
            true_final_rul = int(self.datasets[fd_id]['rul'].iloc[engine_id - 1]['RUL'])
            max_c = eng_df['cycle'].max()
            true_cur_rul = float(true_final_rul + (max_c - cycle))
            result['ground_truth_rul'] = round(true_cur_rul, 1)
            result['prediction_error'] = round(abs(result['predicted_rul'] - true_cur_rul), 1)

        self._prediction_cache[cache_key] = result
        return result

    def get_engine_rul_trend(self, fd_id, engine_id, dataset_type='test', step=2):
        """
        Generate full historical RUL curve across all valid cycles for the engine.
        Uses cached/batched model inference for rapid client loading.
        """
        fd_id = int(fd_id)
        engine_id = int(engine_id)
        eng_df = self.get_engine_data(fd_id, engine_id, dataset_type)
        W = self.scalers[fd_id]['window_size']
        max_cycle = int(eng_df['cycle'].max())

        trend_points = []
        # Sample every `step` cycles for performance, but always include latest cycle
        cycles_to_eval = list(range(W, max_cycle + 1, step))
        if max_cycle not in cycles_to_eval:
            cycles_to_eval.append(max_cycle)

        for c in cycles_to_eval:
            # Use faster 15-sample MC Dropout for historical curve, 50 for latest
            mc_s = 50 if c == max_cycle else 20
            pred = self.predict_engine_cycle(fd_id, engine_id, c, dataset_type, mc_samples=mc_s)
            if pred['status'] == 'success':
                trend_points.append({
                    "cycle": c,
                    "predicted_rul": pred['predicted_rul'],
                    "uncertainty": pred['uncertainty'],
                    "ci_lower": pred['confidence_interval']['lower'],
                    "ci_upper": pred['confidence_interval']['upper'],
                    "ground_truth_rul": pred.get('ground_truth_rul')
                })

        return {
            "dataset": f"FD00{fd_id}",
            "engine_id": engine_id,
            "window_size": W,
            "max_cycle": max_cycle,
            "warning_threshold": 55,
            "critical_threshold": 25,
            "trend": trend_points
        }

    def predict_fleet_summary(self, fd_id, dataset_type='test', force_refresh=False):
        """
        Predict RUL and operational status for all engines in the dataset at their latest cycle.
        Uses fast batched tensor inference with MC Dropout (T=30).
        """
        fd_id = int(fd_id)
        cache_key = (fd_id, dataset_type)
        if not force_refresh and cache_key in self._fleet_cache:
            return self._fleet_cache[cache_key]

        info = self.scalers[fd_id]
        test_df = self.datasets[fd_id][dataset_type]
        engine_ids = sorted(test_df['engine_id'].unique().tolist())
        W = info['window_size']
        sensors = info['selected_sensors']
        km = info['km']
        op_scaler = info['op_scaler']
        cs = info['cond_scalers']
        hc_scaler = info['hc_scaler']
        RM = info['rul_max']

        # 1. Preprocess all engines
        valid_indices = []
        valid_seqs = []
        valid_hcs = []
        engine_records = []

        for idx, eid in enumerate(engine_ids):
            eng_sub = test_df[test_df['engine_id'] == eid]
            max_c = int(eng_sub['cycle'].max())

            true_rul = None
            if dataset_type == 'test':
                true_final = float(self.datasets[fd_id]['rul'].iloc[eid - 1]['RUL'])
                true_rul = round(true_final, 1)

            if len(eng_sub) < W:
                engine_records.append({
                    "engine_id": eid,
                    "dataset": f"FD00{fd_id}",
                    "current_cycle": max_c,
                    "predicted_rul": None,
                    "uncertainty": None,
                    "confidence_interval": None,
                    "health_status": "INSUFFICIENT_DATA",
                    "priority": "LOW",
                    "alert_type": "INSUFFICIENT_DATA",
                    "alert_label": "⚠️ INSUFFICIENT HISTORY",
                    "action": f"Accumulate Telemetry (Need {W}c)",
                    "estimated_failure_cycle": None,
                    "ground_truth_rul": true_rul
                })
                continue

            win = eng_sub.iloc[-W:].copy()
            for s in sensors:
                win[s] = win[s].astype(np.float64)

            normed_win = np.zeros((W, len(sensors)), dtype=np.float32)
            if km is None:
                normed_win = cs[0].transform(win[sensors]).astype(np.float32)
                detected_cond = 0
            else:
                op_norm = op_scaler.transform(win[OP_SETTINGS].values)
                cond_labels = km.predict(op_norm)
                detected_cond = int(cond_labels[-1])
                for cond in cs:
                    mask = (cond_labels == cond)
                    if mask.any():
                        normed_win[mask] = cs[cond].transform(win.loc[mask, sensors]).astype(np.float32)

            raw_hc = compute_window_hc(normed_win)
            normed_hc = np.clip(hc_scaler.transform([raw_hc]), 0, 1).astype(np.float32)

            valid_indices.append(len(engine_records))
            valid_seqs.append(normed_win)
            valid_hcs.append(normed_hc[0])

            engine_records.append({
                "engine_id": eid,
                "dataset": f"FD00{fd_id}",
                "current_cycle": max_c,
                "predicted_rul": None,
                "uncertainty": None,
                "confidence_interval": None,
                "health_status": None,
                "priority": None,
                "alert_type": None,
                "alert_label": None,
                "action": None,
                "estimated_failure_cycle": None,
                "ground_truth_rul": true_rul,
                "operating_condition_cluster": detected_cond
            })

        # 2. Batched MC Dropout Inference
        if valid_seqs:
            x_seq = torch.tensor(np.array(valid_seqs), dtype=torch.float32).to(self.device)
            x_hc = torch.tensor(np.array(valid_hcs), dtype=torch.float32).to(self.device)
            model = self.models[fd_id]
            model.train()
            with torch.no_grad():
                preds = torch.stack([model(x_seq, x_hc, training=False).squeeze(-1) for _ in range(30)]) * RM
            preds_np = preds.cpu().numpy()  # [30, N_valid]
            means = np.mean(preds_np, axis=0)
            stds = np.std(preds_np, axis=0)
            lo_cis = np.quantile(preds_np, 0.05, axis=0)
            hi_cis = np.quantile(preds_np, 0.95, axis=0)

            for i, rec_idx in enumerate(valid_indices):
                m_rul = round(float(means[i]), 1)
                s_rul = round(float(stds[i]), 1)
                lo_ci = round(float(lo_cis[i]), 1)
                hi_ci = round(float(hi_cis[i]), 1)
                cur_c = engine_records[rec_idx]["current_cycle"]
                est_fail = int(round(cur_c + m_rul))

                # Health status
                if m_rul < 25 or (m_rul < 35 and s_rul > 10):
                    h_status = "CRITICAL"
                    priority = "HIGH"
                    action = "Immediate Overhaul"
                    alert_type = "LOW_RUL"
                    alert_label = "🔴 CRITICAL LOW RUL"
                elif m_rul < 55:
                    h_status = "WARNING"
                    priority = "MEDIUM" if s_rul < 8.0 else "HIGH"
                    action = "Schedule Inspection"
                    alert_type = "LOW_RUL"
                    alert_label = "🟡 LOW RUL WARNING"
                else:
                    h_status = "HEALTHY"
                    if s_rul >= 8.0:
                        priority = "MEDIUM"
                        action = "Telemetry Verification"
                        alert_type = "HIGH_UNCERTAINTY"
                        alert_label = "⚠️ HIGH UNCERTAINTY"
                    else:
                        priority = "LOW"
                        action = "Routine Monitoring"
                        alert_type = "NOMINAL"
                        alert_label = "🟢 NOMINAL"

                rec = engine_records[rec_idx]
                rec["predicted_rul"] = m_rul
                rec["uncertainty"] = s_rul
                rec["confidence_interval"] = {"lower": lo_ci, "upper": hi_ci}
                rec["health_status"] = h_status
                rec["priority"] = priority
                rec["alert_type"] = alert_type
                rec["alert_label"] = alert_label
                rec["action"] = action
                rec["estimated_failure_cycle"] = est_fail

                # Seed cache for single-engine lookups
                eng_cache_key = (fd_id, rec["engine_id"], cur_c, dataset_type)
                if eng_cache_key not in self._prediction_cache:
                    self._prediction_cache[eng_cache_key] = {
                        "status": "success",
                        "dataset": f"FD00{fd_id}",
                        "engine_id": rec["engine_id"],
                        "current_cycle": cur_c,
                        "estimated_failure_cycle": est_fail,
                        "predicted_rul": m_rul,
                        "uncertainty": s_rul,
                        "confidence_interval": {"lower": lo_ci, "upper": hi_ci},
                        "health_status": h_status,
                        "operating_condition_cluster": rec.get("operating_condition_cluster", 0),
                        "mc_samples": 30,
                        "alerts": [
                            {"type": alert_type, "severity": "critical" if h_status == "CRITICAL" else "warning", "title": alert_label, "message": f"{action} for Engine {rec['engine_id']}"}
                        ] if alert_type != "NOMINAL" else [],
                        "maintenance_recommendation": {
                            "action": action,
                            "priority": priority,
                            "recommended_window": f"Within 0 - {max(1, int(m_rul * 0.5))} cycles" if h_status == "CRITICAL" else f"Next check (~{int(m_rul * 0.75)} cycles)",
                            "status": "Urgent Action Required" if h_status == "CRITICAL" else "Operational"
                        },
                        "ground_truth_rul": rec.get("ground_truth_rul")
                    }

        # 3. Sort: CRITICAL first, then WARNING, then HEALTHY, then INSUFFICIENT_DATA
        # Within each category, sort by lowest predicted RUL first
        priority_order = {"CRITICAL": 0, "WARNING": 1, "HEALTHY": 2, "INSUFFICIENT_DATA": 3}
        engine_records.sort(key=lambda x: (
            priority_order.get(x['health_status'], 4),
            x['predicted_rul'] if x['predicted_rul'] is not None else 9999
        ))

        # Assign Rank
        for rank_idx, r in enumerate(engine_records, start=1):
            r["rank"] = rank_idx

        # Compute summary counts
        crit_count = sum(1 for e in engine_records if e['health_status'] == 'CRITICAL')
        warn_count = sum(1 for e in engine_records if e['health_status'] == 'WARNING')
        hlth_count = sum(1 for e in engine_records if e['health_status'] == 'HEALTHY')
        uncert_count = sum(1 for e in engine_records if e.get('uncertainty') is not None and e['uncertainty'] >= 8.0)

        attention_req = [e for e in engine_records if e['health_status'] in ['CRITICAL', 'WARNING']][:5]
        if not attention_req:
            attention_req = engine_records[:5]

        result = {
            "dataset": f"FD00{fd_id}",
            "total_engines": len(engine_ids),
            "critical_count": crit_count,
            "warning_count": warn_count,
            "healthy_count": hlth_count,
            "high_uncertainty_count": uncert_count,
            "attention_required": attention_req,
            "fleet": engine_records
        }
        self._fleet_cache[cache_key] = result
        return result

    def predict_all_fleets_summary(self, dataset_type='test', force_refresh=False):
        """
        Predict and aggregate fleet status across ALL 4 C-MAPSS datasets (FD001–FD004).
        Produces a unified operational fleet priority ranking with the most critical engines at the top.
        """
        cache_key = ("ALL", dataset_type)
        if not force_refresh and cache_key in self._fleet_cache:
            return self._fleet_cache[cache_key]

        all_engines = []
        for fd_id in range(1, 5):
            res = self.predict_fleet_summary(fd_id, dataset_type=dataset_type, force_refresh=force_refresh)
            all_engines.extend(res['fleet'])

        # Unified sorting: Critical first, then lowest predicted RUL
        priority_order = {"CRITICAL": 0, "WARNING": 1, "HEALTHY": 2, "INSUFFICIENT_DATA": 3}
        all_engines.sort(key=lambda x: (
            priority_order.get(x['health_status'], 4),
            x['predicted_rul'] if x['predicted_rul'] is not None else 9999
        ))

        # Re-assign global ranks
        for rank_idx, r in enumerate(all_engines, start=1):
            r["rank"] = rank_idx

        crit_count = sum(1 for e in all_engines if e['health_status'] == 'CRITICAL')
        warn_count = sum(1 for e in all_engines if e['health_status'] == 'WARNING')
        hlth_count = sum(1 for e in all_engines if e['health_status'] == 'HEALTHY')
        uncert_count = sum(1 for e in all_engines if e.get('uncertainty') is not None and e['uncertainty'] >= 8.0)

        attention_req = [e for e in all_engines if e['health_status'] in ['CRITICAL', 'WARNING']][:5]
        if not attention_req:
            attention_req = all_engines[:5]

        result = {
            "dataset": "ALL",
            "total_engines": len(all_engines),
            "critical_count": crit_count,
            "warning_count": warn_count,
            "healthy_count": hlth_count,
            "high_uncertainty_count": uncert_count,
            "attention_required": attention_req,
            "fleet": all_engines
        }
        self._fleet_cache[cache_key] = result
        return result



# Singleton engine instance
_ENGINE_INSTANCE = None

def get_engine():
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = RULInferenceEngine()
    return _ENGINE_INSTANCE
