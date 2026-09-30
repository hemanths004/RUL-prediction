"""
SHAP-based AI Explainability Module for Turbofan Engine RUL Predictions.
Explains the multi-input PyTorch architecture (Multi-Scale CNN + BiLSTM + 3D Attention + MoE)
using GradientExplainer with dataset-aware background baselines and sensor-level aggregation.
"""
import os
import sys
import numpy as np
import torch
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OP_SETTINGS
from dataset import compute_window_hc

SENSOR_META = {
    's1':  {'name': 'Fan Inlet Temperature', 'unit': '°R', 'desc': 'Total temperature at fan inlet'},
    's2':  {'name': 'LPC Outlet Temperature', 'unit': '°R', 'desc': 'Total temperature at LPC outlet'},
    's3':  {'name': 'HPC Outlet Temperature', 'unit': '°R', 'desc': 'Total temperature at HPC outlet'},
    's4':  {'name': 'LPT Outlet Temperature', 'unit': '°R', 'desc': 'Total temperature at LPT outlet'},
    's5':  {'name': 'Fan Inlet Pressure', 'unit': 'psia', 'desc': 'Pressure at fan inlet'},
    's6':  {'name': 'Bypass Duct Pressure', 'unit': 'psia', 'desc': 'Total pressure in bypass-duct'},
    's7':  {'name': 'HPC Outlet Pressure', 'unit': 'psia', 'desc': 'Total pressure at HPC outlet'},
    's8':  {'name': 'Physical Fan Speed', 'unit': 'rpm', 'desc': 'Physical fan rotational speed'},
    's9':  {'name': 'Physical Core Speed', 'unit': 'rpm', 'desc': 'Physical core rotational speed'},
    's10': {'name': 'Engine Pressure Ratio', 'unit': '--', 'desc': 'EPR (P50/P2)'},
    's11': {'name': 'HPC Static Pressure', 'unit': 'psia', 'desc': 'Static pressure at HPC outlet'},
    's12': {'name': 'Fuel Flow Ratio', 'unit': 'pps/psia', 'desc': 'Ratio of fuel flow to Ps30'},
    's13': {'name': 'Corrected Fan Speed', 'unit': 'rpm', 'desc': 'Corrected fan speed'},
    's14': {'name': 'Corrected Core Speed', 'unit': 'rpm', 'desc': 'Corrected core speed'},
    's15': {'name': 'Bypass Ratio', 'unit': '--', 'desc': 'Bypass ratio'},
    's16': {'name': 'Burner Fuel-Air Ratio', 'unit': '--', 'desc': 'Burner fuel-air ratio'},
    's17': {'name': 'Bleed Enthalpy', 'unit': '--', 'desc': 'Bleed enthalpy'},
    's18': {'name': 'Demanded Fan Speed', 'unit': 'rpm', 'desc': 'Demanded fan speed'},
    's19': {'name': 'Demanded Corr. Fan Speed', 'unit': 'rpm', 'desc': 'Demanded corrected fan speed'},
    's20': {'name': 'HPT Coolant Bleed', 'unit': 'lbm/s', 'desc': 'HPT coolant bleed'},
    's21': {'name': 'LPT Coolant Bleed', 'unit': 'lbm/s', 'desc': 'LPT coolant bleed'}
}


class TurbofanSHAPExplainer:
    """
    Manages SHAP GradientExplainers and background samples across C-MAPSS datasets (FD001–FD004).
    Computes exact, signed sensor-level contributions that explain the current RUL prediction.
    """

    def __init__(self, inference_engine):
        self.engine = inference_engine
        self.explainers = {}
        self.bg_data = {}
        self.base_values = {}
        self._explanation_cache = {}
        self.device = self.engine.device

    def _prepare_background(self, fd_id: int, n_samples: int = 15):
        """Build representative baseline background samples for a dataset."""
        scaler_info = self.engine.scalers[fd_id]
        W = scaler_info['window_size']
        sensors = scaler_info['selected_sensors']
        cs = scaler_info['cond_scalers']
        km = scaler_info['km']
        op_scaler = scaler_info['op_scaler']
        hc_scaler = scaler_info['hc_scaler']
        test_df = self.engine.datasets[fd_id]['test']

        bg_windows = []
        bg_hcs = []
        engine_ids = sorted(test_df['engine_id'].unique())

        for eid in engine_ids:
            sub = test_df[test_df['engine_id'] == eid].sort_values('cycle')
            if len(sub) >= W:
                w_df = sub.iloc[-W:].copy()
                normed_w = np.zeros((W, len(sensors)), dtype=np.float32)
                if km is None:
                    normed_w = cs[0].transform(w_df[sensors]).astype(np.float32)
                else:
                    op_norm = op_scaler.transform(w_df[OP_SETTINGS].values)
                    cond_lbls = km.predict(op_norm)
                    for cond in cs:
                        mask = (cond_lbls == cond)
                        if mask.any():
                            normed_w[mask] = cs[cond].transform(w_df.loc[mask, sensors]).astype(np.float32)

                raw_h = compute_window_hc(normed_w)
                norm_h = np.clip(hc_scaler.transform([raw_h]), 0, 1).astype(np.float32)[0]

                bg_windows.append(normed_w)
                bg_hcs.append(norm_h)
                if len(bg_windows) >= n_samples:
                    break

        bg_seq = torch.tensor(np.array(bg_windows), dtype=torch.float32, device=self.device)
        bg_hc = torch.tensor(np.array(bg_hcs), dtype=torch.float32, device=self.device)
        return bg_seq, bg_hc

    def get_explainer(self, fd_id: int):
        """Retrieve or lazily initialize the SHAP GradientExplainer for dataset fd_id."""
        if fd_id in self.explainers:
            return self.explainers[fd_id]

        fd_id = int(fd_id)
        model = self.engine.models[fd_id]
        model.eval()

        bg_seq, bg_hc = self._prepare_background(fd_id)
        self.bg_data[fd_id] = (bg_seq, bg_hc)

        # Base expected value across background baseline
        with torch.no_grad():
            raw_base = model(bg_seq, bg_hc, training=False).mean().item()
            RM = self.engine.scalers[fd_id]['rul_max']
            self.base_values[fd_id] = float(raw_base * RM)

        explainer = shap.GradientExplainer(model, [bg_seq, bg_hc])
        self.explainers[fd_id] = explainer
        return explainer

    def explain_engine(self, fd_id: int, engine_id: int, cycle: int = None, dataset_type: str = 'test'):
        """
        Compute real SHAP feature contributions explaining the RUL prediction for an engine at a given cycle.
        """
        fd_id = int(fd_id)
        engine_id = int(engine_id)

        eng_df = self.engine.get_engine_data(fd_id, engine_id, dataset_type)
        if cycle is None:
            cycle = int(eng_df['cycle'].max())
        else:
            cycle = int(cycle)

        cache_key = (fd_id, engine_id, cycle, dataset_type)
        if cache_key in self._explanation_cache:
            return self._explanation_cache[cache_key]

        scaler_info = self.engine.scalers[fd_id]
        W = scaler_info['window_size']
        RM = scaler_info['rul_max']
        sensors = scaler_info['selected_sensors']
        cs = scaler_info['cond_scalers']
        km = scaler_info['km']
        op_scaler = scaler_info['op_scaler']
        hc_scaler = scaler_info['hc_scaler']

        sub_df = eng_df[eng_df['cycle'] <= cycle]
        if len(sub_df) < W:
            return {
                "status": "insufficient_history",
                "message": f"Insufficient history for explainability. Minimum {W} cycles required.",
                "dataset": f"FD00{fd_id}",
                "engine_id": engine_id,
                "current_cycle": cycle
            }

        # 1. Prepare exact window inputs identically to inference
        window = sub_df.iloc[-W:].copy()
        for s in sensors:
            window[s] = window[s].astype(np.float64)

        normed_win = np.zeros((W, len(sensors)), dtype=np.float32)
        if km is None:
            normed_win = cs[0].transform(window[sensors]).astype(np.float32)
        else:
            op_norm = op_scaler.transform(window[OP_SETTINGS].values)
            cond_labels = km.predict(op_norm)
            for cond in cs:
                mask = (cond_labels == cond)
                if mask.any():
                    normed_win[mask] = cs[cond].transform(window.loc[mask, sensors]).astype(np.float32)

        raw_hc = compute_window_hc(normed_win)
        normed_hc = np.clip(hc_scaler.transform([raw_hc]), 0, 1).astype(np.float32)

        x_seq = torch.tensor(normed_win[np.newaxis, ...], dtype=torch.float32, device=self.device)
        x_hc = torch.tensor(normed_hc, dtype=torch.float32, device=self.device)

        # 2. Get the actual prediction from the inference engine / cache
        pred_res = self.engine.predict_engine_cycle(fd_id, engine_id, cycle, dataset_type=dataset_type)
        predicted_rul = pred_res.get('predicted_rul', 0.0)

        # 3. Compute SHAP values with GradientExplainer
        explainer = self.get_explainer(fd_id)
        shap_vals = explainer.shap_values([x_seq, x_hc])

        seq_shap = np.array(shap_vals[0]).squeeze() * RM  # Shape: (W, n_sensors)
        hc_shap = np.array(shap_vals[1]).squeeze() * RM   # Shape: (n_sensors * 6,)

        base_val = self.base_values.get(fd_id, float(RM * 0.5))

        # 4. Temporal and Multi-Input Aggregation to Sensor-Level
        sensor_shap_list = []
        total_abs_contrib = 0.0

        latest_row = window.iloc[-1]

        for i, s_id in enumerate(sensors):
            # Sum temporal contributions over window W
            temporal_sum = float(np.sum(seq_shap[:, i]))
            # Mean temporal contribution across timesteps
            temporal_mean = float(np.mean(seq_shap[:, i]))
            # Sum handcrafted statistical features (6 per sensor)
            hc_sum = float(np.sum(hc_shap[i * 6 : (i + 1) * 6]))

            # Total signed SHAP contribution in RUL cycles
            total_shap = temporal_sum + hc_sum
            total_abs_contrib += abs(total_shap)

            cur_raw_val = float(latest_row[s_id])
            s_meta = SENSOR_META.get(s_id, {"name": s_id, "unit": ""})

            effect = "higher_rul" if total_shap >= 0 else "lower_rul"
            effect_text = "Increases predicted RUL" if total_shap >= 0 else "Decreases predicted RUL"
            effect_short = "Higher RUL" if total_shap >= 0 else "Lower RUL"

            sensor_shap_list.append({
                "sensor": s_id.upper(),
                "sensor_id": s_id,
                "name": s_meta["name"],
                "unit": s_meta["unit"],
                "current_value": round(cur_raw_val, 2),
                "shap_value": round(total_shap, 3),
                "temporal_shap": round(temporal_sum, 3),
                "handcrafted_shap": round(hc_sum, 3),
                "abs_shap": round(abs(total_shap), 3),
                "effect": effect,
                "effect_text": effect_text,
                "effect_short": effect_short,
                "is_risk_increasing": total_shap < 0
            })

        # Calculate relative impact percentage
        for item in sensor_shap_list:
            if total_abs_contrib > 1e-6:
                item["impact_pct"] = round((item["abs_shap"] / total_abs_contrib) * 100, 1)
            else:
                item["impact_pct"] = round(100.0 / len(sensors), 1)

        # Sort all sensors by absolute SHAP value descending
        sensor_shap_list.sort(key=lambda x: x["abs_shap"], reverse=True)

        # Top Positive (RUL-supporting) and Top Negative (Risk-increasing)
        top_positive = [s for s in sensor_shap_list if s["shap_value"] > 0]
        top_positive.sort(key=lambda x: x["shap_value"], reverse=True)

        top_negative = [s for s in sensor_shap_list if s["shap_value"] < 0]
        top_negative.sort(key=lambda x: x["shap_value"])  # Most negative first

        # Top contributing (Top 10 for bar chart)
        top_10 = sensor_shap_list[:10]

        # 5. Narrative Explanation Summary
        top_3_names = [s["sensor"] for s in sensor_shap_list[:3]]
        top_risk_names = [s["sensor"] for s in top_negative[:3]]
        top_supp_names = [s["sensor"] for s in top_positive[:3]]

        narrative = f"The current RUL prediction of {predicted_rul:.0f} cycles is primarily influenced by {', '.join(top_3_names)}."
        if top_risk_names:
            narrative += f" Degradation is primarily driven by {', '.join(top_risk_names)}, which push the prediction toward lower RUL (higher maintenance risk)."
        if top_supp_names:
            narrative += f" In contrast, {', '.join(top_supp_names)} contribute positively toward higher predicted RUL."

        result = {
            "status": "success",
            "dataset": f"FD00{fd_id}",
            "engine_id": engine_id,
            "current_cycle": cycle,
            "window_size": W,
            "rul_prediction": predicted_rul,
            "uncertainty": pred_res.get("uncertainty"),
            "health_status": pred_res.get("health_status"),
            "base_value": round(base_val, 2),
            "summary_narrative": narrative,
            "top_contributing": top_10,
            "all_sensors": sensor_shap_list,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "metadata": {
                "explainer_type": "GradientExplainer",
                "temporal_aggregation": "sum_across_window",
                "handcrafted_aggregation": "sum_across_6_features"
            }
        }

        self._explanation_cache[cache_key] = result
        return result
