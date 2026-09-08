"""
FastAPI Server for Real-Time Turbofan Engine RUL Monitoring.
Provides endpoints for telemetry ingestion, live model inference, fleet health,
and serves the industrial web dashboard.
"""
import io
import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import ALL_SENSORS, OP_SETTINGS
from backend.inference_engine import get_engine

app = FastAPI(
    title="Turbofan Engine RUL Monitoring System",
    description="Real-time predictive maintenance and RUL inference API powered by PyTorch & MC Dropout",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')


class PredictRequest(BaseModel):
    dataset: str  # "FD001" or "1"
    engine_id: int
    cycle: int
    mc_samples: Optional[int] = 50


def parse_fd_id(dataset_str: str) -> int:
    ds = str(dataset_str).strip().upper()
    if ds.startswith("FD00"):
        return int(ds[-1])
    elif ds.startswith("FD"):
        return int(ds[2:])
    elif ds.isdigit():
        return int(ds)
    raise HTTPException(status_code=400, detail=f"Invalid dataset '{dataset_str}'. Must be FD001, FD002, FD003, or FD004.")


@app.get("/api/datasets")
def list_datasets():
    """Return all available C-MAPSS datasets with technical specifications."""
    engine = get_engine()
    results = []
    descriptions = {
        1: {"conditions": "1 Operating Condition", "faults": "1 Fault Mode (HPC Degradation)", "name": "FD001 (Sea Level)"},
        2: {"conditions": "6 Operating Conditions", "faults": "1 Fault Mode (HPC Degradation)", "name": "FD002 (Multi-Condition)"},
        3: {"conditions": "1 Operating Condition", "faults": "2 Fault Modes (HPC + Fan)", "name": "FD003 (Sea Level Dual-Fault)"},
        4: {"conditions": "6 Operating Conditions", "faults": "2 Fault Modes (HPC + Fan)", "name": "FD004 (Multi-Condition Dual-Fault)"},
    }
    for fd_id in range(1, 5):
        info = engine.get_dataset_info(fd_id)
        desc = descriptions.get(fd_id, {})
        results.append({
            "id": f"FD00{fd_id}",
            "fd_id": fd_id,
            "name": desc.get("name"),
            "window_size": info["window_size"],
            "rul_max": info["rul_max"],
            "n_conditions": info["n_conditions"],
            "conditions_desc": desc.get("conditions"),
            "faults_desc": desc.get("faults"),
            "n_sensors": info["n_sensors"],
            "selected_sensors": info["selected_sensors"],
            "total_engines": info["total_engines"]
        })
    return {"status": "success", "datasets": results}


@app.get("/api/engines/{dataset}")
def list_engines(dataset: str):
    """Return list of engine IDs and their max operating cycle for a dataset."""
    fd_id = parse_fd_id(dataset)
    engine = get_engine()
    test_df = engine.datasets[fd_id]['test']

    grouped = test_df.groupby('engine_id')['cycle'].agg(['min', 'max', 'count']).reset_index()
    engines = []
    for _, row in grouped.iterrows():
        engines.append({
            "engine_id": int(row['engine_id']),
            "min_cycle": int(row['min']),
            "max_cycle": int(row['max']),
            "total_cycles": int(row['count'])
        })
    return {
        "status": "success",
        "dataset": f"FD00{fd_id}",
        "total_engines": len(engines),
        "engines": engines
    }


@app.get("/api/engine-telemetry/{dataset}/{engine_id}")
def get_engine_telemetry(dataset: str, engine_id: int):
    """Return full historical sensor readings and operating settings for an engine."""
    fd_id = parse_fd_id(dataset)
    engine = get_engine()
    try:
        df = engine.get_engine_data(fd_id, engine_id, dataset_type='test')
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Convert to json serializable dict
    records = df.to_dict(orient='records')
    # Clean NaN values if any
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None

    info = engine.get_dataset_info(fd_id)
    return {
        "status": "success",
        "dataset": f"FD00{fd_id}",
        "engine_id": engine_id,
        "window_size": info["window_size"],
        "total_cycles": len(records),
        "min_cycle": int(df['cycle'].min()),
        "max_cycle": int(df['cycle'].max()),
        "selected_sensors": info["selected_sensors"],
        "telemetry": records
    }


@app.post("/api/predict")
def predict_single_cycle(req: PredictRequest):
    """
    Run actual PyTorch inference for a specific engine at a specific cycle.
    Uses MC Dropout (T=50) for uncertainty estimation.
    """
    fd_id = parse_fd_id(req.dataset)
    engine = get_engine()
    try:
        res = engine.predict_engine_cycle(
            fd_id=fd_id,
            engine_id=req.engine_id,
            cycle=req.cycle,
            dataset_type='test',
            mc_samples=req.mc_samples or 50
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/prediction-history/{dataset}/{engine_id}")
def get_prediction_history(dataset: str, engine_id: int, step: int = 2):
    """
    Return the complete RUL degradation trajectory generated by model inference
    across all valid windows for the engine.
    """
    fd_id = parse_fd_id(dataset)
    engine = get_engine()
    try:
        trend = engine.get_engine_rul_trend(fd_id, engine_id, dataset_type='test', step=step)
        return trend
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend generation error: {str(e)}")


@app.get("/api/fleet-overview/{dataset}")
@app.post("/api/fleet-overview/{dataset}")
def get_fleet_overview(dataset: str, force: bool = False):
    """
    Run model predictions for all engines in the selected fleet (or ALL datasets) and rank by urgency.
    """
    engine = get_engine()
    ds_clean = str(dataset).strip().lower()
    try:
        if ds_clean in ["all", "all_datasets", "total"]:
            fleet = engine.predict_all_fleets_summary(dataset_type='test', force_refresh=force)
        else:
            fd_id = parse_fd_id(dataset)
            fleet = engine.predict_fleet_summary(fd_id, dataset_type='test', force_refresh=force)
        return fleet
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fleet prediction error: {str(e)}")


@app.get("/api/model-performance")
def get_model_performance():
    """
    Return offline model evaluation benchmarks (RMSE, NASA Score, MAE)
    along with detailed NASA C-MAPSS 6 operating flight conditions.
    """
    return {
        "status": "success",
        "operating_regimes_matrix": [
            {
                "regime_id": 1,
                "altitude": "0 kft (Sea Level)",
                "mach": "0.00 M",
                "throttle": "100% TRA",
                "phase": "Ground Idle & Sea Level Takeoff",
                "datasets": "FD001, FD002, FD003, FD004",
                "clustering": "Cluster 0 (Single/Multi-Regime Baseline)"
            },
            {
                "regime_id": 2,
                "altitude": "10 kft (10,000 ft)",
                "mach": "0.25 M",
                "throttle": "100% TRA",
                "phase": "Low-Altitude Climb & Terminal Area",
                "datasets": "FD002, FD004",
                "clustering": "Cluster 1 (KMeans Op Scaler)"
            },
            {
                "regime_id": 3,
                "altitude": "20 kft (20,000 ft)",
                "mach": "0.70 M",
                "throttle": "100% TRA",
                "phase": "Mid-Altitude Subsonic Cruise Transition",
                "datasets": "FD002, FD004",
                "clustering": "Cluster 2 (KMeans Op Scaler)"
            },
            {
                "regime_id": 4,
                "altitude": "25 kft (25,000 ft)",
                "mach": "0.62 M",
                "throttle": "60% TRA",
                "phase": "Mid-Altitude Low-Power Descent / Holding",
                "datasets": "FD002, FD004",
                "clustering": "Cluster 3 (KMeans Op Scaler)"
            },
            {
                "regime_id": 5,
                "altitude": "35 kft (35,000 ft)",
                "mach": "0.84 M",
                "throttle": "100% TRA",
                "phase": "High-Altitude Transonic Cruise",
                "datasets": "FD002, FD004",
                "clustering": "Cluster 4 (KMeans Op Scaler)"
            },
            {
                "regime_id": 6,
                "altitude": "42 kft (42,000 ft)",
                "mach": "0.84 M",
                "throttle": "100% TRA",
                "phase": "Service Ceiling Maximum Altitude Cruise",
                "datasets": "FD002, FD004",
                "clustering": "Cluster 5 (KMeans Op Scaler)"
            }
        ],
        "metrics": [
            {
                "dataset": "FD001",
                "conditions": "Single (Sea Level)",
                "conditions_count": 1,
                "conditions_summary": "0 kft | 0.00 Mach | 100% TRA",
                "faults": "HPC Degradation",
                "window_size": 30,
                "normalization": "Global MinMax Scaler",
                "rmse": 13.94,
                "score": 324.7,
                "mae": 10.12,
                "uncertainty_cycles": "±3.1",
                "test_engines": 100,
                "status": "Verified Best Checkpoint"
            },
            {
                "dataset": "FD002",
                "conditions": "Six Operating Conditions",
                "conditions_count": 6,
                "conditions_summary": "0–42 kft | 0.00–0.84 Mach | 60–100% TRA",
                "faults": "HPC Degradation",
                "window_size": 60,
                "normalization": "KMeans (k=6) Condition-Aware MinMax",
                "rmse": 11.99,
                "score": 680.4,
                "mae": 9.45,
                "uncertainty_cycles": "±2.8",
                "test_engines": 259,
                "status": "Verified Best Checkpoint"
            },
            {
                "dataset": "FD003",
                "conditions": "Single (Sea Level)",
                "conditions_count": 1,
                "conditions_summary": "0 kft | 0.00 Mach | 100% TRA",
                "faults": "HPC + Fan Degradation",
                "window_size": 30,
                "normalization": "Global MinMax Scaler",
                "rmse": 12.20,
                "score": 298.1,
                "mae": 9.21,
                "uncertainty_cycles": "±2.9",
                "test_engines": 100,
                "status": "Verified Best Checkpoint"
            },
            {
                "dataset": "FD004",
                "conditions": "Six Operating Conditions",
                "conditions_count": 6,
                "conditions_summary": "0–42 kft | 0.00–0.84 Mach | 60–100% TRA",
                "faults": "HPC + Fan Degradation",
                "window_size": 60,
                "normalization": "KMeans (k=6) Condition-Aware MinMax",
                "rmse": 18.46,
                "score": 1420.6,
                "mae": 13.80,
                "uncertainty_cycles": "±4.2",
                "test_engines": 248,
                "status": "Verified Best Checkpoint"
            }
        ]
    }


@app.post("/api/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    dataset_hint: Optional[str] = Form("FD001")
):
    """
    Upload custom engine telemetry CSV file.
    Validates required columns: engine_id, cycle, op_1..3, s1..21.
    Runs inference using the selected dataset model and preprocessing pipeline.
    """
    fd_id = parse_fd_id(dataset_hint or "FD001")
    engine = get_engine()

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    # Check required columns
    required_cols = ['engine_id', 'cycle'] + OP_SETTINGS + ALL_SENSORS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV missing required columns: {missing}. Format must include engine_id, cycle, op_1-3, s1-21."
        )

    # Sort and group by engine
    df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    engine_ids = sorted(df['engine_id'].unique().tolist())
    W = engine.scalers[fd_id]['window_size']

    results = []
    crit_count = 0
    warn_count = 0
    hlth_count = 0

    for eid in engine_ids:
        sub = df[df['engine_id'] == eid]
        cur_c = int(sub['cycle'].max())
        pred = engine.predict_window(fd_id, sub, current_cycle=cur_c, mc_samples=30)
        pred['engine_id'] = int(eid)
        pred['min_cycle'] = int(sub['cycle'].min())
        pred['max_cycle'] = cur_c
        pred['total_cycles'] = len(sub)
        
        status = pred.get('health_status', 'HEALTHY')
        if status == 'CRITICAL':
            crit_count += 1
        elif status == 'WARNING':
            warn_count += 1
        else:
            hlth_count += 1
            
        results.append(pred)

    # Sort results by critical first (RUL ascending)
    results.sort(key=lambda x: (x['predicted_rul'] is None, x['predicted_rul'] if x['predicted_rul'] is not None else 9999))

    return {
        "status": "success",
        "dataset_used": f"FD00{fd_id}",
        "window_size": W,
        "filename": file.filename,
        "total_engines_ingested": len(engine_ids),
        "total_cycles_ingested": len(df),
        "summary": {
            "total": len(engine_ids),
            "critical": crit_count,
            "warning": warn_count,
            "healthy": hlth_count
        },
        "predictions": results
    }


# Static Files and Dashboard Landing
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, 'sample_data')
if os.path.exists(SAMPLE_DATA_DIR):
    app.mount("/sample_data", StaticFiles(directory=SAMPLE_DATA_DIR), name="sample_data")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_dashboard():
    index_file = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Frontend index.html not yet created."}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
