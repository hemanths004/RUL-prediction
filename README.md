# 🔧 Turbofan Engine RUL Prediction

A deep learning pipeline for **Remaining Useful Life (RUL) prediction** of turbofan engines using the NASA C-MAPSS benchmark dataset. The system uses a hybrid **CNN-BiLSTM with 3D Attention + Mixture of Experts (MoE)** architecture with **Monte Carlo (MC) Dropout** for uncertainty quantification.

---

## 📌 Project Description

Predicting when an aircraft engine will fail is a critical problem in **predictive maintenance**. This project tackles it by estimating the **Remaining Useful Life (RUL)** — the number of operational cycles remaining before failure — for each engine in the C-MAPSS dataset.

### Problem Solved

Traditional threshold-based maintenance strategies lead to either **premature replacements** (costly) or **unexpected failures** (dangerous). This system provides:
- Accurate cycle-level RUL estimates
- Calibrated **uncertainty bounds** (5th–95th percentile confidence intervals)
- Per-dataset tuned models covering single and multi-condition operating regimes

---

## ✨ Features

- **Multi-scale CNN** — parallel convolutions with kernel sizes 3, 5, 7 for capturing short- and long-term degradation patterns
- **Bidirectional LSTM** — 2-layer BiLSTM (hidden=128) for sequential temporal modelling
- **3D Attention + AttentionPool** — highlights the most degradation-informative time steps
- **True Mixture of Experts (MoE)** — 4 specialist expert MLPs with entropy regularisation to prevent expert collapse
- **Handcrafted Features** — mean, slope, std, % deviation, CUSUM, late-slope per sensor window
- **MC Dropout Inference** — T=50 stochastic forward passes for uncertainty estimation
- **Condition-aware Normalisation** — KMeans-based clustering + per-condition MinMaxScaling for multi-regime datasets (FD002, FD004)
- **Piecewise-linear RUL** — capped RUL target for stable training
- **Early Stopping + LR Scheduling** — Linear warmup (5 epochs) → Cosine annealing
- **Full EDA Suite** — missing value analysis, duplicate checks, sensor degradation plots
- **Prediction Export** — RUL predictions saved to `.txt` files per dataset

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.7 (CUDA 11.8) |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Visualisation** | Matplotlib, Seaborn |
| **Dataset** | NASA C-MAPSS (FD001–FD004) |
| **Hardware** | CUDA GPU (CPU fallback supported) |

---

## 📁 Project Structure

```
RUL-prediction/
│
├── main.py                   # Entry point — orchestrates model training pipeline
├── config.py                 # Hyperparameters, sensor configs, device setup
│
├── backend/                  # Real-time FastAPI backend & inference engine
│   ├── server.py             # FastAPI REST API & telemetry ingestion routes
│   ├── inference_engine.py   # MC Dropout inference engine & uncertainty estimator
│   └── scaler_manager.py     # Condition scaler loader & manager
│
├── frontend/                 # Real-time industrial web dashboard
│   ├── index.html            # Fleet health monitoring & sensor analytics UI
│   ├── css/style.css         # Glassmorphism dark theme styling
│   └── js/dashboard.js       # Live charts, telemetry parser, & API client
│
├── model_FD001–FD004.pt      # Pretrained PyTorch model weights
├── scalers/all_scalers.pkl   # Condition-aware KMeans & MinMax scalers
├── sample_data/              # Sample CSV datasets for dashboard uploads
│
├── model.py                  # Neural network: Multi-Scale CNN + BiLSTM + 3DAttn + MoE
├── dataset.py                # PyTorch Dataset (sliding window + 6 HC features)
├── data_processing.py        # Piecewise RUL & KMeans condition normalization
├── data_loading.py           # Loads FD001–FD004 raw data & selects informative sensors
├── train.py                  # Training loop, linear warmup + cosine scheduler
├── evaluate.py               # MC Dropout evaluation (T=50 stochastic passes)
├── metrics.py                # RMSE and NASA asymmetric score functions
├── visualize.py              # Training curve & uncertainty band visualizers
│
├── dashboard/                # Plotly/Dash analytics dashboard
│   ├── app.py                # Dash entry point
│   ├── components.py         # Visual components & charts
│   └── data_loader.py        # Results loader
│
└── data/                     # NASA C-MAPSS raw benchmark files
    ├── train_FD001–FD004.txt
    ├── test_FD001–FD004.txt
    └── RUL_FD001–FD004.txt
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hemanths004/RUL-prediction.git
cd RUL-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Verify GPU Availability

Before installing dependencies, check whether your machine has a CUDA-capable GPU:

```bash
python test_gpu.py
```

Read the output:
- ✅ **CUDA is available** — proceed with the GPU installation below
- ❌ **No CUDA / CPU only** — proceed with the CPU installation below

### 4. Install Dependencies

**Option A — GPU (CUDA 11.8, recommended for faster training):**
```bash
pip install torch==2.7.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn
```

**Option B — CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or install everything at once using the requirements file (GPU build):
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Full Pipeline

```bash
python main.py
```

This executes the pipeline steps automatically:

1. Loads datasets and selects informative sensors
2. Runs EDA (missing values, duplicates, degradation plots)
3. Preprocesses data (RUL labels, condition normalisation)
4. Creates PyTorch DataLoaders with sliding windows
5. Trains models for all 4 FD datasets
6. Checks MoE expert utilisation balance
7. Runs MC Dropout evaluation (T=50 samples)
8. Plots results with uncertainty bands
9. Prints final summary, saves model weights and RUL predictions

### 🖥️ Training on CPU

The pipeline runs on CPU automatically if no GPU is detected. However, training is significantly slower.


### Output Files

| Path | Description |
|---|---|
| `model_FD001.pt` – `model_FD004.pt` | Trained model weights (saved in project folder) |
| `predictions/remaininguselife_fd00X.txt` | Predicted RUL values |
| `eda_fd00X.png` | EDA distribution plots |
| `eda_degradation_fd00X.png` | Sensor degradation curves |
| `rul_results.png` | RUL prediction vs ground truth with CI |
---

## 🌐 Real-Time Monitoring Web App & REST API

The project features a **FastAPI backend** paired with an aerospace-grade **real-time industrial web dashboard**:

### Launch the Web App
```bash
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
```
- **Web Dashboard:** [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI / Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Web App Capabilities
- **Fleet-wide Health Telemetry:** Live health status indicators (`HEALTHY`, `WARNING`, `CRITICAL`)
- **Interactive Degradation Curves:** Multi-sensor time series tracking with zoom & pan
- **Uncertainty Quantification:** Visual 90% confidence bands (5th–95th percentile) via MC Dropout
- **Telemetry CSV File Upload:** Ingest custom sensor telemetry and receive instantaneous batch predictions
- **RESTful Endpoints:** `/api/predict`, `/api/datasets`, `/api/engine/{fd_id}/{engine_id}`, and `/api/upload`

---

## 📊 Interactive Dashboard

The project includes a **Dash/Plotly** interactive dashboard for exploring model results, metrics, and EDA visualisations — all in your browser.

### Launch the Dashboard

```bash
python -m dashboard.app
```

Then open **http://127.0.0.1:8050** in your browser.

> **Note:** The dashboard automatically runs live inference on first launch if no cached results exist. This requires trained model weights (`model_FD001.pt` – `model_FD004.pt`) in the project root.

### Dashboard Tabs

| Tab | Description |
|---|---|
| **Overview** | KPI cards (Avg RMSE, Score, MAE, Uncertainty), performance summary table, metrics comparison bar charts, multi-metric radar chart, and dataset property cards |
| **FD001 – FD004** | Per-dataset deep dive — KPIs, RUL prediction vs ground truth with 90% confidence interval bands, predicted vs true scatter plot, uncertainty analysis (colour-coded by σ), and error distribution / residual plots |
| **Experts** | MoE gate utilisation analysis — expert weight heatmap across datasets, grouped bar chart of per-expert weights, and entropy balance gauge (% of perfect uniformity) |
| **EDA** | Pre-generated exploratory data analysis images — sensor distributions and degradation trajectory plots per dataset (selectable via dropdown) |

### Dashboard Tech Stack

| Component | Technology |
|---|---|
| **Framework** | Dash 2.x |
| **Charts** | Plotly.js (via `plotly.graph_objects`) |
| **Tables** | Dash DataTable |
| **Styling** | Custom inline styles (dark glassmorphism theme) |
| **Server** | Flask (built into Dash) |

---

## 🏗️ Architecture & Workflow

### Model Architecture — `TurbofanRULModel`

```
Input: [Batch, Window, Sensors]
                 │
                 ▼
┌─────────────────────────────────┐
│  Multi-scale CNN                │  ← Conv1D (k=3, 5, 7) in parallel
│  + GELU + Dropout + LayerNorm   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  BiLSTM (2-layer, hidden=128)   │  ← Bidirectional, captures time dependencies
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  3D Attention                   │  ← Highlights degradation-informative steps
│  + AttentionPool                │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  Learned FC          HC Feature FC     ← Dual-stream: sequential + handcrafted
        └────────┬────────┘
                 │ Concatenate
                 ▼
┌─────────────────────────────────┐
│  MoE Output Head                │  ← 4 Expert MLPs + gated router
│  (entropy reg. + var. penalty)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Predicted RUL (Engine Cycles)  │
└─────────────────────────────────┘
```

### Solving the Expert Utilisation Problem (MoE Collapse)

A common flaw in standard Mixture of Experts (MoE) architectures is **Expert Collapse**, where the gating network becomes mathematically "lazy" and routes all data to only 1 or 2 experts, leaving the others completely unutilised (dead). 

This pipeline successfully guarantees even distribution across all 4 experts using three strategic mechanics implemented in `model.py`:

1. **Variance Penalty (`var_penalty`)**: We calculate the statistical variance of the mean gate weights across the batch. Minimising this variance explicitly forces the router to spread the workload uniformly (aiming for exactly 25% throughput per expert).
2. **Entropy Bonus (`entropy_bonus`)**: We calculate the entropy of the mean gate distribution. The loss function includes a penalty if the entropy falls short of the theoretical maximum (`np.log(4)`), actively rewarding the network for maintaining balanced expert probabilities.
3. **Smart Bias Initialization**: Inside `ExpertMLP`, the final output bias of every expert is hard-initialized to `0.5` instead of random noise. Because the final network outputs are clamped between `[0, 1]`, starting at `0.5` ensures every expert begins training dead-center in the "active" gradient zone. This prevents experts from dying in the very first optimization step.

### Training Details

| Parameter | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Window Size | 30 | 60 | 30 | 60 |
| Max RUL | 130 | 125 | 125 | 150 |
| Epochs | 300 | 400 | 300 | 450 |
| Learning Rate | 5e-4 | 2e-4 | 5e-4 | 1e-4 |
| Dropout | 0.35 | 0.35 | 0.40 | 0.35 |
| Conditions | 1 | 6 | 1 | 6 |

### Evaluation Metrics

- **RMSE** — Root Mean Squared Error (in engine cycles)
- **NASA Score** — Asymmetric scoring function that penalises late predictions (under-predictions) more than early ones. Here, **`d = Predicted RUL - True RUL`** (the error difference in cycles):

```
Score = Σ (exp(d/10) - 1)   if d ≥ 0  (late prediction)
        Σ (exp(-d/13) - 1)  if d < 0  (early prediction)
```

### 🏆 Final Experimental Results

| Dataset | RMSE | Score | Uncertainty (Avg) |
|:---:|:---:|:---:|:---:|
| **FD001** | 13.99 | 329.34 | ± 3 cycles |
| **FD002** | 11.90 | 659.17 | ± 3 cycles |
| **FD003** | 12.00 | 259.70 | ± 4 cycles |
| **FD004** | 18.44 | 2209.97 | ± 6 cycles |
| **Average** | **14.08** | **864.55** | - |