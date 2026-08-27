# ⚡ SeaSentinel: AI-Powered Ghost Net & Marine Debris Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Edge_INT8-blue.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision, deep learning, and acoustic sound physics pipeline designed to automatically detect, classify, geotag, and report man-made marine debris in **Side Scan Sonar (SSS)** and **Forward-Looking Sonar (FLS)** imagery.

The system is **100% offline edge-deployable** on Autonomous Underwater Vehicles (AUVs), marine survey drones, and vessel laptops without requiring continuous cloud or internet connectivity.

---

## 🌟 Key Features

1. **Robust Ingestion & Acoustic Preprocessing**:
   - **Slant-Range to Ground-Range Geometric Correction**: Un-warps diagonal sound travel distance into true horizontal seabed distance ($R_g = \sqrt{R_s^2 - h^2}$).
   - **Time-Varying Gain (TVG) Equalization**: Compensates for acoustic transmission loss through seawater ($20 \log_{10} R + 2\alpha R$).
   - **Adaptive Lee Speckle Filter**: Smooths acoustic speckle noise while preserving razor-thin ghost net filaments.
   - **Nadir Gap Masking**: Rejects false triggers from the water column gap directly beneath the tow-fish.

2. **Dual-Mode Deep Learning Engine**:
   - **`SSSDebrisNet` (Wide Swath)**: Lightweight PyTorch ResNet-FPN detector with Squeeze-and-Excitation acoustic channel attention for wide-area surveying.
   - **`ResNet-18 FLS Classifier` (Fine-Grained)**: Transfer-learning classifier trained on Kaggle FLS marine debris for 18 specific categories (*bottles, cans, pipes, tires, valves, wrenches, metal boxes*).
   - **ONNX INT8 Quantization**: Optimized for sub-50ms inference on low-power embedded AUV computers.

3. **Physics-Informed Confidence Scoring & False-Positive Suppression**:
   - **Directional Nadir Verification**: Validates that acoustic shadows cast strictly *away* from the nadir path (left on Port, right on Starboard).
   - **Highlight-to-Shadow Contrast Ratio**: Verifies that specular returns are at least 2.5× brighter than acoustic shadows ($R_{hs} \ge 2.5$).
   - **Rock Clutter Rejection**: Eliminates $>85\%$ of false alarms from irregular rock piles and sand ripples.

4. **WGS84 GPS Geodesy & Real-World Sizing**:
   - Direct spherical geodesy equations converting 2D pixel coordinates $(u, v)$ to global WGS84 coordinates $(\text{Lat}, \text{Lon})$.
   - Acoustic shadow length trigonometry estimating true target height above the seafloor ($H_o = \frac{h \cdot L_s}{R_g + L_s}$).
   - Multi-format exports: **Structured JSON**, **Flat Operational CSV**, **QGIS GeoJSON**, and **Cropped Thumbnail ZIP bundles**.

5. **Mission-Control Web Dashboard & GIS Console**:
   - **Mission Workspace**: Real-time Sonar Waterfall viewer alongside a dark ocean Leaflet GIS map.
   - **Review Queue**: Interactive triage list with candidate thumbnails and quick `✓ Confirm` / `✗ Flag False Alarm` active learning feedback.
   - **Detection Reports & Model Health**: Full database view with column sorting and edge hardware performance diagnostics.

---

## 🏛️ System Architecture

```
Raw Sonar File (PNG/TIFF/JPG) + Navigation Track (CSV/NMEA)
                         │
                         ▼
       ┌────────────────────────────────────┐
       │   1. Preprocessing & Ingestion     │
       │   - Slant-to-Ground Remapping      │
       │   - Time-Varying Gain (TVG)        │
       │   - Adaptive Lee Speckle Filter    │
       │   - Sliding Window Tiling Engine   │
       └─────────────────┬──────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────┐
       │   2. Dual Deep Learning Detectors  │
       │   - SSSDebrisNet (Macro Swath)     │
       │   - ResNet-18 (18 Fine Classes)    │
       │   - ONNX INT8 Quantized Runtime    │
       └─────────────────┬──────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────┐
       │   3. Acoustic Physics Verification │
       │   - Directional Shadow Alignment   │
       │   - Highlight-to-Shadow Contrast   │
       │   - Rock Clutter Rejection & NMS   │
       │   - Calibrated 0-100% Confidence   │
       └─────────────────┬──────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────┐
       │   4. Geotagging & Export Engine    │
       │   - WGS84 Latitude / Longitude     │
       │   - Estimated Physical Dimensions  │
       │   - JSON, CSV, GeoJSON, Crop ZIP   │
       └─────────────────┬──────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────┐
       │   5. SeaSentinel Web Dashboard     │
       │   - Interactive Waterfall Canvas   │
       │   - Leaflet Ocean GIS Mapping      │
       │   - Operator Active Learning Loop  │
       └────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/jimmykhanpara/SeaSentinel-Sonar-AI.git
cd SeaSentinel-Sonar-AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

---

### 2. Launch the Web Dashboard

```bash
# Windows / Linux / macOS:
python run_dashboard.py
```
*(On Windows, you can also simply double-click **`run.bat`**)*.

Open your browser at **`http://localhost:8000`** to access the mission console.

---

## 🌐 Free Cloud Hosting (Deployment Guide)

You can host SeaSentinel 24/7 online for free on cloud platforms:

### Option A: Render.com (Recommended Free Hosting)
1. Push your repository to GitHub.
2. Sign in to [Render.com](https://render.com/) and click **New +** $\to$ **Web Service**.
3. Connect your GitHub repository: `jimmykhanpara/SeaSentinel-Sonar-AI`.
4. Configure settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -e .`
   - **Start Command**: `uvicorn sonar_debris.server.app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**. Your app will be live with a public HTTPS URL!

### Option B: Railway.app
1. Go to [Railway.app](https://railway.app/) and click **New Project** $\to$ **Deploy from GitHub Repo**.
2. Select your repository. Railway automatically detects `pyproject.toml` and deploys your FastAPI web app.

---

## 💻 Command Line Interface (CLI)

For headless batch processing on AUVs or survey vessels:

```bash
# Run detection on a sonar image and navigation file
python -m sonar_debris.cli --input datasets/samples/scenario_ghost_nets.png --nav datasets/samples/scenario_ghost_nets_nav.csv --output-dir ./exports --conf 60.0

# Generate a synthetic mission and benchmark edge processing latency
python -m sonar_debris.cli --generate-synthetic --benchmark-edge --output-dir ./exports
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/system-status` | `GET` | Hardware telemetry, device type (CPU/GPU), and runtime status |
| `/api/generate-sample` | `POST` | Simulates a live acoustic sonar survey mission |
| `/api/upload` | `POST` | Processes user-uploaded sonar scans and navigation logs |
| `/api/classify-crop` | `POST` | Classifies an acoustic crop into 18 fine-grained debris classes |
| `/api/export/{id}/json` | `GET` | Downloads hierarchical JSON mission report |
| `/api/export/{id}/csv` | `GET` | Downloads flat operational CSV spreadsheet |
| `/api/export/{id}/geojson`| `GET` | Downloads QGIS/ArcGIS compatible GeoJSON vector layer |
| `/api/export/{id}/zip` | `GET` | Downloads full bundle with high-res cropped thumbnails |
| `/api/feedback` | `POST` | Submits operator active-learning corrections |

---

## 🧪 Automated Test Suite

Run the full automated test suite (20 tests covering preprocessing, neural networks, sound physics, geodesy, and API endpoints):

```bash
pytest -v
```

```
======================= 20 passed in 15.62s =======================
```

---

## 📂 Project Structure

```
SeaSentinel-Sonar-AI/
├── datasets/            <-- Sample sonar missions & online dataset downloaders
├── docs/                <-- Complete Word document handbook & pitch script
├── exports/             <-- Mission export reports (CSV, GeoJSON, JSON)
├── notebooks/           <-- Teammate's Kaggle training notebook (.ipynb)
├── sonar_debris/        <-- Core Python package (Backend, AI/ML, Physics, UI)
│   ├── filtering/       <-- Acoustic sound physics & shadow verification
│   ├── geotagging/      <-- WGS84 GPS geodesy & report generators
│   ├── models/          <-- SSSDebrisNet AI & ResNet-18 classifier
│   ├── preprocessing/   <-- Slant-range & Adaptive Lee filters
│   ├── server/          <-- FastAPI REST API server
│   └── ui/              <-- Frontend HTML, CSS, JS
├── tests/               <-- Automated test suites (20 tests)
├── pyproject.toml       <-- Project dependencies & packaging
├── README.md            <-- Project documentation
├── run.bat              <-- 1-click Windows desktop launcher
└── run_dashboard.py     <-- Dashboard server runner
```

---

## 📜 License
MIT License. Built for Marine Environmental Conservation and Autonomous Survey Missions.
