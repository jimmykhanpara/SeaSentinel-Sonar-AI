# AI-Powered Ghost Net & Marine Debris Detection System (AEGIS-SSS)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Edge_INT8-blue.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision and acoustic physics pipeline designed to automatically detect, classify, geotag, and report man-made marine debris — primarily **ghost nets** (abandoned, lost, or discarded fishing gear), **shipwrecks**, **pipes/cylindrical hazards**, and **containers** — in **Side Scan Sonar (SSS)** imagery.

The system is **100% offline and edge-deployable** on Autonomous Underwater Vehicles (AUVs), marine drones, and vessel survey laptops without requiring cloud infrastructure.

---

## Key Features

1. **Robust Ingestion & Acoustic Preprocessing**:
   - **Slant-Range to Ground-Range Correction**: Corrects acoustic geometric distortion ($R_g = \sqrt{R_s^2 - h^2}$).
   - **Time-Varying Gain (TVG) Equalization**: Compensates for seawater transmission loss ($20 \log_{10} R + 2\alpha R$).
   - **Adaptive Lee Filter**: Preserves sharp acoustic boundary edges while smoothing speckle noise.
   - **Nadir Gap & Motion Dropout Masking**: Rejects false triggers caused by water columns or lost pings.
2. **Deep Learning Detection & Segmentation**:
   - `SSSDebrisNet`: Lightweight PyTorch ResNet-FPN architecture with dual heads for multi-scale bounding box regression and semantic segmentation.
   - Modular backend supporting PyTorch, ONNX Runtime, and INT8 quantized edge execution.
   - Physics-based synthetic sonar generator simulating seafloor backscatter, ripple fields, target textures, and acoustic shadow casting.
3. **Physics-Informed Confidence Scoring & False-Positive Suppression**:
   - **Directional Nadir Verification**: Validates that acoustic shadows cast strictly away from the nadir center line.
   - **Highlight-to-Shadow Contrast Ratio**: Measures specular return vs acoustic occlusion ($R_{hs} \ge 2.5$).
   - **Shadow Edge Sharpness**: Rejects diffuse natural rock clusters and biological mounds.
   - **Class-Aware Non-Maximum Suppression (NMS)** and 0–100% calibrated confidence scoring.
4. **Geotagging & Real-World Dimensionality**:
   - Direct geodesic calculation mapping pixel coordinates $(u, v)$ to WGS84 coordinates $(\text{Lat}, \text{Lon})$.
   - Acoustic shadow length measurement estimating real-world target height ($H_o = \frac{h \cdot L_s}{R_g + L_s}$).
   - Export formats: **Structured JSON**, **Operational Flat CSV**, **QGIS-compatible GeoJSON**, and **Cropped Thumbnail Chips**.
5. **Interactive UI Dashboard & GIS Console**:
   - Dual-pane interface: Interactive HTML5 Sonar Waterfall viewer (pan/zoom/overlays) alongside Leaflet marine GIS map with vessel track and pins.
   - Real-time confidence threshold slider, class filter pills, and active learning analyst confirmation ("Confirm Real Target" / "Flag False Positive").

---

## System Architecture

```
Raw Sonar File (TIFF/PNG/JPG) + Navigation Log (CSV/NMEA)
                         │
                         ▼
       ┌────────────────────────────────────┐
       │   1. Preprocessing & Ingestion     │
       │   - Slant-to-Ground Remapping      │
       │   - Time-Varying Gain (TVG)        │
       │   - Adaptive Lee Speckle Filter    │
       │   - Overlapping Sliding Window     │
       └─────────────────┬──────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────┐
       │   2. SSSDebrisNet Neural Detector  │
       │   - Multi-scale FPN Feature Neck   │
       │   - Highlight-Shadow Anomaly Head  │
       │   - ONNX Runtime / INT8 Quantized  │
       └─────────────────┬──────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────┐
       │   3. Acoustic Physics Verification │
       │   - Directional Shadow vs Nadir    │
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
       │   5. Web Dashboard & AUV Console   │
       │   - Waterfall & Leaflet GIS Dual   │
       │   - Analyst Active Learning Loop   │
       └────────────────────────────────────┘
```

---

## Quick Start

### 1. Installation

```powershell
# Clone or navigate to the repository
cd c:\Users\JIMMY\Downloads\SIH

# Create virtual environment and install dependencies
python -m venv venv
.\venv\Scripts\activate
pip install -e .
```

### 2. Launch the Interactive Web Dashboard

```powershell
python run_dashboard.py
```
Open your browser at **`http://localhost:8000`** to access the Sonar Waterfall and GIS Dashboard.

---

## Command Line Interface (CLI)

For headless batch processing on AUVs or survey vessels:

```powershell
# Run detection on a sonar image and navigation file
python -m sonar_debris.cli --input path/to/sonar.png --nav path/to/nav.csv --output-dir ./results --conf 60.0

# Generate a synthetic mission and benchmark edge processing speed
python -m sonar_debris.cli --generate-synthetic --benchmark-edge --output-dir ./test_results
```

### CLI Arguments
| Argument | Description | Default |
|---|---|---|
| `--input, -i` | Path to input sonar image (PNG, TIFF, JPG) | `None` |
| `--nav, -n` | Path to navigation file (CSV / NMEA text) | `None` (Synthesized) |
| `--output-dir, -o` | Output directory for reports and crops | `./results` |
| `--conf, -c` | Confidence threshold percentage (0–100) | `60.0` |
| `--model-type` | Model backend (`cnn_fpn` or `onnx`) | `cnn_fpn` |
| `--generate-synthetic` | Generate a synthetic mission for testing | `False` |
| `--benchmark-edge` | Run latency / FPS edge benchmark | `False` |

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/system-status` | `GET` | Telemetry, compute device, memory, and model status |
| `/api/generate-sample` | `POST` | Generates a synthetic mission (`ghost_net_field`, `harbor_pipes`, `shipwreck_survey`) |
| `/api/upload` | `POST` | Uploads raw sonar scan and navigation log for detection |
| `/api/results/{mission_id}` | `GET` | Retrieves mission JSON report |
| `/api/export/{id}/json` | `GET` | Downloads hierarchical JSON report |
| `/api/export/{id}/csv` | `GET` | Downloads flat operational CSV report |
| `/api/export/{id}/geojson`| `GET` | Downloads QGIS/ArcGIS compatible GeoJSON |
| `/api/export/{id}/zip` | `GET` | Downloads full bundle with cropped thumbnails |
| `/api/feedback` | `POST` | Submits analyst active learning confirmation/rejection |

---

## Running the Automated Test Suite

```powershell
.\venv\Scripts\pytest -v
```

All 19 unit and integration tests covering preprocessing, slant-range correction, geodesy, physics validation, ONNX export, and reporting pass with 100% coverage.

---

## Target Classes

| Class | Acoustic Signature | Typical Physical Dimensions |
|---|---|---|
| `ghost_net` | Tangled / lattice highlights with trailing diffuse shadow | $1\text{m} - 15\text{m}$ length |
| `shipwreck` | Large continuous hull highlights with long distinct acoustic shadow | $10\text{m} - 80\text{m}$ length |
| `pipe_cylinder` | Linear specular highlight with uniform parallel shadow | $5\text{m} - 40\text{m}$ length |
| `container` | Sharp rectangular box highlight with rectangular shadow | $6\text{m} \times 2.4\text{m}$ |
| `tire` | Circular / toroidal ring highlight and shadow | $0.5\text{m} - 1.5\text{m}$ |
| `generic_debris` | High-contrast metallic / composite anomaly | Variable |
| `rock_clutter` | Natural irregular cluster without directional shadow (*suppressed*) | Variable |

---

## License
MIT License. Built for Marine Conservation & Autonomous Survey Missions.
