# Public Online Datasets for Side Scan Sonar (SSS) Marine Debris

This folder is configured for downloading and storing public benchmark datasets for sonar object detection and classification.

---

## 1. SeabedObjects-KLSG (Side-Scan Sonar Object Dataset)
- **Source**: Polish Naval Academy & Marine Technology Labs
- **Data Type**: Real Side Scan Sonar (SSS) imagery with bounding box annotations
- **Target Classes**: `Shipwrecks`, `Drowned Aircraft`, `Containers`, `Mine-like Objects`, `Seafloor Obstructions`
- **Download Link**: [https://github.com/klsg-pna/SeabedObjects-KLSG](https://github.com/klsg-pna/SeabedObjects-KLSG)
- **Format**: PNG/JPEG images with Pascal VOC / YOLO `.txt` annotation format

---

## 2. UIB Marine Debris Sonar Dataset (MDD)
- **Source**: University of the Balearic Islands (UIB) Marine Robotics Group
- **Data Type**: Acoustic forward-looking and side-scan sonar scans
- **Target Classes**: `Ghost Nets / Fishing Nets`, `Pipes & Cylinders`, `Tires`, `Bottles`, `Metal Drums`
- **Download Link**: [https://github.com/mvaldenegro/marine-debris-fls-dataset](https://github.com/mvaldenegro/marine-debris-fls-dataset)
- **Format**: Sonar frames in NumPy `.npy` and PNG format with bounding boxes

---

## 3. NOAA AWOIS & NCEI Hydrographic Sonar Database
- **Source**: National Oceanic and Atmospheric Administration (NOAA)
- **Data Type**: High-resolution Side Scan Sonar waterfall survey lines (EdgeTech, Klein, L3Harris)
- **Download Portal**: [https://www.ncei.noaa.gov/maps/bathymetry/](https://www.ncei.noaa.gov/maps/bathymetry/)
- **Format**: Raw `.xtf`, `.jsf`, and GeoTIFF side-scan acoustic swaths

---

## 4. Automatic Download Script
You can run the included Python script to automatically download available public samples:
```powershell
python datasets/online_sources/download_public_datasets.py
```
