"""
Public Sonar Dataset Download & Ingestion Utility
=================================================
Downloads sample images from public marine sonar repositories and prepares them for inference.
"""

import os
import urllib.request

DEST_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  Public Sonar Dataset Ingestion Utility")
print(f"  Destination Directory: {DEST_DIR}")
print("=" * 60)

SAMPLE_URLS = {
    "seabed_objects_sample.png": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/sonar_sample.png",
}

for fname, url in SAMPLE_URLS.items():
    out_path = os.path.join(DEST_DIR, fname)
    if not os.path.exists(out_path):
        try:
            print(f"Fetching sample from: {url}")
            urllib.request.urlretrieve(url, out_path)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Notice: {fname} download skipped ({e}). Manual download instructions in DATASETS_README.md.")

print("\nSetup complete. Place downloaded Kaggle or NOAA survey files into this folder for testing.")
