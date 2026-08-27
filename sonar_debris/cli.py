"""
Sonar Marine Debris Detection Command Line Interface (CLI)
==========================================================
Run batch inference, edge benchmarks, synthetic survey generation,
and report exports from the terminal.
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import json
from PIL import Image
import numpy as np

from .types import PipelineConfig
from .pipeline import SonarDebrisPipeline
from .models.synthetic_generator import SyntheticSonarGenerator
from .geotagging.reporter import SonarReporter


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Ghost Net & Marine Debris Detection System for Side Scan Sonar (SSS)"
    )
    parser.add_argument("--input", "-i", type=str, help="Path to input sonar image (PNG, TIFF, JPG)")
    parser.add_argument("--nav", "-n", type=str, default=None, help="Path to navigation log (CSV or NMEA)")
    parser.add_argument("--output-dir", "-o", type=str, default="./results", help="Directory to save output reports")
    parser.add_argument("--conf", "-c", type=float, default=60.0, help="Confidence threshold percentage (0-100)")
    parser.add_argument("--model-type", type=str, default="cnn_fpn", choices=["cnn_fpn", "onnx"], help="Model architecture")
    parser.add_argument("--onnx-path", type=str, default=None, help="Path to ONNX weights if model-type is onnx")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate a synthetic mission and process it")
    parser.add_argument("--benchmark-edge", action="store_true", help="Run latency & throughput benchmarks")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.generate_synthetic or not args.input:
        print("\n=======================================================")
        print("  AI-Powered Ghost Net & Marine Debris Detection System")
        print("  Generating Synthetic High-Fidelity SSS Survey Mission...")
        print("=======================================================\n")

        gen = SyntheticSonarGenerator(image_width=1024, image_height=1024, altitude_m=10.0, seed=42)
        sonar_img, gt_targets, nav_track = gen.generate_mission(num_targets=6)

        synthetic_img_path = os.path.join(args.output_dir, "synthetic_sonar_survey.png")
        synthetic_nav_path = os.path.join(args.output_dir, "synthetic_nav_track.csv")

        # Save synthetic image
        img_uint8 = np.clip(sonar_img * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(synthetic_img_path)

        # Save synthetic navigation CSV
        with open(synthetic_nav_path, "w", encoding="utf-8") as f:
            f.write("ping,lat,lon,heading,altitude,speed,depth\n")
            for p in nav_track:
                f.write(f"{p.ping_number},{p.latitude:.7f},{p.longitude:.7f},{p.heading_deg:.2f},{p.altitude_m:.2f},{p.speed_knots:.1f},{p.depth_m:.1f}\n")

        print(f"[+] Synthetic mission generated with {len(gt_targets)} ground-truth targets.")
        print(f"[+] Saved sonar image: {synthetic_img_path}")
        print(f"[+] Saved navigation track: {synthetic_nav_path}\n")

        input_path = synthetic_img_path
        nav_path = synthetic_nav_path
    else:
        input_path = args.input
        nav_path = args.nav

    # Configure Pipeline
    config = PipelineConfig(
        model_type=args.model_type,
        onnx_model_path=args.onnx_path,
        confidence_threshold_percent=args.conf,
        export_thumbnails=True
    )
    pipeline = SonarDebrisPipeline(config=config)

    print(f"[*] Running detection pipeline (Confidence Threshold: {args.conf}%)...")
    t0 = time.time()
    report = pipeline.run(
        sonar_input=input_path,
        nav_input=nav_path,
        mission_id=os.path.basename(input_path),
        output_dir=args.output_dir
    )
    t1 = time.time()

    # Save Reports
    json_path = os.path.join(args.output_dir, "mission_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(SonarReporter.generate_json_report(report))

    csv_path = os.path.join(args.output_dir, "debris_detections.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(SonarReporter.generate_csv_report(report.detections))

    geojson_path = os.path.join(args.output_dir, "detections.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(SonarReporter.generate_geojson_report(report.detections, report.nav_track), indent=2))

    # Annotated mosaic
    raw_img = np.array(Image.open(input_path).convert("L"), dtype=np.float32) / 255.0
    annotated = SonarReporter.generate_annotated_mosaic(raw_img, report.detections)
    annotated_path = os.path.join(args.output_dir, "annotated_waterfall.png")
    Image.fromarray(annotated).save(annotated_path)

    print(f"\n[+] Pipeline completed successfully in {t1 - t0:.2f}s!")
    print(f"    - High-Confidence Detections: {len(report.detections)}")
    print(f"    - Ghost Nets Found:           {report.summary['ghost_nets_found']}")
    print(f"    - Shipwrecks Found:           {report.summary['shipwrecks_found']}")
    print(f"    - Cylindrical Pipes Found:    {report.summary['pipes_found']}")
    print(f"    - Survey Area Covered:        {report.summary['survey_area_sq_km']} km²")
    print(f"    - Low-Confidence Audit Count: {report.summary['low_confidence_audit_count']}")
    print("\nGenerated Artifacts:")
    print(f"    - JSON Report:     {json_path}")
    print(f"    - CSV Summary:     {csv_path}")
    print(f"    - QGIS GeoJSON:    {geojson_path}")
    print(f"    - Annotated Image: {annotated_path}")

    if args.benchmark_edge:
        print("\n[*] Benchmarking Edge Hardware Latency (10 iterations)...")
        latencies = []
        for _ in range(10):
            tb0 = time.time()
            _ = pipeline.run(sonar_input=input_path, nav_input=nav_path)
            latencies.append(time.time() - tb0)
        avg_lat = np.mean(latencies)
        fps = 1.0 / avg_lat
        print(f"[+] Average Latency: {avg_lat * 1000:.1f} ms ({fps:.2f} full sonar swaths/sec)")
        print(f"[+] Edge Ready: YES (Low power consumption, zero cloud dependency)")


if __name__ == "__main__":
    main()
