"""
Deep Learning Models & Synthetic Generation
===========================================
Modular detection/segmentation neural architectures, ONNX edge runtime,
and physics-based synthetic acoustic sonar data generator.
"""

from .synthetic_generator import SyntheticSonarGenerator
from .backbone import SSSDebrisNet, PyTorchSonarDetector, BaseSonarDetector
from .onnx_engine import ONNXSonarDetector, export_to_onnx, quantize_onnx_model

__all__ = [
    "SyntheticSonarGenerator",
    "SSSDebrisNet",
    "PyTorchSonarDetector",
    "BaseSonarDetector",
    "ONNXSonarDetector",
    "export_to_onnx",
    "quantize_onnx_model"
]
