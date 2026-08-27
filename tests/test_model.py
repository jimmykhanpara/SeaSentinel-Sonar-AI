import pytest
import os
import torch
import numpy as np

from sonar_debris.models.backbone import SSSDebrisNet, PyTorchSonarDetector
from sonar_debris.models.synthetic_generator import SyntheticSonarGenerator
from sonar_debris.models.onnx_engine import export_to_onnx, quantize_onnx_model, ONNXSonarDetector
from sonar_debris.models.resnet_classifier import ResNet18DebrisClassifier, FLS_DEBRIS_CLASSES
from PIL import Image


def test_debris_net_forward():
    net = SSSDebrisNet(in_channels=1, num_classes=7)
    x = torch.randn(2, 1, 128, 128)
    cls_logits, box_preds, obj_logits, mask_logits = net(x)

    assert cls_logits.shape == (2, 7, 32, 32)
    assert box_preds.shape == (2, 4, 32, 32)
    assert obj_logits.shape == (2, 1, 32, 32)
    assert mask_logits.shape == (2, 8, 128, 128)


def test_resnet18_classifier():
    classifier = ResNet18DebrisClassifier(num_classes=len(FLS_DEBRIS_CLASSES), freeze_backbone=True)
    dummy_crop = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    top_class, top_conf, probs = classifier.predict_crop(dummy_crop)

    assert top_class in FLS_DEBRIS_CLASSES
    assert 0.0 <= top_conf <= 1.0
    assert len(probs) == len(FLS_DEBRIS_CLASSES)


def test_synthetic_generator():
    gen = SyntheticSonarGenerator(image_width=256, image_height=256, seed=123)
    img, targets, nav = gen.generate_mission(num_targets=3)

    assert img.shape == (256, 256)
    assert len(nav) == 256
    assert isinstance(targets, list)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_onnx_export_and_quantization(tmp_path):
    net = SSSDebrisNet()
    onnx_file = str(tmp_path / "model.onnx")
    export_to_onnx(net, onnx_file, input_size=(128, 128))
    assert os.path.exists(onnx_file)

    quant_file = str(tmp_path / "model_quant.onnx")
    quantize_onnx_model(onnx_file, quant_file)
    assert os.path.exists(quant_file)

