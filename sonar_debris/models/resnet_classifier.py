"""
Teammate's ResNet-18 Marine Debris Neural Classifier
===================================================
Integrated from teammate's sonar_detection notebook.
Provides transfer learning on ResNet-18 with custom deep dense head
for 18 fine-grained marine debris categories.
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 18 Fine-Grained Marine Debris Classes from Kaggle FLS Dataset
FLS_DEBRIS_CLASSES = [
    "brown-glass-bottle",
    "can",
    "drink-carton",
    "drink-sachet",
    "glass-bottle",
    "glass-jar",
    "large-tire",
    "metal-bottle",
    "metal-box",
    "plastic-bidon",
    "plastic-bottle",
    "plastic-pipe",
    "plastic-propeller",
    "potion-glass-bottle",
    "rotating-platform",
    "small-tire",
    "valve",
    "wrench"
]

class ResNet18DebrisClassifier(nn.Module):
    """
    Teammate's ResNet-18 architecture with custom 3-layer dense head,
    BatchNorm, Dropout, and unfreezed layers 3 & 4.
    """
    def __init__(self, num_classes: int = len(FLS_DEBRIS_CLASSES), freeze_backbone: bool = True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if hasattr(models, 'ResNet18_Weights') else True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_crop(self, crop_img: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on an acoustic crop and returns:
        (top_predicted_class, confidence_score, all_class_probabilities)
        """
        self.eval()
        device = next(self.parameters()).device

        if crop_img.mode != "L":
            crop_img = crop_img.convert("L")

        tensor = self.eval_transform(crop_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_class = FLS_DEBRIS_CLASSES[top_idx]
        top_conf = float(probs[top_idx])

        all_probs = {FLS_DEBRIS_CLASSES[i]: float(probs[i]) for i in range(len(FLS_DEBRIS_CLASSES))}
        return top_class, top_conf, all_probs
