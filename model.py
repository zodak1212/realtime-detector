"""
model.py
========
EfficientNet-B0 emotion classifier.

Pretrained on ImageNet (14M images), fine-tuned for 7-class emotion detection.
Two-phase training: freeze backbone first, then unfreeze for full fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models


EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
NUM_CLASSES = len(EMOTION_LABELS)


class EmotionNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        in_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze backbone — only train classifier head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen.")

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen.")

    def get_optimizer_param_groups(self, head_lr: float, backbone_lr: float):
        """Different learning rates for backbone vs head."""
        return [
            {'params': list(self.backbone.features.parameters()), 'lr': backbone_lr},
            {'params': list(self.backbone.classifier.parameters()), 'lr': head_lr},
        ]


def build_model(device: torch.device = None) -> EmotionNet:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmotionNet()
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] EfficientNet-B0 — {total:,} params on {device}")

    return model
