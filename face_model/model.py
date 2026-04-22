"""
model.py — EfficientNet-B0 emotion classifier
"""

import torch
import torch.nn as nn
from torchvision import models

# RAF-DB Folder Mapping: 1:Surprise, 2:Fear, 3:Disgust, 4:Happy, 5:Sad, 6:Angry, 7:Neutral
# ImageFolder assigns index 0 to folder '1', index 1 to folder '2', etc.
EMOTION_LABELS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
NUM_CLASSES = len(EMOTION_LABELS)

class EmotionNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def get_optimizer_param_groups(self, head_lr, backbone_lr):
        return [
            {'params': list(self.backbone.features.parameters()), 'lr': backbone_lr},
            {'params': list(self.backbone.classifier.parameters()), 'lr': head_lr},
        ]