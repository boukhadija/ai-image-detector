"""Baseline and pretrained CIFAKE image classifiers."""

import torch
from torch import nn
from torchvision import models


class BaselineCNN(nn.Module):
	"""Three-block CNN returning fake/real class logits."""

	def __init__(self, num_classes: int = 2) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

	def forward(self, images):
		return self.classifier(self.features(images))


class ResNet18(nn.Module):
	"""ResNet-18 with ImageNet weights and a CIFAKE binary classifier head."""

	def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> None:
		super().__init__()
		weights = models.ResNet18_Weights.DEFAULT if pretrained else None
		self.backbone = models.resnet18(weights=weights)
		self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

		if freeze_backbone:
			for name, parameter in self.backbone.named_parameters():
				if not name.startswith("fc."):
					parameter.requires_grad_(False)

	def forward(self, images):
		return self.backbone(images)


def create_model(name: str, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
	name = name.lower()
	if name == "baseline":
		return BaselineCNN(num_classes=num_classes)
	if name == "resnet18":
		return ResNet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
	raise ValueError(f"Unknown model '{name}'. Choose baseline or resnet18.")