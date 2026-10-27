import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # assuming binary classification (real vs fake)


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

test_dataset = datasets.ImageFolder(
    "data/test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

checkpoint = torch.load(
    "checkpoints/resnet18_cifake.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []
all_fake_scores = []

fake_index = test_dataset.class_to_idx["fake"]
real_index = test_dataset.class_to_idx["real"]

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        # fake_index = test_dataset.class_to_idx["fake"]

        fake_scores = torch.softmax(logits, dim=1)[:, fake_index]
        all_fake_scores.extend(fake_scores.cpu().tolist())





fake_labels = [
    1 if label == fake_index else 0
    for label in all_labels
]
accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(
    all_labels,
    all_predictions,
    pos_label=fake_index
)

recall = recall_score(
    all_labels,
    all_predictions,
    pos_label=fake_index
)

f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=fake_index
)

auc_roc = roc_auc_score(
    fake_labels,
    all_fake_scores
)
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[fake_index, real_index]
)

tp, fn = cm[0]
fp, tn = cm[1]

print(f"Test images: {len(all_labels)}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")

print(f"accuracy: {accuracy:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1: {f1:.4f}")
print(f"auc_roc: {auc_roc:.4f}")

print(f"TP: {tp}")
print(f"FN: {fn}")
print(f"FP: {fp}")
print(f"TN: {tn}")