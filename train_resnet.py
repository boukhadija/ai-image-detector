import argparse
import platform
import torchvision
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import random
import numpy as np


parser = argparse.ArgumentParser(
    description="Train ResNet18 on CIFAKE"
)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/resnet18_cifake_repro.pt"
)

args = parser.parse_args()

SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Random seed: {SEED}")


weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # assuming binary classification (real vs fake)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
train_dataset = datasets.ImageFolder(
    Path(args.data_dir) / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    Path(args.data_dir) / "val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    Path(args.data_dir) / "test",
    transform=eval_transform
)
print(train_dataset.class_to_idx)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


train_generator = torch.Generator()
train_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    generator=train_generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

print("Device:", device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr
)
epochs = args.epochs

checkpoint_path = Path(args.checkpoint)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

training_config = {
    "seed": args.seed,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "optimizer": "AdamW",
    "architecture": "ResNet18",
    "pretrained_weights": str(weights),
    "data_dir": args.data_dir,
    "class_to_idx": train_dataset.class_to_idx,
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "test_samples": len(test_dataset),
    "train_transform": str(train_transform),
    "eval_transform": str(eval_transform),
    "device": str(device),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torchvision_version": torchvision.__version__,
}


best_val_accuracy = 0.0
for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)


    train_loss = running_loss / total
    train_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"train_loss={train_loss:.4f} "
        f"train_acc={train_accuracy:.4f}"
    )

    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            val_loss_sum += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "epoch": epoch + 1,
        "val_accuracy": val_accuracy,
        "training_config": training_config,
    },
    checkpoint_path
)

        print("Saved new best model.")

    print(
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_accuracy:.4f}"
    )