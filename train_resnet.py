import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

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
    "data/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "data/val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    "data/test",
    transform=eval_transform
)
print(train_dataset.class_to_idx)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

print("Device:", device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001
)
epochs = 5
checkpoint_path = Path("checkpoints/resnet18_cifake.pt")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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
            },
            checkpoint_path
        )

        print("Saved new best model.")

    print(
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_accuracy:.4f}"
    )