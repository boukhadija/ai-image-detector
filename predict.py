"""Run CIFAKE inference for one image using a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.detector import create_model
from train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one image as real or fake")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/baseline_cnn.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_name = checkpoint.get("model_name", "baseline")
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    image_size = checkpoint.get("image_size", 32)
    imagenet_normalization = checkpoint.get("imagenet_normalization", model_name == "resnet18")
    if imagenet_normalization:
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    model = create_model(model_name, num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        logits = model(transform(image).unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = probabilities.argmax().item()

    print(f"Prediction: {idx_to_class[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item():.4f}")
    for index, class_name in idx_to_class.items():
        print(f"{class_name}: {probabilities[index].item():.4f}")


if __name__ == "__main__":
    main()