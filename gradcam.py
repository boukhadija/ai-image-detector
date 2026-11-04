
"""Generate Grad-CAM explanations for the trained CIFAKE ResNet18."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.detector import create_model
from train import choose_device



def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    class_to_idx = checkpoint["class_to_idx"]
    state_dict = checkpoint["model_state_dict"]

    # Older checkpoints saved ResNet18 without the wrapper.
    is_old_checkpoint = "conv1.weight" in state_dict

    model_name = checkpoint.get("model_name", "resnet18")

    if model_name != "resnet18":
        raise ValueError("This script requires a ResNet18 checkpoint.")

    model = create_model(
        "resnet18",
        num_classes=len(class_to_idx),
        pretrained=False,
    ).to(device)

    if is_old_checkpoint:
        # Add the prefix expected by Ismail's ResNet18 wrapper.
        state_dict = {
            f"backbone.{key}": value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint, class_to_idx



def prepare_image(image_path, checkpoint):
    original = Image.open(image_path).convert("RGB")

    image_size = checkpoint.get("image_size", 224)

    if checkpoint.get("imagenet_normalization", True):
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    else:
        normalize = transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    image_tensor = transform(original).unsqueeze(0)

    return original, image_tensor


def generate_gradcam(model, image_tensor, device):
    activations = {}
    gradients = {}

    target_layer = model.backbone.layer4[-1]

    def forward_hook(module, inputs, output):
        activations["value"] = output

        def save_gradient(gradient):
            gradients["value"] = gradient

        output.register_hook(save_gradient)

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        image_tensor = image_tensor.to(device)
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)

            predicted_index = int(logits.argmax(dim=1).item())
            score = logits[0, predicted_index]

            score.backward()

        feature_maps = activations["value"].detach()
        feature_gradients = gradients["value"].detach()

        weights = feature_gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)

        heatmap = cam.cpu().numpy()

        return predicted_index, probabilities[0].detach().cpu(), heatmap

    finally:
        handle.remove()


def save_visualizations(original, heatmap, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = original.size
    original.save(output_dir / "original.png")

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255), mode="L"
    ).resize((width, height), Image.Resampling.BILINEAR)

    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    colored_heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    colored_heatmap[:, :, 0] = np.uint8(255 * heatmap_array)
    colored_heatmap[:, :, 1] = np.uint8(
        80 * (1.0 - heatmap_array)
    )
    colored_heatmap[:, :, 2] = np.uint8(
        255 * (1.0 - heatmap_array)
    )

    heatmap_rgb = Image.fromarray(colored_heatmap, mode="RGB")
    heatmap_rgb.save(output_dir / "heatmap.png")

    overlay = Image.blend(original, heatmap_rgb, alpha=0.45)
    overlay.save(output_dir / "overlay.png")


def main():
    parser = argparse.ArgumentParser(
        description="Explain ResNet18 predictions using Grad-CAM"
    )

    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/resnet18_cifake.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gradcam_outputs"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    device = choose_device(args.device)

    model, checkpoint, class_to_idx = load_model(
        args.checkpoint, device
    )

    original, image_tensor = prepare_image(args.image, checkpoint)

    predicted_index, probabilities, heatmap = generate_gradcam(
        model, image_tensor, device
    )

    idx_to_class = {
        index: name for name, index in class_to_idx.items()
    }

    predicted_class = idx_to_class[predicted_index]
    confidence = probabilities[predicted_index].item()

    save_visualizations(original, heatmap, args.output_dir)

    print(f"Device: {device}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.4f}")

    for class_name, index in class_to_idx.items():
        print(f"{class_name}: {probabilities[index].item():.4f}")

    print(f"Grad-CAM images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
