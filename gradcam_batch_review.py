import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from gradcam import load_model, prepare_image, generate_gradcam, save_visualizations
from train import choose_device


class TestImages(Dataset):
    def __init__(self, items, checkpoint):
        self.items = items
        size = checkpoint.get("image_size", 224)
        if checkpoint.get("imagenet_normalization", True):
            mean, std = (.485, .456, .406), (.229, .224, .225)
        else:
            mean, std = (.5, .5, .5), (.5, .5, .5)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, actual = self.items[i]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, actual, str(path)


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Stratified ResNet18 Grad-CAM review")
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/resnet18_cifake.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gradcam_review"))
    parser.add_argument("--per-group", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    if args.per_group < 1 or args.batch_size < 1:
        parser.error("--per-group and --batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    device = choose_device(args.device)
    model, checkpoint, class_to_idx = load_model(args.checkpoint, device)
    class_to_idx = {str(k).lower(): int(v) for k, v in class_to_idx.items()}
    if set(class_to_idx) != {"fake", "real"}:
        parser.error(f"Expected fake and real labels, got {class_to_idx}")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    if len(idx_to_class) != 2:
        parser.error("Class indices must be distinct")

    items = []
    for name in ("fake", "real"):
        folder = args.test_dir / name
        if not folder.is_dir():
            parser.error(f"Test folder missing: {folder}")
        items.extend((p, class_to_idx[name]) for p in sorted(folder.rglob("*"))
                     if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not items:
        parser.error("No images found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(TestImages(items, checkpoint), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    rows = []
    print(f"Evaluating {len(items)} images on {device}...", flush=True)
    with torch.inference_mode():
        for batch_idx, (images, labels, paths) in enumerate(loader):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            for i, path in enumerate(paths):
                actual = int(labels[i])
                pred = int(preds[i])
                actual_name = idx_to_class[actual]
                pred_name = idx_to_class[pred]
                group = ("TP" if pred_name == "fake" else "FN") if actual_name == "fake" else (
                    "FP" if pred_name == "fake" else "TN")
                rows.append({
                    "path": path, "actual": actual_name.upper(), "predicted": pred_name.upper(),
                    "confidence": f"{float(probs[i, pred]):.6f}",
                    "fake_probability": f"{float(probs[i, class_to_idx['fake']]):.6f}",
                    "group": group,
                })
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {len(rows)}/{len(items)}", flush=True)

    fields = ["path", "actual", "predicted", "confidence", "fake_probability", "group"]
    write_csv(args.output_dir / "all_predictions.csv", rows, fields)
    counts = Counter(row["group"] for row in rows)
    print("Confusion counts:", dict(counts), flush=True)

    rng = random.Random(args.seed)
    selected = []
    for group in ("TP", "TN", "FP", "FN"):
        candidates = [row for row in rows if row["group"] == group]
        if len(candidates) < args.per_group:
            raise RuntimeError(f"Not enough {group} examples: {len(candidates)} < {args.per_group}")
        selected.extend(rng.sample(candidates, args.per_group))
    write_csv(args.output_dir / "sample_200.csv", selected, fields)

    summary = {
        "total_test_images": len(rows),
        "confusion_counts": {group: counts[group] for group in ("TP", "TN", "FP", "FN")},
        "accuracy": (counts["TP"] + counts["TN"]) / len(rows),
        "selected_per_group": args.per_group,
        "selected_total": len(selected),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "test_dir": str(args.test_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Generating Grad-CAM for {len(selected)} images...", flush=True)
    for n, row in enumerate(selected, 1):
        original, image_tensor = prepare_image(Path(row["path"]), checkpoint)
        pred_idx, _, heatmap = generate_gradcam(model, image_tensor, device)
        if idx_to_class[pred_idx].upper() != row["predicted"]:
            raise RuntimeError(f"Prediction mismatch for {row['path']}")
        target = args.output_dir / "images" / row["group"] / f"{n:03d}_{Path(row['path']).stem}"
        save_visualizations(original, heatmap, target)
        if n % 20 == 0:
            print(f"  Generated {n}/{len(selected)}", flush=True)
    print(f"Done. Results in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
