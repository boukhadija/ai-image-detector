"""Evaluate a trained FFT/DCT feature classifier on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.frequency_features import (
    DEFAULT_DCT_SIZE,
    DEFAULT_FFT_BINS,
    create_frequency_dataloaders,
    extract_loader_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAKE FFT/DCT classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/frequency_features.joblib")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers cannot be negative")

    checkpoint = joblib.load(args.checkpoint)
    class_to_idx = checkpoint["class_to_idx"]
    loaders, dataset_class_to_idx = create_frequency_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=checkpoint["image_size"],
    )
    if class_to_idx != dataset_class_to_idx:
        raise ValueError("Checkpoint class mapping does not match the dataset class mapping")

    features, labels = extract_loader_features(
        loaders["test"],
        checkpoint.get("fft_bins", DEFAULT_FFT_BINS),
        checkpoint.get("dct_size", DEFAULT_DCT_SIZE),
    )
    model = checkpoint["model"]
    predictions = model.predict(features)
    class_indices = [class_to_idx["fake"], class_to_idx["real"]]
    fake_scores = model.predict_proba(features)[:, class_to_idx["fake"]]
    fake_labels = (labels == class_to_idx["fake"]).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=class_indices)
    tp, fn = matrix[0]
    fp, tn = matrix[1]
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, pos_label=class_to_idx["fake"]),
        "auc_roc": roc_auc_score(fake_labels, fake_scores),
        "precision": precision_score(labels, predictions, pos_label=class_to_idx["fake"]),
        "recall": recall_score(labels, predictions, pos_label=class_to_idx["fake"]),
    }

    print(f"Test images: {len(labels)}; checkpoint epoch: {checkpoint['epoch']}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"TP: {tp}")
    print(f"FN: {fn}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")


if __name__ == "__main__":
    main()
