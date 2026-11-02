"""Train a logistic-regression classifier on FFT and DCT image features."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.frequency_features import (
    DEFAULT_DCT_SIZE,
    DEFAULT_FFT_BINS,
    create_frequency_dataloaders,
    extract_loader_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CIFAKE classifier on FFT and DCT features"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/frequency_features.joblib")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--fft-bins", type=int, default=DEFAULT_FFT_BINS)
    parser.add_argument("--dct-size", type=int, default=DEFAULT_DCT_SIZE)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers cannot be negative")
    if (
        args.image_size < max(DEFAULT_DCT_SIZE, args.dct_size)
        or args.dct_size < 1
        or args.fft_bins < 1
    ):
        parser.error(
            "image-size must be >= 8 and >= positive dct-size; fft-bins must be positive"
        )
    if args.regularization <= 0 or args.max_iterations < 1:
        parser.error("regularization and max-iterations must be positive")

    loaders, class_to_idx = create_frequency_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
    )
    train_features, train_labels = extract_loader_features(
        loaders["train"], args.fft_bins, args.dct_size
    )
    val_features, val_labels = extract_loader_features(
        loaders["val"], args.fft_bins, args.dct_size
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.regularization,
            max_iter=args.max_iterations,
            random_state=args.seed,
        ),
    )
    model.fit(train_features, train_labels)
    val_accuracy = accuracy_score(val_labels, model.predict(val_features))
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "class_to_idx": class_to_idx,
            "image_size": args.image_size,
            "fft_bins": args.fft_bins,
            "dct_size": args.dct_size,
            "epoch": 1,
            "val_accuracy": val_accuracy,
        },
        args.checkpoint,
    )

    print(f"Training images: {len(train_labels)}; features per image: {train_features.shape[1]}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
