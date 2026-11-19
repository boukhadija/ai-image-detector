"""Measure frequency-classifier accuracy drops under common image corruptions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from src.dataset import IMAGE_EXTENSIONS
from src.frequency_features import (
    CLASS_TO_IDX,
    DEFAULT_DCT_SIZE,
    DEFAULT_FFT_BINS,
    extract_frequency_features,
)
from src.robustness import CORRUPTION_LEVELS, apply_corruption


def _test_samples(data_dir: Path, class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
    test_dir = data_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test split: {test_dir}")
    if class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            f"Expected checkpoint class mapping {CLASS_TO_IDX}, found {class_to_idx}"
        )
    class_directories = {
        entry.name for entry in test_dir.iterdir() if entry.is_dir()
    }
    if class_directories != set(class_to_idx):
        raise ValueError(
            f"Test split must contain only {sorted(class_to_idx)} class directories; "
            f"found {sorted(class_directories)}"
        )

    samples = []
    for class_name, label in class_to_idx.items():
        class_dir = test_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing test class directory: {class_dir}")
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No test images found in {class_dir}")
        samples.extend((path, label) for path in images)
    return samples


def _accuracy_for_condition(
    samples: list[tuple[Path, int]],
    model,
    image_size: int,
    fft_bins: int,
    dct_size: int,
    batch_size: int,
    corruption: str,
    severity: int | float | None,
) -> float:
    correct = 0
    sample_count = 0
    for offset in range(0, len(samples), batch_size):
        batch_samples = samples[offset : offset + batch_size]
        images = []
        labels = []
        for path, label in batch_samples:
            with Image.open(path) as source:
                transformed = apply_corruption(
                    source, corruption, severity, image_size
                )
                images.append(np.asarray(transformed, dtype=np.float32) / 255.0)
            labels.append(label)
        image_array = np.transpose(np.stack(images), (0, 3, 1, 2))
        features = extract_frequency_features(image_array, fft_bins, dct_size)
        predictions = model.predict(features)
        correct += int(np.count_nonzero(predictions == np.asarray(labels)))
        sample_count += len(labels)
    if sample_count == 0:
        raise ValueError("The test split is empty")
    return correct / sample_count


def run_robustness_evaluation(
    data_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 128,
) -> list[dict[str, str | int | float]]:
    """Evaluate clean and corrupted versions of the same test images."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint = joblib.load(checkpoint_path)
    class_to_idx = checkpoint["class_to_idx"]
    samples = _test_samples(data_dir, class_to_idx)
    model = checkpoint["model"]
    image_size = checkpoint["image_size"]
    fft_bins = checkpoint.get("fft_bins", DEFAULT_FFT_BINS)
    dct_size = checkpoint.get("dct_size", DEFAULT_DCT_SIZE)

    clean_accuracy = _accuracy_for_condition(
        samples,
        model,
        image_size,
        fft_bins,
        dct_size,
        batch_size,
        "clean",
        None,
    )
    results: list[dict[str, str | int | float]] = [
        {
            "corruption": "clean",
            "severity": "none",
            "images": len(samples),
            "accuracy": clean_accuracy,
            "accuracy_drop_pp": 0.0,
            "clean_accuracy": clean_accuracy,
        }
    ]
    for corruption, severities in CORRUPTION_LEVELS.items():
        for severity in severities:
            accuracy = _accuracy_for_condition(
                samples,
                model,
                image_size,
                fft_bins,
                dct_size,
                batch_size,
                corruption,
                severity,
            )
            results.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "images": len(samples),
                    "accuracy": accuracy,
                    "accuracy_drop_pp": (clean_accuracy - accuracy) * 100,
                    "clean_accuracy": clean_accuracy,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "corruption",
        "severity",
        "images",
        "accuracy",
        "accuracy_drop_pp",
        "clean_accuracy",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure clean and corruption accuracy for a frequency classifier"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/frequency_features.joblib")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/frequency_robustness.csv"),
        help="CSV report path",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    results = run_robustness_evaluation(
        args.data_dir, args.checkpoint, args.output, args.batch_size
    )
    print(f"Test images: {results[0]['images']}")
    print(f"{'corruption':<14} {'severity':>10} {'accuracy':>10} {'drop (pp)':>12}")
    for result in results:
        print(
            f"{str(result['corruption']):<14} "
            f"{str(result['severity']):>10} "
            f"{float(result['accuracy']):>10.4f} "
            f"{float(result['accuracy_drop_pp']):>12.2f}"
        )
    print(f"CSV report saved to {args.output.resolve()}")
    print(
        "Screenshot results use a synthetic framed-display simulation, "
        "not real captured screenshots."
    )


if __name__ == "__main__":
    main()
