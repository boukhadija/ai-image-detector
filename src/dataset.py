"""Prepare CIFAKE splits and create PyTorch data loaders."""

from __future__ import annotations

import argparse
import filecmp
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ("fake", "real")


def _class_directory(split_directory: Path, class_name: str) -> Path:
    if not split_directory.is_dir():
        raise FileNotFoundError(f"Missing CIFAKE split directory: {split_directory}")

    for directory in split_directory.iterdir():
        if directory.is_dir() and directory.name.casefold() == class_name.casefold():
            return directory

    raise FileNotFoundError(
        f"Expected a {class_name.upper()} class directory inside {split_directory}"
    )


def _images_in(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )


def prepare_cifake_split(
    source_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Materialize train/val/test class folders from an extracted CIFAKE dataset.

    CIFAKE's official test split is preserved. Validation is sampled independently
    from each class in the official training split for reproducible stratification.
    Files are hard-linked when possible and copied otherwise.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_train = source_dir / "train"
    source_test = source_dir / "test"
    planned: dict[str, dict[str, list[Path]]] = {
        "train": {class_name: [] for class_name in CLASS_NAMES},
        "val": {class_name: [] for class_name in CLASS_NAMES},
        "test": {class_name: [] for class_name in CLASS_NAMES},
    }

    randomizer = random.Random(seed)
    for class_name in CLASS_NAMES:
        train_images = _images_in(_class_directory(source_train, class_name))
        test_images = _images_in(_class_directory(source_test, class_name))
        if len(train_images) < 2:
            raise ValueError(
                f"Need at least two training images for {class_name}; found {len(train_images)}"
            )
        if not test_images:
            raise ValueError(f"No test images found for {class_name}")

        randomizer.shuffle(train_images)
        val_count = min(len(train_images) - 1, max(1, round(len(train_images) * val_fraction)))
        planned["val"][class_name] = train_images[:val_count]
        planned["train"][class_name] = train_images[val_count:]
        planned["test"][class_name] = test_images

    for split_name, classes in planned.items():
        for class_name, images in classes.items():
            destination = output_dir / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            expected_names = {image.name for image in images}
            unexpected = [
                entry.name
                for entry in destination.iterdir()
                if entry.name != ".gitkeep" and entry.name not in expected_names
            ]
            if unexpected:
                raise FileExistsError(
                    f"Unexpected files in {destination}; remove or move them before preparing "
                    "the split: " + ", ".join(sorted(unexpected)[:5])
                )

            for source_image in images:
                destination_image = destination / source_image.name
                if destination_image.exists():
                    try:
                        matches = os.path.samefile(source_image, destination_image)
                        if not matches:
                            matches = filecmp.cmp(source_image, destination_image, shallow=False)
                    except OSError:
                        matches = False
                    if not matches:
                        raise FileExistsError(
                            f"Destination already contains different data: {destination_image}"
                        )
                    continue

                try:
                    os.link(source_image, destination_image)
                except OSError:
                    shutil.copy2(source_image, destination_image)

    return {
        split_name: {class_name: len(images) for class_name, images in classes.items()}
        for split_name, classes in planned.items()
    }


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = 32,
    imagenet_normalization: bool = False,
):
    """Create train, validation, and test CIFAKE loaders."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    data_dir = Path(data_dir)
    if imagenet_normalization:
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
    else:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )
    evaluation_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )

    datasets_by_split = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_transform),
        "val": datasets.ImageFolder(data_dir / "val", transform=evaluation_transform),
        "test": datasets.ImageFolder(data_dir / "test", transform=evaluation_transform),
    }
    expected_mapping = {"fake": 0, "real": 1}
    for split_name, image_dataset in datasets_by_split.items():
        if image_dataset.class_to_idx != expected_mapping:
            raise ValueError(
                f"{split_name} must contain both fake and real class folders; "
                f"found {image_dataset.class_to_idx}"
            )
        if set(image_dataset.targets) != {0, 1}:
            raise ValueError(f"{split_name} must contain images from both classes")

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders, expected_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reproducible CIFAKE train/val/test folders")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Extracted CIFAKE directory containing train/REAL, train/FAKE, test/REAL, test/FAKE",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = prepare_cifake_split(args.source_dir, args.output_dir, args.val_fraction, args.seed)
    for split_name, classes in counts.items():
        print(
            f"{split_name}: "
            + ", ".join(f"{class_name}={count}" for class_name, count in classes.items())
        )
    print(f"Prepared dataset at {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()