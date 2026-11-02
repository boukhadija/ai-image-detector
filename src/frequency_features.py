"""Extract compact FFT and DCT features from CIFAKE image batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASS_TO_IDX = {"fake": 0, "real": 1}
DEFAULT_FFT_BINS = 16
DEFAULT_DCT_SIZE = 8


def create_frequency_dataloaders(
    data_dir: Path,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = 32,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """Create resized RGB image loaders without CNN-specific normalization."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if image_size < DEFAULT_DCT_SIZE:
        raise ValueError(f"image_size must be at least {DEFAULT_DCT_SIZE}")

    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    datasets_by_split = {
        split: datasets.ImageFolder(Path(data_dir) / split, transform=transform)
        for split in ("train", "val", "test")
    }
    for split_name, image_dataset in datasets_by_split.items():
        if image_dataset.class_to_idx != CLASS_TO_IDX:
            raise ValueError(
                f"{split_name} must contain both fake and real class folders; "
                f"found {image_dataset.class_to_idx}"
            )
        if set(image_dataset.targets) != set(CLASS_TO_IDX.values()):
            raise ValueError(f"{split_name} must contain images from both classes")

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        **{
            split: DataLoader(
                datasets_by_split[split],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            for split in ("val", "test")
        },
    }
    return loaders, CLASS_TO_IDX.copy()


def extract_frequency_features(
    images: np.ndarray,
    fft_bins: int = DEFAULT_FFT_BINS,
    dct_size: int = DEFAULT_DCT_SIZE,
) -> np.ndarray:
    """Combine radial log-FFT magnitudes and low-frequency 2D DCT coefficients.

    Args:
        images: Float RGB batches shaped ``(batch, 3, height, width)``.
        fft_bins: Number of radial bands used to summarize each channel's FFT.
        dct_size: Side length of the low-frequency DCT block retained per channel.
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (batch, 3, height, width)")
    if images.shape[2] != images.shape[3]:
        raise ValueError("images must be square")
    if fft_bins < 1 or dct_size < 1:
        raise ValueError("fft_bins and dct_size must be positive")
    if dct_size > images.shape[2]:
        raise ValueError("dct_size cannot exceed the image dimensions")
    if images.shape[0] == 0:
        return np.empty((0, 3 * (fft_bins + dct_size**2)), dtype=np.float32)

    image_size = images.shape[2]
    spectrum = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    log_magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    frequencies = np.fft.fftshift(np.fft.fftfreq(image_size))
    yy, xx = np.meshgrid(frequencies, frequencies, indexing="ij")
    radial_bins = np.minimum(
        (np.hypot(xx, yy) / np.hypot(xx, yy).max() * fft_bins).astype(np.int32),
        fft_bins - 1,
    )
    fft_features = np.stack(
        [
            log_magnitude[..., radial_bins == band].mean(axis=-1)
            if np.any(radial_bins == band)
            else np.zeros(images.shape[:2], dtype=np.float32)
            for band in range(fft_bins)
        ],
        axis=-1,
    ).reshape(images.shape[0], -1)

    positions = np.arange(image_size, dtype=np.float32)
    frequencies = np.arange(image_size, dtype=np.float32)[:, None]
    basis = np.cos(np.pi * (positions + 0.5) * frequencies / image_size)
    basis[0] *= np.sqrt(1 / image_size)
    if image_size > 1:
        basis[1:] *= np.sqrt(2 / image_size)
    dct = np.einsum("ui,bcij,vj->bcuv", basis, images, basis, optimize=True)
    dct_features = dct[:, :, :dct_size, :dct_size].reshape(images.shape[0], -1)

    return np.concatenate((fft_features, dct_features), axis=1).astype(np.float32)


def extract_loader_features(
    loader: DataLoader,
    fft_bins: int = DEFAULT_FFT_BINS,
    dct_size: int = DEFAULT_DCT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from an image loader."""
    feature_batches = []
    label_batches = []
    for images, labels in loader:
        feature_batches.append(
            extract_frequency_features(images.detach().cpu().numpy(), fft_bins, dct_size)
        )
        label_batches.append(labels.numpy())

    if not feature_batches:
        raise ValueError("A data loader is empty")
    return np.concatenate(feature_batches), np.concatenate(label_batches)
