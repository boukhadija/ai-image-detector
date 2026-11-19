"""Save a side-by-side image, FFT spectrum, radial profile, and DCT view."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.frequency_features import (
    DEFAULT_DCT_SIZE,
    DEFAULT_FFT_BINS,
    extract_frequency_features,
)


def _heatmap(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    span = float(values.max() - values.min())
    normalized = (values - values.min()) / span if span > 0 else np.zeros_like(values)
    pixels = np.uint8(normalized * 255)
    return Image.fromarray(pixels, mode="L").resize(size, Image.Resampling.NEAREST)


def create_visualization(
    image_path: Path,
    output_path: Path,
    image_size: int = 32,
    fft_bins: int = DEFAULT_FFT_BINS,
    dct_size: int = DEFAULT_DCT_SIZE,
) -> None:
    """Create a PNG showing the input image and its frequency features."""
    if image_size < max(DEFAULT_DCT_SIZE, dct_size) or dct_size < 1 or fft_bins < 1:
        raise ValueError("image-size must be >= 8 and >= positive dct-size; fft-bins must be positive")

    with Image.open(image_path) as source:
        resized = source.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    image_batch = np.transpose(rgb, (2, 0, 1))[None, ...]
    features = extract_frequency_features(image_batch, fft_bins, dct_size)[0]

    spectrum = np.fft.fftshift(
        np.fft.fft2(image_batch, axes=(-2, -1)), axes=(-2, -1)
    )
    log_magnitude = np.log1p(np.abs(spectrum))[0].mean(axis=0)
    radial_features = features[: 3 * fft_bins].reshape(3, fft_bins).mean(axis=0)
    dct_features = features[3 * fft_bins :].reshape(3, dct_size, dct_size)
    dct_magnitude = np.abs(dct_features).mean(axis=0)

    panel_size = 240
    margin = 30
    label_height = 30
    width = margin * 2 + panel_size * 3 + 30 * 2
    height = margin * 2 + label_height * 2 + panel_size + 190
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    panels = [
        ("Input (resized)", resized),
        ("FFT log magnitude", _heatmap(log_magnitude, (panel_size, panel_size))),
        ("DCT low frequencies (|value|)", _heatmap(dct_magnitude, (panel_size, panel_size))),
    ]

    x = margin
    y = margin + label_height
    for label, panel in panels:
        draw.text((x, margin), label, fill="black")
        canvas.paste(panel, (x, y))
        draw.rectangle(
            (x, y, x + panel_size - 1, y + panel_size - 1),
            outline="gray",
            width=1,
        )
        x += panel_size + 30

    chart_top = y + panel_size + 55
    chart_left = margin + 35
    chart_width = width - 2 * margin - 70
    chart_height = 115
    draw.text((margin, chart_top - 28), "Mean radial FFT feature by frequency band", fill="black")
    draw.line(
        (chart_left, chart_top, chart_left, chart_top + chart_height),
        fill="black",
        width=1,
    )
    draw.line(
        (chart_left, chart_top + chart_height, chart_left + chart_width, chart_top + chart_height),
        fill="black",
        width=1,
    )
    value_span = float(radial_features.max() - radial_features.min())
    if value_span > 0:
        normalized = (radial_features - radial_features.min()) / value_span
    else:
        normalized = np.zeros_like(radial_features)
    points = [
        (
            chart_left + int(index * chart_width / max(1, fft_bins - 1)),
            chart_top + chart_height - int(value * (chart_height - 5)),
        )
        for index, value in enumerate(normalized)
    ]
    if len(points) > 1:
        draw.line(points, fill=(40, 100, 200), width=3)
    for px, py in points:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(40, 100, 200))
    draw.text((chart_left, chart_top + chart_height + 8), "lower frequency", fill="black")
    draw.text(
        (chart_left + chart_width - 100, chart_top + chart_height + 8),
        "higher frequency",
        fill="black",
    )
    draw.text(
        (margin, height - margin - 18),
        f"Features: {3 * fft_bins} FFT + {3 * dct_size**2} DCT = {len(features)} total",
        fill="black",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize FFT and DCT features extracted from one image"
    )
    parser.add_argument("image", type=Path, help="Image file to visualize")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frequency_visualization.png"),
        help="Output PNG path",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--fft-bins", type=int, default=DEFAULT_FFT_BINS)
    parser.add_argument("--dct-size", type=int, default=DEFAULT_DCT_SIZE)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the saved image with the operating system's default viewer",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")

    create_visualization(
        args.image,
        args.output,
        image_size=args.image_size,
        fft_bins=args.fft_bins,
        dct_size=args.dct_size,
    )
    print(f"Visualization saved to {args.output.resolve()}")
    if args.show:
        with Image.open(args.output) as result:
            result.show()


if __name__ == "__main__":
    main()
