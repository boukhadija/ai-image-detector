"""Deterministic image corruptions for frequency-classifier robustness tests."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter

# Levels are intentionally fixed so repeated runs are directly comparable.
CORRUPTION_LEVELS = {
    "jpeg": (95, 80, 60, 40),
    "resize": (0.75, 0.5, 0.25),
    "blur": (0.5, 1.0, 2.0),
    "screenshot": (0.9, 0.75, 0.6),
}


def apply_corruption(
    image: Image.Image,
    corruption: str,
    severity: int | float | None,
    image_size: int,
) -> Image.Image:
    """Resize an image to model input size and apply one reproducible corruption.

    The screenshot case is a synthetic display-capture proxy: the image is
    centered in a framed 16:9 canvas with margins, then resized to model input.
    """
    if image_size < 1:
        raise ValueError("image_size must be positive")
    image = image.convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    if corruption == "clean":
        if severity is not None:
            raise ValueError("clean images do not have a severity")
        return image
    if corruption == "jpeg":
        if not isinstance(severity, int) or not 1 <= severity <= 100:
            raise ValueError("JPEG quality must be an integer from 1 to 100")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=severity)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if corruption == "resize":
        if not isinstance(severity, (int, float)) or not 0 < severity < 1:
            raise ValueError("resize scale must be between 0 and 1")
        reduced_size = max(1, round(image_size * severity))
        reduced = image.resize(
            (reduced_size, reduced_size), Image.Resampling.BILINEAR
        )
        return reduced.resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
    if corruption == "blur":
        if not isinstance(severity, (int, float)) or severity < 0:
            raise ValueError("blur radius cannot be negative")
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    if corruption == "screenshot":
        if not isinstance(severity, (int, float)) or not 0 < severity <= 1:
            raise ValueError("screenshot display scale must be between 0 and 1")
        canvas_height = max(96, image_size * 4)
        canvas_width = round(canvas_height * 16 / 9)
        canvas = Image.new("RGB", (canvas_width, canvas_height), (232, 235, 240))
        draw = ImageDraw.Draw(canvas)
        header_height = max(8, canvas_height // 12)
        draw.rectangle(
            (0, 0, canvas_width, header_height),
            fill=(205, 210, 218),
        )
        content_top = header_height + canvas_height // 20
        content_height = canvas_height - content_top - canvas_height // 20
        display_side = max(
            1, round(min(canvas_width * 0.9, content_height * 0.9) * severity)
        )
        display = image.resize(
            (display_side, display_side), Image.Resampling.BILINEAR
        )
        left = (canvas_width - display_side) // 2
        top = content_top + (content_height - display_side) // 2
        canvas.paste(display, (left, top))
        draw.rectangle(
            (left, top, left + display_side - 1, top + display_side - 1),
            outline=(110, 115, 125),
            width=max(1, canvas_width // 500),
        )
        return canvas.resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
    raise ValueError(f"Unknown corruption: {corruption}")
