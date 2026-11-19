"""Tests for deterministic robustness image transformations."""

from __future__ import annotations

import unittest

from PIL import Image

from src.robustness import CORRUPTION_LEVELS, apply_corruption


class RobustnessTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (80, 48), (100, 150, 200))

    def test_all_conditions_return_model_sized_rgb_images(self) -> None:
        self.assertEqual(
            apply_corruption(self.image, "clean", None, 32).size,
            (32, 32),
        )
        for corruption, severities in CORRUPTION_LEVELS.items():
            for severity in severities:
                with self.subTest(corruption=corruption, severity=severity):
                    transformed = apply_corruption(
                        self.image, corruption, severity, 32
                    )
                    self.assertEqual(transformed.mode, "RGB")
                    self.assertEqual(transformed.size, (32, 32))

    def test_unknown_corruption_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown corruption"):
            apply_corruption(self.image, "unknown", 1, 32)

    def test_invalid_corruption_severity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "JPEG quality"):
            apply_corruption(self.image, "jpeg", 0, 32)
        with self.assertRaisesRegex(ValueError, "resize scale"):
            apply_corruption(self.image, "resize", 1.0, 32)


if __name__ == "__main__":
    unittest.main()
