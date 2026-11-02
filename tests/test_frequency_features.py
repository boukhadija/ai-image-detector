"""Tests for the FFT/DCT feature extractor."""

from __future__ import annotations

import unittest

import numpy as np

from src.frequency_features import extract_frequency_features


class FrequencyFeatureTests(unittest.TestCase):
    def test_extracts_combined_fft_and_dct_features(self) -> None:
        images = np.zeros((2, 3, 32, 32), dtype=np.float32)
        images[1, :, ::2, ::2] = 1
        features = extract_frequency_features(images)

        self.assertEqual(features.shape, (2, 3 * (16 + 8**2)))
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(np.array_equal(features[0], features[1]))

    def test_empty_batch_preserves_feature_width(self) -> None:
        images = np.empty((0, 3, 16, 16), dtype=np.float32)

        features = extract_frequency_features(images, fft_bins=4, dct_size=4)

        self.assertEqual(features.shape, (0, 3 * (4 + 4**2)))

    def test_empty_radial_bands_produce_finite_features(self) -> None:
        images = np.ones((1, 3, 8, 8), dtype=np.float32)

        features = extract_frequency_features(images, fft_bins=16, dct_size=8)

        self.assertTrue(np.isfinite(features).all())

    def test_rejects_non_rgb_or_non_square_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            extract_frequency_features(np.zeros((1, 1, 32, 32), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "square"):
            extract_frequency_features(np.zeros((1, 3, 32, 16), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
