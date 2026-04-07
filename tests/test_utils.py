"""Tests for image utility functions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climbassist.utils.image_utils import enhance_frame, convert_bgr_to_rgb
import numpy as np


class TestImageUtils:
    """Tests for image_utils module."""

    def test_enhance_frame_shape(self):
        """Output shape should equal input shape."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = enhance_frame(frame)
        assert result.shape == frame.shape

    def test_convert_bgr_to_rgb_swaps_channels(self):
        """Blue channel (index 0 in BGR) should move to index 2 in RGB."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # Blue channel in BGR
        frame[:, :, 1] = 100  # Green
        frame[:, :, 2] = 50   # Red channel in BGR
        result = convert_bgr_to_rgb(frame)
        # After conversion: R=200, G=100, B=50 => channel 0=50, channel 2=200
        assert result[0, 0, 0] == 50,  "Red channel should be former Blue value"
        assert result[0, 0, 2] == 200, "Blue channel should be former Red value"
        assert result[0, 0, 1] == 100, "Green channel should be unchanged"

    def test_enhance_frame_brightness(self):
        """Enhanced frame should have higher mean pixel value than input."""
        rng = np.random.RandomState(42)
        frame = rng.randint(20, 100, (100, 100, 3), dtype=np.uint8)
        result = enhance_frame(frame, alpha=1.3, beta=30)
        assert result.mean() > frame.mean()

    def test_enhance_frame_dtype(self):
        """Output dtype should be uint8."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = enhance_frame(frame)
        assert result.dtype == np.uint8

    def test_convert_bgr_to_rgb_preserves_shape(self):
        """Conversion should not change array shape."""
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        result = convert_bgr_to_rgb(frame)
        assert result.shape == frame.shape
