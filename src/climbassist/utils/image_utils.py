"""Image processing utility functions."""

import numpy as np


def enhance_frame(frame: np.ndarray, alpha: float = 1.2, beta: int = 20) -> np.ndarray:
    """Enhance frame contrast and brightness."""
    try:
        import cv2
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    except ImportError:
        # Fallback: simple numpy-based enhancement
        enhanced = np.clip(frame.astype(np.float32) * alpha + beta, 0, 255)
        return enhanced.astype(np.uint8)


def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB."""
    try:
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except ImportError:
        return frame[:, :, ::-1].copy()


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to given dimensions."""
    try:
        import cv2
        return cv2.resize(frame, (width, height))
    except ImportError:
        raise ImportError("OpenCV required for frame resizing")
