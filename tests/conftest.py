"""
Shared test fixtures for ClimbAssist AI test suite.
All fixtures use plain numpy -- no cv2/tensorflow/torch imports.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_keypoints():
    """
    Returns a (17, 3) numpy array with realistic MoveNet keypoints.
    Format per row: [y, x, confidence].
    All confidence values > 0.5, reasonable body positions for a climber.
    """
    # Keypoint order: nose, left_eye, right_eye, left_ear, right_ear,
    #   left_shoulder, right_shoulder, left_elbow, right_elbow,
    #   left_wrist, right_wrist, left_hip, right_hip,
    #   left_knee, right_knee, left_ankle, right_ankle
    kp = np.array([
        [0.15, 0.50, 0.90],  # 0  nose
        [0.13, 0.48, 0.85],  # 1  left_eye
        [0.13, 0.52, 0.85],  # 2  right_eye
        [0.14, 0.45, 0.80],  # 3  left_ear
        [0.14, 0.55, 0.80],  # 4  right_ear
        [0.25, 0.40, 0.88],  # 5  left_shoulder
        [0.25, 0.60, 0.88],  # 6  right_shoulder
        [0.35, 0.38, 0.82],  # 7  left_elbow
        [0.35, 0.62, 0.82],  # 8  right_elbow
        [0.40, 0.35, 0.78],  # 9  left_wrist
        [0.40, 0.65, 0.78],  # 10 right_wrist
        [0.50, 0.43, 0.90],  # 11 left_hip
        [0.50, 0.57, 0.90],  # 12 right_hip
        [0.65, 0.42, 0.85],  # 13 left_knee
        [0.65, 0.58, 0.85],  # 14 right_knee
        [0.80, 0.44, 0.80],  # 15 left_ankle
        [0.80, 0.56, 0.80],  # 16 right_ankle
    ], dtype=np.float32)
    return kp


@pytest.fixture
def sample_keypoints_wide_stance():
    """
    Keypoints where ankles are spread > 0.35 apart (y-axis distance).
    Left ankle y=0.55, right ankle y=0.95 => diff = 0.40 > 0.35.
    """
    kp = np.array([
        [0.15, 0.50, 0.90],  # nose
        [0.13, 0.48, 0.85],  # left_eye
        [0.13, 0.52, 0.85],  # right_eye
        [0.14, 0.45, 0.80],  # left_ear
        [0.14, 0.55, 0.80],  # right_ear
        [0.25, 0.40, 0.88],  # left_shoulder
        [0.25, 0.60, 0.88],  # right_shoulder
        [0.35, 0.38, 0.82],  # left_elbow
        [0.35, 0.62, 0.82],  # right_elbow
        [0.40, 0.35, 0.78],  # left_wrist
        [0.40, 0.65, 0.78],  # right_wrist
        [0.50, 0.43, 0.90],  # left_hip
        [0.50, 0.57, 0.90],  # right_hip
        [0.65, 0.42, 0.85],  # left_knee
        [0.65, 0.58, 0.85],  # right_knee
        [0.55, 0.44, 0.80],  # left_ankle  (y=0.55)
        [0.95, 0.56, 0.80],  # right_ankle (y=0.95) => diff=0.40
    ], dtype=np.float32)
    return kp


@pytest.fixture
def sample_keypoints_low_confidence():
    """All keypoints have confidence < 0.3."""
    kp = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        kp[i] = [0.5, 0.5, 0.20]
    return kp


@pytest.fixture
def sample_frame():
    """Returns a synthetic 480x640x3 uint8 BGR numpy array."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_cost_map():
    """Returns a 100x100 float32 array with values in [0, 1]."""
    rng = np.random.RandomState(42)
    return rng.rand(100, 100).astype(np.float32)
