"""Tests for MovementAnalyzer module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climbassist.modules.movement_analyzer import MovementAnalyzer
import numpy as np


def _make_keypoints(overrides=None):
    """
    Build a (17, 3) keypoints array with good defaults, then apply overrides.
    All confidence > 0.5 by default, reasonable body position.
    overrides: dict mapping index (int) -> (y, x, conf) tuple.
    """
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
        [0.65, 0.44, 0.85],  # 13 left_knee
        [0.65, 0.56, 0.85],  # 14 right_knee
        [0.80, 0.44, 0.80],  # 15 left_ankle
        [0.80, 0.56, 0.80],  # 16 right_ankle
    ], dtype=np.float32)
    if overrides:
        for idx, vals in overrides.items():
            kp[idx] = vals
    return kp


class TestMovementAnalyzer:
    """Tests for MovementAnalyzer."""

    def test_reset_clears_counts(self):
        """After analyze, reset() zeros everything."""
        ma = MovementAnalyzer()
        kp = _make_keypoints()
        ma.analyze_keypoints(kp)
        assert ma.analyzed_frames > 0
        ma.reset()
        assert ma.analyzed_frames == 0
        assert ma.valid_frames == 0
        for v in ma.issue_counts.values():
            assert v == 0

    def test_low_confidence_ignored(self):
        """Keypoints with nose conf < 0.3 produce no valid_frames increment."""
        ma = MovementAnalyzer()
        kp = _make_keypoints({0: (0.15, 0.50, 0.20)})  # nose conf = 0.20
        issues = ma.analyze_keypoints(kp)
        assert ma.analyzed_frames == 1
        assert ma.valid_frames == 0
        assert issues == []

    def test_wide_stance_detected(self):
        """Keypoints with ankle y-spread > 0.35 triggers wide_stance."""
        ma = MovementAnalyzer()
        # left_ankle y=0.50, right_ankle y=0.90 => diff=0.40 > 0.35
        kp = _make_keypoints({
            15: (0.50, 0.44, 0.80),
            16: (0.90, 0.56, 0.80),
        })
        issues = ma.analyze_keypoints(kp)
        assert "wide_stance" in issues

    def test_hips_away_detected(self):
        """Keypoints with avg hip x > 0.65 triggers hips_away."""
        ma = MovementAnalyzer()
        # left_hip x=0.70, right_hip x=0.75 => avg=0.725 > 0.65
        kp = _make_keypoints({
            11: (0.50, 0.70, 0.90),
            12: (0.50, 0.75, 0.90),
        })
        issues = ma.analyze_keypoints(kp)
        assert "hips_away" in issues

    def test_overreaching_detected(self):
        """Keypoints with arm length > 0.2 triggers overreaching."""
        ma = MovementAnalyzer()
        # left_shoulder x=0.30, left_elbow x=0.60 => arm_len=0.30 > 0.2
        kp = _make_keypoints({
            5: (0.25, 0.30, 0.88),
            7: (0.35, 0.60, 0.82),
        })
        issues = ma.analyze_keypoints(kp)
        assert "overreaching" in issues

    def test_unstable_knees_detected(self):
        """Knees wider than 1.5x hips triggers unstable_knees."""
        ma = MovementAnalyzer()
        # hips: left_hip y=0.50, right_hip y=0.52 => hip_dist=0.02
        # knees: left_knee y=0.55, right_knee y=0.65 => knee_dist=0.10
        # 0.10 > 0.02 * 1.5 = 0.03 => detected
        kp = _make_keypoints({
            11: (0.50, 0.43, 0.90),
            12: (0.52, 0.57, 0.90),
            13: (0.55, 0.42, 0.85),
            14: (0.65, 0.58, 0.85),
        })
        issues = ma.analyze_keypoints(kp)
        assert "unstable_knees" in issues

    def test_no_issues_for_good_form(self):
        """Well-positioned keypoints produce no issues."""
        ma = MovementAnalyzer()
        # Ankles close: diff = |0.80-0.82| = 0.02 < 0.35
        # Hips close to wall: avg x = (0.43+0.47)/2 = 0.45 < 0.65
        # Wrists confident: both > 0.25
        # Knees proportional: knee_dist ~ hip_dist
        # Arms short: elbow close to shoulder
        kp = _make_keypoints({
            5:  (0.25, 0.45, 0.88),  # left_shoulder
            6:  (0.25, 0.55, 0.88),  # right_shoulder
            7:  (0.35, 0.43, 0.82),  # left_elbow  (arm_len = |0.45-0.43|=0.02)
            8:  (0.35, 0.57, 0.82),  # right_elbow (arm_len = |0.55-0.57|=0.02)
            9:  (0.40, 0.42, 0.78),  # left_wrist
            10: (0.40, 0.58, 0.78),  # right_wrist
            11: (0.50, 0.43, 0.90),  # left_hip
            12: (0.50, 0.47, 0.90),  # right_hip   (avg x = 0.45 < 0.65)
            13: (0.65, 0.44, 0.85),  # left_knee
            14: (0.65, 0.48, 0.85),  # right_knee  (knee_dist=0.04, hip_dist=0.07)
            15: (0.80, 0.44, 0.80),  # left_ankle
            16: (0.82, 0.46, 0.80),  # right_ankle (diff=0.02)
        })
        issues = ma.analyze_keypoints(kp)
        assert issues == []

    def test_multiple_issues_detected(self):
        """Keypoints with both wide stance and hips away."""
        ma = MovementAnalyzer()
        kp = _make_keypoints({
            11: (0.50, 0.70, 0.90),  # left_hip x=0.70
            12: (0.50, 0.75, 0.90),  # right_hip x=0.75 => avg=0.725 > 0.65
            15: (0.50, 0.44, 0.80),  # left_ankle y=0.50
            16: (0.90, 0.56, 0.80),  # right_ankle y=0.90 => diff=0.40 > 0.35
        })
        issues = ma.analyze_keypoints(kp)
        assert "wide_stance" in issues
        assert "hips_away" in issues

    def test_get_significant_issues(self):
        """After multiple frames, only frequent issues are returned."""
        ma = MovementAnalyzer()
        wide_kp = _make_keypoints({
            15: (0.50, 0.44, 0.80),
            16: (0.90, 0.56, 0.80),
        })
        good_kp = _make_keypoints()

        # Feed 20 wide-stance frames and 80 good frames
        for _ in range(20):
            ma.analyze_keypoints(wide_kp)
        for _ in range(80):
            ma.analyze_keypoints(good_kp)

        sig = ma.get_significant_issues()
        # threshold = max(1, int(100 * 0.10)) = 10; wide_stance count=20 > 10
        assert "wide_stance" in sig

    def test_get_summary(self):
        """Returns correct dict structure."""
        ma = MovementAnalyzer()
        kp = _make_keypoints()
        ma.analyze_keypoints(kp)
        summary = ma.get_summary()
        assert "analyzed_frames" in summary
        assert "valid_frames" in summary
        assert "detection_rate" in summary
        assert "issue_counts" in summary
        assert "threshold" in summary
        assert "significant_issues" in summary
        assert summary["analyzed_frames"] == 1
        assert summary["valid_frames"] == 1
