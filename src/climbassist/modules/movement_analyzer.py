"""
Movement Analyzer Module
=========================
Analyzes climbing movement from MoveNet keypoints.
Detects technique issues without any Streamlit dependency.
"""

import numpy as np
from typing import Dict, List, Tuple

from climbassist.config.constants import (
    POSE_CONFIDENCE_THRESHOLD,
    WIDE_STANCE_THRESHOLD,
    HIPS_AWAY_THRESHOLD,
    POOR_HAND_CONFIDENCE,
    UNSTABLE_KNEES_RATIO,
    OVERREACHING_THRESHOLD,
    ISSUE_DETECTION_RATE,
)


class MovementAnalyzer:
    """Analyze climbing technique from MoveNet keypoint sequences."""

    ISSUE_KEYS = [
        "wide_stance",
        "hips_away",
        "poor_hand_use",
        "unstable_knees",
        "overreaching",
    ]

    def __init__(self):
        self.issue_counts: Dict[str, int] = {k: 0 for k in self.ISSUE_KEYS}
        self.analyzed_frames = 0
        self.valid_frames = 0

    def reset(self) -> None:
        """Reset counters for a new analysis session."""
        self.issue_counts = {k: 0 for k in self.ISSUE_KEYS}
        self.analyzed_frames = 0
        self.valid_frames = 0

    def analyze_keypoints(self, keypoints: np.ndarray) -> List[str]:
        """
        Analyze a single frame's keypoints and return detected issues.

        Args:
            keypoints: (17, 3) array [y, x, confidence] from MoveNet.

        Returns:
            List of issue keys detected in this frame.
        """
        self.analyzed_frames += 1
        detected: List[str] = []

        # Nose confidence check — is a person visible?
        if keypoints[0, 2] < POSE_CONFIDENCE_THRESHOLD:
            return detected

        self.valid_frames += 1

        # Extract keypoints
        left_ankle, right_ankle = keypoints[15], keypoints[16]
        left_hip, right_hip = keypoints[11], keypoints[12]
        left_wrist, right_wrist = keypoints[9], keypoints[10]
        left_knee, right_knee = keypoints[13], keypoints[14]
        left_elbow, right_elbow = keypoints[7], keypoints[8]
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]

        # Issue 1: Wide Stance
        if abs(left_ankle[0] - right_ankle[0]) > WIDE_STANCE_THRESHOLD:
            self.issue_counts["wide_stance"] += 1
            detected.append("wide_stance")

        # Issue 2: Hips Away from Wall
        if (left_hip[1] + right_hip[1]) / 2 > HIPS_AWAY_THRESHOLD:
            self.issue_counts["hips_away"] += 1
            detected.append("hips_away")

        # Issue 3: Poor Hand Usage
        if left_wrist[2] < POOR_HAND_CONFIDENCE or right_wrist[2] < POOR_HAND_CONFIDENCE:
            self.issue_counts["poor_hand_use"] += 1
            detected.append("poor_hand_use")

        # Issue 4: Unstable Knees
        if left_knee[2] > POSE_CONFIDENCE_THRESHOLD and right_knee[2] > POSE_CONFIDENCE_THRESHOLD:
            knee_dist = abs(left_knee[0] - right_knee[0])
            hip_dist = abs(left_hip[0] - right_hip[0])
            if hip_dist > 0 and knee_dist > hip_dist * UNSTABLE_KNEES_RATIO:
                self.issue_counts["unstable_knees"] += 1
                detected.append("unstable_knees")

        # Issue 5: Overreaching
        if left_elbow[2] > POSE_CONFIDENCE_THRESHOLD and right_elbow[2] > POSE_CONFIDENCE_THRESHOLD:
            left_arm = abs(left_shoulder[1] - left_elbow[1])
            right_arm = abs(right_shoulder[1] - right_elbow[1])
            if left_arm > OVERREACHING_THRESHOLD or right_arm > OVERREACHING_THRESHOLD:
                self.issue_counts["overreaching"] += 1
                detected.append("overreaching")

        return detected

    def get_threshold(self) -> int:
        """Minimum count for an issue to be considered significant."""
        return max(1, int(self.valid_frames * ISSUE_DETECTION_RATE))

    def get_significant_issues(self) -> List[str]:
        """Return issues that exceed the significance threshold."""
        threshold = self.get_threshold()
        return [k for k, v in self.issue_counts.items() if v > threshold]

    def get_summary(self) -> Dict:
        """Return full analysis summary."""
        threshold = self.get_threshold()
        return {
            "analyzed_frames": self.analyzed_frames,
            "valid_frames": self.valid_frames,
            "detection_rate": (
                int((self.valid_frames / self.analyzed_frames) * 100)
                if self.analyzed_frames > 0 else 0
            ),
            "issue_counts": dict(self.issue_counts),
            "threshold": threshold,
            "significant_issues": self.get_significant_issues(),
        }
