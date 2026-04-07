"""Tests for PoseVisualizer and ImprovementAdvisor (no cv2 dependency)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# We import the module itself to access class-level constants without
# instantiating PoseVisualizer (which requires cv2).
from climbassist.modules import pose_visualizer as pv_module
from climbassist.modules.pose_visualizer import ImprovementAdvisor


class TestImprovementAdvisor:
    """Tests for ImprovementAdvisor.get_advice()."""

    def test_advisor_known_issue(self):
        """get_advice for a known issue returns an advice dict."""
        result = ImprovementAdvisor.get_advice(["wide_stance"])
        assert result["total_issues"] == 1
        assert "wide_stance" in result["issues"]
        tip = result["issues"]["wide_stance"]
        assert "issue" in tip
        assert "description" in tip

    def test_advisor_multiple_issues(self):
        """Returns advice for each known issue."""
        issues = ["wide_stance", "overreaching", "unstable_knees"]
        result = ImprovementAdvisor.get_advice(issues)
        assert result["total_issues"] == 3
        for issue in issues:
            assert issue in result["issues"]

    def test_advisor_unknown_issue(self):
        """get_advice for an unknown issue key returns empty issues dict."""
        result = ImprovementAdvisor.get_advice(["nonexistent_issue"])
        assert result["total_issues"] == 1
        assert len(result["issues"]) == 0

    def test_advisor_empty_list(self):
        """get_advice with empty list returns total_issues=0."""
        result = ImprovementAdvisor.get_advice([])
        assert result["total_issues"] == 0
        assert len(result["issues"]) == 0

    def test_advisor_person_not_detected(self):
        """get_advice for 'person_not_detected' returns empty issues."""
        result = ImprovementAdvisor.get_advice(["person_not_detected"])
        assert len(result["issues"]) == 0

    def test_advisor_structure(self):
        """Each returned issue has required keys."""
        required_keys = {"issue", "description", "why_bad",
                         "key_points_to_avoid", "improvement_steps", "drill"}
        result = ImprovementAdvisor.get_advice(["wide_stance"])
        tip = result["issues"]["wide_stance"]
        for key in required_keys:
            assert key in tip, f"Missing key '{key}' in advice for wide_stance"

    def test_advisor_all_known_issues(self):
        """All keys in IMPROVEMENT_TIPS can be retrieved."""
        all_issues = list(ImprovementAdvisor.IMPROVEMENT_TIPS.keys())
        result = ImprovementAdvisor.get_advice(all_issues)
        assert result["total_issues"] == len(all_issues)
        assert len(result["issues"]) == len(all_issues)

    def test_advisor_none_input(self):
        """get_advice with None returns total_issues=0."""
        result = ImprovementAdvisor.get_advice(None)
        assert result["total_issues"] == 0


class TestPoseVisualizerClassAttributes:
    """Test PoseVisualizer class-level constants without instantiation."""

    def test_keypoint_names_count(self):
        """PoseVisualizer.KEYPOINT_NAMES has 17 entries."""
        assert len(pv_module.PoseVisualizer.KEYPOINT_NAMES) == 17

    def test_skeleton_edges_count(self):
        """PoseVisualizer.SKELETON_EDGES has 16 entries."""
        assert len(pv_module.PoseVisualizer.SKELETON_EDGES) == 16

    def test_keypoint_names_are_strings(self):
        """All keypoint names are non-empty strings."""
        for name in pv_module.PoseVisualizer.KEYPOINT_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_skeleton_edges_are_valid_tuples(self):
        """Each skeleton edge is a tuple of two valid keypoint indices."""
        num_keypoints = len(pv_module.PoseVisualizer.KEYPOINT_NAMES)
        for edge in pv_module.PoseVisualizer.SKELETON_EDGES:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert 0 <= edge[0] < num_keypoints
            assert 0 <= edge[1] < num_keypoints

    def test_color_constants_defined(self):
        """Color constants are 3-element tuples."""
        for attr in ("COLOR_GOOD", "COLOR_WARNING", "COLOR_BAD", "COLOR_SKELETON"):
            color = getattr(pv_module.PoseVisualizer, attr)
            assert isinstance(color, tuple)
            assert len(color) == 3
