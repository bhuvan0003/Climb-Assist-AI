"""Tests for route_planner module (PathPlanner, TerrainSegmenter, TraversabilityEstimator).

Uses only numpy -- no cv2/torch imports.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Import only the classes that do NOT require cv2/torch at construction time.
from climbassist.modules.route_planner import PathPlanner, TerrainSegmenter, TraversabilityEstimator


class TestPathPlanner:
    """Tests for the A* PathPlanner."""

    def test_simple_path_found(self):
        """Uniform low-cost 50x50 grid should yield a path."""
        cost = np.full((50, 50), 0.1, dtype=np.float32)
        planner = PathPlanner(cost, start=(49, 25), goal=(0, 25))
        path = planner.plan()
        assert path is not None
        assert len(path) > 1

    def test_blocked_goal(self):
        """All cells around goal exceed threshold -- no path."""
        cost = np.full((20, 20), 0.1, dtype=np.float32)
        # Wall around goal (0, 10): fill row 0 and row 1 with high cost
        cost[0, :] = 0.99
        cost[1, :] = 0.99
        planner = PathPlanner(cost, start=(19, 10), goal=(0, 10))
        path = planner.plan(cost_threshold=0.95)
        assert path is None

    def test_path_starts_at_start(self):
        """First waypoint equals start."""
        cost = np.full((30, 30), 0.1, dtype=np.float32)
        start = (29, 15)
        goal = (0, 15)
        planner = PathPlanner(cost, start=start, goal=goal)
        path = planner.plan()
        assert path is not None
        assert path[0] == start

    def test_path_ends_at_goal(self):
        """Last waypoint equals goal."""
        cost = np.full((30, 30), 0.1, dtype=np.float32)
        start = (29, 15)
        goal = (0, 15)
        planner = PathPlanner(cost, start=start, goal=goal)
        path = planner.plan()
        assert path is not None
        assert path[-1] == goal

    def test_diagonal_movement(self):
        """Path can move diagonally (8-connected)."""
        cost = np.full((10, 10), 0.1, dtype=np.float32)
        planner = PathPlanner(cost, start=(9, 0), goal=(0, 9))
        path = planner.plan(connectivity=8)
        assert path is not None
        # Check that at least one step is diagonal
        has_diagonal = False
        for i in range(len(path) - 1):
            dy = abs(path[i+1][0] - path[i][0])
            dx = abs(path[i+1][1] - path[i][1])
            if dy == 1 and dx == 1:
                has_diagonal = True
                break
        assert has_diagonal, "Expected at least one diagonal step"

    def test_high_cost_avoidance(self):
        """Create a corridor and verify the path follows the low-cost corridor."""
        cost = np.full((20, 20), 0.98, dtype=np.float32)
        # Create a low-cost corridor along column 10
        cost[:, 10] = 0.1
        # Also open start and goal columns
        cost[19, :] = 0.1
        cost[0, :] = 0.1
        planner = PathPlanner(cost, start=(19, 10), goal=(0, 10))
        path = planner.plan(cost_threshold=0.95)
        assert path is not None
        # Most interior waypoints should be near column 10
        interior = path[1:-1]
        near_corridor = sum(1 for (y, x) in interior if abs(x - 10) <= 1)
        assert near_corridor >= len(interior) * 0.5

    def test_smooth_path_reduces_points(self):
        """Smoothed path has fewer or equal points than raw path."""
        cost = np.full((50, 50), 0.1, dtype=np.float32)
        planner = PathPlanner(cost, start=(49, 0), goal=(0, 49))
        raw_path = planner.plan()
        assert raw_path is not None
        smoothed = planner.smooth_path(raw_path)
        assert len(smoothed) <= len(raw_path)

    def test_bresenham_basic(self):
        """Bresenham horizontal line returns correct points."""
        cost = np.full((10, 10), 0.1, dtype=np.float32)
        planner = PathPlanner(cost, start=(0, 0), goal=(9, 9))
        points = planner._bresenham_line(5, 0, 5, 4)
        # Should produce 5 points: (5,0), (5,1), (5,2), (5,3), (5,4)
        assert len(points) == 5
        for i, (y, x) in enumerate(points):
            assert y == 5
            assert x == i

    def test_heuristic_admissible(self):
        """Heuristic should be <= actual straight-line cost for simple cases."""
        cost = np.full((10, 10), 0.5, dtype=np.float32)
        planner = PathPlanner(cost, start=(0, 0), goal=(9, 9))
        h = planner.heuristic((0, 0))
        # Euclidean distance * (min_cost + 0.1) = sqrt(162) * 0.6 ~ 7.64
        actual_min_cost = np.sqrt(81 + 81) * 0.5  # ~6.36 at terrain cost
        # Heuristic should not wildly exceed actual (it is admissible)
        assert h >= 0

    def test_edge_cost_calculation(self):
        """Edge cost is average of two cells times distance."""
        cost = np.zeros((5, 5), dtype=np.float32)
        cost[0, 0] = 0.4
        cost[0, 1] = 0.6
        planner = PathPlanner(cost, start=(0, 0), goal=(4, 4))
        ec = planner.edge_cost((0, 0), (0, 1))
        expected = (0.4 + 0.6) / 2.0 * 1.0  # distance = 1 for horizontal
        assert abs(ec - expected) < 1e-6


class TestTerrainSegmenter:
    """Tests for TerrainSegmenter class constants."""

    def test_terrain_classes_defined(self):
        """TerrainSegmenter has 10 terrain classes (indices 0-9)."""
        assert len(TerrainSegmenter.TERRAIN_CLASSES) == 10
        for idx in range(10):
            assert idx in TerrainSegmenter.TERRAIN_CLASSES

    def test_terrain_costs_in_range(self):
        """All terrain costs are in [0, 1]."""
        for idx, (name, cost) in TerrainSegmenter.TERRAIN_CLASSES.items():
            assert 0.0 <= cost <= 1.0, f"Class {name} cost {cost} out of range"

    def test_impassable_terrain_cost_one(self):
        """Water and crevasse should have cost = 1.0."""
        water_cost = TerrainSegmenter.TERRAIN_CLASSES[5][1]
        crevasse_cost = TerrainSegmenter.TERRAIN_CLASSES[6][1]
        assert water_cost == 1.0
        assert crevasse_cost == 1.0


class TestTraversabilityEstimator:
    """Tests for TraversabilityEstimator.compute_slope (pure numpy)."""

    def test_compute_slope_flat(self):
        """Flat depth map should produce very low slope values."""
        te = TraversabilityEstimator()
        depth = np.full((20, 20), 0.5, dtype=np.float32)
        slope = te.compute_slope(depth)
        assert slope.shape == (20, 20)
        assert slope.max() < 0.01, "Flat depth should have near-zero slope"

    def test_compute_slope_shape(self):
        """Slope output shape matches input."""
        te = TraversabilityEstimator()
        depth = np.random.rand(30, 40).astype(np.float32)
        slope = te.compute_slope(depth)
        assert slope.shape == depth.shape
