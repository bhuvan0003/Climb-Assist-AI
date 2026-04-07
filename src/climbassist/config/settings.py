"""
Feature availability detection.
Checks for optional heavy dependencies at import time.
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from climbassist.modules.pose_visualizer import PoseVisualizer
    POSE_VISUALIZER_AVAILABLE = CV2_AVAILABLE
except ImportError:
    POSE_VISUALIZER_AVAILABLE = False

try:
    from climbassist.modules.route_planner import SafeRouteAnalyzer
    ROUTE_PLANNER_AVAILABLE = True
except ImportError:
    ROUTE_PLANNER_AVAILABLE = False
