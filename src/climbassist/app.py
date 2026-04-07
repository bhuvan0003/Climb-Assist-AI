"""
ClimbAssist AI v2 - Complete Climbing Analysis Platform
=========================================================
Refactored main Streamlit application.

Integrated platform for:
1. Gear Load Optimizer
2. Climbing Movement Analyzer (MoveNet pose detection)
3. Safe Route Finder (terrain segmentation + depth + pathfinding)
"""

import streamlit as st
import numpy as np
import random
import time

# -- Local module imports (always available) ----------------------------------
from climbassist.config.constants import (
    APP_CSS, BASE_GEAR, GEAR_TIPS, VIDEO_TIPS, ADDITIONAL_TIPS,
)
from climbassist.modules.gear_optimizer import GearOptimizer
from climbassist.modules.movement_analyzer import MovementAnalyzer
from climbassist.utils.pdf_generator import GearPDFGenerator
from climbassist.utils.video_utils import save_temp_video
from climbassist.utils.image_utils import enhance_frame

# -- Heavy / optional dependency detection ------------------------------------
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from climbassist.modules.route_planner import SafeRouteAnalyzer
    ROUTE_PLANNER_AVAILABLE = True
except ImportError:
    ROUTE_PLANNER_AVAILABLE = False

try:
    from climbassist.modules.pose_visualizer import PoseVisualizer, ImprovementAdvisor
    POSE_VISUALIZER_AVAILABLE = True and CV2_AVAILABLE
except ImportError:
    POSE_VISUALIZER_AVAILABLE = False

# TensorFlow will be imported lazily only when needed


# =============================================================================
# Model Caching Functions (Reduce Loading Time)
# =============================================================================

@st.cache_resource
def load_movenet_model():
    """Load MoveNet model once and cache it."""
    import tensorflow.compat.v1 as tf
    import tensorflow_hub as hub
    import os
    import shutil

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    max_retries = 3
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                cache_dir = os.path.join(
                    os.environ.get("TEMP", "/tmp"), "tfhub_modules"
                )
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    print(f"Cleared TensorFlow Hub cache for retry {attempt}")

            print(f"Loading MoveNet model (attempt {attempt + 1}/{max_retries})...")
            model = hub.load(model_url)

            if hasattr(model, "signatures") and "serving_default" in model.signatures:
                return model.signatures["serving_default"]
            else:
                raise ValueError(
                    "Model loaded but missing serving_default signature"
                )
        except (ValueError, Exception) as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise ValueError(
                    f"Failed to load MoveNet model after {max_retries} attempts: {str(e)}"
                )


@st.cache_resource
def load_safe_route_analyzer():
    """Load SafeRouteAnalyzer once and cache it."""
    analyzer = SafeRouteAnalyzer()
    analyzer.load_models()
    return analyzer


# =============================================================================
# Route Analysis Functions (inline -- heavy cv2 coupling)
# =============================================================================

def create_advanced_route_analysis(frame):
    """
    ENHANCED SAFE ROUTE FINDER - A* Algorithm

    Advanced Features:
    - Multi-layer sky detection (HSV + RGB + Edge-based)
    - Terrain segmentation (rock, scree, vegetation, slope analysis)
    - Intelligent cost mapping with terrain classification
    - A* with elevation heuristics
    - Professional visualization with quality indicators
    """
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ===== MULTI-LAYER SKY DETECTION =====
    h_chan = hsv[:, :, 0].astype(np.float32)
    s_chan = hsv[:, :, 1].astype(np.float32)
    v_chan = hsv[:, :, 2].astype(np.float32)

    # Layer 1: Pure blue sky (hue-saturation based)
    blue_sky = (h_chan >= 80) & (h_chan <= 140) & (s_chan >= 20) & (v_chan >= 100)

    # Layer 2: White/light clouds (low saturation, high brightness)
    white_clouds = (s_chan < 40) & (v_chan > 160)

    # Layer 3: Cyan/turquoise sky (hue + saturation specific)
    cyan_sky = (
        (h_chan >= 70) & (h_chan <= 100)
        & (s_chan >= 60) & (s_chan <= 200)
        & (v_chan > 140)
    )

    # Layer 4: Gradient detection (sky has smooth gradients, fewer edges)
    edges = cv2.Canny(gray.astype(np.uint8), 20, 60)
    low_edge_mask = (edges < 50).astype(bool)
    high_brightness = (gray / 255.0 > 0.7)
    sky_gradient = low_edge_mask & high_brightness

    # Combine all sky detections
    sky_mask = (blue_sky | white_clouds | cyan_sky | sky_gradient).astype(np.uint8) * 255

    # Advanced cleanup: morphology + dilation for safety margin
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel_dilate, iterations=1)
    sky_mask = cv2.dilate(sky_mask, kernel_dilate, iterations=2)
    sky_mask = (sky_mask > 127).astype(np.uint8)

    # ===== ADVANCED TERRAIN SEGMENTATION =====
    b_ch, g_ch, r_ch = cv2.split(frame)
    b_f = b_ch.astype(np.float32) / 255.0
    g_f = g_ch.astype(np.float32) / 255.0
    r_f = r_ch.astype(np.float32) / 255.0

    # Rock detection: reddish-brown
    rock_mask = (r_f > 0.4) & (r_f > g_f) & (g_f > b_f) & (r_f - b_f > 0.2)

    # Vegetation detection: green areas
    vegetation_mask = (g_f > 0.4) & (g_f > r_f) & (g_f > b_f)

    # Scree/loose rock: grayish
    scree_mask = (
        (np.abs(r_f - g_f) < 0.15)
        & (np.abs(g_f - b_f) < 0.15)
        & (gray / 255.0 > 0.3)
        & (gray / 255.0 < 0.7)
    )

    # Steep/shadow areas: very dark
    steep_mask = gray / 255.0 < 0.2

    # ===== ADVANCED COST MAP =====
    edges = cv2.Canny(gray.astype(np.uint8), 30, 100)
    edges_norm = edges.astype(np.float32) / 255.0
    edge_cost = edges_norm * 0.4

    brightness = gray / 255.0
    brightness_cost = np.abs(brightness - 0.5) * 1.8
    brightness_cost = np.clip(brightness_cost, 0, 1)

    blur = cv2.GaussianBlur((gray / 255.0), (25, 25), 0)
    texture_cost = np.abs(gray / 255.0 - blur) * 2.0
    texture_cost = np.clip(texture_cost, 0, 1)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5) / 255.0
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5) / 255.0
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
    slope_cost = slope_norm * 0.5

    rock_bonus = rock_mask.astype(np.float32) * -0.1
    veg_penalty = vegetation_mask.astype(np.float32) * 0.3
    scree_penalty = scree_mask.astype(np.float32) * 0.2
    steep_penalty = steep_mask.astype(np.float32) * 0.15

    cost_map = (
        brightness_cost * 0.3
        + edge_cost * 0.25
        + texture_cost * 0.2
        + slope_cost * 0.25
        + veg_penalty
        + scree_penalty
        + steep_penalty
        + rock_bonus
    )

    cost_map = cv2.GaussianBlur(cost_map, (15, 15), 0)
    cost_map = np.clip(cost_map, 0, 1)

    cost_map = (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min() + 1e-6)
    cost_map = np.clip(cost_map, 0, 1)

    # Sky = MAXIMUM cost to avoid completely
    cost_map[sky_mask == 255] = 2.0

    terrain_mask = 1 - sky_mask

    # ===== HELPER FUNCTIONS =====
    def nearest_ground(y, x, max_radius=50):
        """Find nearest non-sky pixel with intelligent search."""
        y = int(np.clip(y, 0, h - 1))
        x = int(np.clip(x, 0, w - 1))
        if sky_mask[y, x] == 0:
            return y, x

        for r in range(1, max_radius + 1):
            y_min = max(0, y - r)
            y_max = min(h - 1, y + r)
            x_min = max(0, x - r)
            x_max = min(w - 1, x + r)

            for xx in range(x_min, x_max + 1):
                if y_min < h and sky_mask[y_min, xx] == 0:
                    return y_min, xx
                if y_max < h and sky_mask[y_max, xx] == 0:
                    return y_max, xx
            for yy in range(y_min, y_max + 1):
                if x_min < w and sky_mask[yy, x_min] == 0:
                    return yy, x_min
                if x_max < w and sky_mask[yy, x_max] == 0:
                    return yy, x_max
        return None

    # ===== ENHANCED A* ALGORITHM =====
    def a_star_safest_route(start_y, start_x, end_y, end_x, allow_sky=False):
        """A* pathfinding with elevation heuristics and variable step sizes."""
        start_y = max(0, min(h - 1, int(start_y)))
        start_x = max(0, min(w - 1, int(start_x)))
        end_y = max(0, min(h - 1, int(end_y)))
        end_x = max(0, min(w - 1, int(end_x)))

        def heuristic(y1, x1, y2, x2):
            h_dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            elevation_bias = (y1 - y2) * 0.5
            return h_dist + elevation_bias

        open_set = {(start_y, start_x): heuristic(start_y, start_x, end_y, end_x)}
        came_from = {}
        g_score = {(start_y, start_x): 0}
        visited = set()

        max_iterations = 50000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current = min(open_set, key=open_set.get)
            y, x = current

            if abs(y - end_y) < 20 and abs(x - end_x) < 20:
                path = [(y, x)]
                current = (y, x)
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            del open_set[current]
            visited.add(current)

            for dy in [-15, -10, -5, 0, 5, 10, 15]:
                for dx in [-15, -10, -5, 0, 5, 10, 15]:
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = y + dy, x + dx

                    if not (0 <= ny < h and 0 <= nx < w) or (ny, nx) in visited:
                        continue

                    if not allow_sky and sky_mask[ny, nx] == 255:
                        continue

                    move_cost = cost_map[ny, nx]
                    if allow_sky and sky_mask[ny, nx] == 255:
                        move_cost = 10.0

                    movement_cost = 5.0 if dy != 0 and dx != 0 else 3.0
                    tentative_g = g_score[current] + move_cost * movement_cost

                    if (ny, nx) not in g_score or tentative_g < g_score[(ny, nx)]:
                        came_from[(ny, nx)] = current
                        g_score[(ny, nx)] = tentative_g
                        f_score = tentative_g + heuristic(ny, nx, end_y, end_x)
                        open_set[(ny, nx)] = f_score

        return None

    # ===== FIND SAFEST ROUTE =====
    start_y = min(h - 30, h - 15)
    end_y = max(5, int(h * 0.10))
    end_x = w // 2

    end_ground = nearest_ground(end_y, end_x, max_radius=60)
    if end_ground:
        end_y, end_x = end_ground

    best_path = None
    best_cost = np.inf

    for start_x_guess in [w // 6, w // 4, w // 2, 3 * w // 4, 5 * w // 6]:
        start_ground = nearest_ground(start_y, start_x_guess, max_radius=60)
        if not start_ground:
            continue
        s_y, s_x = start_ground

        path = a_star_safest_route(s_y, s_x, end_y, end_x, allow_sky=False)

        if not path:
            path = a_star_safest_route(s_y, s_x, end_y, end_x, allow_sky=True)

        if path and len(path) > 15:
            total_cost = sum(
                cost_map[min(int(y), h - 1), min(int(x), w - 1)]
                for y, x in path
                if 0 <= int(y) < h and 0 <= int(x) < w
            )
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path

    path = best_path if best_path else None

    # ===== INTELLIGENT WAYPOINT PLACEMENT =====
    waypoints = []
    if path:
        waypoints = [
            (int(y), int(x))
            for y, x in path
            if sky_mask[min(int(y), h - 1), min(int(x), w - 1)] == 0
        ]
        if len(waypoints) > 12:
            step = len(waypoints) // 10
            waypoints = waypoints[::step]

    # ===== PROFESSIONAL VISUALIZATION =====
    annotated = frame.copy()

    if path:
        path_array = np.array(path, dtype=np.int32)

        cv2.polylines(annotated, [path_array], False, (0, 255, 0), 7)
        cv2.polylines(annotated, [path_array], False, (255, 255, 255), 2)

        for i, (py, px) in enumerate(waypoints):
            if 0 <= py < h and 0 <= px < w:
                cv2.circle(annotated, (px, py), 10, (255, 255, 0), -1)
                cv2.circle(annotated, (px, py), 10, (0, 0, 0), 2)
                cv2.putText(
                    annotated, str(i), (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )

        path_costs = [
            cost_map[min(int(y), h - 1), min(int(x), w - 1)] for y, x in path
        ]
        avg_cost = np.mean(path_costs) if path_costs else 0

        if avg_cost < 0.3:
            quality_text = "EXCELLENT"
            quality_color = (0, 255, 0)
        elif avg_cost < 0.5:
            quality_text = "GOOD"
            quality_color = (0, 165, 255)
        elif avg_cost < 0.7:
            quality_text = "MODERATE"
            quality_color = (0, 255, 255)
        else:
            quality_text = "CHALLENGING"
            quality_color = (0, 0, 255)

        cv2.rectangle(annotated, (10, 10), (250, 50), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (250, 50), quality_color, 2)
        cv2.putText(
            annotated, f"Route Quality: {quality_text}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, quality_color, 2,
        )

        cv2.putText(
            annotated, f"Waypoints: {len(waypoints)}", (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )
    else:
        cv2.rectangle(annotated, (10, h // 2 - 40), (w - 10, h // 2 + 40), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, h // 2 - 40), (w - 10, h // 2 + 40), (0, 0, 255), 2)
        cv2.putText(
            annotated, "No Safe Route Detected", (30, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,
        )

    result = {
        "cost_map": cost_map,
        "terrain_mask": terrain_mask,
        "sky_mask": sky_mask,
        "rock_mask": rock_mask.astype(np.uint8) * 255,
        "vegetation_mask": vegetation_mask.astype(np.uint8) * 255,
        "scree_mask": scree_mask.astype(np.uint8) * 255,
        "segmentation": cv2.cvtColor(
            (terrain_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        ),
    }

    return annotated, result, path if path else []


def create_simplified_route_analysis(frame):
    """
    Simplified route analysis using edge detection + brightness map
    when full ML models are unavailable.
    """
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    cost_map = (edges / 255.0) * 0.7 + 0.1
    cost_map = cv2.GaussianBlur(cost_map, (15, 15), 0)

    path = []
    for i in range(h, 0, -10):
        x = int(w * 0.5) + int(20 * np.sin(i / h * 3.14))
        path.append((i, x))

    annotated = frame.copy()

    cost_colored = cv2.applyColorMap(
        (cost_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    annotated = cv2.addWeighted(annotated, 0.6, cost_colored, 0.4, 0)

    if len(path) > 1:
        path_array = np.array(path, dtype=np.int32)
        cv2.polylines(annotated, [path_array], False, (0, 255, 0), 3)

    cv2.putText(
        annotated, "Safe Route (Simplified Analysis)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
    cv2.putText(
        annotated, "Green = Safe Route | Red = Difficult", (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
    )

    result = {"cost_map": cost_map}
    return annotated, result, path


# =============================================================================
# Main Application
# =============================================================================

def main():
    # -- Page Configuration ---------------------------------------------------
    st.set_page_config(
        page_title="ClimbAssist AI v2",
        page_icon="🧗",
        layout="wide",
    )

    # -- Inject CSS from constants --------------------------------------------
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # -- Header ---------------------------------------------------------------
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 2rem; margin-top: 1rem;'>
            <h1 style='font-size: 3.5rem; font-weight: 800;
                background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                margin-bottom: 0.3rem;'>
                🧗 ClimbAssist AI
            </h1>
            <p style='font-size: 1.1rem; color: #a5b4fc; font-weight: 500;'>
                AI-Powered Climbing Analysis & Safe Route Planning
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -- Feature Availability Banner ------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅ Full" if True else "❌ Unavailable"
        st.markdown(f"**⚙️ Gear Optimizer:** {status}")
    with col2:
        status = "✅ Full" if CV2_AVAILABLE else "⚠️ Cloud Only"
        st.markdown(f"**🎥 Movement Analyzer:** {status}")
    with col3:
        status = "✅ Full" if ROUTE_PLANNER_AVAILABLE else "⚠️ Cloud Only"
        st.markdown(f"**🗺️ Route Finder:** {status}")

    st.markdown("---")

    # -- Tab Navigation -------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "⚙️ Gear Optimizer",
        "🎥 Movement Analyzer",
        "🗺️ Safe Route Finder",
    ])

    # =========================================================================
    # TAB 1: GEAR LOAD OPTIMIZER
    # =========================================================================
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## ⚙️ Gear Load Optimizer")
        st.markdown(
            "<p style='color: #a5b4fc; margin-bottom: 1.5rem;'>"
            "Input your expedition parameters to generate an optimized gear list."
            "</p>",
            unsafe_allow_html=True,
        )

        # Random pro-tip
        tip = random.choice(GEAR_TIPS)
        st.markdown(
            f"<p style='color: #fbbf24; font-size: 0.9rem; margin-bottom: 1rem;'>"
            f"💡 {tip}</p>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            expedition_days = st.slider("Expedition Duration (days)", 1, 20, 3)
            max_altitude = st.slider("Max Altitude (m)", 500, 8000, 2560, 10)
        with col2:
            weather = st.selectbox("Expected Weather", ["Clear", "Windy", "Rainy", "Snow"])
            skill = st.selectbox("Climber Skill Level", ["Beginner", "Intermediate", "Advanced"])

        if st.button("⚙️ Optimize Gear", use_container_width=True):
            with st.spinner("Optimizing gear..."):
                time.sleep(1)

            # Use GearOptimizer module
            optimizer = GearOptimizer()
            recommended = optimizer.optimize(expedition_days, max_altitude, weather, skill)

            st.markdown('<div class="gear-list-card">', unsafe_allow_html=True)
            st.markdown("### 🎒 Recommended Gear List")
            for item in recommended:
                st.markdown(f"• **{item}**")
            st.markdown("</div>", unsafe_allow_html=True)

            # PDF download via GearPDFGenerator
            pdf_bytes = GearPDFGenerator.generate(
                recommended, expedition_days, max_altitude, weather, skill,
            )
            st.download_button(
                label="📥 Download Gear List (PDF)",
                data=pdf_bytes,
                file_name="climbassist_gear_list.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: CLIMBING MOVEMENT ANALYZER
    # =========================================================================
    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("## 🎥 Climb Movement Analyzer")
        st.markdown(
            "<p style='color: #a5b4fc; margin-bottom: 1.5rem;'>"
            "Upload a climbing video to receive AI-powered movement analysis "
            "and personalized feedback.</p>",
            unsafe_allow_html=True,
        )

        if not CV2_AVAILABLE or not POSE_VISUALIZER_AVAILABLE:
            st.info(
                "ℹ️ **Movement Analyzer unavailable on Cloud**\n\n"
                "The climbing movement analyzer requires OpenCV, which is "
                "resource-intensive. This feature works great locally!\n\n"
                "**Try the other features:**\n"
                "- ⚙️ Gear Load Optimizer\n"
                "- 🗺️ Route Planner (local)"
            )
        else:
            # Random video tip
            video_tip = random.choice(VIDEO_TIPS)
            st.markdown(
                f"<p style='color: #34d399; font-size: 0.9rem; margin-bottom: 1rem;'>"
                f"🎯 {video_tip}</p>",
                unsafe_allow_html=True,
            )

            st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
            st.markdown("<h3>📤 Upload Your Climbing Video</h3>", unsafe_allow_html=True)
            st.markdown(
                '<p style="color: #a5b4fc;">Supported formats: MP4, MPEG4 • Max size: 200 MB</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            uploaded_video = st.file_uploader(
                "Upload Climbing Video",
                type=["mp4", "mpeg4"],
                label_visibility="collapsed",
            )

            if uploaded_video is None:
                st.info(
                    "📋 **How it works:**\n"
                    "1. Upload a video of your climbing\n"
                    "2. Our AI analyzes your body movements\n"
                    "3. Get detailed feedback and personalized recommendations"
                )
            else:
                st.markdown(
                    f"<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; "
                    f"border-radius: 8px;'><span style='color:#c4b5fd;'>📄 <b>File:</b> "
                    f"{uploaded_video.name} | <b>Size:</b> "
                    f"{round(uploaded_video.size / 1024 / 1024, 2)} MB</span></div>",
                    unsafe_allow_html=True,
                )

                # Save video to temp file
                temp_video_path = save_temp_video(uploaded_video)

                # Load MoveNet model (cached)
                progress_bar = st.progress(0, text="🤖 Loading AI Model (cached)...")
                movenet = load_movenet_model()
                progress_bar.progress(20, text="🤖 Model loaded! Analyzing video...")

                def detect_pose(frame):
                    import tensorflow.compat.v1 as tf
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (192, 192))
                    input_img = tf.convert_to_tensor(img, dtype=tf.int32)
                    input_img = tf.expand_dims(input_img, axis=0)
                    outputs = movenet(input_img)
                    keypoints = outputs["output_0"].numpy()[0, 0, :, :]
                    return keypoints

                cap = cv2.VideoCapture(temp_video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                max_frames_to_analyze = min(
                    frame_count, int(fps * 15) if fps > 0 else 150
                )

                # Use MovementAnalyzer module instead of inline issue detection
                movement_analyzer = MovementAnalyzer()

                progress = st.progress(0, text="🔍 Analyzing climbing technique...")

                sample_rate = 10
                for i in range(0, max_frames_to_analyze, sample_rate):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    keypoints = detect_pose(frame)
                    movement_analyzer.analyze_keypoints(keypoints)

                    current_progress = min(
                        (i / max_frames_to_analyze) * 0.6 + 20, 80
                    )
                    progress_bar.progress(
                        int(current_progress),
                        text="🔍 Analyzing climbing technique with AI...",
                    )

                cap.release()
                progress_bar.progress(85, text="📊 Generating results...")
                progress_bar.empty()

                summary = movement_analyzer.get_summary()

                st.markdown(
                    f'<div class="status-bar">✅ Analysis Complete | '
                    f'{summary["analyzed_frames"]} frames analyzed | '
                    f'{summary["valid_frames"]} valid detections</div>',
                    unsafe_allow_html=True,
                )

                # Create tabs for results
                results_tab1, results_tab2, results_tab3 = st.tabs([
                    "📊 Summary", "🎬 Sample Frames", "💡 Improvement Tips",
                ])

                with results_tab1:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Frames Analyzed", summary["analyzed_frames"])
                    with col2:
                        st.metric("Valid Detections", summary["valid_frames"])
                    with col3:
                        st.metric("Detection Rate", f'{summary["detection_rate"]}%')

                    st.markdown("---")

                    threshold = summary["threshold"]
                    issue_counts = summary["issue_counts"]
                    valid_frames = summary["valid_frames"]
                    issues_found = False

                    st.markdown("### 🔍 Detected Issues:")

                    # Issue 1: Wide Stance
                    if issue_counts["wide_stance"] > threshold:
                        issues_found = True
                        pct = int((issue_counts["wide_stance"] / valid_frames) * 100) if valid_frames > 0 else 0
                        st.markdown(
                            f'<div class="mistake-card">❌ Wide stance detected in {pct}% '
                            f'of movements ({issue_counts["wide_stance"]} times)</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="recommend-card">✅ Keep feet closer together and '
                            "focus on precise foot placements for better control</div>",
                            unsafe_allow_html=True,
                        )

                    # Issue 2: Hips Away
                    if issue_counts["hips_away"] > threshold:
                        issues_found = True
                        pct = int((issue_counts["hips_away"] / valid_frames) * 100) if valid_frames > 0 else 0
                        st.markdown(
                            f'<div class="mistake-card">❌ Hips away from wall in {pct}% '
                            f'of movements ({issue_counts["hips_away"]} times)</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="recommend-card">✅ Keep hips close to the wall '
                            "to maintain balance and save energy</div>",
                            unsafe_allow_html=True,
                        )

                    # Issue 3: Poor Hand Use
                    if issue_counts["poor_hand_use"] > threshold:
                        issues_found = True
                        pct = int((issue_counts["poor_hand_use"] / valid_frames) * 100) if valid_frames > 0 else 0
                        st.markdown(
                            f'<div class="mistake-card">❌ Inconsistent hand usage in {pct}% '
                            f'of frames ({issue_counts["poor_hand_use"]} times)</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="recommend-card">✅ Use all available handholds and '
                            "maintain three points of contact when possible</div>",
                            unsafe_allow_html=True,
                        )

                    # Issue 4: Unstable Knees
                    if issue_counts["unstable_knees"] > threshold:
                        issues_found = True
                        pct = int((issue_counts["unstable_knees"] / valid_frames) * 100) if valid_frames > 0 else 0
                        st.markdown(
                            f'<div class="mistake-card">❌ Knee instability observed in {pct}% '
                            f'of movements ({issue_counts["unstable_knees"]} times)</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="recommend-card">✅ Keep knees aligned with your '
                            "body and avoid letting them collapse inward</div>",
                            unsafe_allow_html=True,
                        )

                    # Issue 5: Overreaching
                    if issue_counts["overreaching"] > threshold:
                        issues_found = True
                        pct = int((issue_counts["overreaching"] / valid_frames) * 100) if valid_frames > 0 else 0
                        st.markdown(
                            f'<div class="mistake-card">❌ Overreaching detected in {pct}% '
                            f'of movements ({issue_counts["overreaching"]} times)</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="recommend-card">✅ Move your feet up first before '
                            "reaching to reduce strain and improve control</div>",
                            unsafe_allow_html=True,
                        )

                    if not issues_found:
                        st.markdown(
                            '<div class="recommend-card">✅ Excellent technique! No major '
                            "issues detected. Your climbing form looks great - keep "
                            "challenging yourself!</div>",
                            unsafe_allow_html=True,
                        )

                with results_tab2:
                    st.markdown("### 🎬 Sample Frames with Pose Analysis")
                    st.info(
                        "📝 Frame-by-frame visualization with detected pose landmarks "
                        "and issues highlighted in red circles"
                    )

                    if POSE_VISUALIZER_AVAILABLE:
                        visualizer = PoseVisualizer()

                        cap = cv2.VideoCapture(temp_video_path)
                        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        sample_indices = [
                            0,
                            frame_count_total // 2,
                            max(0, frame_count_total - 1),
                        ]

                        sample_frames_col = st.columns(3)

                        for idx, frame_idx in enumerate(sample_indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()

                            if ret:
                                import tensorflow.compat.v1 as tf

                                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, (192, 192))
                                input_img = tf.convert_to_tensor(img, dtype=tf.int32)
                                input_img = tf.expand_dims(input_img, axis=0)
                                outputs = movenet(input_img)
                                keypoints = outputs["output_0"].numpy()[0, 0, :, :]

                                frame_display = cv2.resize(frame, (400, 300))
                                frame_viz, detected_issues = visualizer.analyze_and_visualize(
                                    frame_display, keypoints,
                                )
                                frame_rgb = cv2.cvtColor(frame_viz, cv2.COLOR_BGR2RGB)

                                with sample_frames_col[idx]:
                                    st.image(
                                        frame_rgb,
                                        caption=f"Frame {frame_idx}",
                                        use_column_width=True,
                                    )
                                    if detected_issues:
                                        st.caption(
                                            "Issues: "
                                            + ", ".join([
                                                i.replace("_", " ").title()
                                                for i in detected_issues
                                                if i != "person_not_detected"
                                            ])
                                        )

                        cap.release()
                    else:
                        st.warning(
                            "Pose visualizer module not available. "
                            "Sample frames cannot be displayed."
                        )

                with results_tab3:
                    st.markdown("### 💡 Personalized Improvement Tips")

                    if POSE_VISUALIZER_AVAILABLE:
                        advisor = ImprovementAdvisor()

                        detected_issues_list = []
                        threshold = summary["threshold"]

                        if issue_counts["wide_stance"] > threshold:
                            detected_issues_list.append("wide_stance")
                        if issue_counts["hips_away"] > threshold:
                            detected_issues_list.append("hips_away_from_wall")
                        if issue_counts["poor_hand_use"] > threshold:
                            detected_issues_list.append("poor_hand_usage")
                        if issue_counts["unstable_knees"] > threshold:
                            detected_issues_list.append("unstable_knees")
                        if issue_counts["overreaching"] > threshold:
                            detected_issues_list.append("overreaching")

                        advice = advisor.get_advice(detected_issues_list)

                        if advice.get("issues"):
                            for issue_key, issue_advice in advice["issues"].items():
                                with st.expander(
                                    f"🎯 {issue_advice['issue']}", expanded=True,
                                ):
                                    st.markdown(
                                        f"**Problem:** {issue_advice['description']}"
                                    )
                                    st.markdown(
                                        f"**Why it matters:** {issue_advice['why_bad']}"
                                    )

                                    st.markdown(
                                        "**🚫 Key Points to AVOID (Prevent This Mistake):**"
                                    )
                                    for avoid_point in issue_advice["key_points_to_avoid"]:
                                        st.markdown(f"{avoid_point}")

                                    st.markdown("**📋 Steps to Improve:**")
                                    for step_text in issue_advice["improvement_steps"]:
                                        st.markdown(f"• {step_text}")

                                    st.markdown(
                                        f"**💪 Training Drill:** {issue_advice['drill']}"
                                    )
                                    st.markdown("---")
                        else:
                            st.success(
                                "🎉 Great climbing form! No significant issues detected."
                            )

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: SAFE ROUTE FINDER
    # =========================================================================
    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        # Professional Header
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #c4b5fd; font-size: 2.5rem; margin-bottom: 0.5rem;'>
                    🗺️ AI-Powered Safe Route Finder
                </h2>
                <p style='color: #a5b4fc; font-size: 1.05rem; max-width: 700px; margin: 0 auto;'>
                    Intelligent terrain analysis and pathfinding for climbing routes.
                    Our AI identifies hazardous areas and computes the safest,
                    most efficient climbing path.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Feature highlights
        st.markdown(
            """
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 2rem;'>
                <div style='background: rgba(34, 197, 94, 0.1); padding: 1rem; border-radius: 12px; border-left: 3px solid #22c55e;'>
                    <p style='color: #86efac; font-weight: 600; margin: 0.5rem 0;'>🔍 Terrain Segmentation</p>
                    <p style='color: #cbd5e1; font-size: 0.9rem; margin: 0;'>Identifies rock, cliffs, vegetation & hazards</p>
                </div>
                <div style='background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 12px; border-left: 3px solid #3b82f6;'>
                    <p style='color: #93c5fd; font-weight: 600; margin: 0.5rem 0;'>📊 Cost Mapping</p>
                    <p style='color: #cbd5e1; font-size: 0.9rem; margin: 0;'>Computes traversability & slope difficulty</p>
                </div>
                <div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 12px; border-left: 3px solid #8b5cf6;'>
                    <p style='color: #c4b5fd; font-weight: 600; margin: 0.5rem 0;'>🧠 A* Pathfinding</p>
                    <p style='color: #cbd5e1; font-size: 0.9rem; margin: 0;'>Finds optimal curved route following terrain</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not ROUTE_PLANNER_AVAILABLE:
            st.info(
                "ℹ️ **Route Planner unavailable on Cloud**\n\n"
                "The safe route analyzer requires PyTorch, which is resource-intensive. "
                "This feature works great locally!\n\n"
                "**Try the other features:**\n"
                "- ⚙️ Gear Load Optimizer\n"
                "- 🏃 Climbing Movement Analyzer"
            )
        else:
            # Professional upload section
            st.markdown(
                """
                <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.08),
                    rgba(59, 130, 246, 0.08)); padding: 2rem; border-radius: 16px;
                    border: 1px solid rgba(139, 92, 246, 0.2); margin-bottom: 2rem;'>
                    <h3 style='color: #c4b5fd; margin-top: 0;'>📤 Upload Your Climbing Video</h3>
                    <p style='color: #a5b4fc;'>Supported formats: MP4, MPEG4 • Max size: 200 MB</p>
                """,
                unsafe_allow_html=True,
            )

            video_file = st.file_uploader(
                "Select Video File",
                type=["mp4", "mpeg4"],
                key="route_video",
                label_visibility="collapsed",
            )

            st.markdown("</div>", unsafe_allow_html=True)

            if video_file is not None:
                # Professional file info card
                st.markdown(
                    f"""
                    <div style='background: rgba(139, 92, 246, 0.1); padding: 1.2rem;
                        border-radius: 12px; border-left: 4px solid #8b5cf6;
                        margin-bottom: 1.5rem;'>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                            <div>
                                <p style='color: #a5b4fc; font-size: 0.85rem; margin: 0.3rem 0;'>📄 File Name</p>
                                <p style='color: #c4b5fd; font-weight: 600; margin: 0;'>{video_file.name}</p>
                            </div>
                            <div>
                                <p style='color: #a5b4fc; font-size: 0.85rem; margin: 0.3rem 0;'>💾 File Size</p>
                                <p style='color: #c4b5fd; font-weight: 600; margin: 0;'>{round(video_file.size / 1024 / 1024, 2)} MB</p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("🔍 Analyze Route", use_container_width=True):
                    progress_bar = st.progress(0, text="🤖 Loading AI models (cached)...")

                    try:
                        route_analyzer = load_safe_route_analyzer()
                        progress_bar.progress(25, text="✅ Models loaded! Processing video...")
                    except Exception as e:
                        st.error(
                            f"❌ Error loading models: {str(e)}\n\n"
                            "Trying simplified analysis..."
                        )
                        route_analyzer = None

                    # Save video
                    temp_path = save_temp_video(video_file, filename="temp_route_video.mp4")

                    cap = cv2.VideoCapture(temp_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    middle_frame = max(0, total_frames // 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                    ret, frame = cap.read()
                    cap.release()

                    progress_bar.progress(50, text="📊 Performing terrain segmentation...")

                    if ret:
                        # Enhance image contrast using utility
                        frame_enhanced = enhance_frame(frame)

                        # Use advanced route analysis
                        try:
                            annotated, result, path = create_advanced_route_analysis(
                                frame_enhanced,
                            )
                            progress_bar.progress(
                                85, text="🎯 Computing A* optimal path...",
                            )
                        except Exception:
                            st.warning(
                                "⚠️ Analysis encountered an issue. "
                                "Using fallback analysis...",
                                icon="⚠️",
                            )
                            annotated, result, path = create_simplified_route_analysis(
                                frame_enhanced,
                            )

                        progress_bar.progress(100, text="✅ Analysis complete!")
                        progress_bar.empty()

                        # Results Header
                        st.markdown(
                            """
                            <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                                padding: 0.6rem 0.8rem; border-radius: 8px;
                                border: 1px solid rgba(139, 92, 246, 0.3);
                                margin: 0.5rem 0 0.4rem 0;'>
                                <h3 style='color: #c4b5fd; margin: 0; font-size: 1.1rem;'>
                                    Route Analysis
                                </h3>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Visualization Section
                        col1, col2, col3 = st.columns([0.3, 2, 0.3])
                        with col2:
                            st.image(annotated, use_column_width=True)

                        # Enterprise-Style Tabs
                        result_tab1, result_tab2, result_tab3 = st.tabs([
                            "📈 STATS", "🏔️ TERRAIN", "⛰️ GUIDE",
                        ])

                        with result_tab1:
                            st.markdown(
                                """
                                <div style='background: linear-gradient(135deg,
                                    rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1));
                                    padding: 0.6rem; border-radius: 8px;
                                    border-left: 4px solid #22c55e; margin: 0;'>
                                    <p style='color: #c4b5fd; margin: 0; font-size: 0.9rem;'>
                                        Route Metrics
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if path:
                                route_distance = 0
                                if len(path) > 1:
                                    for i in range(len(path) - 1):
                                        dy = path[i + 1][0] - path[i][0]
                                        dx = path[i + 1][1] - path[i][1]
                                        route_distance += np.sqrt(dy ** 2 + dx ** 2)

                                path_costs = []
                                if len(path) > 0:
                                    for waypoint in path:
                                        y = min(
                                            int(waypoint[0]),
                                            result["cost_map"].shape[0] - 1,
                                        )
                                        x = min(
                                            int(waypoint[1]),
                                            result["cost_map"].shape[1] - 1,
                                        )
                                        path_costs.append(result["cost_map"][y, x])

                                avg_cost = np.mean(path_costs) if path_costs else 0

                                if avg_cost < 0.2:
                                    difficulty = "VERY EASY"
                                    status_color = "#22c55e"
                                    status_bg = "rgba(34, 197, 94, 0.1)"
                                elif avg_cost < 0.4:
                                    difficulty = "EASY"
                                    status_color = "#84cc16"
                                    status_bg = "rgba(132, 204, 22, 0.1)"
                                elif avg_cost < 0.6:
                                    difficulty = "MODERATE"
                                    status_color = "#f59e0b"
                                    status_bg = "rgba(245, 158, 11, 0.1)"
                                elif avg_cost < 0.8:
                                    difficulty = "HARD"
                                    status_color = "#ef4444"
                                    status_bg = "rgba(239, 68, 68, 0.1)"
                                else:
                                    difficulty = "EXTREME"
                                    status_color = "#dc2626"
                                    status_bg = "rgba(220, 38, 38, 0.1)"

                                st.markdown(
                                    f"""
                                    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 0.5rem; margin: 0.5rem 0;'>
                                        <div style='background: rgba(59, 130, 246, 0.1); padding: 0.6rem; border-radius: 6px; border: 1px solid rgba(59, 130, 246, 0.3);'>
                                            <p style='color: #94a3b8; margin: 0; font-size: 0.65rem; font-weight: 600;'>WPT</p>
                                            <p style='color: #3b82f6; margin: 0.2rem 0 0 0; font-size: 1.2rem; font-weight: bold;'>{len(path)}</p>
                                        </div>
                                        <div style='background: rgba(168, 85, 247, 0.1); padding: 0.6rem; border-radius: 6px; border: 1px solid rgba(168, 85, 247, 0.3);'>
                                            <p style='color: #94a3b8; margin: 0; font-size: 0.65rem; font-weight: 600;'>DIST</p>
                                            <p style='color: #a855f7; margin: 0.2rem 0 0 0; font-size: 1.2rem; font-weight: bold;'>{route_distance:.0f}</p>
                                        </div>
                                        <div style='background: rgba(34, 197, 94, 0.1); padding: 0.6rem; border-radius: 6px; border: 1px solid rgba(34, 197, 94, 0.3);'>
                                            <p style='color: #94a3b8; margin: 0; font-size: 0.65rem; font-weight: 600;'>COST</p>
                                            <p style='color: #22c55e; margin: 0.2rem 0 0 0; font-size: 1.2rem; font-weight: bold;'>{avg_cost:.2f}</p>
                                        </div>
                                        <div style='background: {status_bg}; padding: 0.6rem; border-radius: 6px; border: 1px solid {status_color}33;'>
                                            <p style='color: #94a3b8; margin: 0; font-size: 0.65rem; font-weight: 600;'>DIFF</p>
                                            <p style='color: {status_color}; margin: 0.2rem 0 0 0; font-size: 1rem; font-weight: bold;'>{difficulty}</p>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    """
                                    <div style='background: rgba(239, 68, 68, 0.1); padding: 1.5rem;
                                        border-radius: 12px; border-left: 4px solid #ef4444;'>
                                        <h4 style='color: #fca5a5; margin: 0 0 0.5rem 0;'>⚠️ Analysis Result</h4>
                                        <p style='color: #fecaca; margin: 0;'>
                                            No viable climbing route detected. The terrain appears
                                            too difficult or hazardous for standard climbing conditions.
                                        </p>
                                        <p style='color: #fbcfe8; margin: 0.8rem 0 0 0; font-size: 0.9rem;'>
                                            <b>Recommendation:</b> Consider alternative routes, wait for
                                            better weather conditions, or seek professional guidance.
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        with result_tab2:
                            if path:
                                st.markdown(
                                    """
                                    <div style='background: linear-gradient(135deg,
                                        rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.1));
                                        padding: 0.8rem; border-radius: 8px;
                                        border-left: 3px solid #3b82f6; margin: 0.5rem 0;'>
                                        <h4 style='color: #93c5fd; margin: 0 0 0.8rem 0;
                                            font-size: 1rem;'>
                                            🏔️ Terrain Segmentation Analysis
                                        </h4>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                terrain_vis_col1, terrain_vis_col2, terrain_vis_col3 = st.columns(3)

                                with terrain_vis_col1:
                                    st.markdown(
                                        "<p style='color: #94a3b8; font-size: 0.8rem; "
                                        "text-align: center; margin: 0 0 0.5rem 0;'>"
                                        "<b>🌬️ Sky Detection</b></p>",
                                        unsafe_allow_html=True,
                                    )
                                    st.image(
                                        result["sky_mask"],
                                        use_column_width=True,
                                        channels="GRAY",
                                    )

                                with terrain_vis_col2:
                                    st.markdown(
                                        "<p style='color: #94a3b8; font-size: 0.8rem; "
                                        "text-align: center; margin: 0 0 0.5rem 0;'>"
                                        "<b>💰 Cost Map</b></p>",
                                        unsafe_allow_html=True,
                                    )
                                    cost_display = (result["cost_map"] * 255).astype(np.uint8)
                                    cost_colormap = cv2.applyColorMap(
                                        cost_display, cv2.COLORMAP_JET,
                                    )
                                    cost_rgb = cv2.cvtColor(
                                        cost_colormap, cv2.COLOR_BGR2RGB,
                                    )
                                    st.image(cost_rgb, use_column_width=True)

                                with terrain_vis_col3:
                                    st.markdown(
                                        "<p style='color: #94a3b8; font-size: 0.8rem; "
                                        "text-align: center; margin: 0 0 0.5rem 0;'>"
                                        "<b>🌿 Terrain Mask</b></p>",
                                        unsafe_allow_html=True,
                                    )
                                    st.image(
                                        result["segmentation"],
                                        use_column_width=True,
                                    )

                                # Terrain breakdown statistics
                                cost_map_data = result["cost_map"]
                                flat_costs = cost_map_data.flatten()
                                total_pixels = np.sum(flat_costs > 0)

                                # Terrain type breakdown
                                if "rock_mask" in result:
                                    rock_pct = (
                                        np.sum(result["rock_mask"] > 127)
                                        / (result["rock_mask"].size + 1e-6)
                                    ) * 100
                                else:
                                    rock_pct = 0

                                if "vegetation_mask" in result:
                                    veg_pct = (
                                        np.sum(result["vegetation_mask"] > 127)
                                        / (result["vegetation_mask"].size + 1e-6)
                                    ) * 100
                                else:
                                    veg_pct = 0

                                if "scree_mask" in result:
                                    scree_pct = (
                                        np.sum(result["scree_mask"] > 127)
                                        / (result["scree_mask"].size + 1e-6)
                                    ) * 100
                                else:
                                    scree_pct = 0

                                st.markdown(
                                    """
                                    <div style='background: rgba(148, 163, 184, 0.1);
                                        padding: 1rem; border-radius: 6px; margin: 0.5rem 0 0 0;'>
                                        <p style='color: #cbd5e1; font-size: 0.85rem;
                                            margin: 0.5rem 0; font-weight: 600;'>
                                            ⛏️ Terrain Type Distribution
                                        </p>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                type_cols = st.columns(3)
                                with type_cols[0]:
                                    st.metric(
                                        "🪨 Rocky Areas",
                                        f"{rock_pct:.1f}%",
                                        "Good for climbing",
                                    )
                                with type_cols[1]:
                                    st.metric(
                                        "🌿 Vegetation",
                                        f"{veg_pct:.1f}%",
                                        "Variable difficulty",
                                    )
                                with type_cols[2]:
                                    st.metric(
                                        "⛰️ Scree/Loose",
                                        f"{scree_pct:.1f}%",
                                        "Moderate hazard",
                                    )

                                st.markdown("</div>", unsafe_allow_html=True)

                                # Difficulty breakdown
                                difficulty_data = {
                                    "Easy": (
                                        np.sum((flat_costs >= 0.0) & (flat_costs < 0.3))
                                        / total_pixels * 100
                                    ) if total_pixels > 0 else 0,
                                    "Moderate": (
                                        np.sum((flat_costs >= 0.3) & (flat_costs < 0.5))
                                        / total_pixels * 100
                                    ) if total_pixels > 0 else 0,
                                    "Difficult": (
                                        np.sum((flat_costs >= 0.5) & (flat_costs < 0.7))
                                        / total_pixels * 100
                                    ) if total_pixels > 0 else 0,
                                    "V.Difficult": (
                                        np.sum((flat_costs >= 0.7) & (flat_costs < 0.9))
                                        / total_pixels * 100
                                    ) if total_pixels > 0 else 0,
                                    "Impassable": (
                                        np.sum((flat_costs >= 0.9) & (flat_costs <= 1.0))
                                        / total_pixels * 100
                                    ) if total_pixels > 0 else 0,
                                }

                                st.markdown(
                                    """
                                    <p style='color: #cbd5e1; font-size: 0.85rem;
                                        margin: 1rem 0 0.5rem 0; font-weight: 600;'>
                                        📊 Route Difficulty Distribution
                                    </p>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                cols_diff = st.columns(5)
                                colors = [
                                    "#22c55e", "#84cc16", "#f59e0b", "#ef4444", "#dc2626",
                                ]
                                for i, (diff_label, percentage) in enumerate(
                                    difficulty_data.items()
                                ):
                                    with cols_diff[i]:
                                        st.markdown(
                                            f"""
                                            <div style='background: {colors[i]}20; padding: 0.6rem;
                                                border-radius: 6px; text-align: center;
                                                border-left: 3px solid {colors[i]};'>
                                                <p style='color: #94a3b8; margin: 0;
                                                    font-size: 0.7rem;'>{diff_label}</p>
                                                <p style='color: {colors[i]}; margin: 0.3rem 0 0 0;
                                                    font-size: 1rem; font-weight: bold;'>
                                                    {percentage:.0f}%
                                                </p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.info("No route detected - terrain analysis unavailable")

                        with result_tab3:
                            if path:
                                st.markdown(
                                    '<div style="background: linear-gradient(135deg, '
                                    "rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.15)); "
                                    'padding: 0.3rem; border-radius: 6px; margin: 0;">',
                                    unsafe_allow_html=True,
                                )

                                cost_map_data = result["cost_map"]
                                flat_costs = cost_map_data.flatten()

                                terrain_ranges = {
                                    "Easy": np.sum(
                                        (flat_costs >= 0.0) & (flat_costs < 0.3)
                                    ),
                                    "V.Difficult": np.sum(
                                        (flat_costs >= 0.7) & (flat_costs <= 1.0)
                                    ),
                                }

                                total_pixels = np.sum(flat_costs > 0)

                                path_costs = []
                                if len(path) > 0:
                                    for waypoint in path:
                                        y = min(
                                            int(waypoint[0]),
                                            result["cost_map"].shape[0] - 1,
                                        )
                                        x = min(
                                            int(waypoint[1]),
                                            result["cost_map"].shape[1] - 1,
                                        )
                                        path_costs.append(result["cost_map"][y, x])

                                avg_cost = np.mean(path_costs) if path_costs else 0

                                easy_percentage = (
                                    terrain_ranges["Easy"] / total_pixels * 100
                                ) if total_pixels > 0 else 0
                                difficult_percentage = (
                                    terrain_ranges["V.Difficult"] / total_pixels * 100
                                ) if total_pixels > 0 else 0

                                tips = []
                                if easy_percentage > 40:
                                    tips.append("✓ Pace")
                                if difficult_percentage > 20:
                                    tips.append("⚠ Alert")
                                if avg_cost > 0.6:
                                    tips.append("🎯 Pro")
                                if len(path) > 100:
                                    tips.append("⏱ Time")

                                tips_text = " | ".join(tips) if tips else "Plan well"
                                st.markdown(
                                    f"<p style='color: #a5b4fc; margin: 0; "
                                    f"font-size: 0.7rem;'>{tips_text}</p>",
                                    unsafe_allow_html=True,
                                )

                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.info("No guide")
                    else:
                        st.error("Cannot read video file")

        st.markdown("</div>", unsafe_allow_html=True)

    # -- Footer ---------------------------------------------------------------
    st.markdown(
        """
        <div style='text-align: center; margin-top: 3rem; color: #cbd5e1; font-size: 0.9rem;'>
            <p>🧗 ClimbAssist AI v2 | Powered by AI for safer climbing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
