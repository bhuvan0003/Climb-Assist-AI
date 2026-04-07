"""
Application constants: CSS, gear data, thresholds, tips.
"""

# ============================================================================
# GEAR DATA
# ============================================================================

BASE_GEAR = [
    "Helmet", "Harness", "60 m Rope", "Quickdraws (12-15)",
    "Belay Device (ATC/GriGri)", "Carabiners (Locking)",
    "Climbing Shoes", "Chalk Bag",
]

GEAR_TIPS = [
    "Pack 20% lighter than you think you need",
    "Consider weather changes at high altitude",
    "Multi-purpose gear saves weight and space",
    "Always pack backup safety equipment",
    "Test all gear before your expedition",
]

VIDEO_TIPS = [
    "Best results: Side-angle view with full body visible",
    "Ensure good lighting for accurate pose detection",
    "Capture 10-30 seconds of continuous climbing",
]

ADDITIONAL_TIPS = [
    "Focus on footwork precision",
    "Keep hips close to the wall",
    "Practice silent climbing for better technique",
]

# ============================================================================
# DETECTION THRESHOLDS
# ============================================================================

POSE_CONFIDENCE_THRESHOLD = 0.3
WIDE_STANCE_THRESHOLD = 0.35
HIPS_AWAY_THRESHOLD = 0.65
POOR_HAND_CONFIDENCE = 0.25
UNSTABLE_KNEES_RATIO = 1.5
OVERREACHING_THRESHOLD = 0.2
ISSUE_DETECTION_RATE = 0.10  # 10% of valid frames

# ============================================================================
# ROUTE ANALYSIS
# ============================================================================

ROUTE_QUALITY_THRESHOLDS = {
    "EXCELLENT": 0.3,
    "GOOD": 0.5,
    "MODERATE": 0.7,
}

DIFFICULTY_THRESHOLDS = {
    "VERY EASY": (0.0, 0.2),
    "EASY": (0.2, 0.4),
    "MODERATE": (0.4, 0.6),
    "HARD": (0.6, 0.8),
    "EXTREME": (0.8, 1.0),
}

# ============================================================================
# APPLICATION CSS
# ============================================================================

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #ffffff;
}

.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 2rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    border: 1px solid rgba(139, 92, 246, 0.5);
    box-shadow: 0 8px 32px 0 rgba(139, 92, 246, 0.2);
}

.gear-list-card {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
    border-radius: 16px;
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 1.5rem;
    margin-top: 1rem;
    color: #ffffff;
}

.value-indicator {
    display: inline-block;
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
    color: #fff;
    border-radius: 12px;
    padding: 4px 16px;
    font-size: 0.9rem;
    margin-left: 8px;
    font-weight: 600;
}

.upload-zone {
    border: 2px dashed rgba(139, 92, 246, 0.5);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    background: rgba(139, 92, 246, 0.05);
}

.status-bar {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    font-weight: 600;
    margin: 1.5rem 0;
}

.mistake-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
    border-left: 4px solid #ef4444;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    border-radius: 8px;
    color: #fecaca;
}

.recommend-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.15) 100%);
    border-left: 4px solid #22c55e;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    border-radius: 8px;
    color: #bbf7d0;
}

.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
}

h2 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

h3 {
    color: #e0e7ff !important;
    font-weight: 600 !important;
}

.block-container {
    padding-top: 0rem !important;
    max-width: 100% !important;
}
</style>
"""
