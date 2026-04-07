"""
Gear Load Optimizer Module
===========================
Pure logic for climbing gear recommendation based on expedition parameters.
No Streamlit dependency — fully testable independently.
"""

from typing import List
from climbassist.config.constants import BASE_GEAR


class GearOptimizer:
    """Recommend climbing gear based on expedition parameters."""

    @staticmethod
    def optimize(expedition_days: int, max_altitude: int,
                 weather: str, skill: str) -> List[str]:
        """
        Generate an optimized gear list.

        Args:
            expedition_days: Duration in days (1-20).
            max_altitude: Maximum altitude in meters.
            weather: Expected weather ('Clear', 'Windy', 'Rainy', 'Snow').
            skill: Climber skill level ('Beginner', 'Intermediate', 'Advanced').

        Returns:
            List of recommended gear items.
        """
        extra_gear: List[str] = []

        if expedition_days > 7:
            extra_gear.extend(["Portaledge / Bivy Gear", "Slings/Runners"])

        if max_altitude > 4000:
            extra_gear.extend(["Down Jacket", "High-Altitude Boots", "Gloves"])

        if max_altitude > 5000:
            extra_gear.extend(["Ice Axe", "Crampons"])

        if weather in ("Rainy", "Snow"):
            extra_gear.append("Waterproof Shell")

        if skill == "Beginner":
            extra_gear.extend([
                "Extra Anchors / Assisted Belay Device",
                "Crash Pad",
            ])

        return list(BASE_GEAR) + extra_gear
