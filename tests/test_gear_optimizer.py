"""Tests for GearOptimizer module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climbassist.modules.gear_optimizer import GearOptimizer


class TestGearOptimizer:
    """Tests for GearOptimizer.optimize()."""

    def test_base_gear_always_included(self):
        """Verify 8 base items present for default params."""
        result = GearOptimizer.optimize(
            expedition_days=3, max_altitude=2000,
            weather="Clear", skill="Intermediate"
        )
        base_items = [
            "Helmet", "Harness", "60 m Rope", "Quickdraws (12-15)",
            "Belay Device (ATC/GriGri)", "Carabiners (Locking)",
            "Climbing Shoes", "Chalk Bag",
        ]
        for item in base_items:
            assert item in result, f"Base item '{item}' missing from gear list"
        assert len([i for i in result if i in base_items]) == 8

    def test_altitude_above_4000_adds_gear(self):
        """Down Jacket, High-Altitude Boots, Gloves added above 4000m."""
        result = GearOptimizer.optimize(
            expedition_days=3, max_altitude=4500,
            weather="Clear", skill="Intermediate"
        )
        assert "Down Jacket" in result
        assert "High-Altitude Boots" in result
        assert "Gloves" in result

    def test_altitude_above_5000_adds_ice_gear(self):
        """Ice Axe and Crampons added above 5000m."""
        result = GearOptimizer.optimize(
            expedition_days=3, max_altitude=5500,
            weather="Clear", skill="Intermediate"
        )
        assert "Ice Axe" in result
        assert "Crampons" in result

    def test_rainy_weather_adds_shell(self):
        """Waterproof Shell added for Rainy weather."""
        result = GearOptimizer.optimize(
            expedition_days=3, max_altitude=2000,
            weather="Rainy", skill="Intermediate"
        )
        assert "Waterproof Shell" in result

    def test_snow_weather_adds_shell(self):
        """Waterproof Shell added for Snow weather."""
        result = GearOptimizer.optimize(
            expedition_days=3, max_altitude=2000,
            weather="Snow", skill="Intermediate"
        )
        assert "Waterproof Shell" in result

    def test_beginner_adds_extra_gear(self):
        """Extra Anchors and Crash Pad added for Beginner skill."""
        result = GearOptimizer.optimize(
            expedition_days=3, max_altitude=2000,
            weather="Clear", skill="Beginner"
        )
        assert "Extra Anchors / Assisted Belay Device" in result
        assert "Crash Pad" in result

    def test_advanced_no_beginner_gear(self):
        """No Crash Pad for Advanced skill level."""
        result = GearOptimizer.optimize(
            expedition_days=3, max_altitude=2000,
            weather="Clear", skill="Advanced"
        )
        assert "Crash Pad" not in result
        assert "Extra Anchors / Assisted Belay Device" not in result

    def test_long_expedition_adds_bivy(self):
        """Portaledge / Bivy Gear added for expeditions > 7 days."""
        result = GearOptimizer.optimize(
            expedition_days=10, max_altitude=2000,
            weather="Clear", skill="Intermediate"
        )
        assert "Portaledge / Bivy Gear" in result
        assert "Slings/Runners" in result

    def test_no_duplicates(self):
        """Gear list should contain no duplicate items."""
        result = GearOptimizer.optimize(
            expedition_days=10, max_altitude=6000,
            weather="Snow", skill="Beginner"
        )
        assert len(result) == len(set(result)), "Duplicate items found in gear list"

    def test_clear_weather_no_shell(self):
        """No Waterproof Shell for Clear weather."""
        result = GearOptimizer.optimize(
            expedition_days=3, max_altitude=2000,
            weather="Clear", skill="Intermediate"
        )
        assert "Waterproof Shell" not in result
