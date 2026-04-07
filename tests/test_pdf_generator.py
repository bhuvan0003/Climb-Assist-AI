"""Tests for GearPDFGenerator utility."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climbassist.utils.pdf_generator import GearPDFGenerator


class TestGearPDFGenerator:
    """Tests for PDF generation."""

    def test_generate_returns_bytes(self):
        """Result should be bytes."""
        gear = ["Helmet", "Harness", "Rope"]
        result = GearPDFGenerator.generate(gear)
        assert isinstance(result, bytes)

    def test_pdf_not_empty(self):
        """Generated PDF should have non-zero length."""
        gear = ["Helmet", "Harness", "Rope"]
        result = GearPDFGenerator.generate(gear)
        assert len(result) > 0

    def test_generate_with_params(self):
        """No error when all expedition params are provided."""
        gear = [
            "Helmet", "Harness", "60 m Rope", "Quickdraws (12-15)",
            "Down Jacket", "Ice Axe", "Crampons", "Waterproof Shell",
        ]
        result = GearPDFGenerator.generate(
            gear_list=gear,
            expedition_days=10,
            max_altitude=5500,
            weather="Snow",
            skill="Advanced",
        )
        assert isinstance(result, bytes)
        assert len(result) > 100  # A real PDF is at least several hundred bytes

    def test_generate_empty_gear_list(self):
        """Empty gear list should still produce valid PDF bytes."""
        result = GearPDFGenerator.generate([])
        assert isinstance(result, bytes)
        assert len(result) > 0
