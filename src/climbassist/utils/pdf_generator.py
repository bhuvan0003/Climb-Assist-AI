"""PDF report generation for gear lists."""

from fpdf import FPDF


class GearPDFGenerator:
    """Generate PDF reports for climbing gear recommendations."""

    @staticmethod
    def generate(gear_list: list, expedition_days: int = 0,
                 max_altitude: int = 0, weather: str = "",
                 skill: str = "") -> bytes:
        """
        Generate a PDF containing the recommended gear list.

        Returns:
            PDF content as bytes.
        """
        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(200, 10, txt="ClimbAssist AI - Recommended Gear List",
                 ln=True, align="C")
        pdf.ln(5)

        # Expedition parameters
        if expedition_days or max_altitude or weather or skill:
            pdf.set_font("Arial", "B", size=12)
            pdf.cell(200, 8, txt="Expedition Parameters:", ln=True)
            pdf.set_font("Arial", size=11)
            if expedition_days:
                pdf.cell(200, 7, txt=f"  Duration: {expedition_days} days", ln=True)
            if max_altitude:
                pdf.cell(200, 7, txt=f"  Max Altitude: {max_altitude} m", ln=True)
            if weather:
                pdf.cell(200, 7, txt=f"  Weather: {weather}", ln=True)
            if skill:
                pdf.cell(200, 7, txt=f"  Skill Level: {skill}", ln=True)
            pdf.ln(5)

        # Gear list
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(200, 8, txt="Recommended Gear:", ln=True)
        pdf.set_font("Arial", size=11)
        for item in gear_list:
            pdf.cell(200, 8, txt=f"- {item}", ln=True)

        return bytes(pdf.output(dest="S"))
