# ClimbAssist AI

AI-Powered Climbing Analysis and Safe Route Planning.

ClimbAssist AI is a Streamlit-based application that combines computer vision, biomechanical analysis, and graph-based route planning to help climbers optimize their gear, improve movement technique, and find safe routes on climbing walls.

## Features

The application provides three main analysis tabs:

### Gear Optimizer
Analyzes climbing gear from uploaded images, identifies equipment, and provides weight-optimized recommendations based on climb type (sport, trad, bouldering, alpine).

### Movement Analyzer
Processes climbing videos to evaluate body position, joint angles, balance metrics, and movement efficiency. Generates a downloadable PDF report with frame-by-frame analysis and technique scores.

### Route Finder
Builds a graph model of detected holds on a climbing wall image and computes the safest path using hold quality scores, distances, and biomechanical constraints. Visualizes the recommended route overlaid on the wall photo.

## Tech Stack

- **Frontend**: Streamlit
- **Core Analysis**: NumPy, SciPy, scikit-learn
- **Computer Vision**: OpenCV, Pillow
- **Graph & Routing**: NetworkX
- **Reporting**: FPDF2, Matplotlib
- **Deployment**: Streamlit Community Cloud / Vercel (API stub)

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/climbassist-ai.git
cd climbassist-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the application
streamlit run app.py
```

### Cloud Deployment

The application is designed for Streamlit Community Cloud. Connect the repository and set the main file to `app.py`.

A Vercel deployment stub is included in `api/index.py` for basic health-check hosting.

## Project Structure

```
climbassist-ai/
├── app.py                  # Thin entry point
├── pyproject.toml          # Build and tool configuration
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Test dependencies
├── vercel.json             # Vercel deployment config
├── api/
│   └── index.py            # Vercel serverless stub
├── src/
│   └── climbassist/        # Main package
│       ├── app.py          # Streamlit application
│       ├── gear_optimizer/ # Gear analysis module
│       ├── movement_analyzer/ # Movement analysis module
│       └── route_finder/   # Route planning module
├── tests/                  # Test suite
├── docs/                   # Project documentation
│   ├── DEPLOYMENT_GUIDE.md
│   ├── SETUP_STATUS.md
│   ├── ENHANCEMENTS.md
│   └── WHAT_IS_NEW.md
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions CI
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=climbassist --cov-report=term-missing
```

## Deployment

| Platform                  | Method                                    |
|---------------------------|-------------------------------------------|
| Streamlit Community Cloud | Connect repo, set main file to `app.py`   |
| Vercel                    | Auto-deploys via `vercel.json`            |
| Local                     | `streamlit run app.py`                    |

See `docs/DEPLOYMENT_GUIDE.md` for detailed instructions.

## License

MIT
