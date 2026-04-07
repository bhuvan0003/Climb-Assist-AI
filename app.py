"""ClimbAssist AI v2 - Entry Point"""
import sys
import os

# Add src/ to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from climbassist.app import main

main()
