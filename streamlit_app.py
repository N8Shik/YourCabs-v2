"""
YourCabs v2.0 - Streamlit App Entry Point
Alternative entry point for platforms that prefer streamlit_app.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main application
from app import *
