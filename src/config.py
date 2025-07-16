"""
Configuration settings for YourCabs Prediction Application
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Model paths
MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_info.json"

# Data paths
CLEANED_DATA_PATH = DATA_DIR / "YourCabs_cleaned.csv"
ORIGINAL_DATA_PATH = DATA_DIR / "YourCabs.csv"
CLEANING_REPORT_PATH = DATA_DIR / "cleaning_report.json"

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "YourCabs - Cancellation Prediction",
    "page_icon": "🚗",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model configuration
MODEL_CONFIG = {
    "probability_threshold": 0.3,
    "risk_thresholds": {
        "low": 0.15,
        "medium": 0.30,
        "high": 0.50,
        "critical": 1.0
    },
    "risk_colors": {
        "Low": "#28a745",
        "Medium": "#ffc107", 
        "High": "#fd7e14",
        "Critical": "#dc3545"
    }
}

# Feature configuration
REQUIRED_FEATURES = [
    'online_booking', 'mobile_site_booking', 'vehicle_model_id', 
    'travel_type_id', 'from_area_id', 'to_area_id', 'from_lat', 
    'from_long', 'to_lat', 'to_long', 'from_city_id', 'booking_hour',
    'booking_day_of_week', 'booking_month', 'is_weekend', 
    'is_late_booking', 'is_round_trip', 'is_business_travel', 
    'is_leisure_travel', 'channel_mobile', 'channel_online', 'channel_other'
]

# UI configuration
UI_CONFIG = {
    "max_file_size_mb": 50,
    "max_records_display": 1000,
    "batch_size": 5000,
    "chart_height": 400,
    "gauge_height": 300
}

# Validation rules
VALIDATION_RULES = {
    "vehicle_model_id": {"min": 1, "max": 100},
    "travel_type_id": {"values": [1, 2, 3]},
    "from_area_id": {"min": 1, "max": 1000},
    "to_area_id": {"min": 1, "max": 1000},
    "from_city_id": {"min": 1, "max": 100},
    "from_lat": {"min": -90, "max": 90},
    "from_long": {"min": -180, "max": 180},
    "to_lat": {"min": -90, "max": 90},
    "to_long": {"min": -180, "max": 180}
}

# Application messages
MESSAGES = {
    "model_load_success": "✅ Model loaded successfully!",
    "model_load_error": "❌ Failed to load model",
    "prediction_success": "🔮 Prediction completed successfully!",
    "prediction_error": "❌ Prediction failed",
    "file_upload_success": "✅ File uploaded successfully!",
    "file_upload_error": "❌ File upload failed",
    "validation_error": "❌ Input validation failed"
}

# Help texts
HELP_TEXTS = {
    "booking_date": "Date when the booking was created",
    "booking_time": "Time when the booking was created",
    "online_booking": "Was this booking made online?",
    "mobile_site_booking": "Was this booking made via mobile site?",
    "booking_channel": "Channel used for booking",
    "is_round_trip": "Is this a round trip booking?",
    "vehicle_model_id": "ID of the vehicle model (1-100)",
    "travel_type_id": "Purpose of travel: 1=Business, 2=Leisure, 3=Other",
    "from_area_id": "Source area identifier",
    "to_area_id": "Destination area identifier",
    "from_lat": "Source latitude (-90 to 90)",
    "from_long": "Source longitude (-180 to 180)",
    "to_lat": "Destination latitude (-90 to 90)",
    "to_long": "Destination longitude (-180 to 180)",
    "from_city_id": "Source city identifier"
}

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, SRC_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_status():
    """Check if model files exist"""
    return {
        "model_exists": MODEL_PATH.exists(),
        "metadata_exists": MODEL_METADATA_PATH.exists(),
        "data_exists": CLEANED_DATA_PATH.exists() or ORIGINAL_DATA_PATH.exists()
    }
