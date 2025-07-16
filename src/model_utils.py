"""
YourCabs Model Utilities
Utility functions for model loading, prediction, and data processing
"""

import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional


class ModelPredictor:
    """Main class for loading model and making predictions"""
    
    def __init__(self, model_path: str = None, metadata_path: str = None):
        # Get project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default paths if not provided
        if model_path is None:
            self.model_path = os.path.join(self.project_root, "models", "best_model.joblib")
        else:
            self.model_path = model_path
            
        if metadata_path is None:
            self.metadata_path = os.path.join(self.project_root, "models", "model_info.json")
        else:
            self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.features = None
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model and metadata"""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                # Use the exact feature set that the model was trained with
                self.features = [
                    'online_booking', 'mobile_site_booking', 'vehicle_model_id', 
                    'travel_type_id', 'from_area_id', 'to_area_id', 'booking_hour',
                    'booking_day_of_week', 'booking_month', 'is_round_trip', 
                    'channel_mobile', 'channel_online', 'channel_other', 'from_lat', 
                    'from_long', 'to_lat', 'to_long', 'from_city_id'
                ]
                print(f"✅ Metadata loaded: {len(self.features)} features (model-corrected)")
            else:
                print("⚠️ Metadata file not found, using default features")
                self.features = self._get_default_features()
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def _get_default_features(self) -> List[str]:
        """Default feature list if metadata is not available"""
        return [
            'online_booking', 'mobile_site_booking', 'vehicle_model_id', 
            'travel_type_id', 'from_area_id', 'to_area_id', 'booking_hour',
            'booking_day_of_week', 'booking_month', 'is_round_trip', 
            'channel_mobile', 'channel_online', 'channel_other', 'from_lat', 
            'from_long', 'to_lat', 'to_long', 'from_city_id'
        ]
    
    def preprocess_input(self, booking_data: Dict) -> pd.DataFrame:
        """Preprocess single booking input for prediction"""
        # Create empty dataframe with required features
        df = pd.DataFrame(index=[0])
        
        # Initialize all features with 0
        for feature in self.features:
            df[feature] = 0
        
        # Time-based features (only create the ones the model expects)
        if 'booking_created' in booking_data and booking_data['booking_created']:
            booking_time = pd.to_datetime(booking_data['booking_created'])
            df['booking_hour'] = booking_time.hour
            df['booking_day_of_week'] = booking_time.dayofweek
            df['booking_month'] = booking_time.month
            # Note: is_weekend and is_late_booking are not in the model's feature set
        
        # Direct feature mappings
        feature_mappings = {
            'online_booking': 'online_booking',
            'mobile_site_booking': 'mobile_site_booking',
            'vehicle_model_id': 'vehicle_model_id',
            'travel_type_id': 'travel_type_id',
            'from_area_id': 'from_area_id',
            'to_area_id': 'to_area_id',
            'from_lat': 'from_lat',
            'from_long': 'from_long',
            'to_lat': 'to_lat',
            'to_long': 'to_long',
            'from_city_id': 'from_city_id'
        }
        
        for input_key, feature_name in feature_mappings.items():
            if input_key in booking_data and feature_name in df.columns:
                df[feature_name] = booking_data[input_key]
        
        # Travel type features (not in model's feature set, so skip)
        # The model doesn't use is_business_travel and is_leisure_travel
        
        # Round trip feature
        if 'is_round_trip' in booking_data:
            df['is_round_trip'] = 1 if booking_data['is_round_trip'] else 0
        
        # Booking channel features
        if 'booking_channel' in booking_data:
            channel = booking_data['booking_channel']
            df['channel_mobile'] = 1 if channel == 'mobile' else 0
            df['channel_online'] = 1 if channel == 'online' else 0
            df['channel_other'] = 1 if channel not in ['mobile', 'online'] else 0
        
        return df
    
    def predict(self, booking_data: Dict) -> Tuple[float, str, Dict]:
        """Make prediction for a single booking"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess input
        df = self.preprocess_input(booking_data)
        
        # Make prediction (using median as threshold: 10.9%)
        probability = self.model.predict_proba(df)[0, 1]
        prediction = 1 if probability > 0.109 else 0
        
        # Determine risk category (adjusted thresholds for better classification)
        # New thresholds based on observed model behavior
        if probability >= 0.50:
            risk_category = "Critical"
            risk_color = "🔴"
        elif probability >= 0.30:
            risk_category = "High" 
            risk_color = "🟠"
        elif probability >= 0.15:
            risk_category = "Medium"
            risk_color = "🟡"
        elif probability >= 0.05:
            risk_category = "Low"
            risk_color = "🟢"
        else:
            risk_category = "Very Low"
            risk_color = "🟢"
        
        # Create detailed result
        result = {
            'probability': probability,
            'prediction': prediction,
            'risk_category': risk_category,
            'risk_color': risk_color,
            'confidence': 'High' if probability < 0.1 or probability > 0.9 else 'Medium',
            'recommendation': self._get_recommendation(probability, risk_category)
        }
        
        return probability, f"{risk_color} {risk_category}", result
    
    def _get_recommendation(self, probability: float, risk_category: str) -> str:
        """Get business recommendation based on risk level (adjusted thresholds)"""
        if probability >= 0.50:
            return "Critical risk - Immediate intervention required, contact customer proactively"
        elif probability >= 0.30:
            return "High risk - Consider offering incentives or flexible booking options"
        elif probability >= 0.15:
            return "Medium risk - Send confirmation reminders and monitor closely"
        elif probability >= 0.05:
            return "Low risk - Standard monitoring, but keep an eye on booking status"
        else:
            return "Very low risk - Standard processing, minimal intervention needed"
    
    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for multiple bookings"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Prepare features
        X = pd.DataFrame(index=df.index)
        for feature in self.features:
            if feature in df.columns:
                X[feature] = df[feature]
            else:
                X[feature] = 0
        
        # Fill missing values
        X = X.fillna(0)
        
        # Make predictions (using model's median as threshold)
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities > 0.109).astype(int)
        
        # Add risk categories
        risk_categories = pd.cut(probabilities, 
                               bins=[0, 0.15, 0.3, 0.5, 1.0], 
                               labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Create results dataframe
        results = df.copy()
        results['cancellation_probability'] = probabilities
        results['high_risk_prediction'] = predictions
        results['risk_category'] = risk_categories
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information for display"""
        if self.metadata:
            return {
                'Model Type': self.metadata.get('model_type', 'XGBoost'),
                'Version': self.metadata.get('model_version', '2.0'),
                'Training Date': self.metadata.get('training_date', 'Unknown'),
                'Test AUC': f"{self.metadata.get('test_auc', 0):.1%}",
                'CV AUC': f"{self.metadata.get('cv_auc', 0):.1%}",
                'Feature Count': len(self.features),
                'Configuration': self.metadata.get('configuration', 'XGBoost with scale_pos_weight')
            }
        else:
            return {
                'Model Type': 'XGBoost',
                'Version': '2.0',
                'Status': 'Loaded without metadata',
                'Feature Count': len(self.features) if self.features else 0
            }


def get_sample_booking_data() -> Dict:
    """Generate sample booking data for testing"""
    return {
        'booking_created': datetime.now(),
        'online_booking': 1,
        'mobile_site_booking': 0,
        'vehicle_model_id': 15,
        'travel_type_id': 1,  # Business travel
        'from_area_id': 5,
        'to_area_id': 12,
        'from_lat': 12.9716,
        'from_long': 77.5946,
        'to_lat': 12.2958,
        'to_long': 76.6394,
        'from_city_id': 1,
        'is_round_trip': True,
        'booking_channel': 'online'
    }


def validate_input_data(data: Dict) -> Tuple[bool, List[str]]:
    """Validate input data for prediction"""
    errors = []
    
    # Check required fields
    required_fields = ['vehicle_model_id', 'travel_type_id', 'from_area_id', 'to_area_id']
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Check numeric ranges
    if 'vehicle_model_id' in data and data['vehicle_model_id'] < 0:
        errors.append("Vehicle model ID must be positive")
    
    if 'travel_type_id' in data and data['travel_type_id'] not in [1, 2, 3]:
        errors.append("Travel type ID must be 1 (Business), 2 (Leisure), or 3 (Other)")
    
    # Check coordinates if provided
    if 'from_lat' in data and data['from_lat'] and (data['from_lat'] < -90 or data['from_lat'] > 90):
        errors.append("Latitude must be between -90 and 90")
    
    if 'from_long' in data and data['from_long'] and (data['from_long'] < -180 or data['from_long'] > 180):
        errors.append("Longitude must be between -180 and 180")
    
    return len(errors) == 0, errors
