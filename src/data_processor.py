"""
Data Processing Utilities for YourCabs
Functions for data cleaning, validation, and preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import os


class DataProcessor:
    """Data processing utilities for YourCabs booking data"""
    
    def __init__(self):
        self.cleaning_stats = {}
        self.feature_mappings = self._get_feature_mappings()
    
    def _get_feature_mappings(self) -> Dict:
        """Define feature mappings for data processing"""
        return {
            'travel_types': {1: 'Business', 2: 'Leisure', 3: 'Other'},
            'booking_channels': ['online', 'mobile', 'phone', 'other'],
            'boolean_fields': ['online_booking', 'mobile_site_booking', 'is_round_trip']
        }
    
    def clean_booking_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean and preprocess booking data"""
        
        print("🧹 Starting data cleaning process...")
        df_clean = df.copy()
        cleaning_report = {
            'original_shape': df.shape,
            'cleaning_steps': [],
            'feature_engineering': [],
            'final_shape': None,
            'cleaned_at': datetime.now().isoformat()
        }
        
        # 1. Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        print(f"📊 Missing values before cleaning: {missing_before:,}")
        
        # Fill numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                cleaning_report['cleaning_steps'].append(f"Filled {col} missing values with median: {median_val}")
        
        # Fill categorical columns with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
                cleaning_report['cleaning_steps'].append(f"Filled {col} missing values with mode: {mode_val}")
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"✅ Missing values after cleaning: {missing_after:,}")
        
        # 2. Handle outliers in numeric columns
        for col in numeric_cols:
            if col not in ['Car_Cancellation']:  # Skip target variable
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if outliers_before > 0:
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                    cleaning_report['cleaning_steps'].append(f"Clipped {outliers_before} outliers in {col}")
        
        # 3. Feature Engineering
        df_clean = self._engineer_features(df_clean, cleaning_report)
        
        # 4. Data type optimization
        df_clean = self._optimize_dtypes(df_clean, cleaning_report)
        
        cleaning_report['final_shape'] = df_clean.shape
        print(f"✅ Data cleaning completed: {df.shape} → {df_clean.shape}")
        
        return df_clean, cleaning_report
    
    def _engineer_features(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Engineer new features from existing data"""
        
        # Time-based features
        if 'booking_created' in df.columns:
            df['booking_created'] = pd.to_datetime(df['booking_created'], errors='coerce')
            df['booking_hour'] = df['booking_created'].dt.hour
            df['booking_day_of_week'] = df['booking_created'].dt.dayofweek
            df['booking_month'] = df['booking_created'].dt.month
            df['is_weekend'] = (df['booking_day_of_week'] >= 5).astype(int)
            df['is_late_booking'] = ((df['booking_hour'] <= 6) | (df['booking_hour'] >= 22)).astype(int)
            
            report['feature_engineering'].append("Created time-based features: hour, day_of_week, month, is_weekend, is_late_booking")
        
        # Travel type features
        if 'travel_type_id' in df.columns:
            df['is_business_travel'] = (df['travel_type_id'] == 1).astype(int)
            df['is_leisure_travel'] = (df['travel_type_id'] == 2).astype(int)
            report['feature_engineering'].append("Created travel type binary features")
        
        # Booking channel features
        if 'booking_channel' in df.columns:
            for channel in ['mobile', 'online', 'other']:
                df[f'channel_{channel}'] = (df['booking_channel'] == channel).astype(int)
            report['feature_engineering'].append("Created booking channel dummy variables")
        
        # Round trip feature
        if 'travel_type_id' in df.columns:
            # Assume round trip more likely for leisure travel
            df['is_round_trip'] = (df['travel_type_id'] == 2).astype(int)
            report['feature_engineering'].append("Created is_round_trip feature based on travel type")
        
        # Distance features (if coordinates available)
        if all(col in df.columns for col in ['from_lat', 'from_long', 'to_lat', 'to_long']):
            df['trip_distance'] = self._calculate_distance(
                df['from_lat'], df['from_long'], df['to_lat'], df['to_long']
            )
            df['is_long_distance'] = (df['trip_distance'] > 50).astype(int)  # 50km threshold
            report['feature_engineering'].append("Created distance-based features")
        
        return df
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    
    def _optimize_dtypes(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        
        memory_before = df.memory_usage(deep=True).sum()
        
        # Convert appropriate columns to categorical
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 50:
                df[col] = df[col].astype('category')
        
        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0 and df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].min() >= -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -2147483648 and df[col].max() < 2147483647:
                df[col] = df[col].astype('int32')
        
        memory_after = df.memory_usage(deep=True).sum()
        memory_saved = memory_before - memory_after
        
        if memory_saved > 0:
            report['cleaning_steps'].append(f"Optimized data types - saved {memory_saved/1024/1024:.1f} MB")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and return report"""
        
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'quality_score': 0,
            'issues': []
        }
        
        # Calculate quality score
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        duplicate_pct = quality_report['duplicate_rows'] / df.shape[0]
        
        quality_score = 100
        if missing_pct > 0.05:  # More than 5% missing
            quality_score -= 20
            quality_report['issues'].append(f"High missing data: {missing_pct:.1%}")
        
        if duplicate_pct > 0.01:  # More than 1% duplicates
            quality_score -= 10
            quality_report['issues'].append(f"Duplicate rows found: {duplicate_pct:.1%}")
        
        # Check for data consistency
        if 'Car_Cancellation' in df.columns:
            if df['Car_Cancellation'].isnull().sum() > 0:
                quality_score -= 30
                quality_report['issues'].append("Missing values in target variable")
            
            if not df['Car_Cancellation'].isin([0, 1]).all():
                quality_score -= 20
                quality_report['issues'].append("Invalid values in target variable")
        
        quality_report['quality_score'] = max(0, quality_score)
        
        return quality_report
    
    def export_cleaned_data(self, df: pd.DataFrame, output_path: str, 
                          cleaning_report: Dict) -> bool:
        """Export cleaned data and cleaning report"""
        
        try:
            # Save cleaned data
            df.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to: {output_path}")
            
            # Save cleaning report
            report_path = output_path.replace('.csv', '_cleaning_report.json')
            with open(report_path, 'w') as f:
                # Convert any non-serializable objects to strings
                serializable_report = self._make_serializable(cleaning_report)
                json.dump(serializable_report, f, indent=2)
            
            print(f"✅ Cleaning report saved to: {report_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving cleaned data: {str(e)}")
            return False
    
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'dtype'):  # numpy/pandas objects
            return str(obj)
        else:
            return obj
    
    def generate_sample_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate sample booking data for testing"""
        
        np.random.seed(42)
        
        sample_data = {
            'booking_created': pd.date_range(
                start='2024-01-01', 
                end='2024-12-31', 
                periods=n_samples
            ),
            'online_booking': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'mobile_site_booking': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'vehicle_model_id': np.random.randint(1, 51, n_samples),
            'travel_type_id': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
            'from_area_id': np.random.randint(1, 101, n_samples),
            'to_area_id': np.random.randint(1, 101, n_samples),
            'from_lat': np.random.uniform(12.0, 13.5, n_samples),
            'from_long': np.random.uniform(77.0, 78.5, n_samples),
            'to_lat': np.random.uniform(12.0, 13.5, n_samples),
            'to_long': np.random.uniform(77.0, 78.5, n_samples),
            'from_city_id': np.random.randint(1, 11, n_samples),
            'booking_channel': np.random.choice(['online', 'mobile', 'phone', 'other'], n_samples),
            'Car_Cancellation': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        return pd.DataFrame(sample_data)


def load_and_clean_data(file_path: str, export_path: str = None) -> Tuple[pd.DataFrame, Dict]:
    """Load and clean booking data from file"""
    
    processor = DataProcessor()
    
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel files.")
    
    print(f"📁 Loaded data from: {file_path}")
    print(f"📊 Original shape: {df.shape}")
    
    # Clean data
    df_clean, cleaning_report = processor.clean_booking_data(df)
    
    # Export if path provided
    if export_path:
        processor.export_cleaned_data(df_clean, export_path, cleaning_report)
    
    return df_clean, cleaning_report


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Generate sample data
    sample_df = processor.generate_sample_data(1000)
    print("📊 Generated sample data:")
    print(sample_df.info())
    
    # Clean the sample data
    clean_df, report = processor.clean_booking_data(sample_df)
    
    # Validate quality
    quality = processor.validate_data_quality(clean_df)
    print(f"\n📈 Data Quality Score: {quality['quality_score']}/100")
    
    if quality['issues']:
        print("⚠️ Quality Issues:")
        for issue in quality['issues']:
            print(f"  • {issue}")
    else:
        print("✅ No quality issues found!")
