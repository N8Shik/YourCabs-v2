"""
YourCabs Cancellation Prediction - Streamlit Application (Simplified)
A robust, error-handling version of the prediction app
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import random

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(current_dir)

# Page configuration
st.set_page_config(
    page_title="YourCabs - Cancellation Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        color: #155724;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 2px solid #ffc107;
        color: #856404;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.predictor = None
    st.session_state.load_sample = False
    st.session_state.sample_data = None
    st.session_state.show_input_guide = False
    st.session_state.clear_form = False

def get_sample_data(risk_level: str):
    """Generate sample data for different risk levels - using REALISTIC training data ranges"""
    base_time = datetime.now()
    
    if risk_level == "low_risk":
        return {
            'booking_date': base_time.date(),
            'booking_time': base_time.replace(hour=14, minute=30).time(),  # Afternoon booking
            'online_booking': 1,
            'mobile_site_booking': 1,
            'booking_channel': 'online',
            'is_round_trip': True,
            'vehicle_model_id': 90,  # CORRECTED: 0% cancellation rate (safest)
            'travel_type_id': 1,  # Business travel (0% cancellation rate)
            'from_area_id': 100,  # CORRECTED: Within training range (6-1391)
            'to_area_id': 150,   # CORRECTED: Within training range (25-1390)
            'from_lat': 12.97,   # CORRECTED: Within training range (12.78-13.24)
            'from_long': 77.59,  # CORRECTED: Within training range (77.47-77.79)
            'to_lat': 13.05,     # CORRECTED: Within training range
            'to_long': 77.63,    # CORRECTED: Within training range
            'from_city_id': 5    # CORRECTED: Within training range (1-15)
        }
    elif risk_level == "medium_risk":
        return {
            'booking_date': base_time.date(),
            'booking_time': base_time.replace(hour=21, minute=0).time(),  # Evening booking (9 PM)
            'online_booking': 1,
            'mobile_site_booking': 0,
            'booking_channel': 'online',
            'is_round_trip': False,
            'vehicle_model_id': 2,   # Vehicle with actual medium risk profile
            'travel_type_id': 2,     # Leisure travel
            'from_area_id': 804,     # Real medium risk area combination
            'to_area_id': 177,
            'from_lat': 12.890,
            'from_long': 77.601,
            'to_lat': 12.934,
            'to_long': 77.611,
            'from_city_id': 1
        }
    elif risk_level == "high_risk":
        return {
            'booking_date': base_time.date(),
            'booking_time': base_time.replace(hour=23, minute=45).time(),  # Late night booking (high risk)
            'online_booking': 1,
            'mobile_site_booking': 0,
            'booking_channel': 'online',
            'is_round_trip': False,
            'vehicle_model_id': 89,  # Vehicle with highest cancellation rate (~40-50%)
            'travel_type_id': 2,  # Leisure travel
            'from_area_id': 1347,  # Remote area combination
            'to_area_id': 1192,
            'from_lat': 12.987,
            'from_long': 77.736,
            'to_lat': 12.977,
            'to_long': 77.573,
            'from_city_id': 1
        }
    else:  # random
        import random
        return {
            'booking_date': base_time.date(),
            'booking_time': base_time.replace(hour=random.randint(6, 23), minute=random.randint(0, 59)).time(),
            'online_booking': random.choice([0, 1]),
            'mobile_site_booking': random.choice([0, 1]),
            'booking_channel': random.choice(['online', 'mobile', 'phone', 'other']),
            'is_round_trip': random.choice([True, False]),
            'vehicle_model_id': random.randint(1, 91),  # CORRECTED: Full training range
            'travel_type_id': random.choice([1, 2, 3]),
            'from_area_id': random.randint(50, 1000),   # CORRECTED: Within training range
            'to_area_id': random.randint(100, 1200),   # CORRECTED: Within training range
            'from_lat': round(random.uniform(12.80, 13.20), 4),  # CORRECTED: Within training range
            'from_long': round(random.uniform(77.50, 77.75), 4), # CORRECTED: Within training range
            'to_lat': round(random.uniform(12.80, 13.20), 4),    # CORRECTED: Within training range
            'to_long': round(random.uniform(77.50, 77.75), 4),   # CORRECTED: Within training range
            'from_city_id': random.randint(1, 15)  # CORRECTED: Within training range
        }

def show_input_guide():
    """Display detailed input guide"""
    st.markdown("""
    ## 📖 **Input Field Guide**
    
    ### 📅 **Time Information**
    - **Booking Date**: When the booking was created
    - **Booking Time**: Specific time of booking creation
      - *Impact*: Late night/early morning bookings have higher cancellation risk
    
    ### 💻 **Booking Channel**
    - **Online Booking**: Made through website (0=No, 1=Yes)
    - **Mobile Booking**: Made through mobile site (0=No, 1=Yes)
    - **Booking Channel**: Primary channel used
      - *online*: Web portal
      - *mobile*: Mobile app/site
      - *phone*: Call center
      - *other*: Walk-in, partner sites
    
    ### 🚗 **Vehicle & Travel**
    - **Vehicle Model ID**: Specific vehicle type (1-50)
      - *Lower IDs*: Economy vehicles
      - *Higher IDs*: Premium vehicles
    - **Travel Type**: Purpose of travel
      - *1*: Business travel (lower risk)
      - *2*: Leisure travel (medium risk)  
      - *3*: Other purposes (higher risk)
    - **Round Trip**: Return journey included
      - *Yes*: Often more committed bookings
      - *No*: One-way trips
    
    ### 📍 **Location Data**
    - **From/To Area ID**: Location identifiers (1-100)
      - *Lower IDs*: Central/popular areas
      - *Higher IDs*: Remote/less common areas
    - **From/To City ID**: City identifiers (1-50)
    - **Coordinates**: Precise location (optional)
      - *Latitude*: North-South position
      - *Longitude*: East-West position
    
    ### 🎯 **Risk Factors**
    **High Risk Indicators:**
    - Late night bookings (10 PM - 6 AM)
    - Phone/other booking channels
    - Remote pickup/drop locations
    - Non-business travel
    - Weekend bookings
    
    **Low Risk Indicators:**
    - Daytime bookings (9 AM - 6 PM)
    - Online bookings
    - Central city locations
    - Business travel
    - Round trip bookings
    """)

def get_default_values():
    """Get default form values"""
    return {
        'booking_date': datetime.now().date(),
        'booking_time': datetime.now().time(),
        'online_booking': 1,
        'mobile_site_booking': 0,
        'booking_channel': 'online',
        'is_round_trip': True,
        'vehicle_model_id': 50,  # Changed to be within range 1-91
        'travel_type_id': 1,
        'from_area_id': 500,     # Changed from 5 to 500 to be within range 6-1391
        'to_area_id': 600,       # Changed from 12 to 600 to be within range 25-1390
        'from_lat': 12.9716,     # Within Bangalore training range (12.78-13.24)
        'from_long': 77.5946,    # Within Bangalore training range (77.47-77.79)
        'to_lat': 12.9500,       # Changed from 12.2958 to be within range (12.78-13.24)
        'to_long': 77.6100,      # Changed from 76.6394 to be within range (77.47-77.79)
        'from_city_id': 1
    }

# Header
st.markdown('<div class="main-header">🚗 YourCabs - Smart Cancellation Prediction</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Navigation")
    
    # Model loading section
    st.subheader("📊 Model Status")
    
    if not st.session_state.model_loaded:
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading ML model..."):
                try:
                    # Try to import and load model
                    from model_utils import ModelPredictor
                    st.session_state.predictor = ModelPredictor()
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to load model: {str(e)}")
                    
                    # Show troubleshooting info
                    st.info("""
                    **Troubleshooting:**
                    1. Ensure model files exist in models/ folder
                    2. Check if all dependencies are installed
                    3. Verify the working directory is correct
                    """)
    else:
        st.success("✅ Model Ready")
        
        # Show model info if available
        try:
            model_info = st.session_state.predictor.get_model_info()
            with st.expander("📋 Model Details"):
                for key, value in model_info.items():
                    st.write(f"**{key}:** {value}")
        except:
            st.warning("Model info not available")
    
    # Navigation
    st.subheader("📱 Application Modes")
    if st.session_state.model_loaded:
        app_mode = st.selectbox(
            "Choose Mode:",
            ["🔮 Single Prediction", "📊 Batch Analysis", "📈 Analytics Dashboard", "ℹ️ About"]
        )
    else:
        st.info("Load model first to access features")
        app_mode = "ℹ️ About"
    
    # Quick Actions Section
    if st.session_state.model_loaded and app_mode == "🔮 Single Prediction":
        st.subheader("⚡ Quick Actions")
        
        st.write("**🎲 Sample Data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🟢 Low Risk", help="Load sample data for low cancellation risk"):
                st.session_state.sample_data = get_sample_data("low_risk")
                st.session_state.load_sample = True
                st.rerun()
            
            if st.button("🟡 Medium Risk", help="Load sample data for medium cancellation risk"):
                st.session_state.sample_data = get_sample_data("medium_risk")
                st.session_state.load_sample = True
                st.rerun()
        
        with col2:
            if st.button("🔴 High Risk", help="Load sample data for high cancellation risk"):
                st.session_state.sample_data = get_sample_data("high_risk")
                st.session_state.load_sample = True
                st.rerun()
            
            if st.button("🎯 Random", help="Load random sample data"):
                st.session_state.sample_data = get_sample_data("random")
                st.session_state.load_sample = True
                st.rerun()
        
        st.write("**📋 Help:**")
        if st.button("📖 Input Guide", help="Show detailed input explanations"):
            st.session_state.show_input_guide = True
            st.rerun()
        
        if st.button("🔄 Clear Form", help="Reset all form fields"):
            st.session_state.clear_form = True
            st.session_state.load_sample = False
            st.session_state.sample_data = None
            st.rerun()

# Main content
if not st.session_state.model_loaded and app_mode != "ℹ️ About":
    st.warning("⚠️ Please load the model first using the sidebar.")
    st.info("""
    👋 **Welcome to YourCabs Cancellation Prediction System!**
    
    This application uses advanced machine learning to predict booking cancellation risk.
    
    **Features:**
    - 🎯 Real-time cancellation risk prediction
    - 📊 Batch processing for multiple bookings
    - 📈 Interactive analytics dashboard
    - 🎨 Beautiful and intuitive interface
    
    **Getting Started:**
    1. Click "Load Model" in the sidebar
    2. Choose your preferred mode
    3. Start predicting!
    """)

elif app_mode == "🔮 Single Prediction":
    st.header("🔮 Single Booking Prediction")
    
    # Show Input Guide if requested
    if st.session_state.show_input_guide:
        with st.expander("📖 Input Field Guide", expanded=True):
            show_input_guide()
        
        if st.button("✖️ Close Guide"):
            st.session_state.show_input_guide = False
            st.rerun()
        
        st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Booking Information")
        
        # Handle sample data loading and form clearing
        default_values = get_default_values()
        
        # Use session state to track current form values
        if 'current_form_values' not in st.session_state:
            st.session_state.current_form_values = default_values
            
        if st.session_state.load_sample and st.session_state.sample_data:
            st.session_state.current_form_values = st.session_state.sample_data
            st.session_state.load_sample = False  # Reset after loading
            # Show notification about loaded sample
            st.success(f"✅ Sample data loaded! Form values updated.")
        elif st.session_state.clear_form:
            st.session_state.current_form_values = default_values
            st.session_state.clear_form = False  # Reset after clearing
            st.info("🔄 Form cleared to default values.")
            
        form_values = st.session_state.current_form_values
        
        # Create input form with stable key
        with st.form("prediction_form", clear_on_submit=False):
            # Time information
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                booking_date = st.date_input("📅 Booking Date", value=form_values['booking_date'])
            with col_time2:
                booking_time = st.time_input("⏰ Booking Time", value=form_values['booking_time'])
            
            # Booking details
            col_booking1, col_booking2 = st.columns(2)
            with col_booking1:
                online_booking = st.selectbox("💻 Online Booking", [0, 1], 
                                            index=form_values['online_booking'],
                                            format_func=lambda x: "Yes" if x == 1 else "No")
                mobile_site_booking = st.selectbox("📱 Mobile Booking", [0, 1], 
                                                 index=form_values['mobile_site_booking'],
                                                 format_func=lambda x: "Yes" if x == 1 else "No")
            
            with col_booking2:
                booking_channel_options = ["online", "mobile", "phone", "other"]
                booking_channel = st.selectbox("📢 Booking Channel", booking_channel_options,
                                             index=booking_channel_options.index(form_values['booking_channel']))
                is_round_trip = st.selectbox("🔄 Round Trip", [False, True], 
                                           index=1 if form_values['is_round_trip'] else 0,
                                           format_func=lambda x: "Yes" if x else "No")
            
            # Vehicle and travel information
            col_vehicle1, col_vehicle2 = st.columns(2)
            with col_vehicle1:
                vehicle_model_id = st.number_input("🚗 Vehicle Model ID", min_value=1, max_value=91, 
                                                 value=form_values['vehicle_model_id'])
                travel_type_id = st.selectbox("✈️ Travel Type", [1, 2, 3], 
                                            index=form_values['travel_type_id']-1,
                                            format_func=lambda x: {"1": "Business", "2": "Leisure", "3": "Other"}[str(x)])
            
            with col_vehicle2:
                from_area_id = st.number_input("📍 From Area ID", min_value=6, max_value=1391, 
                                              value=form_values['from_area_id'])
                to_area_id = st.number_input("🎯 To Area ID", min_value=25, max_value=1390, 
                                           value=form_values['to_area_id'])
            
            # Geographic coordinates (optional)
            with st.expander("🗺️ Geographic Coordinates (Optional)"):
                col_geo1, col_geo2 = st.columns(2)
                with col_geo1:
                    from_lat = st.number_input("📍 From Latitude", min_value=12.78, max_value=13.24, 
                                             value=form_values['from_lat'], format="%.4f")
                    from_long = st.number_input("📍 From Longitude", min_value=77.47, max_value=77.79, 
                                              value=form_values['from_long'], format="%.4f")
                
                with col_geo2:
                    to_lat = st.number_input("🎯 To Latitude", min_value=12.78, max_value=13.24, 
                                           value=form_values['to_lat'], format="%.4f")
                    to_long = st.number_input("🎯 To Longitude", min_value=77.47, max_value=77.79, 
                                            value=form_values['to_long'], format="%.4f")
                
                from_city_id = st.number_input("🏙️ From City ID", min_value=1, max_value=15, 
                                              value=form_values['from_city_id'])
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Cancellation Risk", type="primary")
            
        # Debug: Show form submission status
        if submitted:
            st.write("🔍 Form submitted successfully!")
    
    with col2:
        st.subheader("📊 Prediction Result")
        
        if submitted:
            try:
                # Prepare booking data
                booking_datetime = datetime.combine(booking_date, booking_time)
                booking_data = {
                    'booking_created': booking_datetime,
                    'online_booking': online_booking,
                    'mobile_site_booking': mobile_site_booking,
                    'vehicle_model_id': vehicle_model_id,
                    'travel_type_id': travel_type_id,
                    'from_area_id': from_area_id,
                    'to_area_id': to_area_id,
                    'from_lat': from_lat,
                    'from_long': from_long,
                    'to_lat': to_lat,
                    'to_long': to_long,
                    'from_city_id': from_city_id,
                    'is_round_trip': is_round_trip,
                    'booking_channel': booking_channel
                }
                
                # DEBUG: Show what data is being sent to model
                with st.expander("🔍 Debug Info (Click to see prediction inputs)", expanded=False):
                    st.write("**Input Data Being Sent to Model:**")
                    st.json(booking_data, expanded=False)
                    st.write(f"**Key Identifiers:**")
                    st.write(f"- Vehicle Model ID: {vehicle_model_id}")
                    st.write(f"- Travel Type ID: {travel_type_id}")
                    st.write(f"- From Area ID: {from_area_id}")
                    st.write(f"- To Area ID: {to_area_id}")
                    st.write(f"- Booking Hour: {booking_datetime.hour}")
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    probability, risk_status, result = st.session_state.predictor.predict(booking_data)
                
                # Display prediction result
                risk_class = result['risk_category'].lower()
                if risk_class in ['critical', 'high']:
                    box_class = "danger-box"
                elif risk_class == 'medium':
                    box_class = "warning-box"
                else:
                    box_class = "success-box"
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h3>{risk_status} Risk</h3>
                    <p>Cancellation Probability: {probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                st.metric("🎯 Cancellation Probability", f"{probability:.1%}", delta=f"{probability - 0.5:.1%} vs average", delta_color="inverse")
                st.metric("🔍 Confidence Level", result['confidence'])
                
                # Recommendation
                st.subheader("💡 Recommendation")
                st.info(result['recommendation'])
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Cancellation Risk %"},
                    delta={'reference': 30},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 15], 'color': "lightgreen"},
                            {'range': [15, 30], 'color': "yellow"},
                            {'range': [30, 50], 'color': "orange"},
                            {'range': [50, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please check your input data and try again.")

elif app_mode == "📊 Batch Analysis":
    st.header("📊 Batch Booking Analysis")
    st.info("Upload a CSV file with booking data for batch prediction")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! {len(df):,} records found.")
            
            # Show preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head())
                st.write(f"**Shape:** {df.shape}")
            
            if st.button("🔮 Run Batch Predictions", type="primary"):
                with st.spinner("Processing predictions..."):
                    try:
                        results_df = st.session_state.predictor.batch_predict(df)
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📋 Total Bookings", f"{len(results_df):,}")
                        
                        with col2:
                            high_risk = (results_df['cancellation_probability'] >= 0.3).sum()
                            st.metric("🔴 High Risk", f"{high_risk:,}", delta=f"{high_risk/len(results_df):.1%}")
                        
                        with col3:
                            avg_prob = results_df['cancellation_probability'].mean()
                            st.metric("📈 Avg Risk", f"{avg_prob:.1%}")
                        
                        with col4:
                            critical_risk = (results_df['cancellation_probability'] >= 0.5).sum()
                            st.metric("🚨 Critical Risk", f"{critical_risk:,}", delta=f"{critical_risk/len(results_df):.1%}")
                        
                        # Risk distribution chart
                        risk_dist = results_df['risk_category'].value_counts()
                        fig_pie = px.pie(
                            values=risk_dist.values,
                            names=risk_dist.index,
                            title="Risk Distribution",
                            color_discrete_map={
                                'Low': '#28a745',
                                'Medium': '#ffc107',
                                'High': '#fd7e14',
                                'Critical': '#dc3545'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Results table
                        st.subheader("📋 Results")
                        st.dataframe(results_df[['cancellation_probability', 'risk_category', 'high_risk_prediction']].head(100))
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Batch prediction failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

elif app_mode == "📈 Analytics Dashboard":
    st.header("📈 Analytics Dashboard")
    st.info("Upload and analyze batch data first to see analytics.")

elif app_mode == "ℹ️ About":
    st.header("ℹ️ About YourCabs Prediction System")
    
    st.markdown("""
    ## 🚗 **YourCabs Cancellation Prediction**
    
    This application uses advanced machine learning to predict booking cancellation risk in real-time.
    
    ### 🎯 **Key Features:**
    
    - **🔮 Single Prediction**: Real-time risk assessment for individual bookings
    - **📊 Batch Analysis**: Process multiple bookings simultaneously  
    - **📈 Analytics Dashboard**: Interactive visualizations and insights
    - **🎨 Beautiful UI**: Modern, responsive design
    
    ### 🧠 **Machine Learning Model:**
    
    - **Algorithm**: XGBoost with hyperparameter optimization
    - **Performance**: 85%+ AUC on test data
    - **Features**: 20+ engineered features
    - **Class Handling**: Scale_pos_weight (no SMOTE needed)
    
    ### 📊 **Risk Categories:**
    
    | Risk Level | Probability | Action |
    |------------|-------------|---------|
    | 🟢 Low | 0% - 15% | Standard monitoring |
    | 🟡 Medium | 15% - 30% | Send reminders |
    | 🟠 High | 30% - 50% | Proactive contact |
    | 🔴 Critical | 50%+ | Immediate intervention |
    
    ### 🔧 **Technical Requirements:**
    
    - Python 3.8+
    - Streamlit, XGBoost, Pandas, Plotly
    - Model files in models/ directory
    - Proper project structure
    
    ### 📞 **Support:**
    
    For technical support or feature requests, please refer to the project documentation.
    
    ---
    
    *Built with ❤️ using Streamlit, XGBoost, and Plotly*
    
    **Version:** 2.0 | **Last Updated:** July 2025
    """)

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666; padding: 20px;'>
🚗 <strong>YourCabs Prediction System</strong> | Built with Streamlit & XGBoost
</div>
""", unsafe_allow_html=True)
