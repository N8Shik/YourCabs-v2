# 🚗 YourCabs v2.0 - Smart Cancellation Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A revolutionary upgrade** to the YourCabs cancellation prediction system with **enhanced UI/UX**, **advanced features**, and **production-ready architecture**.

## 🎯 **What's New in v2.0?**

### ⭐ **Major UI/UX Improvements**
- **🎨 Modern Design Language**: Complete visual overhaul with gradient backgrounds, custom CSS styling, and professional color schemes
- **📱 Responsive Layout**: Enhanced mobile-friendly design with improved column layouts and spacing
- **🎛️ Smart Sidebar Navigation**: Organized model status, quick actions, and help sections in an intuitive sidebar
- **⚡ Quick Action Buttons**: One-click sample data loading for Low/Medium/High/Random risk scenarios
- **🔍 Interactive Debug Panel**: Expandable debug information showing exact model inputs for transparency
- **📊 Enhanced Result Display**: Beautiful prediction boxes with color-coded risk levels and visual gauges

### 🚀 **Advanced Features**
- **🧠 Intelligent Form State Management**: Persistent form values using session state for seamless user experience
- **📖 Comprehensive Input Guide**: Built-in help system with detailed field explanations and risk factor analysis
- **🎲 Smart Sample Data**: Realistic sample data based on actual training data patterns and ranges
- **🔄 One-Click Form Reset**: Easy form clearing and restoration to default values
- **📈 Enhanced Analytics**: Interactive Plotly visualizations with risk distribution charts
- **🎯 Improved Predictions**: Recalibrated risk thresholds for more accurate categorization

### 🛠️ **Technical Enhancements**
- **🧹 Clean Architecture**: Modular codebase with separated concerns and utility functions
- **⚡ Performance Optimized**: Faster loading times and improved model inference
- **🔒 Production Ready**: Error handling, validation, and robust state management
- **📊 Enhanced Model Utils**: Better prediction confidence and recommendation system
- **🎨 Custom CSS Framework**: Professional styling with consistent design patterns

## 📊 **v1 vs v2 Comparison**

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **UI Design** | Basic Streamlit components | 🎨 Custom CSS with gradients & modern styling |
| **Navigation** | Single page layout | 🎛️ Organized sidebar with multiple modes |
| **Form Management** | Static forms, manual entry | ⚡ Smart state management + quick actions |
| **Sample Data** | Manual data entry only | 🎲 One-click risk scenario loading |
| **Help System** | No guidance | 📖 Comprehensive input guide & tooltips |
| **Debug Info** | Hidden/minimal | 🔍 Interactive debug panel with transparency |
| **Result Display** | Basic text output | 📊 Color-coded boxes + interactive gauges |
| **Risk Categories** | Generic thresholds | 🎯 Calibrated thresholds based on real data |
| **Error Handling** | Basic try-catch | 🛠️ Comprehensive error management |
| **Performance** | Standard loading | ⚡ Optimized with caching & session state |

## 🎨 **UI/UX Showcase**

### 🌟 **Before vs After**

#### **v1.0 Interface**
```
❌ Basic Streamlit layout
❌ Generic form fields
❌ Minimal visual feedback
❌ No quick actions
❌ Limited help system
❌ Basic result display
```

#### **v2.0 Interface**
```
✅ Professional gradient header with custom styling
✅ Organized sidebar with model status & quick actions
✅ Smart form with persistent state management
✅ Color-coded prediction boxes with visual indicators
✅ Interactive debug panel for transparency
✅ Comprehensive help system with field guides
✅ One-click sample data loading (Low/Medium/High/Random)
✅ Enhanced analytics with Plotly visualizations
✅ Responsive design for all screen sizes
```

### 🎯 **Key UI Improvements**

1. **🎨 Visual Design**
   - Custom CSS with professional color schemes
   - Gradient backgrounds and modern typography
   - Consistent spacing and visual hierarchy

2. **📱 User Experience**
   - Intuitive navigation with organized sidebar
   - Quick action buttons for common tasks
   - Smart form state persistence

3. **🔍 Transparency**
   - Debug panel showing exact model inputs
   - Clear risk categorization with visual indicators
   - Comprehensive help and guidance system

## 🚀 **Quick Start**

### **1️⃣ Clone & Setup**
```bash
git clone https://github.com/N8Shik/YourCabs-v2.git
cd YourCabs-v2
pip install -r requirements.txt
```

### **2️⃣ Launch Application**
```bash
# Option 1: Windows Batch File
start_app.bat

# Option 2: Direct Command
streamlit run src/app_simple.py
```

### **3️⃣ Start Predicting**
1. Click "🔄 Load Model" in sidebar
2. Use "⚡ Quick Actions" for sample data
3. Or enter custom booking details
4. Get instant predictions with visual insights!

## 📁 **Project Architecture**

```
YourCabs-v2/
├── 🎨 src/                      # Enhanced source code
│   ├── app_simple.py           # Main Streamlit app with v2 features
│   ├── model_utils.py          # Enhanced model utilities
│   ├── data_processor.py       # Advanced data processing
│   └── config.py               # Configuration management
├── 📊 models/                  # ML model artifacts
│   ├── best_model.joblib       # Trained XGBoost model
│   └── model_info.json         # Model metadata
├── 📈 data/                    # Clean datasets
│   └── YourCabs_cleaned.csv    # Production-ready data
├── 📚 notebooks/               # Analysis & training
│   ├── eda.ipynb              # Data exploration
│   └── model_training_clean.ipynb  # Model development
├── 🚀 start_app.bat            # One-click startup
├── 📋 requirements.txt         # Dependencies
└── 📖 README.md               # This file
```

## 🎯 **Risk Categories & Thresholds**

| Risk Level | Probability Range | Visual Indicator | Action Required |
|------------|------------------|------------------|-----------------|
| 🟢 **Very Low** | 0% - 5% | Green success box | Standard monitoring |
| 🟢 **Low** | 5% - 15% | Light green | Regular follow-up |
| 🟡 **Medium** | 15% - 30% | Yellow warning box | Send reminders |
| 🟠 **High** | 30% - 50% | Orange danger box | Proactive contact |
| 🔴 **Critical** | 50%+ | Red critical box | Immediate intervention |

## 🧠 **Machine Learning Model**

- **🎯 Algorithm**: XGBoost with optimized hyperparameters
- **📈 Performance**: 85%+ AUC on validation data
- **🔧 Features**: 20+ engineered features from booking patterns
- **⚖️ Class Handling**: Balanced with scale_pos_weight optimization
- **🎨 Prediction**: Real-time inference with confidence scoring

## 🛠️ **Technical Stack**

### **Frontend**
- **🎨 Streamlit**: Modern web interface with custom CSS
- **📊 Plotly**: Interactive visualizations and charts
- **💫 Custom CSS**: Professional styling and animations

### **Backend**
- **🧠 XGBoost**: High-performance gradient boosting
- **🐼 Pandas**: Data manipulation and processing
- **🔢 NumPy**: Numerical computations
- **⚡ Joblib**: Model serialization and loading

### **Infrastructure**
- **🐍 Python 3.8+**: Core programming language
- **📦 pip**: Package management
- **🔧 Session State**: Advanced state management

## 📊 **Sample Data & Testing**

### **🎲 Quick Action Scenarios**

| Scenario | Description | Expected Result |
|----------|-------------|----------------|
| 🟢 **Low Risk** | Business travel, afternoon booking, central areas | ~2-5% cancellation |
| 🟡 **Medium Risk** | Leisure travel, evening booking, mixed areas | ~15-25% cancellation |
| 🔴 **High Risk** | Late night booking, remote areas, premium vehicle | ~40-50% cancellation |
| 🎯 **Random** | Randomized realistic parameters | Variable result |

## 🎨 **Customization**

### **🎨 Styling**
- Modify CSS in `app_simple.py` for custom themes
- Adjust color schemes in prediction boxes
- Customize sidebar layout and navigation

### **🎯 Model Configuration**
- Update risk thresholds in `model_utils.py`
- Modify feature engineering in `data_processor.py`
- Adjust sample data scenarios

## 🔧 **Troubleshooting**

### **Common Issues**
1. **Model Loading Error**: Ensure `models/best_model.joblib` exists
2. **Import Error**: Run `pip install -r requirements.txt`
3. **Port Conflict**: Use `streamlit run src/app_simple.py --server.port 8502`

### **🐛 Debug Mode**
- Use the "🔍 Debug Info" panel to see exact model inputs
- Check browser console for JavaScript errors
- Verify Python environment and dependencies

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **v1.0 Foundation**: Built upon the solid foundation of YourCabs v1
- **Streamlit Community**: For excellent documentation and examples
- **XGBoost Team**: For the powerful machine learning framework
- **Open Source Community**: For inspiration and best practices

---

<div align="center">

### 🚗 **YourCabs v2.0 - Redefining Cancellation Prediction**

**Built with ❤️ using Streamlit, XGBoost, and Modern Web Technologies**

[🌟 Star this repo](https://github.com/N8Shik/YourCabs-v2) • [🐛 Report Bug](https://github.com/N8Shik/YourCabs-v2/issues) • [💡 Request Feature](https://github.com/N8Shik/YourCabs-v2/issues)

</div>
