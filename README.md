# 🌱 CultivaSense - Smart Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![AI Powered](https://img.shields.io/badge/AI-Powered-brightgreen.svg)](https://github.com/SoahamBehera/Cultiva-Sense)

**CultivaSense** is a comprehensive AI-powered agricultural intelligence platform designed specifically for Indian agriculture. Using advanced machine learning algorithms and computer vision, it provides farmers with data-driven insights including crop recommendations, disease detection, and market price predictions to maximize yield and promote sustainable farming practices.

## ✨ Features

### 🧠 **Smart Crop Recommendation**
- **95% Accuracy** in crop recommendations using Random Forest Classifier
- **22 Different Crops** analyzed and supported (Rice, Wheat, Maize, Cotton, and more)
- Real-time soil condition analysis with 7 environmental parameters
- Personalized suggestions based on NPK levels, pH, temperature, humidity, and rainfall
- Instant comparison between your soil values and ideal conditions

### 🔬 **Crop Disease Detection**
- **Computer Vision-based** disease identification from crop images
- Supports **32 disease classes** across major crops (Apple, Corn, Grape, Potato, Rice, Tomato, Wheat)
- **Instant diagnosis** with confidence scores
- Detailed **treatment recommendations** for each detected disease
- Image upload with preview (supports JPG, PNG up to 5MB)
- Covers diseases like:
  - Rice: Brown Spot, Leaf Blast, Neck Blast
  - Wheat: Brown Rust, Yellow Rust
  - Tomato: Early Blight, Late Blight, Leaf Mold, Bacterial Spot, and more
  - Potato, Corn, Apple, Grape diseases

### 💰 **Market Price Prediction**
- **AI-powered price forecasting** using Random Forest Regressor
- Predictions for **8 major crop categories** (Rice, Wheat, Maize, Cotton, Sugarcane, Pulses, Vegetables, Fruits)
- **13 Indian states** coverage (Maharashtra, Punjab, Haryana, UP, Bihar, and more)
- **Seasonal analysis** with month-wise predictions
- Weather-based market trend insights
- Factors considered: Crop type, State, Month, Rainfall, Temperature

### 🧪 **Comprehensive Soil Testing**
- **NPK Analysis** (Nitrogen, Phosphorus, Potassium)
- **pH Level** monitoring and recommendations
- **Nutrient Deficiency** detection with interactive recovery plans
- Environmental factor analysis (temperature, humidity, rainfall)
- Visual comparison charts (Your values vs Ideal values)

### 🌱 **Interactive Recovery Solutions**
- Instant access to **detailed recovery plans** for each nutrient
- Specific **fertilizer recommendations** with dosages
- **Application schedules** and timing guidance
- Separate plans for deficiency and excess conditions
- Covers all parameters: N, P, K, pH, Temperature, Humidity, Rainfall

### 📊 **Modern User Interface**
- **Mobile-responsive** design with glassmorphism effects
- **Dark mode** optimized interface
- **Interactive animations** and smooth transitions
- **Real-time validation** with helpful error messages
- **Sample data** loading for quick testing
- **Accessibility features** with ARIA labels

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Modern web browser
- (Optional) TensorFlow for disease detection model

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SoahamBehera/Cultiva-Sense.git
   cd Cultiva-Sense
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (Optional)**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   Navigate to `http://127.0.0.1:5000`

## 🎯 Usage

### 1. **Crop Recommendation**
- Navigate to the "Soil Analyzer" section
- Enter soil nutrient values (N, P, K)
- Input environmental conditions (temperature, humidity, pH, rainfall)
- Click "Analyze Soil" for instant crop recommendations
- View nutrient comparison and recovery plans

### 2. **Disease Detection**
- Go to the "Crop Disease Detection" section
- Upload a clear image of the affected crop (JPG/PNG, max 5MB)
- Click "Detect Disease" for instant diagnosis
- View disease name, confidence score, and treatment recommendations

### 3. **Market Price Prediction**
- Navigate to "Market Price Prediction" section
- Select crop type and state
- Choose month and enter weather conditions
- Click "Predict Price" for market forecasts
- View predicted price and market trend analysis

## 📁 Project Structure

```
Cultiva-Sense/
│
├── 📄 app.py                          # Main Flask application with all routes
├── 📄 config.py                       # Configuration management
├── 📄 setup.py                        # Project setup script
│
├── 📊 Models & Data
│   ├── model.pkl                      # Trained crop recommendation model
│   ├── minmaxscaler.pkl              # Feature scaling model
│   ├── Crop_recommendation.csv       # Training dataset (2200+ samples)
│   ├── crop_disease_data.csv         # Disease detection database
│   ├── market_price_data.csv         # Market price historical data
│   └── models/
│       ├── plant_disease_model.h5    # CNN model for disease detection (optional)
│       ├── market_price_model.pkl    # Price prediction model
│       ├── price_scaler.pkl          # Price feature scaler
│       ├── disease_classes.json      # Disease class mappings
│       └── price_model_info.json     # Price model metadata
│
├── 📁 static/
│   ├── 🎨 style.css                  # Modern responsive styling (70KB+)
│   ├── ⚡ script.js                   # Interactive functionality (42KB+)
│   └── 🖼️ logo.png                   # Brand logo
│
├── 📁 templates/
│   └── 🌐 index.html                 # Main web interface (all features)
│
├── 📁 uploads/                        # Temporary storage for uploaded images
│
├── 🔧 Configuration
│   ├── .env.example                  # Environment variables template
│   ├── .gitignore                    # Git ignore rules
│   └── requirements.txt              # Python dependencies
│
├── �️ Utilities
│   ├── generate_disease_data.py      # Generate disease dataset
│   ├── generate_market_price_data.py # Generate price dataset
│   └── batch_predict_prices.py       # Batch price predictions
│
├── 📓 Crop Classification...ipynb     # Jupyter notebook for model training
└── �📄 README.md                       # Project documentation
```

## 🛠️ Technology Stack

### **Backend**
- **Flask 2.3.3** - Lightweight web framework
- **Python 3.8+** - Core programming language
- **Scikit-learn 1.3.0** - Machine learning library
- **Pandas 2.0.3** - Data manipulation and analysis
- **NumPy 1.24.3** - Numerical computing
- **Pillow** - Image processing for disease detection
- **h5py** - HDF5 file handling
- **Gunicorn** - Production WSGI server

### **Frontend**
- **HTML5** - Modern semantic markup
- **CSS3** - Responsive design with glassmorphism effects
- **JavaScript (Vanilla)** - Interactive user experience
- **Responsive Design** - Mobile-first approach
- **Accessibility** - ARIA labels and semantic HTML

### **Machine Learning**
- **Random Forest Classifier** - Crop recommendation (95% accuracy)
- **Random Forest Regressor** - Market price prediction
- **Computer Vision** - Image-based disease detection
- **Min-Max Scaling** - Feature normalization
- **Cross-validation** - Model performance optimization

### **Optional**
- **TensorFlow/Keras** - Deep learning for advanced disease detection
- **CNN (Convolutional Neural Network)** - Image classification

## 🌾 Supported Crops

CultivaSense provides recommendations for **22 major Indian crops**:

🌾 **Cereals:** Rice, Wheat, Maize, Barley  
🫘 **Legumes:** Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil  
🥥 **Plantation:** Coconut, Coffee  
🍎 **Fruits:** Apple, Banana, Grapes, Orange, Papaya, Pomegranate, Watermelon, Muskmelon  
🌱 **Cash Crops:** Cotton, Jute

## 📊 Model Performance

### **Crop Recommendation System**
- **Accuracy:** 95%+ on validation dataset
- **Training Data:** 2,200+ soil samples
- **Features:** 7 soil and environmental parameters (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Algorithm:** Random Forest Classifier
- **Validation:** K-fold cross-validation
- **Optimization:** Grid search hyperparameter tuning

### **Disease Detection System**
- **Disease Classes:** 32 different diseases
- **Supported Crops:** 7 major crops (Apple, Corn, Grape, Potato, Rice, Tomato, Wheat)
- **Detection Method:** Computer Vision + CSV-based matching
- **Image Support:** JPG, PNG (up to 5MB)
- **Features:** Color analysis, affected area detection
- **Treatment Database:** Comprehensive recommendations for each disease

### **Market Price Prediction**
- **Crop Categories:** 8 major categories
- **Geographic Coverage:** 13 Indian states
- **Algorithm:** Random Forest Regressor
- **Features:** Crop ID, State ID, Month, Rainfall, Temperature
- **Feature Importance:** Crop type (78.8%), Temperature (7.3%), Month (5.2%), Rainfall (5.3%), State (3.4%)
- **Seasonal Analysis:** Month-wise trend predictions
- **Market Insights:** Weather-based trend analysis

## 🌍 Environmental Impact

### **Sustainable Agriculture Goals:**
- 🌿 Promote eco-friendly farming practices
- 💧 Optimize water usage through smart recommendations
- 🌱 Reduce chemical fertilizer dependency
- 📈 Increase crop yield efficiency
- 🌾 Support biodiversity in agricultural systems

## 📱 Mobile Responsiveness

CultivaSense is designed with a **mobile-first approach**:

- ✅ **Fully Responsive** - Works on all screen sizes
- ✅ **Touch-Friendly** - Optimized for mobile interactions
- ✅ **Fast Loading** - Optimized performance
- ✅ **Progressive Web App** - App-like experience

## 🤝 Contributing

We welcome contributions to make CultivaSense even better!

### **How to Contribute:**

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

### **Areas for Contribution:**
- 🌾 Additional crop support
- 🧠 Model accuracy improvements
- 🎨 UI/UX enhancements
- 🌍 Localization support
- 📱 Mobile app development

## 👨‍💻 Developer

**Souvik Das**
- 🌐 Portfolio: [souvik-das.dev](https://souvik-das.dev)
- 📧 Email: souvik.das@agriculture.ai
- 💼 LinkedIn: [dasouvik122005](https://linkedin.com/in/dasouvik122005)
- 🐙 GitHub: [@dasouvik122005](https://github.com/dasouvik122005)

## 🙏 Acknowledgments

- **Indian Agricultural Research Institute** - Dataset and domain expertise
- **Scikit-learn Community** - Machine learning framework
- **Flask Team** - Web framework development
- **Agricultural Experts** - Domain knowledge and validation

## 🔮 Future Roadmap

### **Version 2.0 - Planned Features:**
- 🌦️ **Weather API Integration** - Real-time weather data from IMD (India Meteorological Department)
- 🛰️ **Satellite Imagery** - Remote sensing capabilities for field analysis
- 📊 **Advanced Analytics Dashboard** - Historical tracking and trend analysis
- 🌍 **Multi-language Support** - Hindi, Tamil, Telugu, Bengali, and other regional languages
- 📱 **Progressive Web App** - Offline capability and app-like experience
- 🤖 **AI Chatbot** - Voice-enabled farming assistant in local languages
- 💾 **User Accounts** - Save predictions and track farming history
- 📧 **Email Reports** - Automated PDF reports via email

### **Version 3.0 - Vision:**
- 🌾 **Advanced Disease Detection** - Deep learning CNN model with 98%+ accuracy
- 🎯 **Precision Farming** - GPS-guided field-specific recommendations
- � **IoT Integration** - Smart sensor connectivity for real-time monitoring
- 📈 **Yield Prediction** - AI-powered harvest forecasting
- 💰 **Financial Planning** - ROI calculator and crop profitability analysis
- 🌐 **Community Platform** - Farmer forums and knowledge sharing
- 🏪 **Marketplace Integration** - Direct buyer-seller connections
- 📱 **Native Mobile Apps** - iOS and Android applications

### **✅ Recently Implemented (Current Version):**
- ✅ Crop Disease Detection with 32 disease classes
- ✅ Market Price Prediction for 8 crop categories across 13 states
- ✅ Interactive Recovery Plans for all soil parameters
- ✅ Mobile-responsive modern UI with glassmorphism design
- ✅ Computer Vision-based disease analysis
- ✅ Seasonal market trend analysis

## 📄 Project Documentation

The project report and additional documentation can be found in the `project_report/` directory.

To upload your project report:
1. Place your report files (PDF, DOCX, etc.) in the `project_report/` folder.
2. Commit and push the changes to the repository.

[📂 View Project Report Folder](project_report/)


---

## 📈 Statistics

- ⭐ **95%** Model Accuracy (Crop Recommendation)
- 🌾 **22** Supported Crops
- � **32** Disease Classes Detected
- 💰 **8** Crop Categories for Price Prediction
- 🗺️ **13** Indian States Coverage
- �📊 **2,200+** Training Samples
- 🇮🇳 **Pan-India** Coverage
- 📱 **100%** Mobile Responsive

---

<div align="center">

**🌱 Empowering Indian Agriculture with AI 🤖**

*Built with ❤️ for sustainable farming and food security*

**Three Powerful Features:**  
🧠 Crop Recommendation | 🔬 Disease Detection | � Price Prediction

---

**📧 Contact:** cultivasense_test@gmail.com  
**🌐 Website:** www.cultivasense.com

---

**© 2025 CultivaSense. All rights reserved. | Built for Indian Agriculture 🇮🇳**

</div>
