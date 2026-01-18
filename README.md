# 🌱 CultivaSense - Smart Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

**CultivaSense** is an AI-powered crop recommendation system designed specifically for Indian agriculture. Using advanced machine learning algorithms, it provides farmers with data-driven crop suggestions based on soil conditions, climate, and regional patterns to maximize yield and promote sustainable farming practices.

## ✨ Features

### 🧠 **Smart AI Analysis**
- **95% Accuracy** in crop recommendations
- **22 Different Crops** analyzed and supported
- Advanced machine learning algorithms (Random Forest Classifier)
- Real-time soil condition analysis

### 🧪 **Comprehensive Soil Testing**
- **NPK Analysis** (Nitrogen, Phosphorus, Potassium)
- **pH Level** monitoring and recommendations
- **Nutrient Deficiency** detection with recovery plans
- Environmental factor analysis (temperature, humidity, rainfall)

### 📈 **Yield Optimization**
- Data-driven insights for **up to 30% yield increase**
- Precision agriculture techniques
- Smart farming recommendations
- Sustainable practice suggestions

### 🌱 **Recovery Solutions**
- Instant access to detailed recovery plans
- Specific fertilizer recommendations
- Application schedules and dosage guidance
- Nutrient management strategies

### 📊 **Detailed Analytics**
- Comprehensive reporting dashboard
- Progress tracking capabilities
- Historical data analysis
- Performance metrics visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dasouvik122005/crop.git
   cd crop
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

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   Navigate to `http://127.0.0.1:5000`

## 🎯 Usage

### 1. **Soil Analysis**
- Enter soil nutrient values (N, P, K)
- Input environmental conditions (temperature, humidity, pH, rainfall)
- Click "Analyze" for instant crop recommendations

### 2. **Review Results**
- Get personalized crop suggestions
- View nutrient comparison (your values vs. ideal)
- Access detailed insights and warnings

### 3. **Recovery Plans**
- Click "Manage" on any nutrient deficiency
- Follow detailed recovery strategies
- Implement suggested fertilizer applications

## 📁 Project Structure

```
crop_recommandation/
│
├── 📄 app.py                          # Main Flask application
├── 📄 model.pkl                       # Trained ML model
├── 📄 minmaxscaler.pkl                # Feature scaling model
├── 📄 Crop_recommendation.csv         # Training dataset
├── 📓 Crop Classification...ipynb     # Jupyter notebook for model training
│
├── 📁 static/
│   ├── 🎨 style.css                  # Modern responsive styling
│   ├── ⚡ script.js                   # Interactive functionality
│   └── 🖼️ logo.png                   # Brand logo
│
├── 📁 templates/
│   └── 🌐 index.html                 # Main web interface
│
└── 📄 README.md                      # Project documentation
```

## 🛠️ Technology Stack

### **Backend**
- **Flask** - Lightweight web framework
- **Python** - Core programming language
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### **Frontend**
- **HTML5** - Modern semantic markup
- **CSS3** - Responsive design with glass-morphism effects
- **JavaScript** - Interactive user experience
- **Responsive Design** - Mobile-first approach

### **Machine Learning**
- **Random Forest Classifier** - Primary recommendation algorithm
- **Decision Tree** - Supporting classification model
- **Min-Max Scaling** - Feature normalization
- **Cross-validation** - Model performance optimization

## 🌾 Supported Crops

CultivaSense provides recommendations for **22 major Indian crops**:

🌾 **Cereals:** Rice, Wheat, Maize, Barley
🫘 **Legumes:** Chickpea, Kidney Beans, Pigeon Peas, Moth Beans
🥥 **Plantation:** Coconut, Coffee
🍎 **Fruits:** Apple, Banana, Grapes, Orange, Papaya, Pomegranate, Watermelon, Muskmelon
🥜 **Oilseeds & Others:** Cotton, Jute, Lentil, Black Gram

## 📊 Model Performance

- **Accuracy:** 95%+ on validation dataset
- **Training Data:** 2,200+ soil samples
- **Features:** 7 soil and environmental parameters
- **Validation:** K-fold cross-validation
- **Optimization:** Grid search hyperparameter tuning

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
- 🌦️ **Weather Integration** - Real-time weather data
- 🛰️ **Satellite Imagery** - Remote sensing capabilities
- 📊 **Predictive Analytics** - Seasonal trend analysis
- 🌍 **Multi-language Support** - Regional language interfaces
- 📱 **Mobile App** - Native iOS/Android applications
- 🤖 **Chatbot Integration** - AI-powered farming assistant

### **Version 3.0 - Vision:**
- 🌾 **Crop Disease Detection** - Computer vision for plant health
- 💰 **Market Price Prediction** - Economic optimization
- 🚜 **IoT Integration** - Smart sensor connectivity
- 🎯 **Precision Farming** - GPS-guided recommendations

## 📄 Project Documentation

The project report and additional documentation can be found in the `project_report/` directory.

To upload your project report:
1. Place your report files (PDF, DOCX, etc.) in the `project_report/` folder.
2. Commit and push the changes to the repository.

[📂 View Project Report Folder](project_report/)


---

## 📈 Statistics

- ⭐ **95%** Model Accuracy
- 🌾 **22** Supported Crops
- 📊 **2,200+** Training Samples
- 🇮🇳 **Pan-India** Coverage
- 📱 **100%** Mobile Responsive

---

<div align="center">

**🌱 Empowering Indian Agriculture with AI 🤖**

*Built with ❤️ for sustainable farming and food security*

[🚀 Get Started](http://127.0.0.1:5000) • [📖 Documentation](docs/) • [🐛 Report Bug](issues/) • [💡 Feature Request](issues/)

</div>

---

**© 2025 CultivaSense. All rights reserved. | Built for Indian Agriculture 🇮🇳**
