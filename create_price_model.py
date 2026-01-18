"""
Create market price prediction model
This creates a trained model for price prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("🔨 Creating Market Price Prediction Model...")

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Check if market_price_data.csv exists, otherwise generate synthetic data
if os.path.exists('market_price_data.csv'):
    print("📊 Loading existing market price data...")
    df = pd.read_csv('market_price_data.csv')
else:
    print("📊 Generating synthetic market price data...")
    # Generate synthetic training data
    np.random.seed(42)
    
    n_samples = 1000
    crop_ids = np.random.randint(1, 9, n_samples)
    state_ids = np.random.randint(1, 14, n_samples)
    months = np.random.randint(1, 13, n_samples)
    rainfall = np.random.uniform(0, 400, n_samples)
    temperature = np.random.uniform(10, 45, n_samples)
    
    # Generate prices based on features
    base_prices = [2500, 2800, 1800, 4500, 3200, 3500, 2000, 3800]
    prices = []
    for crop_id, temp, rain in zip(crop_ids, temperature, rainfall):
        base = base_prices[crop_id - 1]
        temp_factor = 1.0 + (temp - 30) / 100
        rain_factor = 1.0 - (rain - 200) / 2000
        price = base * temp_factor * rain_factor
        price = max(500, min(10000, price))  # Clamp between 500 and 10000
        prices.append(price)
    
    df = pd.DataFrame({
        'crop_id': crop_ids,
        'state_id': state_ids,
        'month': months,
        'rainfall': rainfall,
        'temperature': temperature,
        'price': prices
    })

# Prepare features and target
# Prepare features and target
if 'crop_id' in df.columns:
    # Use lowercase columns if available (synthetic data)
    X = df[['crop_id', 'state_id', 'month', 'rainfall', 'temperature']]
    y = df['price']
else:
    # Use CSV columns (real data)
    X = df[['Crop_ID', 'State_ID', 'Month', 'Rainfall', 'Temperature']]
    y = df['Price_Per_Quintal']

print(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest model
print("🔄 Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_scaled, y)

print(f"✅ Model trained successfully!")
print(f"   - Number of features: {model.n_features_in_}")
print(f"   - Number of estimators: {model.n_estimators}")
print(f"   - Training R² score: {model.score(X_scaled, y):.4f}")

# Save the model
model_path = 'models/market_price_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\n✅ Price prediction model saved to: {model_path}")

# Save the scaler
scaler_path = 'models/price_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Feature scaler saved to: {scaler_path}")

# Save feature names
import json
feature_info = {
    'features': ['crop_id', 'state_id', 'month', 'rainfall', 'temperature'],
    'model_type': 'RandomForestRegressor',
    'n_estimators': model.n_estimators,
    'feature_importances': model.feature_importances_.tolist()
}
with open('models/price_model_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"✅ Model info saved to: models/price_model_info.json")

print("\n✅ All model files created successfully!")
