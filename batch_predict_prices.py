import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

# Mappings must match app.py
CROP_TO_ID = {
    'Rice':1, 'Wheat':2, 'Maize':3, 'Cotton':4,
    'Sugarcane':5, 'Pulses':6, 'Vegetables':7, 'Fruits':8
}

STATE_TO_ID = {
    'Andhra Pradesh':1, 'Karnataka':2, 'Kerala':3, 'Tamil Nadu':4,
    'Maharashtra':5, 'Gujarat':6, 'Rajasthan':7, 'Madhya Pradesh':8,
    'Uttar Pradesh':9, 'Bihar':10, 'West Bengal':11, 'Punjab':12, 'Haryana':13
}

# Load model and scaler
MODEL_PATH = 'models/market_price_model.pkl'
SCALER_PATH = 'models/price_scaler.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Price model not found at {MODEL_PATH}. Run create_price_model.py first.")

with open(MODEL_PATH, 'rb') as f:
    price_model = pickle.load(f)

price_scaler = None
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as f:
        price_scaler = pickle.load(f)

# Simple market trend helper (same logic as app)
def get_market_trend(crop_id, month, rainfall, temperature):
    trends = []
    if month in [6,7,8,9]:
        if rainfall > 200:
            trends.append("High supply expected due to good rainfall")
        else:
            trends.append("Prices may rise due to water scarcity")
    if month in [10,11,12,1]:
        trends.append("Prices typically lower during harvest season")
    if month in [3,4,5] and temperature > 35:
        trends.append("Heat stress may reduce supply and increase prices")

    crop_name = {v:k for k,v in CROP_TO_ID.items()}.get(crop_id, '')
    if crop_name in ['Rice', 'Wheat'] and month in [10,11,12]:
        trends.append("Peak harvest season - expect competitive pricing")
    elif crop_name == 'Cotton' and month in [10,11,12,1]:
        trends.append("Cotton harvest season - market prices stabilizing")

    return " | ".join(trends) if trends else "Stable market conditions expected"

# Read input CSV
INPUT_CSV = 'input_market_requests.csv'
OUT_CSV = 'predicted_prices.csv'

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

# Map crop/state to IDs
missing = []

def map_crop(x):
    if x in CROP_TO_ID:
        return CROP_TO_ID[x]
    else:
        missing.append(("crop", x))
        return np.nan

def map_state(x):
    if x in STATE_TO_ID:
        return STATE_TO_ID[x]
    else:
        missing.append(("state", x))
        return np.nan

df['crop_id'] = df['crop_type'].map(map_crop)
df['state_id'] = df['state'].map(map_state)

if missing:
    raise ValueError(f"Unknown crop or state names encountered: {missing}")

# Validate month, rainfall, temperature
for col in ['month','rainfall','temperature']:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Prepare features
features = df[['crop_id','state_id','month','rainfall','temperature']].astype(float).values

if price_scaler is not None:
    features_scaled = price_scaler.transform(features)
    preds = price_model.predict(features_scaled)
else:
    preds = price_model.predict(features)

# Ensure non-negative and round
preds = np.maximum(0, preds)

# Add predictions and market trend
df['predicted_price'] = np.round(preds, 2)

def crop_id_to_name(cid):
    inv = {v:k for k,v in CROP_TO_ID.items()}
    return inv.get(int(cid), 'Unknown')

def state_id_to_name(sid):
    inv = {v:k for k,v in STATE_TO_ID.items()}
    return inv.get(int(sid), 'Unknown')

rows = []
for idx, row in df.iterrows():
    market_trend = get_market_trend(int(row['crop_id']), int(row['month']), float(row['rainfall']), float(row['temperature']))
    rows.append(market_trend)

df['market_trend'] = rows
# --- New: add previous month and month-to-month crop differences using historical data ---
HIST_PATHS = ['market_price_data.csv']
hist_df = None
for p in HIST_PATHS:
    if os.path.exists(p):
        try:
            hist_df = pd.read_csv(p)
            break
        except Exception:
            hist_df = None

if hist_df is not None:
    # Normalize column names
    cols_lower = {c.lower(): c for c in hist_df.columns}
    # determine price column
    price_col = None
    for candidate in ['price_per_quintal', 'price', 'price_per_quintal'.lower()]:
        if candidate in cols_lower:
            price_col = cols_lower[candidate]
            break
    # fallback: try any column containing 'price'
    if price_col is None:
        for c in hist_df.columns:
            if 'price' in c.lower():
                price_col = c
                break

    # determine crop and state name columns
    crop_col = None
    state_col = None
    for c in hist_df.columns:
        if c.lower() in ('crop_name', 'crop', 'cropname'):
            crop_col = c
        if c.lower() in ('state_name', 'state', 'state_name'):
            state_col = c

    month_col = None
    for c in hist_df.columns:
        if c.lower() in ('month', 'month_name'):
            month_col = c

    # create normalized working df if we have required columns
    if price_col is not None and crop_col is not None and month_col is not None:
        # map crop/state names to IDs in historical df if needed
        hist = hist_df.copy()
        # ensure month numeric
        if hist[month_col].dtype == object:
            # try to map month names to numbers where possible
            month_map = {m.lower(): i for i,m in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'], start=1)}
            hist[month_col] = hist[month_col].apply(lambda x: month_map.get(str(x).strip().lower(), x))
        # try to compute numeric price
        hist[price_col] = pd.to_numeric(hist[price_col], errors='coerce')

        # helper to get mean price for crop/state/month
        def lookup_mean_price(crop_name, state_name, month_num):
            sub = hist[(hist[crop_col].str.lower() == str(crop_name).lower()) & (hist[month_col] == month_num)]
            if state_name is not None:
                sub_state = sub[sub[state_col].str.lower() == str(state_name).lower()] if state_col in hist.columns else sub
                if not sub_state.empty:
                    return float(sub_state[price_col].mean())
            if not sub.empty:
                return float(sub[price_col].mean())
            return np.nan

        prev_month_prices = []
        crop_prev_mean = []
        crop_month_diff = []
        crop_month_diff_pct = []
        for idx, row in df.iterrows():
            m = int(row['month'])
            prev_m = 12 if m == 1 else m - 1
            crop_name = row['crop_type']
            state_name = row['state']

            prev_price = lookup_mean_price(crop_name, state_name, prev_m)
            # fallback: lookup by crop only
            if np.isnan(prev_price):
                prev_price = lookup_mean_price(crop_name, None, prev_m)

            # crop-level month means (current and previous) across all states
            cur_mean = lookup_mean_price(crop_name, None, m)
            prev_mean = lookup_mean_price(crop_name, None, prev_m)

            # calculate differences
            if np.isnan(prev_price):
                price_change = np.nan
                price_change_pct = np.nan
            else:
                price_change = float(df.at[idx, 'predicted_price']) - float(prev_price)
                price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else np.nan

            # crop month diff (current mean - prev mean)
            if np.isnan(cur_mean) or np.isnan(prev_mean):
                c_diff = np.nan
                c_diff_pct = np.nan
            else:
                c_diff = float(cur_mean) - float(prev_mean)
                c_diff_pct = (c_diff / prev_mean * 100) if prev_mean != 0 else np.nan

            prev_month_prices.append(prev_price)
            crop_prev_mean.append(prev_mean)
            crop_month_diff.append(c_diff)
            crop_month_diff_pct.append(c_diff_pct)

        df['previous_month_price'] = np.round(prev_month_prices, 2)
        df['price_change'] = np.round(df['predicted_price'] - df['previous_month_price'], 2)
        df['price_change_percent'] = np.round((df['price_change'] / df['previous_month_price'] * 100).replace([np.inf, -np.inf], np.nan), 2)
        df['crop_prev_month_mean'] = np.round(crop_prev_mean, 2)
        df['crop_month_diff'] = np.round(crop_month_diff, 2)
        df['crop_month_diff_percent'] = np.round(crop_month_diff_pct, 2)
    else:
        # If we cannot find historical data, fill with NaN
        df['previous_month_price'] = np.nan
        df['price_change'] = np.nan
        df['price_change_percent'] = np.nan
        df['crop_prev_month_mean'] = np.nan
        df['crop_month_diff'] = np.nan
        df['crop_month_diff_percent'] = np.nan
else:
    # no historical file found
    df['previous_month_price'] = np.nan
    df['price_change'] = np.nan
    df['price_change_percent'] = np.nan
    df['crop_prev_month_mean'] = np.nan
    df['crop_month_diff'] = np.nan
    df['crop_month_diff_percent'] = np.nan

# Reorder columns for output (include new fields)
out_cols = ['crop_type','state','month','rainfall','temperature','predicted_price','previous_month_price','price_change','price_change_percent','crop_prev_month_mean','crop_month_diff','crop_month_diff_percent','market_trend']
df[out_cols].to_csv(OUT_CSV, index=False)

print(f"✅ Predictions saved to {OUT_CSV} ({len(df)} rows) at {datetime.now().isoformat()}")
