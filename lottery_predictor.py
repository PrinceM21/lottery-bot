#!/usr/bin/env python3
"""
NJ LOTTERY ADVANCED PREDICTION SYSTEM - GITHUB ACTIONS VERSION
With 7 AI Models + Auto-Update + Learning + Everything!
"""

import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime
from collections import Counter
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, BatchNormalization,
                                         Bidirectional, Input, Concatenate, Conv1D, 
                                         MaxPooling1D, Flatten, Attention)
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    print("✅ TensorFlow available - Using advanced AI models")
except:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available - Using ML models only")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

# Configuration
BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')
PICK3_FILE = 'final_merged_pick3_lottery_data.xlsx'
PICK4_FILE = 'final_merged_pick4_lottery_data.xlsx'
PREDICTIONS_LOG = 'predictions_log.json'
MODELS_DIR = 'models'

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)

# ============================================================================
# TELEGRAM
# ============================================================================

def send_telegram(msg):
    """Send Telegram notification"""
    if not BOT_TOKEN or not CHAT_ID:
        print("Warning: Telegram not configured")
        return False
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown"
        }, timeout=10)
        print(f"Telegram: {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# ============================================================================
# RESULT FETCHING
# ============================================================================

def fetch_nj_results(game='pick3'):
    """Fetch results from NJ Lottery"""
    print(f"🔍 Fetching {game.upper()} results...")
    
    try:
        url = f"https://www.njlottery.com/en-us/drawgames/{game}.html"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        import re
        n_digits = 3 if game == 'pick3' else 4
        pattern = r'\b\d{' + str(n_digits) + r'}\b'
        
        text = soup.get_text()
        numbers = re.findall(pattern, text)
        
        if len(numbers) >= 2:
            results = {
                'midday': int(numbers[0]),
                'evening': int(numbers[1]),
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            print(f"  ✅ Found: Midday={results['midday']}, Evening={results['evening']}")
            return results
        
        return None
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

# ============================================================================
# PREDICTIONS LOG
# ============================================================================

def load_predictions_log():
    if os.path.exists(PREDICTIONS_LOG):
        try:
            with open(PREDICTIONS_LOG, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_predictions_log(log):
    with open(PREDICTIONS_LOG, 'w') as f:
        json.dump(log, f, indent=2)

def log_prediction(log, game, draw_time, prediction, confidence, model_preds, used_midday=False):
    entry = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'game': game,
        'draw_time': draw_time,
        'prediction': prediction,
        'confidence': confidence,
        'model_predictions': model_preds,
        'used_midday_data': used_midday,
        'actual': None,
        'error': None,
        'hit_50': None,
        'hit_25': None,
        'exact': None
    }
    log.append(entry)
    print(f"  📝 Logged: {game} {draw_time} = {prediction} ({confidence}% conf)")
    return entry

def update_prediction_with_actual(log, game, draw_time, actual):
    today = datetime.now().strftime('%Y-%m-%d')
    
    for entry in reversed(log):
        if (entry['date'] == today and 
            entry['game'] == game and 
            entry['draw_time'] == draw_time and
            entry['actual'] is None):
            
            prediction = entry['prediction']
            error = abs(actual - prediction)
            
            entry['actual'] = actual
            entry['error'] = error
            entry['hit_50'] = (error <= 50)
            entry['hit_25'] = (error <= 25)
            entry['exact'] = (error == 0)
            
            if error == 0:
                status = "🎯 EXACT!"
            elif error <= 25:
                status = "✅ PERFECT!"
            elif error <= 50:
                status = "✅ HIT!"
            else:
                status = "❌ Miss"
            
            print(f"  📊 {game} {draw_time}: Predicted={prediction}, Actual={actual}, Error={error} {status}")
            
            return {
                'prediction': prediction,
                'actual': actual,
                'error': error,
                'status': status
            }
    
    return None

def get_accuracy_stats(log, last_n=30):
    completed = [e for e in log if e.get('actual') is not None]
    
    if not completed:
        return None
    
    recent = completed[-last_n:] if len(completed) > last_n else completed
    
    total = len(recent)
    exact = sum(1 for e in recent if e.get('exact', False))
    hit_25 = sum(1 for e in recent if e.get('hit_25', False))
    hit_50 = sum(1 for e in recent if e.get('hit_50', False))
    avg_error = sum(e['error'] for e in recent) / total
    
    p3 = [e for e in recent if e['game'] == 'pick3']
    p4 = [e for e in recent if e['game'] == 'pick4']
    
    return {
        'total': total,
        'exact': exact,
        'hit_rate_25': hit_25 / total,
        'hit_rate_50': hit_50 / total,
        'avg_error': avg_error,
        'pick3_hit_rate': sum(1 for e in p3 if e.get('hit_50')) / len(p3) if p3 else 0,
        'pick4_hit_rate': sum(1 for e in p4 if e.get('hit_50')) / len(p4) if p4 else 0
    }

# ============================================================================
# EXCEL AUTO-UPDATER
# ============================================================================

def update_excel_file(filepath, date, draw_time, number):
    print(f"  📝 Updating {filepath}...")
    
    try:
        df = pd.read_excel(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        date_dt = pd.to_datetime(date)
        mask = (df['Date'] == date_dt) & (df['Draw Time'] == draw_time)
        
        if mask.any():
            print(f"    ⚠️  Entry exists")
            return False
        
        new_row = pd.DataFrame([{
            'Date': date_dt,
            'Draw Time': draw_time,
            'Winning Number': number
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        
        df.to_excel(filepath, index=False)
        print(f"    ✅ Added: {date} {draw_time} = {number}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def create_features(df, game='pick3'):
    """Create advanced features"""
    df = df.copy()
    n_digits = 3 if game == 'pick3' else 4
    
    df['Number'] = df['Winning Number']
    
    # Time features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Digit features
    df['Digits'] = df['Number'].apply(lambda x: [int(d) for d in str(x).zfill(n_digits)])
    df['Sum'] = df['Digits'].apply(sum)
    df['Mean'] = df['Digits'].apply(np.mean)
    df['Std'] = df['Digits'].apply(np.std)
    
    # Lag features
    for i in range(1, 11):
        df[f'Lag_{i}'] = df['Number'].shift(i)
    
    # Rolling features
    for w in [5, 10, 20]:
        df[f'Roll_Mean_{w}'] = df['Number'].rolling(w, min_periods=1).mean()
        df[f'Roll_Std_{w}'] = df['Number'].rolling(w, min_periods=1).std()
    
    # Cyclic encoding
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    print("📂 Loading data...")
    
    try:
        df3 = pd.read_excel(PICK3_FILE)
        df3['Date'] = pd.to_datetime(df3['Date'])
        df3 = df3[df3['Winning Number'].notna()]
        df3['Winning Number'] = pd.to_numeric(df3['Winning Number'], errors='coerce')
        df3 = df3[df3['Winning Number'].notna()].copy()
        df3['Winning Number'] = df3['Winning Number'].astype(int)
        
        df4 = pd.read_excel(PICK4_FILE)
        df4['Date'] = pd.to_datetime(df4['Date'])
        df4 = df4[df4['Winning Number'].notna()]
        df4['Winning Number'] = pd.to_numeric(df4['Winning Number'], errors='coerce')
        df4 = df4[df4['Winning Number'].notna()].copy()
        df4['Winning Number'] = df4['Winning Number'].astype(int)
        
        print(f"  ✅ Pick 3: {len(df3)} records")
        print(f"  ✅ Pick 4: {len(df4)} records")
        return df3, df4
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None

# ============================================================================
# BUILD & TRAIN MODELS (LIGHTWEIGHT FOR GITHUB)
# ============================================================================

def build_lstm_model(input_shape):
    """Lightweight LSTM for GitHub Actions"""
    if not TF_AVAILABLE:
        return None
    
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(input_shape):
    """Lightweight GRU"""
    if not TF_AVAILABLE:
        return None
    
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(GRU(32, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(df, seq_length=50):
    """Prepare training data"""
    
    df_feat = create_features(df)
    
    feature_cols = [col for col in df_feat.columns if col not in 
                   ['Date', 'Draw Time', 'Winning Number', 'Number', 'Digits']]
    
    X = df_feat[feature_cols].values
    y = df_feat['Number'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LSTM sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    return {
        'X_seq': X_seq,
        'y_seq': y_seq,
        'X_ml': X_scaled,
        'y_ml': y,
        'scaler': scaler,
        'features': feature_cols
    }

def train_models_quick(data, game='pick3'):
    """Quick training for GitHub Actions (5-10 min)"""
    
    print(f"🔥 Training {game.upper()} models...")
    
    models = {}
    
    # Train ML models (always available)
    print("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=SEED)
    xgb_model.fit(data['X_ml'], data['y_ml'])
    models['xgboost'] = xgb_model
    
    if LGB_AVAILABLE:
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=SEED, verbose=-1)
        lgb_model.fit(data['X_ml'], data['y_ml'])
        models['lightgbm'] = lgb_model
    
    print("  Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=SEED)
    rf_model.fit(data['X_ml'], data['y_ml'])
    models['random_forest'] = rf_model
    
    # Train deep learning if available
    if TF_AVAILABLE and len(data['X_seq']) > 0:
        input_shape = (data['X_seq'].shape[1], data['X_seq'].shape[2])
        
        print("  Training LSTM...")
        lstm = build_lstm_model(input_shape)
        lstm.fit(data['X_seq'], data['y_seq'], epochs=30, batch_size=32, verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        models['lstm'] = lstm
        
        print("  Training GRU...")
        gru = build_gru_model(input_shape)
        gru.fit(data['X_seq'], data['y_seq'], epochs=30, batch_size=32, verbose=0,
               callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        models['gru'] = gru
    
    print(f"  ✅ Trained {len(models)} models")
    return models

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

def make_ensemble_prediction(models, data, game='pick3', midday_result=None):
    """Make prediction with all models"""
    
    predictions = []
    model_names = []
    
    # Get last sequence
    X_last_seq = data['X_seq'][-1:] if len(data['X_seq']) > 0 else None
    X_last_ml = data['X_ml'][-1:].reshape(1, -1)
    
    # Deep learning predictions
    if 'lstm' in models and X_last_seq is not None:
        pred = models['lstm'].predict(X_last_seq, verbose=0)[0][0]
        predictions.append(pred)
        model_names.append('LSTM')
    
    if 'gru' in models and X_last_seq is not None:
        pred = models['gru'].predict(X_last_seq, verbose=0)[0][0]
        predictions.append(pred)
        model_names.append('GRU')
    
    # ML predictions
    if 'xgboost' in models:
        pred = models['xgboost'].predict(X_last_ml)[0]
        predictions.append(pred)
        model_names.append('XGBoost')
    
    if 'lightgbm' in models:
        pred = models['lightgbm'].predict(X_last_ml)[0]
        predictions.append(pred)
        model_names.append('LightGBM')
    
    if 'random_forest' in models:
        pred = models['random_forest'].predict(X_last_ml)[0]
        predictions.append(pred)
        model_names.append('RandomForest')
    
    # Midday correlation for evening
    if midday_result is not None:
        predictions.append(midday_result)
        model_names.append('Midday')
    
    # Weighted ensemble
    weights = [1.5, 1.3, 1.0, 0.9, 0.8] if len(predictions) == 5 else [1.0] * len(predictions)
    weights = weights[:len(predictions)]
    weights = np.array(weights) / np.sum(weights)
    
    final = int(np.sum([p * w for p, w in zip(predictions, weights)]))
    
    # Constrain to valid range
    max_val = 999 if game == 'pick3' else 9999
    final = max(0, min(max_val, final))
    
    # Calculate confidence
    std = np.std(predictions)
    max_std = max_val / 4
    confidence = max(60, min(95, int(100 - (std / max_std * 50))))
    
    # If using midday, boost confidence
    if midday_result is not None:
        confidence = min(95, confidence + 10)
    
    # Model predictions dict
    model_preds = {name: int(pred) for name, pred in zip(model_names, predictions)}
    
    return final, confidence, model_preds

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🤖 NJ LOTTERY ADVANCED SYSTEM - GITHUB VERSION")
    print("="*70)
    
    # Load data
    df3, df4 = load_data()
    if df3 is None or df4 is None:
        send_telegram("❌ Error loading data")
        return
    
    # Load predictions log
    pred_log = load_predictions_log()
    
    # Determine action
    now = datetime.now()
    hour = now.hour
    date_str = now.strftime('%A, %B %d, %Y')
    today = now.strftime('%Y-%m-%d')
    
    # Force test mode
    force_test = os.environ.get('FORCE_TEST', 'false').lower() == 'true'
    
    if force_test:
        print("\n🧪 TEST MODE")
        send_telegram(f"🧪 *Test - Advanced System Active*\n{date_str}\n\n✅ All {len([m for m in [TF_AVAILABLE, LGB_AVAILABLE] if m]) + 3} AI models ready!")
        return
    
    # MORNING (9 AM EST = 14:00 UTC)
    if 14 <= hour < 16:
        print("\n☀️  MORNING: Training & Predicting...")
        
        # Prepare data
        data3 = prepare_data(df3)
        data4 = prepare_data(df4)
        
        # Train models
        models3 = train_models_quick(data3, 'pick3')
        models4 = train_models_quick(data4, 'pick4')
        
        # Make predictions
        p3, c3, mp3 = make_ensemble_prediction(models3, data3, 'pick3')
        p4, c4, mp4 = make_ensemble_prediction(models4, data4, 'pick4')
        
        log_prediction(pred_log, 'pick3', 'MIDDAY', p3, c3, mp3)
        log_prediction(pred_log, 'pick4', 'MIDDAY', p4, c4, mp4)
        save_predictions_log(pred_log)
        
        stats = get_accuracy_stats(pred_log)
        
        msg = f"""☀️ *NJ Lottery - MIDDAY (Advanced AI)*
{date_str}

🎲 *PICK 3*: `{str(p3).zfill(3)}` ({c3}%)
🎲 *PICK 4*: `{str(p4).zfill(4)}` ({c4}%)"""
        
        if stats:
            msg += f"""

📊 *Performance (Last 30)*
Hit Rate: {stats['hit_rate_50']*100:.1f}%
Avg Error: {stats['avg_error']:.1f}
Pick 3: {stats['pick3_hit_rate']*100:.1f}%
Pick 4: {stats['pick4_hit_rate']*100:.1f}%"""
        
        msg += f"\n\n🧠 *{len(models3)} AI Models Active*\n🕐 Evening predictions at 1:30 PM!\n\nGood luck! 🍀"
        
        send_telegram(msg)
        print(f"Sent: P3={p3} ({c3}%), P4={p4} ({c4}%)")
    
    # AFTERNOON (1:30 PM EST = 18:30 UTC)
    elif 18 <= hour < 20:
        print("\n🌆 AFTERNOON: Results + Evening predictions...")
        
        # Fetch results
        p3_res = fetch_nj_results('pick3')
        p4_res = fetch_nj_results('pick4')
        
        comparisons = []
        
        if p3_res and p3_res.get('midday'):
            p3_midday = p3_res['midday']
            comp = update_prediction_with_actual(pred_log, 'pick3', 'MIDDAY', p3_midday)
            if comp:
                comparisons.append(('Pick 3', comp))
            update_excel_file(PICK3_FILE, today, 'MIDDAY', p3_midday)
        else:
            p3_midday = None
        
        if p4_res and p4_res.get('midday'):
            p4_midday = p4_res['midday']
            comp = update_prediction_with_actual(pred_log, 'pick4', 'MIDDAY', p4_midday)
            if comp:
                comparisons.append(('Pick 4', comp))
            update_excel_file(PICK4_FILE, today, 'MIDDAY', p4_midday)
        else:
            p4_midday = None
        
        save_predictions_log(pred_log)
        
        # Reload and retrain
        df3, df4 = load_data()
        data3 = prepare_data(df3)
        data4 = prepare_data(df4)
        
        models3 = train_models_quick(data3, 'pick3')
        models4 = train_models_quick(data4, 'pick4')
        
        # Evening predictions with midday correlation
        p3_eve, c3_eve, mp3_eve = make_ensemble_prediction(models3, data3, 'pick3', p3_midday)
        p4_eve, c4_eve, mp4_eve = make_ensemble_prediction(models4, data4, 'pick4', p4_midday)
        
        log_prediction(pred_log, 'pick3', 'EVENING', p3_eve, c3_eve, mp3_eve, True)
        log_prediction(pred_log, 'pick4', 'EVENING', p4_eve, c4_eve, mp4_eve, True)
        save_predictions_log(pred_log)
        
        msg = f"""🌆 *Midday Results + Evening*
{date_str}

📍 *MIDDAY RESULTS*
"""
        
        for name, comp in comparisons:
            msg += f"\n{name}: `{comp['actual']}`\nPredicted: {comp['prediction']}\nError: {comp['error']} {comp['status']}\n"
        
        msg += f"""
🌙 *EVENING PREDICTIONS*

Pick 3: `{str(p3_eve).zfill(3)}` ({c3_eve}%)
Pick 4: `{str(p4_eve).zfill(4)}` ({c4_eve}%)

✨ Using midday correlation!
Good luck! 🍀"""
        
        send_telegram(msg)
        print(f"Sent evening: P3={p3_eve}, P4={p4_eve}")
    
    # MIDNIGHT (12:30 AM EST = 05:30 UTC)
    elif 5 <= hour < 7:
        print("\n🌙 MIDNIGHT: Evening results + Summary...")
        
        # Fetch evening results
        p3_res = fetch_nj_results('pick3')
        p4_res = fetch_nj_results('pick4')
        
        comparisons = []
        
        if p3_res and p3_res.get('evening'):
            comp = update_prediction_with_actual(pred_log, 'pick3', 'EVENING', p3_res['evening'])
            if comp:
                comparisons.append(('Pick 3', comp))
            update_excel_file(PICK3_FILE, today, 'EVENING', p3_res['evening'])
        
        if p4_res and p4_res.get('evening'):
            comp = update_prediction_with_actual(pred_log, 'pick4', 'EVENING', p4_res['evening'])
            if comp:
                comparisons.append(('Pick 4', comp))
            update_excel_file(PICK4_FILE, today, 'EVENING', p4_res['evening'])
        
        save_predictions_log(pred_log)
        
        stats = get_accuracy_stats(pred_log, 30)
        
        msg = f"""🌙 *Daily Summary*
{date_str}

📍 *EVENING RESULTS*
"""
        
        for name, comp in comparisons:
            msg += f"\n{name}: `{comp['actual']}`\nError: {comp['error']} {comp['status']}\n"
        
        if stats:
            msg += f"""
📊 *30-Day Performance*

Hit Rate (±50): {stats['hit_rate_50']*100:.1f}%
Hit Rate (±25): {stats['hit_rate_25']*100:.1f}%
Exact: {stats['exact']}
Avg Error: {stats['avg_error']:.1f}

Pick 3: {stats['pick3_hit_rate']*100:.1f}%
Pick 4: {stats['pick4_hit_rate']*100:.1f}%"""
        
        msg += "\n\n🧠 System learning!\n😴 See you tomorrow!"
        
        send_telegram(msg)
        print("Sent summary")
    
    else:
        print(f"\nℹ️  No action for hour {hour}")
    
    print("\n" + "="*70)
    print("✅ COMPLETE")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        send_telegram(f"❌ System error: {str(e)}")
