#!/usr/bin/env python3
"""
🎯 NJ LOTTERY: LSTM + XGBOOST WITH SELF-LEARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on your original Colab code with:
✅ LSTM model
✅ XGBoost model  
✅ Self-learning (adaptive weights)
✅ Telegram bot
✅ Auto-save to Excel
"""

import os
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available, will use XGBoost only")

import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

PICK3_FILE = 'final_merged_pick3_lottery_data.xlsx'
PICK4_FILE = 'final_merged_pick4_lottery_data.xlsx'
WEIGHTS_FILE = 'learning_weights.json'
PREDICTIONS_LOG = 'predictions_log.json'

SEED = 42
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)

# Model parameters
SEQ_LENGTH = 50
LEARNING_RATE = 0.1  # For weight adaptation

print("="*80)
print("🎯 LSTM + XGBOOST WITH SELF-LEARNING")
print("="*80)
print(f"TensorFlow: {'✅' if TF_AVAILABLE else '❌'}")

# ══════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════

def send_message(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        print(f"MSG: {msg}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
    except:
        pass

def wait_for_reply(prompt: str, timeout: int = 3600) -> Optional[str]:
    print(f"\n📱 {prompt}")
    send_message(prompt)
    
    if not BOT_TOKEN or not CHAT_ID:
        try:
            return input("Reply: ").strip()
        except:
            return None
    
    try:
        updates = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params={"timeout": 30},
            timeout=35
        ).json()
    except:
        updates = None
    
    last_id = 0
    if updates and updates.get('result'):
        for u in updates['result']:
            last_id = max(last_id, u['update_id'])
    
    start = time.time()
    reminder_sent = False
    
    while time.time() - start < timeout:
        elapsed = time.time() - start
        time.sleep(5)
        
        if elapsed >= 1800 and not reminder_sent:
            send_message("⏰ Reminder: " + prompt + " (30 min left)")
            reminder_sent = True
        
        try:
            new_updates = requests.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                params={"timeout": 30, "offset": last_id + 1},
                timeout=35
            ).json()
        except:
            continue
        
        if new_updates and new_updates.get('result'):
            for update in new_updates['result']:
                last_id = update['update_id']
                if 'message' in update:
                    msg = update['message']
                    if str(msg.get('chat', {}).get('id', '')) == str(CHAT_ID):
                        return msg.get('text', '').strip()
    
    send_message("⏰ No reply in 1 hour. Skipping.")
    return None

def validate_number(text: Optional[str], digits: int) -> Optional[int]:
    if not text:
        return None
    text = text.strip().replace(' ', '')
    if not text.isdigit() or len(text) > digits:
        return None
    return int(text.zfill(digits))

# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    print("📂 Loading data...")
    try:
        df3 = pd.read_excel(PICK3_FILE)
        df4 = pd.read_excel(PICK4_FILE)
        
        df3['Date'] = pd.to_datetime(df3['Date'])
        df4['Date'] = pd.to_datetime(df4['Date'])
        
        df3['Winning Number'] = pd.to_numeric(df3['Winning Number'], errors='coerce')
        df4['Winning Number'] = pd.to_numeric(df4['Winning Number'], errors='coerce')
        
        df3 = df3[df3['Winning Number'].notna()].copy()
        df4 = df4[df4['Winning Number'].notna()].copy()
        
        df3['Winning Number'] = df3['Winning Number'].astype(int)
        df4['Winning Number'] = df4['Winning Number'].astype(int)
        
        print(f"  ✅ Pick 3: {len(df3):,} records")
        print(f"  ✅ Pick 4: {len(df4):,} records")
        return df3, df4
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None

def save_result(filepath: str, date: str, draw_time: str, number: int):
    try:
        df = pd.read_excel(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        date_dt = pd.to_datetime(date)
        
        if ((df['Date'] == date_dt) & (df['Draw Time'] == draw_time)).any():
            return False
        
        new_row = pd.DataFrame([{
            'Date': date_dt,
            'Draw Time': draw_time,
            'Winning Number': int(number)
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df.to_excel(filepath, index=False)
        
        print(f"  ✅ Saved: {draw_time} = {number}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

# ══════════════════════════════════════════════════════════════════════════
# SELF-LEARNING WEIGHTS
# ══════════════════════════════════════════════════════════════════════════

def load_weights(game: str):
    """Load learned model weights"""
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                all_weights = json.load(f)
                if game in all_weights:
                    return all_weights[game]
        except:
            pass
    
    # Default: equal weights
    return {
        'lstm_weight': 1.0,
        'xgb_weight': 1.0,
        'iterations': 0,
        'history': []
    }

def save_weights(game: str, weights: Dict):
    """Save learned weights"""
    all_weights = {}
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                all_weights = json.load(f)
        except:
            pass
    
    all_weights[game] = weights
    
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(all_weights, f, indent=2)

def update_weights(game: str, lstm_error: float, xgb_error: float):
    """
    🧠 SELF-LEARNING: Update weights based on which model performed better
    """
    weights = load_weights(game)
    
    # Which model was better?
    if lstm_error < xgb_error:
        # LSTM was better, increase its weight
        adjustment = LEARNING_RATE * (1 - lstm_error / (lstm_error + xgb_error))
        weights['lstm_weight'] *= (1.0 + adjustment)
        weights['xgb_weight'] *= (1.0 - adjustment * 0.5)
    else:
        # XGBoost was better, increase its weight
        adjustment = LEARNING_RATE * (1 - xgb_error / (lstm_error + xgb_error))
        weights['xgb_weight'] *= (1.0 + adjustment)
        weights['lstm_weight'] *= (1.0 - adjustment * 0.5)
    
    # Keep weights in reasonable range
    weights['lstm_weight'] = max(0.3, min(3.0, weights['lstm_weight']))
    weights['xgb_weight'] = max(0.3, min(3.0, weights['xgb_weight']))
    
    # Normalize
    total = weights['lstm_weight'] + weights['xgb_weight']
    weights['lstm_weight'] = weights['lstm_weight'] / total * 2
    weights['xgb_weight'] = weights['xgb_weight'] / total * 2
    
    weights['iterations'] += 1
    weights['history'].append({
        'timestamp': datetime.now().isoformat(),
        'lstm_error': lstm_error,
        'xgb_error': xgb_error,
        'lstm_weight': weights['lstm_weight'],
        'xgb_weight': weights['xgb_weight']
    })
    
    # Keep last 100 history
    if len(weights['history']) > 100:
        weights['history'] = weights['history'][-100:]
    
    save_weights(game, weights)
    
    print(f"  🧠 Weights updated (iter {weights['iterations']}):")
    print(f"     LSTM: {weights['lstm_weight']:.2f} | XGB: {weights['xgb_weight']:.2f}")
    
    return weights

# ══════════════════════════════════════════════════════════════════════════
# PREDICTIONS LOG
# ══════════════════════════════════════════════════════════════════════════

def load_log():
    if os.path.exists(PREDICTIONS_LOG):
        try:
            with open(PREDICTIONS_LOG, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_log(log):
    with open(PREDICTIONS_LOG, 'w') as f:
        json.dump(log, f, indent=2)

def add_prediction(game: str, draw_time: str, prediction: int, lstm_pred: int, xgb_pred: int):
    log = load_log()
    log.append({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'game': game,
        'draw_time': draw_time,
        'prediction': prediction,
        'lstm_pred': lstm_pred,
        'xgb_pred': xgb_pred,
        'actual': None,
        'error': None,
        'lstm_error': None,
        'xgb_error': None,
        'timestamp': datetime.now().isoformat()
    })
    save_log(log)

def update_actual(game: str, draw_time: str, actual: int):
    log = load_log()
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    for date_str in [today, yesterday]:
        for entry in reversed(log):
            if (entry['date'] == date_str and 
                entry['game'] == game and 
                entry['draw_time'] == draw_time and
                entry['actual'] is None):
                
                entry['actual'] = actual
                entry['error'] = abs(actual - entry['prediction'])
                entry['lstm_error'] = abs(actual - entry['lstm_pred'])
                entry['xgb_error'] = abs(actual - entry['xgb_pred'])
                
                save_log(log)
                
                # 🧠 SELF-LEARNING: Update weights
                update_weights(game, entry['lstm_error'], entry['xgb_error'])
                
                return entry
    
    return None

# ══════════════════════════════════════════════════════════════════════════
# PREPARE DATA FOR TRAINING
# ══════════════════════════════════════════════════════════════════════════

def prepare_lstm_data(data, seq_length):
    """Prepare sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def calculate_batch_size(X, target_batches=874):
    """Calculate batch size to get ~874 batches"""
    batch_size = max(1, len(X) // target_batches)
    return batch_size

# ══════════════════════════════════════════════════════════════════════════
# BUILD MODELS
# ══════════════════════════════════════════════════════════════════════════

def build_lstm_model(seq_length):
    """Build LSTM model (from your code)"""
    if not TF_AVAILABLE:
        return None
    
    model = Sequential([
        LSTM(200, return_sequences=True, input_shape=(seq_length, 1)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(150, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='mean_squared_error'
    )
    
    return model

# ══════════════════════════════════════════════════════════════════════════
# TRAIN AND PREDICT
# ══════════════════════════════════════════════════════════════════════════

def train_and_predict(df, game: str):
    """Train LSTM + XGBoost and make prediction"""
    print(f"\n🔥 Training {game.upper()}...")
    
    # Load learned weights
    weights = load_weights(game)
    print(f"  🧠 Using weights - LSTM: {weights['lstm_weight']:.2f}, XGB: {weights['xgb_weight']:.2f}")
    
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Winning Number']])
    
    X, y = prepare_lstm_data(scaled_data, SEQ_LENGTH)
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
    X_xgb = X.reshape((X.shape[0], -1))
    
    batch_size = calculate_batch_size(X, 874)
    
    lstm_pred = None
    xgb_pred = None
    
    # ══════════════════════════════════════════════════════════════════════
    # TRAIN LSTM
    # ══════════════════════════════════════════════════════════════════════
    if TF_AVAILABLE:
        print("  📊 Training LSTM...")
        model_lstm = build_lstm_model(SEQ_LENGTH)
        
        callbacks = [EarlyStopping(patience=20, restore_best_weights=True)]
        
        model_lstm.fit(
            X_lstm, y,
            epochs=200,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks
        )
        
        # Predict
        last_seq = X_lstm[-1].reshape(1, SEQ_LENGTH, 1)
        lstm_pred_scaled = model_lstm.predict(last_seq, verbose=0)
        lstm_pred = int(scaler.inverse_transform(lstm_pred_scaled)[0][0])
        
        print(f"     LSTM prediction: {lstm_pred}")
    
    # ══════════════════════════════════════════════════════════════════════
    # TRAIN XGBOOST
    # ══════════════════════════════════════════════════════════════════════
    print("  📊 Training XGBoost...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=SEED
    )
    
    model_xgb.fit(X_xgb, y)
    
    # Predict
    last_seq_xgb = X_xgb[-1].reshape(1, -1)
    xgb_pred_scaled = model_xgb.predict(last_seq_xgb)
    xgb_pred = int(scaler.inverse_transform([[xgb_pred_scaled[0]]])[0][0])
    
    print(f"     XGBoost prediction: {xgb_pred}")
    
    # ══════════════════════════════════════════════════════════════════════
    # WEIGHTED ENSEMBLE (SELF-LEARNING!)
    # ══════════════════════════════════════════════════════════════════════
    if lstm_pred is not None and xgb_pred is not None:
        # Use learned weights
        final_pred = int(
            (lstm_pred * weights['lstm_weight'] + xgb_pred * weights['xgb_weight']) / 
            (weights['lstm_weight'] + weights['xgb_weight'])
        )
    elif lstm_pred is not None:
        final_pred = lstm_pred
    else:
        final_pred = xgb_pred
    
    # Ensure valid range
    n_digits = 3 if game == 'pick3' else 4
    max_val = 999 if game == 'pick3' else 9999
    final_pred = max(0, min(max_val, final_pred))
    
    print(f"  🎯 Final prediction: {str(final_pred).zfill(n_digits)}")
    
    return {
        'prediction': final_pred,
        'lstm_pred': lstm_pred if lstm_pred else final_pred,
        'xgb_pred': xgb_pred,
        'lstm_weight': weights['lstm_weight'],
        'xgb_weight': weights['xgb_weight'],
        'iterations': weights['iterations']
    }

# ══════════════════════════════════════════════════════════════════════════
# WORKFLOWS
# ══════════════════════════════════════════════════════════════════════════

def morning_workflow():
    """10 AM: Get last night results + Predict midday"""
    print("\n☀️ MORNING WORKFLOW")
    print("="*80)
    
    yesterday = datetime.now() - timedelta(days=1)
    y_str = yesterday.strftime('%Y-%m-%d')
    y_display = yesterday.strftime('%A, %B %d, %Y')
    
    send_message("☀️ *Good morning!*\n\nEnter last night's results:")
    
    # Get last night's results
    results = {}
    for game, digits, path in [('pick3', 3, PICK3_FILE), ('pick4', 4, PICK4_FILE)]:
        num = None
        tries = 0
        while num is None and tries < 3:
            reply = wait_for_reply(f"🌙 {game.upper()} EVENING ({y_display})?")
            num = validate_number(reply, digits)
            tries += 1
            if num is None:
                send_message(f"❌ Invalid! Need {digits} digits ({tries}/3)")
        
        if num:
            results[game] = num
            save_result(path, y_str, 'EVENING', num)
    
    # Update log and learn
    for game, actual in results.items():
        entry = update_actual(game, 'EVENING', actual)
    
    # Reload data
    df3, df4 = load_data()
    if df3 is None:
        return
    
    # Make predictions
    print("\n🔮 Predicting MIDDAY...")
    r3 = train_and_predict(df3, 'pick3')
    r4 = train_and_predict(df4, 'pick4')
    
    # Log predictions
    add_prediction('pick3', 'MIDDAY', r3['prediction'], r3['lstm_pred'], r3['xgb_pred'])
    add_prediction('pick4', 'MIDDAY', r4['prediction'], r4['lstm_pred'], r4['xgb_pred'])
    
    # Send message
    msg = f"☀️ *TODAY MIDDAY Predictions*\n\n"
    msg += f"🎲 *PICK 3*: `{str(r3['prediction']).zfill(3)}`\n"
    msg += f"   🧠 Learning iteration: {r3['iterations']}\n\n"
    msg += f"🎲 *PICK 4*: `{str(r4['prediction']).zfill(4)}`\n"
    msg += f"   🧠 Learning iteration: {r4['iterations']}\n\n"
    msg += "🕐 Midday results at 2 PM!"
    
    send_message(msg)
    print("\n✅ Morning complete!")

def afternoon_workflow():
    """2 PM: Get midday results + Predict evening"""
    print("\n🌆 AFTERNOON WORKFLOW")
    print("="*80)
    
    today = datetime.now()
    t_str = today.strftime('%Y-%m-%d')
    
    send_message("🌆 *Midday Results Time!*")
    
    # Get midday results
    results = {}
    for game, digits, path in [('pick3', 3, PICK3_FILE), ('pick4', 4, PICK4_FILE)]:
        num = None
        tries = 0
        while num is None and tries < 3:
            reply = wait_for_reply(f"🎲 {game.upper()} MIDDAY?")
            num = validate_number(reply, digits)
            tries += 1
            if num is None:
                send_message(f"❌ Invalid! Need {digits} digits ({tries}/3)")
        
        if num:
            results[game] = num
            save_result(path, t_str, 'MIDDAY', num)
    
    # Update and learn
    for game, actual in results.items():
        entry = update_actual(game, 'MIDDAY', actual)
    
    # Reload
    df3, df4 = load_data()
    if df3 is None:
        return
    
    # Evening predictions
    print("\n🔮 Predicting EVENING...")
    r3 = train_and_predict(df3, 'pick3')
    r4 = train_and_predict(df4, 'pick4')
    
    # Log
    add_prediction('pick3', 'EVENING', r3['prediction'], r3['lstm_pred'], r3['xgb_pred'])
    add_prediction('pick4', 'EVENING', r4['prediction'], r4['lstm_pred'], r4['xgb_pred'])
    
    # Send
    msg = f"🌙 *EVENING Predictions*\n\n"
    msg += f"🎲 *PICK 3*: `{str(r3['prediction']).zfill(3)}`\n"
    msg += f"   🧠 Learning iteration: {r3['iterations']}\n\n"
    msg += f"🎲 *PICK 4*: `{str(r4['prediction']).zfill(4)}`\n"
    msg += f"   🧠 Learning iteration: {r4['iterations']}\n\n"
    msg += "🍀 Good luck!"
    
    send_message(msg)
    print("\n✅ Afternoon complete!")

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    hour = datetime.now().hour
    
    if 15 <= hour < 17:  # 10 AM EST
        morning_workflow()
    elif 19 <= hour < 21:  # 2 PM EST
        afternoon_workflow()
    else:
        print(f"No action for hour {hour}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        send_message(f"❌ Error: {str(e)}")
