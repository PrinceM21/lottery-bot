#!/usr/bin/env python3
"""
NJ LOTTERY TELEGRAM BOT INPUT SYSTEM
- Bot asks you for the draw results
- Saves to Excel automatically
- Runs AI predictions
- Sends results back to you
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Try to import deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

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

SEED = 42
np.random.seed(SEED)

# ============================================================================
# TELEGRAM FUNCTIONS
# ============================================================================

def send_message(msg, parse_mode="Markdown"):
    """Send a message via Telegram"""
    if not BOT_TOKEN or not CHAT_ID:
        print(f"MSG: {msg}")
        return None
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": parse_mode
        }, timeout=10)
        return r.json()
    except Exception as e:
        print(f"Send error: {e}")
        return None

def get_updates(offset=None):
    """Get new messages from Telegram"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": 30, "offset": offset}
    try:
        r = requests.get(url, params=params, timeout=35)
        return r.json()
    except Exception as e:
        print(f"Update error: {e}")
        return None

def wait_for_reply(prompt, timeout=3600):
    """
    Send a prompt and wait for user reply
    - Sends reminder at 30 minutes
    - Stops after 1 hour (3600 seconds)
    """
    print(f"\n📱 Asking: {prompt}")
    send_message(prompt)
    
    # Get current update ID
    updates = get_updates()
    last_id = 0
    if updates and updates.get('result'):
        for u in updates['result']:
            last_id = max(last_id, u['update_id'])
    
    start_time = time.time()
    reminder_sent = False
    
    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        time.sleep(5)
        
        # Send reminder at 30 minutes
        if elapsed >= 1800 and not reminder_sent:
            send_message(
                "⏰ *Reminder!*\n\n" + prompt +
                "\n\n_You have 30 minutes left before this is skipped!_"
            )
            reminder_sent = True
            print("   Sent 30-min reminder")
        
        # Check for new messages
        new_updates = get_updates(offset=last_id + 1)
        
        if new_updates and new_updates.get('result'):
            for update in new_updates['result']:
                last_id = update['update_id']
                
                if 'message' in update:
                    msg = update['message']
                    
                    if str(msg.get('chat', {}).get('id', '')) == str(CHAT_ID):
                        text = msg.get('text', '').strip()
                        print(f"   Got reply: {text}")
                        return text
    
    # Timed out after 1 hour
    send_message("⏰ *No reply received in 1 hour.*\n\nSkipping result entry for now.\n\n_You can enter results tomorrow or the system will ask again next time!_")
    print("   Timed out after 1 hour!")
    return None

def validate_number(text, digits):
    """Validate that text is a valid lottery number"""
    if not text:
        return None
    
    # Remove spaces
    text = text.strip().replace(' ', '')
    
    # Check if all digits
    if not text.isdigit():
        return None
    
    # Check length
    if len(text) != digits:
        # Try to pad with zeros
        if len(text) < digits:
            text = text.zfill(digits)
        else:
            return None
    
    return int(text)

# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def load_data():
    """Load Excel files"""
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

def save_result(filepath, date, draw_time, number):
    """Save a result to Excel"""
    try:
        df = pd.read_excel(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        date_dt = pd.to_datetime(date)
        
        # Check if exists
        if ((df['Date'] == date_dt) & (df['Draw Time'] == draw_time)).any():
            print(f"  ⚠️  Already exists: {draw_time} {number}")
            return False
        
        # Add new row
        new_row = pd.DataFrame([{
            'Date': date_dt,
            'Draw Time': draw_time,
            'Winning Number': number
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df.to_excel(filepath, index=False)
        
        print(f"  ✅ Saved: {draw_time} = {number}")
        return True
    
    except Exception as e:
        print(f"  ❌ Save error: {e}")
        return False

def create_features(df, game='pick3'):
    """Create features for ML"""
    df = df.copy()
    n_digits = 3 if game == 'pick3' else 4
    
    df['Number'] = df['Winning Number']
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    df['Digits'] = df['Number'].apply(lambda x: [int(d) for d in str(x).zfill(n_digits)])
    df['Sum'] = df['Digits'].apply(sum)
    df['Mean'] = df['Digits'].apply(np.mean)
    
    for i in range(1, 11):
        df[f'Lag_{i}'] = df['Number'].shift(i)
    
    for w in [5, 10, 20]:
        df[f'Roll_Mean_{w}'] = df['Number'].rolling(w, min_periods=1).mean()
        df[f'Roll_Std_{w}'] = df['Number'].rolling(w, min_periods=1).std()
    
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def train_and_predict(df, game='pick3', midday_result=None):
    """Train models and make prediction"""
    
    print(f"  🔥 Training {game.upper()} models...")
    
    n_digits = 3 if game == 'pick3' else 4
    max_val = 999 if game == 'pick3' else 9999
    
    df_feat = create_features(df, game)
    
    feature_cols = [col for col in df_feat.columns if col not in 
                   ['Date', 'Draw Time', 'Winning Number', 'Number', 'Digits']]
    
    X = df_feat[feature_cols].values
    y = df_feat['Number'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    predictions = []
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, 
                                  max_depth=5, random_state=SEED)
    xgb_model.fit(X_scaled, y)
    xgb_pred = xgb_model.predict(X_scaled[-1:].reshape(1, -1))[0]
    predictions.append(('XGBoost', xgb_pred, 1.0))
    
    # LightGBM
    if LGB_AVAILABLE:
        lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, 
                                      random_state=SEED, verbose=-1)
        lgb_model.fit(X_scaled, y)
        lgb_pred = lgb_model.predict(X_scaled[-1:].reshape(1, -1))[0]
        predictions.append(('LightGBM', lgb_pred, 0.9))
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=SEED)
    rf_model.fit(X_scaled, y)
    rf_pred = rf_model.predict(X_scaled[-1:].reshape(1, -1))[0]
    predictions.append(('RandomForest', rf_pred, 0.8))
    
    # LSTM if available
    if TF_AVAILABLE:
        seq_len = 50
        X_seq = []
        for i in range(len(X_scaled) - seq_len):
            X_seq.append(X_scaled[i:i+seq_len])
        
        if len(X_seq) > 0:
            X_seq = np.array(X_seq)
            y_seq = y[seq_len:]
            
            lstm = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), 
                            input_shape=(seq_len, X_scaled.shape[1])),
                Dropout(0.2),
                Bidirectional(LSTM(32)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            lstm.compile(optimizer='adam', loss='mse')
            lstm.fit(X_seq, y_seq, epochs=20, batch_size=32, verbose=0,
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
            
            lstm_pred = lstm.predict(X_seq[-1:], verbose=0)[0][0]
            predictions.append(('LSTM', lstm_pred, 1.5))
    
    # Midday correlation for evening
    if midday_result is not None:
        predictions.append(('Midday', midday_result, 1.2))
    
    # Weighted ensemble
    total_weight = sum(w for _, _, w in predictions)
    final = sum(p * w for _, p, w in predictions) / total_weight
    final = max(0, min(max_val, int(final)))
    
    # Confidence
    pred_values = [p for _, p, _ in predictions]
    std = np.std(pred_values)
    confidence = max(60, min(95, int(100 - (std / (max_val/4) * 50))))
    if midday_result:
        confidence = min(95, confidence + 10)
    
    # Model breakdown
    model_preds = {name: int(p) for name, p, _ in predictions}
    
    print(f"  ✅ Prediction: {final} ({confidence}%)")
    return final, confidence, model_preds

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

def update_log_with_actual(log, game, draw_time, actual):
    today = datetime.now().strftime('%Y-%m-%d')
    for entry in reversed(log):
        if (entry['date'] == today and 
            entry['game'] == game and 
            entry['draw_time'] == draw_time and
            entry['actual'] is None):
            
            error = abs(actual - entry['prediction'])
            entry['actual'] = actual
            entry['error'] = error
            entry['hit_50'] = (error <= 50)
            entry['hit_25'] = (error <= 25)
            entry['exact'] = (error == 0)
            
            if error == 0:
                return entry, "🎯 EXACT!"
            elif error <= 25:
                return entry, "✅ PERFECT!"
            elif error <= 50:
                return entry, "✅ HIT!"
            else:
                return entry, "❌ Miss"
    return None, None

def get_stats(log):
    completed = [e for e in log if e.get('actual') is not None]
    if not completed:
        return None
    recent = completed[-30:]
    total = len(recent)
    hit_50 = sum(1 for e in recent if e.get('hit_50', False))
    avg_error = sum(e['error'] for e in recent) / total
    return {
        'total': total,
        'hit_rate': hit_50 / total,
        'avg_error': avg_error
    }

# ============================================================================
# MAIN - MORNING MODE (9 AM)
# ============================================================================

def morning_predictions():
    """Make morning predictions without asking for input"""
    print("\n☀️  MORNING: Making midday predictions...")
    
    df3, df4 = load_data()
    if df3 is None:
        send_message("❌ Error loading data!")
        return
    
    log = load_log()
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    display_date = now.strftime('%A, %B %d, %Y')
    
    p3, c3, mp3 = train_and_predict(df3, 'pick3')
    p4, c4, mp4 = train_and_predict(df4, 'pick4')
    
    # Log predictions
    log.append({
        'date': date_str,
        'game': 'pick3',
        'draw_time': 'MIDDAY',
        'prediction': p3,
        'confidence': c3,
        'actual': None,
        'error': None,
        'hit_50': None,
        'hit_25': None,
        'exact': None
    })
    log.append({
        'date': date_str,
        'game': 'pick4',
        'draw_time': 'MIDDAY',
        'prediction': p4,
        'confidence': c4,
        'actual': None,
        'error': None,
        'hit_50': None,
        'hit_25': None,
        'exact': None
    })
    save_log(log)
    
    stats = get_stats(log)
    
    msg = f"""☀️ *NJ Lottery - MIDDAY Predictions*
{display_date}

🎲 *PICK 3*: `{str(p3).zfill(3)}` ({c3}%)
🎲 *PICK 4*: `{str(p4).zfill(4)}` ({c4}%)"""

    if stats:
        msg += f"""

📊 *Performance*
Hit Rate: {stats['hit_rate']*100:.1f}%
Avg Error: {stats['avg_error']:.1f}"""

    msg += "\n\n🕐 *I'll ask for results at 2 PM!*\n\nGood luck! 🍀"
    
    send_message(msg)
    print(f"✅ Sent morning predictions")

# ============================================================================
# MAIN - AFTERNOON MODE (2 PM)
# ============================================================================

def afternoon_results_and_predictions():
    """Ask for midday results, then predict evening"""
    print("\n🌆 AFTERNOON: Asking for midday results...")
    
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    display_date = now.strftime('%A, %B %d, %Y')
    
    log = load_log()
    
    # Ask for Pick 3 Midday
    send_message(f"🌆 *It's 2 PM! Time for today's results!*\n{display_date}")
    time.sleep(2)
    
    # Get Pick 3 Midday
    p3_midday = None
    attempts = 0
    while p3_midday is None and attempts < 3:
        reply = wait_for_reply("🎲 What was today's *PICK 3 MIDDAY* number?\n\n_(Enter 3 digits, e.g. 534)_")
        
        if reply is None:
            send_message("⏰ Skipping Pick 3 Midday - no reply received.")
            break
        
        p3_midday = validate_number(reply, 3)
        attempts += 1
        
        if p3_midday is None:
            send_message(f"❌ Invalid number '{reply}'. Please enter exactly 3 digits!\n_(Attempt {attempts}/3)_")
    
    # Get Pick 4 Midday
    p4_midday = None
    attempts = 0
    while p4_midday is None and attempts < 3:
        reply = wait_for_reply("🎲 What was today's *PICK 4 MIDDAY* number?\n\n_(Enter 4 digits, e.g. 3891)_")
        
        if reply is None:
            send_message("⏰ Skipping Pick 4 Midday - no reply received.")
            break
        
        p4_midday = validate_number(reply, 4)
        attempts += 1
        
        if p4_midday is None:
            send_message(f"❌ Invalid number '{reply}'. Please enter exactly 4 digits!\n_(Attempt {attempts}/3)_")
    
    # Save results to Excel
    if p3_midday:
        save_result(PICK3_FILE, date_str, 'MIDDAY', p3_midday)
    if p4_midday:
        save_result(PICK4_FILE, date_str, 'MIDDAY', p4_midday)
    
    # Update log with actual results
    comp_p3 = comp_p4 = None
    status_p3 = status_p4 = ""
    
    if p3_midday:
        _, status_p3 = update_log_with_actual(log, 'pick3', 'MIDDAY', p3_midday)
        for entry in reversed(log):
            if entry['date'] == date_str and entry['game'] == 'pick3' and entry['draw_time'] == 'MIDDAY':
                comp_p3 = entry
                break
    
    if p4_midday:
        _, status_p4 = update_log_with_actual(log, 'pick4', 'MIDDAY', p4_midday)
        for entry in reversed(log):
            if entry['date'] == date_str and entry['game'] == 'pick4' and entry['draw_time'] == 'MIDDAY':
                comp_p4 = entry
                break
    
    save_log(log)
    
    # Reload and train with new data
    df3, df4 = load_data()
    
    # Make evening predictions
    p3_eve, c3_eve, mp3 = train_and_predict(df3, 'pick3', p3_midday)
    p4_eve, c4_eve, mp4 = train_and_predict(df4, 'pick4', p4_midday)
    
    # Log evening predictions
    log.append({
        'date': date_str,
        'game': 'pick3',
        'draw_time': 'EVENING',
        'prediction': p3_eve,
        'confidence': c3_eve,
        'actual': None,
        'error': None,
        'hit_50': None,
        'hit_25': None,
        'exact': None
    })
    log.append({
        'date': date_str,
        'game': 'pick4',
        'draw_time': 'EVENING',
        'prediction': p4_eve,
        'confidence': c4_eve,
        'actual': None,
        'error': None,
        'hit_50': None,
        'hit_25': None,
        'exact': None
    })
    save_log(log)
    
    # Send notification
    msg = f"""🌆 *Midday Results + Evening Predictions*
{display_date}

📍 *MIDDAY RESULTS*
"""
    
    if comp_p3 and p3_midday:
        msg += f"\nPick 3: `{str(p3_midday).zfill(3)}`"
        msg += f"\nPredicted: {str(comp_p3['prediction']).zfill(3)}"
        msg += f"\nError: {comp_p3['error']} {status_p3}\n"
    
    if comp_p4 and p4_midday:
        msg += f"\nPick 4: `{str(p4_midday).zfill(4)}`"
        msg += f"\nPredicted: {str(comp_p4['prediction']).zfill(4)}"
        msg += f"\nError: {comp_p4['error']} {status_p4}\n"
    
    msg += f"""
🌙 *EVENING PREDICTIONS*

Pick 3: `{str(p3_eve).zfill(3)}` ({c3_eve}%)
Pick 4: `{str(p4_eve).zfill(4)}` ({c4_eve}%)

✨ Using midday correlation!
Good luck! 🍀"""
    
    send_message(msg)
    print("✅ Sent afternoon update")

# ============================================================================
# MAIN - MIDNIGHT MODE (12 AM)
# ============================================================================

def midnight_results():
    """Ask for evening results and send summary"""
    print("\n🌙 MIDNIGHT: Asking for evening results...")
    
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    display_date = now.strftime('%A, %B %d, %Y')
    
    log = load_log()
    
    send_message(f"🌙 *Evening Results Time!*\n{display_date}")
    time.sleep(2)
    
    # Get Pick 3 Evening
    p3_evening = None
    attempts = 0
    while p3_evening is None and attempts < 3:
        reply = wait_for_reply("🎲 What was today's *PICK 3 EVENING* number?\n\n_(Enter 3 digits, e.g. 789)_")
        
        if reply is None:
            send_message("⏰ Skipping Pick 3 Evening - no reply received.")
            break
        
        p3_evening = validate_number(reply, 3)
        attempts += 1
        
        if p3_evening is None:
            send_message(f"❌ Invalid! Please enter exactly 3 digits!\n_(Attempt {attempts}/3)_")
    
    # Get Pick 4 Evening
    p4_evening = None
    attempts = 0
    while p4_evening is None and attempts < 3:
        reply = wait_for_reply("🎲 What was today's *PICK 4 EVENING* number?\n\n_(Enter 4 digits, e.g. 4567)_")
        
        if reply is None:
            send_message("⏰ Skipping Pick 4 Evening - no reply received.")
            break
        
        p4_evening = validate_number(reply, 4)
        attempts += 1
        
        if p4_evening is None:
            send_message(f"❌ Invalid! Please enter exactly 4 digits!\n_(Attempt {attempts}/3)_")
    
    # Save to Excel
    if p3_evening:
        save_result(PICK3_FILE, date_str, 'EVENING', p3_evening)
    if p4_evening:
        save_result(PICK4_FILE, date_str, 'EVENING', p4_evening)
    
    # Update log
    status_p3 = status_p4 = ""
    comp_p3 = comp_p4 = None
    
    if p3_evening:
        _, status_p3 = update_log_with_actual(log, 'pick3', 'EVENING', p3_evening)
        for entry in reversed(log):
            if entry['date'] == date_str and entry['game'] == 'pick3' and entry['draw_time'] == 'EVENING':
                comp_p3 = entry
                break
    
    if p4_evening:
        _, status_p4 = update_log_with_actual(log, 'pick4', 'EVENING', p4_evening)
        for entry in reversed(log):
            if entry['date'] == date_str and entry['game'] == 'pick4' and entry['draw_time'] == 'EVENING':
                comp_p4 = entry
                break
    
    save_log(log)
    
    # Stats
    stats = get_stats(log)
    
    # Send summary
    msg = f"""🌙 *Daily Summary*
{display_date}

📍 *EVENING RESULTS*
"""
    
    if comp_p3 and p3_evening:
        msg += f"\nPick 3: `{str(p3_evening).zfill(3)}`"
        msg += f"\nPredicted: {str(comp_p3['prediction']).zfill(3)}"
        msg += f"\nError: {comp_p3['error']} {status_p3}\n"
    
    if comp_p4 and p4_evening:
        msg += f"\nPick 4: `{str(p4_evening).zfill(4)}`"
        msg += f"\nPredicted: {str(comp_p4['prediction']).zfill(4)}"
        msg += f"\nError: {comp_p4['error']} {status_p4}\n"
    
    if stats:
        msg += f"""
📊 *30-Day Performance*

Hit Rate (±50): {stats['hit_rate']*100:.1f}%
Avg Error: {stats['avg_error']:.1f}
Total Predictions: {stats['total']}"""
    
    msg += "\n\n🧠 System learning!\n😴 See you tomorrow!"
    
    send_message(msg)
    print("✅ Sent midnight summary")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🤖 NJ LOTTERY BOT INPUT SYSTEM")
    print("="*70)
    
    now = datetime.now()
    hour = now.hour
    force_test = os.environ.get('FORCE_TEST', 'false').lower() == 'true'
    
    print(f"Current hour (UTC): {hour}")
    
    if force_test:
        print("\n🧪 TEST MODE")
        send_message("🧪 *Bot Input System - Active!*\n\nI'll ask you for results automatically at:\n🕘 9:00 AM - Morning predictions\n🕐 2:00 PM - Midday results + Evening predictions\n🕛 12:00 AM - Evening results + Summary\n\n✅ All systems ready!")
        return
    
    # 9 AM EST = 14 UTC
    if 14 <= hour < 16:
        morning_predictions()
    
    # 2 PM EST = 19 UTC
    elif 19 <= hour < 21:
        afternoon_results_and_predictions()
    
    # 12 AM EST = 5 UTC
    elif 5 <= hour < 6:
        midnight_results()
    
    else:
        print(f"\nNo action for hour {hour}")
    
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
        send_message(f"❌ System error: {str(e)}")
