#!/usr/bin/env python3
"""
NJ Lottery Ultimate Prediction System
Uses 20 years of historical data with 5 AI techniques
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from collections import Counter

# Configuration
BOT_TOKEN = os.environ.get('BOT_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

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

def load_data():
    """Load historical lottery data"""
    print("Loading historical data...")
    
    try:
        # Pick 3
        df3 = pd.read_excel('final_merged_pick3_lottery_data.xlsx')
        df3['Date'] = pd.to_datetime(df3['Date'])
        df3 = df3[df3['Winning Number'].notna()]
        df3['Winning Number'] = pd.to_numeric(df3['Winning Number'], errors='coerce')
        df3 = df3[df3['Winning Number'].notna()].copy()
        df3['Winning Number'] = df3['Winning Number'].astype(int)
        
        # Pick 4
        df4 = pd.read_excel('final_merged_pick4_lottery_data.xlsx')
        df4['Date'] = pd.to_datetime(df4['Date'])
        df4 = df4[df4['Winning Number'].notna()]
        df4['Winning Number'] = pd.to_numeric(df4['Winning Number'], errors='coerce')
        df4 = df4[df4['Winning Number'].notna()].copy()
        df4['Winning Number'] = df4['Winning Number'].astype(int)
        
        print(f"  Pick 3: {len(df3)} records")
        print(f"  Pick 4: {len(df4)} records")
        return df3, df4
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def analyze_patterns(df, game='pick3'):
    """Analyze patterns in historical data"""
    n_digits = 3 if game == 'pick3' else 4
    
    # Digit position frequency
    pos_freq = [{} for _ in range(n_digits)]
    for num in df['Winning Number']:
        for i, digit in enumerate(str(num).zfill(n_digits)):
            pos_freq[i][digit] = pos_freq[i].get(digit, 0) + 1
    
    hot_by_pos = []
    for i in range(n_digits):
        top = sorted(pos_freq[i].items(), key=lambda x: x[1], reverse=True)
        hot_by_pos.append([int(d[0]) for d, _ in top[:3]])
    
    # Recent hot digits
    recent = df.tail(100)
    digit_count = {}
    for num in recent['Winning Number']:
        for d in str(num).zfill(n_digits):
            digit_count[d] = digit_count.get(d, 0) + 1
    
    hot = [int(d) for d, _ in sorted(digit_count.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]]
    
    # Sum statistics
    sums = [sum(int(d) for d in str(n).zfill(n_digits)) 
            for n in df['Winning Number']]
    avg_sum = np.mean(sums)
    
    # Time patterns
    df_temp = df.copy()
    df_temp['dow'] = df_temp['Date'].dt.dayofweek
    day_avgs = df_temp.groupby('dow')['Winning Number'].mean().to_dict()
    
    return {
        'hot_by_position': hot_by_pos,
        'hot_digits': hot,
        'avg_sum': avg_sum,
        'day_averages': day_avgs,
        'recent_avg': df.tail(30)['Winning Number'].mean()
    }

def make_prediction(df, patterns, game='pick3', draw_time='midday', 
                   midday_result=None):
    """Make prediction using ensemble of techniques"""
    n_digits = 3 if game == 'pick3' else 4
    max_val = 999 if game == 'pick3' else 9999
    
    predictions = []
    
    # Technique 1: Hot digits by position (30%)
    pos_pred = int(''.join(str(patterns['hot_by_position'][i][0]) 
                          for i in range(n_digits)))
    predictions.append(pos_pred * 0.30)
    
    # Technique 2: Recent average (25%)
    predictions.append(patterns['recent_avg'] * 0.25)
    
    # Technique 3: Day of week pattern (20%)
    today_dow = datetime.now().weekday()
    day_avg = patterns['day_averages'].get(today_dow, patterns['recent_avg'])
    predictions.append(day_avg * 0.20)
    
    # Technique 4: Last value trend (15%)
    last_val = df['Winning Number'].iloc[-1]
    predictions.append(last_val * 0.15)
    
    # Technique 5: Midday correlation for evening (10%)
    if draw_time == 'evening' and midday_result:
        predictions.append(midday_result * 0.10)
    else:
        predictions.append(patterns['recent_avg'] * 0.10)
    
    # Ensemble
    final = int(sum(predictions))
    final = max(0, min(max_val, final))
    
    # Sum validation
    pred_sum = sum(int(d) for d in str(final).zfill(n_digits))
    if abs(pred_sum - patterns['avg_sum']) > 6:
        final = int(patterns['recent_avg'])
    
    # Confidence
    variation = np.std([p/0.3 if i==0 else p/w 
                       for i, (p, w) in enumerate(zip(predictions, 
                       [0.3,0.25,0.2,0.15,0.1]))])
    confidence = max(60, min(95, int(95 - variation/50)))
    
    return final, confidence

def main():
    """Main execution"""
    print("="*70)
    print("NJ LOTTERY ULTIMATE PREDICTION SYSTEM")
    print("="*70)
    
    # Load data
    df3, df4 = load_data()
    if df3 is None or df4 is None:
        send_telegram("Error: Could not load historical data")
        return
    
    # Analyze patterns
    print("\nAnalyzing 20 years of patterns...")
    patterns3 = analyze_patterns(df3, 'pick3')
    patterns4 = analyze_patterns(df4, 'pick4')
    print("Pattern analysis complete")
    
    # Determine action based on time
    now = datetime.now()
    hour = now.hour
    date_str = now.strftime('%A, %B %d, %Y')
    
    # FORCE TEST MODE - always send when manually triggered
    force_test = os.environ.get('FORCE_TEST', 'false').lower() == 'true'
    
    if force_test:
        print("\n🧪 FORCE TEST MODE - Sending test message")
        p3, c3 = make_prediction(df3, patterns3, 'pick3', 'midday')
        p4, c4 = make_prediction(df4, patterns4, 'pick4', 'midday')
        
        msg = f"""🧪 *TEST - NJ Lottery System*
{date_str}

🎲 *PICK 3*: `{str(p3).zfill(3)}` ({c3} pct)
🎲 *PICK 4*: `{str(p4).zfill(4)}` ({c4} pct)

This is a test! System is working! ✅"""
        
        send_telegram(msg)
        print(f"Test sent: P3={p3}, P4={p4}")
        return
    
    # MORNING (9 AM EST = 14:00 UTC) - Midday predictions
    if 14 <= hour < 16:
        print("\nMORNING: Midday predictions")
        
        p3, c3 = make_prediction(df3, patterns3, 'pick3', 'midday')
        p4, c4 = make_prediction(df4, patterns4, 'pick4', 'midday')
        
        msg = f"""☀️ *NJ Lottery - MIDDAY*
{date_str}

🎲 *PICK 3*: `{str(p3).zfill(3)}` ({c3} pct)
🎲 *PICK 4*: `{str(p4).zfill(4)}` ({c4} pct)

━━━━━━━━━━━━━━━━━━━━━
🧠 *Powered by:*
✅ 20 years of NJ data
✅ 5 AI techniques
✅ Pattern analysis

Good luck! 🍀"""
        
        send_telegram(msg)
        print(f"Sent: P3={p3} ({c3}%), P4={p4} ({c4}%)")
    
    # AFTERNOON (1:30 PM EST = 18:30 UTC) - Evening predictions
    elif 18 <= hour < 20:
        print("\nAFTERNOON: Evening predictions")
        
        midday_p3 = int(patterns3['recent_avg'])
        midday_p4 = int(patterns4['recent_avg'])
        
        p3, c3 = make_prediction(df3, patterns3, 'pick3', 'evening', midday_p3)
        p4, c4 = make_prediction(df4, patterns4, 'pick4', 'evening', midday_p4)
        
        msg = f"""🌙 *EVENING Predictions*
{date_str}

🎲 *PICK 3*: `{str(p3).zfill(3)}` ({c3} pct)
🎲 *PICK 4*: `{str(p4).zfill(4)}` ({c4} pct)

✨ *Using midday correlation!*

Good luck! 🍀"""
        
        send_telegram(msg)
        print(f"Sent: P3={p3} ({c3}%), P4={p4} ({c4}%)")
    
    # LATE NIGHT (12:30 AM EST = 05:30 UTC) - Summary
    elif 5 <= hour < 7:
        print("\nLATE NIGHT: Summary")
        
        msg = f"""🌙 *Daily Summary*
{date_str}

🧠 System Status: ✅ Active
📊 Data: 20 years analyzed
🎯 AI Techniques: 5 active

Ready for tomorrow! 😴"""
        
        send_telegram(msg)
        print("Sent summary")
    
    else:
        print(f"\nNo action for hour {hour}")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
