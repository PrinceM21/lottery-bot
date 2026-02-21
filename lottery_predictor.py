#!/usr/bin/env python3
"""
🎓 ACADEMICALLY CORRECT NJ LOTTERY PREDICTION SYSTEM (V2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V2 fixes remaining issues from V1:
✅ Removes remaining leakage (target-derived stats shifted or excluded)
✅ Correct historic Midday→Evening feature join (per-date, not constant)
✅ Beam-search Top-K (fast; no enumerating 10,000 numbers per row)
✅ XGBoost forced 10-class multiclass (stable log-loss)
✅ Proper chronological train/test split + scaler fit on train only
✅ Proper ordering by Date + Draw Time everywhere (including saving)
✅ Confidence is interpretable (Top1 prob share among TopK)

Notes:
- If the lottery is fair IID, models should not beat strong baselines.
- This script provides a reproducible evaluation pipeline and a "next draw" prediction.
"""

import os
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss

import xgboost as xgb

warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

PICK3_FILE = "final_merged_pick3_lottery_data.xlsx"
PICK4_FILE = "final_merged_pick4_lottery_data.xlsx"

SEED = 42
np.random.seed(SEED)

DRAW_ORDER = {"MIDDAY": 0, "EVENING": 1}

LAGS = [1, 2, 3, 5, 7, 10]
ROLL_WINDOWS = [5, 10, 20, 30]
MIN_HISTORY_ROWS = 30

TOPK = 5
BEAM_WIDTH = 50  # speed vs exactness; 50 is plenty for Top5


# ==============================
# TELEGRAM (optional)
# ==============================

def send_message(msg: str, parse_mode: str = "Markdown"):
    if not BOT_TOKEN or not CHAT_ID:
        print(f"MSG: {msg}")
        return None
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": parse_mode},
            timeout=10,
        )
        return r.json()
    except Exception as e:
        print(f"Telegram error: {e}")
        return None


def wait_for_reply(prompt: str, timeout: int = 3600) -> Optional[str]:
    """
    If BOT_TOKEN / CHAT_ID are missing, fallback to console input.
    """
    print(f"\n📱 Asking: {prompt}")
    send_message(prompt)

    if not BOT_TOKEN or not CHAT_ID:
        try:
            return input("Reply: ").strip()
        except Exception:
            return None

    # Telegram polling loop
    try:
        updates = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params={"timeout": 30, "offset": None},
            timeout=35,
        ).json()
    except Exception:
        updates = None

    last_id = 0
    if updates and updates.get("result"):
        for u in updates["result"]:
            last_id = max(last_id, u["update_id"])

    start = time.time()
    reminder_sent = False

    while time.time() - start < timeout:
        elapsed = time.time() - start
        time.sleep(5)

        if elapsed >= 1800 and not reminder_sent:
            send_message("⏰ *Reminder!*\n\n" + prompt + "\n\n_30 minutes left!_")
            reminder_sent = True

        try:
            new_updates = requests.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                params={"timeout": 30, "offset": last_id + 1},
                timeout=35,
            ).json()
        except Exception:
            new_updates = None

        if new_updates and new_updates.get("result"):
            for update in new_updates["result"]:
                last_id = update["update_id"]
                if "message" in update:
                    msg = update["message"]
                    if str(msg.get("chat", {}).get("id", "")) == str(CHAT_ID):
                        return msg.get("text", "").strip()

    send_message("⏰ No reply in 1 hour. Skipping.")
    return None


def validate_number(text: Optional[str], digits: int) -> Optional[int]:
    if not text:
        return None
    t = text.strip().replace(" ", "")
    if not t.isdigit():
        return None
    if len(t) > digits:
        return None
    return int(t.zfill(digits))


# ==============================
# DATA LOADING / SAVING
# ==============================

def _sort_draws(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["draw_order"] = df["Draw Time"].map(DRAW_ORDER).fillna(99).astype(int)
    df = df.sort_values(["Date", "draw_order"]).drop(columns=["draw_order"]).reset_index(drop=True)
    return df


def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    print("📂 Loading data...")
    try:
        df3 = pd.read_excel(PICK3_FILE)
        df4 = pd.read_excel(PICK4_FILE)

        for df in (df3, df4):
            df["Date"] = pd.to_datetime(df["Date"])
            df["Winning Number"] = pd.to_numeric(df["Winning Number"], errors="coerce")

        df3 = df3[df3["Winning Number"].notna()].copy()
        df4 = df4[df4["Winning Number"].notna()].copy()
        df3["Winning Number"] = df3["Winning Number"].astype(int)
        df4["Winning Number"] = df4["Winning Number"].astype(int)

        df3 = _sort_draws(df3)
        df4 = _sort_draws(df4)

        print(f"  ✅ Pick 3: {len(df3):,} records")
        print(f"  ✅ Pick 4: {len(df4):,} records")
        return df3, df4
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None


def save_result(filepath: str, date: str, draw_time: str, number: int) -> bool:
    try:
        df = pd.read_excel(filepath)
        df["Date"] = pd.to_datetime(df["Date"])
        date_dt = pd.to_datetime(date)

        if ((df["Date"] == date_dt) & (df["Draw Time"] == draw_time)).any():
            print("  ⚠️  Already exists")
            return False

        new_row = pd.DataFrame([{"Date": date_dt, "Draw Time": draw_time, "Winning Number": int(number)}])
        df = pd.concat([df, new_row], ignore_index=True)
        df = _sort_draws(df)
        df.to_excel(filepath, index=False)

        print(f"  ✅ Saved: {draw_time} = {number}")
        return True
    except Exception as e:
        print(f"  ❌ Error saving: {e}")
        return False


# ==============================
# FEATURE ENGINEERING (leakage-safe)
# ==============================

def add_digits(df: pd.DataFrame, n_digits: int) -> pd.DataFrame:
    df = df.copy()
    s = df["Winning Number"].astype(int).astype(str).str.zfill(n_digits)
    for i in range(n_digits):
        df[f"Digit{i+1}"] = s.str[i].astype(int)
    return df


def add_midday_features(df: pd.DataFrame, n_digits: int) -> pd.DataFrame:
    """
    Historic per-date Midday→Evening join:
      evening rows receive that date's midday digits.
    """
    df = df.copy()
    mid = df[df["Draw Time"].eq("MIDDAY")][["Date"] + [f"Digit{i+1}" for i in range(n_digits)]].copy()
    mid = mid.rename(columns={f"Digit{i+1}": f"Midday_Digit{i+1}" for i in range(n_digits)})
    df = df.merge(mid, on="Date", how="left")
    df["Midday_Sum"] = df[[f"Midday_Digit{i+1}" for i in range(n_digits)]].sum(axis=1, min_count=1)
    return df


def create_features(df: pd.DataFrame, game: str) -> pd.DataFrame:
    """
    Rule: any outcome-derived predictor must come from the past (shifted).
    """
    df = df.copy()
    n_digits = 3 if game == "pick3" else 4

    df = _sort_draws(df)
    df = add_digits(df, n_digits)
    df = add_midday_features(df, n_digits)

    digit_cols = [f"Digit{i+1}" for i in range(n_digits)]

    # Time features (safe)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Outcome-derived stats (compute but DO NOT use current row values as features)
    df["Sum"] = df[digit_cols].sum(axis=1)
    df["NumEven"] = (df[digit_cols] % 2 == 0).sum(axis=1)
    df["NumUnique"] = df[digit_cols].nunique(axis=1)

    # Shifted stats are safe predictors
    df["Sum_lag1"] = df["Sum"].shift(1)
    df["NumEven_lag1"] = df["NumEven"].shift(1)
    df["NumUnique_lag1"] = df["NumUnique"].shift(1)

    # Lag digit features
    for lag in LAGS:
        df[f"Lag_Sum_{lag}"] = df["Sum"].shift(lag)
        for i in range(n_digits):
            df[f"Lag_Digit{i+1}_{lag}"] = df[f"Digit{i+1}"].shift(lag)

    # Rolling on shifted sum
    for w in ROLL_WINDOWS:
        df[f"Roll_SumMean_{w}"] = df["Sum"].shift(1).rolling(w, min_periods=1).mean()
        df[f"Roll_SumStd_{w}"] = df["Sum"].shift(1).rolling(w, min_periods=1).std()

    # Drop early rows + NaNs from lagging
    df = df.iloc[MIN_HISTORY_ROWS:].copy()
    df = df.dropna().reset_index(drop=True)
    return df


# ==============================
# MODELING: Digit-wise multiclass
# ==============================

def train_xgb_digit_models(
    X_train: np.ndarray,
    y_train_digits: np.ndarray,
    X_test: np.ndarray,
    y_test_digits: np.ndarray,
    n_digits: int,
) -> Tuple[List[xgb.XGBClassifier], List[np.ndarray], MinMaxScaler, Dict]:
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models: List[xgb.XGBClassifier] = []
    probs_test: List[np.ndarray] = []
    labels = np.arange(10)

    per_digit = []
    for pos in range(n_digits):
        ytr = y_train_digits[:, pos]
        yte = y_test_digits[:, pos]

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=10,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=SEED,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        model.fit(X_train_s, ytr)

        p = model.predict_proba(X_test_s)
        # Ensure [n,10]
        if p.shape[1] != 10:
            full = np.full((p.shape[0], 10), 1e-12, dtype=float)
            present = model.classes_.astype(int)
            full[:, present] = p
            p = full

        pred = np.argmax(p, axis=1)
        acc = accuracy_score(yte, pred)
        ll = log_loss(yte, p, labels=labels)

        per_digit.append({"digit_pos": pos + 1, "accuracy": float(acc), "logloss": float(ll)})
        models.append(model)
        probs_test.append(p)

    metrics = {
        "per_digit": per_digit,
        "mean_digit_accuracy": float(np.mean([m["accuracy"] for m in per_digit])),
        "mean_digit_logloss": float(np.mean([m["logloss"] for m in per_digit])),
    }
    return models, probs_test, scaler, metrics


# ==============================
# TOP-K: Beam Search
# ==============================

def beam_topk(digit_probs: List[np.ndarray], k: int = TOPK, beam_width: int = BEAM_WIDTH):
    beams = [("", 1.0)]
    for probs in digit_probs:
        top_d = np.argsort(probs)[::-1][:beam_width]
        new = []
        for prefix, p_prefix in beams:
            for d in top_d:
                new.append((prefix + str(int(d)), p_prefix * float(probs[int(d)])))
        new.sort(key=lambda x: x[1], reverse=True)
        beams = new[:beam_width]
    beams.sort(key=lambda x: x[1], reverse=True)
    return beams[:k]


def rel_confidence(topk):
    if not topk:
        return 0.0
    s = sum(p for _, p in topk)
    return (topk[0][1] / s) if s > 0 else 0.0


def evaluate_topk_rates(y_test_digits: np.ndarray, probs_test: List[np.ndarray], n_digits: int, k: int = TOPK) -> Dict:
    n = len(y_test_digits)
    exact = 0
    hitk = 0
    for i in range(n):
        actual = "".join(str(int(d)) for d in y_test_digits[i])
        digit_probs_i = [probs_test[pos][i] for pos in range(n_digits)]
        topk_list = beam_topk(digit_probs_i, k=k)
        preds = [s for s, _ in topk_list]
        if preds and preds[0] == actual:
            exact += 1
        if actual in preds:
            hitk += 1
    return {
        "k": k,
        "n_test": n,
        "exact_match_rate": exact / n if n else 0.0,
        "topk_hit_rate": hitk / n if n else 0.0,
    }


# ==============================
# TRAIN + PREDICT
# ==============================

def train_and_predict(df: pd.DataFrame, game: str, target_draw: str, todays_midday_number: Optional[int] = None) -> Dict:
    assert target_draw in ("MIDDAY", "EVENING")
    n_digits = 3 if game == "pick3" else 4

    df_feat = create_features(df, game)

    # Subset to target draw type
    df_target = df_feat[df_feat["Draw Time"].eq(target_draw)].copy().reset_index(drop=True)

    # For MIDDAY prediction, midday features are unknown → drop them
    midday_cols = [f"Midday_Digit{i+1}" for i in range(n_digits)] + ["Midday_Sum"]
    if target_draw == "MIDDAY":
        for c in midday_cols:
            if c in df_target.columns:
                df_target = df_target.drop(columns=[c])

    # Targets (current digits)
    y_digits = df_target[[f"Digit{i+1}" for i in range(n_digits)]].values.astype(int)

    # Exclude columns that would leak current outcome
    exclude = set(
        ["Date", "Draw Time", "Winning Number"]
        + [f"Digit{i+1}" for i in range(n_digits)]
        + ["Sum", "NumEven", "NumUnique"]  # unshifted outcome stats excluded
    )
    feature_cols = [c for c in df_target.columns if c not in exclude]
    X = df_target[feature_cols].values.astype(float)

    # Chronological split 80/20
    split = int(0.8 * len(df_target))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_digits[:split], y_digits[split:]

    models, probs_test, scaler, metrics = train_xgb_digit_models(X_train, y_train, X_test, y_test, n_digits)
    topk_metrics = evaluate_topk_rates(y_test, probs_test, n_digits, k=TOPK)

    # Next-draw prediction context: last row for that draw type
    row_last = df_target.tail(1).copy()

    # If predicting EVENING and today's midday is known, overwrite midday feature cols just for prediction row
    if target_draw == "EVENING" and todays_midday_number is not None:
        md = [int(d) for d in str(int(todays_midday_number)).zfill(n_digits)]
        for i in range(n_digits):
            col = f"Midday_Digit{i+1}"
            if col in row_last.columns:
                row_last[col] = md[i]
        if "Midday_Sum" in row_last.columns:
            row_last["Midday_Sum"] = sum(md)

    X_last = row_last[feature_cols].values.astype(float)
    X_last_s = scaler.transform(X_last)

    next_digit_probs = []
    for m in models:
        p = m.predict_proba(X_last_s)[0]
        if p.shape[0] != 10:
            full = np.full((10,), 1e-12, dtype=float)
            present = m.classes_.astype(int)
            full[present] = p
            p = full
        next_digit_probs.append(p)

    topk_list = beam_topk(next_digit_probs, k=TOPK, beam_width=BEAM_WIDTH)
    conf = rel_confidence(topk_list)

    return {
        "game": game,
        "target_draw": target_draw,
        "n_digits": n_digits,
        "feature_count": len(feature_cols),
        "train_size": int(split),
        "test_size": int(len(df_target) - split),
        "metrics": metrics,
        "topk_metrics": topk_metrics,
        "prediction_top1": topk_list[0][0] if topk_list else "0" * n_digits,
        "prediction_topk": topk_list,
        "relative_confidence_top1_in_topk": float(conf),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# ==============================
# WORKFLOWS
# ==============================

def morning_workflow():
    """10 AM: enter last night's EVENING results, predict today's MIDDAY."""
    print("\n☀️ MORNING WORKFLOW (V2)")
    print("=" * 80)

    yesterday = datetime.now() - timedelta(days=1)
    y_str = yesterday.strftime("%Y-%m-%d")
    y_display = yesterday.strftime("%A, %B %d, %Y")

    today = datetime.now()
    t_display = today.strftime("%A, %B %d, %Y")

    send_message("☀️ *Good morning!*\n\nLet's enter last night's results!")

    for game, digits, path in [("pick3", 3, PICK3_FILE), ("pick4", 4, PICK4_FILE)]:
        num = None
        tries = 0
        while num is None and tries < 3:
            reply = wait_for_reply(f"🌙 *{game.upper()} EVENING* from last night?\n({y_display})\n\n_({digits} digits)_")
            num = validate_number(reply, digits)
            tries += 1
            if num is None:
                send_message(f"❌ Invalid! {digits} digits needed. ({tries}/3)")
        if num is not None:
            save_result(path, y_str, "EVENING", num)

    df3, df4 = load_data()
    if df3 is None:
        send_message("❌ Error loading data")
        return

    r3 = train_and_predict(df3, "pick3", target_draw="MIDDAY")
    r4 = train_and_predict(df4, "pick4", target_draw="MIDDAY")

    msg = f"☀️ *TODAY MIDDAY Predictions*\n{t_display}\n\n"
    msg += f"🎲 *PICK 3*: `{r3['prediction_top1']}` (rel.conf: {r3['relative_confidence_top1_in_topk']*100:.1f}%)\n"
    msg += f"   Top{TOPK}: {', '.join(s for s,_ in r3['prediction_topk'])}\n"
    msg += f"   Mean digit acc (test): {r3['metrics']['mean_digit_accuracy']*100:.1f}% | logloss: {r3['metrics']['mean_digit_logloss']:.3f}\n\n"
    msg += f"🎲 *PICK 4*: `{r4['prediction_top1']}` (rel.conf: {r4['relative_confidence_top1_in_topk']*100:.1f}%)\n"
    msg += f"   Top{TOPK}: {', '.join(s for s,_ in r4['prediction_topk'])}\n"
    msg += f"   Mean digit acc (test): {r4['metrics']['mean_digit_accuracy']*100:.1f}% | logloss: {r4['metrics']['mean_digit_logloss']:.3f}\n\n"
    msg += "🎓 *V2 leakage-safe evaluation included.*"
    send_message(msg)
    print("✅ Morning workflow complete!")


def afternoon_workflow():
    """2 PM: enter today's MIDDAY results, predict today's EVENING."""
    print("\n🌆 AFTERNOON WORKFLOW (V2)")
    print("=" * 80)

    today = datetime.now()
    t_str = today.strftime("%Y-%m-%d")
    t_display = today.strftime("%A, %B %d, %Y")

    send_message(f"🌆 *Midday Results Time!*\n{t_display}")

    midday_results = {}
    for game, digits, path in [("pick3", 3, PICK3_FILE), ("pick4", 4, PICK4_FILE)]:
        num = None
        tries = 0
        while num is None and tries < 3:
            reply = wait_for_reply(f"🎲 *{game.upper()} MIDDAY* result?\n\n_({digits} digits)_")
            num = validate_number(reply, digits)
            tries += 1
            if num is None:
                send_message(f"❌ Invalid! {digits} digits needed. ({tries}/3)")
        if num is not None:
            midday_results[game] = num
            save_result(path, t_str, "MIDDAY", num)

    df3, df4 = load_data()
    if df3 is None:
        send_message("❌ Error loading data")
        return

    r3 = train_and_predict(df3, "pick3", target_draw="EVENING", todays_midday_number=midday_results.get("pick3"))
    r4 = train_and_predict(df4, "pick4", target_draw="EVENING", todays_midday_number=midday_results.get("pick4"))

    msg = f"🌙 *TODAY EVENING Predictions*\n{t_display}\n\n"
    msg += f"🎲 *PICK 3*: `{r3['prediction_top1']}` (rel.conf: {r3['relative_confidence_top1_in_topk']*100:.1f}%)\n"
    msg += f"   Top{TOPK}: {', '.join(s for s,_ in r3['prediction_topk'])}\n"
    msg += f"   Top{TOPK} hit rate (test): {r3['topk_metrics']['topk_hit_rate']*100:.2f}%\n\n"
    msg += f"🎲 *PICK 4*: `{r4['prediction_top1']}` (rel.conf: {r4['relative_confidence_top1_in_topk']*100:.1f}%)\n"
    msg += f"   Top{TOPK}: {', '.join(s for s,_ in r4['prediction_topk'])}\n"
    msg += f"   Top{TOPK} hit rate (test): {r4['topk_metrics']['topk_hit_rate']*100:.2f}%\n\n"
    msg += "✨ Midday used as per-date feature (historic join).\n🎓 V2 evaluation included."
    send_message(msg)
    print("✅ Afternoon workflow complete!")


def main():
    print("=" * 80)
    print("🎓 ACADEMICALLY CORRECT LOTTERY PREDICTION SYSTEM (V2)")
    print("=" * 80)

    hour = datetime.now().hour
    force_test = os.environ.get("FORCE_TEST", "false").lower() == "true"

    if force_test:
        df3, df4 = load_data()
        if df3 is None:
            return
        for df, game in [(df3, "pick3"), (df4, "pick4")]:
            for draw in ("MIDDAY", "EVENING"):
                res = train_and_predict(df, game, target_draw=draw)
                print(f"\n[{game.upper()} {draw}] top1={res['prediction_top1']} rel_conf={res['relative_confidence_top1_in_topk']*100:.1f}%")
                print(f"  mean_digit_acc={res['metrics']['mean_digit_accuracy']*100:.2f}% mean_digit_logloss={res['metrics']['mean_digit_logloss']:.3f}")
                print(f"  top{TOPK}_hit_rate={res['topk_metrics']['topk_hit_rate']*100:.2f}% exact={res['topk_metrics']['exact_match_rate']*100:.2f}%")
        return

    # Scheduling (UTC-based example like your original)
    # 10 AM EST ≈ 15 UTC
    if 15 <= hour < 17:
        morning_workflow()
    # 2 PM EST ≈ 19 UTC
    elif 19 <= hour < 21:
        afternoon_workflow()
    else:
        print(f"No action for hour {hour}")

    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        send_message(f"❌ Error: {str(e)}")
        raise
