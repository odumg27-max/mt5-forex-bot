import os, time, math, json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from loguru import logger
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import indicators as ind  # <-- our pure-Python indicators (no pandas_ta)

# ---------- Config ----------
load_dotenv()

SYMBOL = os.getenv("SYMBOL", "EURUSD")
TF_H1 = os.getenv("TF_H1", "H1")
TF_M15 = os.getenv("TF_M15", "M15")
TF_M1 = os.getenv("TF_M1", "M1")

LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() == "true"
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "3"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.01"))
RR = float(os.getenv("RR", "3.0"))
ATR_MULT = float(os.getenv("ATR_MULT", "1.5"))
ATR_TRAIL_MULT = float(os.getenv("ATR_TRAIL_MULT", "1.0"))
DAILY_LOSS_STOP_USD = float(os.getenv("DAILY_LOSS_STOP_USD", "100"))
MAX_SPREAD_POINTS = float(os.getenv("MAX_SPREAD_POINTS", "25"))
MAGIC = int(os.getenv("MAGIC", "20250910"))

MODEL_ON = os.getenv("MODEL_ON", "false").lower() == "true"
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
LEDGER_CSV = os.getenv("LEDGER_CSV", "logs/ledger.csv")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH")

logger.remove()
logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)
logger.add("logs/run.log", level=LOG_LEVEL, rotation="1 MB", retention=5)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
}

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return ind.add_indicators(df)

# ---------- MT5 Init ----------
def init_mt5():
    if MT5_TERMINAL_PATH:
        mt5.initialize(MT5_TERMINAL_PATH, login=int(MT5_LOGIN) if MT5_LOGIN else None,
                       password=MT5_PASSWORD, server=MT5_SERVER)
    else:
        mt5.initialize(login=int(MT5_LOGIN) if MT5_LOGIN else None,
                       password=MT5_PASSWORD, server=MT5_SERVER)
    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        raise SystemExit(1)
    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"Failed to select symbol {SYMBOL}")
        raise SystemExit(1)
    logger.info("MT5 initialized.")

# ---------- Data ----------
def rates_df(symbol, tf_str, bars=1000):
    tf = TIMEFRAME_MAP[tf_str]
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})

# ---------- Top-down filters ----------
def h1_trend(df_h1: pd.DataFrame):
    df = add_indicators(df_h1).dropna()
    if df.empty: return None
    last = df.iloc[-1]
    trend = "up" if (last.ema50 > last.ema200 and last.adx14 > 18) else \
            "down" if (last.ema50 < last.ema200 and last.adx14 > 18) else "chop"
    return trend

def m15_bos(df_m15: pd.DataFrame, direction: str) -> bool:
    df = add_indicators(df_m15).dropna()
    if df.empty: return False
    prior_high = df.high.rolling(20).max().shift(1)
    prior_low  = df.low.rolling(20).min().shift(1)
    c = df.close.iloc[-1]
    if direction == "up":
        return bool(c > prior_high.iloc[-1])
    if direction == "down":
        return bool(c < prior_low.iloc[-1])
    return False

# ---------- Entry logic on M1 ----------
def m1_signal(df_m1: pd.DataFrame, direction: str):
    df = add_indicators(df_m1).dropna()
    if df.empty or direction not in ("up","down"): 
        return None
    last, prev = df.iloc[-1], df.iloc[-2]

    ema_align_up   = last.ema9 > last.ema21 > last.ema50
    ema_align_down = last.ema9 < last.ema21 < last.ema50
    macd_up   = last.macdh > 0 and last.macdh > prev.macdh
    macd_down = last.macdh < 0 and last.macdh < prev.macdh
    st_up   = last.st_dir == 1
    st_down = last.st_dir == -1
    adx_ok  = last.adx14 > 18
    vol_ok  = last.volume > (last.vol_ma20 * 1.05)

    breakout_up   = last.close > last.don_high and prev.close <= prev.don_high
    breakout_down = last.close < last.don_low  and prev.close >= prev.don_low

    if direction == "up" and ema_align_up and macd_up and st_up and adx_ok and vol_ok and breakout_up:
        return {"side":"buy"}
    if direction == "down" and ema_align_down and macd_down and st_down and adx_ok and vol_ok and breakout_down:
        return {"side":"sell"}
    return None

# ---------- Risk, sizing, SL/TP ----------
def symbol_meta(symbol):
    info = mt5.symbol_info(symbol)
    if info is None: raise RuntimeError(f"symbol_info None for {symbol}")
    return info

def price_tick():
    return mt5.symbol_info_tick(SYMBOL)

def spread_points():
    t = price_tick()
    info = symbol_meta(SYMBOL)
    return (t.ask - t.bid) / info.point

def atr_sl_tp(df_m1: pd.DataFrame, side: str):
    df = add_indicators(df_m1).dropna()
    last = df.iloc[-1]
    atr = last.atr14
    t = price_tick()
    if side == "buy":
        entry = t.ask
        sl = entry - ATR_MULT * atr
        tp = entry + RR * (entry - sl)
    else:
        entry = t.bid
        sl = entry + ATR_MULT * atr
        tp = entry - RR * (sl - entry)
    return entry, sl, tp, atr

def lot_size(entry, sl):
    acc = mt5.account_info()
    bal = float(acc.balance) if acc else 0.0
    risk_amount = bal * RISK_PCT
    info = symbol_meta(SYMBOL)
    val_per_point_1lot = info.trade_tick_value / info.trade_tick_size
    sl_points = abs(entry - sl) / info.point
    risk_per_lot = sl_points * val_per_point_1lot
    if risk_per_lot <= 0: return info.volume_min
    lots_raw = risk_amount / risk_per_lot
    lots = max(info.volume_min, min(lots_raw, info.volume_max))
    step = info.volume_step
    lots = math.floor(lots / step) * step
    return max(lots, info.volume_min)

# ---------- Orders ----------
def send_order(side, lots, sl, tp):
    t = price_tick()
    price = t.ask if side=="buy" else t.bid
    order_type = mt5.ORDER_TYPE_BUY if side=="buy" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "type": order_type,
        "volume": lots,
        "price": price,
        "sl": sl, "tp": tp,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "m1_topdown_scalp",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    if not LIVE_MODE:
        logger.info(f"[PAPER] {side} {lots} {SYMBOL} @ {price:.5f} SL {sl:.5f} TP {tp:.5f}")
        return {"retcode":0, "order":0, "deal":0}
    res = mt5.order_send(request)
    logger.info(f"order_send: {res}")
    if res is None:
        logger.error(f"order_send None: {mt5.last_error()}")
    return res

def trail_update(atr_current):
    pos = mt5.positions_get(symbol=SYMBOL)
    if pos is None: return
    info = symbol_meta(SYMBOL)
    for p in pos:
        if p.magic != MAGIC: continue
        ticket = p.ticket
        if p.type == mt5.POSITION_TYPE_BUY:
            new_sl = mt5.symbol_info_tick(SYMBOL).bid - ATR_TRAIL_MULT * atr_current
            if new_sl > p.sl and LIVE_MODE:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP, "position": ticket,
                    "sl": new_sl, "tp": p.tp
                })
        if p.type == mt5.POSITION_TYPE_SELL:
            new_sl = mt5.symbol_info_tick(SYMBOL).ask + ATR_TRAIL_MULT * atr_current
            if (p.sl == 0 or new_sl < p.sl) and LIVE_MODE:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP, "position": ticket,
                    "sl": new_sl, "tp": p.tp
                })

# ---------- Kill switch ----------
def pnl_today_usd():
    now = datetime.utcnow()
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    deals = mt5.history_deals_get(day_start, now)
    if deals is None: return 0.0
    return float(sum(d.profit for d in deals if d.magic == MAGIC and d.symbol == SYMBOL))

# ---------- Ledger ----------
def log_fill(side, entry, sl, tp, lots, note=""):
    os.makedirs(os.path.dirname(LEDGER_CSV), exist_ok=True)
    row = {
        "ts": datetime.utcnow().isoformat(),
        "symbol": SYMBOL, "side": side, "entry": entry, "sl": sl, "tp": tp,
        "lots": lots, "note": note
    }
    write_header = not os.path.exists(LEDGER_CSV)
    pd.DataFrame([row]).to_csv(LEDGER_CSV, mode="a", header=write_header, index=False)

# ---------- Optional ML gate ----------
def ml_ok(features: pd.Series) -> bool:
    if not MODEL_ON: return True
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        proba = model.predict_proba([features.values])[0][1]
        return proba >= float(os.getenv("PROB_THRESHOLD","0.55"))
    except Exception as e:
        logger.warning(f"ML gate skipped: {e}")
        return True

# ---------- Main Loop ----------
def main():
    init_mt5()
    logger.info(f"Running on SYMBOL={SYMBOL} LIVE_MODE={LIVE_MODE}")

    while True:
        try:
            if pnl_today_usd() <= -DAILY_LOSS_STOP_USD:
                logger.warning("Daily loss limit hit. Pausing.")
                time.sleep(60); continue

            if spread_points() > MAX_SPREAD_POINTS:
                time.sleep(POLL_SECONDS); continue

            h1 = rates_df(SYMBOL, TF_H1, 500)
            m15 = rates_df(SYMBOL, TF_M15, 500)
            m1  = rates_df(SYMBOL, TF_M1,  800)

            if h1.empty or m15.empty or m1.empty:
                time.sleep(POLL_SECONDS); continue

            dir_h1 = h1_trend(h1)
            if dir_h1 not in ("up","down"):
                time.sleep(POLL_SECONDS); continue

            if not m15_bos(m15, dir_h1):
                time.sleep(POLL_SECONDS); continue

            sig = m1_signal(m1, dir_h1)
            if not sig:
                time.sleep(POLL_SECONDS); continue

            entry, sl, tp, atr = atr_sl_tp(m1, sig["side"])
            lots = lot_size(entry, sl)

            feats_row = add_indicators(m1).dropna().iloc[-1][[
                "ema9","ema21","ema50","ema200","wma100","hma55","macdh","adx14","+DI","-DI",
                "st_dir","don_high","don_low","don_mid","kijun","atr14","vol_ma20","close"
            ]].fillna(0.0)
            if not ml_ok(feats_row):
                time.sleep(POLL_SECONDS); continue

            res = send_order(sig["side"], lots, sl, tp)
            log_fill(sig["side"], entry, sl, tp, lots, note=f"atr={atr:.5f}")

            trail_update(atr)
            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.exception(e)
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
