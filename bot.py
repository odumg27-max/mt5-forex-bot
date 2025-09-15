# bot.py — integrates OB/OS v2 strategy as the main live signal source.
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from bot_mt5 import init_mt5, add_indicators, SYMBOL     # uses your existing infra
from strategy import build_signals_v2, filter_signal_times
# optional live trade executor (if you already have it)
try:
    from compounding_rr import place_market_rr_trade
    _HAVE_EXEC = True
except Exception:
    _HAVE_EXEC = False

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def rates_range_df(symbol, tf_key, start, end):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, end)
    if rates is None or len(rates)==0: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df.rename(columns={"tick_volume":"volume"})
    return df[["open","high","low","close","volume"]]

def run_once(warm_days=30, adx_min=20, vol_mult=1.10):
    init_mt5()
    sym = SYMBOL
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=warm_days)

    # pull H1/M15/M1
    h1  = rates_range_df(sym, "H1",  start, now)
    m15 = rates_range_df(sym, "M15", start, now)
    m1  = rates_range_df(sym, "M1",  start, now)
    if h1.empty or m15.empty or m1.empty:
        print("No history — open charts / check MT5 connection.")
        return

    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    long_sig, short_sig = build_signals_v2(
        h1i, m15i, m1i, warmup=300, adx_min=adx_min, vol_mult=vol_mult,
        don_len=20, supertrend_req=True
    )

    # look only at the latest closed bar
    last_ts = m1i.index[-1]
    sig_times = filter_signal_times(m1i.index, long_sig, short_sig)
    if len(sig_times)==0 or sig_times[-1] != last_ts:
        print(f"{last_ts.isoformat()} — No new signal.")
        return

    is_long = bool(long_sig.loc[last_ts])
    side = "buy" if is_long else "sell"
    price = float(m1i.loc[last_ts, "close"])
    atr = float(m1i.loc[last_ts, "atr14"])

    print(f"{last_ts.isoformat()} — SIGNAL: {side.upper()}  price={price}  ATR={atr}")

    # Optional live execution — 2% risk, 1:3 RR, SL=1.5*ATR
    if _HAVE_EXEC and np.isfinite(atr) and atr>0:
        try:
            place_market_rr_trade(sym, side, atr=atr, rr=3.0, atr_mult=1.5, dry_run=False)
            print("Order sent (2% risk, RR=1:3, SL=1.5*ATR).")
        except Exception as e:
            print(f"Execution failed: {e}")
    else:
        print("Execution module not available; signal logged only.")

if __name__ == "__main__":
    run_once()
