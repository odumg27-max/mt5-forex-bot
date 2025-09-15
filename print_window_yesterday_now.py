import os
from datetime import datetime, time, timedelta, timezone
import pandas as pd
import MetaTrader5 as mt5
from bot_mt5 import init_mt5, SYMBOL

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

# Resolve Africa/Lagos timezone safely (fallback to UTC+1 if zoneinfo missing)
try:
    from zoneinfo import ZoneInfo
    LAGOS = ZoneInfo("Africa/Lagos")
except Exception:
    LAGOS = timezone(timedelta(hours=1))

def rates_range_df(symbol, tf_key, start_utc, end_utc):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start_utc, end_utc)
    if rates is None or len(rates)==0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def main():
    init_mt5()

    now_utc = datetime.now(timezone.utc)
    now_lagos = now_utc.astimezone(LAGOS)

    # Yesterday 00:00 in Lagos → UTC
    yday_date = (now_lagos - timedelta(days=1)).date()
    start_lagos = datetime.combine(yday_date, time(0,0), tzinfo=LAGOS)
    start_utc = start_lagos.astimezone(timezone.utc)

    print("=== WINDOW (Yesterday → Now) ===")
    print(f"Symbol: {SYMBOL}")
    print(f"Start (Lagos): {start_lagos.isoformat()}")
    print(f"Start (UTC)  : {start_utc.isoformat()}")
    print(f"Now   (Lagos): {now_lagos.isoformat()}")
    print(f"Now   (UTC)  : {now_utc.isoformat()}")

    # Pull bars
    h1  = rates_range_df(SYMBOL, "H1",  start_utc, now_utc)
    m15 = rates_range_df(SYMBOL, "M15", start_utc, now_utc)
    m1  = rates_range_df(SYMBOL, "M1",  start_utc, now_utc)

    print("\n=== BAR COUNTS ===")
    print(f"H1 : {len(h1)}")
    print(f"M15: {len(m15)}")
    print(f"M1 : {len(m1)}")

    # Show first/last timestamps actually returned (in Lagos)
    def ts_bounds(df):
        if df.empty: return ("—","—")
        first = df.index[0].astimezone(LAGOS).isoformat()
        last  = df.index[-1].astimezone(LAGOS).isoformat()
        return (first, last)

    f_h1, l_h1   = ts_bounds(h1)
    f_m15, l_m15 = ts_bounds(m15)
    f_m1, l_m1   = ts_bounds(m1)

    print("\n=== ACTUAL DATA RANGE (Africa/Lagos) ===")
    print(f"H1 : first {f_h1}   | last {l_h1}")
    print(f"M15: first {f_m15}  | last {l_m15}")
    print(f"M1 : first {f_m1}   | last {l_m1}")

    # Preview last 5 M1 bars (Lagos time)
    if not m1.empty:
        tail = m1.tail(5).copy()
        tail.index = tail.index.tz_convert(LAGOS)
        print("\n=== LAST 5 M1 BARS (Africa/Lagos) ===")
        print(tail.to_string())

if __name__ == "__main__":
    main()
