import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone

from bot_mt5 import init_mt5, add_indicators, SYMBOL

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def rates_range_df(symbol, tf_key, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days+7)  # buffer over weekends/holidays
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, now)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})

def run(days=120, warmup=300):
    init_mt5()
    sym = SYMBOL
    print(f"Backtest symbol={sym} days={days} warmup={warmup}")

    h1  = rates_range_df(sym, "H1",  days)
    m15 = rates_range_df(sym, "M15", days)
    m1  = rates_range_df(sym, "M1",  days)

    print(f"Bars fetched -> H1:{len(h1)}  M15:{len(m15)}  M1:{len(m1)}")
    if min(len(h1), len(m15), len(m1)) < warmup:
        print("Still not enough bars pre-indicators. Open H1/M15/M1 charts to download more history or increase days.")
        return

    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    # Only require the columns we actually use (instead of dropna on ALL)
    h1_req  = ["ema50","ema200","adx14","close"]
    m15_req = ["don_high","don_low","close","high","low"]
    m1_req  = ["ema9","ema21","ema50","macdh","st_dir","adx14","don_high","don_low","atr14","volume","vol_ma20","close"]

    h1i  = h1i.dropna(subset=h1_req).iloc[-(len(h1i)-5):]   # tolerate a few NaNs elsewhere
    m15i = m15i.dropna(subset=m15_req)
    m1i  = m1i.dropna(subset=m1_req)

    print(f"After indicators -> H1:{len(h1i)}  M15:{len(m15i)}  M1:{len(m1i)}")
    if h1i.empty or m15i.empty or m1i.empty:
        print("A timeframe is still empty after indicators. Increase days further (e.g., 240) and keep charts open to cache history.")
        return

    pnl_R, wins, losses, trades = 0.0, 0, 0, 0
    start_idx = max(warmup, 220)

    for ts, _ in m1i.iloc[start_idx:].iterrows():
        h1u  = h1i.loc[:ts].iloc[-warmup:]
        m15u = m15i.loc[:ts].iloc[-warmup:]
        m1u  = m1i.loc[:ts].iloc[-warmup:]

        # H1 trend filter
        lh1 = h1u.iloc[-1]
        if   (lh1.ema50 > lh1.ema200 and lh1.adx14 > 18): dir_h1 = "up"
        elif (lh1.ema50 < lh1.ema200 and lh1.adx14 > 18): dir_h1 = "down"
        else: continue

        # M15 BOS / shift
        prior_high = m15u.high.rolling(20).max().shift(1).iloc[-1]
        prior_low  = m15u.low.rolling(20).min().shift(1).iloc[-1]
        c15 = m15u.close.iloc[-1]
        if   dir_h1=="up"   and not (c15 > prior_high): continue
        elif dir_h1=="down" and not (c15 < prior_low):  continue

        # M1 momentum + Donchian breakout
        prev = m1u.iloc[-2]; last = m1u.iloc[-1]
        breakout_up = last.close > last.don_high and prev.close <= prev.don_high
        breakout_dn = last.close < last.don_low  and prev.close >= prev.don_low

        if dir_h1=="up":
            cond = (last.ema9>last.ema21>last.ema50) and (last.macdh>0) and (last.st_dir==1) and (last.adx14>18) and breakout_up and (last.volume > 1.05*last.vol_ma20)
            side = "buy" if cond else None
        else:
            cond = (last.ema9<last.ema21<last.ema50) and (last.macdh<0) and (last.st_dir==-1) and (last.adx14>18) and breakout_dn and (last.volume > 1.05*last.vol_ma20)
            side = "sell" if cond else None
        if not side: continue

        # ATR-based SL/TP, R:R=1:3
        atr = last.atr14
        if np.isnan(atr) or atr <= 0: continue
        entry = last.close
        if side=="buy":
            sl = entry - 1.5*atr
            tp = entry + 3.0*(entry - sl)
        else:
            sl = entry + 1.5*atr
            tp = entry - 3.0*(sl - entry)

        fw = m1i.loc[ts:].iloc[1:200]
        if fw.empty: continue

        if side=="buy":
            hit_tp = (fw.high >= tp)
            hit_sl = (fw.low  <= sl)
        else:
            hit_tp = (fw.low  <= tp)
            hit_sl = (fw.high >= sl)

        res_R = 0.0
        if hit_tp.any() and hit_sl.any():
            t_tp = fw.index[hit_tp.argmax()]
            t_sl = fw.index[hit_sl.argmax()]
            res_R = 3.0 if t_tp <= t_sl else -1.0
        elif hit_tp.any():
            res_R = 3.0
        elif hit_sl.any():
            res_R = -1.0

        pnl_R += res_R; trades += 1
        if   res_R > 0: wins += 1
        elif res_R < 0: losses += 1

    print(f"Trades:{trades}  Wins:{wins}  Losses:{losses}  PnL (R):{pnl_R:.1f}")
    if trades: print(f"Win rate: {wins/trades*100:.1f}%")

if __name__ == "__main__":
    run(days=120, warmup=300)
