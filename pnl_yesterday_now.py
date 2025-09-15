import os, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone, time
from bot_mt5 import init_mt5, add_indicators, SYMBOL
from os import getenv

# --- Timezone: Africa/Lagos ---
try:
    from zoneinfo import ZoneInfo
    LAGOS = ZoneInfo("Africa/Lagos")
except Exception:
    LAGOS = timezone(timedelta(hours=1))

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def rates_range_df(symbol, tf_key, start_utc, end_utc):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start_utc, end_utc)
    if rates is None or len(rates)==0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def donchian_bounds(high, low, n):
    dch = high.rolling(n).max()
    dcl = low.rolling(n).min()
    return dch, dcl

def build_signals(h1i, m15i, m1i, *, warmup=300, adx_min=14, don_len=15, bos_window_m15=3,
                  vol_mult=1.00, entry_mode="either", supertrend_req=True):
    # H1 regime
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_up_m1   = h1_up.reindex(m1i.index, method="ffill").fillna(False)
    h1_down_m1 = h1_down.reindex(m1i.index, method="ffill").fillna(False)

    # M15 BOS (recent)
    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(bos_window_m15).max().astype(bool)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(bos_window_m15).max().astype(bool)
    bos_up_m1   = bos_up_recent.reindex(m1i.index, method="ffill").fillna(False)
    bos_down_m1 = bos_down_recent.reindex(m1i.index, method="ffill").fillna(False)

    # M1 momentum + breakout
    ema_up   = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    ema_dn   = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])
    macd_up  = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    macd_dn  = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    st_up    = (m1i["st_dir"]==1)
    st_dn    = (m1i["st_dir"]==-1)
    adx_ok   = (m1i["adx14"]>adx_min)
    vol_ok   = (m1i["volume"] > vol_mult*m1i["vol_ma20"])

    dch_m1 = m1i["high"].rolling(don_len).max()
    dcl_m1 = m1i["low"].rolling(don_len).min()
    brk_up   = (m1i["close"]>dch_m1) & (m1i["close"].shift(1)<=dch_m1.shift(1))
    brk_dn   = (m1i["close"]<dcl_m1) & (m1i["close"].shift(1)>=dcl_m1.shift(1))

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False

    long_momo  = valid & ema_up & macd_up & adx_ok & vol_ok & (st_up if supertrend_req else True)
    short_momo = valid & ema_dn & macd_dn & adx_ok & vol_ok & (st_dn if supertrend_req else True)

    long_brk   = valid & ema_up   & adx_ok & vol_ok & brk_up   & (st_up if supertrend_req else True)
    short_brk  = valid & ema_dn   & adx_ok & vol_ok & brk_dn   & (st_dn if supertrend_req else True)

    if entry_mode=="breakout":
        long_core, short_core = long_brk, short_brk
    elif entry_mode=="momentum":
        long_core, short_core = long_momo, short_momo
    else:
        long_core, short_core = (long_brk | long_momo), (short_brk | short_momo)

    long_sig  = long_core  & h1_up_m1   & bos_up_m1
    short_sig = short_core & h1_down_m1 & bos_down_m1
    return long_sig, short_sig

def main():
    init_mt5()
    sym = SYMBOL

    # --- Window: yesterday 00:00 (Africa/Lagos) → now ---
    now_utc = datetime.now(timezone.utc)
    now_lagos = now_utc.astimezone(LAGOS)
    yday_date = (now_lagos - timedelta(days=1)).date()
    start_lagos = datetime.combine(yday_date, time(0,0), tzinfo=LAGOS)
    start_utc   = start_lagos.astimezone(timezone.utc)

    # Extra history (for indicator warmup only)
    warmup_days = 20
    start_ext = start_utc - timedelta(days=warmup_days)

    print("=== WINDOW (Yesterday → Now) ===")
    print(f"Symbol: {sym}")
    print(f"Start (Lagos): {start_lagos.isoformat()}   Start (UTC): {start_utc.isoformat()}")
    print(f"Now   (Lagos): {now_lagos.isoformat()}   Now   (UTC): {now_utc.isoformat()}")

    # Pull data with warmup lookback
    h1  = rates_range_df(sym, "H1",  start_ext, now_utc)
    m15 = rates_range_df(sym, "M15", start_ext, now_utc)
    m1  = rates_range_df(sym, "M1",  start_ext, now_utc)

    if h1.empty or m15.empty or m1.empty:
        print("No bars returned. Make sure the symbol is visible and history is cached.")
        return

    # Indicators
    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    # Build signals (same params you’ve been using)
    long_sig, short_sig = build_signals(
        h1i, m15i, m1i,
        warmup=300, adx_min=14, don_len=15, bos_window_m15=3,
        vol_mult=1.00, entry_mode="either", supertrend_req=True
    )

    # Only evaluate trades inside the actual window
    trade_mask = (long_sig | short_sig)
    trade_idx = [ts for ts in m1i.index[trade_mask] if ts >= start_utc]

    if len(trade_idx) == 0:
        print("\nNo entry signals from yesterday to now with current params.")
        return

    # Risk from .env (default 1% if not set)
    try:
        risk_pct = float(getenv("RISK_PCT", "0.01"))
    except Exception:
        risk_pct = 0.01

    start_equity = 100.0
    equity = start_equity
    lookahead = 150  # minutes to check TP/SL
    wins=losses=0; pnl_total=0.0; pnl_R_total=0.0; trades=0

    for ts in trade_idx:
        row = m1i.loc[ts]
        atr = row.get("atr14", np.nan)
        if not np.isfinite(atr) or atr <= 0:
            continue

        side = "buy" if long_sig.loc[ts] else "sell"
        entry = float(row["close"])

        if side=="buy":
            sl = entry - 1.5*atr
            tp = entry + 3.0*(entry - sl)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            tp_hit = (fw["high"] >= tp); sl_hit = (fw["low"] <= sl)
        else:
            sl = entry + 1.5*atr
            tp = entry - 3.0*(sl - entry)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            tp_hit = (fw["low"] <= tp);  sl_hit = (fw["high"] >= sl)

        res_R = 0.0
        if tp_hit.any() and sl_hit.any():
            t_tp = fw.index[tp_hit.argmax()]; t_sl = fw.index[sl_hit.argmax()]
            res_R = 3.0 if t_tp <= t_sl else -1.0
        elif tp_hit.any():
            res_R = 3.0
        elif sl_hit.any():
            res_R = -1.0
        else:
            res_R = 0.0  # timeout inside lookahead

        risk_usd = equity * risk_pct
        pnl_usd  = res_R * risk_usd
        equity  += pnl_usd

        pnl_total += pnl_usd
        pnl_R_total += res_R
        trades += 1
        if   res_R>0: wins+=1
        elif res_R<0: losses+=1

    print("\n=== SUMMARY (Yesterday → Now) ===")
    print(f"Trades: {trades}  Wins: {wins}  Losses: {losses}")
    print(f"Risk per trade: {risk_pct*100:.2f}% of current equity (compounding)")
    print(f"Total PnL (R): {pnl_R_total:.1f}")
    print(f"Total PnL ($): ${pnl_total:.2f}")
    print(f"Starting equity: ${start_equity:.2f}")
    print(f"Final equity:    ${equity:.2f}")

if __name__ == "__main__":
    main()
