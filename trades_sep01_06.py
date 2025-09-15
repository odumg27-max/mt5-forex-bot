import os, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from bot_mt5 import init_mt5, add_indicators, SYMBOL

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def rates_range_df(symbol, tf_key, start, end):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, end)
    if rates is None or len(rates)==0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})

def _to_bool_mask(series_like, index):
    s = series_like.reindex(index).ffill().fillna(False)
    return pd.Series(np.asarray(s, dtype=bool), index=index)

def build_signals(h1i, m15i, m1i, *, warmup=300, adx_min=18, don_len=20, bos_window_m15=2,
                  vol_mult=1.10, entry_mode="either", supertrend_req=True):
    # H1 regime
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_up_m1   = _to_bool_mask(h1_up,   m1i.index)
    h1_down_m1 = _to_bool_mask(h1_down, m1i.index)

    # M15 BOS
    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(bos_window_m15).max().astype(bool)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(bos_window_m15).max().astype(bool)
    bos_up_m1   = _to_bool_mask(bos_up_recent,   m1i.index)
    bos_down_m1 = _to_bool_mask(bos_down_recent, m1i.index)

    # M1 momentum + volume + Donchian
    ema_up = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    ema_dn = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])
    macd_up = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    macd_dn = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    st_up = (m1i["st_dir"]==1); st_dn = (m1i["st_dir"]==-1)
    adx_ok = (m1i["adx14"]>adx_min)
    vol_ok = (m1i["volume"] > vol_mult*m1i["vol_ma20"])

    dch = m1i["high"].rolling(don_len).max()
    dcl = m1i["low"].rolling(don_len).min()
    brk_up = (m1i["close"]>dch) & (m1i["close"].shift(1)<=dch.shift(1))
    brk_dn = (m1i["close"]<dcl) & (m1i["close"].shift(1)>=dcl.shift(1))

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False

    long_momo  = valid & ema_up & macd_up & adx_ok & vol_ok & (st_up if supertrend_req else True)
    short_momo = valid & ema_dn & macd_dn & adx_ok & vol_ok & (st_dn if supertrend_req else True)
    long_brk   = valid & ema_up & adx_ok & vol_ok & brk_up & (st_up if supertrend_req else True)
    short_brk  = valid & ema_dn & adx_ok & vol_ok & brk_dn & (st_dn if supertrend_req else True)

    if entry_mode=="breakout":
        long_core, short_core = long_brk, short_brk
    elif entry_mode=="momentum":
        long_core, short_core = long_momo, short_momo
    else:
        long_core, short_core = (long_brk | long_momo), (short_brk | short_momo)

    long_sig  = (long_core  & h1_up_m1   & bos_up_m1).astype(bool)
    short_sig = (short_core & h1_down_m1 & bos_down_m1).astype(bool)
    return long_sig, short_sig

def simulate_and_log(m1i, long_sig, short_sig, start_utc, end_utc, *, lookahead=150, tp_R=3.0,
                     start_equity=100.0, risk_pct=0.02):
    # Only signals within [start_utc, end_utc]
    sig = (long_sig | short_sig).copy()
    sig.loc[sig.index < start_utc] = False
    sig.loc[sig.index > end_utc] = False
    idx = m1i.index[sig]

    equity = float(start_equity)
    rows = []
    wins = losses = be_to = 0
    total_R = 0.0
    i = 0

    for ts in idx:
        row = m1i.loc[ts]; atr = row.get("atr14", np.nan)
        if not np.isfinite(atr) or atr<=0:
            continue

        i += 1
        long = bool(long_sig.loc[ts]); entry = float(row["close"])
        fw = m1i.loc[ts:].iloc[1:lookahead]

        if long:
            side="buy"; sl = entry - 1.5*atr; tp = entry + tp_R*(entry - sl)
            hit_tp = (fw["high"]>=tp); hit_sl = (fw["low"]<=sl)
        else:
            side="sell"; sl = entry + 1.5*atr; tp = entry - tp_R*(sl - entry)
            hit_tp = (fw["low"]<=tp);  hit_sl = (fw["high"]>=sl)

        res_R = 0.0; outcome="timeout"
        if hit_tp.any() and hit_sl.any():
            t_tp = fw.index[hit_tp.argmax()]; t_sl = fw.index[hit_sl.argmax()]
            if t_tp <= t_sl: res_R, outcome = tp_R, "TP"
            else:            res_R, outcome = -1.0, "SL"
        elif hit_tp.any():   res_R, outcome = tp_R, "TP"
        elif hit_sl.any():   res_R, outcome = -1.0, "SL"

        # Compounding position size from equity
        risk_usd = equity * risk_pct
        pnl_usd  = res_R * risk_usd
        equity  += pnl_usd
        total_R += res_R
        if   res_R>0: wins+=1
        elif res_R<0: losses+=1
        else: be_to+=1

        # Optional: Lagos local time (won't crash if tz not present)
        try:
            time_lagos = pd.Timestamp(ts).tz_convert("Africa/Lagos").isoformat()
        except Exception:
            time_lagos = ts.isoformat()

        rows.append({
            "i": i,
            "time_utc": ts.isoformat(),
            "time_lagos": time_lagos,
            "side": side,
            "entry": round(entry, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "atr": round(float(atr), 5),
            "R": res_R,
            "outcome": outcome,
            "risk_usd": round(risk_usd, 2),
            "pnl_usd": round(pnl_usd, 2),
            "equity_after": round(equity, 2)
        })

    return rows, wins, losses, be_to, total_R, equity

def main():
    init_mt5()
    sym = SYMBOL
    risk_pct = float(os.getenv("RISK_PCT","0.02"))

    # Window: Sept 1 → Sept 6 (current year), inclusive, UTC
    year = datetime.now(timezone.utc).year
    start = datetime(year, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    end   = datetime(year, 9, 6, 23, 59, 59, tzinfo=timezone.utc)

    # Warmup history for indicators
    start_ext = start - timedelta(days=7)

    print(f"Trades for {sym} | window (UTC): {start} → {end}")
    print(f"Start equity: $100.00  Risk: {risk_pct*100:.1f}% (compounding)\n")

    # Pull data
    h1  = rates_range_df(sym, "H1",  start_ext, end)
    m15 = rates_range_df(sym, "M15", start_ext, end)
    m1  = rates_range_df(sym, "M1",  start_ext, end)
    if h1.empty or m15.empty or m1.empty:
        print("No history returned; open charts / ensure symbol has data.")
        return

    # Indicators
    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    # Signals (robust defaults from earlier run)
    long_sig, short_sig = build_signals(
        h1i, m15i, m1i,
        warmup=300, adx_min=18, don_len=20, bos_window_m15=2,
        vol_mult=1.10, entry_mode="either", supertrend_req=True
    )

    rows, wins, losses, be_to, total_R, final_equity = simulate_and_log(
        m1i, long_sig, short_sig, start, end,
        lookahead=150, tp_R=3.0, start_equity=100.0, risk_pct=risk_pct
    )

    if not rows:
        print("No trades in this window.")
        return

    df = pd.DataFrame(rows)
    os.makedirs("logs", exist_ok=True)
    out_csv = "logs/trades_sep01_06.csv"
    df.to_csv(out_csv, index=False)

    # Print all trades
    print("=== TRADES (Sept 1 → Sept 6, UTC) ===")
    pd.set_option("display.max_rows", None)
    cols = ["i","time_utc","time_lagos","side","entry","sl","tp","atr","R","outcome","risk_usd","pnl_usd","equity_after"]
    print(df[cols].to_string(index=False))

    # Summary (winrate & total PnL)
    considered = wins + losses
    winrate = (wins/considered*100.0) if considered>0 else 0.0
    print("\n=== SUMMARY ===")
    print(f"Trades: {len(rows)}  Wins: {wins}  Losses: {losses}  BE/Timeout: {be_to}")
    print(f"Win rate: {winrate:.2f}%  Total PnL: {total_R:.1f} R")
    print(f"Final equity (start $100, compounding): ${final_equity:.2f}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
