import os, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from bot_mt5 import init_mt5, add_indicators, SYMBOL

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def rates_range_df(symbol, tf_key, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days+7)
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, now)
    if rates is None or len(rates)==0: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})

def donchian_bounds(high, low, n):
    dch = high.rolling(n).max()
    dcl = low.rolling(n).min()
    return dch, dcl

def build_signals(h1i, m15i, m1i, *, warmup=300, adx_min=14, don_len=15, bos_window_m15=3,
                  vol_mult=1.00, entry_mode="either", supertrend_req=True):
    # H1 top regime
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_up_m1   = h1_up.reindex(m1i.index, method="ffill").fillna(False)
    h1_down_m1 = h1_down.reindex(m1i.index, method="ffill").fillna(False)

    # M15 BOS (any of last K bars)
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

    dch_m1, dcl_m1 = donchian_bounds(m1i["high"], m1i["low"], don_len)
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

def run(days=45, warmup=300, lookahead=150, max_trades=200,
        adx_min=14, don_len=15, bos_window_m15=3, vol_mult=1.00,
        entry_mode="either", supertrend_req=True,
        start_equity=100.0, risk_pct=0.02, out_csv="logs/trade_log.csv"):

    os.makedirs("logs", exist_ok=True)
    init_mt5()
    sym = SYMBOL
    print(f"Equity report for {sym} | days={days} warmup={warmup} lookahead={lookahead} max_trades={max_trades}")
    print(f"Risk: {risk_pct*100:.1f}%  Start equity: ${start_equity:.2f}  (compounding)")

    h1  = rates_range_df(sym, "H1",  days)
    m15 = rates_range_df(sym, "M15", days)
    m1  = rates_range_df(sym, "M1",  days)
    if min(len(h1), len(m15), len(m1)) < warmup:
        print("Not enough history; open charts and/or increase days.")
        return

    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    long_sig, short_sig = build_signals(h1i, m15i, m1i, warmup=warmup, adx_min=adx_min,
                                        don_len=don_len, bos_window_m15=bos_window_m15,
                                        vol_mult=vol_mult, entry_mode=entry_mode, supertrend_req=supertrend_req)
    sig_mask = (long_sig | short_sig)
    sig_idx = m1i.index[sig_mask]
    if len(sig_idx)==0:
        print("No signals found with current parameters.")
        return

    equity = float(start_equity)
    rows = []
    wins=losses=0; pnl_total=0.0; trades=0
    for ts in sig_idx:
        if trades >= max_trades: break
        last = m1i.loc[ts]
        atr = last.get("atr14", np.nan)
        if not np.isfinite(atr) or atr<=0: continue

        side = "buy" if long_sig.loc[ts] else "sell"
        entry = float(last["close"])

        # ATR-based SL/TP (R:R = 1:3)
        if side=="buy":
            sl = entry - 1.5*atr
            tp = entry + 3.0*(entry - sl)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            tp_hit = (fw["high"] >= tp)
            sl_hit = (fw["low"]  <= sl)
        else:
            sl = entry + 1.5*atr
            tp = entry - 3.0*(sl - entry)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            tp_hit = (fw["low"]  <= tp)
            sl_hit = (fw["high"] >= sl)

        # First-hit wins logic
        res_R = 0.0; outcome = "timeout"
        if tp_hit.any() and sl_hit.any():
            t_tp = fw.index[tp_hit.argmax()]
            t_sl = fw.index[sl_hit.argmax()]
            if t_tp <= t_sl:
                res_R = 3.0; outcome = "TP"
            else:
                res_R = -1.0; outcome = "SL"
        elif tp_hit.any():
            res_R = 3.0; outcome = "TP"
        elif sl_hit.any():
            res_R = -1.0; outcome = "SL"

        risk_usd = equity * risk_pct
        pnl_usd  = res_R * risk_usd
        equity  += pnl_usd
        pnl_total += pnl_usd
        trades += 1
        if   res_R>0: wins+=1
        elif res_R<0: losses+=1

        rows.append({
            "i": trades, "time": ts.isoformat(), "side": side,
            "entry": round(entry, 5), "sl": round(sl, 5), "tp": round(tp, 5),
            "atr": round(float(atr), 5),
            "R": res_R, "risk_usd": round(risk_usd, 2),
            "pnl_usd": round(pnl_usd, 2), "equity_after": round(equity, 2),
            "outcome": outcome
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== TRADE LOG (every trade) ===")
    pd.set_option("display.max_rows", None)
    print(df.to_string(index=False))

    # Summary
    peak = df["equity_after"].cummax()
    dd = (df["equity_after"] - peak)
    max_dd = dd.min()
    print("\n=== SUMMARY ===")
    print(f"Trades: {trades}  Wins: {wins}  Losses: {losses}")
    print(f"Total PnL: ${pnl_total:.2f}  Final equity: ${equity:.2f}  Max drawdown: ${max_dd:.2f}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    # Defaults match the run that produced ~200 trades and +94R
    run(days=45, warmup=300, lookahead=150, max_trades=200,
        adx_min=14, don_len=15, bos_window_m15=3, vol_mult=1.00,
        entry_mode="either", supertrend_req=True,
        start_equity=100.0, risk_pct=0.02, out_csv="logs/trade_log.csv")
