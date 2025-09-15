import numpy as np, pandas as pd, MetaTrader5 as mt5
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

def run(days=45, warmup=300, lookahead=150, max_trades=250, progress_every=25,
        adx_min=14, don_len=15, bos_window_m15=3, vol_mult=1.00,
        entry_mode="either", supertrend_req=True):
    """
    entry_mode: 'breakout' (needs Donchian cross) | 'momentum' (EMA+MACD+ADX+vol) | 'either'
    """
    init_mt5()
    sym = SYMBOL
    print(f"Fast backtest symbol={sym} days={days} warmup={warmup} lookahead={lookahead} max_trades={max_trades}")
    print(f"Params: adx_min={adx_min} don_len={don_len} bos_window_m15={bos_window_m15} vol_mult={vol_mult} entry_mode={entry_mode} supertrend_req={supertrend_req}")

    h1  = rates_range_df(sym, "H1",  days)
    m15 = rates_range_df(sym, "M15", days)
    m1  = rates_range_df(sym, "M1",  days)
    print(f"Bars -> H1:{len(h1)}  M15:{len(m15)}  M1:{len(m1)}")
    if min(len(h1), len(m15), len(m1)) < warmup:
        print("Not enough bars; increase --days or open charts to cache more history."); return

    # Indicators
    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    # --- Vectorized top-down filters (align to M1) ---
    # H1: trend regime
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_up_m1   = h1_up.reindex(m1i.index, method="ffill").fillna(False)
    h1_down_m1 = h1_down.reindex(m1i.index, method="ffill").fillna(False)

    # M15: BOS in recent window (any of last bos_window_m15 bars broke prior swing)
    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_m15_raw   = (m15i["close"] > prior_high)
    bos_down_m15_raw = (m15i["close"] < prior_low)
    bos_up_recent   = bos_up_m15_raw.rolling(bos_window_m15).max().astype(bool)
    bos_down_recent = bos_down_m15_raw.rolling(bos_window_m15).max().astype(bool)
    bos_up_m1   = bos_up_recent.reindex(m1i.index, method="ffill").fillna(False)
    bos_down_m1 = bos_down_recent.reindex(m1i.index, method="ffill").fillna(False)

    # M1: momentum & breakout
    ema_align_up   = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    ema_align_down = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])
    macd_up   = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    macd_down = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    st_up   = (m1i["st_dir"]==1)
    st_down = (m1i["st_dir"]==-1)
    adx_ok  = (m1i["adx14"]>adx_min)
    vol_ok  = (m1i["volume"] > vol_mult * m1i["vol_ma20"])

    # Donchian breakout with tunable length (computed here)
    dch_m1, dcl_m1 = donchian_bounds(m1i["high"], m1i["low"], don_len)
    breakout_up   = (m1i["close"]>dch_m1) & (m1i["close"].shift(1)<=dch_m1.shift(1))
    breakout_down = (m1i["close"]<dcl_m1) & (m1i["close"].shift(1)>=dcl_m1.shift(1))

    valid_mask = pd.Series(True, index=m1i.index); valid_mask.iloc[:warmup] = False

    # Momentum-only conditions
    long_momo  = valid_mask & ema_align_up   & macd_up   & adx_ok & vol_ok & (st_up if supertrend_req else True)
    short_momo = valid_mask & ema_align_down & macd_down & adx_ok & vol_ok & (st_down if supertrend_req else True)

    # Breakout-only conditions (still trend-aligned with EMAs)
    long_break  = valid_mask & ema_align_up   & adx_ok & vol_ok & breakout_up   & (st_up if supertrend_req else True)
    short_break = valid_mask & ema_align_down & adx_ok & vol_ok & breakout_down & (st_down if supertrend_req else True)

    # Top-down gating
    if entry_mode == "breakout":
        long_sig_core  = long_break
        short_sig_core = short_break
    elif entry_mode == "momentum":
        long_sig_core  = long_momo
        short_sig_core = short_momo
    else:  # either
        long_sig_core  = long_break | long_momo
        short_sig_core = short_break | short_momo

    long_sig  = long_sig_core  & h1_up_m1   & bos_up_m1
    short_sig = short_sig_core & h1_down_m1 & bos_down_m1
    sig_mask = (long_sig | short_sig)

    # --- Diagnostics ---
    total_m1 = len(m1i) - warmup
    def cnt(s): return int(s.iloc[warmup:].sum()) if len(s)>warmup else 0
    print("\nFilter counts (M1 bars after warmup):")
    print(f"H1 up: {cnt(h1_up_m1)}  H1 down: {cnt(h1_down_m1)}")
    print(f"BOS up (recent): {cnt(bos_up_m1)}  BOS down (recent): {cnt(bos_down_m1)}")
    print(f"EMA align up: {cnt(ema_align_up)}  down: {cnt(ema_align_down)}")
    print(f"MACD up: {cnt(macd_up)}  down: {cnt(macd_down)}  ADX ok: {cnt(adx_ok)}  Vol ok: {cnt(vol_ok)}")
    print(f"Breakout up: {cnt(breakout_up)}  down: {cnt(breakout_down)}")
    print(f"Momentum long: {cnt(long_momo)}  short: {cnt(short_momo)}")
    print(f"Breakout long: {cnt(long_break)}  short: {cnt(short_break)}")
    print(f"Signals before top-down: {cnt(long_sig_core|short_sig_core)}")
    print(f"Signals after top-down:  {cnt(sig_mask)}")

    sig_idx = m1i.index[sig_mask]
    if len(sig_idx)==0:
        print("\nNo entry signals with current params. Try lowering --adx_min, --vol_mult, or use --entry_mode momentum.")
        return

    # --- Trade sim only over signal bars ---
    pnl_R = 0.0; wins=losses=0; trades=0
    for ts in sig_idx:
        if trades >= max_trades: break
        row = m1i.loc[ts]
        atr = row.get("atr14", np.nan)
        if not np.isfinite(atr) or atr<=0: continue
        side = "buy" if long_sig.loc[ts] else "sell"
        entry = float(row["close"])
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

        res = 0.0
        if tp_hit.any() and sl_hit.any():
            t_tp = fw.index[tp_hit.argmax()]
            t_sl = fw.index[sl_hit.argmax()]
            res = 3.0 if t_tp <= t_sl else -1.0
        elif tp_hit.any():
            res = 3.0
        elif sl_hit.any():
            res = -1.0

        pnl_R += res; trades += 1
        if   res>0: wins+=1
        elif res<0: losses+=1
        if trades % progress_every == 0:
            print(f"Processed {trades}/{max_trades} signals... PnL(R)={pnl_R:.1f}")

    print(f"\nRESULTS  Trades:{trades}  Wins:{wins}  Losses:{losses}  PnL (R):{pnl_R:.1f}")
    if trades:
        print(f"Win rate: {wins/trades*100:.1f}%  Avg R/trade: {pnl_R/trades:.2f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lookahead", type=int, default=150)
    ap.add_argument("--max_trades", type=int, default=250)
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--adx_min", type=int, default=14)
    ap.add_argument("--don_len", type=int, default=15)
    ap.add_argument("--bos_window_m15", type=int, default=3)
    ap.add_argument("--vol_mult", type=float, default=1.00)
    ap.add_argument("--entry_mode", choices=["breakout","momentum","either"], default="either")
    ap.add_argument("--supertrend_req", action="store_true")      # add flag to require supertrend
    ap.add_argument("--no_supertrend", dest="supertrend_req", action="store_false")
    ap.set_defaults(supertrend_req=True)
    args = ap.parse_args()
    run(**vars(args))
