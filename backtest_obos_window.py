import os, math, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from bot_mt5 import init_mt5, add_indicators, SYMBOL

# -------- Timeframes ----------
TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

# -------- Minimal data fetchers ----------
def rates_range_df(symbol, tf_key, start, end):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, end)
    if rates is None or len(rates)==0: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df.rename(columns={"tick_volume":"volume"})
    return df[["open","high","low","close","volume"]]

# -------- OB/OS helper indicators (vectorized) ----------
def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")

def stoch_kd(high, low, close, n=14, d=3):
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d_ = k.rolling(d).mean()
    return k.fillna(method="bfill"), d_.fillna(method="bfill")

def williams_r(high, low, close, n=14):
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    wr = -100 * (hh - close) / (hh - ll).replace(0, np.nan)
    return wr.fillna(method="bfill")

def cci(high, low, close, n=20):
    tp = (high + low + close) / 3.0
    sma = tp.rolling(n).mean()
    md = (tp - sma).abs().rolling(n).mean()
    cci_ = (tp - sma) / (0.015 * md.replace(0, np.nan))
    return cci_.fillna(method="bfill")

def mfi(high, low, close, volume, n=14):
    tp = (high + low + close) / 3.0
    pmf = ((tp > tp.shift(1)) * (tp * volume)).fillna(0.0)
    nmf = ((tp < tp.shift(1)) * (tp * volume)).fillna(0.0)
    pmf_n = pmf.rolling(n).sum()
    nmf_n = nmf.rolling(n).sum().replace(0, np.nan)
    mr = pmf_n / nmf_n
    out = 100 - (100 / (1 + mr))
    return out.replace([np.inf, -np.inf], np.nan).fillna(method="bfill")

def bbands(close, n=20, k=2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + k*sd
    lower = ma - k*sd
    pb = (close - lower) / (upper - lower)
    return upper, ma, lower, pb

def add_obos_cols(df):
    out = df.copy()
    out["rsi14"] = rsi(out["close"], 14)
    out["stochK"], out["stochD"] = stoch_kd(out["high"], out["low"], out["close"], 14, 3)
    out["wr14"]  = williams_r(out["high"], out["low"], out["close"], 14)
    out["cci20"] = cci(out["high"], out["low"], out["close"], 20)
    out["mfi14"] = mfi(out["high"], out["low"], out["close"], out["volume"], 14)
    out["bbU"], out["bbM"], out["bbL"], out["pb"] = bbands(out["close"], 20, 2.0)
    return out

# -------- BOS on M15 and H1 regime ----------
def topdown_masks(h1i, m15i, m1i, adx_min=14):
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_up_m1   = h1_up.reindex(m1i.index).ffill().fillna(False).astype(bool)
    h1_down_m1 = h1_down.reindex(m1i.index).ffill().fillna(False).astype(bool)

    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(3).max().astype(bool)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(3).max().astype(bool)
    bos_up_m1   = bos_up_recent.reindex(m1i.index).ffill().fillna(False).astype(bool)
    bos_down_m1 = bos_down_recent.reindex(m1i.index).ffill().fillna(False).astype(bool)
    return h1_up_m1, h1_down_m1, bos_up_m1, bos_down_m1

# -------- Core signals (existing momentum+breakout) + NEW OB/OS pullback entries ----------
def build_signals_with_obos(h1i, m15i, m1i, *, warmup=300, adx_min=18, vol_mult=1.05, don_len=20, supertrend_req=True):
    h1_up_m1, h1_down_m1, bos_up_m1, bos_down_m1 = topdown_masks(h1i, m15i, m1i, adx_min=adx_min)

    # Momentum stack & volume/ADX gates
    ema_up   = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    ema_dn   = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])
    macd_up  = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    macd_dn  = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    st_up    = (m1i["st_dir"]==1)
    st_dn    = (m1i["st_dir"]==-1)
    adx_ok   = (m1i["adx14"]>adx_min)
    vol_ok   = (m1i["volume"] > vol_mult*m1i["vol_ma20"])

    # Donchian breakout
    dch = m1i["high"].rolling(don_len).max()
    dcl = m1i["low"].rolling(don_len).min()
    brk_up = (m1i["close"]>dch) & (m1i["close"].shift(1)<=dch.shift(1))
    brk_dn = (m1i["close"]<dcl) & (m1i["close"].shift(1)>=dcl.shift(1))

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False

    long_momo  = valid & ema_up & macd_up & adx_ok & vol_ok & (st_up if supertrend_req else True)
    short_momo = valid & ema_dn & macd_dn & adx_ok & vol_ok & (st_dn if supertrend_req else True)
    long_brk   = valid & ema_up   & adx_ok & vol_ok & brk_up & (st_up if supertrend_req else True)
    short_brk  = valid & ema_dn   & adx_ok & vol_ok & brk_dn & (st_dn if supertrend_req else True)

    # ---------- NEW: OB/OS cluster pullbacks with recross triggers ----------
    m1 = add_obos_cols(m1i)

    # Oversold cluster score (for longs)
    os_flags = [
        (m1["rsi14"] < 30),
        (m1["stochK"] < 20),
        (m1["wr14"] < -80),
        (m1["cci20"] < -100),
        (m1["mfi14"] < 20),
        (m1["close"] <= m1["bbL"])
    ]
    os_score = sum(f.astype(int) for f in os_flags)

    # Overbought cluster score (for shorts)
    ob_flags = [
        (m1["rsi14"] > 70),
        (m1["stochK"] > 80),
        (m1["wr14"] > -20),
        (m1["cci20"] > 100),
        (m1["mfi14"] > 80),
        (m1["close"] >= m1["bbU"])
    ]
    ob_score = sum(f.astype(int) for f in ob_flags)

    # Recross/trigger back with trend
    recross_up = ((m1["stochK"] > m1["stochD"]) & (m1["stochK"].shift(1) <= m1["stochD"].shift(1))) | \
                 ((m1["rsi14"] > 30) & (m1["rsi14"].shift(1) <= 30)) | \
                 ((m1["close"] > m1["bbL"]) & (m1["close"].shift(1) <= m1["bbL"].shift(1)))

    recross_dn = ((m1["stochK"] < m1["stochD"]) & (m1["stochK"].shift(1) >= m1["stochD"].shift(1))) | \
                 ((m1["rsi14"] < 70) & (m1["rsi14"].shift(1) >= 70)) | \
                 ((m1["close"] < m1["bbU"]) & (m1["close"].shift(1) >= m1["bbU"].shift(1)))

    # Require decent cluster (>=3 of 6) + ADX trend context
    long_obos  = valid & (os_score >= 3) & recross_up  & adx_ok & (st_up if supertrend_req else True)
    short_obos = valid & (ob_score >= 3) & recross_dn  & adx_ok & (st_dn if supertrend_req else True)

    # Combine cores (existing + new OB/OS pullback), then gate by top-down masks
    long_core  = (long_momo  | long_brk  | long_obos)
    short_core = (short_momo | short_brk | short_obos)

    long_sig  = (long_core  & h1_up_m1   & bos_up_m1).astype(bool)
    short_sig = (short_core & h1_down_m1 & bos_down_m1).astype(bool)
    return long_sig, short_sig

# -------- Backtest (compounding, 1:3 RR, ATR 1.5×) ----------
def simulate_window(start_equity=100.0, risk_pct=None):
    init_mt5()
    sym = SYMBOL

    # Window: 8 Sep to 12 Sep (UTC). We fetch extra history for indicator warmup.
    start = datetime(2025, 9, 8, 0, 0, tzinfo=timezone.utc)
    end   = datetime(2025, 9, 12, 23, 59, tzinfo=timezone.utc)
    warmback = start - timedelta(days=20)

    print(f"Symbol={sym}  Window={start.date()} → {end.date()}  (OB/OS module ON)")
    h1  = rates_range_df(sym, "H1",  warmback, end)
    m15 = rates_range_df(sym, "M15", warmback, end)
    m1  = rates_range_df(sym, "M1",  warmback, end)
    if h1.empty or m15.empty or m1.empty:
        print("No history returned; open charts / ensure MT5 is connected and cache is filled.")
        return

    # Add base indicators from your project (emas, macd, adx, atr, supertrend, vol_ma20, etc.)
    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    # Signals with OB/OS entries added
    long_sig, short_sig = build_signals_with_obos(h1i, m15i, m1i,
        warmup=300, adx_min=18, vol_mult=1.05, don_len=20, supertrend_req=True)

    # Restrict to [start, end]
    mask_window = (m1i.index >= start) & (m1i.index <= end)
    sig_idx = m1i.index[mask_window & (long_sig | short_sig)]
    if len(sig_idx)==0:
        print("No signals found in this window with OB/OS rules.")
        return

    # Risk & compounding
    if risk_pct is None:
        risk_pct = float(os.getenv("RISK_PCT", "0.02"))
    equity = float(start_equity)

    rows = []
    wins=losses=be_to=0
    total_R = 0.0
    lookahead = 150  # ~2.5 hours on M1

    for ts in sig_idx:
        row = m1i.loc[ts]
        atr = float(row.get("atr14", np.nan))
        if not np.isfinite(atr) or atr<=0: continue

        side = "buy" if bool(long_sig.loc[ts]) else "sell"
        entry = float(row["close"])

        # ATR SL/TP with 1:3 RR
        if side=="buy":
            sl = entry - 1.5*atr
            tp = entry + 3.0*(entry - sl)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            hit_tp = (fw["high"]>=tp)
            hit_sl = (fw["low"]<=sl)
        else:
            sl = entry + 1.5*atr
            tp = entry - 3.0*(sl - entry)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            hit_tp = (fw["low"]<=tp)
            hit_sl = (fw["high"]>=sl)

        # First-hit logic
        res_R, outcome = 0.0, "timeout"
        if hit_tp.any() and hit_sl.any():
            t_tp = fw.index[hit_tp.argmax()]
            t_sl = fw.index[hit_sl.argmax()]
            res_R, outcome = (3.0,"TP") if t_tp<=t_sl else (-1.0,"SL")
        elif hit_tp.any():
            res_R, outcome = 3.0, "TP"
        elif hit_sl.any():
            res_R, outcome = -1.0, "SL"

        # Compounding PnL
        risk_usd = equity * risk_pct
        pnl_usd  = res_R * risk_usd
        equity  += pnl_usd

        if   res_R>0: wins+=1
        elif res_R<0: losses+=1
        else: be_to+=1
        total_R += res_R

        rows.append({
            "time": ts.isoformat(),
            "side": side,
            "entry": round(entry,5),
            "sl": round(sl,5),
            "tp": round(tp,5),
            "atr": round(atr,5),
            "R": res_R,
            "risk_usd": round(risk_usd,2),
            "pnl_usd": round(pnl_usd,2),
            "equity_after": round(equity,2),
            "outcome": outcome
        })

    # Output
    df = pd.DataFrame(rows)
    os.makedirs("logs", exist_ok=True)
    out_csv = "logs/trades_obos_2025-09-08_2025-09-12.csv"
    df.to_csv(out_csv, index=False)

    considered = wins + losses
    winrate = (wins/considered*100.0) if considered>0 else 0.0

    print("\n=== TRADES (OB/OS + base) ===")
    pd.set_option("display.max_rows", None)
    print(df[["time","side","entry","sl","tp","atr","R","outcome","equity_after"]].to_string(index=False))

    print("\n=== SUMMARY ===")
    print(f"Start equity: $100.00   Risk per trade: {risk_pct*100:.1f}%   RR=1:3   SL=1.5*ATR")
    print(f"Trades: {len(df)}   Wins: {wins}   Losses: {losses}   BE/TO: {be_to}")
    print(f"Win rate: {winrate:.1f}%   Total PnL: {total_R:.1f} R   Final equity: ${equity:.2f}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    simulate_window(start_equity=100.0)
