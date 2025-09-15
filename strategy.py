# strategy.py — OB/OS v2 + Momentum/Breakout + BOS + Sessions + Cooldown + BE@+1R filters
# Mirrors your backtest_obos_window_v2 logic.
import numpy as np
import pandas as pd
from datetime import timedelta, time
from indicators import add_obos_cols, bbands

# Candle confirmations
def bullish_engulf(o,h,l,c):
    prev_c = c.shift(1); prev_o = o.shift(1)
    return (c>o) & (prev_c<prev_o) & (c>=prev_o) & (o<=prev_c)

def bearish_engulf(o,h,l,c):
    prev_c = c.shift(1); prev_o = o.shift(1)
    return (c<o) & (prev_c>prev_o) & (c<=prev_o) & (o>=prev_c)

def pin_bull(o,h,l,c, thresh=2.0):
    body = (c-o).abs()
    low_wick  = (o.where(c>=o, c) - l).abs()
    high_wick = (h - c.where(c>=o, o)).abs()
    return (c>=o) & (low_wick > thresh*body) & (high_wick < thresh*body)

def pin_bear(o,h,l,c, thresh=2.0):
    body = (c-o).abs()
    high_wick = (h - c.where(c<=o, o)).abs()
    low_wick  = (o.where(c<=o, c) - l).abs()
    return (c<=o) & (high_wick > thresh*body) & (low_wick < thresh*body)

# HTF regime + M15 BOS into M1
def topdown_masks(h1i, m15i, m1i, adx_min=20):
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_up_m1   = h1_up.reindex(m1i.index).ffill().astype("boolean").fillna(False).astype(bool)
    h1_down_m1 = h1_down.reindex(m1i.index).ffill().astype("boolean").fillna(False).astype(bool)

    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(3).max().astype(bool)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(3).max().astype(bool)
    bos_up_m1   = bos_up_recent.reindex(m1i.index).ffill().astype("boolean").fillna(False).astype(bool)
    bos_down_m1 = bos_down_recent.reindex(m1i.index).ffill().astype("boolean").fillna(False).astype(bool)
    return h1_up_m1, h1_down_m1, bos_up_m1, bos_down_m1

# Sessions & cooldown
LAGOS_TZ = "Africa/Lagos"
def session_mask(index_utc, windows_lagos):
    idx = index_utc.tz_convert(LAGOS_TZ)
    t = pd.Series(idx.time, index=index_utc)
    mask = pd.Series(False, index=index_utc, dtype=bool)
    for a,b in windows_lagos:
        h1,m1 = map(int, a.split(":")); h2,m2 = map(int, b.split(":"))
        t1 = time(h1,m1); t2 = time(h2,m2)
        mask = mask | ((t>=t1) & (t<=t2))
    return mask

def session_mask_index(index_utc):
    return session_mask(index_utc, [("07:00","11:30"), ("13:30","17:00")])

def cool_down_filter(index, minutes=20, per_hour_cap=3):
    allow = pd.Series(True, index=index)
    last_ts = None
    count_hour = 0
    hour_start = None
    for ts in index:
        if hour_start is None or (ts - hour_start) >= timedelta(hours=1):
            hour_start = ts
            count_hour = 0
        if last_ts is not None and (ts - last_ts) < timedelta(minutes=minutes):
            allow.loc[ts] = False
            continue
        if count_hour >= per_hour_cap:
            allow.loc[ts] = False
            continue
        allow.loc[ts] = True
        if allow.loc[ts]:
            last_ts = ts
            count_hour += 1
    return allow

# Core signal builder (OB/OS v2 + momentum + breakout)
def build_signals_v2(h1i, m15i, m1i, *, warmup=300, adx_min=20, vol_mult=1.10, don_len=20, supertrend_req=True):
    h1_up_m1, h1_down_m1, bos_up_m1, bos_down_m1 = topdown_masks(h1i, m15i, m1i, adx_min=adx_min)

    ema_up   = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    ema_dn   = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])
    macd_up  = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    macd_dn  = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    st_up    = (m1i["st_dir"]==1)
    st_dn    = (m1i["st_dir"]==-1)
    adx_ok   = (m1i["adx14"]>adx_min)
    vol_ok   = (m1i["volume"] > vol_mult*m1i["vol_ma20"])

    dch = m1i["high"].rolling(don_len).max()
    dcl = m1i["low"].rolling(don_len).min()
    brk_up = (m1i["close"]>dch) & (m1i["close"].shift(1)<=dch.shift(1))
    brk_dn = (m1i["close"]<dcl) & (m1i["close"].shift(1)>=dcl.shift(1))

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False

    long_momo  = valid & ema_up & macd_up & adx_ok & vol_ok & (st_up if supertrend_req else True)
    short_momo = valid & ema_dn & macd_dn & adx_ok & vol_ok & (st_dn if supertrend_req else True)
    long_brk   = valid & ema_up   & adx_ok & vol_ok & brk_up & (st_up if supertrend_req else True)
    short_brk  = valid & ema_dn   & adx_ok & vol_ok & brk_dn & (st_dn if supertrend_req else True)

    m1 = add_obos_cols(m1i)

    ob_flags = [
        (m1["rsi14"] > 70),
        (m1["stochK"] > 80),
        (m1["wr14"] > -20),
        (m1["cci20"] > 100),
        (m1["mfi14"] > 80),
        (m1["close"] >= m1["bbU"])
    ]
    os_flags = [
        (m1["rsi14"] < 30),
        (m1["stochK"] < 20),
        (m1["wr14"] < -80),
        (m1["cci20"] < -100),
        (m1["mfi14"] < 20),
        (m1["close"] <= m1["bbL"])
    ]
    ob_score = sum(f.astype(int) for f in ob_flags)
    os_score = sum(f.astype(int) for f in os_flags)

    m15_anchor_ema = m15i["ema21"].reindex(m1i.index).ffill()
    m15_atr = m15i["atr14"].reindex(m1i.index).ffill()
    m15_bbU, m15_bbM, m15_bbL, _ = bbands(m15i["close"], 20, 2.0)
    m15_bbU = m15_bbU.reindex(m1i.index).ffill()
    m15_bbL = m15_bbL.reindex(m1i.index).ffill()

    dist_to_m15ema = (m1i["close"] - m15_anchor_ema).abs()
    long_anchor  = ( (dist_to_m15ema <= 0.5*m15_atr) | (m1i["close"] <= m15_bbL) )
    short_anchor = ( (dist_to_m15ema <= 0.5*m15_atr) | (m1i["close"] >= m15_bbU) )

    recross_up = ((m1["stochK"] > m1["stochD"]) & (m1["stochK"].shift(1) <= m1["stochD"].shift(1))) | \
                 ((m1["rsi14"] > 30) & (m1["rsi14"].shift(1) <= 30)) | \
                 ((m1i["close"] > m1["bbL"]) & (m1i["close"].shift(1) <= m1["bbL"].shift(1)))
    recross_dn = ((m1["stochK"] < m1["stochD"]) & (m1["stochK"].shift(1) >= m1["stochD"].shift(1))) | \
                 ((m1["rsi14"] < 70) & (m1["rsi14"].shift(1) >= 70)) | \
                 ((m1i["close"] < m1["bbU"]) & (m1i["close"].shift(1) >= m1["bbU"].shift(1)))

    bull_candle = bullish_engulf(m1i["open"], m1i["high"], m1i["low"], m1i["close"]) | pin_bull(m1i["open"], m1i["high"], m1i["low"], m1i["close"])
    bear_candle = bearish_engulf(m1i["open"], m1i["high"], m1i["low"], m1i["close"]) | pin_bear(m1i["open"], m1i["high"], m1i["low"], m1i["close"])

    long_obos  = valid & (os_score>=4) & recross_up  & bull_candle & adx_ok & long_anchor  & (st_up if supertrend_req else True)
    short_obos = valid & (ob_score>=4) & recross_dn  & bear_candle & adx_ok & short_anchor & (st_dn if supertrend_req else True)

    long_core  = (long_momo  | long_brk  | long_obos)
    short_core = (short_momo | short_brk | short_obos)

    long_sig  = (long_core  & h1_up_m1   & bos_up_m1).astype(bool)
    short_sig = (short_core & h1_down_m1 & bos_down_m1).astype(bool)
    return long_sig, short_sig

def filter_signal_times(m1_index, long_sig, short_sig, *, start=None, end=None, cooldown_min=20, per_hour_cap=3):
    mask = pd.Series(True, index=m1_index)
    if start is not None:
        mask &= (m1_index >= start)
    if end is not None:
        mask &= (m1_index <= end)
    session_ok = session_mask_index(m1_index)
    raw_idx = m1_index[mask & session_ok & (long_sig | short_sig)]
    allowed = cool_down_filter(raw_idx, minutes=cooldown_min, per_hour_cap=per_hour_cap)
    return raw_idx[allowed.loc[raw_idx].values]
