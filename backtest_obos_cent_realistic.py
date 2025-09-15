import os, math, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone, time
from bot_mt5 import init_mt5, add_indicators, SYMBOL

def sample_exit_slip_pts(base_pts, atr_now, atr_ma):
    # ATR sensitivity + fat tails (Laplace) to mimic real exit slippage variability
    scale = 1.0 + 1.5*max(0.0, (atr_now/atr_ma - 1.0)) if (atr_ma and atr_ma > 0) else 1.0
    jitter = np.random.laplace(loc=0.0, scale=base_pts*0.35)
    slip = max(0.0, base_pts*scale + jitter)
    return min(slip, base_pts*4.0)


TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def rates_range_df(symbol, tf_key, start, end):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, end)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df.rename(columns={"tick_volume": "volume"})
    cols = ["open", "high", "low", "close", "volume"]
    if "spread" in df.columns:
        cols.append("spread")
    return df[cols]

# ---------- OB/OS indicators ----------
def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill()

def stoch_kd(h, l, c, n=14, d=3):
    hh = h.rolling(n).max(); ll = l.rolling(n).min()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d_ = k.rolling(d).mean()
    return k.bfill(), d_.bfill()

def williams_r(h, l, c, n=14):
    hh = h.rolling(n).max(); ll = l.rolling(n).min()
    wr = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    return wr.bfill()

def cci(h, l, c, n=20):
    tp = (h + l + c) / 3.0
    sma = tp.rolling(n).mean()
    md  = (tp - sma).abs().rolling(n).mean()
    cci_ = (tp - sma) / (0.015 * md.replace(0, np.nan))
    return cci_.bfill()

def mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3.0
    pmf = ((tp > tp.shift(1)) * (tp * v)).fillna(0.0)
    nmf = ((tp < tp.shift(1)) * (tp * v)).fillna(0.0)
    pmf_n = pmf.rolling(n).sum()
    nmf_n = nmf.rolling(n).sum().replace(0, np.nan)
    mr = pmf_n / nmf_n
    out = 100 - (100 / (1 + mr))
    return out.replace([np.inf, -np.inf], np.nan).bfill()

def bbands(c, n=20, k=2.0):
    ma = c.rolling(n).mean(); sd = c.rolling(n).std(ddof=0)
    upper = ma + k*sd; lower = ma - k*sd
    pb = (c - lower) / (upper - lower)
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

# ---------- Top-down + BOS ----------
def topdown_masks(h1i, m15i, m1i, adx_min=18):
    # Fallbacks if HTF indicators missing
    if "ema50" not in h1i.columns:
        h1i["ema50"] = h1i["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    if "ema200" not in h1i.columns:
        h1i["ema200"] = h1i["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    if "adx14" not in h1i.columns:
        tr = pd.concat([(h1i["high"] - h1i["low"]),
                        (h1i["high"] - h1i["close"].shift(1)).abs(),
                        (h1i["low"]  - h1i["close"].shift(1)).abs()], axis=1).max(axis=1)
        plus_dm  = h1i["high"].diff().clip(lower=0).fillna(0)
        minus_dm = (-h1i["low"].diff()).clip(lower=0).fillna(0)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di  = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        h1i["adx14"] = dx.ewm(alpha=1/14, adjust=False).mean().bfill()

    h1_up   = (h1i["ema50"] > h1i["ema200"]) & (h1i["adx14"] > adx_min)
    h1_down = (h1i["ema50"] < h1i["ema200"]) & (h1i["adx14"] > adx_min)

    h1_up_m1   = h1_up.reindex(m1i.index).ffill().astype("boolean").fillna(False).astype(bool)
    h1_down_m1 = h1_down.reindex(m1i.index).ffill().astype("boolean").fillna(False).astype(bool)

    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(3).max().astype(bool)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(3).max().astype(bool)

    bos_up_m1   = bos_up_recent.reindex(m1i.index).ffill().astype("boolean").fillna(False).astype(bool)
    bos_down_m1 = bos_down_recent.reindex(m1i.index).ffill().astype("boolean").fillna(False).astype(bool)
    return h1_up_m1, h1_down_m1, bos_up_m1, bos_down_m1

# ---------- Candles ----------
def bullish_engulf(o, h, l, c):
    prev_c = c.shift(1); prev_o = o.shift(1)
    return (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)

def bearish_engulf(o, h, l, c):
    prev_c = c.shift(1); prev_o = o.shift(1)
    return (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)

def pin_bull(o, h, l, c, t=2.0):
    body = (c - o).abs()
    lw = (o.where(c >= o, c) - l).abs()
    hw = (h - c.where(c >= o, o)).abs()
    return (c >= o) & (lw > t*body) & (hw < t*body)

def pin_bear(o, h, l, c, t=2.0):
    body = (c - o).abs()
    hw = (h - c.where(c <= o, o)).abs()
    lw = (o.where(c <= o, c) - l).abs()
    return (c <= o) & (hw > t*body) & (lw < t*body)

# ---------- Sessions / cooldown ----------
LAGOS = "Africa/Lagos"
def session_mask(index_utc, windows_lagos):
    idx = index_utc.tz_convert(LAGOS)
    t = pd.Series(idx.time, index=index_utc)
    mask = pd.Series(False, index=index_utc, dtype=bool)
    for a, b in windows_lagos:
        h1, m1 = map(int, a.split(":")); h2, m2 = map(int, b.split(":"))
        t1 = time(h1, m1); t2 = time(h2, m2)
        mask = mask | ((t >= t1) & (t <= t2))
    return mask

def session_mask_index(index_utc):
    return session_mask(index_utc, [("07:00","11:30"), ("13:30","17:00")])

def cool_down_filter(index, minutes=20, per_hour_cap=3):
    allow = pd.Series(True, index=index)
    last_ts = None; count_hour = 0; hour_start = None
    for ts in index:
        if hour_start is None or (ts - hour_start) >= timedelta(hours=1):
            hour_start = ts; count_hour = 0
        if last_ts is not None and (ts - last_ts) < timedelta(minutes=minutes):
            allow.loc[ts] = False; continue
        if count_hour >= per_hour_cap:
            allow.loc[ts] = False; continue
        allow.loc[ts] = True
        if allow.loc[ts]:
            last_ts = ts; count_hour += 1
    return allow

# ---------- Build signals (OB/OS v2 + core) ----------
def build_signals_v2(h1i, m15i, m1i, *, warmup=300, adx_min=20, vol_mult=1.10, don_len=20, supertrend_req=True):
    h1_up_m1, h1_down_m1, bos_up_m1, bos_down_m1 = topdown_masks(h1i, m15i, m1i, adx_min=adx_min)

    ema_up   = (m1i["ema9"] > m1i["ema21"]) & (m1i["ema21"] > m1i["ema50"])
    ema_dn   = (m1i["ema9"] < m1i["ema21"]) & (m1i["ema21"] < m1i["ema50"])
    macd_up  = (m1i["macdh"] > 0) & (m1i["macdh"] > m1i["macdh"].shift(1))
    macd_dn  = (m1i["macdh"] < 0) & (m1i["macdh"] < m1i["macdh"].shift(1))
    st_up    = (m1i["st_dir"] == 1)
    st_dn    = (m1i["st_dir"] == -1)
    adx_ok   = (m1i["adx14"] > adx_min)
    vol_ok   = (m1i["volume"] > (vol_mult * m1i["vol_ma20"]))

    dch = m1i["high"].rolling(don_len).max()
    dcl = m1i["low"].rolling(don_len).min()
    brk_up = (m1i["close"] > dch) & (m1i["close"].shift(1) <= dch.shift(1))
    brk_dn = (m1i["close"] < dcl) & (m1i["close"].shift(1) >= dcl.shift(1))

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False
    long_momo  = valid & ema_up & macd_up & adx_ok & vol_ok & (st_up if supertrend_req else True)
    short_momo = valid & ema_dn & macd_dn & adx_ok & vol_ok & (st_dn if supertrend_req else True)
    long_brk   = valid & ema_up   & adx_ok & vol_ok & brk_up & (st_up if supertrend_req else True)
    short_brk  = valid & ema_dn   & adx_ok & vol_ok & brk_dn & (st_dn if supertrend_req else True)

    m1 = add_obos_cols(m1i)
    ob_flags = [
        (m1["rsi14"] > 70), (m1["stochK"] > 80), (m1["wr14"] > -20),
        (m1["cci20"] > 100), (m1["mfi14"] > 80), (m1["close"] >= m1["bbU"])
    ]
    os_flags = [
        (m1["rsi14"] < 30), (m1["stochK"] < 20), (m1["wr14"] < -80),
        (m1["cci20"] < -100), (m1["mfi14"] < 20), (m1["close"] <= m1["bbL"])
    ]
    ob_score = sum(f.astype(int) for f in ob_flags)
    os_score = sum(f.astype(int) for f in os_flags)

    m15_anchor_ema = m15i["ema21"].reindex(m1i.index).ffill()
    m15_atr = m15i["atr14"].reindex(m1i.index).ffill()
    m15_bbU, _, m15_bbL, _ = bbands(m15i["close"], 20, 2.0)
    m15_bbU = m15_bbU.reindex(m1i.index).ffill()
    m15_bbL = m15_bbL.reindex(m1i.index).ffill()

    dist_to_m15ema = (m1i["close"] - m15_anchor_ema).abs()
    long_anchor  = ((dist_to_m15ema <= 0.5*m15_atr) | (m1i["close"] <= m15_bbL))
    short_anchor = ((dist_to_m15ema <= 0.5*m15_atr) | (m1i["close"] >= m15_bbU))

    recross_up = ((m1["stochK"] > m1["stochD"]) & (m1["stochK"].shift(1) <= m1["stochD"].shift(1))) | \
                 ((m1["rsi14"] > 30) & (m1["rsi14"].shift(1) <= 30)) | \
                 ((m1i["close"] > m1["bbL"]) & (m1i["close"].shift(1) <= m1["bbL"].shift(1)))
    recross_dn = ((m1["stochK"] < m1["stochD"]) & (m1["stochK"].shift(1) >= m1["stochD"].shift(1))) | \
                 ((m1["rsi14"] < 70) & (m1["rsi14"].shift(1) >= 70)) | \
                 ((m1i["close"] < m1["bbU"]) & (m1i["close"].shift(1) >= m1["bbU"].shift(1)))

    bull_candle = bullish_engulf(m1i["open"], m1i["high"], m1i["low"], m1i["close"]) | pin_bull(m1i["open"], m1i["high"], m1i["low"], m1i["close"])
    bear_candle = bearish_engulf(m1i["open"], m1i["high"], m1i["low"], m1i["close"]) | pin_bear(m1i["open"], m1i["high"], m1i["low"], m1i["close"])

    long_obos  = valid & (os_score >= 4) & recross_up  & bull_candle & adx_ok & long_anchor  & (st_up if supertrend_req else True)
    short_obos = valid & (ob_score >= 4) & recross_dn  & bear_candle & adx_ok & short_anchor & (st_dn if supertrend_req else True)

    long_core  = (long_momo  | long_brk  | long_obos)
    short_core = (short_momo | short_brk | short_obos)

    long_sig  = (long_core  & h1_up_m1   & bos_up_m1).astype(bool)
    short_sig = (short_core & h1_down_m1 & bos_down_m1).astype(bool)

    sig_type = pd.Series("core", index=m1i.index, dtype="object")
    sig_type[long_obos | short_obos] = "obos"
    return long_sig, short_sig, sig_type

# ---------- Symbol helpers ----------
def _sym_info(symbol):
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError("symbol_info is None; is MT5 connected & symbol visible?")
    return si

def usd_per_lot_for_move(symbol, price_move):
    si = _sym_info(symbol)
    ticks = abs(price_move) / (si.trade_tick_size or si.point)
    return ticks * (si.trade_tick_value or 0.0)

def round_lots(symbol, lots):
    si = _sym_info(symbol)
    step = si.volume_step or 0.01
    lots = math.floor(lots / step) * step
    lots = max(si.volume_min or 0.01, min(lots, si.volume_max or lots))
    return round(lots, 2)

# ---------- Backtest (Cent realistic, Jan 1 -> Sep 12) ----------
def simulate_window(start_equity=20.0, risk_pct=None):
    np.random.seed(0)
    init_mt5()
    sym = SYMBOL

    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    end   = datetime(2025, 9, 12, 23, 59, tzinfo=timezone.utc)
    warmback = start - timedelta(days=30)

    risk_pct = float(os.getenv("RISK_PCT", "0.02")) if risk_pct is None else risk_pct
    LEVERAGE          = float(os.getenv("LEVERAGE", "2000"))
    SPREAD_CAP_POINTS = float(os.getenv("SPREAD_CAP_POINTS", "35"))
    SLIP_POINTS       = float(os.getenv("SLIP_POINTS", "5"))       # entry slip
    EXIT_SLIP_POINTS  = float(os.getenv("EXIT_SLIP_POINTS", "8"))  # exit slip (worse)
    COMM_PER_LOT_USD  = float(os.getenv("COMM_PER_LOT_USD", "0"))
    NEWS_ATR_MULT     = float(os.getenv("NEWS_ATR_MULT", "4"))


