# indicators.py  — OB/OS helpers used by the main strategy
import numpy as np
import pandas as pd

__all__ = [
    "rsi","stoch_kd","williams_r","cci","mfi","bbands","add_obos_cols"
]

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill()

def stoch_kd(h,l,c,n=14,d=3):
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d_ = k.rolling(d).mean()
    return k.bfill(), d_.bfill()

def williams_r(h,l,c,n=14):
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    wr = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    return wr.bfill()

def cci(h,l,c,n=20):
    tp = (h + l + c) / 3.0
    sma = tp.rolling(n).mean()
    md = (tp - sma).abs().rolling(n).mean()
    cci_ = (tp - sma) / (0.015 * md.replace(0, np.nan))
    return cci_.bfill()

def mfi(h,l,c,v,n=14):
    tp = (h + l + c) / 3.0
    pmf = ((tp > tp.shift(1)) * (tp * v)).fillna(0.0)
    nmf = ((tp < tp.shift(1)) * (tp * v)).fillna(0.0)
    pmf_n = pmf.rolling(n).sum()
    nmf_n = nmf.rolling(n).sum().replace(0, np.nan)
    mr = pmf_n / nmf_n
    out = 100 - (100 / (1 + mr))
    return out.replace([np.inf,-np.inf], np.nan).bfill()

def bbands(c,n=20,k=2.0):
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    upper = ma + k*sd
    lower = ma - k*sd
    pb = (c - lower) / (upper - lower)
    return upper, ma, lower, pb

def add_obos_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi14"] = rsi(out["close"], 14)
    out["stochK"], out["stochD"] = stoch_kd(out["high"], out["low"], out["close"], 14, 3)
    out["wr14"]  = williams_r(out["high"], out["low"], out["close"], 14)
    out["cci20"] = cci(out["high"], out["low"], out["close"], 20)
    out["mfi14"] = mfi(out["high"], out["low"], out["close"], out["volume"], 14)
    out["bbU"], out["bbM"], out["bbL"], out["pb"] = bbands(out["close"], 20, 2.0)
    return out
