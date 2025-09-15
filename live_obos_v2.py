import os, math, time as _time, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone, time
HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_SEC","5"))  # how often to print a heartbeat
POLL_SEC      = float(os.getenv("POLL_SEC","1.0"))     # loop sleep seconds

_next_hb = 0.0
def maybe_heartbeat(m1df=None):
    """Periodic tiny log so you know it's alive."""
    import time as _t
    global _next_hb
    now = _t.time()
    if now < _next_hb:
        return
    _next_hb = now + HEARTBEAT_SEC
    try:
        si = mt5.symbol_info(SYMBOL)
        spr = si.spread if si else None
    except Exception:
        spr = None
    lastc = "n/a"
    try:
        if isinstance(m1df, pd.DataFrame) and not m1df.empty:
            lastc = m1df.index[-1].isoformat()
    except Exception:
        pass
    print(f"[hb] {datetime.now(timezone.utc).strftime('%H:%M:%S')} waiting for M1 close | last_bar={lastc} | spread={spr} pts", flush=True)
import traceback

SYMBOL  = os.getenv("SYMBOL", "EURUSD")
RISK_PCT = float(os.getenv("RISK_PCT", "0.02"))
SPREAD_CAP_POINTS = float(os.getenv("SPREAD_CAP_POINTS", "25"))
NEWS_ATR_MULT     = float(os.getenv("NEWS_ATR_MULT", "4.0"))
DEVIATION_POINTS  = int(os.getenv("DEVIATION_POINTS", "10"))
MAGIC             = int(os.getenv("MAGIC", "20250912"))
COMMENT           = os.getenv("COMMENT", "obos_v2_live")
LAGOS_TZ          = "Africa/Lagos"

def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    si = mt5.symbol_info(SYMBOL)
    if si is None: raise RuntimeError("symbol_info is None; is MT5 connected?")
    if not si.visible: mt5.symbol_select(SYMBOL, True)

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
def rates_range_df(symbol, tf_key, start, end):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, end)
    if rates is None or len(rates)==0: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df.rename(columns={"tick_volume":"volume"})
    return df[["open","high","low","close","volume"]]

def rsi(close, n=14):
    d = close.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean().replace(0,np.nan)
    return (100 - 100/(1+rs)).bfill()

def stoch_kd(h,l,c,n=14,d=3):
    hh=h.rolling(n).max(); ll=l.rolling(n).min()
    k = 100*(c-ll)/(hh-ll).replace(0,np.nan); d_=k.rolling(d).mean()
    return k.bfill(), d_.bfill()

def williams_r(h,l,c,n=14):
    hh=h.rolling(n).max(); ll=l.rolling(n).min()
    return (-100*(hh-c)/(hh-ll).replace(0,np.nan)).bfill()

def cci(h,l,c,n=20):
    tp=(h+l+c)/3; sma=tp.rolling(n).mean(); md=(tp-sma).abs().rolling(n).mean()
    return ((tp-sma)/(0.015*md.replace(0,np.nan))).bfill()

def mfi(h,l,c,v,n=14):
    tp=(h+l+c)/3; pmf=((tp>tp.shift(1))*(tp*v)).fillna(0.0); nmf=((tp<tp.shift(1))*(tp*v)).fillna(0.0)
    mr=pmf.rolling(n).sum()/nmf.rolling(n).sum().replace(0,np.nan)
    return (100-100/(1+mr)).replace([np.inf,-np.inf],np.nan).bfill()

def bbands(c,n=20,k=2.0):
    ma=c.rolling(n).mean(); sd=c.rolling(n).std(ddof=0)
    upper=ma+k*sd; lower=ma-k*sd; pb=(c-lower)/(upper-lower)
    return upper, ma, lower, pb

def add_obos_cols(df):
    out=df.copy()
    out["rsi14"]=rsi(out["close"],14)
    out["stochK"], out["stochD"]=stoch_kd(out["high"],out["low"],out["close"],14,3)
    out["wr14"]=williams_r(out["high"],out["low"],out["close"],14)
    out["cci20"]=cci(out["high"],out["low"],out["close"],20)
    out["mfi14"]=mfi(out["high"],out["low"],out["close"],out["volume"],14)
    out["bbU"], out["bbM"], out["bbL"], out["pb"]=bbands(out["close"],20,2.0)
    out["ema9"]=out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"]=out["close"].ewm(span=21, adjust=False).mean()
    out["ema50"]=out["close"].ewm(span=50, adjust=False).mean()
    tr = pd.concat([
        (out["high"]-out["low"]),
        (out["high"]-out["close"].shift(1)).abs(),
        (out["low"]-out["close"].shift(1)).abs()
    ],axis=1).max(axis=1)
    out["atr14"]=tr.rolling(14).mean().bfill()
    # ensure Series (not ndarray) types:
    out["adx14"]=pd.Series(25.0, index=out.index)                 # placeholder; replace with real ADX if you have it
    st = pd.Series(np.sign(out["ema21"].diff()), index=out.index)
    out["st_dir"]=st.replace(0,1)
    out["vol_ma20"]=out["volume"].rolling(20).mean().bfill()
    return out

def bullish_engulf(o,h,l,c):
    prev_c = c.shift(1); prev_o = o.shift(1)
    return (c>o) & (prev_c<prev_o) & (c>=prev_o) & (o<=prev_c)

def bearish_engulf(o,h,l,c):
    prev_c = c.shift(1); prev_o = o.shift(1)
    return (c<o) & (prev_c>prev_o) & (c<=prev_o) & (o>=prev_c)

def pin_bull(o,h,l,c,t=2.0):
    body=(c-o).abs(); lw=(o.where(c>=o,c)-l).abs(); hw=(h-c.where(c>=o,o)).abs()
    return (c>=o)&(lw>t*body)&(hw<t*body)

def pin_bear(o,h,l,c,t=2.0):
    body=(c-o).abs(); hw=(h-c.where(c<=o,o)).abs(); lw=(o.where(c<=o,c)-l).abs()
    return (c<=o)&(hw>t*body)&(lw<t*body)

def topdown_masks(h1i, m15i, m1i, adx_min=18):
    """
    Build HTF (H1) regime and M15 BOS masks.
    Ensures required H1 columns exist (ema50, ema200, adx14) so live runs don't crash.
    """
    # --- Ensure H1 indicators exist ---
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

    # Align to M1 index
    h1_up_m1   = h1_up.reindex(m1i.index).ffill().astype(bool)
    h1_down_m1 = h1_down.reindex(m1i.index).ffill().astype(bool)

    # --- M15 BOS (recent breakout of prior Donchian) ---
    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(3).max().astype(bool)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(3).max().astype(bool)

    bos_up_m1   = bos_up_recent.reindex(m1i.index).ffill().astype(bool)
    bos_down_m1 = bos_down_recent.reindex(m1i.index).ffill().astype(bool)

    return h1_up_m1, h1_down_m1, bos_up_m1, bos_down_m1
def ensure_m1_cols(m1i: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required M1 columns exist for live trading:
      ema9/ema21/ema50, atr14, adx14, vol_ma20, macdh, st_dir (fallback).
    """
    out = m1i.copy()

    # EMAs
    for span, name in [(9,"ema9"), (21,"ema21"), (50,"ema50")]:
        if name not in out.columns:
            out[name] = out["close"].ewm(span=span, adjust=False, min_periods=span).mean()

    # True Range base
    tr = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"]  - out["close"].shift(1)).abs()
    ], axis=1).max(axis=1)

    # ATR14
    if "atr14" not in out.columns:
        out["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # ADX14
    if "adx14" not in out.columns:
        plus_dm  = out["high"].diff().clip(lower=0).fillna(0)
        minus_dm = (-out["low"].diff()).clip(lower=0).fillna(0)
        atr_for_di = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di  = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_for_di)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_for_di)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        out["adx14"] = dx.ewm(alpha=1/14, adjust=False).mean().bfill()

    # Volume MA
    if "vol_ma20" not in out.columns:
        out["vol_ma20"] = out["volume"].rolling(20).mean().bfill()

    # MACD histogram (12,26,9)
    if "macdh" not in out.columns:
        ema12 = out["close"].ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = out["close"].ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        out["macdh"] = macd - signal

    # Supertrend direction fallback: use EMA21 as simple proxy if missing
    if "st_dir" not in out.columns:
        out["st_dir"] = np.where(out["close"] >= out["ema21"], 1, -1)

    return out
def build_signals_v2(h1i,m15i,m1i,*,warmup=300,adx_min=20,vol_mult=1.10,don_len=20,supertrend_req=True):
    m1i = ensure_m1_cols(m1i)
    h1_up_m1,h1_dn_m1,bos_up_m1,bos_dn_m1 = topdown_masks(h1i,m15i,m1i,adx_min)
    ema_up=(m1i["ema9"]>m1i["ema21"])&(m1i["ema21"]>m1i["ema50"])
    ema_dn=(m1i["ema9"]<m1i["ema21"])&(m1i["ema21"]<m1i["ema50"])
    macd_up=(m1i["macdh"]>0)&(m1i["macdh"]>m1i["macdh"].shift(1))
    macd_dn=(m1i["macdh"]<0)&(m1i["macdh"]<m1i["macdh"].shift(1))
    st_up=(m1i["st_dir"]==1); st_dn=(m1i["st_dir"]==-1)
    adx_ok=(m1i["adx14"]>adx_min)
    vol_ok=(m1i["volume"]>vol_mult*m1i["vol_ma20"])
    dch=m1i["high"].rolling(don_len).max(); dcl=m1i["low"].rolling(don_len).min()
    brk_up=(m1i["close"]>dch)&(m1i["close"].shift(1)<=dch.shift(1))
    brk_dn=(m1i["close"]<dcl)&(m1i["close"].shift(1)>=dcl.shift(1))
    valid=pd.Series(True,index=m1i.index); valid.iloc[:warmup]=False
    long_momo=valid&ema_up&macd_up&adx_ok&vol_ok&(st_up if supertrend_req else True)
    short_momo=valid&ema_dn&macd_dn&adx_ok&vol_ok&(st_dn if supertrend_req else True)
    long_brk=valid&ema_up&adx_ok&vol_ok&brk_up&(st_up if supertrend_req else True)
    short_brk=valid&ema_dn&adx_ok&vol_ok&brk_dn&(st_dn if supertrend_req else True)

    m1=add_obos_cols(m1i)
    ob_flags=[(m1["rsi14"]>70),(m1["stochK"]>80),(m1["wr14"]>-20),(m1["cci20"]>100),(m1["mfi14"]>80),(m1["close"]>=m1["bbU"])]
    os_flags=[(m1["rsi14"]<30),(m1["stochK"]<20),(m1["wr14"]<-80),(m1["cci20"]<-100),(m1["mfi14"]<20),(m1["close"]<=m1["bbL"])]
    ob_score=sum(f.astype(int) for f in ob_flags)
    os_score=sum(f.astype(int) for f in os_flags)
    m15_ema = m15i["ema21"].reindex(m1i.index).ffill()
    m15_atr = m15i["atr14"].reindex(m1i.index).ffill()
    m15_bbU,_,m15_bbL,_ = bbands(m15i["close"],20,2.0)
    m15_bbU=m15_bbU.reindex(m1i.index).ffill(); m15_bbL=m15_bbL.reindex(m1i.index).ffill()
    dist=(m1i["close"]-m15_ema).abs()
    long_anchor=((dist<=0.5*m15_atr)|(m1i["close"]<=m15_bbL))
    short_anchor=((dist<=0.5*m15_atr)|(m1i["close"]>=m15_bbU))
    rec_up=((m1["stochK"]>m1["stochD"])&(m1["stochK"].shift(1)<=m1["stochD"].shift(1)))|((m1["rsi14"]>30)&(m1["rsi14"].shift(1)<=30))|((m1i["close"]>m1["bbL"])&(m1i["close"].shift(1)<=m1["bbL"].shift(1)))
    rec_dn=((m1["stochK"]<m1["stochD"])&(m1["stochK"].shift(1)>=m1["stochD"].shift(1)))|((m1["rsi14"]<70)&(m1["rsi14"].shift(1)>=70))|((m1i["close"]<m1["bbU"])&(m1i["close"].shift(1)>=m1["bbU"].shift(1)))
    bull = bullish_engulf(m1i["open"],m1i["high"],m1i["low"],m1i["close"]) | pin_bull(m1i["open"],m1i["high"],m1i["low"],m1i["close"])
    bear = bearish_engulf(m1i["open"],m1i["high"],m1i["low"],m1i["close"]) | pin_bear(m1i["open"],m1i["high"],m1i["low"],m1i["close"])
    long_obos = valid&(os_score>=4)&rec_up&bull&adx_ok&long_anchor&(st_up if supertrend_req else True)
    short_obos= valid&(ob_score>=4)&rec_dn&bear&adx_ok&short_anchor&(st_dn if supertrend_req else True)
    long_core = long_momo|long_brk|long_obos
    short_core= short_momo|short_brk|short_obos
    long_sig =(long_core & h1_up_m1 & bos_up_m1).astype(bool)
    short_sig=(short_core& h1_dn_m1 & bos_dn_m1).astype(bool)
    return long_sig, short_sig, m1

def session_mask_index(index_utc):
    idx = index_utc.tz_convert(LAGOS_TZ)
    t = pd.Series(idx.time, index=index_utc)
    mask = pd.Series(False, index=index_utc, dtype=bool)
    for a,b in [("07:00","11:30"),("13:30","17:00")]:
        h1,m1 = map(int,a.split(":")); h2,m2=map(int,b.split(":"))
        t1=time(h1,m1); t2=time(h2,m2)
        mask = mask | ((t>=t1)&(t<=t2))
    return mask

def _sym_info(symbol):
    si=mt5.symbol_info(symbol)
    if si is None: raise RuntimeError("symbol_info is None")
    return si

def usd_per_lot_for_move(symbol, price_move):
    si=_sym_info(symbol)
    ticks = abs(price_move)/(si.trade_tick_size or si.point)
    return ticks*(si.trade_tick_value or 0.0)

def round_lots(symbol, lots):
    """
    If the raw lot size is below broker min volume, return 0.0 (skip the trade)
    unless ALLOW_MIN_LOT_IF_BELOW_RISK=1, which will force min lot (can over-risk).
    """
    si = _sym_info(symbol)
    step = si.volume_step or 0.01
    minv = si.volume_min or 0.01
    maxv = si.volume_max or lots
    allow_min = os.getenv("ALLOW_MIN_LOT_IF_BELOW_RISK", "0") == "1"

    # If we can't reach min lot without exceeding risk, skip by default
    if lots < minv:
        return (minv if allow_min else 0.0)

    lots = math.floor(lots/step) * step
    lots = max(minv, min(lots, maxv))
    return round(lots, 2)

def place_market_rr(symbol, side, atr, rr=3.0, atr_mult=1.5):
    tick=mt5.symbol_info_tick(symbol)
    if tick is None: return False,"no_tick"
    si=_sym_info(symbol)
    pt=si.point
    bid=tick.bid; ask=tick.ask
    price = ask if side=="buy" else bid
    sl = price - atr_mult*atr if side=="buy" else price + atr_mult*atr
    tp = price + rr*(price - sl) if side=="buy" else price - rr*(sl - price)
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.0,
        "type": mt5.ORDER_TYPE_BUY if side=="buy" else mt5.ORDER_TYPE_SELL,
        "price": price, "sl": sl, "tp": tp,
        "deviation": DEVIATION_POINTS,
        "magic": MAGIC, "comment": COMMENT,
        "type_filling": mt5.ORDER_FILLING_FOK if (si.trade_fill_mode & mt5.SYMBOL_FILLING_FOK) else mt5.ORDER_FILLING_IOC,
    }
    risk_usd = max(0.0, float(getattr(place_market_rr,"equity",100.0))*RISK_PCT)
    stop_px = abs(price - sl)
    usd_per_lot = usd_per_lot_for_move(symbol, stop_px) or 0.0
    if usd_per_lot<=0: return False,"bad_usd_per_lot"
    lots = round_lots(symbol, risk_usd/usd_per_lot)
    if lots<=0: return False,"lots_le_zero"
    req["volume"]=lots
    spread_pts = (ask-bid)/pt
    if spread_pts>SPREAD_CAP_POINTS:
        return False, f"spread_cap({spread_pts:.1f}pts)"
    res=mt5.order_send(req)
    ok = (res is not None) and (res.retcode==mt5.TRADE_RETCODE_DONE)
    return ok, {"retcode": getattr(res,'retcode',None), "price": price, "sl": sl, "tp": tp, "lots": lots, "spread_pts": spread_pts}

def breakeven_manager():
    poss = mt5.positions_get(symbol=SYMBOL)
    if not poss: return
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None: return
    for p in poss:
        if p.magic!=MAGIC: continue
        entry=p.price_open; sl=p.sl; tp=p.tp
        if sl is None or tp is None: continue
        r = abs(entry - sl)
        if p.type==mt5.POSITION_TYPE_BUY:
            cur = tick.bid; hit1R = cur >= entry + r; new_sl = entry if hit1R and (sl < entry) else sl
        else:
            cur = tick.ask; hit1R = cur <= entry - r; new_sl = entry if hit1R and (sl > entry) else sl
        if new_sl!=sl:
            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": p.symbol, "sl": new_sl, "tp": tp, "magic": MAGIC, "comment": COMMENT+"|BE"})

def _as_bool_at(series, ts):
    """Return a safe scalar bool for series value at timestamp ts (handles duplicates)."""
    try:
        v = series.loc[ts]
    except KeyError:
        return False
    if isinstance(v, pd.Series):
        if v.empty: return False
        return bool(v.iloc[-1])
    if isinstance(v, (np.ndarray, list, tuple)):
        return bool(np.asarray(v).ravel()[-1])
    return bool(v)

def main():
    init_mt5()
    print(f"Live OB/OS v2 runner on {SYMBOL}  (candle-close only, spread cap, SL/TP immediate, BE@1R)")
    last_bar_time=None; last_trade_ts=None; hour_count=0; hour_start=None
    while True:
        _obj = locals().get("m1i", None)
        if _obj is None:
            _obj = locals().get("m1", None)
        maybe_heartbeat(_obj)
        try:
            m1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 3)
            if m1 is None or len(m1) < 2:
                _time.sleep(POLL_SEC); continue
            closed = m1[-2]
            bar_time = datetime.fromtimestamp(closed["time"], tz=timezone.utc)
            if last_bar_time is not None and bar_time<=last_bar_time:
                breakeven_manager(); _time.sleep(POLL_SEC); continue
            last_bar_time = bar_time

            end = bar_time; start = end - timedelta(days=30)
            h1  = rates_range_df(SYMBOL,"H1", start, end)
            m15 = rates_range_df(SYMBOL,"M15",start, end)
            m1  = rates_range_df(SYMBOL,"M1", start, end)
            if h1.empty or m15.empty or m1.empty:
                print("No history; skipping."); _time.sleep(POLL_SEC); continue

            h1i = add_obos_cols(h1); m15i = add_obos_cols(m15); m1i = add_obos_cols(m1)
            long_sig, short_sig, _ = build_signals_v2(h1i, m15i, m1i, warmup=300, adx_min=20, vol_mult=1.10, don_len=20, supertrend_req=True)

            ts = m1i.index[-2]
            in_sess = session_mask_index(m1i.index).iloc[-2]
            now = bar_time
            if hour_start is None or (now - hour_start)>=timedelta(hours=1):
                hour_start = now; hour_count=0
            cooldown_ok = (last_trade_ts is None) or ((now - last_trade_ts) >= timedelta(minutes=20))
            cap_ok = hour_count < 3
            if (not bool(in_sess)) or (not cooldown_ok) or (not cap_ok):
                breakeven_manager()
                _time.sleep(POLL_SEC); continue

            row = m1i.loc[ts]
            atr = float(row.get("atr14", np.nan))
            if not np.isfinite(atr) or atr<=0:
                _time.sleep(POLL_SEC); continue

            if (float(row["high"]) - float(row["low"])) > NEWS_ATR_MULT * atr:
                breakeven_manager(); _time.sleep(POLL_SEC); continue

            # SAFE scalar booleans for signals
            is_long  = _as_bool_at(long_sig, ts)
            is_short = _as_bool_at(short_sig, ts)
            side = "buy" if is_long else ("sell" if is_short else None)
            if side is None:
                breakeven_manager(); _time.sleep(POLL_SEC); continue

            if not hasattr(place_market_rr, "equity"):
                place_market_rr.equity = 100.0  # local equity tracker for sizing
            ok, info = place_market_rr(SYMBOL, side, atr, rr=3.0, atr_mult=1.5)
            if ok:
                last_trade_ts = now; hour_count += 1
                print(f"{now.isoformat()}  SENT {side.upper()}  lots={info['lots']}  px={info['price']:.5f}  sl={info['sl']:.5f}  tp={info['tp']:.5f}  spr={info['spread_pts']:.1f}pts  ret={info['retcode']}")
            else:
                print(f"{now.isoformat()}  SKIP {side}  reason={info}")
            breakeven_manager()
        except Exception as e:
            print("loop error:", e)
            traceback.print_exc()
        _time.sleep(POLL_SEC)

if __name__=="__main__":
    init_mt5()
    print(f"MT5 initialized. Watching {SYMBOL}")
    main()










