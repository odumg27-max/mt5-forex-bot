import os, sys, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone, time
from bot_mt5 import init_mt5, add_indicators, SYMBOL

print = lambda *a, **k: (sys.stdout.write((" ".join(map(str,a)) + "\n")), sys.stdout.flush())
pd.set_option("future.no_silent_downcasting", True)

try:
    from zoneinfo import ZoneInfo
    LAGOS = ZoneInfo("Africa/Lagos")
except Exception:
    LAGOS = timezone(timedelta(hours=1))

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def _ffill_bool_to_index(s: pd.Series, index) -> pd.Series:
    return (s.astype("boolean").reindex(index).ffill().fillna(False).astype(bool))

def rates_range_df(symbol, tf_key, start_utc, end_utc):
    r = mt5.copy_rates_range(symbol, TF[tf_key], start_utc, end_utc)
    if r is None or len(r)==0: return pd.DataFrame()
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def precompute_common(m1i):
    base = {}
    base["ema_up"]   = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    base["ema_dn"]   = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])
    base["macd_up"]  = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    base["macd_dn"]  = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    base["st_up"]    = (m1i["st_dir"]==1)
    base["st_dn"]    = (m1i["st_dir"]==-1)
    dch = m1i["high"].rolling(20).max()
    dcl = m1i["low"].rolling(20).min()
    base["brk_up"] = (m1i["close"]>dch) & (m1i["close"].shift(1)<=dch.shift(1))
    base["brk_dn"] = (m1i["close"]<dcl) & (m1i["close"].shift(1)>=dcl.shift(1))
    base["adx14"] = m1i["adx14"]
    base["vol_ratio"] = (m1i["volume"] / m1i["vol_ma20"].replace(0,np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0)
    base["atr14"] = m1i["atr14"]
    base["atr_rel"] = m1i["atr14"] > m1i["atr14"].rolling(1000).median()
    ma = m1i["close"].rolling(20).mean(); std = m1i["close"].rolling(20).std()
    bbw = (4*std)/ma.clip(lower=1e-9)
    def _pctl(x):
        x = pd.Series(x).dropna()
        if len(x)==0: return np.nan
        return (x.rank(pct=True).iloc[-1])
    pctl = bbw.rolling(150).apply(_pctl, raw=False)
    base["squeeze_true"]  = pctl.lt(0.35).rolling(15).max().fillna(False).astype(bool)
    base["squeeze_false"] = pd.Series(True, index=m1i.index, dtype=bool)
    return base

def build_bos_m15(m15i):
    out = {}
    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    for w in (1,2,3):
        bos_up_recent   = (m15i["close"] > prior_high).rolling(w).sum().gt(0)
        bos_down_recent = (m15i["close"] < prior_low ).rolling(w).sum().gt(0)
        out[w] = (bos_up_recent, bos_down_recent)
    return out

def regime_h1(h1i, adx_min):
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    slope200 = h1i["ema200"].ewm(span=20, adjust=False).mean() - h1i["ema200"].shift(20)
    macd_pos = (h1i.get("macdh", pd.Series(index=h1i.index, data=0)) > 0)
    macd_neg = ~macd_pos
    h1_up   = (h1_up  & (slope200>0) & macd_pos)
    h1_down = (h1_down & (slope200<0) & macd_neg)
    return h1_up, h1_down

def build_signals_cached(h1i, m15i, m1i, base, bos_cache, *, warmup=300,
                         adx_min=18, bos_w=2, vol_mult=1.10,
                         entry_mode="either", supertrend_req=True, squeeze=False):
    h1_up, h1_down = regime_h1(h1i, adx_min)
    h1_up_m1   = _ffill_bool_to_index(h1_up,   m1i.index)
    h1_down_m1 = _ffill_bool_to_index(h1_down, m1i.index)
    bos_up_recent, bos_down_recent = bos_cache[bos_w]
    bos_up_m1   = _ffill_bool_to_index(bos_up_recent,   m1i.index)
    bos_down_m1 = _ffill_bool_to_index(bos_down_recent, m1i.index)

    adx_ok   = base["adx14"] > adx_min
    vol_ok   = base["vol_ratio"] > vol_mult
    atr_rel  = base["atr_rel"]
    is_sq    = base["squeeze_true"] if squeeze else base["squeeze_false"]

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False

    st_up, st_dn   = base["st_up"], base["st_dn"]
    ema_up, ema_dn = base["ema_up"], base["ema_dn"]
    macd_up, macd_dn = base["macd_up"], base["macd_dn"]
    brk_up, brk_dn   = base["brk_up"], base["brk_dn"]

    long_momo  = (valid & ema_up & macd_up & adx_ok & vol_ok & atr_rel & is_sq & (st_up if supertrend_req else True))
    short_momo = (valid & ema_dn & macd_dn & adx_ok & vol_ok & atr_rel & is_sq & (st_dn if supertrend_req else True))
    long_brk   = (valid & ema_up & adx_ok & vol_ok & atr_rel & is_sq & brk_up & (st_up if supertrend_req else True))
    short_brk  = (valid & ema_dn & adx_ok & vol_ok & atr_rel & is_sq & brk_dn & (st_dn if supertrend_req else True))

    if entry_mode=="breakout":
        long_core, short_core = long_brk, short_brk
    elif entry_mode=="momentum":
        long_core, short_core = long_momo, short_momo
    elif entry_mode=="pullback":
        near_long  = (m1i["close"] <= (m1i["ema21"] + 0.2*m1i["atr14"]))
        recross_up = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema9"].shift(1)<=m1i["ema21"].shift(1))
        long_pull  = (valid & ema_up & near_long & recross_up & adx_ok & vol_ok & atr_rel & (st_up if supertrend_req else True))
        near_short = (m1i["close"] >= (m1i["ema21"] - 0.2*m1i["atr14"]))
        recross_dn = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema9"].shift(1)>=m1i["ema21"].shift(1))
        short_pull = (valid & ema_dn & near_short & recross_dn & adx_ok & vol_ok & atr_rel & (st_dn if supertrend_req else True))
        long_core, short_core = long_pull, short_pull
    else:
        long_core, short_core = (long_brk | long_momo), (short_brk | short_momo)

    long_sig  = (long_core  & h1_up_m1   & bos_up_m1).astype(bool)
    short_sig = (short_core & h1_down_m1 & bos_down_m1).astype(bool)
    return long_sig, short_sig

def simulate_fast(m1i, long_sig, short_sig, start_utc, lookahead=150, tp_R=3.0, move_BE_at_1R=True,
                  sessions=None):
    idx = m1i.index
    sig = (long_sig | short_sig).copy()
    sig.loc[idx < start_utc] = False

    if sessions:
        # FIX: use NumPy arrays and align back to a Series; no .values
        idx_lagos = idx.tz_convert(LAGOS)
        times = np.array(idx_lagos.time)  # ndarray of datetime.time
        mask = np.zeros(len(idx), dtype=bool)
        for a,b in sessions:
            h1,m1 = map(int,a.split(":")); h2,m2 = map(int,b.split(":"))
            t1 = time(h1,m1); t2 = time(h2,m2)
            mask |= ((times>=t1) & (times<=t2))
        sig &= pd.Series(mask, index=idx)

    close = m1i["close"].values
    high  = m1i["high"].values
    low   = m1i["low"].values
    atr   = m1i["atr14"].values
    long_mask = long_sig.values
    sig_pos = np.where(sig.values)[0]

    wins=losses=timeouts=0; total_R=0.0
    for pos in sig_pos:
        a = atr[pos]
        if not np.isfinite(a) or a<=0: 
            continue
        entry = close[pos]
        if long_mask[pos]:
            sl = entry - 1.5*a
            r1 = entry + (entry - sl)
            tp = entry + tp_R*(entry - sl)
            h = high[pos+1:pos+1+lookahead]
            l = low [pos+1:pos+1+lookahead]
            has_tp = (h>=tp).any(); has_sl = (l<=sl).any(); has_1r = (h>=r1).any()
            hit_tp = np.argmax(h>=tp); hit_sl = np.argmax(l<=sl)
        else:
            sl = entry + 1.5*a
            r1 = entry - (sl - entry)
            tp = entry - tp_R*(sl - entry)
            h = high[pos+1:pos+1+lookahead]
            l = low [pos+1:pos+1+lookahead]
            has_tp = (l<=tp).any(); has_sl = (h>=sl).any(); has_1r = (l<=r1).any()
            hit_tp = np.argmax(l<=tp); hit_sl = np.argmax(h>=sl)

        res_R=0.0
        if move_BE_at_1R and has_1r:
            if has_tp:
                res_R = tp_R; wins += 1
            else:
                timeouts += 1
        else:
            if has_tp and has_sl:
                if hit_tp <= hit_sl:
                    res_R = tp_R; wins+=1
                else:
                    res_R = -1.0; losses+=1
            elif has_tp:
                res_R = tp_R; wins+=1
            elif has_sl:
                res_R = -1.0; losses+=1
            else:
                timeouts += 1
        total_R += res_R

    considered = wins+losses
    winrate = (wins/considered*100.0) if considered>0 else 0.0
    return {"trades":len(sig_pos), "wins":wins, "losses":losses, "be_or_to":timeouts,
            "winrate":winrate, "total_R":total_R}

def main():
    init_mt5()
    sym = SYMBOL
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=45)
    start_ext = start - timedelta(days=20)

    print(f"Symbol={sym}  Window={start.date()} → {now.date()}  (45d)")
    h1  = rates_range_df(sym, "H1",  start_ext, now)
    m15 = rates_range_df(sym, "M15", start_ext, now)
    m1  = rates_range_df(sym, "M1",  start_ext, now)
    if h1.empty or m15.empty or m1.empty:
        print("No history returned; open charts / increase market data cache.")
        return

    print("Adding indicators...", end=""); h1i=add_indicators(h1); m15i=add_indicators(m15); m1i=add_indicators(m1); print("done.")
    print("Precomputing caches...", end=""); base=precompute_common(m1i); bos_cache=build_bos_m15(m15i); print("done.")

    grid=[]
    tp_Rs     = [2.5, 3.0]
    adx_mins  = [18]
    vol_mults = [1.05, 1.10]
    bos_ws    = [2]
    modes     = ["either","pullback"]
    squeezes  = [False, True]
    for tp_R in tp_Rs:
        for adx_min in adx_mins:
            for vol_mult in vol_mults:
                for bos_w in bos_ws:
                    for entry_mode in modes:
                        for sq in squeezes:
                            grid.append((tp_R, adx_min, vol_mult, bos_w, entry_mode, sq))

    print(f"Running {len(grid)} configs…")
    sessions = [("07:00","11:30"), ("13:30","17:00")]
    results=[]
    for i, (tp_R, adx_min, vol_mult, bos_w, entry_mode, sq) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] tp_R={tp_R} adx_min={adx_min} vol_mult={vol_mult} bos_w={bos_w} mode={entry_mode} squeeze={sq}")
        long_sig, short_sig = build_signals_cached(
            h1i, m15i, m1i, base, bos_cache,
            warmup=300, adx_min=adx_min, bos_w=bos_w, vol_mult=vol_mult,
            entry_mode=entry_mode, supertrend_req=True, squeeze=sq
        )
        res = simulate_fast(m1i, long_sig, short_sig, start, lookahead=150, tp_R=tp_R,
                            move_BE_at_1R=True, sessions=sessions)
        res.update({"tp_R":tp_R,"adx_min":adx_min,"vol_mult":vol_mult,"bos_w":bos_w,
                    "entry_mode":entry_mode,"squeeze":sq})
        print(f"   → trades={res['trades']}  winrate={res['winrate']:.1f}%  total_R={res['total_R']:.1f}")
        results.append(res)

    df = pd.DataFrame(results).sort_values(["winrate","total_R"], ascending=[False,False])
    print("\n=== Top (quick) configs ===")
    cols = ["winrate","trades","wins","losses","be_or_to","total_R","tp_R","adx_min","vol_mult","bos_w","entry_mode","squeeze"]
    print(df[cols].head(10).to_string(index=False))
    os.makedirs("logs", exist_ok=True); df.to_csv("logs/winrate_top10_quick.csv", index=False)
    print("Saved: logs/winrate_top10_quick.csv")

if __name__ == "__main__":
    main()
