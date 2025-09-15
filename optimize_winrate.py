import os, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone, time
from bot_mt5 import init_mt5, add_indicators, SYMBOL

# Opt into the future behavior to silence the downcasting warning
pd.set_option("future.no_silent_downcasting", True)

try:
    from zoneinfo import ZoneInfo
    LAGOS = ZoneInfo("Africa/Lagos")
except Exception:
    LAGOS = timezone(timedelta(hours=1))

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def _ffill_bool_to_index(s: pd.Series, index) -> pd.Series:
    """
    Reindex/ffill a boolean-like Series without triggering pandas FutureWarning.
    Works by using pandas nullable boolean dtype during NA handling, then casting to bool.
    """
    return (
        s.astype("boolean")        # allow <NA>
         .reindex(index)           # may introduce <NA>
         .ffill()                  # forward-fill <NA>
         .fillna(False)            # still nullable-bool
         .astype(bool)             # plain numpy bool mask
    )

def rates_range_df(symbol, tf_key, start_utc, end_utc):
    r = mt5.copy_rates_range(symbol, TF[tf_key], start_utc, end_utc)
    if r is None or len(r)==0: return pd.DataFrame()
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def ema_slope(series, n=20):
    ema = series.ewm(span=n, adjust=False).mean()
    return ema - ema.shift(n)

def build_signals(h1i, m15i, m1i, *,
                  warmup=300, adx_min=14, don_len=20, bos_window_m15=2,
                  vol_mult=1.10, entry_mode="either", supertrend_req=True,
                  use_pullback=False, squeeze=False):

    # H1 regime + slope + MACD sign
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    slope200 = ema_slope(h1i["ema200"], 20)
    macd_pos = (h1i.get("macdh", pd.Series(index=h1i.index, data=0)) > 0)
    macd_neg = ~macd_pos
    h1_up   = (h1_up  & (slope200>0) & macd_pos)
    h1_down = (h1_down & (slope200<0) & macd_neg)

    # Reindex safely to M1 (no warnings)
    h1_up_m1   = _ffill_bool_to_index(h1_up,   m1i.index)
    h1_down_m1 = _ffill_bool_to_index(h1_down, m1i.index)

    # M15 BOS (stay boolean throughout)
    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(bos_window_m15).sum().gt(0)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(bos_window_m15).sum().gt(0)

    bos_up_m1   = _ffill_bool_to_index(bos_up_recent,   m1i.index)
    bos_down_m1 = _ffill_bool_to_index(bos_down_recent, m1i.index)

    # Volatility & volume filters
    adx_ok   = (m1i["adx14"]>adx_min)
    vol_ok   = (m1i["volume"] > vol_mult*m1i["vol_ma20"])
    atr_med  = m1i["atr14"].rolling(1000).median()
    atr_rel  = (m1i["atr14"] > atr_med)

    # Optional squeeze filter (computed safely; False when insufficient history)
    if squeeze:
        ma = m1i["close"].rolling(20).mean()
        std = m1i["close"].rolling(20).std()
        bbw = (4*std)/ma.clip(lower=1e-9)

        def _pctl(x):
            x = pd.Series(x).dropna()
            if len(x)==0: return np.nan
            return (x.rank(pct=True).iloc[-1])

        pctl = bbw.rolling(150).apply(_pctl, raw=False)
        is_squeeze = pctl.lt(0.35).rolling(15).max().fillna(False).astype(bool)
    else:
        is_squeeze = pd.Series(True, index=m1i.index, dtype=bool)

    # M1 momentum stacks
    ema_up   = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    ema_dn   = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])
    macd_up  = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    macd_dn  = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    st_up    = (m1i["st_dir"]==1)
    st_dn    = (m1i["st_dir"]==-1)

    # Donchian breakout
    dch = m1i["high"].rolling(don_len).max()
    dcl = m1i["low"].rolling(don_len).min()
    brk_up = (m1i["close"]>dch) & (m1i["close"].shift(1)<=dch.shift(1))
    brk_dn = (m1i["close"]<dcl) & (m1i["close"].shift(1)>=dcl.shift(1))

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False

    long_momo  = (valid & ema_up & macd_up & adx_ok & vol_ok & atr_rel & is_squeeze & (st_up if supertrend_req else True))
    short_momo = (valid & ema_dn & macd_dn & adx_ok & vol_ok & atr_rel & is_squeeze & (st_dn if supertrend_req else True))
    long_brk   = (valid & ema_up & adx_ok & vol_ok & atr_rel & is_squeeze & brk_up & (st_up if supertrend_req else True))
    short_brk  = (valid & ema_dn & adx_ok & vol_ok & atr_rel & is_squeeze & brk_dn & (st_dn if supertrend_req else True))

    if use_pullback:
        near_long  = (m1i["close"] <= (m1i["ema21"] + 0.2*m1i["atr14"]))
        recross_up = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema9"].shift(1)<=m1i["ema21"].shift(1))
        long_pull  = (valid & ema_up & near_long & recross_up & adx_ok & vol_ok & atr_rel & (st_up if supertrend_req else True))

        near_short = (m1i["close"] >= (m1i["ema21"] - 0.2*m1i["atr14"]))
        recross_dn = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema9"].shift(1)>=m1i["ema21"].shift(1))
        short_pull = (valid & ema_dn & near_short & recross_dn & adx_ok & vol_ok & atr_rel & (st_dn if supertrend_req else True))
    else:
        long_pull = short_pull = pd.Series(False, index=m1i.index, dtype=bool)

    if entry_mode=="breakout":
        long_core, short_core = long_brk, short_brk
    elif entry_mode=="momentum":
        long_core, short_core = long_momo, short_momo
    elif entry_mode=="pullback":
        long_core, short_core = long_pull, short_pull
    else:
        long_core, short_core = (long_brk | long_momo | long_pull), (short_brk | short_momo | short_pull)

    long_sig  = (long_core  & h1_up_m1   & bos_up_m1).astype(bool)
    short_sig = (short_core & h1_down_m1 & bos_down_m1).astype(bool)
    return long_sig, short_sig

def session_mask(index_utc, windows_lagos):
    idx_lagos = index_utc.tz_convert(LAGOS)
    times = idx_lagos.time
    mask = pd.Series(False, index=index_utc, dtype=bool)
    for a,b in windows_lagos:
        h1,m1 = map(int, a.split(":")); h2,m2 = map(int, b.split(":"))
        t1 = time(h1,m1); t2 = time(h2,m2)
        mask = mask | ((times>=t1) & (times<=t2))
    return mask

def simulate(m1i, long_sig, short_sig, start_utc, lookahead=150, tp_R=3.0, move_BE_at_1R=True,
             sessions=None, risk_pct=0.02):
    sig = (long_sig | short_sig).copy()
    sig.loc[sig.index < start_utc] = False
    if sessions:
        sig = sig & session_mask(sig.index, sessions)

    rows=[]; wins=losses=timeouts=0; total_R=0.0
    for ts in m1i.index[sig]:
        row = m1i.loc[ts]; atr = row.get("atr14", np.nan)
        if not np.isfinite(atr) or atr<=0: continue
        long = bool(long_sig.loc[ts]); entry=float(row["close"])
        if long:
            sl = entry - 1.5*atr
            tp = entry + tp_R*(entry - sl)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            r1 = entry + (entry - sl)
            hit_tp = (fw["high"]>=tp); hit_sl = (fw["low"]<=sl); hit_1R = (fw["high"]>=r1)
        else:
            sl = entry + 1.5*atr
            tp = entry - tp_R*(sl - entry)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            r1 = entry - (sl - entry)
            hit_tp = (fw["low"]<=tp);  hit_sl = (fw["high"]>=sl); hit_1R = (fw["low"]<=r1)

        res_R=0.0; outcome="timeout"
        if move_BE_at_1R and hit_1R.any():
            t_1R = fw.index[hit_1R.argmax()]
            if hit_tp.any():
                t_tp = fw.index[hit_tp.argmax()]
                res_R, outcome = (tp_R, "TP") if t_tp>=t_1R else (tp_R, "TP")
            else:
                res_R, outcome = 0.0, "BE"
        else:
            if hit_tp.any() and hit_sl.any():
                t_tp = fw.index[hit_tp.argmax()]; t_sl = fw.index[hit_sl.argmax()]
                res_R, outcome = (tp_R,"TP") if t_tp<=t_sl else (-1.0,"SL")
            elif hit_tp.any():
                res_R, outcome = tp_R, "TP"
            elif hit_sl.any():
                res_R, outcome = -1.0, "SL"

        rows.append({"time": ts, "side":"buy" if long else "sell", "R":res_R, "outcome":outcome})
        if   res_R>0: wins+=1
        elif res_R<0: losses+=1
        else: timeouts+=1
        total_R += res_R

    considered = wins+losses
    winrate = (wins/considered*100.0) if considered>0 else 0.0
    return {"trades":len(rows), "wins":wins, "losses":losses, "be_or_to":timeouts,
            "winrate":winrate, "total_R":total_R, "rows":rows}

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

    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    params=[]
    for tp_R in [2.0, 2.5, 3.0]:
        for adx_min in [14, 18, 22]:
            for vol_mult in [1.00, 1.10, 1.20]:
                for bos_w in [1, 2, 3]:
                    for entry_mode in ["either","pullback","breakout"]:
                        for squeeze in [False, True]:
                            params.append((tp_R, adx_min, vol_mult, bos_w, entry_mode, squeeze))

    results=[]
    sessions = [("07:00","11:30"), ("13:30","17:00")]
    for (tp_R, adx_min, vol_mult, bos_w, entry_mode, squeeze) in params:
        long_sig, short_sig = build_signals(
            h1i, m15i, m1i,
            warmup=300, adx_min=adx_min, don_len=20, bos_window_m15=bos_w,
            vol_mult=vol_mult, entry_mode=entry_mode, supertrend_req=True,
            use_pullback=(entry_mode=="pullback"), squeeze=squeeze
        )
        res = simulate(m1i, long_sig, short_sig, start, lookahead=150, tp_R=tp_R,
                       move_BE_at_1R=True, sessions=sessions)
        if res["trades"]<50:
            continue
        if res["total_R"]<=0:
            continue
        res["tp_R"]=tp_R; res["adx_min"]=adx_min; res["vol_mult"]=vol_mult
        res["bos_w"]=bos_w; res["entry_mode"]=entry_mode; res["squeeze"]=squeeze
        results.append(res)

    if not results:
        print("No positive-edge configs found in this quick sweep.")
        return

    df = pd.DataFrame(results).sort_values(["winrate","total_R"], ascending=[False,False]).head(10)
    print("\n=== Top configs by Win-Rate (edge>0) ===")
    cols = ["winrate","trades","wins","losses","be_or_to","total_R",
            "tp_R","adx_min","vol_mult","bos_w","entry_mode","squeeze"]
    pd.set_option("display.max_rows", None)
    print(df[cols].to_string(index=False))

    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/winrate_top10.csv", index=False)
    print("Saved: logs/winrate_top10.csv")

if __name__ == "__main__":
    main()
