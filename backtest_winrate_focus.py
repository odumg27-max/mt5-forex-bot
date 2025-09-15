import numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from bot_mt5 import init_mt5, add_indicators, SYMBOL

# Opt in to pandas future behavior to avoid downcasting warnings
pd.set_option("future.no_silent_downcasting", True)

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

try:
    from zoneinfo import ZoneInfo
    LAGOS = ZoneInfo("Africa/Lagos")
except Exception:
    from datetime import timedelta, timezone
    LAGOS = timezone(timedelta(hours=1))

def rates_range_df(symbol, tf_key, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days+7)
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, now)
    if rates is None or len(rates)==0: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})

# Safe boolean alignment (no deprecation warnings)
def align_bool(src: pd.Series, target_idx) -> pd.Series:
    s = src.astype("boolean")                   # pandas BooleanDtype
    s = s.reindex(target_idx)                   # align index
    s = s.ffill()                               # forward fill
    s = s.fillna(False).astype(bool)            # final plain bool
    return s

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    s = pd.Series(series, dtype="float64")
    delta = s.diff()
    up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    ru = up.ewm(alpha=1/max(n,1), adjust=False).mean()
    rd = dn.ewm(alpha=1/max(n,1), adjust=False).mean()
    rs = ru / (rd.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def run(days=60, warmup=350, lookahead=150, max_trades=150,
        adx_min_h1=22, adx_min_m1=20, vol_mult=1.10,
        don_len=15, bos_window_m15=3,
        rsi_len=14, rsi_up=55, rsi_dn=45,
        ema_align_bars=3, pullback_k_atr=0.35,
        session_start=8, session_end=18,
        entry_mode="pullback", supertrend_req=True,
        progress_every=25):

    init_mt5()
    sym = SYMBOL
    print(f"Winrate-focused backtest on {sym} | days={days} warmup={warmup} lookahead={lookahead} max_trades={max_trades}")
    print(f"Filters: H1_ADX>{adx_min_h1}, M1_ADX>{adx_min_m1}, Vol>{vol_mult}×ma, RSI({rsi_len})>={rsi_up}/<={rsi_dn}, EMA align {ema_align_bars} bars, session {session_start}-{session_end} Lagos, entry={entry_mode}")

    h1  = rates_range_df(sym, "H1",  days)
    m15 = rates_range_df(sym, "M15", days)
    m1  = rates_range_df(sym, "M1",  days)
    print(f"Bars -> H1:{len(h1)}  M15:{len(m15)}  M1:{len(m1)}")
    if min(len(h1), len(m15), len(m1)) < warmup:
        print("Not enough bars; increase --days or open charts to cache more history."); return

    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    m1i["rsi"] = rsi(m1i["close"], rsi_len)

    # H1 regime, aligned to M1 (warning-safe)
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min_h1)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min_h1)
    h1_up_m1   = align_bool(h1_up,   m1i.index).to_numpy()
    h1_down_m1 = align_bool(h1_down, m1i.index).to_numpy()

    # M15 BOS recent, aligned to M1 (warning-safe)
    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(bos_window_m15).max().astype(bool)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(bos_window_m15).max().astype(bool)
    bos_up_m1   = align_bool(bos_up_recent,   m1i.index).to_numpy()
    bos_down_m1 = align_bool(bos_down_recent, m1i.index).to_numpy()

    # M1 momentum / structure
    ema_up   = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    ema_dn   = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])

    ema_up_sus = ema_up.rolling(ema_align_bars).apply(lambda x: float(x.all()), raw=False).astype(bool)
    ema_dn_sus = ema_dn.rolling(ema_align_bars).apply(lambda x: float(x.all()), raw=False).astype(bool)

    macd_up  = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    macd_dn  = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    st_up    = (m1i["st_dir"]==1)
    st_dn    = (m1i["st_dir"]==-1)
    adx_ok   = (m1i["adx14"]>adx_min_m1)
    vol_ok   = (m1i["volume"] > vol_mult*m1i["vol_ma20"])
    rsi_up_ok = (m1i["rsi"] >= rsi_up)
    rsi_dn_ok = (m1i["rsi"] <= rsi_dn)

    dch = m1i["high"].rolling(don_len).max()
    dcl = m1i["low"].rolling(don_len).min()
    brk_up   = (m1i["close"]>dch) & (m1i["close"].shift(1)<=dch.shift(1))
    brk_dn   = (m1i["close"]<dcl) & (m1i["close"].shift(1)>=dcl.shift(1))

    # Pullback (fixed precedence + dtype-safe)
    pb_up  = (
        ema_up_sus
        & adx_ok
        & rsi_up_ok
        & (((m1i["close"] - m1i["ema21"]).abs()) <= (pullback_k_atr * m1i["atr14"]))
        & (m1i["volume"] > (vol_mult * m1i["vol_ma20"]))
        & (st_up if supertrend_req else True)
    )
    pb_dn  = (
        ema_dn_sus
        & adx_ok
        & rsi_dn_ok
        & (((m1i["close"] - m1i["ema21"]).abs()) <= (pullback_k_atr * m1i["atr14"]))
        & (m1i["volume"] > (vol_mult * m1i["vol_ma20"]))
        & (st_dn if supertrend_req else True)
    )

    long_momo  = ema_up_sus  & macd_up  & adx_ok & vol_ok & rsi_up_ok  & (st_up if supertrend_req else True)
    short_momo = ema_dn_sus  & macd_dn  & adx_ok & vol_ok & rsi_dn_ok  & (st_dn if supertrend_req else True)
    long_brk   = ema_up_sus  & adx_ok   & vol_ok & rsi_up_ok  & brk_up  & (st_up if supertrend_req else True)
    short_brk  = ema_dn_sus  & adx_ok   & vol_ok & rsi_dn_ok  & brk_dn  & (st_dn if supertrend_req else True)
    long_pull  = pb_up
    short_pull = pb_dn

    if entry_mode == "breakout":
        long_core, short_core = long_brk, short_brk
    elif entry_mode == "pullback":
        long_core, short_core = long_pull, short_pull
    else:
        long_core, short_core = (long_brk | long_pull | long_momo), (short_brk | short_pull | short_momo)

    idx_lagos = m1i.index.tz_convert(LAGOS)
    in_session = (idx_lagos.hour >= session_start) & (idx_lagos.hour < session_end)

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False
    long_sig  = valid & in_session & pd.Series(bos_up_m1, index=m1i.index)   & pd.Series(h1_up_m1, index=m1i.index)   & long_core
    short_sig = valid & in_session & pd.Series(bos_down_m1, index=m1i.index) & pd.Series(h1_down_m1, index=m1i.index) & short_core

    sig_mask = (long_sig | short_sig)
    sig_idx = m1i.index[sig_mask]
    if len(sig_idx)==0:
        print("No signals with current conservative filters. Loosen params slightly if needed.")
        return

    pnl_R = 0.0; wins=losses=0; trades=0
    for ts in sig_idx:
        if trades >= max_trades: break
        row = m1i.loc[ts]
        atr = row.get("atr14", np.nan)
        if not np.isfinite(atr) or atr <= 0: continue

        side = "buy" if long_sig.loc[ts] else "sell"
        entry = float(row["close"])
        if side=="buy":
            sl = entry - 1.5*atr
            tp = entry + 3.0*(entry - sl)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            tp_hit = (fw["high"] >= tp); sl_hit = (fw["low"] <= sl)
        else:
            sl = entry + 1.5*atr
            tp = entry - 3.0*(sl - entry)
            fw = m1i.loc[ts:].iloc[1:lookahead]
            tp_hit = (fw["low"] <= tp);  sl_hit = (fw["high"] >= sl)

        res = 0.0
        if tp_hit.any() and sl_hit.any():
            t_tp = fw.index[tp_hit.argmax()]; t_sl = fw.index[sl_hit.argmax()]
            res = 3.0 if t_tp <= t_sl else -1.0
        elif tp_hit.any(): res = 3.0
        elif sl_hit.any(): res = -1.0

        pnl_R += res; trades += 1
        if res>0: wins+=1
        elif res<0: losses+=1
        if trades % progress_every == 0:
            print(f"Processed {trades}/{max_trades} signals... PnL(R)={pnl_R:.1f}, Win%={(wins/max(trades,1))*100:.1f}")

    print(f"\nRESULTS  Trades:{trades}  Wins:{wins}  Losses:{losses}  Win%:{(wins/max(trades,1))*100:.1f}%  PnL(R):{pnl_R:.1f}  Avg R/trade:{(pnl_R/max(trades,1)):.2f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=350)
    ap.add_argument("--lookahead", type=int, default=150)
    ap.add_argument("--max_trades", type=int, default=150)
    ap.add_argument("--adx_min_h1", type=int, default=22)
    ap.add_argument("--adx_min_m1", type=int, default=20)
    ap.add_argument("--vol_mult", type=float, default=1.10)
    ap.add_argument("--don_len", type=int, default=15)
    ap.add_argument("--bos_window_m15", type=int, default=3)
    ap.add_argument("--rsi_len", type=int, default=14)
    ap.add_argument("--rsi_up", type=float, default=55)
    ap.add_argument("--rsi_dn", type=float, default=45)
    ap.add_argument("--ema_align_bars", type=int, default=3)
    ap.add_argument("--pullback_k_atr", type=float, default=0.35)
    ap.add_argument("--session_start", type=int, default=8)
    ap.add_argument("--session_end", type=int, default=18)
    ap.add_argument("--entry_mode", choices=["pullback","breakout","either"], default="pullback")
    ap.add_argument("--supertrend_req", action="store_true")
    ap.add_argument("--no_supertrend", dest="supertrend_req", action="store_false")
    ap.set_defaults(supertrend_req=True)
    args = ap.parse_args()
    run(**vars(args))
