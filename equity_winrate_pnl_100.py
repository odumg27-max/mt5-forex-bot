import os, sys, argparse, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from bot_mt5 import init_mt5, add_indicators, SYMBOL

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def rates_range_df(symbol, tf_key, start, end):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, end)
    if rates is None or len(rates)==0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})
    return df

def _to_bool_mask(series_like, index):
    s = series_like.reindex(index)
    s = s.ffill()
    s = s.fillna(False)
    # Avoid pandas future downcasting warnings by going through numpy
    return pd.Series(np.asarray(s, dtype=bool), index=index)

def build_signals(h1i, m15i, m1i, *, warmup=300, adx_min=18, don_len=20, bos_window_m15=2,
                  vol_mult=1.10, entry_mode="either", supertrend_req=True):
    # H1 regime filter
    h1_up   = (h1i["ema50"]>h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_down = (h1i["ema50"]<h1i["ema200"]) & (h1i["adx14"]>adx_min)
    h1_up_m1   = _to_bool_mask(h1_up,   m1i.index)
    h1_down_m1 = _to_bool_mask(h1_down, m1i.index)

    # M15 structure/BOS
    prior_high = m15i["high"].rolling(20).max().shift(1)
    prior_low  = m15i["low"].rolling(20).min().shift(1)
    bos_up_recent   = (m15i["close"] > prior_high).rolling(bos_window_m15).max().astype(bool)
    bos_down_recent = (m15i["close"] < prior_low ).rolling(bos_window_m15).max().astype(bool)
    bos_up_m1   = _to_bool_mask(bos_up_recent,   m1i.index)
    bos_down_m1 = _to_bool_mask(bos_down_recent, m1i.index)

    # M1 momentum stacks
    ema_up   = (m1i["ema9"]>m1i["ema21"]) & (m1i["ema21"]>m1i["ema50"])
    ema_dn   = (m1i["ema9"]<m1i["ema21"]) & (m1i["ema21"]<m1i["ema50"])
    macd_up  = (m1i["macdh"]>0) & (m1i["macdh"]>m1i["macdh"].shift(1))
    macd_dn  = (m1i["macdh"]<0) & (m1i["macdh"]<m1i["macdh"].shift(1))
    st_up    = (m1i["st_dir"]==1)
    st_dn    = (m1i["st_dir"]==-1)
    adx_ok   = (m1i["adx14"]>adx_min)
    vol_ok   = (m1i["volume"] > vol_mult*m1i["vol_ma20"])

    # Donchian breakout (M1)
    dch = m1i["high"].rolling(don_len).max()
    dcl = m1i["low"].rolling(don_len).min()
    brk_up = (m1i["close"]>dch) & (m1i["close"].shift(1)<=dch.shift(1))
    brk_dn = (m1i["close"]<dcl) & (m1i["close"].shift(1)>=dcl.shift(1))

    valid = pd.Series(True, index=m1i.index); valid.iloc[:warmup] = False

    long_momo  = valid & ema_up & macd_up & adx_ok & vol_ok & (st_up if supertrend_req else True)
    short_momo = valid & ema_dn & macd_dn & adx_ok & vol_ok & (st_dn if supertrend_req else True)
    long_brk   = valid & ema_up   & adx_ok & vol_ok & brk_up   & (st_up if supertrend_req else True)
    short_brk  = valid & ema_dn   & adx_ok & vol_ok & brk_dn   & (st_dn if supertrend_req else True)

    if entry_mode=="breakout":
        long_core, short_core = long_brk, short_brk
    elif entry_mode=="momentum":
        long_core, short_core = long_momo, short_momo
    else:  # "either"
        long_core, short_core = (long_brk | long_momo), (short_brk | short_momo)

    long_sig  = (long_core  & h1_up_m1   & bos_up_m1)
    short_sig = (short_core & h1_down_m1 & bos_down_m1)
    return long_sig.astype(bool), short_sig.astype(bool)

def simulate_usd(m1i, long_sig, short_sig, start_utc, *, lookahead=150, tp_R=3.0,
                 start_equity=100.0, risk_pct=0.02, move_BE_at_1R=False):
    sig = (long_sig | short_sig).copy()
    sig.loc[sig.index < start_utc] = False
    idx = m1i.index[sig]

    equity = float(start_equity)
    wins = losses = be_to = 0
    total_R = 0.0
    for ts in idx:
        row = m1i.loc[ts]
        atr = row.get("atr14", np.nan)
        if not np.isfinite(atr) or atr<=0: 
            continue
        long = bool(long_sig.loc[ts])
        entry = float(row["close"])
        fw = m1i.loc[ts:].iloc[1:lookahead]

        if long:
            sl = entry - 1.5*atr
            r1 = entry + (entry - sl)
            tp = entry + tp_R*(entry - sl)
            hit_tp = (fw["high"]>=tp)
            hit_sl = (fw["low"]<=sl)
            hit_1R = (fw["high"]>=r1)
        else:
            sl = entry + 1.5*atr
            r1 = entry - (sl - entry)
            tp = entry - tp_R*(sl - entry)
            hit_tp = (fw["low"]<=tp)
            hit_sl = (fw["high"]>=sl)
            hit_1R = (fw["low"]<=r1)

        res_R = 0.0
        if move_BE_at_1R and hit_1R.any():
            # Move to BE after 1R; if no TP later, count as BE (0R)
            if hit_tp.any():
                res_R = tp_R
                wins += 1
            else:
                be_to += 1
        else:
            if hit_tp.any() and hit_sl.any():
                t_tp = fw.index[hit_tp.argmax()]
                t_sl = fw.index[hit_sl.argmax()]
                if t_tp <= t_sl:
                    res_R = tp_R; wins += 1
                else:
                    res_R = -1.0; losses += 1
            elif hit_tp.any():
                res_R = tp_R; wins += 1
            elif hit_sl.any():
                res_R = -1.0; losses += 1
            else:
                be_to += 1

        # Compounding PnL in USD
        risk_usd = equity * risk_pct
        pnl_usd  = res_R * risk_usd
        equity  += pnl_usd
        total_R += res_R

    considered = wins + losses
    winrate = (wins/considered*100.0) if considered>0 else 0.0
    return {
        "trades": int(sig.sum()),
        "wins": wins, "losses": losses, "be_or_to": be_to,
        "winrate": winrate, "total_R": total_R,
        "final_equity": equity
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45, help="lookback window (days)")
    ap.add_argument("--yesterday", action="store_true", help="override window to yesterday->now")
    ap.add_argument("--risk", type=float, default=float(os.getenv("RISK_PCT","0.02")),
                    help="risk per trade as fraction (default .env RISK_PCT or 0.02)")
    ap.add_argument("--tpR", type=float, default=3.0, help="take profit in R (default 3.0)")
    args = ap.parse_args()

    init_mt5()
    sym = SYMBOL
    now = datetime.now(timezone.utc)
    if args.yesterday:
        start = now - timedelta(days=1)
    else:
        start = now - timedelta(days=args.days)
    start_ext = start - timedelta(days=7)  # warmup fetch

    print(f"Equity report for {sym} | window: {start.date()} → {now.date()}")
    h1  = rates_range_df(sym, "H1",  start_ext, now)
    m15 = rates_range_df(sym, "M15", start_ext, now)
    m1  = rates_range_df(sym, "M1",  start_ext, now)
    if h1.empty or m15.empty or m1.empty:
        print("No history returned; open MT5 charts and ensure the symbol has data.")
        return

    # Indicators
    h1i  = add_indicators(h1)
    m15i = add_indicators(m15)
    m1i  = add_indicators(m1)

    # Signals (defaults mirror your strategy: top-down + momo/breakout, ADX & volume, Donchian)
    long_sig, short_sig = build_signals(
        h1i, m15i, m1i,
        warmup=300, adx_min=18, don_len=20, bos_window_m15=2,
        vol_mult=1.10, entry_mode="either", supertrend_req=True
    )

    res = simulate_usd(
        m1i, long_sig, short_sig, start,
        lookahead=150, tp_R=args.tpR, start_equity=100.0, risk_pct=args.risk,
        move_BE_at_1R=False
    )

    print("\n=== SUMMARY (start $100, compounding) ===")
    print(f"Trades: {res['trades']}  Wins: {res['wins']}  Losses: {res['losses']}  BE/Timeout: {res['be_or_to']}")
    print(f"Win rate: {res['winrate']:.2f}%")
    print(f"Total PnL: {res['total_R']:.1f} R  |  Final equity: ${res['final_equity']:.2f}  (risk {args.risk*100:.1f}% per trade)")
    # Optional CSV
    os.makedirs("logs", exist_ok=True)
    with open("logs/equity_winrate_100.txt","w") as f:
        f.write(f"{res}\n")
    print("Saved: logs/equity_winrate_100.txt")
if __name__ == "__main__":
    main()
