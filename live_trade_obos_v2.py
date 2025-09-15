import os, time as pyt, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone, time
from bot_mt5 import init_mt5, add_indicators, SYMBOL
from backtest_obos_window_v2 import build_signals_v2, session_mask_index

TF = {"M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

def rates_range_df(symbol, tf_key, start, end):
    rates = mt5.copy_rates_range(symbol, TF[tf_key], start, end)
    if rates is None or len(rates)==0: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df.rename(columns={"tick_volume":"volume"})
    return df[["open","high","low","close","volume"]]

def _pip_and_point(symbol):
    info = mt5.symbol_info(symbol)
    if not info: raise RuntimeError("symbol_info failed")
    point = info.point
    # 5/3 digits -> 1 pip = 10 points; else 1 pip = 1 point
    pip = point*10 if info.digits in (3,5) else point
    return info, point, pip

def _spread_pips(symbol, pip):
    tk = mt5.symbol_info_tick(symbol)
    if not tk: return np.inf
    return (tk.ask - tk.bid) / pip

def _pip_value_per_lot(symbol, pip):
    info = mt5.symbol_info(symbol)
    tv = info.trade_tick_value or 0.0
    ts = info.trade_tick_size or info.point
    if ts == 0: ts = info.point
    return tv * (pip/ts)

def _quantize_lot(symbol, lots):
    info = mt5.symbol_info(symbol)
    step = info.volume_step or 0.01
    lots = max(info.volume_min, min(info.volume_max, lots))
    return round(np.floor(lots/step)*step, 2)

def _calc_rr_prices(side, price, atr, atr_mult=1.5, rr=3.0):
    if side=="buy":
        sl = price - atr_mult*atr
        tp = price + rr*(price - sl)
    else:
        sl = price + atr_mult*atr
        tp = price - rr*(sl - price)
    return sl, tp

def _allowed_now(idx_utc, cooldown_min=20, per_hour_cap=3, state={}):
    now = idx_utc[-1]
    # cooldown
    last_ts = state.get("last_trade_ts")
    if last_ts and (now - last_ts) < timedelta(minutes=cooldown_min):
        return False, "cooldown"
    # per-hour cap
    hour_start = state.get("hour_start")
    hour_ct = state.get("hour_ct", 0)
    if (hour_start is None) or (now - hour_start) >= timedelta(hours=1):
        state["hour_start"] = now
        state["hour_ct"] = 0
        hour_ct = 0
    if hour_ct >= per_hour_cap:
        return False, "per_hour_cap"
    return True, ""

def _barchange_pips(bar, pip):
    return (bar["high"] - bar["low"]) / pip

def run_live():
    init_mt5()
    sym = SYMBOL
    info, point, pip = _pip_and_point(sym)

    # --- Realism knobs (env-overridable)
    risk_pct            = float(os.getenv("RISK_PCT",            "0.02"))   # 2%/trade
    max_spread_pips     = float(os.getenv("MAX_SPREAD_PIPS",     "1.2"))    # reject > 1.2 pips
    slippage_pips       = float(os.getenv("SLIPPAGE_PIPS",       "0.2"))    # budgeted slippage
    commission_per_lot  = float(os.getenv("COMMISSION_PER_LOT",  "7.0"))    # USD / 1.0 lot (round-turn)
    deviation_points    = int(os.getenv("DEVIATION_POINTS",      "20"))     # allowed price deviation (points)
    news_spike_pips     = float(os.getenv("NEWS_SPIKE_PIPS",     "5.0"))    # reject if last M1 range > this
    warm_days           = int(os.getenv("WARM_DAYS",             "30"))     # indicator warmback
    atr_mult            = float(os.getenv("ATR_MULT",            "1.5"))    # 1.5*ATR stop
    rr                  = float(os.getenv("RR",                  "3.0"))    # 1:3 RR

    os.makedirs("logs", exist_ok=True)
    fills_path = "logs/live_fills.csv"
    if not os.path.exists(fills_path):
        pd.DataFrame(columns=[
            "time","symbol","side","entry","sl","tp","lot","spread_pips",
            "risk_pct","risk_usd","pnl_target_usd","comment","ticket","retcode","reason"
        ]).to_csv(fills_path, index=False)

    last_closed_ts = None
    state = {}  # cooldown/hour counters

    print(f"Live OB/OS v2 runner on {sym}  (candle-close only, spread cap, slippage/commission, SL/TP immediate)")
    while True:
        # wait for a new CLOSED M1 bar
        rates = mt5.copy_rates_from_pos(sym, TF["M1"], 0, 3)
        if rates is None or len(rates) < 2:
            pyt.sleep(1); continue

        r2 = pd.DataFrame(rates)
        r2["time"] = pd.to_datetime(r2["time"], unit="s", utc=True)
        r2.set_index("time", inplace=True)

        closed_ts = r2.index[-2]  # last CLOSED bar
        if last_closed_ts is not None and closed_ts <= last_closed_ts:
            pyt.sleep(1); continue

        # -------- Build indicators up to the new bar (warmback -> now)
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=warm_days)
        h1  = rates_range_df(sym, "H1",  start, end)
        m15 = rates_range_df(sym, "M15", start, end)
        m1  = rates_range_df(sym, "M1",  start, end)
        if h1.empty or m15.empty or m1.empty:
            pyt.sleep(1); continue

        h1i  = add_indicators(h1)
        m15i = add_indicators(m15)
        m1i  = add_indicators(m1)

        # compute signals with your exact logic
        long_sig, short_sig = build_signals_v2(
            h1i, m15i, m1i,
            warmup=300, adx_min=20, vol_mult=1.10, don_len=20, supertrend_req=True
        )

        # candle-close only: evaluate exactly on the just-closed bar
        ts = closed_ts
        in_sess = session_mask_index(m1i.index).reindex(m1i.index).ffill().reindex([ts]).iloc[0]
        if not in_sess:
            last_closed_ts = ts;  pyt.sleep(0.2);  continue

        allowed, reason = _allowed_now(m1i.index)
        if not allowed:
            last_closed_ts = ts;  pyt.sleep(0.2);  continue

        # news spike guard: reject if the just-closed bar range is too large
        last_bar = m1i.loc[ts][["high","low"]]
        if _barchange_pips(last_bar, pip) > news_spike_pips:
            last_closed_ts = ts;  pyt.sleep(0.2);  continue

        # if no signal on this bar -> move on
        is_long  = bool(long_sig.reindex([ts]).iloc[0]) if ts in long_sig.index else False
        is_short = bool(short_sig.reindex([ts]).iloc[0]) if ts in short_sig.index else False
        if not (is_long or is_short):
            last_closed_ts = ts;  pyt.sleep(0.2);  continue

        # spread cap
        sp = _spread_pips(sym, pip)
        if sp > max_spread_pips or not np.isfinite(sp):
            last_closed_ts = ts;  pyt.sleep(0.2);  continue

        # entry price & ATR
        atr = float(m1i.loc[ts, "atr14"]) if "atr14" in m1i.columns else float(pd.Series(m1i["close"]).rolling(14).apply(lambda x: np.std(x)).reindex([ts]).iloc[0])
        tk = mt5.symbol_info_tick(sym)
        if not tk or not np.isfinite(atr):
            last_closed_ts = ts; pyt.sleep(0.2); continue

        side = "buy" if is_long else "sell"
        price = tk.ask if side=="buy" else tk.bid
        sl, tp = _calc_rr_prices(side, price, atr, atr_mult=atr_mult, rr=rr)

        # risk sizing incl. commission & slippage & spread
        equity = (mt5.account_info().equity or 0.0)
        risk_budget = equity * risk_pct
        sl_dist_pips = abs(price - sl) / pip
        pip_val_lot = _pip_value_per_lot(sym, pip)
        # budget entry-costs (full spread + slippage)
        extra_pips = sp + slippage_pips
        risk_cost_per_lot = (sl_dist_pips + extra_pips) * pip_val_lot + commission_per_lot
        lots = _quantize_lot(sym, max(0.0, risk_budget / max(1e-9, risk_cost_per_lot)))
        if lots <= 0:
            last_closed_ts = ts;  pyt.sleep(0.2);  continue

        # send market order with immediate SL/TP
        order_type = mt5.ORDER_TYPE_BUY if side=="buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "type": order_type,
            "volume": float(lots),
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation_points,
            "magic": 9012025,
            "comment": f"OBOSv2 rr={rr} risk={int(risk_pct*100)}% live",
            "type_filling": mt5.ORDER_FILLING_FOK if (mt5.symbol_info(sym).filling_mode in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC)) else mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = mt5.order_send(request)

        ticket = getattr(result, "order", 0)
        ret    = getattr(result, "retcode", -1)
        reason = "" if ret==mt5.TRADE_RETCODE_DONE else str(result)
        pnl_target = rr * risk_budget

        # log the attempt/fill
        row = {
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": sym, "side": side,
            "entry": round(price, 6), "sl": round(sl, 6), "tp": round(tp, 6),
            "lot": float(lots), "spread_pips": round(sp, 2),
            "risk_pct": risk_pct, "risk_usd": round(risk_budget, 2),
            "pnl_target_usd": round(pnl_target, 2),
            "comment": request["comment"], "ticket": ticket, "retcode": ret, "reason": reason
        }
        try:
            pd.DataFrame([row]).to_csv("logs/live_fills.csv", mode="a", header=False, index=False)
        except Exception:
            pass

        # advance cooldown counters only on a *send*
        state["last_trade_ts"] = ts
        state["hour_ct"] = state.get("hour_ct", 0) + 1

        print(f"{ts}  {side.upper()} lots={lots} price={price:.6f} sl={sl:.6f} tp={tp:.6f} spread={sp:.2f}p ret={ret}")

        last_closed_ts = ts
        pyt.sleep(0.5)

if __name__ == "__main__":
    run_live()
