import os, math
import MetaTrader5 as mt5

def _select(symbol:str)->bool:
    info = mt5.symbol_info(symbol)
    if not info:
        print(f"[ERR] symbol_info failed for {symbol}")
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True

def _round_volume(symbol:str, vol:float)->float:
    info = mt5.symbol_info(symbol)
    step = info.volume_step or 0.01
    vol_min = info.volume_min or 0.01
    vol_max = info.volume_max or 100.0
    # floor to avoid risking more than intended
    steps = math.floor(vol / step)
    vol_rounded = steps * step
    if vol_rounded < vol_min: vol_rounded = vol_min
    if vol_rounded > vol_max: vol_rounded = vol_max
    # round to 2/3 dp for clean printing
    return float(f"{vol_rounded:.3f}")

def compute_volume_for_risk(symbol:str, entry:float, sl:float, *, equity:float=None, risk_pct:float=None)->float:
    """Return volume (lots) so that SL loss ≈ risk_pct * equity.
       Uses SYMBOL_TRADE_TICK_SIZE / SYMBOL_TRADE_TICK_VALUE (per 1 lot)."""
    if not _select(symbol):
        return 0.0
    ai = mt5.account_info()
    if not ai:
        print("[ERR] account_info failed")
        return 0.0
    eq = float(equity if equity is not None else ai.equity or ai.balance)
    rp = float(risk_pct if risk_pct is not None else float(os.getenv("RISK_PCT", "0.02")))
    risk_usd = max(0.0, eq * rp)

    info = mt5.symbol_info(symbol)
    tick_size = info.trade_tick_size or info.point or 0.0001
    tick_val  = info.trade_tick_value or 0.0

    # Fallback if broker doesn't populate tick_value
    if tick_val <= 0.0:
        contract = info.trade_contract_size or 100000.0  # 1 lot nominal
        tick_val = contract * tick_size  # crude fallback

    ticks = abs(entry - sl) / tick_size
    loss_per_lot = ticks * tick_val
    if loss_per_lot <= 0:
        print("[ERR] non-positive loss_per_lot; check prices/tick settings.")
        return 0.0

    vol = risk_usd / loss_per_lot
    return _round_volume(symbol, vol)

def rr_prices(side:str, entry:float, atr:float, *, atr_mult:float=1.5, rr:float=3.0):
    """Return (sl, tp) prices for 1:R:rr using ATR for SL distance."""
    if side.lower()=="buy":
        sl = entry - atr_mult*atr
        tp = entry + rr*(entry - sl)
    else:
        sl = entry + atr_mult*atr
        tp = entry - rr*(sl - entry)
    return (float(sl), float(tp))

def place_market_rr_trade(symbol:str, side:str, atr:float, *,
                          rr:float=3.0, atr_mult:float=1.5, risk_pct:float=None,
                          deviation_points:int=20, dry_run:bool=True, comment:str="bot-rr"):
    """Market order risking risk_pct of (current) equity with 1:rr RR using ATR-based SL.
       dry_run=True prints the plan without sending."""
    if not _select(symbol):
        return None

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("[ERR] symbol_info_tick failed")
        return None

    side = side.lower()
    entry = float(tick.ask if side=="buy" else tick.bid)
    sl, tp = rr_prices(side, entry, atr, atr_mult=atr_mult, rr=rr)

    ai = mt5.account_info()
    eq_now = float(ai.equity or ai.balance)

    rp = float(risk_pct if risk_pct is not None else float(os.getenv("RISK_PCT", "0.02")))
    vol = compute_volume_for_risk(symbol, entry, sl, equity=eq_now, risk_pct=rp)
    if vol <= 0.0:
        print("[ERR] computed volume <= 0; aborting.")
        return None

    print(f"[PLAN] {side.upper()} {symbol} @ {entry:.5f} | SL {sl:.5f} | TP {tp:.5f} | RR 1:{rr} | "
          f"risk {rp*100:.2f}% of ${eq_now:.2f} -> vol {vol} lots")

    if dry_run:
        return {"dry_run": True, "symbol": symbol, "side": side, "entry": entry, "sl": sl, "tp": tp,
                "volume": vol, "risk_pct": rp, "equity": eq_now}

    req = {
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   symbol,
        "volume":   vol,
        "type":     mt5.ORDER_TYPE_BUY if side=="buy" else mt5.ORDER_TYPE_SELL,
        "price":    entry,
        "sl":       sl,
        "tp":       tp,
        "deviation": deviation_points,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment":  comment,
    }
    res = mt5.order_send(req)
    if res is None:
        print("[ERR] order_send returned None")
        return None
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERR] order_send failed: retcode={res.retcode}  comment={getattr(res,'comment', '')}")
    else:
        print(f"[OK ] ticket={res.order or res.order}  price={entry:.5f}  vol={vol}  sl={sl:.5f}  tp={tp:.5f}")
    return res
