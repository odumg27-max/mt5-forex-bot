import os, time, math, numpy as np, pandas as pd, MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

# --- project imports (reuse your indicators + signals + sessions) ---
from bot_mt5 import init_mt5, add_indicators, SYMBOL as BOT_SYMBOL
from backtest_obos_cent_realistic import build_signals_v2, session_mask_index

ASCII_TS = lambda: datetime.now().strftime("%H:%M:%S")

# ---------- helpers ----------
def _sym_info(symbol):
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError("symbol_info is None; is MT5 connected & symbol visible?")
    if not si.visible:
        mt5.symbol_select(symbol, True)
    return si

def round_lots(symbol, lots):
    si = _sym_info(symbol)
    step = si.volume_step or 0.01
    lots = math.floor(lots/step)*step
    lots = max(si.volume_min or 0.01, min(lots, si.volume_max or lots))
    return round(lots, 2)

def usd_per_lot_for_move(symbol, price_move):
    si = _sym_info(symbol)
    ticks = abs(price_move) / (si.trade_tick_size or si.point)
    return ticks * (si.trade_tick_value or 0.0)

def df_from_pos(symbol, timeframe, bars=2000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates)==0: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})

def last_closed_m1_time(symbol):
    rr = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 3)
    if rr is None or len(rr) < 2: return None
    # penultimate bar is the last CLOSED bar
    return pd.to_datetime(rr[-2]["time"], unit="s", utc=True)

def place_market(symbol, long, lots, sl, tp, deviation_pts=10):
    si = _sym_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: 
        return None, "no tick"
    price = tick.ask if long else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "type": mt5.ORDER_TYPE_BUY if long else mt5.ORDER_TYPE_SELL,
        "price": price,
        "volume": float(lots),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": int(deviation_pts),
        "magic": 880015,
        "comment": "fronttest_obos_v2",
        "type_filling": mt5.ORDER_FILLING_IOC if si.trade_fill_mode in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK) else mt5.ORDER_FILLING_RETURN
    }
    res = mt5.order_send(request)
    return res, None

def move_sl_to_be_if_needed(symbol, be_book):
    """be_book: {ticket: {long, entry, r1, moved}}"""
    if not be_book: return
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return
    price_up  = tick.bid  # for shorts BE checks vs bid
    price_dn  = tick.ask  # for longs BE checks vs ask

    positions = mt5.positions_get(symbol=symbol)
    if positions is None: positions = []
    pos_by_ticket = {p.ticket: p for p in positions}

    for ticket, st in list(be_book.items()):
        if ticket not in pos_by_ticket:
            # position is closed
            be_book.pop(ticket, None)
            continue
        if st["moved"]: 
            continue
        long = st["long"]; r1 = st["r1"]; entry = st["entry"]
        hit = (price_dn >= r1) if long else (price_up <= r1)
        if not hit: 
            continue

        # modify SL to entry (BE)
        p = pos_by_ticket[ticket]
        new_sl = entry if long else entry
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": symbol,
            "sl": float(new_sl),
            "tp": float(p.tp),
            "magic": 880015,
            "comment": "BE_move"
        }
        r = mt5.order_send(req)
        if r and r.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, 10009, 10008):
            st["moved"] = True
            print(f"[{ASCII_TS()}] BE moved on #{ticket} -> SL={new_sl}")
        else:
            print(f"[{ASCII_TS()}] BE move failed on #{ticket}: {getattr(r,'retcode',None)}")

# ---------- main loop ----------
def main():
    init_mt5()
    symbol = os.getenv("SYMBOL") or BOT_SYMBOL or "EURUSD"
    si = _sym_info(symbol)
    pt = si.point

    RISK_PCT = float(os.getenv("RISK_PCT", "0.02"))
    SPREAD_CAP_POINTS = float(os.getenv("SPREAD_CAP_POINTS", "35"))
    DEVIATION_POINTS = int(os.getenv("DEVIATION_POINTS", "10"))
    COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "20"))
    PER_HOUR_CAP = int(os.getenv("PER_HOUR_CAP", "3"))
    NEWS_ATR_MULT = float(os.getenv("NEWS_ATR_MULT", "4"))
    DRY_RUN = os.getenv("DRY_RUN","0") == "1"

    print(f"MT5 ready. Forward-testing {symbol} (risk={RISK_PCT*100:.1f}%, spread cap={SPREAD_CAP_POINTS} pts, dev={DEVIATION_POINTS} pts, BE@+1R, dry_run={DRY_RUN})")

    last_bar = None
    last_trade_time = None
    hour_count = 0
    hour_anchor = None
    be_book = {}

    while True:
        try:
            # heartbeat + last closed m1
            lc = last_closed_m1_time(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                spr_pts = (tick.ask - tick.bid) / pt
            else:
                spr_pts = float("inf")

            if lc != last_bar:
                # new M1 close
                last_bar = lc
                ts = lc

                print(f"[{ASCII_TS()}] M1 closed @ {str(ts)} | spread={spr_pts:.0f} pts")

                # fetch fresh history (enough for warmups)
                h1 = df_from_pos(symbol, mt5.TIMEFRAME_H1,  600)
                m15= df_from_pos(symbol, mt5.TIMEFRAME_M15, 800)
                m1 = df_from_pos(symbol, mt5.TIMEFRAME_M1,  2500)
                if h1.empty or m15.empty or m1.empty:
                    print(f"[{ASCII_TS()}] Not enough history yet.")
                    time.sleep(2); 
                    move_sl_to_be_if_needed(symbol, be_book)
                    continue

                # indicators
                h1i  = add_indicators(h1)
                m15i = add_indicators(m15)
                m1i  = add_indicators(m1)

                # signals
                long_sig, short_sig, _ = build_signals_v2(
                    h1i, m15i, m1i,
                    warmup=300, adx_min=20, vol_mult=1.10, don_len=20, supertrend_req=True
                )

                # allowed sessions + cooldown
                in_session = session_mask_index(m1i.index)
                allowed_now = False
                if ts in m1i.index:
                    allowed_now = bool(in_session.loc[ts])

                # cooldown limiter
                if hour_anchor is None or (ts - hour_anchor) >= pd.Timedelta(hours=1):
                    hour_anchor = ts
                    hour_count = 0

                if last_trade_time is not None and (ts - last_trade_time) < pd.Timedelta(minutes=COOLDOWN_MIN):
                    allowed_now = False
                if hour_count >= PER_HOUR_CAP:
                    allowed_now = False

                # spread cap
                if spr_pts > SPREAD_CAP_POINTS:
                    allowed_now = False

                # resolve signal at `ts`
                go_long = bool(long_sig.loc[ts]) if ts in long_sig.index else False
                go_short= bool(short_sig.loc[ts]) if ts in short_sig.index else False
                will_trade = allowed_now and (go_long or go_short)

                # news-spike rejection (on the signal bar)
                if will_trade:
                    row = m1i.loc[ts]
                    atr = float(row.get("atr14", np.nan))
                    if not np.isfinite(atr) or atr <= 0:
                        will_trade = False
                    elif (row["high"] - row["low"]) > NEWS_ATR_MULT * atr:
                        will_trade = False

                # avoid stacking positions on same symbol (one-at-a-time)
                if will_trade:
                    open_pos = mt5.positions_get(symbol=symbol)
                    if open_pos and len(open_pos) > 0:
                        will_trade = False
                        print(f"[{ASCII_TS()}] Position already open -> skip new entry.")

                if not will_trade:
                    move_sl_to_be_if_needed(symbol, be_book)
                    time.sleep(1.5)
                    continue

                long = go_long
                side = "BUY" if long else "SELL"
                entry = m1i.loc[ts, "close"]

                # position sizing
                atr = float(m1i.loc[ts, "atr14"])
                sl  = entry - 1.5*atr if long else entry + 1.5*atr
                oneR = abs(entry - sl)
                usd_per_lot = usd_per_lot_for_move(symbol, oneR) or 0.0
                if usd_per_lot <= 0:
                    print(f"[{ASCII_TS()}] Bad usd_per_lot -> skip.")
                    move_sl_to_be_if_needed(symbol, be_book); time.sleep(1.5); continue

                # risk allocation
                equity_info = mt5.account_info()
                balance = float(getattr(equity_info, "balance", 100.0))
                risk_usd = balance * RISK_PCT
                lots = round_lots(symbol, risk_usd / usd_per_lot)
                if lots <= 0:
                    print(f"[{ASCII_TS()}] Lots<=0 -> skip.")
                    move_sl_to_be_if_needed(symbol, be_book); time.sleep(1.5); continue

                # actual market entry price (bid/ask) and SL/TP as prices
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print(f"[{ASCII_TS()}] No tick -> skip.")
                    move_sl_to_be_if_needed(symbol, be_book); time.sleep(1.5); continue

                price = tick.ask if long else tick.bid
                # recompute SL/TP off real entry
                sl  = price - 1.5*atr if long else price + 1.5*atr
                tp  = price + 3.0*(price - sl) if long else price - 3.0*(sl - price)
                r1  = price + (price - sl) if long else price - (sl - price)

                print(f"[{ASCII_TS()}] {side} {symbol} lots={lots} price={price:.5f} sl={sl:.5f} tp={tp:.5f} (spread={spr_pts:.0f} pts)")

                if DRY_RUN:
                    # pretend we placed order; add to BE watch (demo only)
                    fake_ticket = int(time.time())
                    be_book[fake_ticket] = {"long": long, "entry": price, "r1": r1, "moved": False}
                    last_trade_time = ts; hour_count += 1
                    move_sl_to_be_if_needed(symbol, be_book)
                    time.sleep(1.5)
                    continue

                res, err = place_market(symbol, long, lots, sl, tp, deviation_pts=DEVIATION_POINTS)
                if err:
                    print(f"[{ASCII_TS()}] order_send error: {err}")
                elif res and res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, 10009, 10008):
                    ticket = getattr(res, "order", None) or getattr(res, "deal", None) or 0
                    be_book[int(ticket)] = {"long": long, "entry": price, "r1": r1, "moved": False}
                    last_trade_time = ts
                    hour_count += 1
                    print(f"[{ASCII_TS()}] ORDER OK ticket={ticket}")
                else:
                    print(f"[{ASCII_TS()}] ORDER FAIL retcode={getattr(res,'retcode',None)}")

                # after entry, try BE checks
                move_sl_to_be_if_needed(symbol, be_book)

            else:
                # no new bar yet: do BE checks and heartbeat every ~5s
                move_sl_to_be_if_needed(symbol, be_book)
                time.sleep(2.0)

        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"[{ASCII_TS()}] loop error: {e}")
            time.sleep(2.0)

if __name__ == "__main__":
    main()
