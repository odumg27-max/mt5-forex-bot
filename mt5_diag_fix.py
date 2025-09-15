import os, re
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import MetaTrader5 as mt5

load_dotenv()
base = os.getenv("SYMBOL", "EURUSD").upper()

def init():
    from bot_mt5 import MT5_TERMINAL_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    if MT5_TERMINAL_PATH:
        mt5.initialize(MT5_TERMINAL_PATH, login=int(MT5_LOGIN) if MT5_LOGIN else None,
                       password=MT5_PASSWORD, server=MT5_SERVER)
    else:
        mt5.initialize(login=int(MT5_LOGIN) if MT5_LOGIN else None,
                       password=MT5_PASSWORD, server=MT5_SERVER)
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error()); raise SystemExit(1)

def bars_count(sym, tf, days=30):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days+5)  # buffer over weekends
    rates = mt5.copy_rates_range(sym, tf, start, now)
    return 0 if rates is None else len(rates)

def pick_best(cands):
    # Choose the symbol with the MOST M1 bars
    from bot_mt5 import TIMEFRAME_MAP
    scored = []
    for s in cands:
        mt5.symbol_select(s, True)
        m1  = bars_count(s, TIMEFRAME_MAP["M1"])
        m15 = bars_count(s, TIMEFRAME_MAP["M15"])
        h1  = bars_count(s, TIMEFRAME_MAP["H1"])
        total = m1 + m15 + h1
        scored.append((s, m1, m15, h1, total))
    scored.sort(key=lambda x: x[4], reverse=True)
    return scored

def update_env_symbol(new_sym):
    path = ".env"
    if not os.path.exists(path):
        print("No .env found; skipping update."); return
    txt = open(path, "r", encoding="utf-8").read()
    if re.search(r"^SYMBOL=", txt, flags=re.M):
        txt = re.sub(r"^SYMBOL=.*$", f"SYMBOL={new_sym}", txt, flags=re.M)
    else:
        txt += f"\nSYMBOL={new_sym}\n"
    open(path, "w", encoding="utf-8").write(txt)
    print(f"Updated .env → SYMBOL={new_sym}")

def main():
    init()
    # Gather candidates that contain the base (EURUSD) in their name
    syms = mt5.symbols_get()
    cands = [s.name for s in syms if base in s.name.upper()]
    if not cands:
        print(f"No symbols matching '{base}' found. Open MT5 → Market Watch → Symbols and enable a {base} symbol.")
        return
    best = pick_best(cands)
    print("Candidates (symbol, M1, M15, H1, total):")
    for row in best:
        print(row)
    chosen = best[0][0]
    print(f"\nChosen symbol: {chosen}")
    update_env_symbol(chosen)

if __name__ == "__main__":
    main()
