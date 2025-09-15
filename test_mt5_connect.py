import os, sys
from dotenv import load_dotenv
import MetaTrader5 as mt5

load_dotenv()
symbol = os.getenv("SYMBOL","EURUSD")
term   = os.getenv("MT5_TERMINAL_PATH")
login  = os.getenv("MT5_LOGIN")
pwd    = os.getenv("MT5_PASSWORD")
server = os.getenv("MT5_SERVER")

# Initialize once (with or without credentials/terminal path)
if term:
    ok = mt5.initialize(term, login=int(login) if login else None,
                        password=pwd if pwd else None,
                        server=server if server else None)
else:
    ok = mt5.initialize(login=int(login) if login else None,
                        password=pwd if pwd else None,
                        server=server if server else None)

if not ok:
    print("❌ MT5 init failed:", mt5.last_error()); sys.exit(1)

if not mt5.symbol_select(symbol, True):
    print(f"❌ Failed to select {symbol}"); sys.exit(1)

tick = mt5.symbol_info_tick(symbol)
info = mt5.symbol_info(symbol)
if not tick or not info:
    print("❌ Could not read symbol info/tick"); sys.exit(1)

spread_pts = (tick.ask - tick.bid) / info.point
print("✅ Connected.")
print(f"Symbol: {symbol}  Bid: {tick.bid:.5f}  Ask: {tick.ask:.5f}  Spread(points): {spread_pts:.1f}")
mt5.shutdown()
