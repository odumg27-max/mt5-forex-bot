import os, MetaTrader5 as mt5, sys, time
path = os.getenv("MT5_TERMINAL_PATH")
login = int(os.getenv("MT5_LOGIN","0") or "0")
password = os.getenv("MT5_PASSWORD","")
server = os.getenv("MT5_SERVER","")
symbol = os.getenv("SYMBOL","EURUSD")

print("Diag: init...", flush=True)
ok = mt5.initialize(path) if (path and os.path.exists(path)) else mt5.initialize()
print("initialize:", ok, "last_error:", mt5.last_error(), flush=True)
if not ok: sys.exit(2)

if login and password and server:
    ok2 = mt5.login(login, password=password, server=server)
    print("login:", ok2, "last_error:", mt5.last_error(), flush=True)
else:
    print("Missing login/server/password in env.", flush=True)

info = mt5.account_info()
print("account_info:", bool(info), "| login:", getattr(info,'login',None), "| server:", getattr(info,'server',None), flush=True)

si = mt5.symbol_info(symbol)
print("symbol_info before select:", bool(si), "| visible:", getattr(si,'visible',None), flush=True)
if si is None or not si.visible:
    mt5.symbol_select(symbol, True)
    si = mt5.symbol_info(symbol)
print("symbol_info after select:", bool(si), "| visible:", getattr(si,'visible',None), flush=True)

mt5.shutdown()
