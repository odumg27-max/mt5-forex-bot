import numpy as np, pandas as pd, MetaTrader5 as mt5
from bot_mt5 import init_mt5, SYMBOL, add_indicators
from compounding_rr import place_market_rr_trade

def latest_atr_m1(symbol: str, bars: int = 1000) -> float:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars)
    if rates is None or len(rates) == 0:
        print("No M1 data; open M1 chart / ensure MT5 is connected.")
        return float("nan")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    i = add_indicators(df)
    return float(i["atr14"].iloc[-1])

def main():
    init_mt5()
    atr = latest_atr_m1(SYMBOL)
    if not np.isfinite(atr) or atr <= 0:
        print("ATR not ready.")
        return

    # Demo: pretend we have a long signal now. Set dry_run=False to actually send.
    print("=== DRY RUN (no order sent) ===")
    place_market_rr_trade(SYMBOL, "buy", atr=atr, rr=3.0, atr_mult=1.5, dry_run=True)
    # Example if your signal is short:
    # place_market_rr_trade(SYMBOL, "sell", atr=atr, rr=3.0, atr_mult=1.5, dry_run=True)

if __name__ == "__main__":
    main()
