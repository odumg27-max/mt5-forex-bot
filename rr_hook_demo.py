import pandas as pd, numpy as np, MetaTrader5 as mt5
from bot_mt5 import init_mt5, SYMBOL, add_indicators
from compounding_rr import place_market_rr_trade

def main(side="buy", dry_run=True):
    init_mt5()
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 1500)
    if rates is None or len(rates) == 0:
        print("No M1 data; open the M1 chart and ensure MT5 is connected.")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume":"volume"}, inplace=True)

    i = add_indicators(df)
    atr_now = float(i["atr14"].iloc[-1])
    print(f"ATR (M1) now: {atr_now:.6f}")

    # 2% risk (from your .env or default inside compounding_rr), RR=1:3, SL=1.5*ATR
    place_market_rr_trade(
        SYMBOL, side,
        atr=atr_now, rr=3.0, atr_mult=1.5,
        dry_run=dry_run
    )

if __name__ == "__main__":
    # Change to "sell" or set dry_run=False when you're ready
    main(side="buy", dry_run=True)
