import os, pandas as pd, numpy as np
from os import getenv

def load_env():
    # Try python-dotenv first, then fall back to a tiny parser
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        if os.path.exists(".env"):
            with open(".env","r",encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith("#") or "=" not in line: 
                        continue
                    k,v=line.split("=",1)
                    os.environ.setdefault(k.strip(), v.strip())

def main():
    load_env()
    # Defaults: $100 start, 2% risk per trade if RISK_PCT is not set in .env
    start_equity = float(getenv("START_EQUITY", "100"))
    risk_pct     = float(getenv("RISK_PCT", "0.02"))

    path = os.path.join("logs","trades_45d.csv")
    if not os.path.exists(path):
        print("Missing logs/trades_45d.csv. Run trades_45days.py first.")
        return

    df = pd.read_csv(path)
    if "R" not in df.columns:
        print("File found but no 'R' column present.")
        return

    # Ensure numeric R and ordered by i if available
    df["R"] = pd.to_numeric(df["R"], errors="coerce").fillna(0.0)
    if "i" in df.columns:
        df = df.sort_values("i")

    equity = start_equity
    eq_before=[]; eq_after=[]; risk_usd_list=[]; pnl_usd_list=[]
    for r in df["R"].tolist():
        eq_before.append(equity)
        risk_usd = equity * risk_pct
        pnl_usd  = r * risk_usd
        equity  += pnl_usd
        eq_after.append(equity)
        risk_usd_list.append(risk_usd)
        pnl_usd_list.append(pnl_usd)

    df["equity_before"] = np.round(eq_before, 2)
    df["risk_usd"]      = np.round(risk_usd_list, 2)
    df["pnl_usd"]       = np.round(pnl_usd_list, 2)
    df["equity_after"]  = np.round(eq_after, 2)

    wins     = int((df["R"] > 0).sum())
    losses   = int((df["R"] < 0).sum())
    timeouts = int((df["R"] == 0).sum())
    considered = wins + losses
    winrate = (wins/considered*100.0) if considered > 0 else 0.0

    total_R    = float(df["R"].sum())
    total_pnl  = float(df["pnl_usd"].sum())
    final_eq   = float(df["equity_after"].iloc[-1]) if len(df) else start_equity

    # Max drawdown in dollars (peak-to-valley on equity_after)
    eq_vals = df["equity_after"].values
    if len(eq_vals):
        peaks = np.maximum.accumulate(eq_vals)
        dd = eq_vals - peaks
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    out_path = os.path.join("logs","trades_45d_with_equity.csv")
    os.makedirs("logs", exist_ok=True)
    df.to_csv(out_path, index=False)

    print("=== EQUITY REPORT (last 45 days) ===")
    print(f"Start equity: ${start_equity:.2f}")
    print(f"Risk/Trade: {risk_pct*100:.2f}% (compounding)")
    print(f"Trades: {len(df)}  Wins: {wins}  Losses: {losses}  Timeouts: {timeouts}")
    print(f"Win rate (excl. timeouts): {winrate:.1f}%")
    print(f"Total PnL (R): {total_R:.1f}")
    print(f"Total PnL ($): ${total_pnl:.2f}")
    print(f"Final equity: ${final_eq:.2f}  Max drawdown: ${max_dd:.2f}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
