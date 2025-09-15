# Phase 3 — Strategy Overview (Top-Down H1 → M15 → M1)

**Goal:** 1-minute scalping that trades *with* higher-timeframe trend, only after a structure shift on M15, and confirms momentum + breakout on M1. Includes breakout trading, momentum following, and strict risk controls.

---

## 1) Higher Timeframe Trend Gate — H1
- **Trend filter:** EMA(50) vs EMA(200) + ADX(14)
  - **Uptrend:** EMA50 > EMA200 **and** ADX(14) > 18
  - **Downtrend:** EMA50 < EMA200 **and** ADX(14) > 18
  - Otherwise = **chop** (no trades)

---

## 2) Structure/BOS Gate — M15
- Detect **market-structure shift** using **Donchian(20)**:
  - For **longs:** price closes **above** prior Donchian high (break of resistance)
  - For **shorts:** price closes **below** prior Donchian low (break of support)

---

## 3) Entry — M1 Momentum + Breakout
**Long (uptrend path) requires all:**
1. EMAs aligned: **EMA9 > EMA21 > EMA50**
2. **MACD(12,26,9) histogram > 0** and rising vs prior bar
3. **Supertrend(10,3)** direction = **up**
4. **ADX(14) > 18**
5. **Volume** > 20-bar volume MA × **1.05**
6. **Breakout:** close > **Donchian(20) high** (prior bar not broken)

**Short (downtrend path) requires all:**
1. EMAs aligned: **EMA9 < EMA21 < EMA50**
2. **MACD histogram < 0** and falling vs prior bar
3. **Supertrend** direction = **down**
4. **ADX(14) > 18**
5. **Volume** > 20-bar volume MA × **1.05**
6. **Breakout:** close < **Donchian(20) low** (prior bar not broken)

**Spread filter:** skip if current spread > `MAX_SPREAD_POINTS` (from .env).

---

## 4) Risk Management (strict)
- **Position size:** 1–2% risk per trade (uses account balance & SL distance)
- **SL/TP:** ATR(14)-based dynamic **SL = ATR × 1.5**, **TP = 3R** (R:R = **1:3**)
- **Trailing stop:** ATR trail (moves SL as price advances)
- **Daily kill-switch:** stop trading if daily PnL ≤ `DAILY_LOSS_STOP_USD`

---

## 5) Breakout & Momentum Mapping (your rules)
- **Breakout trading:** enter on strong S/R breaks with **Donchian(20)** + volume confirm
- **Momentum following:** only trade when EMAs are aligned, MACD supports, Supertrend agrees, and ADX confirms trend strength

---

## 6) Indicators Used (≥ 10 trend-following)
1. **EMA(9)**
2. **EMA(21)**
3. **EMA(50)**
4. **EMA(200)**
5. **WMA(100)**
6. **HMA(55)**
7. **MACD(12,26,9) histogram**
8. **Supertrend(10,3)**
9. **ADX(14)** (+DI/−DI internally)
10. **Donchian(20)**
11. **Ichimoku Kijun (baseline)**
12. **ATR(14)** (for SL/TP & trailing)
13. **Volume MA(20)** (confirmation)

> All of the above are already wired in `bot_mt5.py`:
> - Multi-TF pulls (H1, M15, M1)
> - H1 trend gate → M15 BOS → M1 momentum + breakout
> - ATR SL/TP, trailing stop, spread filter, daily loss kill-switch
> - Optional ML gate (set `MODEL_ON=true` and provide `models/model.joblib`)
