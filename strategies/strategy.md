# HighWinRateStrategy Logic

## Key Parameters and Timeframe

- **EMA (Exponential Moving Average):**
  - Fast EMA period: 9
  - Slow EMA period: 21
- **RSI (Relative Strength Index):**
  - Period: 14
  - Overbought threshold: 70
  - Oversold threshold: 30
- **VWAP (Volume Weighted Average Price):**
  - Calculated on rolling tick data (no fixed period)
- **Volume Moving Average (MA):**
  - Period: 20
- **Open Interest (OI) Moving Average (MA):**
  - Period: 20
- **ATR (Average True Range):**
  - Period: 14
- **Support/Resistance:**
  - Window: 20 (rolling window for local highs/lows)
  - Threshold: 2% (0.02) for grouping nearby levels
- **Timeframe:** Tick-based (every tick from WebSocket, not fixed candles)
- **Rolling window:** Last 100 ticks are kept for indicator calculations

| Indicator         | Period/Length | Usage in Logic                |
|-------------------|---------------|-------------------------------|
| Fast EMA          | 9             | Trend/crossover               |
| Slow EMA          | 21            | Trend/crossover               |
| RSI               | 14            | Overbought/oversold filter    |
| Volume MA         | 20            | Confirmation                  |
| OI MA             | 20            | Confirmation                  |
| ATR               | 14            | Volatility/regime detection   |
| Support/Resistance| 20 (window)   | Price action filter           |
| VWAP              | Rolling       | Price action filter           |

---

## Strategy Flow Diagram

```
+-------------------+
|  Tick Received    |
+-------------------+
          |
          v
+-----------------------------+
|  Update DataFrame with Tick |
+-----------------------------+
          |
          v
+-----------------------------+
|  Calculate Indicators:      |
|  - EMA (fast/slow)          |
|  - RSI                      |
|  - VWAP                     |
|  - Volume/OI MAs            |
|  - Support/Resistance       |
+-----------------------------+
          |
          v
+-----------------------------+
|  Generate Signals:          |
|  - Buy if:                  |
|    * EMA fast crosses above |
|      slow                   |
|    * RSI < 70               |
|    * Volume > MA            |
|    * OI > MA                |
|    * Price > VWAP           |
|    * Near support           |
|  - Sell if:                 |
|    * EMA fast crosses below |
|      slow                   |
|    * RSI > 30               |
|    * Volume > MA            |
|    * OI > MA                |
|    * Price < VWAP           |
|    * Near resistance        |
+-----------------------------+
          |
          v
+-----------------------------+
|  If Buy/Sell Signal:        |
|  - Place order              |
|  - Update trade state       |
+-----------------------------+
          |
          v
+-----------------------------+
|  Monitor for exit:          |
|  - Stop loss                |
|  - Target                   |
|  - Signal reversal          |
|  - Trailing SL/Partial exit |
+-----------------------------+
```

## Why You Might Not Be Getting Trades

- **All conditions must be met** for a buy/sell signal:
  - EMA crossover (fast/slow)
  - RSI threshold
  - Volume and OI above their moving averages
  - Price above/below VWAP
  - Price near support/resistance
- **If even one condition is not met, no trade is triggered.**
- **Market may be ranging or not volatile enough** for the signals to trigger.
- **Strategy is conservative by design** (to avoid false signals).

## What You Can Do

1. **Check logs:**  Look for lines like `Current Signals: ...` and `Trade Status: ...` to see if signals are being generated but not triggering trades.
2. **Relax conditions:**  Temporarily comment out or adjust some conditions (e.g., remove support/resistance check, lower volume/OI thresholds) to see if trades trigger.
3. **Add debug prints:**  Print indicator values and which condition is failing for each tick.
4. **Test with historical data:**  Use a backtest or replay mode to see if the strategy would trigger trades in a more volatile period.

---

**Summary:**

HighWinRateStrategy is a multi-indicator, high-confidence system. It will only trade when all the following align:
- Trend (EMA crossover)
- Momentum (RSI)
- Confirmation (Volume/OI)
- Price action (VWAP, support/resistance)

This is good for safety, but can mean long waits between trades. Tune thresholds or conditions for more/less frequent signals as needed. 