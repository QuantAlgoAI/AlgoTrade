# AlgoTrade Project Architecture

## Overview
AlgoTrade is a modular, event-driven options trading bot for Indian markets (NSE, BSE, MCX) using the Angel One SmartAPI. It is designed for automated trading with a focus on high win-rate strategies, risk management, and robust error handling. The codebase is structured for extensibility, maintainability, and real-time execution.

---

## High-Level Architecture Diagram

```
+-------------------+         +-------------------+         +-------------------+
|                   |         |                   |         |                   |
|   Data Sources    +-------->+   Data Handler    +-------->+   Strategy Engine |
| (SmartAPI, Files) |         |                   |         |                   |
+-------------------+         +-------------------+         +-------------------+
                                                                |
                                                                v
                                                    +-----------------------+
                                                    |   Trade Execution     |
                                                    |  (Order Placement)    |
                                                    +-----------------------+
                                                                |
                                                                v
                                                    +-----------------------+
                                                    |  Trade State Manager  |
                                                    +-----------------------+
                                                                |
                                                                v
                                                    +-----------------------+
                                                    | Notification/Logging  |
                                                    +-----------------------+
```

---

## Components

### 1. Data Sources
- **SmartAPI WebSocket:** Real-time tick data for subscribed tokens.
- **Instrument Files:** Daily CSV files with contract metadata.

### 2. Data Handler
- Downloads and parses instrument files.
- Filters and selects relevant contracts.
- Maintains a contract hub for quick lookup.
- Handles expiry and token selection logic.

### 3. Strategy Engine
- **HighWinRateStrategy:**
  - Maintains a rolling DataFrame of recent ticks.
  - Calculates indicators (EMA, RSI, VWAP, ATR, volume/OI MAs, support/resistance).
  - Generates buy/sell signals based on multi-indicator logic.
  - Implements risk management (trailing stop, daily loss cap, max trades).
- Extensible: New strategies can be added as classes.

### 4. Trade Execution
- Places market orders via SmartAPI.
- Handles order status, error handling, and logging.
- Supports both live and paper trading.

### 5. Trade State Manager
- Tracks the status, entry/exit, and parameters for each monitored token.
- Maintains a list of all trades (open and closed).
- Exports trade history and statistics.

### 6. Notification/Logging
- **Logging:** File and console logging with error tracebacks.
- **TelegramNotifier:** Sends trade and error notifications to Telegram.

---

## Data Flow

1. **Startup:**
   - Logging and notifier initialized.
   - Logs in to SmartAPI and processes instrument files.
2. **Instrument Selection:**
   - User selects instrument (e.g., NIFTY 50).
   - Expiry and tokens for ATM CE/PE are selected.
3. **WebSocket Initialization:**
   - Subscribes to tokens for live tick data.
   - Initializes trade state for each token.
4. **Tick Processing:**
   - On each tick:
     - Updates trade state with latest LTP and timestamp.
     - Updates strategy data and generates signals.
     - Handles trade entries/exits based on signals and risk management.
     - Sends notifications for trade actions.
5. **Trade Management:**
   - Tracks open/closed trades, PnL, and trade statistics.
   - Exports trades to CSV for analysis.
6. **Shutdown:**
   - Handles graceful shutdown and sends Telegram notification.

---

## Extensibility
- **Strategy:** Add new strategy classes implementing the same interface as `HighWinRateStrategy`.
- **Data Sources:** Integrate new data feeds by extending the Data Handler.
- **Notification:** Add new notification channels (e.g., email, Slack) by extending the notifier system.
- **Risk Management:** Parameterize or add new risk modules.

---

## Key Files
- `AlgoTrade.py`: Main application logic, trading loop, and orchestration.
- `ARCHITECTURE.md`: This architecture document.
- `logs/`: Log files for each trading day.
- `instruments/`: Instrument files for contract metadata.

---

## Summary
AlgoTrade is a robust, extensible, and modular trading bot designed for real-time options trading in Indian markets. Its architecture separates data handling, strategy, execution, and notification, making it easy to maintain and extend. 