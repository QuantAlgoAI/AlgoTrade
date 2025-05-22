# AlgoTrade - Advanced Algorithmic Trading Bot

A sophisticated algorithmic trading bot with real-time market data processing, multiple trading strategies, and comprehensive analytics.

## Features

### Core Trading
- Multi-symbol trading (NIFTY, BANKNIFTY, FINNIFTY)
- Real-time market data processing
- Multiple trading strategies:
  - RSI-based trading
  - MACD strategy
  - Bollinger Bands
- Automated trade execution
- Risk management system

### Analytics & Monitoring
- Real-time performance tracking
- Trade distribution analysis
- Performance metrics visualization
- Pattern recognition
- Win/loss ratio analysis
- Symbol-specific performance tracking
- Time-based analysis (hourly/daily patterns)

### Risk Management
- Position sizing based on volatility
- Dynamic stop-loss and take-profit
- Portfolio-level risk controls
- Maximum drawdown limits

### Technical Features
- Real-time market data using yfinance
- SmartAPI integration for order execution
- SQLite database for trade tracking
- Flask web interface
- Docker containerization
- Comprehensive logging system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/QuantAlgoAI/AlgoTrade.git
cd AlgoTrade
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the trading bot:
```bash
python AlgoTrade.py
```

2. Access the web interface:
```bash
python app.py
```

3. View analytics:
```bash
python analytics.py
```

## Project Structure

```
AlgoTrade/
├── AlgoTrade.py          # Main trading bot
├── analytics.py          # Analytics and visualization
├── config.py            # Configuration settings
├── models.py            # Database models
├── notifier.py          # Notification system
├── security.py          # Security utilities
├── requirements.txt     # Dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker compose setup
├── data_cache/         # Market data cache
├── logs/              # Log files
└── reports/           # Generated reports
```

## Development Roadmap

### Phase 1: Core System Enhancement
- [x] Risk management system
- [x] Strategy optimization
- [x] Performance analytics

### Phase 2: Advanced Analytics
- [ ] Machine learning integration
- [ ] Enhanced visualization
- [ ] Pattern recognition

### Phase 3: Monitoring & Alerts
- [ ] Alert system
- [ ] Real-time monitoring
- [ ] Performance tracking

### Phase 4: Testing & Quality
- [ ] Testing framework
- [ ] Quality assurance
- [ ] Documentation

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- yfinance for market data
- SmartAPI for trading integration
- Flask for web interface
- Plotly for visualizations 