# Production Trading System

A comprehensive, modular Python trading system with multi-timeframe analysis, automated learning, and dual-platform execution (MT5 for metals, Binance for crypto).

## 🚀 Features

### Core Capabilities
- **Multi-Timeframe Analysis**: Analyzes 1H, 15m, and 5m timeframes for high-probability setups
- **15+ Technical Indicators**: EMA, RSI, MACD, SuperTrend, Bollinger Bands, ATR, and more
- **Dual Platform Execution**: 
  - MT5 via Socket/EA Bridge for metals (XAU/USD)
  - Binance REST & WebSocket for crypto (BTC, ETH, etc.)
- **Advanced Risk Management**:
  - Dynamic position sizing
  - Trailing stops with RR activation
  - Partial profit-taking
  - Break-even moves
  - Global risk limits
- **Automated Learning Engine**: 
  - Parameter optimization (Grid Search, Random Search, RL Bandit)
  - Performance-based strategy adaptation
  - Safe versioning and rollback
- **Production-Ready Logging**: SQLite database with full audit trail
- **Interactive Dashboard**: Real-time Streamlit interface
- **Backtesting Framework**: Realistic simulation with slippage and commission

## 📁 Project Structure

```
trader_project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variables template
├── config/
│   └── config.yaml             # Runtime configuration
├── data_feed/
│   └── market_client.py        # Market data (historical + live WebSocket)
├── indicators/
│   └── indicators.py           # Technical indicators library
├── strategy/
│   └── engine.py               # Multi-TF analysis & decision logic
├── risk_management/
│   ├── money_manager.py        # Position sizing
│   └── stop_manager.py         # SL/TP/trailing stop logic
├── execution/
│   ├── mt5_bridge.py           # MT5 Socket/EA bridge
│   └── binance_api.py          # Binance REST & WebSocket
├── logger/
│   ├── db.py                   # SQLite database manager
│   └── audit_logger.py         # Structured logging
├── learning/
│   └── learner.py              # Parameter optimization engine
├── backtest/
│   └── backtester.py           # Backtesting framework
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── tests/
│   ├── test_indicators.py
│   └── test_risk.py
└── main.py                     # Main orchestrator
```

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd trader_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required environment variables:

```bash
# Binance
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_MODE=testnet  # or live

# MT5 Bridge
MT5_BRIDGE_HOST=localhost
MT5_BRIDGE_PORT=9090
MT5_ACCOUNT=your_mt5_account
MT5_PASSWORD=your_password
MT5_SERVER=MetaQuotes-Demo

# General
ENVIRONMENT=demo  # or live
DATABASE_PATH=data/trading.db
```

### 3. Configure Strategy

Edit `config/config.yaml` to adjust:
- Risk parameters
- Indicator settings
- Entry logic
- Timeframes
- Learning engine parameters

## 🚀 Usage

### Running the System

#### Demo Mode (Paper Trading)
```bash
# Set demo mode in .env
ENVIRONMENT=demo

# Start main system
python main.py
```

#### Live Mode
```bash
# Set live mode in .env
ENVIRONMENT=live

# Start main system (ensure you understand risks!)
python main.py
```

### Dashboard

Launch the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

Access at: http://localhost:8501

Dashboard features:
- Real-time position monitoring
- Trade history and analytics
- Performance metrics
- Configuration management
- Learning engine controls
- Manual overrides

### Backtesting

Run backtest on historical data:

```python
python backtest/backtester.py
```

Or use programmatically:

```python
from backtest.backtester import Backtester
from strategy.engine import StrategyEngine
from risk_management.money_manager import MoneyManager
from risk_management.stop_manager import StopManager

# Initialize components
config = load_config('config/config.yaml')
strategy_engine = StrategyEngine(config)
money_manager = MoneyManager(config)
stop_manager = StopManager(config)
backtester = Backtester(config)

# Run backtest
results = backtester.run(
    strategy_engine,
    money_manager,
    stop_manager,
    multi_tf_data,
    symbol='BTC/USDT',
    initial_balance=10000
)

print(results)
```

## 🔧 Configuration

### Risk Management

Key parameters in `config/config.yaml`:

```yaml
risk_management:
  max_risk_percent_per_trade: 1.0
  
  stop_loss:
    method: conservative  # atr, structure, or conservative
    atr_multiplier: 2.0
    
  trailing_stop:
    enabled: true
    activation_rr: 1.0  # Activate at 1:1 R:R
    method: atr
    atr_multiplier: 1.5
    
    breakeven:
      enabled: true
      trigger_rr: 1.0
      buffer_pips: 1
      
  take_profit:
    targets:
      - {name: TP1, rr_ratio: 1.5, close_percent: 50}
      - {name: TP2, rr_ratio: 3.0, close_percent: 30}
      
  global_limits:
    daily_max_drawdown_percent: 5.0
    max_concurrent_trades: 3
    max_trades_per_day: 10
```

### Strategy Parameters

```yaml
strategy:
  entry_types:
    - breakout_retest
    - pullback_to_sr
    - momentum_breakout
    
  confluence_required: 2
  
  filters:
    respect_higher_tf_bias: true
    min_atr_threshold: 0.0005
    max_atr_threshold: 0.05
```

### Learning Engine

```yaml
learning:
  enabled: true
  auto_apply: false  # Require manual approval
  
  optimization:
    method: grid_search
    param_ranges:
      atr_multiplier: [1.5, 2.0, 2.5, 3.0]
      rsi_oversold: [25, 30, 35]
      confluence_required: [2, 3]
      
  guardrails:
    min_sample_size: 50
    min_expectancy: 0.5
    max_drawdown_threshold: 20.0
```

## 📊 Performance Metrics

The system tracks comprehensive metrics:

- **Win Rate**: Percentage of winning trades
- **Avg R:R**: Average risk-to-reward ratio
- **Expectancy**: Average P&L per trade
- **Profit Factor**: Gross profit / gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Max Consecutive Wins/Losses**
- **Average Trade Duration**

## 🧠 Learning Engine

The learning engine automatically optimizes strategy parameters:

### Optimization Methods

1. **Grid Search**: Test all parameter combinations
2. **Random Search**: Sample random parameter sets
3. **RL Bandit**: Multi-armed bandit approach

### Safety Features

- Minimum sample size requirement
- Positive expectation check
- Maximum drawdown threshold
- Parameter versioning with rollback
- Manual approval for parameter changes

### Running Optimization

Via dashboard or programmatically:

```python
from learning.learner import StrategyLearner

learner = StrategyLearner(db, config)

result = learner.run_learning_cycle(
    backtest_function,
    historical_data,
    days_lookback=30
)

print(f"Best parameters: {result['best_params']}")
print(f"Improvement: {result['improvement_percent']:.2f}%")
```

## 🔌 MT5 Bridge Setup

The system communicates with MT5 via a socket-based EA bridge:

### EA Bridge Requirements

1. Install custom EA on MT5 that:
   - Opens socket server on specified port
   - Accepts JSON commands
   - Returns order confirmations
   - Streams live price data

2. Configure bridge in `.env`:
```bash
MT5_BRIDGE_HOST=localhost
MT5_BRIDGE_PORT=9090
```

### Command Protocol

```json
{
  "action": "place_order",
  "symbol": "XAUUSD",
  "order_type": "ORDER_TYPE_BUY",
  "volume": 0.1,
  "sl": 2040.0,
  "tp": 2060.0
}
```

## 📈 Binance Integration

### Testnet Setup

1. Create testnet account: https://testnet.binance.vision
2. Generate API keys
3. Configure in `.env`:

```bash
BINANCE_MODE=testnet
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
```

### Live Trading

```bash
BINANCE_MODE=live
BINANCE_API_KEY=your_live_key
BINANCE_API_SECRET=your_live_secret
```

**⚠️ Warning**: Use extreme caution with live keys. Start with small amounts.

## 🧪 Testing

Run unit tests:

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_indicators.py

# With coverage
pytest --cov=. tests/
```

## 📝 Logging

All decisions and trades are logged to SQLite:

- **Analysis Logs**: Every market analysis with indicator states
- **Trade Logs**: Full lifecycle from entry to exit
- **Order Events**: Placements, fills, modifications
- **SL/TP Adjustments**: All stop/target changes
- **System Events**: Errors, warnings, manual interventions

### Accessing Logs

```python
from logger.db import DatabaseManager

db = DatabaseManager("data/trading.db")
db.connect()

# Get recent trades
trades = db.get_trades(limit=50)

# Get statistics
stats = db.get_trade_statistics(days=30)

# Export to CSV
db.export_to_csv('trades', 'exports/trades.csv')
```

## 🚨 Safety Features

- **Emergency Shutdown**: Closes all positions on critical errors
- **Daily Drawdown Limit**: Stops trading at threshold
- **Max Concurrent Trades**: Prevents overexposure
- **Cooldown After Losses**: Pauses after consecutive losses
- **Demo Mode**: Full simulation without real execution
- **Manual Override**: Dashboard controls for immediate action

## 🔄 Workflow

### Typical Trading Day

1. **System Start**: `python main.py`
2. **Monitor Dashboard**: Track positions and metrics
3. **Analysis Loop**: System analyzes markets every minute
4. **Entry Signals**: Executes trades meeting criteria
5. **Position Management**: Updates stops, takes profits
6. **End of Day**: Generates summary, saves logs
7. **Learning Cycle**: Optimizes parameters (scheduled)

### Manual Intervention

Use dashboard to:
- Pause/resume trading
- Close individual positions
- Emergency close all
- Adjust risk parameters
- Force learning run

## 📊 Performance Optimization

### Tips for Better Results

1. **Start Conservative**: Use 0.5% risk per trade initially
2. **Test Thoroughly**: Run backtests on 6+ months of data
3. **Monitor Drawdown**: Act before hitting limits
4. **Review Logs**: Analyze what works and what doesn't
5. **Let Learning Run**: Give system time to optimize (50+ trades)
6. **Stay Disciplined**: Trust the system, avoid emotional overrides

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## ⚠️ Disclaimer

**Trading financial instruments involves substantial risk of loss. This system is provided for educational purposes. Past performance does not guarantee future results. Always:**

- Start with demo mode
- Test thoroughly before live trading
- Never risk more than you can afford to lose
- Understand all system components
- Monitor positions actively
- Keep API keys secure

## 📄 License

[Your License Here]

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check documentation in `/docs`
- Review code comments

## 🎯 Roadmap

Future enhancements:

- [ ] Sentiment analysis integration
- [ ] Order book depth analysis  
- [ ] Multi-asset portfolio optimization
- [ ] GPU-accelerated RLstrategy
- [ ] Mobile app for monitoring
- [ ] Advanced chart pattern recognition
- [ ] News event filtering
- [ ] Social trading integration

---

**Built with ❤️ for algorithmic traders**