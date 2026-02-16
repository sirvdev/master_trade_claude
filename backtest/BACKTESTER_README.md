# Backtester - Fixed Version

## Overview

The backtester now properly loads real market data from CSV files and automatically resamples it to the configured timeframes (4H, 1H, 15m, etc.). This allows you to test your strategy on actual historical market data.

## Quick Start

### 1. Prepare Your Data

You have two options:

#### Option A: Use Your Own Data

Create a CSV file with 1-minute (or any base timeframe) OHLCV data:

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,2000.50,2001.00,2000.00,2000.75,1234
2024-01-01 00:01:00,2000.75,2001.50,2000.50,2001.00,1567
2024-01-01 00:02:00,2001.00,2001.25,2000.75,2001.10,1890
...
```

**Requirements:**
- Must have columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Timestamp must be parseable (YYYY-MM-DD HH:MM:SS format recommended)
- Data should be sorted chronologically

#### Option B: Generate Sample Data

Use the provided data generator:

```bash
python generate_sample_data.py
```

This creates sample data files in the `data/` directory:
- `XAU_USD_1m.csv` - Gold 1-minute data (30 days)
- `XAU_USD_5m.csv` - Gold 5-minute data (90 days)  
- `BTC_USDT_1m.csv` - Bitcoin 1-minute data (30 days)
- `BTC_USDT_15m.csv` - Bitcoin 15-minute data (180 days)

### 2. Run the Backtester

Replace the old `backtest/backtester.py` with the new version, then:

```bash
python -m backtest.backtester
```

Or use it programmatically:

```python
from backtest.backtester import Backtester
from strategy.engine import StrategyEngine
from risk_management.money_manager import MoneyManager
from risk_management.stop_manager import StopManager

# Initialize components
backtester = Backtester(config)
strategy_engine = StrategyEngine(config)
money_manager = MoneyManager(config)
stop_manager = StopManager(config)

# Load and prepare data
multi_tf_data = backtester.prepare_multi_timeframe_data(
    csv_path='data/XAU_USD_1m.csv',
    timeframes=['4h', '1h', '15m'],  # Configure as needed
    base_timeframe='1m'  # Timeframe of your CSV
)

# Run backtest
results = backtester.run(
    strategy_engine,
    money_manager,
    stop_manager,
    multi_tf_data,
    'XAU/USD',
    initial_balance=10000
)

print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Total Return: {results['return_percent']:.2f}%")
```

## How It Works

### Data Resampling

The backtester automatically resamples your base data to higher timeframes:

**Example:**
- CSV contains: 1-minute bars
- You configure: `['4h', '1h', '15m']`
- Backtester creates:
  - **4H bars**: Uses 240 x 1m bars (4 hours = 240 minutes)
  - **1H bars**: Uses 60 x 1m bars
  - **15m bars**: Uses 15 x 1m bars

**Resampling Rules:**
- `open`: First bar's open in period
- `high`: Maximum high in period
- `low`: Minimum low in period
- `close`: Last bar's close in period
- `volume`: Sum of volume in period

### Walk-Forward Testing

The backtester simulates real-time trading:

1. At each bar, it provides data **up to that point** to the strategy
2. Strategy analyzes using configured timeframes (e.g., 4H, 1H, 15m)
3. If entry signal, it places a simulated order with:
   - Realistic slippage
   - Commission costs
   - Latency (configurable bar delay)
4. Manages open positions:
   - Checks stop loss hits
   - Checks take profit hits
   - Updates trailing stops
5. Logs all trades and generates performance metrics

## Configuration

Edit `config/config.yaml`:

```yaml
backtest:
  simulation:
    slippage_percent: 0.05  # 0.05% slippage
    commission_percent: 0.1  # 0.1% commission per trade
    latency_bars: 1  # 1 bar delay for execution
```

## Output

The backtester provides:

### Console Output
```
=== Backtest Complete: XAU/USD ===
Total Trades: 45
Win Rate: 62.22%
Final Balance: $12,340.50
Return: 23.41%
Max Drawdown: -5.67%
Profit Factor: 2.15
Sharpe Ratio: 1.82
```

### Detailed Results

```python
results = {
    'total_trades': 45,
    'winning_trades': 28,
    'losing_trades': 17,
    'win_rate': 0.6222,
    'total_pnl': 2340.50,
    'avg_win': 125.30,
    'avg_loss': -67.80,
    'avg_rr': 1.85,
    'profit_factor': 2.15,
    'expectancy': 52.01,
    'sharpe_ratio': 1.82,
    'max_drawdown': -5.67,
    'max_consecutive_wins': 7,
    'max_consecutive_losses': 4,
    'avg_duration_hours': 2.3,
    'final_balance': 12340.50,
    'return_percent': 23.41,
    'trades': [...],  # List of all trades
    'equity_curve': [...]  # Balance over time
}
```

### Export Trades

```python
backtester.export_trades('results/backtest_trades.csv')
```

Creates CSV with:
- trade_id, symbol, direction
- entry_time, entry_price
- exit_time, exit_price
- stop_loss, pnl, realized_rr
- exit_reason (stop_loss, take_profit, etc.)

## Supported Timeframes

Base timeframe can be any of:
- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day

You can resample to any higher timeframe. For example:
- Base: `1m` → Can create: `5m`, `15m`, `1h`, `4h`, `1d`
- Base: `5m` → Can create: `15m`, `30m`, `1h`, `4h`, `1d`
- Cannot resample to lower timeframes (e.g., `1h` → `5m` won't work)

## Tips for Good Backtests

1. **Use enough data**: Minimum 30 days recommended, ideally 90-180 days
2. **Include various market conditions**: Trending, ranging, volatile
3. **Don't overfit**: If win rate is >80%, you may be overfitting
4. **Check drawdown**: Max DD >20% is risky for live trading
5. **Verify trade count**: Need 30+ trades for statistical significance
6. **Test multiple symbols**: Don't optimize for just one market

## Common Issues

### "Insufficient data for XAU/USD @ 1H"

**Cause**: Not enough base data to create higher timeframe bars

**Solution**: 
- For 1H bars, need at least 100 hours of 1m data (~4 days)
- For 4H bars, need at least 400 hours of 1m data (~17 days)
- Use more data or test on lower timeframes

### FutureWarning about 'H'

**Fixed** in the new version - now uses 'h' instead of 'H'

### "No trades executed"

**Possible causes**:
- Strategy is too restrictive (too many filters)
- Not enough confluence signals
- Risk limits blocking trades

**Solutions**:
- Check strategy configuration
- Review logs to see why signals are filtered
- Reduce `confluence_required` for testing

## Integration with Learning Engine

Backtest results feed directly into the learning engine:

```python
from learning.learner import StrategyLearner

learner = StrategyLearner(db, config)

# Calculate metrics from backtest trades
metrics = learner.calculate_performance_metrics(
    results['trades'],
    initial_balance=10000
)

print(f"Expectancy: ${metrics['expectancy']:.2f}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
```

## Next Steps

1. **Run initial backtest** with default parameters
2. **Analyze results** - what worked, what didn't?
3. **Adjust parameters** in `config/config.yaml`
4. **Re-run backtest** to validate improvements
5. **Use learning engine** to find optimal parameters
6. **Paper trade** before going live

---

**Remember**: Past performance does not guarantee future results. Always start with small position sizes when going live!
