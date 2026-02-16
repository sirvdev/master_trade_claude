"""
Complete working backtest example.
Generates sufficient data and runs a full backtest.

Run this directly: python backtest_complete_example.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set backtest mode environment variable
os.environ['BACKTEST_MODE'] = 'true'

print("="*70)
print(" COMPLETE BACKTEST EXAMPLE")
print("="*70)

# Step 1: Generate sufficient test data
print("\n[1/5] Generating test data...")

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# Generate 7 days of 1-minute data (that's 10,080 bars - plenty for resampling)
dates = pd.date_range(start='2024-01-01', periods=10080, freq='1min')
np.random.seed(42)

print(f"  - Creating 10,080 bars (7 days) of 1-minute data...")

# Create realistic price movement
price = 2000.0
prices = []
for i in range(10080):
    # Add some trend and volatility
    trend = 0.001 if i % 1440 < 720 else -0.001  # Trend changes
    change = np.random.normal(trend, 0.3)
    price += change
    price = max(price, 1500)  # Floor
    prices.append(price)

# Create OHLCV data
data = []
for i in range(len(dates)):
    close = prices[i]
    open_price = prices[i-1] if i > 0 else close
    
    # Generate realistic high/low
    range_size = abs(np.random.normal(0, 0.5))
    high = max(open_price, close) + abs(np.random.uniform(0, range_size))
    low = min(open_price, close) - abs(np.random.uniform(0, range_size))
    
    volume = int(1000 * np.random.uniform(0.5, 2.0))
    
    data.append({
        'timestamp': dates[i],
        'open': round(open_price, 2),
        'high': round(high, 2),
        'low': round(low, 2),
        'close': round(close, 2),
        'volume': volume
    })

df_1m = pd.DataFrame(data)
csv_path = 'data/XAUUSD_1m.csv'
df_1m.to_csv(csv_path, index=False)

print(f"  ✓ Saved to {csv_path}")
print(f"  ✓ Date range: {df_1m['timestamp'].iloc[0]} to {df_1m['timestamp'].iloc[-1]}")
print(f"  ✓ Price range: ${df_1m['low'].min():.2f} - ${df_1m['high'].max():.2f}")

# Step 2: Load and resample data
print("\n[2/5] Loading and resampling data...")

df_1m = pd.read_csv(csv_path)
df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
df_1m.set_index('timestamp', inplace=True)

# Create timeframes
timeframes = {
    '1h': df_1m.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(),
    
    '15m': df_1m.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(),
    
    '5m': df_1m.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
}

for tf, df in timeframes.items():
    print(f"  ✓ {tf}: {len(df)} bars")

# Verify we have enough data
min_required = 30  # Reduced requirement for backtest mode
all_good = all(len(df) >= min_required for df in timeframes.values())

if not all_good:
    print("\n  ✗ ERROR: Insufficient data after resampling")
    for tf, df in timeframes.items():
        status = "✓" if len(df) >= min_required else "✗"
        print(f"    {status} {tf}: {len(df)} bars (need {min_required})")
    sys.exit(1)

print("  ✓ All timeframes have sufficient data")

# Step 3: Initialize components
print("\n[3/5] Initializing trading components...")

try:
    from indicators.indicators import TechnicalIndicators
    from strategy.engine import StrategyEngine
    from risk_management.money_manager import MoneyManager
    from risk_management.stop_manager import StopManager
    from backtest.backtester import Backtester
    
    config = {
        'indicators': TechnicalIndicators._default_config(),
        'strategy': {
            'entry_types': ['breakout_retest'],
            'confluence_required': 2,
            'filters': {}
        },
        'timeframes': {
            'structure_timeframe': '1h',
            'entry_timeframe': '5m'
        },
        'risk_management': {
            'max_risk_percent_per_trade': 1.0,
            'stop_loss': {
                'method': 'conservative',
                'atr_multiplier': 2.0
            },
            'global_limits': {
                'max_concurrent_trades': 3
            }
        },
        'backtest': {
            'simulation': {
                'slippage_percent': 0.05,
                'commission_percent': 0.1,
                'latency_bars': 1
            }
        }
    }
    
    strategy_engine = StrategyEngine(config)
    money_manager = MoneyManager(config)
    stop_manager = StopManager(config)
    backtester = Backtester(config)
    
    print("  ✓ All components initialized")
    
except ImportError as e:
    print(f"\n  ✗ ERROR: Missing required module")
    print(f"    {e}")
    print("\n  Make sure you're running this from the project root directory:")
    print("    cd /path/to/trader_project")
    print("    python backtest_complete_example.py")
    sys.exit(1)

# Step 4: Run backtest
print("\n[4/5] Running backtest...")
print("-"*70)

try:
    results = backtester.run(
        strategy_engine,
        money_manager,
        stop_manager,
        timeframes,
        'XAUUSD',
        initial_balance=10000
    )
    
    print("-"*70)
    
except Exception as e:
    print(f"\n  ✗ Backtest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Display results
print("\n[5/5] Results Summary")
print("="*70)

if results['total_trades'] == 0:
    print("\n⚠ WARNING: No trades were executed!")
    print("\nPossible reasons:")
    print("  • Strategy is too restrictive (confluence_required too high)")
    print("  • Entry conditions never met in this data")
    print("  • All signals filtered out by risk management")
    print("\nTry:")
    print("  • Reduce confluence_required to 1")
    print("  • Generate more data (14+ days)")
    print("  • Check strategy configuration")
else:
    print(f"\n📊 PERFORMANCE METRICS")
    print("-"*70)
    print(f"  Total Trades:        {results['total_trades']}")
    print(f"  Winning Trades:      {results['winning_trades']}")
    print(f"  Losing Trades:       {results['losing_trades']}")
    print(f"  Win Rate:            {results['win_rate']:.2%}")
    print()
    print(f"  Total P&L:           ${results['total_pnl']:,.2f}")
    print(f"  Average Win:         ${results['avg_win']:,.2f}")
    print(f"  Average Loss:        ${results['avg_loss']:,.2f}")
    print(f"  Average R:R:         {results['avg_rr']:.2f}")
    print()
    print(f"  Profit Factor:       {results['profit_factor']:.2f}")
    print(f"  Expectancy:          ${results['expectancy']:.2f}")
    print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
    print()
    print(f"  Max Drawdown:        {results['max_drawdown']:.2f}%")
    print(f"  Max Consecutive Wins:    {results['max_consecutive_wins']}")
    print(f"  Max Consecutive Losses:  {results['max_consecutive_losses']}")
    print()
    print(f"  Final Balance:       ${results['final_balance']:,.2f}")
    print(f"  Total Return:        {results['return_percent']:.2f}%")
    print(f"  Avg Trade Duration:  {results['avg_duration_hours']:.1f} hours")
    
    # Export trades
    if results['total_trades'] > 0:
        trades_file = 'data/backtest_trades.csv'
        backtester.export_trades(trades_file)
        print(f"\n  ✓ Trades exported to: {trades_file}")

print("\n" + "="*70)
print("✓ BACKTEST COMPLETE")
print("="*70)

# Next steps
if results['total_trades'] > 0:
    print("\n📈 NEXT STEPS:")
    print("  1. Review individual trades in data/backtest_trades.csv")
    print("  2. Adjust strategy parameters in config/config.yaml")
    print("  3. Run more backtests with different timeframes")
    print("  4. Use learning engine to optimize parameters")
    print("  5. Paper trade before going live!")
else:
    print("\n🔧 TROUBLESHOOTING:")
    print("  1. Check strategy configuration - may be too restrictive")
    print("  2. Try generating more data (14+ days)")
    print("  3. Review logs to see why signals aren't triggering")
    print("  4. Reduce confluence_required for testing")