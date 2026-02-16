"""
Generate sample OHLCV data for backtesting.
Creates realistic price movement with trends and volatility.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_realistic_ohlcv(
    start_date: str = '2024-01-01',
    days: int = 30,
    timeframe: str = '1m',
    initial_price: float = 2000.0,
    volatility: float = 0.5,
    trend: float = 0.0,
    symbol: str = 'XAU/USD'
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data with trends and patterns.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        days: Number of days to generate
        timeframe: Timeframe ('1m', '5m', '15m', '1h')
        initial_price: Starting price
        volatility: Price volatility (standard deviation)
        trend: Upward/downward trend bias (-1 to 1)
        symbol: Symbol name for filename
        
    Returns:
        DataFrame with OHLCV data
    """
    # Calculate number of bars
    bars_per_day = {
        '1m': 1440,
        '5m': 288,
        '15m': 96,
        '30m': 48,
        '1h': 24,
        '4h': 6,
        '1d': 1
    }
    
    if timeframe not in bars_per_day:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    total_bars = bars_per_day[timeframe] * days
    
    # Generate timestamps
    freq_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '4h': '4h',
        '1d': '1D'
    }
    
    timestamps = pd.date_range(
        start=start_date,
        periods=total_bars,
        freq=freq_map[timeframe]
    )
    
    # Generate price movement
    np.random.seed(42)
    
    # Create realistic price path
    price = initial_price
    prices = []
    
    for i in range(total_bars):
        # Add trend
        drift = trend * volatility / 100
        
        # Add random walk
        change = np.random.normal(drift, volatility)
        
        # Add occasional large moves (fat tails)
        if np.random.random() < 0.05:
            change *= np.random.choice([2, -2, 3, -3])
        
        # Update price
        price += change
        price = max(price, initial_price * 0.5)  # Prevent unrealistic drops
        prices.append(price)
    
    # Generate OHLC from close prices
    data = []
    
    for i, close in enumerate(prices):
        # Generate realistic OHLC
        range_size = abs(np.random.normal(0, volatility * 0.5))
        
        high = close + abs(np.random.uniform(0, range_size))
        low = close - abs(np.random.uniform(0, range_size))
        
        # Open is between previous close and current close
        if i == 0:
            open_price = close
        else:
            prev_close = prices[i-1]
            open_price = prev_close + np.random.uniform(-volatility/2, volatility/2)
        
        # Ensure OHLC validity
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate realistic volume
        base_volume = 1000
        volume = int(base_volume * np.random.uniform(0.5, 2.0))
        
        data.append({
            'timestamp': timestamps[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    print(f"Generated {len(df)} bars of {timeframe} data")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    return df


def main():
    """Generate sample data files."""
    print("=" * 60)
    print("Sample Data Generator for Backtesting")
    print("=" * 60)
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Generate 1-minute data for XAU/USD (Gold)
    print("\n1. Generating XAU/USD 1-minute data (30 days)...")
    df_gold_1m = generate_realistic_ohlcv(
        start_date='2024-01-01',
        days=30,
        timeframe='1m',
        initial_price=2000.0,
        volatility=0.5,
        trend=0.01,  # Slight upward bias
        symbol='XAU/USD'
    )
    
    filepath = 'data/XAU_USD_1m.csv'
    df_gold_1m.to_csv(filepath, index=False)
    print(f"✓ Saved to {filepath}\n")
    
    # Generate 5-minute data for XAU/USD (90 days)
    print("2. Generating XAU/USD 5-minute data (90 days)...")
    df_gold_5m = generate_realistic_ohlcv(
        start_date='2024-01-01',
        days=90,
        timeframe='5m',
        initial_price=2000.0,
        volatility=1.0,
        trend=0.02,
        symbol='XAU/USD'
    )
    
    filepath = 'data/XAU_USD_5m.csv'
    df_gold_5m.to_csv(filepath, index=False)
    print(f"✓ Saved to {filepath}\n")
    
    # Generate 1-minute data for BTC/USDT
    print("3. Generating BTC/USDT 1-minute data (30 days)...")
    df_btc_1m = generate_realistic_ohlcv(
        start_date='2024-01-01',
        days=30,
        timeframe='1m',
        initial_price=45000.0,
        volatility=50.0,
        trend=0.05,  # More volatile with upward trend
        symbol='BTC/USDT'
    )
    
    filepath = 'data/BTC_USDT_1m.csv'
    df_btc_1m.to_csv(filepath, index=False)
    print(f"✓ Saved to {filepath}\n")
    
    # Generate 15-minute data for BTC/USDT (180 days)
    print("4. Generating BTC/USDT 15-minute data (180 days)...")
    df_btc_15m = generate_realistic_ohlcv(
        start_date='2024-01-01',
        days=180,
        timeframe='15m',
        initial_price=45000.0,
        volatility=100.0,
        trend=0.03,
        symbol='BTC/USDT'
    )
    
    filepath = 'data/BTC_USDT_15m.csv'
    df_btc_15m.to_csv(filepath, index=False)
    print(f"✓ Saved to {filepath}\n")
    
    print("=" * 60)
    print("Sample Data Generation Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/XAU_USD_1m.csv (30 days, 1-minute bars)")
    print("  - data/XAU_USD_5m.csv (90 days, 5-minute bars)")
    print("  - data/BTC_USDT_1m.csv (30 days, 1-minute bars)")
    print("  - data/BTC_USDT_15m.csv (180 days, 15-minute bars)")
    print("\nYou can now run the backtester with these files!")


if __name__ == "__main__":
    main()
