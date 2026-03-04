"""
Unit tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from indicators.indicators import TechnicalIndicators


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    
    # Generate realistic price data
    price = 100
    prices = []
    for _ in range(200):
        change = np.random.normal(0, 1)
        price = max(price + change, 50)  # Prevent too low
        prices.append(price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Ensure high is highest, low is lowest
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def test_indicator_initialization():
    """Test indicator initialization with config."""
    config = {
        'ema': {'periods': [20, 50]},
        'rsi': {'period': 14}
    }
    
    indicators = TechnicalIndicators(config)
    assert indicators.config is not None
    assert indicators.config['ema']['periods'] == [20, 50]


def test_ema_calculation(sample_data):
    """Test EMA calculation."""
    indicators = TechnicalIndicators()
    emas = indicators.calculate_ema(sample_data)
    
    # Check that EMAs are calculated
    assert 20 in emas
    assert 50 in emas
    assert 200 in emas
    
    # Check values are Series
    assert isinstance(emas[20], pd.Series)
    
    # Check EMA values are reasonable
    assert len(emas[20]) == len(sample_data)
    assert not emas[20].isna().all()
    
    # EMA should be smoothed (less volatile than price)
    price_std = sample_data['close'].std()
    ema_std = emas[20].dropna().std()
    assert ema_std < price_std


def test_rsi_calculation(sample_data):
    """Test RSI calculation."""
    indicators = TechnicalIndicators()
    rsi = indicators.calculate_rsi(sample_data)
    
    # Check structure
    assert 'value' in rsi
    assert 'overbought' in rsi
    assert 'oversold' in rsi
    assert 'is_overbought' in rsi
    assert 'is_oversold' in rsi
    
    # Check RSI bounds (0-100)
    rsi_values = rsi['value'].dropna()
    assert (rsi_values >= 0).all()
    assert (rsi_values <= 100).all()
    
    # Check overbought/oversold flags are boolean
    assert isinstance(rsi['is_overbought'], (bool, np.bool_))
    assert isinstance(rsi['is_oversold'], (bool, np.bool_))


def test_macd_calculation(sample_data):
    """Test MACD calculation."""
    indicators = TechnicalIndicators()
    macd = indicators.calculate_macd(sample_data)
    
    # Check structure
    assert 'macd' in macd
    assert 'signal' in macd
    assert 'histogram' in macd
    
    # Check all are Series
    assert isinstance(macd['macd'], pd.Series)
    assert isinstance(macd['signal'], pd.Series)
    assert isinstance(macd['histogram'], pd.Series)
    
    # Check histogram = macd - signal
    hist_check = macd['macd'] - macd['signal']
    assert np.allclose(
        macd['histogram'].dropna(),
        hist_check.dropna(),
        rtol=1e-5
    )


def test_atr_calculation(sample_data):
    """Test ATR calculation."""
    indicators = TechnicalIndicators()
    atr = indicators.calculate_atr(sample_data)
    
    # Check structure
    assert 'value' in atr
    assert 'current' in atr
    assert 'percent_of_price' in atr
    
    # ATR should be positive
    assert (atr['value'].dropna() > 0).all()
    assert atr['current'] > 0
    
    # Percent should be reasonable (0-10%)
    assert 0 < atr['percent_of_price'] < 10


def test_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    indicators = TechnicalIndicators()
    bb = indicators.calculate_bollinger_bands(sample_data)
    
    # Check structure
    assert 'upper' in bb
    assert 'middle' in bb
    assert 'lower' in bb
    assert 'bandwidth' in bb
    assert 'squeeze' in bb
    
    # Upper > Middle > Lower
    valid_data = ~(bb['upper'].isna() | bb['middle'].isna() | bb['lower'].isna())
    assert (bb['upper'][valid_data] >= bb['middle'][valid_data]).all()
    assert (bb['middle'][valid_data] >= bb['lower'][valid_data]).all()
    
    # Squeeze is boolean
    assert isinstance(bb['squeeze'], (bool, np.bool_))


def test_supertrend(sample_data):
    """Test SuperTrend calculation."""
    indicators = TechnicalIndicators()
    st = indicators.calculate_supertrend(sample_data)
    
    # Check structure
    assert 'value' in st
    assert 'direction' in st
    assert 'trend' in st
    
    # Check trend values
    assert st['trend'] in ['bullish', 'bearish']
    
    # Direction should be 1 or -1
    directions = st['direction'].dropna().unique()
    assert set(directions).issubset({1, -1})


def test_stochastic_oscillator(sample_data):
    """Test Stochastic Oscillator."""
    indicators = TechnicalIndicators()
    stoch = indicators.calculate_stochastic(sample_data)
    
    # Check structure
    assert 'k' in stoch
    assert 'd' in stoch
    assert 'is_overbought' in stoch
    assert 'is_oversold' in stoch
    
    # K and D should be 0-100
    k_values = stoch['k'].dropna()
    d_values = stoch['d'].dropna()
    
    assert (k_values >= 0).all()
    assert (k_values <= 100).all()
    assert (d_values >= 0).all()
    assert (d_values <= 100).all()


def test_adx_calculation(sample_data):
    """Test ADX calculation."""
    indicators = TechnicalIndicators()
    adx = indicators.calculate_adx(sample_data)
    
    # Check structure
    assert 'value' in adx
    assert 'plus_di' in adx
    assert 'minus_di' in adx
    assert 'trend_strength' in adx
    assert 'direction' in adx
    
    # ADX values should be positive
    assert (adx['value'].dropna() >= 0).all()
    
    # Trend strength should be 'strong' or 'weak'
    assert adx['trend_strength'] in ['strong', 'weak']
    
    # Direction should be 'bullish' or 'bearish'
    assert adx['direction'] in ['bullish', 'bearish']


def test_price_structure_detection(sample_data):
    """Test price structure detection."""
    indicators = TechnicalIndicators()
    structure = indicators.detect_price_structure(sample_data)
    
    # Check structure
    assert 'structure' in structure
    assert 'swing_highs' in structure
    assert 'swing_lows' in structure
    
    # Structure should be valid type
    assert structure['structure'] in ['uptrend', 'downtrend', 'neutral']
    
    # Swing points should be lists
    assert isinstance(structure['swing_highs'], list)
    assert isinstance(structure['swing_lows'], list)


def test_candle_patterns(sample_data):
    """Test candle pattern detection."""
    indicators = TechnicalIndicators()
    patterns = indicators.detect_candle_patterns(sample_data)
    
    # Check all expected patterns
    expected_patterns = [
        'bullish_engulfing',
        'bearish_engulfing',
        'hammer',
        'shooting_star',
        'doji',
        'pin_bar'
    ]
    
    for pattern in expected_patterns:
        assert pattern in patterns
        assert isinstance(patterns[pattern], bool)


def test_fibonacci_levels(sample_data):
    """Test Fibonacci retracement calculation."""
    indicators = TechnicalIndicators()
    fib = indicators.calculate_fibonacci_levels(sample_data)
    
    # Check structure
    assert 'levels' in fib
    assert 'swing_high' in fib
    assert 'swing_low' in fib
    assert 'nearest_level' in fib
    
    # Check Fibonacci levels exist
    expected_levels = ['0.0', '0.236', '0.382', '0.5', '0.618', '0.786', '1.0']
    for level in expected_levels:
        assert level in fib['levels']
    
    # Swing high should be > swing low
    assert fib['swing_high'] > fib['swing_low']
    
    # Levels should be in descending order
    levels = [fib['levels'][k] for k in expected_levels]
    assert levels == sorted(levels, reverse=True)


def test_calculate_all_indicators(sample_data):
    """Test calculating all indicators at once."""
    indicators = TechnicalIndicators()
    results = indicators.calculate_all(sample_data)
    
    # Check all indicator groups are present
    expected_groups = [
        'ema', 'sma', 'macd', 'rsi', 'atr',
        'bollinger', 'supertrend', 'parabolic_sar',
        'stochastic', 'adx', 'vwap', 'obv',
        'price_structure', 'candle_patterns', 'fibonacci'
    ]
    
    for group in expected_groups:
        assert group in results, f"Missing indicator group: {group}"
    
    # Spot check some values
    assert 20 in results['ema']
    assert 'value' in results['rsi']
    assert 'macd' in results['macd']


def test_indicator_edge_cases():
    """Test indicators with edge case data."""
    # Very small dataset
    small_data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    })
    
    indicators = TechnicalIndicators()
    
    # Should handle gracefully without crashing
    try:
        emas = indicators.calculate_ema(small_data)
        assert emas is not None
    except Exception as e:
        pytest.fail(f"Indicator calculation failed on small data: {e}")


def test_indicator_consistency():
    """Test that indicators produce consistent results."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    indicators = TechnicalIndicators()
    
    # Calculate twice
    result1 = indicators.calculate_all(data)
    result2 = indicators.calculate_all(data)
    
    # Should be identical
    assert np.allclose(
        result1['ema'][20].dropna(),
        result2['ema'][20].dropna()
    )
    
    assert np.allclose(
        result1['rsi']['value'].dropna(),
        result2['rsi']['value'].dropna()
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])