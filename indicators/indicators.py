"""
Comprehensive technical indicators library.
All indicators are parameterized and exposed through a unified API.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pandas_ta_classic as ta


class TechnicalIndicators:
    """Unified API for all technical indicators."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize indicators with configuration.
        
        Args:
            config: Dictionary of indicator parameters from config.yaml
        """
        self.config = config or self._default_config()
        
    @staticmethod
    def _default_config() -> Dict:
        """Default indicator parameters."""
        return {
            'ema': {'periods': [20, 50, 200]},
            'sma': {'periods': [50, 100, 200]},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'atr': {'period': 14},
            'bollinger': {'period': 20, 'std_dev': 2},
            'supertrend': {'period': 10, 'multiplier': 3.0},
            'parabolic_sar': {'acceleration': 0.02, 'maximum': 0.2},
            'stochastic': {'k_period': 14, 'd_period': 3, 'smooth_k': 3},
            'adx': {'period': 14, 'trend_strength_min': 25},
            'fibonacci': {'lookback_bars': 50}
        }
        
    def calculate_all(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate all indicators for given OHLCV data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            
        Returns:
            Dictionary of all indicator values
        """
        indicators = {}
        
        # Moving Averages
        indicators['ema'] = self.calculate_ema(df)
        indicators['sma'] = self.calculate_sma(df)
        
        # Momentum
        indicators['macd'] = self.calculate_macd(df)
        indicators['rsi'] = self.calculate_rsi(df)
        indicators['stochastic'] = self.calculate_stochastic(df)
        
        # Volatility
        indicators['atr'] = self.calculate_atr(df)
        indicators['bollinger'] = self.calculate_bollinger_bands(df)
        
        # Trend
        indicators['supertrend'] = self.calculate_supertrend(df)
        indicators['parabolic_sar'] = self.calculate_parabolic_sar(df)
        indicators['adx'] = self.calculate_adx(df)
        
        # Volume
        indicators['vwap'] = self.calculate_vwap(df)
        indicators['obv'] = self.calculate_obv(df)
        
        # Price Action
        indicators['price_structure'] = self.detect_price_structure(df)
        indicators['candle_patterns'] = self.detect_candle_patterns(df)
        indicators['fibonacci'] = self.calculate_fibonacci_levels(df)
        
        return indicators
        
    def calculate_ema(self, df: pd.DataFrame) -> Dict[int, pd.Series]:
        """Calculate Exponential Moving Averages."""
        periods = self.config['ema']['periods']
        emas = {}
        for period in periods:
            emas[period] = df['close'].ewm(span=period, adjust=False).mean()
        return emas
        
    def calculate_sma(self, df: pd.DataFrame) -> Dict[int, pd.Series]:
        """Calculate Simple Moving Averages."""
        periods = self.config['sma']['periods']
        smas = {}
        for period in periods:
            smas[period] = df['close'].rolling(window=period).mean()
        return smas
        
    def calculate_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        fast = self.config['macd'].get('fast', 12)  # ✅ Correct
        slow = self.config['macd'].get('slow', 26)  # ✅ Correct
        signal = self.config['macd'].get('signal', 9)  # ✅ Correct
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
        
    def calculate_rsi(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate RSI indicator."""
        period = self.config['rsi']['period']
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'value': rsi,
            'overbought': self.config['rsi']['overbought'],
            'oversold': self.config['rsi']['oversold'],
            'is_overbought': rsi.iloc[-1] > self.config['rsi']['overbought'],
            'is_oversold': rsi.iloc[-1] < self.config['rsi']['oversold']
        }
        
    def calculate_atr(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate Average True Range."""
        period = self.config['atr']['period']
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return {
            'value': atr,
            'current': atr.iloc[-1],
            'percent_of_price': (atr.iloc[-1] / df['close'].iloc[-1]) * 100
        }
        
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        period = self.config.get('bollinger', {}).get('period', 20)  # ✅ Correct
        std_dev = self.config.get('bollinger', {}).get('std_dev', 2) # ✅ Correct
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bandwidth = ((upper_band - lower_band) / sma) * 100
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'bandwidth': bandwidth,
            'squeeze': bandwidth.iloc[-1] < bandwidth.rolling(100).quantile(0.2).iloc[-1]
        }
        
    def calculate_supertrend(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate SuperTrend indicator."""
        period = self.config['supertrend']['period']
        multiplier = self.config['supertrend']['multiplier']
        
        atr = self.calculate_atr(df)['value']
        hl_avg = (df['high'] + df['low']) / 2
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = lower_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                
        return {
            'value': supertrend,
            'direction': direction,
            'trend': 'bullish' if direction.iloc[-1] == 1 else 'bearish'
        }
        
    def calculate_parabolic_sar(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Parabolic SAR."""
        accel = self.config['parabolic_sar']['acceleration']
        max_accel = self.config['parabolic_sar']['maximum']
        
        sar = pd.Series(index=df.index, dtype=float)
        ep = pd.Series(index=df.index, dtype=float)
        af = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        # Initialize
        sar.iloc[0] = df['low'].iloc[0]
        ep.iloc[0] = df['high'].iloc[0]
        af.iloc[0] = accel
        trend.iloc[0] = 1
        
        for i in range(1, len(df)):
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
            
            if trend.iloc[i-1] == 1:  # Uptrend
                if df['low'].iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = df['low'].iloc[i]
                    af.iloc[i] = accel
                else:
                    trend.iloc[i] = 1
                    if df['high'].iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = df['high'].iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + accel, max_accel)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                if df['high'].iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = df['high'].iloc[i]
                    af.iloc[i] = accel
                else:
                    trend.iloc[i] = -1
                    if df['low'].iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = df['low'].iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + accel, max_accel)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
                        
        return {
            'value': sar,
            'trend': trend,
            'current_trend': 'bullish' if trend.iloc[-1] == 1 else 'bearish'
        }
        
    def calculate_stochastic(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate Stochastic Oscillator."""
        k_period = self.config['stochastic']['k_period']
        d_period = self.config['stochastic']['d_period']
        smooth_k = self.config['stochastic']['smooth_k']
        
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k_raw = 100 * (df['close'] - low_min) / (high_max - low_min)
        k = k_raw.rolling(window=smooth_k).mean()
        d = k.rolling(window=d_period).mean()
        
        return {
            'k': k,
            'd': d,
            'is_overbought': k.iloc[-1] > 80,
            'is_oversold': k.iloc[-1] < 20
        }
        
    def calculate_adx(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate Average Directional Index."""
        period = self.config['adx']['period']
        
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self.calculate_atr(df)['value']
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        trend_strength = 'strong' if adx.iloc[-1] > self.config['adx']['trend_strength_min'] else 'weak'
        
        return {
            'value': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'trend_strength': trend_strength,
            'direction': 'bullish' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'bearish'
        }
        
    def calculate_vwap(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        return {
            'value': vwap,
            'current': vwap.iloc[-1],
            'price_vs_vwap': 'above' if df['close'].iloc[-1] > vwap.iloc[-1] else 'below'
        }
        
    def calculate_obv(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        obv_ema = obv.ewm(span=20).mean()
        
        return {
            'value': obv,
            'ema': obv_ema,
            'divergence': self._detect_divergence(df['close'], obv)
        }
        
    def detect_price_structure(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, any]:
        """
        Detect market structure: HH/HL (uptrend) or LH/LL (downtrend).
        
        Returns:
            Dictionary with structure information
        """
        highs = df['high'].rolling(window=5, center=True).max() == df['high']
        lows = df['low'].rolling(window=5, center=True).min() == df['low']
        
        swing_highs = df['high'][highs].tail(3)
        swing_lows = df['low'][lows].tail(3)
        
        structure = 'neutral'
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for Higher Highs and Higher Lows (uptrend)
            hh = swing_highs.iloc[-1] > swing_highs.iloc[-2]
            hl = swing_lows.iloc[-1] > swing_lows.iloc[-2]
            
            # Check for Lower Highs and Lower Lows (downtrend)
            lh = swing_highs.iloc[-1] < swing_highs.iloc[-2]
            ll = swing_lows.iloc[-1] < swing_lows.iloc[-2]
            
            if hh and hl:
                structure = 'uptrend'
            elif lh and ll:
                structure = 'downtrend'
                
        return {
            'structure': structure,
            'swing_highs': swing_highs.tolist() if len(swing_highs) > 0 else [],
            'swing_lows': swing_lows.tolist() if len(swing_lows) > 0 else [],
            'last_swing_high': swing_highs.iloc[-1] if len(swing_highs) > 0 else None,
            'last_swing_low': swing_lows.iloc[-1] if len(swing_lows) > 0 else None
        }
        
    def detect_candle_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Detect common candlestick patterns."""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        
        patterns = {
            'bullish_engulfing': False,
            'bearish_engulfing': False,
            'hammer': False,
            'shooting_star': False,
            'doji': False,
            'pin_bar': False
        }
        
        if prev is not None:
            body = abs(last['close'] - last['open'])
            body_prev = abs(prev['close'] - prev['open'])
            range_curr = last['high'] - last['low']
            
            # Bullish Engulfing
            if (prev['close'] < prev['open'] and  # Previous bearish
                last['close'] > last['open'] and   # Current bullish
                last['close'] > prev['open'] and
                last['open'] < prev['close']):
                patterns['bullish_engulfing'] = True
                
            # Bearish Engulfing
            if (prev['close'] > prev['open'] and  # Previous bullish
                last['close'] < last['open'] and   # Current bearish
                last['close'] < prev['open'] and
                last['open'] > prev['close']):
                patterns['bearish_engulfing'] = True
                
            # Hammer (bullish reversal)
            upper_shadow = last['high'] - max(last['close'], last['open'])
            lower_shadow = min(last['close'], last['open']) - last['low']
            if lower_shadow > 2 * body and upper_shadow < 0.3 * body:
                patterns['hammer'] = True
                
            # Shooting Star (bearish reversal)
            if upper_shadow > 2 * body and lower_shadow < 0.3 * body:
                patterns['shooting_star'] = True
                
            # Doji
            if body < 0.1 * range_curr:
                patterns['doji'] = True
                
            # Pin Bar
            if lower_shadow > 2 * body or upper_shadow > 2 * body:
                patterns['pin_bar'] = True
                
        return patterns
        
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels from recent swing."""
        lookback = self.config['fibonacci']['lookback_bars']
        recent_df = df.tail(lookback)
        
        swing_high = recent_df['high'].max()
        swing_low = recent_df['low'].min()
        diff = swing_high - swing_low
        
        levels = {
            '0.0': swing_high,
            '0.236': swing_high - 0.236 * diff,
            '0.382': swing_high - 0.382 * diff,
            '0.5': swing_high - 0.5 * diff,
            '0.618': swing_high - 0.618 * diff,
            '0.786': swing_high - 0.786 * diff,
            '1.0': swing_low
        }
        
        # Find nearest level to current price
        current_price = df['close'].iloc[-1]
        nearest_level = min(levels.values(), key=lambda x: abs(x - current_price))
        
        return {
            'levels': levels,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'nearest_level': nearest_level,
            'distance_to_nearest': abs(current_price - nearest_level)
        }
        
    @staticmethod
    def _detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 14) -> str:
        """Detect bullish/bearish divergence between price and indicator."""
        price_trend = price.iloc[-1] - price.iloc[-lookback]
        ind_trend = indicator.iloc[-1] - indicator.iloc[-lookback]
        
        if price_trend > 0 and ind_trend < 0:
            return 'bearish'  # Price up, indicator down
        elif price_trend < 0 and ind_trend > 0:
            return 'bullish'  # Price down, indicator up
        else:
            return 'none'


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open': np.cumsum(np.random.randn(200)) + 100,
        'high': np.cumsum(np.random.randn(200)) + 102,
        'low': np.cumsum(np.random.randn(200)) + 98,
        'close': np.cumsum(np.random.randn(200)) + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # Calculate all indicators
    indicators = TechnicalIndicators()
    results = indicators.calculate_all(df)
    
    print("=== Indicator Results ===")
    print(f"\nRSI: {results['rsi']['value'].iloc[-1]:.2f}")
    print(f"RSI Overbought: {results['rsi']['is_overbought']}")
    print(f"RSI Oversold: {results['rsi']['is_oversold']}")
    
    print(f"\nATR: {results['atr']['current']:.4f}")
    print(f"ATR %: {results['atr']['percent_of_price']:.2f}%")
    
    print(f"\nSuperTrend: {results['supertrend']['trend']}")
    print(f"ADX Trend Strength: {results['adx']['trend_strength']}")
    print(f"ADX Direction: {results['adx']['direction']}")
    
    print(f"\nPrice Structure: {results['price_structure']['structure']}")
    print(f"Candle Patterns: {results['candle_patterns']}")
    
    print(f"\nMACD: {results['macd']['macd'].iloc[-1]:.4f}")
    print(f"MACD Signal: {results['macd']['signal'].iloc[-1]:.4f}")
    
    print("\nTest completed successfully!")