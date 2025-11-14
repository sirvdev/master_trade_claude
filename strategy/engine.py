"""
Strategy engine for multi-timeframe analysis and trade decision making.
Implements confluence-based entry logic with configurable filters.
"""

import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from indicators.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Multi-timeframe strategy engine with confluence-based decisions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize strategy engine.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.indicators = TechnicalIndicators(config.get('indicators', {}))
        self.strategy_config = config.get('strategy', {})
        self.timeframe_config = config.get('timeframes', {})
        
    def analyze_market(self, symbol: str, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Perform complete multi-timeframe market analysis.
        
        Args:
            symbol: Trading symbol
            multi_tf_data: Dictionary mapping timeframe to DataFrame
            
        Returns:
            Complete analysis dictionary with entry signal
        """
        logger.info(f"Analyzing {symbol} across {len(multi_tf_data)} timeframes")
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'primary_timeframe': self.timeframe_config.get('structure_timeframe', '1H'),
            'timeframe_snapshots': {},
            'market_structure': {},
            'indicators_state': {},
            'entry_signal': False,
            'entry_reason': None,
            'confidence_score': 0.0,
            'direction': None
        }
        
        try:
            # Analyze each timeframe
            for tf, df in multi_tf_data.items():
                if df is None or len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol} @ {tf}")
                    continue
                    
                tf_analysis = self._analyze_timeframe(df, tf)
                analysis['timeframe_snapshots'][tf] = tf_analysis
                
            # Determine market structure from higher timeframe
            structure_tf = self.timeframe_config.get('structure_timeframe', '1H')
            if structure_tf in analysis['timeframe_snapshots']:
                analysis['market_structure'] = self._determine_structure(
                    analysis['timeframe_snapshots'][structure_tf]
                )
                
            # Check for entry signals
            entry_decision = self._evaluate_entry(analysis)
            analysis.update(entry_decision)
            
            # Apply filters
            if analysis['entry_signal']:
                analysis = self._apply_filters(analysis, multi_tf_data)
                
            logger.info(
                f"Analysis complete: {symbol} - "
                f"Signal: {analysis['entry_signal']}, "
                f"Direction: {analysis['direction']}, "
                f"Confidence: {analysis['confidence_score']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            
        return analysis
        
    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Analyze single timeframe.
        
        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe string
            
        Returns:
            Timeframe analysis dictionary
        """
        # Calculate all indicators
        indicators = self.indicators.calculate_all(df)
        
        # Get latest values
        latest = {
            'ohlc': {
                'open': df['open'].iloc[-1],
                'high': df['high'].iloc[-1],
                'low': df['low'].iloc[-1],
                'close': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1]
            },
            'indicators': {}
        }
        
        # Extract current indicator values
        latest['indicators']['ema'] = {
            period: values.iloc[-1]
            for period, values in indicators['ema'].items()
        }
        
        latest['indicators']['rsi'] = {
            'value': indicators['rsi']['value'].iloc[-1],
            'overbought': indicators['rsi']['is_overbought'],
            'oversold': indicators['rsi']['is_oversold']
        }
        
        latest['indicators']['macd'] = {
            'macd': indicators['macd']['macd'].iloc[-1],
            'signal': indicators['macd']['signal'].iloc[-1],
            'histogram': indicators['macd']['histogram'].iloc[-1],
            'bullish_cross': indicators['macd']['macd'].iloc[-1] > indicators['macd']['signal'].iloc[-1]
        }
        
        latest['indicators']['atr'] = {
            'value': indicators['atr']['current'],
            'percent': indicators['atr']['percent_of_price']
        }
        
        latest['indicators']['bollinger'] = {
            'upper': indicators['bollinger']['upper'].iloc[-1],
            'middle': indicators['bollinger']['middle'].iloc[-1],
            'lower': indicators['bollinger']['lower'].iloc[-1],
            'squeeze': indicators['bollinger']['squeeze']
        }
        
        latest['indicators']['supertrend'] = indicators['supertrend']
        latest['indicators']['adx'] = indicators['adx']
        latest['indicators']['price_structure'] = indicators['price_structure']
        latest['indicators']['candle_patterns'] = indicators['candle_patterns']
        
        # Determine trend
        latest['trend'] = self._determine_trend(latest['indicators'], df)
        
        return latest
        
    def _determine_trend(self, indicators: Dict, df: pd.DataFrame) -> Dict:
        """
        Determine trend from indicators.
        
        Args:
            indicators: Indicator values
            df: OHLCV data
            
        Returns:
            Trend dictionary
        """
        signals = []
        
        # EMA trend
        ema_20 = indicators['ema'].get(20, 0)
        ema_50 = indicators['ema'].get(50, 0)
        if ema_20 > ema_50:
            signals.append(1)  # Bullish
        elif ema_20 < ema_50:
            signals.append(-1)  # Bearish
            
        # SuperTrend
        if indicators['supertrend']['trend'] == 'bullish':
            signals.append(1)
        else:
            signals.append(-1)
            
        # ADX direction
        if indicators['adx']['direction'] == 'bullish':
            signals.append(1)
        else:
            signals.append(-1)
            
        # Price structure
        structure = indicators['price_structure']['structure']
        if structure == 'uptrend':
            signals.append(1)
        elif structure == 'downtrend':
            signals.append(-1)
            
        # Aggregate
        trend_score = sum(signals) / len(signals) if signals else 0
        
        if trend_score > 0.3:
            trend = 'bullish'
        elif trend_score < -0.3:
            trend = 'bearish'
        else:
            trend = 'neutral'
            
        return {
            'direction': trend,
            'strength': abs(trend_score),
            'signals': signals
        }
        
    def _determine_structure(self, tf_analysis: Dict) -> Dict:
        """
        Determine market structure from timeframe analysis.
        
        Args:
            tf_analysis: Timeframe analysis data
            
        Returns:
            Market structure dictionary
        """
        structure = tf_analysis['indicators']['price_structure']
        trend = tf_analysis['trend']
        
        return {
            'type': structure['structure'],
            'trend': trend['direction'],
            'strength': trend['strength'],
            'last_swing_high': structure.get('last_swing_high'),
            'last_swing_low': structure.get('last_swing_low')
        }
        
    def _evaluate_entry(self, analysis: Dict) -> Dict:
        """
        Evaluate entry signal based on confluence.
        
        Args:
            analysis: Full market analysis
            
        Returns:
            Entry decision dictionary
        """
        decision = {
            'entry_signal': False,
            'entry_reason': None,
            'direction': None,
            'confidence_score': 0.0,
            'confluence_signals': []
        }
        
        # Get timeframe data
        structure_tf = self.timeframe_config.get('structure_timeframe', '1H')
        entry_tf = self.timeframe_config.get('entry_timeframe', '5m')
        
        if structure_tf not in analysis['timeframe_snapshots']:
            return decision
            
        if entry_tf not in analysis['timeframe_snapshots']:
            return decision
            
        htf = analysis['timeframe_snapshots'][structure_tf]
        ltf = analysis['timeframe_snapshots'][entry_tf]
        
        # Get market structure bias
        structure = analysis.get('market_structure', {})
        bias = structure.get('trend')
        
        # Evaluate entry types
        entry_types = self.strategy_config.get('entry_types', ['breakout_retest'])
        
        for entry_type in entry_types:
            if entry_type == 'breakout_retest':
                result = self._check_breakout_retest(htf, ltf, bias)
                if result['signal']:
                    decision = result
                    break
                    
            elif entry_type == 'pullback_to_sr':
                result = self._check_pullback(htf, ltf, bias)
                if result['signal']:
                    decision = result
                    break
                    
            elif entry_type == 'momentum_breakout':
                result = self._check_momentum_breakout(htf, ltf, bias)
                if result['signal']:
                    decision = result
                    break
                    
        # Check confluence requirement
        min_confluence = self.strategy_config.get('confluence_required', 2)
        if len(decision['confluence_signals']) < min_confluence:
            decision['entry_signal'] = False
            decision['entry_reason'] = f"Insufficient confluence ({len(decision['confluence_signals'])} < {min_confluence})"
            
        return decision
        
    def _check_breakout_retest(self, htf: Dict, ltf: Dict, bias: str) -> Dict:
        """Check for breakout + retest entry."""
        decision = {
            'signal': False,
            'entry_signal': False,
            'entry_reason': None,
            'direction': None,
            'confidence_score': 0.0,
            'confluence_signals': []
        }
        
        ltf_ind = ltf['indicators']
        htf_ind = htf['indicators']
        
        # Check if higher timeframe supports direction
        if bias not in ['bullish', 'bearish']:
            return decision
            
        # Look for breakout structure
        structure = ltf_ind['price_structure']
        
        if bias == 'bullish':
            # Check for break above resistance and retest
            last_high = structure.get('last_swing_high')
            current_price = ltf['ohlc']['close']
            
            if last_high and current_price > last_high:
                decision['confluence_signals'].append('price_above_structure')
                
            # EMA support
            if ltf_ind['ema'].get(20, 0) < current_price:
                decision['confluence_signals'].append('ema_support')
                
            # Bullish candle pattern
            if any([
                ltf_ind['candle_patterns'].get('bullish_engulfing'),
                ltf_ind['candle_patterns'].get('hammer')
            ]):
                decision['confluence_signals'].append('bullish_pattern')
                
            # RSI confirmation
            rsi = ltf_ind['rsi']['value']
            if 40 < rsi < 70:
                decision['confluence_signals'].append('rsi_healthy')
                
            direction = 'long'
            
        else:  # bearish
            last_low = structure.get('last_swing_low')
            current_price = ltf['ohlc']['close']
            
            if last_low and current_price < last_low:
                decision['confluence_signals'].append('price_below_structure')
                
            if ltf_ind['ema'].get(20, 0) > current_price:
                decision['confluence_signals'].append('ema_resistance')
                
            if any([
                ltf_ind['candle_patterns'].get('bearish_engulfing'),
                ltf_ind['candle_patterns'].get('shooting_star')
            ]):
                decision['confluence_signals'].append('bearish_pattern')
                
            rsi = ltf_ind['rsi']['value']
            if 30 < rsi < 60:
                decision['confluence_signals'].append('rsi_healthy')
                
            direction = 'short'
            
        if len(decision['confluence_signals']) >= 2:
            decision['signal'] = True
            decision['entry_signal'] = True
            decision['direction'] = direction
            decision['entry_reason'] = f"Breakout+retest {direction} with {len(decision['confluence_signals'])} signals"
            decision['confidence_score'] = min(0.95, len(decision['confluence_signals']) * 0.25)
            
        return decision
        
    def _check_pullback(self, htf: Dict, ltf: Dict, bias: str) -> Dict:
        """Check for pullback to support/resistance entry."""
        decision = {
            'signal': False,
            'entry_signal': False,
            'entry_reason': None,
            'direction': None,
            'confidence_score': 0.0,
            'confluence_signals': []
        }
        
        if bias not in ['bullish', 'bearish']:
            return decision
            
        ltf_ind = ltf['indicators']
        current_price = ltf['ohlc']['close']
        
        if bias == 'bullish':
            # Check if price pulled back to support
            ema_50 = ltf_ind['ema'].get(50, 0)
            if abs(current_price - ema_50) / ema_50 < 0.002:  # Within 0.2%
                decision['confluence_signals'].append('pullback_to_ema50')
                
            # Oversold RSI recovery
            rsi = ltf_ind['rsi']['value']
            if ltf_ind['rsi']['oversold'] or (30 < rsi < 40):
                decision['confluence_signals'].append('rsi_oversold_recovery')
                
            # Volume increase
            if ltf['ohlc']['volume'] > ltf['ohlc'].get('volume_avg', 0):
                decision['confluence_signals'].append('volume_increase')
                
            direction = 'long'
        else:
            ema_50 = ltf_ind['ema'].get(50, 0)
            if abs(current_price - ema_50) / ema_50 < 0.002:
                decision['confluence_signals'].append('pullback_to_ema50')
                
            rsi = ltf_ind['rsi']['value']
            if ltf_ind['rsi']['overbought'] or (60 < rsi < 70):
                decision['confluence_signals'].append('rsi_overbought_rejection')
                
            direction = 'short'
            
        if len(decision['confluence_signals']) >= 2:
            decision['signal'] = True
            decision['entry_signal'] = True
            decision['direction'] = direction
            decision['entry_reason'] = f"Pullback {direction} with {len(decision['confluence_signals'])} signals"
            decision['confidence_score'] = min(0.90, len(decision['confluence_signals']) * 0.25)
            
        return decision
        
    def _check_momentum_breakout(self, htf: Dict, ltf: Dict, bias: str) -> Dict:
        """Check for momentum breakout on volatility expansion."""
        decision = {
            'signal': False,
            'entry_signal': False,
            'entry_reason': None,
            'direction': None,
            'confidence_score': 0.0,
            'confluence_signals': []
        }
        
        ltf_ind = ltf['indicators']
        
        # Check for Bollinger squeeze breakout
        if ltf_ind['bollinger']['squeeze']:
            decision['confluence_signals'].append('bollinger_squeeze')
            
        # ATR expansion
        atr_current = ltf_ind['atr']['value']
        # Would compare to historical ATR percentile
        
        # MACD momentum
        if ltf_ind['macd']['bullish_cross']:
            decision['confluence_signals'].append('macd_bullish_cross')
            direction = 'long'
        elif not ltf_ind['macd']['bullish_cross'] and ltf_ind['macd']['histogram'] < 0:
            decision['confluence_signals'].append('macd_bearish_cross')
            direction = 'short'
        else:
            return decision
            
        # Strong ADX
        if ltf_ind['adx']['trend_strength'] == 'strong':
            decision['confluence_signals'].append('strong_trend')
            
        if len(decision['confluence_signals']) >= 2:
            decision['signal'] = True
            decision['entry_signal'] = True
            decision['direction'] = direction
            decision['entry_reason'] = f"Momentum breakout {direction} with {len(decision['confluence_signals'])} signals"
            decision['confidence_score'] = min(0.85, len(decision['confluence_signals']) * 0.25)
            
        return decision
        
    def _apply_filters(self, analysis: Dict, multi_tf_data: Dict) -> Dict:
        """
        Apply filters to entry signal.
        
        Args:
            analysis: Analysis with entry signal
            multi_tf_data: Multi-timeframe data
            
        Returns:
            Updated analysis
        """
        filters = self.strategy_config.get('filters', {})
        
        # Higher timeframe bias filter
        if filters.get('respect_higher_tf_bias', True):
            structure_tf = self.timeframe_config.get('structure_timeframe', '1H')
            if structure_tf in analysis['timeframe_snapshots']:
                htf_trend = analysis['timeframe_snapshots'][structure_tf]['trend']['direction']
                
                if analysis['direction'] == 'long' and htf_trend == 'bearish':
                    if not filters.get('allow_override', False):
                        analysis['entry_signal'] = False
                        analysis['entry_reason'] = "Filtered: Against higher TF trend"
                        logger.info(f"Filtered entry: Long signal against bearish HTF trend")
                        
                elif analysis['direction'] == 'short' and htf_trend == 'bullish':
                    if not filters.get('allow_override', False):
                        analysis['entry_signal'] = False
                        analysis['entry_reason'] = "Filtered: Against higher TF trend"
                        logger.info(f"Filtered entry: Short signal against bullish HTF trend")
                        
        # ATR volatility filter
        entry_tf = self.timeframe_config.get('entry_timeframe', '5m')
        if entry_tf in analysis['timeframe_snapshots']:
            atr_percent = analysis['timeframe_snapshots'][entry_tf]['indicators']['atr']['percent']
            
            min_atr = filters.get('min_atr_threshold', 0)
            max_atr = filters.get('max_atr_threshold', 100)
            
            if atr_percent < min_atr * 100:
                analysis['entry_signal'] = False
                analysis['entry_reason'] = f"Filtered: ATR too low ({atr_percent:.2f}%)"
                logger.info(f"Filtered entry: ATR {atr_percent:.2f}% < {min_atr*100}%")
                
            elif atr_percent > max_atr * 100:
                analysis['entry_signal'] = False
                analysis['entry_reason'] = f"Filtered: ATR too high ({atr_percent:.2f}%)"
                logger.info(f"Filtered entry: ATR {atr_percent:.2f}% > {max_atr*100}%")
                
        # News blackout filter
        if filters.get('news_blackout', {}).get('enabled', False):
            if self._is_news_blackout():
                analysis['entry_signal'] = False
                analysis['entry_reason'] = "Filtered: News blackout period"
                logger.info("Filtered entry: News blackout active")
                
        return analysis
        
    def _is_news_blackout(self) -> bool:
        """Check if current time is in news blackout window."""
        blackout_config = self.strategy_config.get('filters', {}).get('news_blackout', {})
        windows = blackout_config.get('windows', [])
        
        now = datetime.utcnow()
        current_time = now.time()
        current_day = now.weekday() + 1  # Monday = 1
        
        for window in windows:
            days = window.get('days', [])
            if current_day not in days:
                continue
                
            start = datetime.strptime(window['start'], '%H:%M').time()
            end = datetime.strptime(window['end'], '%H:%M').time()
            
            if start <= current_time <= end:
                return True
                
        return False
        
    def calculate_entry_levels(self, analysis: Dict, multi_tf_data: Dict) -> Dict:
        """
        Calculate entry price, stop loss, and take profit levels.
        
        Args:
            analysis: Market analysis with entry signal
            multi_tf_data: Multi-timeframe data
            
        Returns:
            Dictionary with entry levels
        """
        entry_tf = self.timeframe_config.get('entry_timeframe', '5m')
        if entry_tf not in multi_tf_data:
            return {}
            
        df = multi_tf_data[entry_tf]
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR for dynamic levels
        atr = self.indicators.calculate_atr(df)['current']
        
        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(
            analysis,
            current_price,
            atr,
            df
        )
        
        # Calculate take profit levels
        take_profits = self._calculate_take_profits(
            analysis,
            current_price,
            stop_loss,
            atr
        )
        
        return {
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit_1': take_profits.get('tp1'),
            'take_profit_2': take_profits.get('tp2'),
            'take_profit_3': take_profits.get('tp3'),
            'atr': atr,
            'risk_distance': abs(current_price - stop_loss)
        }
        
    def _calculate_stop_loss(self, analysis: Dict, entry_price: float,
                             atr: float, df: pd.DataFrame) -> float:
        """Calculate stop loss level."""
        direction = analysis['direction']
        
        # Get swing levels from price structure
        entry_tf = self.timeframe_config.get('entry_timeframe', '5m')
        structure = analysis['timeframe_snapshots'][entry_tf]['indicators']['price_structure']
        
        # Structure-based stop
        if direction == 'long':
            structure_stop = structure.get('last_swing_low', entry_price - 2 * atr)
        else:
            structure_stop = structure.get('last_swing_high', entry_price + 2 * atr)
            
        # ATR-based stop
        atr_multiplier = 2.0  # From config
        if direction == 'long':
            atr_stop = entry_price - atr * atr_multiplier
        else:
            atr_stop = entry_price + atr * atr_multiplier
            
        # Use more conservative (wider stop)
        if direction == 'long':
            stop_loss = min(structure_stop, atr_stop)
        else:
            stop_loss = max(structure_stop, atr_stop)
            
        return stop_loss
        
    def _calculate_take_profits(self, analysis: Dict, entry_price: float,
                                stop_loss: float, atr: float) -> Dict:
        """Calculate take profit levels based on R:R ratios."""
        risk = abs(entry_price - stop_loss)
        direction = analysis['direction']
        
        # TP ratios from config (default: 1.5R, 3R, trail)
        tp_targets = [
            {'rr_ratio': 1.5, 'close_percent': 50},
            {'rr_ratio': 3.0, 'close_percent': 30},
        ]
        
        tps = {}
        for i, target in enumerate(tp_targets, 1):
            rr = target['rr_ratio']
            if direction == 'long':
                tp = entry_price + risk * rr
            else:
                tp = entry_price - risk * rr
            tps[f'tp{i}'] = tp
            
        return tps


# Example usage
if __name__ == "__main__":
    # Sample configuration
    config = {
        'indicators': TechnicalIndicators._default_config(),
        'strategy': {
            'entry_types': ['breakout_retest', 'pullback_to_sr'],
            'confluence_required': 2
        },
        'timeframes': {
            'structure_timeframe': '1H',
            'entry_timeframe': '5m'
        }
    }
    
    engine = StrategyEngine(config)
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    df_1h = pd.DataFrame({
        'open': np.cumsum(np.random.randn(200)) + 100,
        'high': np.cumsum(np.random.randn(200)) + 102,
        'low': np.cumsum(np.random.randn(200)) + 98,
        'close': np.cumsum(np.random.randn(200)) + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    df_5m = df_1h.copy()  # Simplified for testing
    
    # Ensure high/low validity
    for df in [df_1h, df_5m]:
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # Perform analysis
    multi_tf_data = {'1H': df_1h, '5m': df_5m}
    analysis = engine.analyze_market('BTC/USDT', multi_tf_data)
    
    print("=== Strategy Analysis ===")
    print(f"Symbol: {analysis['symbol']}")
    print(f"Entry Signal: {analysis['entry_signal']}")
    print(f"Direction: {analysis['direction']}")
    print(f"Reason: {analysis['entry_reason']}")
    print(f"Confidence: {analysis['confidence_score']:.2f}")
    print(f"Confluence Signals: {analysis.get('confluence_signals', [])}")
    
    if analysis['entry_signal']:
        levels = engine.calculate_entry_levels(analysis, multi_tf_data)
        print(f"\n=== Entry Levels ===")
        print(f"Entry: {levels['entry_price']:.2f}")
        print(f"Stop Loss: {levels['stop_loss']:.2f}")
        print(f"TP1: {levels['take_profit_1']:.2f}")
        print(f"TP2: {levels['take_profit_2']:.2f}")
        print(f"Risk: {levels['risk_distance']:.2f}")
    
    print("\nStrategy engine test completed!")