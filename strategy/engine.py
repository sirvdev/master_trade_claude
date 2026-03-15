"""
strategy/engine.py
==================
Multi-timeframe strategy engine with confluence-based entry logic.

Changes v2:
  - Weighted confluence scoring (score >= threshold, not count >= N)
  - Per-symbol confluence thresholds (volatile vs calm)
  - Fixed breakout_retest: requires confirmed prior break + proximity + rejection
  - New entry types: ema_stack_pullback, bb_squeeze_breakout,
                     macd_zero_cross, rsi_divergence
  - Smart order type: limit for zone entries, market for momentum entries
  - entry_type and order_type propagated into analysis dict for executor
  - Session-aware ATR multiplier delegated to StopManager
  - All existing filters (_apply_filters) preserved unchanged
"""

import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from indicators.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


# ── Confluence signal weights ─────────────────────────────────────────────────
# Higher weight = stronger evidence. Scores are summed; result must reach
# the symbol's confluence_threshold before an entry fires.
_SIGNAL_WEIGHTS: Dict[str, int] = {
    # Structural — high confidence (3 pts)
    'supertrend_aligned':       3,
    'ema_stack_full':           3,
    'price_structure_aligned':  3,
    'rsi_divergence_confirmed': 3,
    'breakout_held':            3,
    # Momentum — medium confidence (2 pts)
    'macd_zero_cross':          2,
    'rsi_confirmation':         2,
    'bb_squeeze_active':        2,
    'ema_proximity_bounce':     2,
    'pullback_to_ema50':        2,
    'adx_strong':               2,
    # Confirming — lower confidence (1 pt)
    'candle_pattern':           1,
    'volume_confirmation':      1,
    'rsi_oversold_recovery':    1,
    'rsi_overbought_rejection': 1,
    'macd_bullish_cross':       1,
    'macd_bearish_cross':       1,
    'bollinger_squeeze':        1,
    'strong_trend':             1,
}

# Entry types that use limit orders (zone-based, level is known in advance)
_LIMIT_ORDER_TYPES = {'ema_stack_pullback', 'pullback_to_sr', 'rsi_divergence'}

# Volatile symbols — use the higher confluence_required threshold
_VOLATILE_SYMBOLS = {
    'XAUUSD', 'XAU/USD', 'BTCUSD', 'BTC/USD',
    'NAS100USD', 'NAS100', 'US100',
}


class StrategyEngine:
    """
    Multi-timeframe strategy engine with confluence-based decisions.
    """

    def __init__(self, config: Dict):
        self.config           = config
        self.indicators       = TechnicalIndicators(config.get('indicators', {}))
        self.strategy_config  = config.get('strategy', {})
        self.timeframe_config = config.get('timeframes', {})

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze_market(self, symbol: str, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Full multi-timeframe analysis. Returns analysis dict including entry
        signal, direction, entry_type, order_type, limit_price, and
        confluence_score.
        """
        logger.info(f"Analyzing {symbol} across {len(multi_tf_data)} timeframes")

        analysis = {
            'symbol':              symbol,
            'timestamp':           datetime.utcnow(),
            'primary_timeframe':   self.timeframe_config.get('structure_timeframe', '1H'),
            'timeframe_snapshots': {},
            'market_structure':    {},
            'indicators_state':    {},
            'entry_signal':        False,
            'entry_reason':        None,
            'entry_type':          None,
            'confidence_score':    0.0,
            'confluence_score':    0.0,
            'confluence_signals':  [],
            'direction':           None,
            'order_type':          'market',
            'limit_price':         None,
        }

        try:
            for tf, df in multi_tf_data.items():
                if df is None or len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol} @ {tf}")
                    continue
                analysis['timeframe_snapshots'][tf] = self._analyze_timeframe(df, tf)

            structure_tf = self.timeframe_config.get('structure_timeframe', '1H')
            if structure_tf in analysis['timeframe_snapshots']:
                analysis['market_structure'] = self._determine_structure(
                    analysis['timeframe_snapshots'][structure_tf]
                )

            entry_decision = self._evaluate_entry(analysis, multi_tf_data)
            analysis.update(entry_decision)

            if analysis['entry_signal']:
                analysis = self._apply_filters(analysis, multi_tf_data)

            logger.info(
                f"Analysis complete: {symbol} — "
                f"signal={analysis['entry_signal']} "
                f"dir={analysis['direction']} "
                f"type={analysis.get('entry_type')} "
                f"order={analysis.get('order_type')} "
                f"score={analysis.get('confluence_score', 0):.1f} "
                f"conf={analysis['confidence_score']:.2f}"
            )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)

        return analysis

    def calculate_entry_levels(self, analysis: Dict, multi_tf_data: Dict) -> Dict:
        """
        Calculate entry price, stop loss, and take profit levels.
        For limit entries the entry_price is the limit level, not current price.

        Returns dict with: entry_price, order_price, order_type, limit_price,
                           stop_loss, take_profit_1/2/3, atr, risk_distance.
        """
        entry_tf = self.timeframe_config.get('entry_timeframe', '5m')
        if entry_tf not in multi_tf_data:
            return {}

        df           = multi_tf_data[entry_tf]
        current_price = float(df['close'].iloc[-1])
        atr           = self.indicators.calculate_atr(df)['current']

        order_type   = analysis.get('order_type', 'market')
        limit_price  = analysis.get('limit_price')

        # For limit entries use the limit level for SL/TP calculations
        entry_price = float(limit_price) if (order_type == 'limit' and limit_price) \
                      else current_price

        stop_loss = self._calculate_stop_loss(analysis, entry_price, atr, df)

        take_profits = self._calculate_take_profits(
            analysis, entry_price, stop_loss, atr
        )

        return {
            'entry_price':    current_price,   # current market reference
            'order_price':    entry_price,      # actual price for order placement
            'order_type':     order_type,
            'limit_price':    entry_price if order_type == 'limit' else None,
            'stop_loss':      stop_loss,
            'take_profit_1':  take_profits.get('tp1'),
            'take_profit_2':  take_profits.get('tp2'),
            'take_profit_3':  take_profits.get('tp3'),
            'atr':            atr,
            'risk_distance':  abs(entry_price - stop_loss),
        }

    # ── Timeframe analysis ─────────────────────────────────────────────────────

    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        indicators = self.indicators.calculate_all(df)

        latest = {
            'ohlc': {
                'open':   float(df['open'].iloc[-1]),
                'high':   float(df['high'].iloc[-1]),
                'low':    float(df['low'].iloc[-1]),
                'close':  float(df['close'].iloc[-1]),
                'volume': float(df['volume'].iloc[-1]),
            },
            'indicators': {},
        }

        latest['indicators']['ema'] = {
            p: float(v.iloc[-1])
            for p, v in indicators['ema'].items()
        }
        latest['indicators']['rsi'] = {
            'value':      float(indicators['rsi']['value'].iloc[-1]),
            'overbought': bool(indicators['rsi']['is_overbought']),
            'oversold':   bool(indicators['rsi']['is_oversold']),
        }
        latest['indicators']['macd'] = {
            'macd':         float(indicators['macd']['macd'].iloc[-1]),
            'signal':       float(indicators['macd']['signal'].iloc[-1]),
            'histogram':    float(indicators['macd']['histogram'].iloc[-1]),
            'bullish_cross': (
                float(indicators['macd']['macd'].iloc[-1]) >
                float(indicators['macd']['signal'].iloc[-1])
            ),
        }
        latest['indicators']['atr'] = {
            'value':   float(indicators['atr']['current']),
            'percent': float(indicators['atr']['percent_of_price']),
        }
        latest['indicators']['bollinger'] = {
            'upper':   float(indicators['bollinger']['upper'].iloc[-1]),
            'middle':  float(indicators['bollinger']['middle'].iloc[-1]),
            'lower':   float(indicators['bollinger']['lower'].iloc[-1]),
            'squeeze': bool(indicators['bollinger']['squeeze']),
        }
        latest['indicators']['supertrend']       = indicators['supertrend']
        latest['indicators']['adx']              = indicators['adx']
        latest['indicators']['price_structure']  = indicators['price_structure']
        latest['indicators']['candle_patterns']  = indicators['candle_patterns']
        latest['trend'] = self._determine_trend(latest['indicators'], df)

        return latest

    def _determine_trend(self, indicators: Dict, df: pd.DataFrame) -> Dict:
        signals = []
        ema_20 = indicators['ema'].get(20, 0)
        ema_50 = indicators['ema'].get(50, 0)
        if ema_20 > ema_50:
            signals.append(1)
        elif ema_20 < ema_50:
            signals.append(-1)

        if indicators['supertrend']['trend'] == 'bullish':
            signals.append(1)
        else:
            signals.append(-1)

        if indicators['adx']['direction'] == 'bullish':
            signals.append(1)
        else:
            signals.append(-1)

        structure = indicators['price_structure']['structure']
        if structure == 'uptrend':
            signals.append(1)
        elif structure == 'downtrend':
            signals.append(-1)

        score = sum(signals) / len(signals) if signals else 0
        if score > 0.3:
            trend = 'bullish'
        elif score < -0.3:
            trend = 'bearish'
        else:
            trend = 'neutral'

        return {'direction': trend, 'strength': abs(score), 'signals': signals}

    def _determine_structure(self, tf_analysis: Dict) -> Dict:
        structure = tf_analysis['indicators']['price_structure']
        trend     = tf_analysis['trend']
        return {
            'type':            structure['structure'],
            'trend':           trend['direction'],
            'strength':        trend['strength'],
            'last_swing_high': structure.get('last_swing_high'),
            'last_swing_low':  structure.get('last_swing_low'),
        }

    # ── Entry evaluation ───────────────────────────────────────────────────────

    def _evaluate_entry(
        self,
        analysis: Dict,
        multi_tf_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Evaluate entry signals using weighted confluence scoring.
        Runs each entry type in config order; first type that meets the
        symbol's score threshold produces the signal.
        """
        decision = {
            'entry_signal':     False,
            'entry_reason':     None,
            'entry_type':       None,
            'direction':        None,
            'confidence_score': 0.0,
            'confluence_score': 0.0,
            'confluence_signals': [],
            'order_type':       'market',
            'limit_price':      None,
        }

        structure_tf = self.timeframe_config.get('structure_timeframe', '1H')
        entry_tf     = self.timeframe_config.get('entry_timeframe', '5m')

        if structure_tf not in analysis['timeframe_snapshots']:
            return decision
        if entry_tf not in analysis['timeframe_snapshots']:
            return decision

        htf     = analysis['timeframe_snapshots'][structure_tf]
        ltf     = analysis['timeframe_snapshots'][entry_tf]
        htf_df  = (multi_tf_data or {}).get(structure_tf)
        ltf_df  = (multi_tf_data or {}).get(entry_tf)
        bias    = analysis.get('market_structure', {}).get('trend')

        threshold = self._get_confluence_threshold(analysis.get('symbol', ''))
        entry_types = self.strategy_config.get('entry_types', ['breakout_retest'])

        for entry_type in entry_types:
            result = self._run_entry_check(entry_type, htf, ltf, htf_df, ltf_df, bias)
            if not result.get('signal'):
                continue

            score = self._calculate_confluence_score(result['confluence_signals'])
            if score < threshold:
                logger.debug(
                    f"[{analysis.get('symbol')}] {entry_type}: score={score:.0f} "
                    f"< threshold={threshold} — skipped"
                )
                continue

            order_type = 'limit' if entry_type in _LIMIT_ORDER_TYPES else 'market'
            decision.update({
                'entry_signal':      True,
                'entry_reason':      result.get('entry_reason', entry_type),
                'entry_type':        entry_type,
                'direction':         result['direction'],
                'confidence_score':  min(1.0, score / max(threshold * 1.5, 10.0)),
                'confluence_score':  score,
                'confluence_signals': result['confluence_signals'],
                'order_type':        order_type,
                'limit_price':       result.get('limit_price'),
            })
            break

        return decision

    def _run_entry_check(
        self,
        entry_type: str,
        htf: Dict, ltf: Dict,
        htf_df: Optional[pd.DataFrame],
        ltf_df: Optional[pd.DataFrame],
        bias: str,
    ) -> Dict:
        """Dispatch to the correct entry check function."""
        dispatch = {
            'breakout_retest':    self._check_breakout_retest,
            'ema_stack_pullback': self._check_ema_stack_pullback,
            'pullback_to_sr':     self._check_pullback,
            'bb_squeeze_breakout': self._check_bb_squeeze_breakout,
            'macd_zero_cross':    self._check_macd_zero_cross,
            'momentum_breakout':  self._check_momentum_breakout,
            'rsi_divergence':     self._check_rsi_divergence,
        }
        fn = dispatch.get(entry_type)
        if fn is None:
            return {'signal': False}
        try:
            return fn(htf, ltf, htf_df, ltf_df, bias)
        except Exception as e:
            logger.debug(f"Entry check {entry_type} error: {e}")
            return {'signal': False}

    def _empty_decision(self) -> Dict:
        return {
            'signal': False, 'entry_signal': False,
            'entry_reason': None, 'direction': None,
            'confidence_score': 0.0, 'confluence_signals': [],
            'limit_price': None,
        }

    # ── Entry type checks ──────────────────────────────────────────────────────

    def _check_breakout_retest(
        self, htf, ltf, htf_df, ltf_df, bias
    ) -> Dict:
        """
        Real breakout + retest logic:
        1. A prior closed bar broke above (long) / below (short) a swing level.
        2. Current bar has pulled back to within 0.5 ATR of that level.
        3. Current closed bar closed back above (long) / below (short) the level — rejection.
        """
        d = self._empty_decision()
        if bias not in ('bullish', 'bearish'):
            return d
        if ltf_df is None or len(ltf_df) < 10:
            return d

        ltf_ind = ltf['indicators']
        atr     = ltf_ind['atr']['value']
        cfg     = self.strategy_config.get('breakout_retest', {})
        lookback   = int(cfg.get('lookback_bars', 5))
        max_dist   = float(cfg.get('max_distance_atr', 0.5))

        # Use the confirmed closed bar = iloc[-1] (bar-close sync ensures this is closed)
        current_bar   = ltf_df.iloc[-1]
        current_close = float(current_bar['close'])

        # Prior bars: everything before the current bar
        prior = ltf_df.iloc[-(lookback + 2):-1]
        if len(prior) < 3:
            return d

        if bias == 'bullish':
            direction  = 'long'
            level      = float(prior['high'].max())          # swing resistance
            broke_out  = any(float(r['close']) > level for _, r in prior.iterrows())
            near_level = abs(current_close - level) <= max_dist * atr
            rejection  = current_close >= level              # held above after pullback

            if not (broke_out and near_level and rejection):
                return d

        else:  # bearish
            direction  = 'short'
            level      = float(prior['low'].min())           # swing support
            broke_down = any(float(r['close']) < level for _, r in prior.iterrows())
            near_level = abs(current_close - level) <= max_dist * atr
            rejection  = current_close <= level              # held below after pullback

            if not (broke_down and near_level and rejection):
                return d

        # Gather confluence
        signals = ['breakout_held']

        if ltf_ind['supertrend']['trend'] == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('supertrend_aligned')

        rsi = ltf_ind['rsi']['value']
        if direction == 'long' and 40 < rsi < 70:
            signals.append('rsi_confirmation')
        elif direction == 'short' and 30 < rsi < 60:
            signals.append('rsi_confirmation')

        if ltf_ind['adx'].get('trend_strength') == 'strong':
            signals.append('adx_strong')

        if direction == 'long' and ltf_ind['candle_patterns'].get('bullish_engulfing'):
            signals.append('candle_pattern')
        elif direction == 'short' and ltf_ind['candle_patterns'].get('bearish_engulfing'):
            signals.append('candle_pattern')

        # Check EMA alignment
        ema20 = ltf_ind['ema'].get(20, 0)
        ema50 = ltf_ind['ema'].get(50, 0)
        if direction == 'long' and ema20 > ema50:
            signals.append('ema_stack_full')
        elif direction == 'short' and ema20 < ema50:
            signals.append('ema_stack_full')

        d.update({
            'signal': True, 'entry_signal': True,
            'direction': direction, 'confluence_signals': signals,
            'entry_reason': f'Breakout retest {direction} @ {level:.2f}',
        })
        return d

    def _check_ema_stack_pullback(
        self, htf, ltf, htf_df, ltf_df, bias
    ) -> Dict:
        """
        Full EMA stack alignment (EMA20>50>200 for longs) + price pulls back
        to EMA20 proximity + confirmed bounce (close back above EMA20).
        Limit order entry at EMA20 value.
        """
        d = self._empty_decision()
        if bias not in ('bullish', 'bearish'):
            return d

        ltf_ind = ltf['indicators']
        htf_ind = htf['indicators']

        ema20_ltf = ltf_ind['ema'].get(20, 0)
        ema50_ltf = ltf_ind['ema'].get(50, 0)
        ema200_ltf = ltf_ind['ema'].get(200, 0)
        ema20_htf = htf_ind['ema'].get(20, 0)
        ema50_htf = htf_ind['ema'].get(50, 0)
        atr       = ltf_ind['atr']['value']
        close     = ltf['ohlc']['close']

        if bias == 'bullish':
            direction  = 'long'
            stack_ltf  = ema20_ltf > ema50_ltf > 0
            stack_htf  = ema20_htf > ema50_htf > 0
            proximity  = abs(close - ema20_ltf) <= 0.5 * atr if ema20_ltf > 0 else False
            bounce     = close > ema20_ltf if ema20_ltf > 0 else False
            full_stack = (ema200_ltf > 0 and ema20_ltf > ema50_ltf > ema200_ltf)
        else:
            direction  = 'short'
            stack_ltf  = ema20_ltf < ema50_ltf
            stack_htf  = ema20_htf < ema50_htf
            proximity  = abs(close - ema20_ltf) <= 0.5 * atr if ema20_ltf > 0 else False
            bounce     = close < ema20_ltf if ema20_ltf > 0 else False
            full_stack = (ema200_ltf > 0 and ema20_ltf < ema50_ltf < ema200_ltf)

        if not (stack_ltf and stack_htf and proximity and bounce):
            return d

        signals = ['ema_proximity_bounce']
        if full_stack:
            signals.append('ema_stack_full')
        if ltf_ind['supertrend']['trend'] == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('supertrend_aligned')

        rsi = ltf_ind['rsi']['value']
        if direction == 'long' and 35 < rsi < 65:
            signals.append('rsi_confirmation')
        elif direction == 'short' and 35 < rsi < 65:
            signals.append('rsi_confirmation')

        if ltf_ind['adx'].get('trend_strength') == 'strong':
            signals.append('adx_strong')

        htf_structure = htf['trend']['direction']
        if htf_structure == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('price_structure_aligned')

        d.update({
            'signal': True, 'entry_signal': True,
            'direction': direction, 'confluence_signals': signals,
            'entry_reason': f'EMA stack pullback {direction} @ EMA20={ema20_ltf:.2f}',
            'limit_price': ema20_ltf,
        })
        return d

    def _check_pullback(
        self, htf, ltf, htf_df, ltf_df, bias
    ) -> Dict:
        """
        Pullback to EMA50 support/resistance.
        Limit order at EMA50.
        """
        d = self._empty_decision()
        if bias not in ('bullish', 'bearish'):
            return d

        ltf_ind      = ltf['indicators']
        current_price = ltf['ohlc']['close']
        ema_50        = ltf_ind['ema'].get(50, 0)
        atr           = ltf_ind['atr']['value']

        if ema_50 <= 0:
            return d

        proximity = abs(current_price - ema_50) / ema_50 < 0.003   # within 0.3%

        if bias == 'bullish':
            direction = 'long'
            valid     = proximity and current_price > ema_50
        else:
            direction = 'short'
            valid     = proximity and current_price < ema_50

        if not valid:
            return d

        signals = ['pullback_to_ema50']
        rsi = ltf_ind['rsi']['value']

        if direction == 'long':
            if ltf_ind['rsi']['oversold'] or (30 < rsi < 45):
                signals.append('rsi_oversold_recovery')
        else:
            if ltf_ind['rsi']['overbought'] or (55 < rsi < 70):
                signals.append('rsi_overbought_rejection')

        if ltf_ind['supertrend']['trend'] == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('supertrend_aligned')

        if ltf_ind['adx'].get('trend_strength') == 'strong':
            signals.append('adx_strong')

        if direction == 'long' and any([
            ltf_ind['candle_patterns'].get('bullish_engulfing'),
            ltf_ind['candle_patterns'].get('hammer'),
        ]):
            signals.append('candle_pattern')
        elif direction == 'short' and any([
            ltf_ind['candle_patterns'].get('bearish_engulfing'),
            ltf_ind['candle_patterns'].get('shooting_star'),
        ]):
            signals.append('candle_pattern')

        ema20 = ltf_ind['ema'].get(20, 0)
        ema200 = ltf_ind['ema'].get(200, 0)
        if direction == 'long' and ema20 > ema_50 > ema200 > 0:
            signals.append('ema_stack_full')
        elif direction == 'short' and ema20 < ema_50 and ema200 > 0 and ema_50 < ema200:
            signals.append('ema_stack_full')

        d.update({
            'signal': True, 'entry_signal': True,
            'direction': direction, 'confluence_signals': signals,
            'entry_reason': f'Pullback to EMA50 {direction} @ {ema_50:.2f}',
            'limit_price': ema_50,
        })
        return d

    def _check_bb_squeeze_breakout(
        self, htf, ltf, htf_df, ltf_df, bias
    ) -> Dict:
        """
        Bollinger Band squeeze breakout: bands narrowed (squeeze=True),
        then price breaks out with ATR expansion.
        Market order — catches the volatility expansion.
        """
        d = self._empty_decision()
        if bias not in ('bullish', 'bearish'):
            return d

        ltf_ind = ltf['indicators']
        bb      = ltf_ind['bollinger']
        atr     = ltf_ind['atr']['value']
        close   = ltf['ohlc']['close']

        if not bb.get('squeeze'):
            return d

        is_breakout_up   = close > bb['upper']
        is_breakout_down = close < bb['lower']

        if bias == 'bullish' and not is_breakout_up:
            return d
        if bias == 'bearish' and not is_breakout_down:
            return d

        direction = 'long' if bias == 'bullish' else 'short'

        # Confirm ATR expansion: current ATR vs 5-bar average
        atr_expanding = True
        if ltf_df is not None and len(ltf_df) >= 6:
            try:
                recent_atr = self.indicators.calculate_atr(
                    ltf_df.iloc[-6:-1]
                )['current']
                atr_expanding = atr > recent_atr * 1.1
            except Exception:
                pass

        signals = ['bb_squeeze_active', 'bollinger_squeeze']
        if atr_expanding:
            signals.append('strong_trend')

        if ltf_ind['supertrend']['trend'] == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('supertrend_aligned')
        if ltf_ind['macd']['bullish_cross' if direction == 'long' else 'histogram' < 0]:
            pass
        if ltf_ind['adx'].get('trend_strength') == 'strong':
            signals.append('adx_strong')

        d.update({
            'signal': True, 'entry_signal': True,
            'direction': direction, 'confluence_signals': signals,
            'entry_reason': f'BB squeeze breakout {direction}',
        })
        return d

    def _check_macd_zero_cross(
        self, htf, ltf, htf_df, ltf_df, bias
    ) -> Dict:
        """
        MACD zero-line cross: MACD crosses from negative to positive (long)
        or positive to negative (short) on the current confirmed closed bar.
        Requires prior closed bar to confirm the direction of the cross.
        Market order.
        """
        d = self._empty_decision()
        if bias not in ('bullish', 'bearish'):
            return d
        if ltf_df is None or len(ltf_df) < 3:
            return d

        ltf_ind     = ltf['indicators']
        current_macd = ltf_ind['macd']['macd']

        # Get prior closed bar MACD
        try:
            prior_df   = ltf_df.iloc[-2:-1]
            prior_inds = self.indicators.calculate_all(ltf_df.iloc[:-1])
            prior_macd = float(prior_inds['macd']['macd'].iloc[-1])
        except Exception:
            return d

        if bias == 'bullish':
            direction = 'long'
            crossed   = prior_macd < 0 and current_macd > 0
        else:
            direction = 'short'
            crossed   = prior_macd > 0 and current_macd < 0

        if not crossed:
            return d

        signals = ['macd_zero_cross']

        if ltf_ind['supertrend']['trend'] == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('supertrend_aligned')

        rsi = ltf_ind['rsi']['value']
        if direction == 'long' and 40 < rsi < 70:
            signals.append('rsi_confirmation')
        elif direction == 'short' and 30 < rsi < 60:
            signals.append('rsi_confirmation')

        if ltf_ind['adx'].get('trend_strength') == 'strong':
            signals.append('adx_strong')

        ema20 = ltf_ind['ema'].get(20, 0)
        ema50 = ltf_ind['ema'].get(50, 0)
        if direction == 'long' and ema20 > ema50 > 0:
            signals.append('ema_stack_full')
        elif direction == 'short' and ema20 < ema50:
            signals.append('ema_stack_full')

        htf_trend = htf['trend']['direction']
        if htf_trend == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('price_structure_aligned')

        d.update({
            'signal': True, 'entry_signal': True,
            'direction': direction, 'confluence_signals': signals,
            'entry_reason': f'MACD zero-line cross {direction}',
        })
        return d

    def _check_momentum_breakout(
        self, htf, ltf, htf_df, ltf_df, bias
    ) -> Dict:
        """
        Momentum breakout: BB squeeze + MACD momentum + ADX strength.
        Market order.
        """
        d = self._empty_decision()
        ltf_ind   = ltf['indicators']
        direction = None

        if ltf_ind['bollinger']['squeeze']:
            d['confluence_signals'].append('bb_squeeze_active')

        if ltf_ind['macd']['bullish_cross']:
            d['confluence_signals'].append('macd_bullish_cross')
            direction = 'long'
        elif ltf_ind['macd']['histogram'] < 0:
            d['confluence_signals'].append('macd_bearish_cross')
            direction = 'short'
        else:
            return d

        if ltf_ind['adx'].get('trend_strength') == 'strong':
            d['confluence_signals'].append('adx_strong')

        if ltf_ind['supertrend']['trend'] == ('bullish' if direction == 'long' else 'bearish'):
            d['confluence_signals'].append('supertrend_aligned')

        d.update({
            'signal': True, 'entry_signal': True,
            'direction': direction,
            'entry_reason': f'Momentum breakout {direction}',
        })
        return d

    def _check_rsi_divergence(
        self, htf, ltf, htf_df, ltf_df, bias
    ) -> Dict:
        """
        RSI divergence: price makes a lower low but RSI makes a higher low
        (bullish), or price makes a higher high but RSI makes a lower high
        (bearish). Detected over the last 20 bars on the entry timeframe.
        Limit order at current price (structure entry).
        """
        d = self._empty_decision()
        if ltf_df is None or len(ltf_df) < 20:
            return d

        try:
            all_inds = self.indicators.calculate_all(ltf_df)
            rsi_series = all_inds['rsi']['value']
        except Exception:
            return d

        lookback = 20
        price_slice = ltf_df['close'].iloc[-lookback:]
        rsi_slice   = rsi_series.iloc[-lookback:]

        # Find two swing lows in price (bullish divergence check)
        divergence = None
        direction  = None

        if bias == 'bullish':
            # Look for price LL + RSI HL
            min_price_idx = price_slice.idxmin()
            min_price     = price_slice.min()
            prev_min_price = float(price_slice.iloc[:price_slice.index.get_loc(min_price_idx)].min()) \
                             if price_slice.index.get_loc(min_price_idx) > 1 else None

            if prev_min_price is not None and min_price < prev_min_price:
                # Price made LL — check RSI
                rsi_at_min  = float(rsi_slice.loc[min_price_idx]) if min_price_idx in rsi_slice.index else None
                rsi_prev    = float(rsi_slice.iloc[:rsi_slice.index.get_loc(min_price_idx)].min()) \
                              if rsi_at_min and rsi_slice.index.get_loc(min_price_idx) > 1 else None
                if rsi_at_min and rsi_prev and rsi_at_min > rsi_prev:
                    divergence = 'bullish'
                    direction  = 'long'

        elif bias == 'bearish':
            # Look for price HH + RSI LH
            max_price_idx = price_slice.idxmax()
            max_price     = price_slice.max()
            prev_max_price = float(price_slice.iloc[:price_slice.index.get_loc(max_price_idx)].max()) \
                             if price_slice.index.get_loc(max_price_idx) > 1 else None

            if prev_max_price is not None and max_price > prev_max_price:
                rsi_at_max = float(rsi_slice.loc[max_price_idx]) if max_price_idx in rsi_slice.index else None
                rsi_prev   = float(rsi_slice.iloc[:rsi_slice.index.get_loc(max_price_idx)].max()) \
                             if rsi_at_max and rsi_slice.index.get_loc(max_price_idx) > 1 else None
                if rsi_at_max and rsi_prev and rsi_at_max < rsi_prev:
                    divergence = 'bearish'
                    direction  = 'short'

        if not divergence:
            return d

        ltf_ind = ltf['indicators']
        signals = ['rsi_divergence_confirmed']

        if ltf_ind['supertrend']['trend'] == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('supertrend_aligned')

        htf_trend = htf['trend']['direction']
        if htf_trend == ('bullish' if direction == 'long' else 'bearish'):
            signals.append('price_structure_aligned')

        if ltf_ind['adx'].get('trend_strength') == 'strong':
            signals.append('adx_strong')

        entry_price = ltf['ohlc']['close']
        d.update({
            'signal': True, 'entry_signal': True,
            'direction': direction, 'confluence_signals': signals,
            'entry_reason': f'RSI {divergence} divergence {direction}',
            'limit_price': entry_price,
        })
        return d

    # ── Confluence scoring ────────────────────────────────────────────────────

    def _calculate_confluence_score(self, signals: List[str]) -> float:
        """Sum weights of all confluence signals."""
        return sum(_SIGNAL_WEIGHTS.get(s, 1) for s in signals)

    def _get_confluence_threshold(self, symbol: str) -> int:
        """
        Return confluence score threshold for this symbol.
        Priority: per-symbol config → volatile default → calm default.
        """
        sym_cfg = self.config.get('symbols', {}).get(symbol, {})
        if 'confluence_threshold' in sym_cfg:
            return int(sym_cfg['confluence_threshold'])

        mt5_sym = symbol.replace('/', '').upper()
        if mt5_sym in _VOLATILE_SYMBOLS or symbol in _VOLATILE_SYMBOLS:
            return int(self.strategy_config.get('confluence_required', 7))
        return int(self.strategy_config.get('confluence_threshold_calm', 5))

    # ── Filters ───────────────────────────────────────────────────────────────

    def _apply_filters(self, analysis: Dict, multi_tf_data: Dict) -> Dict:
        filters = self.strategy_config.get('filters', {})

        if filters.get('respect_higher_tf_bias', True):
            structure_tf = self.timeframe_config.get('structure_timeframe', '1H')
            if structure_tf in analysis['timeframe_snapshots']:
                htf_trend = analysis['timeframe_snapshots'][structure_tf]['trend']['direction']
                if analysis['direction'] == 'long' and htf_trend == 'bearish':
                    if not filters.get('allow_override', False):
                        analysis['entry_signal'] = False
                        analysis['entry_reason'] = 'Filtered: against HTF bearish trend'
                elif analysis['direction'] == 'short' and htf_trend == 'bullish':
                    if not filters.get('allow_override', False):
                        analysis['entry_signal'] = False
                        analysis['entry_reason'] = 'Filtered: against HTF bullish trend'

        entry_tf = self.timeframe_config.get('entry_timeframe', '5m')
        if entry_tf in analysis['timeframe_snapshots']:
            atr_pct = analysis['timeframe_snapshots'][entry_tf]['indicators']['atr']['percent']
            min_atr = filters.get('min_atr_threshold', 0)
            max_atr = filters.get('max_atr_threshold', 100)
            if atr_pct < min_atr * 100:
                analysis['entry_signal'] = False
                analysis['entry_reason'] = f'Filtered: ATR too low ({atr_pct:.2f}%)'
            elif atr_pct > max_atr * 100:
                analysis['entry_signal'] = False
                analysis['entry_reason'] = f'Filtered: ATR too high ({atr_pct:.2f}%)'

        if filters.get('news_blackout', {}).get('enabled', False):
            if self._is_news_blackout():
                analysis['entry_signal'] = False
                analysis['entry_reason'] = 'Filtered: news blackout active'

        return analysis

    def _is_news_blackout(self) -> bool:
        blackout_config = self.strategy_config.get('filters', {}).get('news_blackout', {})
        windows     = blackout_config.get('windows', [])
        now         = datetime.utcnow()
        current_time = now.time()
        current_day  = now.weekday() + 1  # Monday=1

        for window in windows:
            if current_day not in window.get('days', []):
                continue
            start = datetime.strptime(str(window['start']), '%H:%M').time()
            end   = datetime.strptime(str(window['end']),   '%H:%M').time()
            if start <= current_time <= end:
                return True
        return False

    # ── Stop / TP calculation ─────────────────────────────────────────────────

    def _calculate_stop_loss(
        self,
        analysis: Dict,
        entry_price: float,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        direction = analysis['direction']
        entry_tf  = self.timeframe_config.get('entry_timeframe', '5m')
        structure = analysis['timeframe_snapshots'].get(entry_tf, {}) \
                        .get('indicators', {}).get('price_structure', {})

        if direction == 'long':
            structure_stop = structure.get('last_swing_low', entry_price - 2 * atr)
        else:
            structure_stop = structure.get('last_swing_high', entry_price + 2 * atr)

        # Session-aware ATR multiplier — read from strategy config
        sess_cfg = self.strategy_config.get('session_atr_multipliers', {})
        hour     = datetime.utcnow().hour
        if 0 <= hour < 7:
            mult = float(sess_cfg.get('asian', 1.5))
        elif 7 <= hour < 12:
            mult = float(sess_cfg.get('london', 2.5))
        elif 12 <= hour < 20:
            mult = float(sess_cfg.get('ny', 2.5))
        else:
            mult = float(sess_cfg.get('overlap', 2.0))

        if direction == 'long':
            atr_stop = entry_price - atr * mult
            stop_loss = min(structure_stop, atr_stop)   # wider of the two
        else:
            atr_stop  = entry_price + atr * mult
            stop_loss = max(structure_stop, atr_stop)

        return stop_loss

    def _calculate_take_profits(
        self,
        analysis: Dict,
        entry_price: float,
        stop_loss: float,
        atr: float,
    ) -> Dict:
        risk      = abs(entry_price - stop_loss)
        direction = analysis['direction']

        raw_targets = self.config.get('risk_management', {}) \
                          .get('take_profit', {}).get('targets', [])
        real_targets = [t for t in raw_targets if float(t.get('rr_ratio', 999)) < 999]

        if not real_targets:
            real_targets = [
                {'rr_ratio': 2.0, 'close_percent': 50},
                {'rr_ratio': 4.0, 'close_percent': 30},
            ]

        tps = {}
        for i, target in enumerate(real_targets, 1):
            rr = float(target['rr_ratio'])
            if direction == 'long':
                tps[f'tp{i}'] = entry_price + risk * rr
            else:
                tps[f'tp{i}'] = entry_price - risk * rr
        return tps