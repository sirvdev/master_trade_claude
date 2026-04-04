"""
strategy/ict_engine.py — ICT (Inner Circle Trader) Concepts Strategy Engine
=============================================================================
Drop-in replacement for StrategyEngine. Implements ICT methodology:
  - Power of Three (Accumulation → Manipulation → Distribution)
  - Optimal Trade Entry (OTE) at 62-79% Fibonacci retracement
  - ICT Killzones (London/NY specific windows)
  - Judas Swing (fake move to sweep liquidity before real move)
  - Breaker Blocks, Mitigation Blocks
  - Asian Range as reference for London/NY expansion

Integration:
  In main.py, change:
    from strategy.engine import StrategyEngine
  to:
    from strategy.ict_engine import ICTStrategyEngine as StrategyEngine

  No other changes required — same analyze_market() / calculate_entry_levels() API.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from indicators.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# ── ICT Signal Weights ────────────────────────────────────────────────────────
_ICT_WEIGHTS: Dict[str, int] = {
    # Core ICT concepts (highest weight)
    'ote_zone_entry':          4,
    'judas_swing_detected':    4,
    'po3_phase_aligned':       4,
    'breaker_block_entry':     3,
    'asian_range_breakout':    3,
    # Structure
    'htf_pd_array_aligned':    3,
    'displacement_confirmed':  3,
    'market_structure_shift':  3,
    # Killzone & timing
    'killzone_active':         2,
    'ny_midnight_open_ref':    2,
    'london_session_bias':     2,
    # Supporting
    'premium_discount':        2,
    'institutional_candle':    2,
    'ema_bias_aligned':        1,
    'rsi_confirmation':        1,
    'volume_spike':            1,
}

# ICT Killzones (UTC) — more precise than generic SMC
_ICT_KILLZONES = {
    'london':    (2, 5),     # 02:00–05:00 UTC (London open)
    'ny_am':     (7, 10),    # 07:00–10:00 UTC (NY morning)
    'ny_lunch':  (11, 13),   # 11:00–13:00 UTC (NY lunch — avoid)
    'ny_pm':     (13, 15),   # 13:00–15:00 UTC (NY afternoon)
}

# Asian session for range reference
_ASIAN_SESSION = (0, 6)  # 00:00–06:00 UTC

_VOLATILE_SYMBOLS = {'XAUUSD','XAU/USD','BTCUSD','BTC/USD','NAS100USD','NAS100','US100'}


class ICTStrategyEngine:
    """
    ICT Concepts strategy engine.
    
    Core ICT methodology:
    1. Power of Three (PO3) — Accumulation, Manipulation, Distribution
    2. Optimal Trade Entry (OTE) — 62%-79% Fibonacci retracement of impulse
    3. Killzones — Trade only during institutional activity windows
    4. Judas Swing — Fake move (manipulation) that sweeps liquidity
    5. Asian Range — Reference range that London/NY expands from
    6. Breaker Blocks — Failed order blocks that become support/resistance
    7. Premium/Discount Arrays — Institutional reference points (FVG, OB, etc.)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.indicators = TechnicalIndicators(config.get('indicators', {}))
        self.strategy_config = config.get('strategy', {})
        self.ict_config = config.get('ict', {})

    def _tf_minutes(self, tf: str) -> int:
        m = {'1m':1,'5m':5,'15m':15,'30m':30,'1H':60,'4H':240,'1D':1440,
             'M1':1,'M5':5,'M15':15,'H1':60,'H4':240,'D1':1440}
        return m.get(tf, 60)

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — Same interface as StrategyEngine
    # ══════════════════════════════════════════════════════════════════════════

    def analyze_market(self, symbol: str, multi_tf_data: Dict[str, pd.DataFrame],
                       symbol_config: Optional[Dict] = None) -> Dict:
        if symbol_config:
            tfs = symbol_config.get('timeframes', [])
            structure_tf = tfs[0] if tfs else '1H'
            primary_tf = symbol_config.get('primary_timeframe', tfs[1] if len(tfs) > 1 else '15m')
            entry_tf = symbol_config.get('entry_timeframe', tfs[-1] if tfs else '5m')
        else:
            avail = sorted(multi_tf_data.keys(), key=lambda t: self._tf_minutes(t), reverse=True)
            structure_tf = avail[0] if avail else '1H'
            primary_tf = avail[1] if len(avail) > 1 else '15m'
            entry_tf = avail[-1] if avail else '5m'

        analysis = {
            'symbol': symbol, 'timestamp': datetime.now().replace(tzinfo=None),
            'primary_timeframe': primary_tf, 'structure_tf': structure_tf,
            'entry_tf': entry_tf, 'timeframe_snapshots': {},
            'market_structure': {}, 'indicators_state': {},
            'entry_signal': False, 'entry_reason': None, 'entry_type': None,
            'confidence_score': 0.0, 'confluence_score': 0.0,
            'confluence_signals': [], 'direction': None,
            'order_type': 'limit', 'limit_price': None,
        }

        try:
            for tf, df in multi_tf_data.items():
                if df is None or len(df) < 50:
                    continue
                analysis['timeframe_snapshots'][tf] = self._build_ict_snapshot(df, tf)

            snaps = analysis['timeframe_snapshots']
            if not snaps:
                return analysis

            if structure_tf in snaps:
                analysis['market_structure'] = snaps[structure_tf].get('structure', {})

            # ── ICT Killzone gate — only trade during institutional hours ─
            if not self._in_killzone():
                analysis['entry_reason'] = 'ICT: Outside killzone — no trade'
                logger.debug(f"[ICT] {symbol}: Outside killzone, skipping")
                return analysis

            decision = self._evaluate_ict_entry(
                analysis, multi_tf_data, snaps,
                structure_tf, primary_tf, entry_tf
            )
            analysis.update(decision)

            if analysis['entry_signal']:
                analysis = self._apply_ict_filters(analysis, snaps, structure_tf)

            logger.info(
                f"[ICT] {symbol} signal={analysis['entry_signal']} "
                f"dir={analysis['direction']} type={analysis.get('entry_type')} "
                f"score={analysis.get('confluence_score',0):.1f}")

        except Exception as e:
            logger.error(f"[ICT] Error analyzing {symbol}: {e}", exc_info=True)

        return analysis

    def calculate_entry_levels(self, analysis: Dict, multi_tf_data: Dict) -> Dict:
        """Same interface as StrategyEngine.calculate_entry_levels()."""
        entry_tf = analysis.get('entry_tf', '5m')
        if entry_tf not in multi_tf_data:
            avail = sorted(multi_tf_data.keys(), key=lambda t: self._tf_minutes(t))
            entry_tf = avail[0] if avail else next(iter(multi_tf_data), None)
        if not entry_tf or entry_tf not in multi_tf_data:
            return {}

        df = multi_tf_data[entry_tf]
        current_price = float(df['close'].iloc[-1])
        atr = self.indicators.calculate_atr(df)['current']
        order_type = analysis.get('order_type', 'limit')
        limit_price = analysis.get('limit_price')
        entry_price = float(limit_price) if (order_type == 'limit' and limit_price) else current_price

        stop_loss = self._calc_ict_sl(analysis, entry_price, atr, df)
        risk = abs(entry_price - stop_loss)

        tp_config = self.config.get('risk_management', {}).get('take_profit', {})
        targets = tp_config.get('targets', [
            {'name': 'TP1', 'rr_ratio': 2.0, 'close_percent': 40},
            {'name': 'TP2', 'rr_ratio': 3.5, 'close_percent': 35},
            {'name': 'TP3', 'rr_ratio': 5.0, 'close_percent': 25},
        ])

        direction = analysis.get('direction', 'long')
        tps = {}

        # ICT prefers targeting liquidity pools for TP
        struct = analysis.get('market_structure', {})
        if direction == 'long':
            # Target buy-side liquidity (swing highs)
            liq_target = struct.get('last_swing_high')
            if liq_target and liq_target > entry_price:
                tps['tp1'] = liq_target  # First target = opposing liquidity
            else:
                tps['tp1'] = entry_price + risk * targets[0].get('rr_ratio', 2.0)
        else:
            liq_target = struct.get('last_swing_low')
            if liq_target and liq_target < entry_price:
                tps['tp1'] = liq_target
            else:
                tps['tp1'] = entry_price - risk * targets[0].get('rr_ratio', 2.0)

        # TP2 and TP3 use R:R multiples
        for i, t in enumerate(targets[1:3], 2):
            rr = t.get('rr_ratio', 3.0)
            if direction == 'long':
                tps[f'tp{i}'] = entry_price + risk * rr
            else:
                tps[f'tp{i}'] = entry_price - risk * rr

        return {
            'entry_price': current_price,
            'order_price': entry_price,
            'order_type': order_type,
            'limit_price': entry_price if order_type == 'limit' else None,
            'stop_loss': stop_loss,
            'take_profit_1': tps.get('tp1'),
            'take_profit_2': tps.get('tp2'),
            'take_profit_3': tps.get('tp3'),
            'atr': atr,
            'risk_distance': risk,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # ICT CORE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ict_snapshot(self, df: pd.DataFrame, tf: str) -> Dict:
        """Build ICT-specific analysis snapshot."""
        ind = self.indicators.calculate_all(df)

        snap = {
            'ohlc': {k: float(df[k].iloc[-1]) for k in ['open','high','low','close','volume']},
            'indicators': {
                'ema': {p: float(v.iloc[-1]) for p, v in ind['ema'].items()},
                'rsi': {'value': float(ind['rsi']['value'].iloc[-1]),
                        'overbought': bool(ind['rsi']['is_overbought']),
                        'oversold': bool(ind['rsi']['is_oversold'])},
                'atr': {'value': float(ind['atr']['current']),
                        'percent': float(ind['atr']['percent_of_price'])},
                'adx': ind['adx'],
            },
        }

        # Swing points
        swing_highs, swing_lows = self._detect_swing_points(df)
        snap['swing_highs'] = swing_highs
        snap['swing_lows'] = swing_lows

        # ICT structure
        snap['structure'] = self._analyze_ict_structure(df, swing_highs, swing_lows)

        # ICT-specific concepts
        snap['asian_range'] = self._detect_asian_range(df, tf)
        snap['fvgs'] = self._detect_fvgs(df)
        snap['order_blocks'] = self._detect_order_blocks(df, ind['atr']['current'])
        snap['breaker_blocks'] = self._detect_breaker_blocks(df, swing_highs, swing_lows,
                                                              ind['atr']['current'])
        snap['ote_zone'] = self._calculate_ote_zone(swing_highs, swing_lows, df)
        snap['judas_swing'] = self._detect_judas_swing(df, snap['asian_range'], ind['atr']['current'])
        snap['displacement'] = self._detect_displacement(df, ind['atr']['current'])
        snap['institutional_candles'] = self._detect_institutional_candles(df, ind['atr']['current'])

        return snap

    # ── Swing Points ───────────────────────────────────────────────────────────

    def _detect_swing_points(self, df: pd.DataFrame, order: int = 5
                             ) -> Tuple[List, List]:
        highs, lows = [], []
        h, l = df['high'].values, df['low'].values
        n = len(df)
        for i in range(order, n - order):
            if all(h[i] > h[i-j] for j in range(1, order+1)) and \
               all(h[i] > h[i+j] for j in range(1, order+1)):
                highs.append((i, float(h[i])))
            if all(l[i] < l[i-j] for j in range(1, order+1)) and \
               all(l[i] < l[i+j] for j in range(1, order+1)):
                lows.append((i, float(l[i])))
        return highs, lows

    # ── ICT Market Structure ───────────────────────────────────────────────────

    def _analyze_ict_structure(self, df, swing_highs, swing_lows) -> Dict:
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'trend': 'neutral', 'bias': None, 'mss': False,
                    'last_swing_high': None, 'last_swing_low': None}

        sh = sorted(swing_highs, key=lambda x: x[0])
        sl = sorted(swing_lows, key=lambda x: x[0])

        last_sh, prev_sh = sh[-1][1], sh[-2][1]
        last_sl, prev_sl = sl[-1][1], sl[-2][1]
        cp = float(df['close'].iloc[-1])

        hh = last_sh > prev_sh
        hl = last_sl > prev_sl
        lh = last_sh < prev_sh
        ll = last_sl < prev_sl

        trend = 'neutral'
        bias = None
        mss = False  # Market Structure Shift (ICT term for CHoCH)

        if hh and hl:
            trend, bias = 'bullish', 'bullish'
        elif lh and ll:
            trend, bias = 'bearish', 'bearish'
        elif hh and ll:
            # Potential MSS
            if cp < prev_sl:
                mss, trend, bias = True, 'bearish', 'bearish'
        elif lh and hl:
            if cp > prev_sh:
                mss, trend, bias = True, 'bullish', 'bullish'

        return {
            'trend': trend, 'bias': bias, 'mss': mss,
            'last_swing_high': last_sh, 'last_swing_low': last_sl,
            'prev_swing_high': prev_sh, 'prev_swing_low': prev_sl,
            'hh': hh, 'hl': hl, 'lh': lh, 'll': ll,
        }

    # ── Asian Range ────────────────────────────────────────────────────────────

    def _detect_asian_range(self, df: pd.DataFrame, tf: str) -> Dict:
        """
        Detect the Asian session range (00:00–06:00 UTC).
        ICT uses this as the manipulation reference — London/NY will break it.
        """
        if not hasattr(df.index, 'hour'):
            return {'high': None, 'low': None, 'range': None, 'broken': None}

        try:
            asian_mask = (df.index.hour >= _ASIAN_SESSION[0]) & (df.index.hour < _ASIAN_SESSION[1])
            # Get today's Asian range (most recent Asian session)
            asian_bars = df[asian_mask]

            if len(asian_bars) < 2:
                return {'high': None, 'low': None, 'range': None, 'broken': None}

            # Use only the last Asian session's bars
            last_asian_date = asian_bars.index[-1].date()
            today_asian = asian_bars[asian_bars.index.date == last_asian_date]

            if len(today_asian) < 2:
                return {'high': None, 'low': None, 'range': None, 'broken': None}

            ar_high = float(today_asian['high'].max())
            ar_low = float(today_asian['low'].min())
            ar_range = ar_high - ar_low
            cp = float(df['close'].iloc[-1])

            broken = None
            if cp > ar_high:
                broken = 'above'
            elif cp < ar_low:
                broken = 'below'

            return {
                'high': ar_high, 'low': ar_low,
                'range': ar_range, 'mid': (ar_high + ar_low) / 2,
                'broken': broken,
            }
        except Exception:
            return {'high': None, 'low': None, 'range': None, 'broken': None}

    # ── Fair Value Gaps ────────────────────────────────────────────────────────

    def _detect_fvgs(self, df: pd.DataFrame) -> List[Dict]:
        fvgs = []
        h, l = df['high'].values, df['low'].values
        n = len(df)
        for i in range(1, n - 1):
            if l[i+1] > h[i-1]:  # Bullish FVG
                fvgs.append({
                    'type': 'bullish',
                    'top': float(l[i+1]), 'bottom': float(h[i-1]),
                    'ce': float((l[i+1] + h[i-1]) / 2),  # Consequent Encroachment
                    'index': i,
                    'filled': float(df['low'].iloc[i+1:].min()) <= float(h[i-1]) if i+2 < n else False,
                })
            if h[i+1] < l[i-1]:  # Bearish FVG
                fvgs.append({
                    'type': 'bearish',
                    'top': float(l[i-1]), 'bottom': float(h[i+1]),
                    'ce': float((l[i-1] + h[i+1]) / 2),
                    'index': i,
                    'filled': float(df['high'].iloc[i+1:].max()) >= float(l[i-1]) if i+2 < n else False,
                })
        return [f for f in fvgs if f['index'] >= n - 25 and not f['filled']][-5:]

    # ── Order Blocks ───────────────────────────────────────────────────────────

    def _detect_order_blocks(self, df: pd.DataFrame, atr: float) -> List[Dict]:
        obs = []
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        n = len(df)

        for i in range(2, n - 2):
            body = abs(c[i] - o[i])
            if body < 0.3 * atr:
                continue  # Skip tiny candles

            # Bullish OB: bearish candle followed by strong bullish move
            if c[i] < o[i]:  # Bearish candle
                future_high = max(h[i+1:min(i+4, n)]) if i+1 < n else 0
                if future_high - l[i] > 2.0 * atr:
                    obs.append({
                        'type': 'bullish', 'top': float(max(o[i], c[i])),
                        'bottom': float(min(o[i], c[i])),
                        'index': i, 'strength': (future_high - l[i]) / atr,
                    })

            # Bearish OB: bullish candle followed by strong bearish move
            if c[i] > o[i]:
                future_low = min(l[i+1:min(i+4, n)]) if i+1 < n else 0
                if h[i] - future_low > 2.0 * atr:
                    obs.append({
                        'type': 'bearish', 'top': float(max(o[i], c[i])),
                        'bottom': float(min(o[i], c[i])),
                        'index': i, 'strength': (h[i] - future_low) / atr,
                    })

        return obs[-6:]  # Last 6

    # ── Breaker Blocks ─────────────────────────────────────────────────────────

    def _detect_breaker_blocks(self, df: pd.DataFrame,
                                swing_highs: List, swing_lows: List,
                                atr: float) -> List[Dict]:
        """
        Breaker Block: A failed order block. When an OB fails to hold and price
        trades through it, the OB flips polarity and becomes a breaker.
        Bullish OB that gets broken to the downside → becomes a bearish breaker.
        """
        breakers = []
        obs = self._detect_order_blocks(df, atr)
        cp = float(df['close'].iloc[-1])

        for ob in obs:
            if ob['type'] == 'bullish' and cp < ob['bottom']:
                # Bullish OB failed → bearish breaker
                breakers.append({
                    'type': 'bearish_breaker',
                    'top': ob['top'], 'bottom': ob['bottom'],
                    'index': ob['index'],
                })
            elif ob['type'] == 'bearish' and cp > ob['top']:
                # Bearish OB failed → bullish breaker
                breakers.append({
                    'type': 'bullish_breaker',
                    'top': ob['top'], 'bottom': ob['bottom'],
                    'index': ob['index'],
                })

        return breakers[-3:]

    # ── OTE Zone (Optimal Trade Entry) ─────────────────────────────────────────

    def _calculate_ote_zone(self, swing_highs: List, swing_lows: List,
                            df: pd.DataFrame) -> Dict:
        """
        ICT OTE = 62%–79% Fibonacci retracement of the most recent impulse leg.
        This is the "sweet spot" where institutions re-enter.
        """
        if not swing_highs or not swing_lows:
            return {'active': False}

        sh = sorted(swing_highs, key=lambda x: x[0])
        sl = sorted(swing_lows, key=lambda x: x[0])

        # Determine the most recent impulse
        last_sh = sh[-1]
        last_sl = sl[-1]

        if last_sh[0] > last_sl[0]:
            # Last structure point is a high → impulse was UP
            # OTE for longs = 62-79% retracement of this up-move
            impulse_high = last_sh[1]
            impulse_low = last_sl[1]
            direction = 'long'
        else:
            # Last structure point is a low → impulse was DOWN
            impulse_high = last_sh[1]
            impulse_low = last_sl[1]
            direction = 'short'

        if impulse_high == impulse_low:
            return {'active': False}

        range_size = impulse_high - impulse_low

        # Fibonacci levels
        fib_618 = impulse_high - range_size * 0.618
        fib_705 = impulse_high - range_size * 0.705  # ICT "sweet spot"
        fib_786 = impulse_high - range_size * 0.786

        cp = float(df['close'].iloc[-1])

        if direction == 'long':
            ote_top = fib_618
            ote_bottom = fib_786
            in_zone = ote_bottom <= cp <= ote_top
        else:
            ote_top = impulse_low + range_size * 0.618
            ote_bottom = impulse_low + range_size * 0.786
            in_zone = ote_bottom >= cp >= ote_top  # Inverted for shorts

        return {
            'active': True,
            'direction': direction,
            'ote_top': ote_top,
            'ote_bottom': ote_bottom,
            'fib_618': fib_618,
            'fib_705': fib_705,
            'fib_786': fib_786,
            'in_zone': in_zone,
            'impulse_high': impulse_high,
            'impulse_low': impulse_low,
        }

    # ── Judas Swing ────────────────────────────────────────────────────────────

    def _detect_judas_swing(self, df: pd.DataFrame, asian_range: Dict,
                            atr: float) -> Dict:
        """
        Judas Swing: A false move at the start of London or NY session that
        sweeps Asian range liquidity, then reverses.
        
        Bullish Judas: Price dips below Asian low (sweeps sell stops), then reverses up.
        Bearish Judas: Price spikes above Asian high (sweeps buy stops), then reverses down.
        """
        if not asian_range.get('high') or not asian_range.get('low'):
            return {'detected': False}

        ar_high = asian_range['high']
        ar_low = asian_range['low']

        if not hasattr(df.index, 'hour'):
            return {'detected': False}

        # Check recent bars (last 5) during London/NY open
        try:
            recent = df.iloc[-5:]
            hour = datetime.utcnow().hour

            if not (2 <= hour <= 10):  # Only during London/NY open
                return {'detected': False}

            recent_low = float(recent['low'].min())
            recent_high = float(recent['high'].max())
            cp = float(df['close'].iloc[-1])

            # Bullish Judas: swept below Asian low, now closing above
            if recent_low < ar_low and cp > ar_low:
                sweep_depth = ar_low - recent_low
                if sweep_depth > 0.3 * atr:  # Meaningful sweep
                    return {
                        'detected': True,
                        'type': 'bullish',
                        'sweep_price': recent_low,
                        'recovery_price': cp,
                        'depth_atr': sweep_depth / atr,
                    }

            # Bearish Judas: swept above Asian high, now closing below
            if recent_high > ar_high and cp < ar_high:
                sweep_depth = recent_high - ar_high
                if sweep_depth > 0.3 * atr:
                    return {
                        'detected': True,
                        'type': 'bearish',
                        'sweep_price': recent_high,
                        'recovery_price': cp,
                        'depth_atr': sweep_depth / atr,
                    }
        except Exception:
            pass

        return {'detected': False}

    # ── Displacement & Institutional Candles ───────────────────────────────────

    def _detect_displacement(self, df: pd.DataFrame, atr: float) -> Dict:
        bodies = abs(df['close'] - df['open'])
        last_body = float(bodies.iloc[-1])
        direction = 'bullish' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'bearish'
        return {
            'detected': last_body > 1.5 * atr,
            'direction': direction,
            'ratio': last_body / atr if atr > 0 else 0,
        }

    def _detect_institutional_candles(self, df: pd.DataFrame, atr: float) -> List[Dict]:
        """Large-body candles with small wicks = institutional momentum."""
        candles = []
        for i in range(-3, 0):  # Last 3 bars
            try:
                body = abs(float(df['close'].iloc[i]) - float(df['open'].iloc[i]))
                upper_wick = float(df['high'].iloc[i]) - max(float(df['close'].iloc[i]),
                                                              float(df['open'].iloc[i]))
                lower_wick = min(float(df['close'].iloc[i]),
                                 float(df['open'].iloc[i])) - float(df['low'].iloc[i])
                total_wick = upper_wick + lower_wick
                full_range = float(df['high'].iloc[i]) - float(df['low'].iloc[i])

                if full_range > 0 and body / full_range > 0.7 and body > 1.2 * atr:
                    direction = 'bullish' if df['close'].iloc[i] > df['open'].iloc[i] else 'bearish'
                    candles.append({
                        'index': len(df) + i,
                        'direction': direction,
                        'body_ratio': body / full_range,
                        'body_atr': body / atr,
                    })
            except (IndexError, ValueError):
                continue
        return candles

    # ── Killzone Check ─────────────────────────────────────────────────────────

    def _in_killzone(self) -> bool:
        """ICT only trades during institutional killzones."""
        hour = datetime.utcnow().hour
        for kz_name, (start, end) in _ICT_KILLZONES.items():
            if kz_name == 'ny_lunch':
                continue  # Skip lunch hour — ICT avoids this
            if start <= hour < end:
                return True
        return False

    def _get_active_killzone(self) -> Optional[str]:
        hour = datetime.utcnow().hour
        for kz_name, (start, end) in _ICT_KILLZONES.items():
            if start <= hour < end:
                return kz_name
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY EVALUATION
    # ══════════════════════════════════════════════════════════════════════════

    def _evaluate_ict_entry(self, analysis, multi_tf_data, snaps,
                            structure_tf, primary_tf, entry_tf) -> Dict:
        out = {
            'entry_signal': False, 'entry_reason': None, 'entry_type': None,
            'direction': None, 'confidence_score': 0.0, 'confluence_score': 0.0,
            'confluence_signals': [], 'order_type': 'limit', 'limit_price': None,
        }

        htf = snaps.get(structure_tf)
        ptf = snaps.get(primary_tf, htf)
        ltf = snaps.get(entry_tf, ptf)

        if not htf or not ptf:
            return out

        htf_bias = htf.get('structure', {}).get('bias')
        if not htf_bias:
            return out

        candidates = []

        # ── Setup 1: OTE Entry (ICT flagship setup) ──────────────────────
        ote = self._check_ote_entry(htf_bias, ptf, ltf)
        if ote['signal']:
            candidates.append((self._ict_score(ote['signals']), 'ict_ote', ote))

        # ── Setup 2: Judas Swing Entry ───────────────────────────────────
        judas = self._check_judas_entry(htf_bias, ptf, ltf)
        if judas['signal']:
            candidates.append((self._ict_score(judas['signals']), 'ict_judas_swing', judas))

        # ── Setup 3: Asian Range Breakout ─────────────────────────────────
        ar = self._check_asian_range_entry(htf_bias, ptf, ltf)
        if ar['signal']:
            candidates.append((self._ict_score(ar['signals']), 'ict_asian_breakout', ar))

        # ── Setup 4: Breaker Block Entry ──────────────────────────────────
        brk = self._check_breaker_entry(htf_bias, ptf, ltf)
        if brk['signal']:
            candidates.append((self._ict_score(brk['signals']), 'ict_breaker', brk))

        # ── Setup 5: MSS + OB/FVG ────────────────────────────────────────
        mss = self._check_mss_entry(htf_bias, ptf, ltf)
        if mss['signal']:
            candidates.append((self._ict_score(mss['signals']), 'ict_mss', mss))

        if not candidates:
            return out

        threshold = self._get_threshold(analysis.get('symbol', ''))
        candidates = [(s, t, r) for s, t, r in candidates if s >= threshold]
        if not candidates:
            return out

        candidates.sort(reverse=True, key=lambda x: x[0])
        score, etype, result = candidates[0]

        out.update({
            'entry_signal': True,
            'entry_reason': result.get('reason', etype),
            'entry_type': etype,
            'direction': result['direction'],
            'confidence_score': min(1.0, score / max(threshold * 1.5, 10)),
            'confluence_score': score,
            'confluence_signals': result['signals'],
            'order_type': result.get('order_type', 'limit'),
            'limit_price': result.get('limit_price'),
        })
        return out

    # ── Entry Setups ───────────────────────────────────────────────────────────

    def _check_ote_entry(self, htf_bias, ptf, ltf) -> Dict:
        """
        ICT Optimal Trade Entry: Price is in the 62-79% Fibonacci retracement
        zone (OTE) of the last impulse, aligned with HTF bias.
        """
        ote = ptf.get('ote_zone', {})
        if not ote.get('active') or not ote.get('in_zone'):
            return {'signal': False}

        # OTE direction must align with HTF bias
        if htf_bias == 'bullish' and ote['direction'] != 'long':
            return {'signal': False}
        if htf_bias == 'bearish' and ote['direction'] != 'short':
            return {'signal': False}

        direction = 'long' if htf_bias == 'bullish' else 'short'
        signals = ['ote_zone_entry']

        # Look for an FVG or OB within the OTE zone for precision entry
        fvgs = ptf.get('fvgs', [])
        obs = ptf.get('order_blocks', [])
        limit_price = ote['fib_705']  # Default: 70.5% = ICT sweet spot

        target_type = 'bullish' if direction == 'long' else 'bearish'
        for fvg in fvgs:
            if fvg['type'] == target_type:
                if ote['ote_bottom'] <= fvg['ce'] <= ote['ote_top']:
                    limit_price = fvg['ce']
                    signals.append('htf_pd_array_aligned')
                    break

        self._add_ict_confluence(signals, htf_bias, ptf, ltf)

        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': limit_price,
            'order_type': 'limit',
            'reason': f'ICT OTE {direction} @ fib 70.5%',
        }

    def _check_judas_entry(self, htf_bias, ptf, ltf) -> Dict:
        """Judas Swing: fake move sweeps Asian range, then reverses."""
        judas = ptf.get('judas_swing', {})
        if not judas.get('detected'):
            return {'signal': False}

        if htf_bias == 'bullish' and judas['type'] != 'bullish':
            return {'signal': False}
        if htf_bias == 'bearish' and judas['type'] != 'bearish':
            return {'signal': False}

        direction = 'long' if judas['type'] == 'bullish' else 'short'
        signals = ['judas_swing_detected', 'po3_phase_aligned']

        # Entry at Asian range mid or at an OB/FVG
        ar = ptf.get('asian_range', {})
        limit_price = ar.get('mid', ltf['ohlc']['close'])

        self._add_ict_confluence(signals, htf_bias, ptf, ltf)

        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': limit_price,
            'order_type': 'market',  # After Judas = market entry
            'reason': f'ICT Judas Swing {direction}',
        }

    def _check_asian_range_entry(self, htf_bias, ptf, ltf) -> Dict:
        """Asian range breakout aligned with HTF bias + displacement."""
        ar = ptf.get('asian_range', {})
        disp = ptf.get('displacement', {})

        if not ar.get('broken'):
            return {'signal': False}

        if htf_bias == 'bullish' and ar['broken'] != 'above':
            return {'signal': False}
        if htf_bias == 'bearish' and ar['broken'] != 'below':
            return {'signal': False}

        if not disp.get('detected'):
            return {'signal': False}  # Need displacement to confirm

        direction = 'long' if htf_bias == 'bullish' else 'short'
        signals = ['asian_range_breakout', 'displacement_confirmed']

        # Enter on pullback to Asian range boundary
        if direction == 'long':
            limit_price = ar['high']  # Pullback to Asian high
        else:
            limit_price = ar['low']

        self._add_ict_confluence(signals, htf_bias, ptf, ltf)

        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': limit_price,
            'order_type': 'limit',
            'reason': f'ICT Asian Range breakout {direction}',
        }

    def _check_breaker_entry(self, htf_bias, ptf, ltf) -> Dict:
        """Enter at a breaker block (failed OB that flipped polarity)."""
        breakers = ptf.get('breaker_blocks', [])
        cp = ltf['ohlc']['close']
        atr = ptf['indicators']['atr']['value']

        target = 'bullish_breaker' if htf_bias == 'bullish' else 'bearish_breaker'
        valid = [b for b in breakers if b['type'] == target]

        if not valid:
            return {'signal': False}

        # Find closest breaker
        best = min(valid, key=lambda b: abs(cp - (b['top'] + b['bottom']) / 2))
        dist = abs(cp - (best['top'] + best['bottom']) / 2)

        if dist > 2.0 * atr:
            return {'signal': False}

        direction = 'long' if htf_bias == 'bullish' else 'short'
        signals = ['breaker_block_entry']

        limit_price = (best['top'] + best['bottom']) / 2

        self._add_ict_confluence(signals, htf_bias, ptf, ltf)

        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': limit_price,
            'order_type': 'limit',
            'reason': f'ICT Breaker Block {direction} @ {limit_price:.2f}',
        }

    def _check_mss_entry(self, htf_bias, ptf, ltf) -> Dict:
        """Market Structure Shift + enter at OB/FVG."""
        struct = ptf.get('structure', {})
        if not struct.get('mss'):
            return {'signal': False}

        if struct.get('bias') != htf_bias:
            return {'signal': False}

        direction = 'long' if htf_bias == 'bullish' else 'short'
        signals = ['market_structure_shift']

        # Find entry point at nearest OB or FVG
        cp = ltf['ohlc']['close']
        limit_price = cp

        target_type = 'bullish' if direction == 'long' else 'bearish'
        obs = ptf.get('order_blocks', [])
        for ob in reversed(obs):
            if ob['type'] == target_type:
                limit_price = (ob['top'] + ob['bottom']) / 2
                signals.append('htf_pd_array_aligned')
                break

        self._add_ict_confluence(signals, htf_bias, ptf, ltf)

        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': limit_price,
            'order_type': 'limit',
            'reason': f'ICT MSS + PD Array {direction}',
        }

    # ── Confluence ─────────────────────────────────────────────────────────────

    def _add_ict_confluence(self, signals, htf_bias, ptf, ltf):
        ind = ptf['indicators']

        # Killzone
        kz = self._get_active_killzone()
        if kz and kz != 'ny_lunch':
            signals.append('killzone_active')

        # Premium/Discount
        struct = ptf.get('structure', {})
        sh = struct.get('last_swing_high')
        sl_price = struct.get('last_swing_low')
        cp = ltf['ohlc']['close']
        if sh and sl_price and sh != sl_price:
            mid = (sh + sl_price) / 2
            if htf_bias == 'bullish' and cp < mid:
                signals.append('premium_discount')
            elif htf_bias == 'bearish' and cp > mid:
                signals.append('premium_discount')

        # EMA alignment
        e20, e50 = ind['ema'].get(20, 0), ind['ema'].get(50, 0)
        if htf_bias == 'bullish' and e20 > e50:
            signals.append('ema_bias_aligned')
        elif htf_bias == 'bearish' and e20 < e50:
            signals.append('ema_bias_aligned')

        # RSI
        rsi = ind['rsi']['value']
        if 30 < rsi < 70:
            signals.append('rsi_confirmation')

        # Institutional candles
        inst = ptf.get('institutional_candles', [])
        if inst:
            target_dir = 'bullish' if htf_bias == 'bullish' else 'bearish'
            if any(c['direction'] == target_dir for c in inst):
                signals.append('institutional_candle')

        # Volume spike (if available)
        vol = ltf['ohlc'].get('volume', 0)
        if vol > 0:
            signals.append('volume_spike')

    # ── Scoring & Filtering ────────────────────────────────────────────────────

    def _ict_score(self, signals: List[str]) -> float:
        return sum(_ICT_WEIGHTS.get(s, 1) for s in signals)

    def _get_threshold(self, symbol: str) -> int:
        sc = self.config.get('symbols', {}).get(symbol, {})
        if 'confluence_threshold' in sc:
            return int(sc['confluence_threshold'])
        ms = symbol.replace('/', '').upper()
        if ms in _VOLATILE_SYMBOLS:
            return int(self.ict_config.get('confluence_required',
                       self.strategy_config.get('confluence_required', 7)))
        return int(self.ict_config.get('confluence_threshold_calm',
                   self.strategy_config.get('confluence_threshold_calm', 5)))

    def _apply_ict_filters(self, analysis, snaps, structure_tf) -> Dict:
        htf = snaps.get(structure_tf)
        if not htf:
            return analysis

        # ADX filter — no trading in chop
        adx = htf['indicators']['adx']
        adx_val = adx.get('value')
        if isinstance(adx_val, pd.Series):
            adx_val = float(adx_val.iloc[-1])
        if adx_val is not None and adx_val < 18:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'ICT Filter: ADX too low ({adx_val:.1f})'
            return analysis

        # NY Lunch filter — ICT avoids 11:00–13:00 UTC
        hour = datetime.utcnow().hour
        if 11 <= hour < 13:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = 'ICT Filter: NY Lunch — no trade'
            return analysis

        return analysis

    # ── Stop Loss ──────────────────────────────────────────────────────────────

    def _calc_ict_sl(self, analysis, entry_price, atr, df) -> float:
        """ICT SL: Beyond the liquidity that was swept, or beyond structure."""
        struct = analysis.get('market_structure', {})
        direction = analysis.get('direction', 'long')
        buffer = atr * 0.2

        if direction == 'long':
            sl = struct.get('last_swing_low')
            if sl:
                return sl - buffer
            return entry_price - 1.5 * atr
        else:
            sh = struct.get('last_swing_high')
            if sh:
                return sh + buffer
            return entry_price + 1.5 * atr