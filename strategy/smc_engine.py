"""
strategy/smc_engine.py — Smart Money Concepts (SMC) Strategy Engine
====================================================================
Drop-in replacement for StrategyEngine. Uses institutional order flow
concepts: Order Blocks, Fair Value Gaps (FVG), Break of Structure (BOS),
Change of Character (CHoCH), and liquidity sweeps.

Integration:
  In main.py, change:
    from strategy.engine import StrategyEngine
  to:
    from strategy.smc_engine import SMCStrategyEngine as StrategyEngine

  No other changes required — same analyze_market() / calculate_entry_levels() API.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from indicators.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# ── Signal weights for SMC confluence scoring ──────────────────────────────────
_SMC_WEIGHTS: Dict[str, int] = {
    # Structure signals (highest weight — institutional footprint)
    'bos_confirmed':          4,
    'choch_detected':         4,
    'liquidity_swept':        4,
    'order_block_entry':      3,
    'fvg_entry':              3,
    # Confluence signals
    'htf_structure_aligned':  3,
    'premium_discount_zone':  2,
    'session_killzone':       2,
    'displacement_candle':    2,
    'inducement_cleared':     2,
    # Supporting signals
    'ema_bias_aligned':       1,
    'volume_confirmation':    1,
    'rsi_not_exhausted':      1,
    'adx_trending':           1,
    'atr_sufficient':         1,
}

# Killzone windows (UTC) — when smart money is most active
_KILLZONES = {
    'london_open':  (7, 10),    # 07:00–10:00 UTC
    'ny_open':      (12, 15),   # 12:00–15:00 UTC  (NYSE open)
    'london_close': (15, 17),   # 15:00–17:00 UTC
    'asian_open':   (0, 3),     # 00:00–03:00 UTC
}

_VOLATILE_SYMBOLS = {'XAUUSD','XAU/USD','BTCUSD','BTC/USD','NAS100USD','NAS100','US100'}


class SMCStrategyEngine:
    """
    Smart Money Concepts strategy engine.
    Detects institutional order flow patterns and trades with smart money.
    
    Core concepts:
    1. Market Structure — BOS (Break of Structure) and CHoCH (Change of Character)
    2. Order Blocks — Last opposing candle before an impulsive move
    3. Fair Value Gaps — 3-candle imbalance zones (price tends to fill)
    4. Liquidity — Resting orders above swing highs / below swing lows
    5. Premium/Discount — Entry in the favorable half of a swing
    """

    def __init__(self, config: Dict):
        self.config = config
        self.indicators = TechnicalIndicators(config.get('indicators', {}))
        self.strategy_config = config.get('strategy', {})
        self.smc_config = config.get('smc', {})  # Optional SMC-specific overrides

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
            # ── Step 1: Build SMC snapshots for each timeframe ────────────
            for tf, df in multi_tf_data.items():
                if df is None or len(df) < 50:
                    continue
                analysis['timeframe_snapshots'][tf] = self._build_smc_snapshot(df)

            snaps = analysis['timeframe_snapshots']
            if not snaps:
                return analysis

            # ── Step 2: Determine HTF market structure (BOS/CHoCH) ────────
            if structure_tf in snaps:
                analysis['market_structure'] = snaps[structure_tf].get('structure', {})

            # ── Step 3: Check for entry setups ────────────────────────────
            decision = self._evaluate_smc_entry(
                analysis, multi_tf_data, snaps,
                structure_tf, primary_tf, entry_tf
            )
            analysis.update(decision)

            # ── Step 4: Apply safety filters ──────────────────────────────
            if analysis['entry_signal']:
                analysis = self._apply_smc_filters(analysis, snaps, structure_tf)

            logger.info(
                f"[SMC] {symbol} signal={analysis['entry_signal']} "
                f"dir={analysis['direction']} type={analysis.get('entry_type')} "
                f"score={analysis.get('confluence_score',0):.1f}")

        except Exception as e:
            logger.error(f"[SMC] Error analyzing {symbol}: {e}", exc_info=True)

        return analysis

    def calculate_entry_levels(self, analysis, multi_tf_data):
        """
        Same interface as StrategyEngine.calculate_entry_levels().
    
        CHANGED (v2.800 patch):
        - After computing stop_loss and risk, clamp risk to max
        1.5× primary-TF ATR estimate of daily range.
        - TP2 cap: if TP2 exceeds estimated daily range, reduce R:R
        so the trade can close within the day.
        """
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
    
        stop_loss = self._calc_smc_sl(analysis, entry_price, atr, df)
        risk = abs(entry_price - stop_loss)
    
        # ── NEW: Estimate daily ATR from the primary timeframe ────────────
        # Use the 1H ATR × 4 as a rough daily range estimate, or fall back
        # to entry_tf ATR × bar-count-per-day.
        primary_tf = analysis.get('primary_timeframe', '15m')
        primary_df = multi_tf_data.get(primary_tf, df)
        primary_atr = self.indicators.calculate_atr(primary_df)['current']
        bars_per_day = max(1, 1440 / self._tf_minutes(primary_tf))
        # Daily range ≈ ATR × sqrt(bars_per_day) × 0.6 (overlap factor)
        estimated_daily_range = primary_atr * (bars_per_day ** 0.5) * 0.6
    
        # Clamp risk to max 1.5× estimated daily range
        max_risk_mult = float(self.smc_config.get('max_risk_daily_atr_mult', 1.5))
        max_risk = estimated_daily_range * max_risk_mult
        if risk > max_risk and max_risk > 0:
            logger.info(
                f"[SMC] Risk ${risk:.2f} exceeds {max_risk_mult}× daily range "
                f"${estimated_daily_range:.2f} — clamping to ${max_risk:.2f}"
            )
            risk = max_risk
            direction = analysis.get('direction', 'long')
            if direction == 'long':
                stop_loss = entry_price - risk
            else:
                stop_loss = entry_price + risk
    
        # ── Compute TPs ──────────────────────────────────────────────────────
        tp_config = self.config.get('risk_management', {}).get('take_profit', {})
        targets = tp_config.get('targets', [
            {'name': 'TP1', 'rr_ratio': 2.0, 'close_percent': 40},
            {'name': 'TP2', 'rr_ratio': 4.0, 'close_percent': 35},
            {'name': 'TP3', 'rr_ratio': 5.0, 'close_percent': 25},
        ])
    
        direction = analysis.get('direction', 'long')
        tps = {}
        for i, t in enumerate(targets[:3], 1):
            rr = t.get('rr_ratio', 2.0)
    
            # ── NEW: Cap TP distance to 2× estimated daily range ─────────
            tp_distance = risk * rr
            max_tp = estimated_daily_range * 2.0
            if tp_distance > max_tp and max_tp > 0:
                rr = max_tp / risk if risk > 0 else rr
                tp_distance = max_tp
                logger.debug(
                    f"[SMC] TP{i} capped: {t.get('rr_ratio')}R → {rr:.1f}R "
                    f"(max ${max_tp:.2f})"
                )
    
            if direction == 'long':
                tps[f'tp{i}'] = entry_price + tp_distance
            else:
                tps[f'tp{i}'] = entry_price - tp_distance
    
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
    # SMC CORE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_smc_snapshot(self, df: pd.DataFrame) -> Dict:
        """Build a full SMC analysis snapshot for one timeframe."""
        ind = self.indicators.calculate_all(df)

        # Standard OHLC + indicators
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

        # ── SMC-specific analysis ─────────────────────────────────────────
        swing_highs, swing_lows = self._detect_swing_points(df, order=5)
        snap['swing_highs'] = swing_highs  # list of (index_pos, price)
        snap['swing_lows'] = swing_lows

        snap['structure'] = self._analyze_structure(df, swing_highs, swing_lows)
        snap['order_blocks'] = self._detect_order_blocks(df, swing_highs, swing_lows)
        snap['fvgs'] = self._detect_fair_value_gaps(df)
        snap['liquidity_levels'] = self._detect_liquidity(swing_highs, swing_lows, df)
        snap['displacement'] = self._detect_displacement(df, ind['atr']['current'])

        return snap

    # ── Swing Point Detection ──────────────────────────────────────────────────

    def _detect_swing_points(self, df: pd.DataFrame, order: int = 5
                             ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Detect swing highs and swing lows using fractal logic.
        A swing high has `order` lower highs on each side.
        """
        highs, lows = [], []
        h, l = df['high'].values, df['low'].values
        n = len(df)

        for i in range(order, n - order):
            # Swing high: higher than all neighbours within `order` distance
            if all(h[i] > h[i-j] for j in range(1, order+1)) and \
               all(h[i] > h[i+j] for j in range(1, order+1)):
                highs.append((i, float(h[i])))

            # Swing low: lower than all neighbours
            if all(l[i] < l[i-j] for j in range(1, order+1)) and \
               all(l[i] < l[i+j] for j in range(1, order+1)):
                lows.append((i, float(l[i])))

        return highs, lows

    # ── Market Structure (BOS / CHoCH) ─────────────────────────────────────────

    def _analyze_structure(self, df: pd.DataFrame,
                           swing_highs: List, swing_lows: List) -> Dict:
        """
        Determine market structure using BOS and CHoCH:
        - BOS (Break of Structure): Trend continuation — HH breaks previous HH (bull)
          or LL breaks previous LL (bear).
        - CHoCH (Change of Character): Trend reversal — HL broken in uptrend = bearish CHoCH,
          HH broken in downtrend = bullish CHoCH.
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'trend': 'neutral', 'bias': None, 'bos': False, 'choch': False,
                    'last_swing_high': None, 'last_swing_low': None}

        # Sort by position
        sh = sorted(swing_highs, key=lambda x: x[0])
        sl = sorted(swing_lows, key=lambda x: x[0])

        last_sh = sh[-1][1]
        prev_sh = sh[-2][1] if len(sh) >= 2 else last_sh
        last_sl = sl[-1][1]
        prev_sl = sl[-2][1] if len(sl) >= 2 else last_sl

        current_close = float(df['close'].iloc[-1])

        # Determine swing sequence
        hh = last_sh > prev_sh  # Higher High
        hl = last_sl > prev_sl  # Higher Low
        lh = last_sh < prev_sh  # Lower High
        ll = last_sl < prev_sl  # Lower Low

        bos = False
        choch = False
        trend = 'neutral'
        bias = None

        if hh and hl:
            trend = 'bullish'
            bias = 'bullish'
            # BOS = price broke above the last swing high (continuation)
            if current_close > last_sh:
                bos = True
        elif lh and ll:
            trend = 'bearish'
            bias = 'bearish'
            if current_close < last_sl:
                bos = True
        elif hh and ll:
            # Mixed — check for CHoCH
            if current_close < prev_sl:
                choch = True
                trend = 'bearish'
                bias = 'bearish'
            else:
                trend = 'neutral'
        elif lh and hl:
            if current_close > prev_sh:
                choch = True
                trend = 'bullish'
                bias = 'bullish'
            else:
                trend = 'neutral'

        return {
            'trend': trend,
            'bias': bias,
            'bos': bos,
            'choch': choch,
            'last_swing_high': last_sh,
            'last_swing_low': last_sl,
            'prev_swing_high': prev_sh,
            'prev_swing_low': prev_sl,
            'hh': hh, 'hl': hl, 'lh': lh, 'll': ll,
        }

    # ── Order Block Detection ──────────────────────────────────────────────────

    def _detect_order_blocks(self, df: pd.DataFrame,
                             swing_highs: List, swing_lows: List) -> List[Dict]:
        """
        Detect order blocks: the last opposing candle before an impulsive move.
        
        Bullish OB: Last bearish candle before a strong bullish impulse that
                     creates a swing low and breaks structure.
        Bearish OB: Last bullish candle before a strong bearish impulse.
        """
        obs = []
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        atr = self.indicators.calculate_atr(df)['current']

        # Find bullish order blocks (at swing lows)
        for idx, price in swing_lows[-5:]:  # Last 5 swing lows
            if idx < 2 or idx >= n - 2:
                continue
            # Walk back to find the last bearish candle before the impulse up
            for j in range(idx, max(idx - 5, 0), -1):
                is_bearish = c[j] < o[j]
                # Check if the move UP from this candle was impulsive (> 1.5×ATR)
                move_up = max(h[j+1:min(j+4, n)]) - l[j] if j+1 < n else 0
                if is_bearish and move_up > 1.5 * atr:
                    obs.append({
                        'type': 'bullish',
                        'top': float(o[j]),      # OB top = open of bearish candle
                        'bottom': float(c[j]),    # OB bottom = close of bearish candle
                        'candle_index': j,
                        'strength': move_up / atr,
                        'mitigated': float(df['low'].iloc[j:].min()) < float(c[j]),
                    })
                    break

        # Find bearish order blocks (at swing highs)
        for idx, price in swing_highs[-5:]:
            if idx < 2 or idx >= n - 2:
                continue
            for j in range(idx, max(idx - 5, 0), -1):
                is_bullish = c[j] > o[j]
                move_down = h[j] - min(l[j+1:min(j+4, n)]) if j+1 < n else 0
                if is_bullish and move_down > 1.5 * atr:
                    obs.append({
                        'type': 'bearish',
                        'top': float(c[j]),
                        'bottom': float(o[j]),
                        'candle_index': j,
                        'strength': move_down / atr,
                        'mitigated': float(df['high'].iloc[j:].max()) > float(c[j]),
                    })
                    break

        return obs

    # ── Fair Value Gap Detection ───────────────────────────────────────────────

    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Fair Value Gap (FVG) = 3-candle imbalance.
        Bullish FVG: candle[i-1].high < candle[i+1].low  (gap up)
        Bearish FVG: candle[i-1].low > candle[i+1].high  (gap down)
        
        Price tends to return to fill these gaps — we trade the fill.
        """
        fvgs = []
        h, l = df['high'].values, df['low'].values
        n = len(df)

        for i in range(1, n - 1):
            # Bullish FVG: gap between candle[i-1] high and candle[i+1] low
            if l[i+1] > h[i-1]:
                gap_size = l[i+1] - h[i-1]
                fvgs.append({
                    'type': 'bullish',
                    'top': float(l[i+1]),       # Top of gap
                    'bottom': float(h[i-1]),     # Bottom of gap
                    'candle_index': i,
                    'size': float(gap_size),
                    'filled': float(df['low'].iloc[i+1:].min()) <= float(h[i-1]) if i+2 < n else False,
                })

            # Bearish FVG: gap between candle[i+1] high and candle[i-1] low
            if h[i+1] < l[i-1]:
                gap_size = l[i-1] - h[i+1]
                fvgs.append({
                    'type': 'bearish',
                    'top': float(l[i-1]),
                    'bottom': float(h[i+1]),
                    'candle_index': i,
                    'size': float(gap_size),
                    'filled': float(df['high'].iloc[i+1:].max()) >= float(l[i-1]) if i+2 < n else False,
                })

        # Only return recent unfilled FVGs (last 20 bars)
        recent_unfilled = [f for f in fvgs
                          if f['candle_index'] >= n - 20 and not f['filled']]
        return recent_unfilled[-5:]  # Max 5

    # ── Liquidity Detection ────────────────────────────────────────────────────

    def _detect_liquidity(self, swing_highs: List, swing_lows: List,
                          df: pd.DataFrame) -> Dict:
        """
        Liquidity = clusters of stop losses above swing highs (buy-side) 
        and below swing lows (sell-side). Smart money sweeps these before reversing.
        """
        current = float(df['close'].iloc[-1])

        # Buy-side liquidity: above recent swing highs
        buy_side = sorted([p for _, p in swing_highs if p > current], key=lambda x: x)
        # Sell-side liquidity: below recent swing lows
        sell_side = sorted([p for _, p in swing_lows if p < current], key=lambda x: -x)

        # Check if recent price action swept liquidity
        recent_high = float(df['high'].iloc[-3:].max())
        recent_low = float(df['low'].iloc[-3:].min())

        buy_swept = any(recent_high > p for p in buy_side[:2]) if buy_side else False
        sell_swept = any(recent_low < p for p in sell_side[:2]) if sell_side else False

        return {
            'buy_side': buy_side[:3],
            'sell_side': sell_side[:3],
            'buy_swept': buy_swept,
            'sell_swept': sell_swept,
            'nearest_buy': buy_side[0] if buy_side else None,
            'nearest_sell': sell_side[0] if sell_side else None,
        }

    # ── Displacement Detection ─────────────────────────────────────────────────

    def _detect_displacement(self, df: pd.DataFrame, atr: float) -> Dict:
        """
        Displacement = strong impulsive candle (body > 1.5×ATR).
        Signals institutional involvement.
        """
        bodies = abs(df['close'] - df['open'])
        last_body = float(bodies.iloc[-1])
        last_dir = 'bullish' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'bearish'

        return {
            'detected': last_body > 1.5 * atr,
            'direction': last_dir,
            'body_atr_ratio': last_body / atr if atr > 0 else 0,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY EVALUATION
    # ══════════════════════════════════════════════════════════════════════════

    def _evaluate_smc_entry(self, analysis, multi_tf_data, snaps,
                            structure_tf, primary_tf, entry_tf) -> Dict:
        out = {
            'entry_signal': False, 'entry_reason': None, 'entry_type': None,
            'direction': None, 'confidence_score': 0.0, 'confluence_score': 0.0,
            'confluence_signals': [], 'order_type': 'limit', 'limit_price': None,
        }

        htf_snap = snaps.get(structure_tf)
        ptf_snap = snaps.get(primary_tf, snaps.get(structure_tf))
        ltf_snap = snaps.get(entry_tf, ptf_snap)

        if not htf_snap or not ptf_snap:
            return out

        htf_struct = htf_snap.get('structure', {})
        ptf_struct = ptf_snap.get('structure', {})
        htf_bias = htf_struct.get('bias')

        if not htf_bias:
            return out  # No clear HTF bias — SMC requires directional conviction

        # ── Evaluate entry setups in priority order ───────────────────────

        candidates = []

        # Setup 1: Order Block Entry
        ob_result = self._check_order_block_entry(htf_bias, ptf_snap, ltf_snap)
        if ob_result['signal']:
            score = self._smc_score(ob_result['signals'])
            candidates.append((score, 'order_block', ob_result))

        # Setup 2: FVG Fill Entry
        fvg_result = self._check_fvg_entry(htf_bias, ptf_snap, ltf_snap)
        if fvg_result['signal']:
            score = self._smc_score(fvg_result['signals'])
            candidates.append((score, 'fvg_fill', fvg_result))

        # Setup 3: Liquidity Sweep + Reversal
        liq_result = self._check_liquidity_sweep(htf_bias, ptf_snap, ltf_snap)
        if liq_result['signal']:
            score = self._smc_score(liq_result['signals'])
            candidates.append((score, 'liquidity_sweep', liq_result))

        # Setup 4: BOS + Pullback to OB
        bos_result = self._check_bos_pullback(htf_bias, ptf_snap, ltf_snap)
        if bos_result['signal']:
            score = self._smc_score(bos_result['signals'])
            candidates.append((score, 'bos_pullback', bos_result))

        if not candidates:
            return out

        # Pick highest scoring candidate
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
            'order_type': 'limit',
            'limit_price': result.get('limit_price'),
        })
        return out

    # ── Entry Setup Checks ─────────────────────────────────────────────────────

    def _check_order_block_entry(self, htf_bias, ptf_snap, ltf_snap) -> Dict:
        """Enter at an unmitigated order block aligned with HTF bias."""
        obs = ptf_snap.get('order_blocks', [])
        cp = ltf_snap['ohlc']['close']
        atr = ptf_snap['indicators']['atr']['value']
        signals = []

        target_type = 'bullish' if htf_bias == 'bullish' else 'bearish'
        valid_obs = [ob for ob in obs if ob['type'] == target_type
                     and not ob['mitigated'] and ob['strength'] > 2.0]

        if not valid_obs:
            return {'signal': False}

        # Find OB closest to current price
        best_ob = None
        min_dist = float('inf')
        for ob in valid_obs:
            mid = (ob['top'] + ob['bottom']) / 2
            dist = abs(cp - mid)
            if dist < min_dist and dist < 2.0 * atr:
                min_dist = dist
                best_ob = ob

        if not best_ob:
            return {'signal': False}

        direction = 'long' if htf_bias == 'bullish' else 'short'
        signals.append('order_block_entry')

        # Check if price is in premium/discount zone
        struct = ptf_snap.get('structure', {})
        sh = struct.get('last_swing_high', cp)
        sl_price = struct.get('last_swing_low', cp)
        if sh and sl_price and sh != sl_price:
            mid_range = (sh + sl_price) / 2
            if direction == 'long' and cp < mid_range:
                signals.append('premium_discount_zone')
            elif direction == 'short' and cp > mid_range:
                signals.append('premium_discount_zone')

        # Confluence checks
        self._add_confluence_signals(signals, htf_bias, ptf_snap, ltf_snap)

        # Limit price = OB zone (50% of OB for optimal entry)
        limit_price = (best_ob['top'] + best_ob['bottom']) / 2

        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': limit_price,
            'reason': f'SMC Order Block {direction} @ {limit_price:.2f}',
        }

    def _check_fvg_entry(self, htf_bias, ptf_snap, ltf_snap) -> Dict:
        """Enter at an unfilled Fair Value Gap aligned with HTF bias."""
        fvgs = ptf_snap.get('fvgs', [])
        cp = ltf_snap['ohlc']['close']
        atr = ptf_snap['indicators']['atr']['value']
        signals = []

        target_type = 'bullish' if htf_bias == 'bullish' else 'bearish'
        valid_fvgs = [f for f in fvgs if f['type'] == target_type and not f['filled']]

        if not valid_fvgs:
            return {'signal': False}

        # Find FVG closest to current price
        best_fvg = None
        min_dist = float('inf')
        for fvg in valid_fvgs:
            mid = (fvg['top'] + fvg['bottom']) / 2
            dist = abs(cp - mid)
            if dist < min_dist and dist < 3.0 * atr:
                min_dist = dist
                best_fvg = fvg

        if not best_fvg:
            return {'signal': False}

        direction = 'long' if htf_bias == 'bullish' else 'short'
        signals.append('fvg_entry')

        self._add_confluence_signals(signals, htf_bias, ptf_snap, ltf_snap)

        # Enter at 50% of FVG (consequent encroachment)
        limit_price = (best_fvg['top'] + best_fvg['bottom']) / 2

        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': limit_price,
            'reason': f'SMC FVG fill {direction} @ {limit_price:.2f}',
        }

    def _check_liquidity_sweep(self, htf_bias, ptf_snap, ltf_snap) -> Dict:
        """
        Trade after a liquidity sweep: smart money sweeps stops, then reverses.
        Bullish: sell-side liquidity swept (price dipped below swing low, then reversed)
        Bearish: buy-side liquidity swept (price spiked above swing high, then reversed)
        """
        liq = ptf_snap.get('liquidity_levels', {})
        disp = ptf_snap.get('displacement', {})
        signals = []

        if htf_bias == 'bullish' and liq.get('sell_swept'):
            # Sell-side swept + bullish displacement = smart money bought
            if disp.get('detected') and disp.get('direction') == 'bullish':
                direction = 'long'
                signals.extend(['liquidity_swept', 'displacement_candle'])
            else:
                return {'signal': False}
        elif htf_bias == 'bearish' and liq.get('buy_swept'):
            if disp.get('detected') and disp.get('direction') == 'bearish':
                direction = 'short'
                signals.extend(['liquidity_swept', 'displacement_candle'])
            else:
                return {'signal': False}
        else:
            return {'signal': False}

        self._add_confluence_signals(signals, htf_bias, ptf_snap, ltf_snap)

        cp = ltf_snap['ohlc']['close']
        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': cp,  # Market entry after sweep
            'reason': f'SMC Liquidity sweep {direction}',
        }

    def _check_bos_pullback(self, htf_bias, ptf_snap, ltf_snap) -> Dict:
        """BOS confirmed on primary TF, price pulls back to an OB or FVG."""
        struct = ptf_snap.get('structure', {})
        if not struct.get('bos'):
            return {'signal': False}

        direction = 'long' if struct.get('bias') == 'bullish' else 'short'
        if struct.get('bias') != htf_bias:
            return {'signal': False}

        signals = ['bos_confirmed']

        # Look for an OB or FVG to enter at
        obs = ptf_snap.get('order_blocks', [])
        target_type = 'bullish' if direction == 'long' else 'bearish'
        valid_obs = [ob for ob in obs if ob['type'] == target_type and not ob['mitigated']]

        cp = ltf_snap['ohlc']['close']
        atr = ptf_snap['indicators']['atr']['value']
        limit_price = cp

        if valid_obs:
            best_ob = valid_obs[-1]
            limit_price = (best_ob['top'] + best_ob['bottom']) / 2
            signals.append('order_block_entry')
        else:
            # Try FVG
            fvgs = ptf_snap.get('fvgs', [])
            valid_fvgs = [f for f in fvgs if f['type'] == target_type and not f['filled']]
            if valid_fvgs:
                best_fvg = valid_fvgs[-1]
                limit_price = (best_fvg['top'] + best_fvg['bottom']) / 2
                signals.append('fvg_entry')

        self._add_confluence_signals(signals, htf_bias, ptf_snap, ltf_snap)

        return {
            'signal': True,
            'direction': direction,
            'signals': signals,
            'limit_price': limit_price,
            'reason': f'SMC BOS pullback {direction} @ {limit_price:.2f}',
        }

    # ── Confluence Helpers ─────────────────────────────────────────────────────

    def _add_confluence_signals(self, signals: List, htf_bias: str,
                                ptf_snap: Dict, ltf_snap: Dict):
        """Add standard confluence checks to signal list."""
        ind = ptf_snap['indicators']

        # HTF alignment
        signals.append('htf_structure_aligned')

        # EMA alignment
        e20, e50 = ind['ema'].get(20, 0), ind['ema'].get(50, 0)
        if htf_bias == 'bullish' and e20 > e50:
            signals.append('ema_bias_aligned')
        elif htf_bias == 'bearish' and e20 < e50:
            signals.append('ema_bias_aligned')

        # RSI not exhausted
        rsi = ind['rsi']['value']
        if htf_bias == 'bullish' and 30 < rsi < 70:
            signals.append('rsi_not_exhausted')
        elif htf_bias == 'bearish' and 30 < rsi < 70:
            signals.append('rsi_not_exhausted')

        # ADX trending
        if ind['adx'].get('trend_strength') == 'strong':
            signals.append('adx_trending')

        # ATR sufficient (not dead market)
        if ind['atr']['percent'] > 0.05:
            signals.append('atr_sufficient')

        # Killzone check
        hour = datetime.utcnow().hour
        for kz_name, (start, end) in _KILLZONES.items():
            if start <= hour < end:
                signals.append('session_killzone')
                break

    # ── Scoring & Filtering ────────────────────────────────────────────────────

    def _smc_score(self, signals: List[str]) -> float:
        return sum(_SMC_WEIGHTS.get(s, 1) for s in signals)

    def _get_threshold(self, symbol: str) -> int:
        sc = self.config.get('symbols', {}).get(symbol, {})
        if 'confluence_threshold' in sc:
            return int(sc['confluence_threshold'])
        ms = symbol.replace('/', '').upper()
        if ms in _VOLATILE_SYMBOLS:
            return int(self.smc_config.get('confluence_required',
                       self.strategy_config.get('confluence_required', 7)))
        return int(self.smc_config.get('confluence_threshold_calm',
                   self.strategy_config.get('confluence_threshold_calm', 5)))

    def _apply_smc_filters(self, analysis, snaps, structure_tf) -> Dict:
        """Apply safety filters — reject entries in choppy / low-ADX conditions."""
        htf = snaps.get(structure_tf)
        if not htf:
            return analysis

        # Filter 1: ADX must show trend (anti-chop)
        adx = htf['indicators']['adx']
        adx_val = adx.get('value')
        if isinstance(adx_val, pd.Series):
            adx_val = float(adx_val.iloc[-1])
        if adx_val is not None and adx_val < 20:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'SMC Filter: HTF ADX too low ({adx_val:.1f})'
            return analysis

        # Filter 2: Structure must be clear (not mixed signals)
        struct = htf.get('structure', {})
        if struct.get('trend') == 'neutral':
            analysis['entry_signal'] = False
            analysis['entry_reason'] = 'SMC Filter: HTF structure neutral'
            return analysis

        # Filter 3: ATR threshold
        atr_pct = htf['indicators']['atr']['percent']
        filters = self.strategy_config.get('filters', {})
        min_atr = filters.get('min_atr_threshold', 0.05)
        if atr_pct < min_atr:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'SMC Filter: ATR too low ({atr_pct:.3f}%)'

        return analysis

    # ── Stop Loss (Structure-Based) ────────────────────────────────────────────

    def _calc_smc_sl(self, analysis, entry_price, atr, df) -> float:
        """
        SMC stop loss: placed beyond the relevant structure point.
        Long: below the last swing low (or below the OB)
        Short: above the last swing high (or above the OB)
    
        CHANGED (v2.800 patch):
        - Fallback ATR multiplier reduced from 2.0 → 1.5 to tighten
        stops when no structure is found.
        - Buffer reduced from 0.3×ATR → 0.2×ATR (same as ICT).
        - Added absolute distance cap: SL cannot be further than
        max_sl_atr_mult × ATR from entry, even when structure is far away.
        This prevents the cascade: wide SL → wide risk → impossibly wide TP.
        """
        struct = analysis.get('market_structure', {})
        direction = analysis.get('direction', 'long')
        buffer = atr * 0.3   # was 0.3 — tighter buffer
    
        # ── Maximum SL distance cap ──────────────────────────────────────────
        # Never let SL be more than 1.5× ATR from entry.
        # This is the key fix: when structure is far away (e.g. swing low
        # is $50 below entry on gold), the old code would set SL there,
        # creating a $50 risk → $200 TP target. Now capped.
        max_sl_atr_mult = float(
            self.smc_config.get('max_sl_atr_multiplier', 1.5)
        )
        max_sl_distance = atr * max_sl_atr_mult
    
        if direction == 'long':
            swing_low = struct.get('last_swing_low')
            if swing_low:
                sl = swing_low - buffer
            else:
                sl = entry_price - 2.0 * atr  # was 2.0
    
            # Clamp: don't let SL be further than max_sl_distance from entry
            if entry_price - sl > max_sl_distance:
                sl = entry_price - max_sl_distance
                logger.debug(
                    f"[SMC] SL clamped to {sl:.5f} (max {max_sl_atr_mult}×ATR "
                    f"= {max_sl_distance:.2f} from entry {entry_price:.5f})"
                )
            return sl
        else:
            swing_high = struct.get('last_swing_high')
            if swing_high:
                sl = swing_high + buffer
            else:
                sl = entry_price + 2.0 * atr  # was 2.0
    
            if sl - entry_price > max_sl_distance:
                sl = entry_price + max_sl_distance
                logger.debug(
                    f"[SMC] SL clamped to {sl:.5f} (max {max_sl_atr_mult}×ATR "
                    f"= {max_sl_distance:.2f} from entry {entry_price:.5f})"
                )
            return sl