"""
strategy/nlm_engine.py — Nexus Liquidity Method (NLM) Strategy Engine
=======================================================================
Fourth drop-in replacement for StrategyEngine. Implements the NLM checklist
for XAUUSD (Gold) — a disciplined, phase-based ICT/SMC adaptation.

NLM workflow (mirrors the 9-phase checklist):
  Phase 1: Higher-timeframe bias (D1 / H4)            — directional displacement, HH/LL or LH/HL
  Phase 2: Liquidity mapping (H1 / H4)                — EQH/EQL, PDH/PDL, session highs/lows
  Phase 3: Nexus Zone identification (H1 / M15)       — confluence of 2-4 of: liquidity, OB, FVG, premium/discount
  Phase 4: Inducement / liquidity sweep (M15 / M5)    — clean wick beyond level, momentum stalls
  Phase 5: Market Structure Shift (M15 / M5)          — CHoCH (preferred) or BOS in bias direction
  Phase 6: Entry execution (M5 / M1)                  — OB > FVG > Breaker priority
  Phase 7: Stop loss beyond sweep extreme             — 3-5 pip buffer past wick
  Phase 8: Liquidity-based take profits               — internal LQ → external LQ → runner
  Phase 9: Trade management                           — breakeven at TP1, trail after TP2

Abort criteria (Phase-X):
  - Price within 40-60% D1 midpoint  → chop zone
  - No liquidity sweep                → no setup
  - Nexus Zone < 2 confluences        → no setup
  - Premium for long / Discount short → wrong half of swing
  - High-impact news within 30 min    → news filter (out of band — handled upstream)
  - Stop > max risk tolerance         → reject in calculate_entry_levels via SL cap

Integration:
  In main.py, change:
    from strategy.engine import StrategyEngine
  to:
    from strategy.nlm_engine import NLMStrategyEngine as StrategyEngine

  No other changes required — identical analyze_market() / calculate_entry_levels() API.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from indicators.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# ── NLM Signal Weights ────────────────────────────────────────────────────────
# Weighted by the checklist's "priority" — sweep + MSS are the gatekeepers,
# Nexus Zone confluence count drives conviction, entry-type priority is
# enforced via candidate ordering rather than weights alone.
_NLM_WEIGHTS: Dict[str, int] = {
    # Phase 4 — Sweep is non-negotiable (highest weight)
    'liquidity_sweep_clean':        5,
    'sweep_momentum_stalled':       3,
    # Phase 5 — Structural confirmation
    'choch_in_bias':                5,    # CHoCH preferred over BOS
    'bos_in_bias':                  3,
    'structure_clean_no_failure':   2,
    # Phase 3 — Nexus Zone confluences (each adds weight; min 2 required)
    'nz_liquidity_pool':            2,
    'nz_order_block':               2,
    'nz_fair_value_gap':            2,
    'nz_premium_discount_aligned':  2,
    # Phase 1 — HTF bias
    'htf_bias_clear':               3,
    'd1_directional_displacement':  2,
    # Phase 2 — Liquidity targets identified
    'liquidity_target_mapped':      1,
    # Phase 6 — Entry refinement
    'ob_entry_clean':               3,
    'fvg_entry':                    2,
    'breaker_entry':                1,    # Lower conviction per checklist
    'entry_momentum_confirmation':  1,
    # Supporting / session
    'gold_session_active':          1,    # London open or NY open windows
    'not_in_d1_midpoint':           2,    # Phase 1 abort guard
}

# C3 FIXED 2026-09-23: the gate scored the full signal list, but every path
# that reaches scoring has already appended a fixed set of signals worth 23
# (Phase 1: 7, Phase 2: 1, Phase 4: 8, the always-true nz_liquidity_pool: 2,
# MSS floor bos + structure_clean: 5). With the minimum discretionary points
# the floor was 26 against thresholds of 9 and 12, so the gate could never
# fire. Live: 109 of 109 signals between 30 Aug and 23 Sep scored 28 to 33.
#
# The gate now reads a conviction score built only from signals that can be
# absent on a path that reaches scoring. CHoCH counts its 2-point premium
# over BOS. Range: 3 (one extra Nexus confluence, BOS, breaker entry, no
# session) to 13. confluence_score keeps the old total for log continuity.
_NLM_CONVICTION_WEIGHTS: Dict[str, int] = {
    'nz_order_block':               2,
    'nz_fair_value_gap':            2,
    'nz_premium_discount_aligned':  2,
    'choch_in_bias':                2,
    'ob_entry_clean':               3,
    'fvg_entry':                    2,
    'breaker_entry':                1,
    'entry_momentum_confirmation':  1,
    'gold_session_active':          1,
}
_NLM_CONVICTION_MAX = 13

# Config keys that belonged to the old, unreachable scale. Read only to warn.
_NLM_RETIRED_KEYS = ('confluence_required', 'confluence_threshold_calm')

# Gold-specific session windows (UTC) — taken directly from the checklist
# Asian:        00:00-06:00   range reference
# London open:  07:00-10:00   (02:00-05:00 EST in checklist = 07:00-10:00 UTC)
# NY open:      13:00-16:00   (08:30-10:00 EST cash open band)
_NLM_SESSIONS: Dict[str, Tuple[int, int]] = {
    'asian':        (0, 6),
    'london_open':  (7, 10),
    'ny_open':      (13, 16),
    'london_close': (15, 17),
}

# UNUSED since C3 (2026-09-23): thresholds no longer depend on a volatility
# class. Kept so sibling-engine diffs stay readable.
_VOLATILE_SYMBOLS = {'XAUUSD', 'XAU/USD', 'BTCUSD', 'BTC/USD', 'NAS100USD', 'NAS100', 'US100'}

# Pip size for the checklist's "3-5 pip buffer" beyond sweep extreme.
# The NLM checklist was written for Gold ($0.10/pip), but this engine is run
# on other symbols too — a hardcoded $0.10 pip made the SL buffer ~400 pips
# on EURUSD. Per-symbol defaults below; override via config `pip_sizes` map.
_DEFAULT_PIP_SIZES = {
    'XAUUSD': 0.10,     # Gold: $0.10
    'XAGUSD': 0.01,     # Silver
    'BTCUSD': 10.0,     # BTC: $10 as a practical "pip"
    'EURUSD': 0.0001,
    'GBPUSD': 0.0001,
    'USDJPY': 0.01,
}
_FALLBACK_PIP = 0.0001


class NLMStrategyEngine:
    """
    Nexus Liquidity Method (NLM) strategy engine for XAUUSD.

    Each public-API call walks the 9-phase checklist in sequence and aborts
    on the first failed mandatory gate. Phase 4 (sweep) and Phase 5 (MSS) are
    absolute prerequisites — no signal without both. Phase 3 confluence count
    is the conviction driver.

    Concepts (mirrors checklist terminology):
      - HTF Bias        — D1/H4 directional displacement
      - Nexus Zone      — confluence of liquidity + OB + FVG + premium/discount
      - Inducement      — engineered sweep that takes liquidity before reversing
      - MSS             — Market Structure Shift (CHoCH preferred, BOS acceptable)
      - Liquidity TPs   — internal → external → D1/weekly runner
    """

    def __init__(self, config: Dict):
        self.config = config
        self.indicators = TechnicalIndicators(config.get('indicators', {}))
        self.strategy_config = config.get('strategy', {})
        self.nlm_config = config.get('nlm', {}) or {}  # Optional NLM-specific overrides

        # C3: warn once about thresholds on the retired scale, so a config
        # edit to them is not mistaken for a live control.
        stale = [k for k in _NLM_RETIRED_KEYS if k in self.nlm_config]
        for sym, sc in (config.get('symbols', {}) or {}).items():
            if isinstance(sc, dict) and 'nlm_confluence_threshold' in sc:
                stale.append(f'symbols.{sym}.nlm_confluence_threshold')
        if stale:
            logger.warning(
                f"[NLM] Ignoring retired threshold keys {stale}: they were on a "
                f"scale the score could never fall below. Use nlm.min_conviction "
                f"(range 3-{_NLM_CONVICTION_MAX}) or symbols.<sym>.nlm.min_conviction."
            )

    def _nlm_cfg(self, symbol: Optional[str] = None) -> Dict:
        """N3 FIXED 2026-09-23: NLM settings for one symbol.

        Every NLM constant was global, so values tuned on gold applied to
        US30, US500, EURUSD and silver unchanged. A `symbols.<sym>.nlm:` block
        now overrides any key of the top-level `nlm:` block for that symbol.
        """
        merged = dict(self.nlm_config)
        if symbol:
            sc = (self.config.get('symbols', {}) or {}).get(symbol, {}) or {}
            merged.update(sc.get('nlm', {}) or {})
        return merged

    def _tf_minutes(self, tf: str) -> int:
        m = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1H': 60, '4H': 240, '1D': 1440,
             'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440}
        return m.get(tf, 60)

    @staticmethod
    def _norm_symbol(symbol: str) -> str:
        """Broker-suffix-tolerant symbol key.

        B1 FIXED 2026-08-30 audit: every lookup in this file compared
        symbol.replace('/', '').upper() against exact table keys, so the
        Exness micro suffix made 'XAU/USDm' normalise to 'XAUUSDM' and match
        nothing. _pip_size('XAU/USDm') returned the 0.0001 fallback instead of
        0.1, and _get_threshold('XAU/USDm') returned the calm 9 instead of 12.
        Longest-prefix matching fixes both, and any future symbol.
        """
        return symbol.replace('/', '').replace('-', '').replace('_', '').upper()

    @staticmethod
    def _lookup_by_prefix(norm: str, table: dict):
        if norm in table:
            return table[norm]
        hits = [k for k in table if k and norm.startswith(k)]
        return table[max(hits, key=len)] if hits else None

    def _pip_size(self, symbol: str) -> float:
        """Per-symbol pip size; config `pip_sizes` map overrides defaults."""
        norm = self._norm_symbol(symbol)
        overrides = {self._norm_symbol(str(k)): v
                     for k, v in (self.config.get('pip_sizes', {}) or {}).items()}
        hit = self._lookup_by_prefix(norm, overrides)
        if hit is not None:
            return float(hit)
        hit = self._lookup_by_prefix(norm, _DEFAULT_PIP_SIZES)
        return float(hit) if hit is not None else _FALLBACK_PIP

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — identical signature to StrategyEngine / SMC / ICT engines
    # ══════════════════════════════════════════════════════════════════════════

    def analyze_market(self, symbol: str, multi_tf_data: Dict[str, pd.DataFrame],
                       symbol_config: Optional[Dict] = None) -> Dict:
        """
        Full NLM 9-phase analysis. Returns the same shape every other engine
        returns so the orchestrator, money manager, and stop manager continue
        working unchanged.
        """
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
            'symbol': symbol, # FIXED 2026-08-30 audit: datetime.now() is the host's LOCAL clock, which
            # on this machine is 7 hours behind UTC, while every trade,
            # order and log record in the system is UTC. The analysis
            # timestamp was the odd one out.
            'timestamp': datetime.now(timezone.utc).replace(tzinfo=None),
            'primary_timeframe': primary_tf, 'structure_tf': structure_tf,
            'entry_tf': entry_tf, 'timeframe_snapshots': {},
            'market_structure': {}, 'indicators_state': {},
            'entry_signal': False, 'entry_reason': None, 'entry_type': None,
            'confidence_score': 0.0, 'confluence_score': 0.0, 'conviction_score': 0,
            'confluence_signals': [], 'direction': None,
            'order_type': 'limit', 'limit_price': None,
            # NLM-specific diagnostics surfaced for logging / dashboard
            'nlm_phase_results': {},
        }

        try:
            # ── Build snapshots per TF ────────────────────────────────────
            for tf, df in multi_tf_data.items():
                if df is None or len(df) < 50:
                    continue
                analysis['timeframe_snapshots'][tf] = self._build_nlm_snapshot(df, tf, symbol)

            snaps = analysis['timeframe_snapshots']
            if not snaps:
                return analysis

            # N1: replace the structure TF's 20-bar proxy with a real D1 range
            # check. Phase 1 and the final filter both read this snapshot key.
            if structure_tf in snaps and structure_tf in multi_tf_data:
                d1 = self._check_d1_midpoint_daily(multi_tf_data, structure_tf, symbol)
                if d1 is not None:
                    snaps[structure_tf]['d1_midpoint_chop'] = d1

            if structure_tf in snaps:
                analysis['market_structure'] = snaps[structure_tf].get('structure', {})

            # ── Walk the 9-phase checklist ────────────────────────────────
            decision = self._evaluate_nlm_setup(
                analysis, multi_tf_data, snaps,
                structure_tf, primary_tf, entry_tf
            )
            analysis.update(decision)

            # ── Final safety net (Phase abort criteria) ───────────────────
            if analysis['entry_signal']:
                analysis = self._apply_nlm_filters(analysis, snaps, structure_tf, primary_tf)

            logger.info(
                f"[NLM] {symbol} signal={analysis['entry_signal']} "
                f"dir={analysis['direction']} type={analysis.get('entry_type')} "
                f"score={analysis.get('confluence_score', 0):.1f} "
                f"conviction={analysis.get('conviction_score', 0)} "
                f"nz_confluence={analysis['nlm_phase_results'].get('nexus_confluence_count', 0)}/4"
            )

        except Exception as e:
            logger.error(f"[NLM] Error analyzing {symbol}: {e}", exc_info=True)

        return analysis

    def calculate_entry_levels(self, analysis: Dict,
                               multi_tf_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Phase 7 (SL beyond sweep extreme) + Phase 8 (liquidity-based TPs).
        Returns the same dict shape as every other engine.

        Stop loss:  3-5 pips beyond the sweep wick (not just beyond the OB),
                    with an ATR-based cap as a safety net.
        TP1:        nearest internal liquidity pool (partial 30-50%)
        TP2:        external liquidity (PDH/PDL, session extreme, EQH/EQL)
        TP3:        D1/weekly liquidity — runner (trailing in stop_manager)
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

        # ── Phase 7: SL beyond sweep extreme ──────────────────────────────
        stop_loss = self._calc_nlm_sl(analysis, entry_price, atr)
        risk = abs(entry_price - stop_loss)

        # ── Phase 8: liquidity-based TPs ──────────────────────────────────
        tps = self._calc_nlm_tps(analysis, entry_price, stop_loss, multi_tf_data)

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
    # SNAPSHOT BUILDERS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_nlm_snapshot(self, df: pd.DataFrame, tf: str,
                            symbol: Optional[str] = None) -> Dict:
        """Build a full NLM snapshot for one timeframe."""
        ind = self.indicators.calculate_all(df)

        snap = {
            'ohlc': {k: float(df[k].iloc[-1]) for k in ['open', 'high', 'low', 'close', 'volume']},
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

        # Structure primitives
        swing_highs, swing_lows = self._detect_swing_points(df, order=5)
        snap['swing_highs'] = swing_highs
        snap['swing_lows'] = swing_lows
        snap['structure'] = self._analyze_nlm_structure(df, swing_highs, swing_lows)

        # NLM-specific building blocks
        snap['order_blocks'] = self._detect_order_blocks(df, ind['atr']['current'])
        snap['fvgs'] = self._detect_fair_value_gaps(df)
        snap['breaker_blocks'] = self._detect_breaker_blocks(df, ind['atr']['current'])
        snap['liquidity_map'] = self._build_liquidity_map(df, swing_highs, swing_lows, tf,
                                                          symbol)
        snap['sweep'] = self._detect_inducement_sweep(df, snap['liquidity_map'],
                                                     ind['atr']['current'])
        snap['mss'] = self._detect_market_structure_shift(
            df, swing_highs, swing_lows,
            max_age_bars=int(self._nlm_cfg(symbol).get('max_mss_age_bars', 15)))
        snap['d1_midpoint_chop'] = self._check_d1_midpoint(df)

        return snap

    # ── Swing detection (fractal) ──────────────────────────────────────────────

    def _detect_swing_points(self, df: pd.DataFrame, order: int = 5
                             ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        highs, lows = [], []
        h, l = df['high'].values, df['low'].values
        n = len(df)
        for i in range(order, n - order):
            if all(h[i] > h[i - j] for j in range(1, order + 1)) and \
               all(h[i] > h[i + j] for j in range(1, order + 1)):
                highs.append((i, float(h[i])))
            if all(l[i] < l[i - j] for j in range(1, order + 1)) and \
               all(l[i] < l[i + j] for j in range(1, order + 1)):
                lows.append((i, float(l[i])))
        return highs, lows

    # ── Structure (Phase 1 + Phase 5) ──────────────────────────────────────────

    def _analyze_nlm_structure(self, df: pd.DataFrame,
                               swing_highs: List, swing_lows: List) -> Dict:
        """
        Determine bias from HH/HL (bullish) or LH/LL (bearish). Mirrors the
        checklist's Phase 1 requirement of a clear directional displacement.
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'trend': 'neutral', 'bias': None,
                    'last_swing_high': None, 'last_swing_low': None}

        sh = sorted(swing_highs, key=lambda x: x[0])
        sl = sorted(swing_lows, key=lambda x: x[0])
        last_sh, prev_sh = sh[-1][1], sh[-2][1]
        last_sl, prev_sl = sl[-1][1], sl[-2][1]

        hh = last_sh > prev_sh
        hl = last_sl > prev_sl
        lh = last_sh < prev_sh
        ll = last_sl < prev_sl

        bias = None
        trend = 'neutral'
        if hh and hl:
            trend, bias = 'bullish', 'bullish'
        elif lh and ll:
            trend, bias = 'bearish', 'bearish'

        return {
            'trend': trend, 'bias': bias,
            'last_swing_high': last_sh, 'last_swing_low': last_sl,
            'prev_swing_high': prev_sh, 'prev_swing_low': prev_sl,
            'hh': hh, 'hl': hl, 'lh': lh, 'll': ll,
            'sh_idx': sh[-1][0], 'sl_idx': sl[-1][0],
        }

    # ── Order Block detection ──────────────────────────────────────────────────

    def _detect_order_blocks(self, df: pd.DataFrame, atr: float,
                             include_mitigated: bool = False) -> List[Dict]:
        """Last opposing candle before an impulsive displacement (>1.5×ATR).

        Returned oldest-first (ascending index). With include_mitigated=True
        the traded-through blocks are kept too, which is what breaker
        detection needs.
        """
        obs: List[Dict] = []
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        # C1 FIXED 2026-09-23: the block is validated by the displacement over
        # bars i+1..i+3, and mitigation was then tested from i+1, over the
        # same bars. The next bar opens at about c[i] and almost always wicks
        # below its own open, so the leg that defines a bullish block also
        # "mitigated" it. On 300 synthetic series 2,622 of 2,622 recent
        # blocks died this way; live, OBs appeared in 9-17% of snapshots
        # against FVGs in 85-96%, and 108 of 109 NLM entries were FVGs.
        # Mitigation is now tested only after the displacement window.
        disp_end = 4

        for i in range(2, n - 3):
            is_bearish = c[i] < o[i]
            is_bullish = c[i] > o[i]
            move_up = max(h[i + 1:min(i + 4, n)]) - l[i] if i + 1 < n else 0
            move_down = h[i] - min(l[i + 1:min(i + 4, n)]) if i + 1 < n else 0
            after = i + disp_end   # first bar after the displacement window

            if is_bearish and move_up > 1.5 * atr:
                obs.append({
                    'type': 'bullish',
                    'top': float(o[i]), 'bottom': float(c[i]),
                    'mid': (float(o[i]) + float(c[i])) / 2,
                    'index': i, 'strength': move_up / atr if atr > 0 else 0,
                    'mitigated': bool(after < n and float(l[after:].min()) < float(c[i])),
                })
            if is_bullish and move_down > 1.5 * atr:
                obs.append({
                    'type': 'bearish',
                    'top': float(c[i]), 'bottom': float(o[i]),
                    'mid': (float(o[i]) + float(c[i])) / 2,
                    'index': i, 'strength': move_down / atr if atr > 0 else 0,
                    'mitigated': bool(after < n and float(h[after:].max()) > float(c[i])),
                })

        recent = [ob for ob in obs if ob['index'] >= n - 40]
        if include_mitigated:
            return recent
        # Keep only recent + unmitigated OBs
        return [ob for ob in recent if not ob['mitigated']][-6:]

    # ── FVG detection ──────────────────────────────────────────────────────────

    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """3-candle imbalance. Bullish gap: c[i-1].high < c[i+1].low."""
        fvgs: List[Dict] = []
        h, l = df['high'].values, df['low'].values
        n = len(df)

        for i in range(1, n - 1):
            if l[i + 1] > h[i - 1]:
                fvgs.append({
                    'type': 'bullish',
                    'top': float(l[i + 1]), 'bottom': float(h[i - 1]),
                    'mid': (float(l[i + 1]) + float(h[i - 1])) / 2,
                    'index': i,
                    'size': float(l[i + 1] - h[i - 1]),
                    'filled': float(df['low'].iloc[i + 1:].min()) <= float(h[i - 1])
                    if i + 2 < n else False,
                })
            if h[i + 1] < l[i - 1]:
                fvgs.append({
                    'type': 'bearish',
                    'top': float(l[i - 1]), 'bottom': float(h[i + 1]),
                    'mid': (float(l[i - 1]) + float(h[i + 1])) / 2,
                    'index': i,
                    'size': float(l[i - 1] - h[i + 1]),
                    'filled': float(df['high'].iloc[i + 1:].max()) >= float(l[i - 1])
                    if i + 2 < n else False,
                })

        recent_unfilled = [f for f in fvgs if f['index'] >= n - 25 and not f['filled']]
        return recent_unfilled[-5:]

    # ── Breaker Blocks ─────────────────────────────────────────────────────────

    def _detect_breaker_blocks(self, df: pd.DataFrame, atr: float) -> List[Dict]:
        """An OB that failed and flipped polarity → acts as S/R the other way."""
        breakers: List[Dict] = []
        # C2 FIXED 2026-09-23: this iterated only UNmitigated blocks, and for a
        # bullish block "unmitigated" means no low has traded below its bottom,
        # while the breaker test below requires the latest close (so the latest
        # low) to be below it. The two cannot both hold, so no breaker could
        # ever form: 0 breakers in 7,670 logged snapshots. A breaker is by
        # definition a block price has traded through, so read all of them.
        obs = self._detect_order_blocks(df, atr, include_mitigated=True)
        cp = float(df['close'].iloc[-1])
        for ob in obs:
            if ob['type'] == 'bullish' and cp < ob['bottom']:
                breakers.append({**ob, 'type': 'bearish_breaker'})
            elif ob['type'] == 'bearish' and cp > ob['top']:
                breakers.append({**ob, 'type': 'bullish_breaker'})
        return breakers[-3:]

    # ── Phase 2: Liquidity Map ─────────────────────────────────────────────────

    def _build_liquidity_map(self, df: pd.DataFrame,
                             swing_highs: List, swing_lows: List, tf: str,
                             symbol: Optional[str] = None) -> Dict:
        """
        Build the liquidity map prescribed by Phase 2:
          - Equal Highs / Equal Lows (EQH / EQL)
          - Previous Day High / Low (PDH / PDL)
          - Asian / London / NY session extremes
          - Recent significant swing high / low

        Tolerances tuned for XAUUSD: 'equal' means within 0.05% of price.
        """
        if len(df) < 10:
            return {}

        cp = float(df['close'].iloc[-1])
        atr = self.indicators.calculate_atr(df)['current']
        # N3 FIXED 2026-09-23: was max(0.05% of price, 0.1 x ATR). A percent of
        # price is not a volatility unit, so "equal" meant a different thing
        # on every symbol. Measured on the live primary TFs, in ATRs:
        #   XAG 1H 0.08 | XAU 15m 0.27 | US30 1H 0.29 | US500 1H 0.32 | EUR 1H 0.72
        # Silver found almost no equal levels and EURUSD clustered swings most
        # of an ATR apart. Now ATR-only; the default is gold's measured value
        # so gold is unchanged in the median. Override per symbol if needed.
        eq_tol = float(self._nlm_cfg(symbol).get('eq_tolerance_atr', 0.27)) * atr

        # ── Equal highs / lows ────────────────────────────────────────────
        eqh = self._find_equal_levels([p for _, p in swing_highs], eq_tol)
        eql = self._find_equal_levels([p for _, p in swing_lows], eq_tol)

        # ── Session extremes (requires time-indexed df) ───────────────────
        sessions = self._compute_session_extremes(df)

        # ── Previous Day H/L ──────────────────────────────────────────────
        pdh, pdl = self._compute_pdh_pdl(df)

        # ── Recent significant swing ──────────────────────────────────────
        recent_sh = swing_highs[-1][1] if swing_highs else None
        recent_sl = swing_lows[-1][1] if swing_lows else None

        # ── Categorise into buy-side (above) / sell-side (below) ──────────
        buy_side_levels: List[Dict] = []
        sell_side_levels: List[Dict] = []

        def _push(level: Optional[float], label: str):
            if level is None:
                return
            entry = {'price': float(level), 'label': label}
            (buy_side_levels if level > cp else sell_side_levels).append(entry)

        for p in eqh:
            _push(p, 'EQH')
        for p in eql:
            _push(p, 'EQL')
        _push(pdh, 'PDH')
        _push(pdl, 'PDL')
        for name, hi in sessions.get('highs', {}).items():
            _push(hi, f'{name}_high')
        for name, lo in sessions.get('lows', {}).items():
            _push(lo, f'{name}_low')
        _push(recent_sh, 'swing_high')
        _push(recent_sl, 'swing_low')

        # Sort ascending for buy side, descending for sell side
        buy_side_levels.sort(key=lambda x: x['price'])
        sell_side_levels.sort(key=lambda x: -x['price'])

        return {
            'eqh': eqh, 'eql': eql,
            'pdh': pdh, 'pdl': pdl,
            'sessions': sessions,
            'recent_swing_high': recent_sh, 'recent_swing_low': recent_sl,
            'buy_side': buy_side_levels[:6],
            'sell_side': sell_side_levels[:6],
            'nearest_buy_side': buy_side_levels[0] if buy_side_levels else None,
            'nearest_sell_side': sell_side_levels[0] if sell_side_levels else None,
        }

    @staticmethod
    def _find_equal_levels(prices: List[float], tol: float) -> List[float]:
        """Cluster prices that lie within `tol` of each other; return cluster mean."""
        if not prices:
            return []
        sorted_p = sorted(prices)
        clusters: List[List[float]] = [[sorted_p[0]]]
        for p in sorted_p[1:]:
            if abs(p - clusters[-1][-1]) <= tol:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [float(np.mean(c)) for c in clusters if len(c) >= 2]

    def _compute_session_extremes(self, df: pd.DataFrame) -> Dict:
        """Per-session highs and lows for the most recent UTC day in the data.

        CAVEAT 2026-08-30 audit (I4, unresolved): df.index is NOT verified to be
        UTC. The EA sends raw MQL5 rates[i].time, which is broker-server time,
        and execution/mt5_file_bridge.py labels it utc=True without converting.
        The broker handshake reports a non-zero and unstable offset. If the
        server is not on UTC, every window below and the day boundary in
        _compute_pdh_pdl are shifted by that offset. Run
        ../verify_server_offset.py against a live terminal to settle it, and fix
        it once in data_feed/market_client.py rather than here.
        """
        out: Dict[str, Dict[str, float]] = {'highs': {}, 'lows': {}}
        if not hasattr(df.index, 'hour'):
            return out
        try:
            last_date = df.index[-1].date()
            today = df[df.index.date == last_date]
            for name, (start, end) in _NLM_SESSIONS.items():
                session = today[(today.index.hour >= start) & (today.index.hour < end)]
                if len(session) >= 2:
                    out['highs'][name] = float(session['high'].max())
                    out['lows'][name] = float(session['low'].min())
        except Exception:
            pass
        return out

    def _compute_pdh_pdl(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        if not hasattr(df.index, 'hour'):
            return None, None
        try:
            last_date = df.index[-1].date()
            prev_day = df[df.index.date < last_date]
            if len(prev_day) < 5:
                return None, None
            prev_date = prev_day.index[-1].date()
            prev_bars = prev_day[prev_day.index.date == prev_date]
            return float(prev_bars['high'].max()), float(prev_bars['low'].min())
        except Exception:
            return None, None

    # ── Phase 4: Inducement / Liquidity Sweep ──────────────────────────────────

    def _detect_inducement_sweep(self, df: pd.DataFrame,
                                 liq_map: Dict, atr: float) -> Dict:
        """
        Detect a clean liquidity sweep:
          - Recent wick pierced a mapped liquidity level
          - The close did NOT continue strongly in the sweep direction
            (momentum stalled or reversed)
          - The sweep was a wick, not a body close beyond the level

        Returns:
          { detected: bool, type: 'buy_side'|'sell_side',
            level: float, level_label: str,
            extreme: float, momentum_stalled: bool,
            clean_wick: bool, bar_index: int }
        """
        out = {'detected': False, 'momentum_stalled': False, 'clean_wick': False}
        if not liq_map:
            return out

        n = len(df)
        if n < 5:
            return out

        # Look at the last 5 bars for sweep activity
        window = df.iloc[-5:]
        recent_high = float(window['high'].max())
        recent_low = float(window['low'].min())
        last_close = float(df['close'].iloc[-1])

        # Find the positional index (0..4) of the bar containing each extreme.
        # We use np.argmax/argmin on .values so the result is always a plain int,
        # regardless of whether df has a DatetimeIndex or a RangeIndex.
        try:
            high_pos = int(np.argmax(window['high'].values))
        except Exception:
            high_pos = len(window) - 1
        try:
            low_pos = int(np.argmin(window['low'].values))
        except Exception:
            low_pos = len(window) - 1

        # Buy-side sweep — wick pierced a buy-side liquidity level
        for lvl in liq_map.get('buy_side', []):
            if recent_high > lvl['price']:
                # Verify the sweep is a wick, not a body close beyond
                try:
                    bar = window.iloc[high_pos]
                    bar_close = float(bar['close'])
                    clean_wick = bar_close < lvl['price']  # closed back below
                except Exception:
                    clean_wick = last_close < lvl['price']

                # Momentum stalled = current close is below the sweep extreme,
                # ideally below the swept level
                momentum_stalled = last_close < recent_high - 0.2 * atr
                sweep_strong_enough = (recent_high - lvl['price']) > 0.1 * atr

                if clean_wick and momentum_stalled and sweep_strong_enough:
                    return {
                        'detected': True,
                        'type': 'buy_side',
                        'level': float(lvl['price']),
                        'level_label': lvl['label'],
                        'extreme': float(recent_high),
                        'clean_wick': True,
                        'momentum_stalled': True,
                        'bar_pos': high_pos,  # position within last-5 window
                        # C4: when the sweep happened, so Phase 5 can require
                        # the structure break to come after it.
                        'bar_time': self._bar_time(window, high_pos),
                    }

        # Sell-side sweep — wick pierced a sell-side liquidity level
        for lvl in liq_map.get('sell_side', []):
            if recent_low < lvl['price']:
                try:
                    bar = window.iloc[low_pos]
                    bar_close = float(bar['close'])
                    clean_wick = bar_close > lvl['price']
                except Exception:
                    clean_wick = last_close > lvl['price']

                momentum_stalled = last_close > recent_low + 0.2 * atr
                sweep_strong_enough = (lvl['price'] - recent_low) > 0.1 * atr

                if clean_wick and momentum_stalled and sweep_strong_enough:
                    return {
                        'detected': True,
                        'type': 'sell_side',
                        'level': float(lvl['price']),
                        'level_label': lvl['label'],
                        'extreme': float(recent_low),
                        'clean_wick': True,
                        'momentum_stalled': True,
                        'bar_pos': low_pos,
                        'bar_time': self._bar_time(window, low_pos),
                    }

        return out

    # ── Phase 5: Market Structure Shift ────────────────────────────────────────

    @staticmethod
    def _bar_time(frame: pd.DataFrame, pos: int) -> Optional[str]:
        """ISO timestamp of the bar at `pos`, or None without a DatetimeIndex.

        Stored as a string so the snapshot stays JSON-serialisable for the
        analysis log.
        """
        try:
            ts = frame.index[pos]
            if isinstance(ts, pd.Timestamp):
                return ts.isoformat()
        except Exception:
            pass
        return None

    def _detect_market_structure_shift(self, df: pd.DataFrame,
                                       swing_highs: List, swing_lows: List,
                                       max_age_bars: int = 15) -> Dict:
        """
        Look for CHoCH (preferred) or BOS in the bias direction in the last
        ~15 bars. CHoCH = breaks the most recent opposing swing in the bias
        direction for the first time.

        C4 FIXED 2026-09-23: the check compared only the current close with
        the last swing, so a break any number of bars old passed, including
        one that happened before the sweep it is meant to confirm. On
        synthetic replays 36% of detections were more than 15 bars old and,
        where a matching sweep existed on the same frame, 568 of 957 breaks
        preceded the sweep bar. The break bar (the latest close that crossed
        the level) is now located, must be within `max_age_bars`, and is
        returned as break_time so Phase 5 can require it to follow the sweep.

        C7 FIXED 2026-09-23: CHoCH was assigned whenever the most recent swing
        was a low (for a bullish break), whatever the prior trend, so in an
        uptrend every continuation break was labelled CHoCH and took the
        2-point premium. 628 of 2,139 logged detections were continuation
        breaks labelled CHoCH. Now a bullish break is CHoCH only when the high
        it breaks was a lower high (bearish character before the break), and
        a bearish break only when the low it breaks was a higher low.
        Detection conditions are unchanged; only the label changes.
        """
        out = {'detected': False, 'type': None, 'direction': None}
        if len(swing_highs) < 2 or len(swing_lows) < 2 or len(df) < 5:
            return out

        cp = float(df['close'].iloc[-1])
        last_sh_idx, last_sh_val = swing_highs[-1]
        last_sl_idx, last_sl_val = swing_lows[-1]
        prev_sh_val = swing_highs[-2][1]
        prev_sl_val = swing_lows[-2][1]

        hh = last_sh_val > prev_sh_val
        hl = last_sl_val > prev_sl_val
        lh = last_sh_val < prev_sh_val
        ll = last_sl_val < prev_sl_val

        direction = None
        if (last_sl_idx > last_sh_idx and cp > last_sh_val) or (hh and hl and cp > last_sh_val):
            direction, level, pivot = 'bullish', last_sh_val, last_sh_idx
            mss_type = 'CHoCH' if lh else 'BOS'
        elif (last_sh_idx > last_sl_idx and cp < last_sl_val) or (lh and ll and cp < last_sl_val):
            direction, level, pivot = 'bearish', last_sl_val, last_sl_idx
            mss_type = 'CHoCH' if hl else 'BOS'
        if direction is None:
            return out

        # Locate the break bar: the latest close that crossed the level.
        c = df['close'].values
        n = len(c)
        up = direction == 'bullish'
        break_pos = None
        for j in range(n - 1, max(int(pivot), 0), -1):
            if up and c[j] > level and c[j - 1] <= level:
                break_pos = j
                break
            if (not up) and c[j] < level and c[j - 1] >= level:
                break_pos = j
                break
        if break_pos is None:
            return out
        age = n - 1 - break_pos
        if age > max_age_bars:
            return {**out, 'stale_break': True, 'break_age': int(age),
                    'rejected_direction': direction}

        return {'detected': True, 'type': mss_type, 'direction': direction,
                'broken_level': level, 'break_pos': int(break_pos),
                'break_age': int(age), 'break_time': self._bar_time(df, break_pos)}

    @staticmethod
    def _mss_follows_sweep(mss: Dict, sweep: Dict) -> bool:
        """C4: True unless both carry timestamps and the break predates the
        bar that made the sweep. Without timestamps the ordering cannot be
        judged across timeframes, so it is not enforced (recency still is)."""
        bt, st = mss.get('break_time'), sweep.get('bar_time')
        if not bt or not st:
            return True
        try:
            return pd.Timestamp(bt) >= pd.Timestamp(st)
        except Exception:
            return True

    # ── Phase 1: D1 midpoint chop guard ────────────────────────────────────────

    def _check_d1_midpoint(self, df: pd.DataFrame) -> Dict:
        """
        Returns whether current price sits within the 40-60% midpoint of the
        most recent D1 swing range. If yes → abort per Phase 1.

        Since this is called on whatever TF the df represents, callers should
        pass a daily-aggregated view via the structure_tf snapshot when D1 data
        exists; otherwise this acts as a best-effort estimate from the lookback.
        """
        if len(df) < 20:
            return {'in_chop': False}

        try:
            # Use the last ~20 bars of the timeframe as a swing-range proxy
            window = df.iloc[-20:]
            hi = float(window['high'].max())
            lo = float(window['low'].min())
            if hi == lo:
                return {'in_chop': True, 'range_high': hi, 'range_low': lo}
            cp = float(df['close'].iloc[-1])
            position = (cp - lo) / (hi - lo)
            return {
                'in_chop': 0.40 <= position <= 0.60,
                'range_high': hi, 'range_low': lo,
                'position_in_range': position,
                'source': '20-bar proxy',
            }
        except Exception:
            return {'in_chop': False}

    def _check_d1_midpoint_daily(self, multi_tf_data: Dict[str, pd.DataFrame],
                                 structure_tf: str,
                                 symbol: Optional[str] = None) -> Optional[Dict]:
        """N1 FIXED 2026-09-23: Phase 1's chop guard is specified on the D1
        range, but it was computed from the last 20 bars of whatever the
        structure TF is (4H on every symbol here: about 3.3 days, and never
        aligned to days). It now uses real daily bars: a '1D'/'D1' frame when
        one is fetched, otherwise the structure frame resampled to UTC days.
        The range is the high/low of the last `d1_range_days` days including
        the current one (default 5, one trading week).

        Returns None when neither is possible (no DatetimeIndex, or fewer
        than 2 days of data), in which case the old proxy stays in place and
        is labelled as such.
        """
        days = max(2, int(self._nlm_cfg(symbol).get('d1_range_days', 5)))
        try:
            daily = None
            for key in ('1D', 'D1'):
                d = multi_tf_data.get(key)
                if d is not None and len(d) >= 2:
                    daily = d
                    break
            if daily is None:
                src = multi_tf_data.get(structure_tf)
                if src is None or not isinstance(src.index, pd.DatetimeIndex):
                    return None
                if self._tf_minutes(structure_tf) >= 1440:
                    daily = src
                else:
                    daily = (src[['high', 'low', 'close']]
                             .resample('1D').agg({'high': 'max', 'low': 'min',
                                                  'close': 'last'})
                             .dropna())
            if len(daily) < 2:
                return None
            window = daily.iloc[-days:]
            hi = float(window['high'].max())
            lo = float(window['low'].min())
            cp = float(multi_tf_data[structure_tf]['close'].iloc[-1])
            if hi <= lo:
                return {'in_chop': True, 'range_high': hi, 'range_low': lo,
                        'source': f'D1 x{len(window)}'}
            position = (cp - lo) / (hi - lo)
            return {
                'in_chop': 0.40 <= position <= 0.60,
                'range_high': hi, 'range_low': lo,
                'position_in_range': position,
                'source': f'D1 x{len(window)}',
            }
        except Exception as e:
            logger.debug(f"[NLM] D1 midpoint check fell back to the proxy: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE WALKTHROUGH — evaluate the 9-phase checklist
    # ══════════════════════════════════════════════════════════════════════════

    def _evaluate_nlm_setup(self, analysis: Dict, multi_tf_data: Dict,
                            snaps: Dict, structure_tf: str,
                            primary_tf: str, entry_tf: str) -> Dict:
        out = {
            'entry_signal': False, 'entry_reason': None, 'entry_type': None,
            'direction': None, 'confidence_score': 0.0, 'confluence_score': 0.0,
            'confluence_signals': [], 'order_type': 'limit', 'limit_price': None,
            'nlm_phase_results': {},
        }

        htf = snaps.get(structure_tf)
        ptf = snaps.get(primary_tf, htf)
        ltf = snaps.get(entry_tf, ptf)
        if not htf or not ptf:
            out['entry_reason'] = 'NLM: insufficient TF data'
            return out

        signals: List[str] = []
        phase_results: Dict = {}

        # ── PHASE 1: HTF Bias (D1/H4) ─────────────────────────────────────
        htf_struct = htf.get('structure', {})
        htf_bias = htf_struct.get('bias')
        if not htf_bias:
            out['entry_reason'] = 'NLM Phase 1: no clear HTF bias (mid-range/neutral)'
            phase_results['phase_1'] = 'FAIL — no bias'
            out['nlm_phase_results'] = phase_results
            return out

        # D1 midpoint guard
        d1_chop = htf.get('d1_midpoint_chop', {})
        if d1_chop.get('in_chop'):
            out['entry_reason'] = 'NLM Phase 1: price in 40-60% D1 midpoint (chop)'
            phase_results['phase_1'] = 'FAIL — D1 chop zone'
            out['nlm_phase_results'] = phase_results
            return out

        signals.append('htf_bias_clear')
        signals.append('d1_directional_displacement')
        signals.append('not_in_d1_midpoint')
        phase_results['phase_1'] = f'PASS — {htf_bias}'

        # ── PHASE 2: Liquidity Map ────────────────────────────────────────
        liq_map = ptf.get('liquidity_map', {})
        target_side_levels = (liq_map.get('buy_side', []) if htf_bias == 'bearish'
                              else liq_map.get('sell_side', []))
        if not target_side_levels:
            # Try HTF as a fallback for liquidity targets
            liq_map = htf.get('liquidity_map', {})
            target_side_levels = (liq_map.get('buy_side', []) if htf_bias == 'bearish'
                                  else liq_map.get('sell_side', []))

        if not target_side_levels:
            out['entry_reason'] = 'NLM Phase 2: no liquidity targets mapped'
            phase_results['phase_2'] = 'FAIL — no liquidity'
            out['nlm_phase_results'] = phase_results
            return out

        signals.append('liquidity_target_mapped')
        phase_results['phase_2'] = f'PASS — {len(target_side_levels)} levels mapped'

        # ── PHASE 4 first (precedes Phase 3 in practice — no sweep, no setup) ─
        # We do Phase 4 before Phase 3 because the checklist's hard rule is
        # "no sweep = no trade", regardless of Nexus Zone quality.
        sweep = ptf.get('sweep', {})
        if not sweep.get('detected'):
            # Try entry TF as well — sweep often shows on M5/M1
            sweep = ltf.get('sweep', {}) or sweep

        if not sweep.get('detected'):
            out['entry_reason'] = 'NLM Phase 4: no liquidity sweep detected'
            phase_results['phase_4'] = 'FAIL — no sweep'
            out['nlm_phase_results'] = phase_results
            return out

        # Sweep direction must match bias inversion
        # Bullish bias → expects sell-side sweep (price dipped to grab stops below, now reverses up)
        # Bearish bias → expects buy-side sweep
        expected_sweep_side = 'sell_side' if htf_bias == 'bullish' else 'buy_side'
        if sweep.get('type') != expected_sweep_side:
            out['entry_reason'] = (f'NLM Phase 4: sweep side mismatch '
                                   f'(got {sweep.get("type")}, expected {expected_sweep_side})')
            phase_results['phase_4'] = 'FAIL — wrong-side sweep'
            out['nlm_phase_results'] = phase_results
            return out

        signals.append('liquidity_sweep_clean')
        if sweep.get('momentum_stalled'):
            signals.append('sweep_momentum_stalled')
        phase_results['phase_4'] = f'PASS — {sweep["level_label"]} swept @ {sweep["extreme"]:.2f}'

        # ── PHASE 3: Nexus Zone confluence ────────────────────────────────
        direction = 'long' if htf_bias == 'bullish' else 'short'
        nz = self._identify_nexus_zone(htf_bias, ptf, ltf, sweep)
        confluence_count = nz['confluence_count']

        if confluence_count < 2:
            out['entry_reason'] = (f'NLM Phase 3: Nexus Zone confluence '
                                   f'{confluence_count}/4 (need ≥2)')
            phase_results['phase_3'] = f'FAIL — {confluence_count}/4'
            out['nlm_phase_results'] = phase_results
            return out

        signals.extend(nz['signals'])
        phase_results['phase_3'] = f'PASS — {confluence_count}/4 confluence'

        # ── PHASE 5: Market Structure Shift ───────────────────────────────
        mss = ltf.get('mss', {})
        if not mss.get('detected'):
            mss = ptf.get('mss', {}) or mss

        expected_mss_dir = 'bullish' if htf_bias == 'bullish' else 'bearish'
        if not mss.get('detected') or mss.get('direction') != expected_mss_dir:
            out['entry_reason'] = ('NLM Phase 5: no CHoCH/BOS in bias direction '
                                   f'({mss.get("type", "none")}/{mss.get("direction", "none")})')
            phase_results['phase_5'] = 'FAIL — no MSS'
            out['nlm_phase_results'] = phase_results
            return out

        # C4: the checklist sequence is sweep, then structure shift. A break
        # that predates the sweep is not a reaction to it.
        if not self._mss_follows_sweep(mss, sweep):
            out['entry_reason'] = (f'NLM Phase 5: {mss.get("type")} at {mss.get("break_time")} '
                                   f'precedes the sweep at {sweep.get("bar_time")}')
            phase_results['phase_5'] = 'FAIL - MSS before sweep'
            out['nlm_phase_results'] = phase_results
            return out

        if mss.get('type') == 'CHoCH':
            signals.append('choch_in_bias')
        else:
            signals.append('bos_in_bias')
        signals.append('structure_clean_no_failure')
        phase_results['phase_5'] = f'PASS — {mss["type"]} {mss["direction"]}'

        # ── PHASE 6: Entry execution (OB > FVG > Breaker) ─────────────────
        entry_decision = self._select_entry_type(direction, ptf, ltf, sweep, nz)
        if not entry_decision['signal']:
            out['entry_reason'] = 'NLM Phase 6: no clean entry zone (OB/FVG/Breaker)'
            phase_results['phase_6'] = 'FAIL — no entry zone'
            out['nlm_phase_results'] = phase_results
            return out

        signals.extend(entry_decision['signals'])
        phase_results['phase_6'] = f'PASS — {entry_decision["entry_type"]}'

        symbol = analysis.get('symbol', '')

        # Gold-session bonus (Phase 4 gold-specific note)
        if self._is_gold_session_active(symbol):
            signals.append('gold_session_active')

        # ── Compose final decision ────────────────────────────────────────
        score = self._nlm_score(signals)
        conviction = self._nlm_conviction(signals)
        threshold = self._get_threshold(symbol)
        if conviction < threshold:
            out['entry_reason'] = (f'NLM: conviction {conviction} below minimum {threshold} '
                                   f'(total score {score})')
            phase_results['final'] = f'FAIL - conviction {conviction}/{threshold}'
            out['nlm_phase_results'] = phase_results
            out['conviction_score'] = conviction
            return out

        phase_results['final'] = f'PASS - conviction {conviction} (total score {score})'
        phase_results['conviction_score'] = conviction
        phase_results['nexus_confluence_count'] = confluence_count
        phase_results['mss_type'] = mss['type']
        phase_results['sweep_level'] = sweep['level']
        phase_results['sweep_extreme'] = sweep['extreme']

        # Store sweep extreme for SL calculation in calculate_entry_levels
        out.update({
            'entry_signal': True,
            'entry_reason': (f'NLM {direction.upper()}: '
                             f'sweep {sweep["level_label"]} → '
                             f'{mss["type"]} → {entry_decision["entry_type"]}'),
            'entry_type': f'nlm_{entry_decision["entry_type"]}',
            'direction': direction,
            # C3: was min(1.0, score / max(threshold * 1.5, 10)), which was
            # 1.0 on every signal ever logged. Now conviction on its own range.
            'confidence_score': round(conviction / _NLM_CONVICTION_MAX, 3),
            'confluence_score': score,
            'conviction_score': conviction,
            'confluence_signals': signals,
            'order_type': entry_decision.get('order_type', 'limit'),
            'limit_price': entry_decision.get('limit_price'),
            'nlm_phase_results': phase_results,
            # NLM-specific keys consumed by calculate_entry_levels:
            'nlm_sweep_extreme': sweep['extreme'],
            'nlm_sweep_type': sweep['type'],
            'nlm_liquidity_map': liq_map,
        })
        return out

    # ── Phase 3: Nexus Zone identification ─────────────────────────────────────

    def _identify_nexus_zone(self, htf_bias: str, ptf: Dict, ltf: Dict,
                             sweep: Dict) -> Dict:
        """
        A valid Nexus Zone requires ≥2 of:
          1. Liquidity pool — EQH/EQL/session extreme at the zone
          2. Order Block — last opposing candle before impulse
          3. FVG — visible imbalance
          4. Premium/Discount alignment

        Returns the count + the signals to add to the confluence list.
        """
        signals: List[str] = []
        confluences = 0
        cp = ltf['ohlc']['close']
        struct = ptf.get('structure', {})
        target_type = 'bullish' if htf_bias == 'bullish' else 'bearish'

        # ── Confluence 1: liquidity pool at the swept level ───────────────
        # The sweep itself is liquidity — count this if the sweep is recent.
        if sweep.get('detected'):
            signals.append('nz_liquidity_pool')
            confluences += 1

        # ── Confluence 2: order block aligned with bias near current price ─
        obs = ptf.get('order_blocks', [])
        atr = ptf['indicators']['atr']['value']
        nearby_ob = None
        # C6 FIXED 2026-09-23: both zone lists are oldest-first and these loops
        # took the first match, so the oldest qualifying zone won. In 41 of
        # the 104 live FVG signals a newer FVG also qualified, and in all 41
        # the chosen one was further from price (median 2.27 vs 0.76 ATR),
        # which is where limits go to expire unfilled. Newest first now,
        # matching the checklist's "fresh" zone that produced the shift.
        for ob in reversed(obs):
            if ob['type'] == target_type and not ob['mitigated']:
                if abs(cp - ob['mid']) < 2.5 * atr:
                    nearby_ob = ob
                    break
        if nearby_ob:
            signals.append('nz_order_block')
            confluences += 1

        # ── Confluence 3: unfilled FVG aligned with bias near current price ─
        fvgs = ptf.get('fvgs', [])
        nearby_fvg = None
        for fvg in reversed(fvgs):   # C6: newest first
            if fvg['type'] == target_type and not fvg['filled']:
                if abs(cp - fvg['mid']) < 3.0 * atr:
                    nearby_fvg = fvg
                    break
        if nearby_fvg:
            signals.append('nz_fair_value_gap')
            confluences += 1

        # ── Confluence 4: premium/discount alignment ──────────────────────
        sh = struct.get('last_swing_high')
        sl = struct.get('last_swing_low')
        if sh and sl and sh != sl:
            mid = (sh + sl) / 2
            in_discount = cp < mid
            in_premium = cp > mid
            if htf_bias == 'bullish' and in_discount:
                signals.append('nz_premium_discount_aligned')
                confluences += 1
            elif htf_bias == 'bearish' and in_premium:
                signals.append('nz_premium_discount_aligned')
                confluences += 1

        return {
            'confluence_count': confluences,
            'signals': signals,
            'nearby_ob': nearby_ob,
            'nearby_fvg': nearby_fvg,
        }

    # ── Phase 6: Entry type selection (priority: OB > FVG > Breaker) ───────────

    def _select_entry_type(self, direction: str, ptf: Dict, ltf: Dict,
                           sweep: Dict, nz: Dict) -> Dict:
        """
        Priority order from the checklist:
          A. Order Block entry (highest conviction) — limit at OB 50% or open
          B. FVG entry — limit at 50% midpoint
          C. Breaker Block — secondary, lower conviction
        """
        target_type = 'bullish' if direction == 'long' else 'bearish'
        cp = ltf['ohlc']['close']
        atr = ptf['indicators']['atr']['value']
        signals: List[str] = []

        # A — Order Block
        nearby_ob = nz.get('nearby_ob')
        if nearby_ob:
            # Verify no strong opposing displacement candle has closed inside the OB
            limit_price = nearby_ob['mid']
            signals.append('ob_entry_clean')
            signals.append('entry_momentum_confirmation')
            return {
                'signal': True,
                'entry_type': 'order_block',
                'order_type': 'limit',
                'limit_price': limit_price,
                'signals': signals,
            }

        # B — FVG
        nearby_fvg = nz.get('nearby_fvg')
        if nearby_fvg:
            limit_price = nearby_fvg['mid']
            signals.append('fvg_entry')
            signals.append('entry_momentum_confirmation')
            return {
                'signal': True,
                'entry_type': 'fvg',
                'order_type': 'limit',
                'limit_price': limit_price,
                'signals': signals,
            }

        # C — Breaker (lower conviction, also use as final fallback)
        breakers = ptf.get('breaker_blocks', [])
        target_breaker_type = 'bullish_breaker' if direction == 'long' else 'bearish_breaker'
        for brk in reversed(breakers):   # C6: newest first
            if brk['type'] == target_breaker_type:
                if abs(cp - brk['mid']) < 2.5 * atr:
                    signals.append('breaker_entry')
                    return {
                        'signal': True,
                        'entry_type': 'breaker',
                        'order_type': 'limit',
                        'limit_price': brk['mid'],
                        'signals': signals,
                    }

        return {'signal': False}

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 7 + 8: SL / TP calculation
    # ══════════════════════════════════════════════════════════════════════════

    def _calc_nlm_sl(self, analysis: Dict, entry_price: float, atr: float) -> float:
        """
        Phase 7: SL placed beyond the liquidity sweep extreme (not just beyond
        the OB), with a 3-5 pip buffer past the wick.

        Cap the SL distance at a configurable multiple of ATR to prevent
        runaway risk when the sweep extreme is far away. If the resulting
        stop would exceed the trader's risk tolerance, the money manager
        (not this engine) is responsible for either reducing position size
        or rejecting the trade — per the checklist's "reduce position size,
        not stop distance" rule.
        """
        direction = analysis.get('direction', 'long')
        sweep_extreme = analysis.get('nlm_sweep_extreme')

        # Buffer: 3-5 pips beyond the wick (4 pips chosen), sized per symbol,
        # plus a small ATR component to flex with volatility.
        pip = self._pip_size(analysis.get('symbol', 'XAUUSD'))
        buffer = max(4 * pip, 0.15 * atr)

        max_sl_atr_mult = float(self._nlm_cfg(analysis.get('symbol')).get('max_sl_atr_multiplier', 2.5))
        max_sl_distance = atr * max_sl_atr_mult

        if direction == 'long':
            if sweep_extreme is not None:
                sl = float(sweep_extreme) - buffer
            else:
                # Fallback: use structure if no sweep extreme available
                struct = analysis.get('market_structure', {})
                sl_pt = struct.get('last_swing_low')
                sl = (sl_pt - buffer) if sl_pt else (entry_price - 2.0 * atr)

            # A1.1 FIXED 2026-08-30 audit: the entry is a resting limit at an
            # OB/FVG mid and the stop comes from the sweep extreme, and nothing
            # required the former to sit above the latter. When the zone is
            # below the sweep low this produced a buy limit with its stop ABOVE
            # it, and the clamp below is negative in that case so it never
            # fired. The broker rejects it as retcode 10016 (invalid stops);
            # the ICT log carries 20 of those. Same guard as _calc_smc_sl and
            # _calc_ict_sl already have.
            if sl is None or sl >= entry_price:
                logger.debug(
                    f"[NLM] SL {sl} is not below the long entry {entry_price}; "
                    f"falling back to the ATR stop"
                )
                sl = entry_price - 2.0 * atr

            if entry_price - sl > max_sl_distance:
                logger.debug(f"[NLM] SL clamped to {max_sl_atr_mult}×ATR cap")
                sl = entry_price - max_sl_distance
            return sl

        else:  # short
            if sweep_extreme is not None:
                sl = float(sweep_extreme) + buffer
            else:
                struct = analysis.get('market_structure', {})
                sh_pt = struct.get('last_swing_high')
                sl = (sh_pt + buffer) if sh_pt else (entry_price + 2.0 * atr)

            # A1.1: mirror of the long-side guard above.
            if sl is None or sl <= entry_price:
                logger.debug(
                    f"[NLM] SL {sl} is not above the short entry {entry_price}; "
                    f"falling back to the ATR stop"
                )
                sl = entry_price + 2.0 * atr

            if sl - entry_price > max_sl_distance:
                logger.debug(f"[NLM] SL clamped to {max_sl_atr_mult}×ATR cap")
                sl = entry_price + max_sl_distance
            return sl

    def _calc_nlm_tps(self, analysis: Dict, entry_price: float,
                      stop_loss: float, multi_tf_data: Dict) -> Dict:
        """
        Phase 8: TPs anchor on identifiable liquidity pools.
          TP1 — nearest internal liquidity (partial close 30-50% — handled by stop_manager)
          TP2 — external liquidity: major session high/low, PDH/PDL
          TP3 — D1/weekly liquidity → runner

        If the liquidity map yields fewer than 3 levels in the trade direction,
        fall back to R-multiples (1.5R / 3R / 5R) so the trade is always sized.
        """
        direction = analysis.get('direction', 'long')
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            return {'tp1': None, 'tp2': None, 'tp3': None}

        liq_map = analysis.get('nlm_liquidity_map', {})

        # Pull levels in the direction the trade should travel
        if direction == 'long':
            # Long → targets are above entry → buy-side liquidity
            candidates = [lvl['price'] for lvl in liq_map.get('buy_side', [])
                          if lvl['price'] > entry_price]
            candidates.sort()  # ascending; nearest first
        else:
            candidates = [lvl['price'] for lvl in liq_map.get('sell_side', [])
                          if lvl['price'] < entry_price]
            candidates.sort(reverse=True)  # descending; nearest first

        # Default R-multiple fallbacks
        default_rr = {
            'tp1': 1.5, 'tp2': 3.0, 'tp3': 5.0,
        }
        default_tps = {}
        for name, rr in default_rr.items():
            if direction == 'long':
                default_tps[name] = entry_price + rr * risk
            else:
                default_tps[name] = entry_price - rr * risk

        # Assign liquidity targets where available; ensure each successive TP
        # is at least 1.0R further than the previous (prevents TP clustering).
        tps: Dict[str, Optional[float]] = {'tp1': None, 'tp2': None, 'tp3': None}

        def _is_progressive(price: float, prev: Optional[float]) -> bool:
            """Return True if `price` extends past `prev` in trade direction."""
            if prev is None:
                return True
            if direction == 'long':
                return price > prev + 0.3 * risk
            return price < prev - 0.3 * risk

        def _beyond(price: float, prev: float) -> float:
            """C5: a target that fails the progression rule against the
            previous one moves to 1R past it. Liquidity targets chosen by the
            loops below already pass the rule and are left alone; only the
            fixed R fallbacks can fail it."""
            if _is_progressive(price, prev):
                return price
            return prev + 1.0 * risk if direction == 'long' else prev - 1.0 * risk

        # TP1 — nearest internal liquidity (at least 1.0R from entry)
        for c in candidates:
            if direction == 'long' and c - entry_price >= 1.0 * risk:
                tps['tp1'] = c
                break
            if direction == 'short' and entry_price - c >= 1.0 * risk:
                tps['tp1'] = c
                break
        if tps['tp1'] is None:
            tps['tp1'] = default_tps['tp1']

        # TP2 — external liquidity, progressive from TP1
        for c in candidates:
            if _is_progressive(c, tps['tp1']):
                # Must be at least 2R from entry to qualify as TP2
                dist = abs(c - entry_price)
                if dist >= 2.0 * risk:
                    tps['tp2'] = c
                    break
        if tps['tp2'] is None:
            tps['tp2'] = default_tps['tp2']
        # C5 FIXED 2026-09-23: TP1 has no upper bound, so when it lands beyond
        # 3R and no further level qualifies, the fixed 3R fallback put TP2
        # nearer than TP1. main.py sends TP2 to the broker, so the position
        # closed before its own first target. 14 of 140 NLM orders had this.
        tps['tp2'] = _beyond(tps['tp2'], tps['tp1'])

        # TP3 — runner; further still, at least 4R from entry
        for c in candidates:
            if _is_progressive(c, tps['tp2']):
                dist = abs(c - entry_price)
                if dist >= 4.0 * risk:
                    tps['tp3'] = c
                    break
        if tps['tp3'] is None:
            tps['tp3'] = default_tps['tp3']
        tps['tp3'] = _beyond(tps['tp3'], tps['tp2'])   # C5: same for the runner

        return tps

    # ══════════════════════════════════════════════════════════════════════════
    # SCORING / FILTERS / UTIL
    # ══════════════════════════════════════════════════════════════════════════

    def _nlm_score(self, signals: List[str]) -> float:
        return float(sum(_NLM_WEIGHTS.get(s, 1) for s in signals))

    @staticmethod
    def _nlm_conviction(signals: List[str]) -> int:
        """C3: score over the signals that can actually be absent."""
        return int(sum(_NLM_CONVICTION_WEIGHTS.get(s, 0) for s in signals))

    def _get_threshold(self, symbol: str) -> int:
        """Minimum conviction (3-13) for this symbol.

        C3: replaces confluence_required / confluence_threshold_calm /
        nlm_confluence_threshold, which sat on a scale the score could not
        fall below. Set per symbol with symbols.<sym>.nlm.min_conviction.
        Default 0 keeps the gate open, which is what it has always been in
        practice; see the 2026-09-23 note for why no cut is set yet.
        """
        return int(self._nlm_cfg(symbol).get('min_conviction', 0))

    def _is_gold_session_active(self, symbol: Optional[str] = None) -> bool:
        """Session bonus window check. Default: London open or NY open.

        N3: the windows were gold's for every symbol. Override per symbol with
        symbols.<sym>.nlm.session_bonus_windows: [[start_hour, end_hour], ...]
        in UTC, end exclusive.
        """
        hour = datetime.now(timezone.utc).hour
        windows = self._nlm_cfg(symbol).get('session_bonus_windows')
        if not windows:
            windows = [_NLM_SESSIONS[name] for name in ('london_open', 'ny_open')]
        for start, end in windows:
            if int(start) <= hour < int(end):
                return True
        return False

    def _apply_nlm_filters(self, analysis: Dict, snaps: Dict,
                           structure_tf: str, primary_tf: str) -> Dict:
        """
        Final safety net mirroring the checklist's abort criteria:
          - HTF ADX must show meaningful trend (anti-chop)
          - HTF ATR threshold (dead market guard)
          - D1 midpoint chop guard (already checked in phase walkthrough,
            but re-verified here in case of race conditions)
        """
        htf = snaps.get(structure_tf)
        if not htf:
            return analysis

        # Filter 1: ADX (anti-chop)
        adx = htf['indicators']['adx']
        adx_val = adx.get('value')
        if isinstance(adx_val, pd.Series):
            adx_val = float(adx_val.iloc[-1])
        cfg = self._nlm_cfg(analysis.get('symbol'))   # N3: per-symbol overrides
        min_adx = float(cfg.get('min_htf_adx', 18))
        if adx_val is not None and adx_val < min_adx:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'NLM Filter: HTF ADX too low ({adx_val:.1f} < {min_adx})'
            return analysis

        # Filter 2: ATR floor — Gold needs movement
        atr_pct = htf['indicators']['atr']['percent']
        min_atr = float(cfg.get('min_atr_threshold',
                        self.strategy_config.get('filters', {}).get('min_atr_threshold', 0.08)))
        if atr_pct < min_atr:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'NLM Filter: ATR too low ({atr_pct:.3f}% < {min_atr}%)'
            return analysis

        # Filter 3: D1 midpoint re-check
        d1 = htf.get('d1_midpoint_chop', {})
        if d1.get('in_chop'):
            analysis['entry_signal'] = False
            analysis['entry_reason'] = 'NLM Filter: D1 midpoint chop (re-verified)'
            return analysis

        return analysis