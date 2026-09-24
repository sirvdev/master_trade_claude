"""
strategy/nlm_matrix_engine.py - The Nexus Liquidity Matrix Strategy, book v1.0
===============================================================================
Implements "The Nexus Liquidity Matrix Strategy - A Comprehensive Technical
Manual for Institutional Order Flow Trading", Specification v1.0 (Expanded
Master Edition), as a drop-in StrategyEngine.

Every gate below is one question of the book's Part XIII pre-flight checklist.
Any "NO" is a stand-down (Part 6.5): there is no scoring, no weighting and no
confluence count anywhere in this file.

  STAGE 1  MACRO ALIGNMENT
    G1  1H/4H macro bias confirmed ................................ 4.1, 4.2, 4.4
    G2  Nexus Axis identifiable on the HTF ........................ 3.1, 2.3, 2.4
    G3  HTF Mitigating Node present beyond the axis ............... 3.3
    G4  Velocity Factor audited on the approach ................... 3.2, 8.1 step 4
  STAGE 2  LIQUIDITY SWEEP VALIDATION
    G5  Breach of the axis by a spike ............................. 5.1, 5.2
    G6  Body closes back inside, same or next candle .............. 5.1, 5.3
    G7  No acceptance / no body close outside, no both-side sweep . 5.3, 6.3, 10.6
    G8  Rule 3: target not reached before the sweep ................ 6.2
  STAGE 3  LOWER TIMEFRAME CONFIRMATION
    G9  5M Market Structure Shift body close ...................... 2.6, 4.3
    G10 Displacement candle expansion ............................. 2.6, 6.4
    G11 FVG formed by the displacement move ....................... 2.7, 8.2
  STAGE 4  RISK
    G12 Limit at the FVG 50% equilibrium .......................... 8.2
    G13 SL 1.5 pips beyond the absolute sweep extreme ............. 8.3
    G14 R:R to the target >= 1:3 ................................... 9.4
    Risk 0.5-1.0% is enforced by risk_management (0.6% configured) 9.1

Deviation by owner instruction (2026-09-23):
  * Take profit is a fixed 1:5 R:R instead of the opposing HTF node (8.4).
    The 9.4 audit (>= 1:3) is applied to that target, and Rule 3 (6.2) uses
    it as the "target destination".
Execution note:
  * 9.3 (close 50% at 1:2, stop to breakeven) is executed as two orders of
    equal size, targets 1:2 and 1:5, with the EA moving the survivor's stop
    to entry at 1:2 (see calculate_entry_levels 'legs').

Where the book states a rule but not a number, the number chosen is in
DEFAULTS with the reason, and every one can be overridden in config
`nlm_matrix:` or per symbol in `symbols.<sym>.nlm_matrix:`.

All times are UTC. Every frame passed in must contain CLOSED bars only (the
market client already drops the forming bar).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULTS: Dict = {
    # Timeframes (4.1): macro bias 4H or 1H, execution 5M or 1M.
    'macro_tf': '4H',          # 4H must not contradict the 1H bias (4.4)
    'bias_tf': '1H',           # the journal template records "1H Macro Bias"
    'sweep_tf': '1H',          # 10.1 source execution model verifies the sweep on 1H
    'session_tf': '15m',       # finest frame with 2+ days of history, for session extremes
    'exec_tf': '5m',           # 4.3, 8.2 "5M Fair Value Gap"
    # Swing point (2.4): "three-to-five candle sequence" -> 5-candle fractal.
    'swing_order': 2,
    # Stop (8.3) and targets.
    'sl_buffer_pips': 1.5,     # 8.3 exact
    'min_rr_to_node': 3.0,     # 9.4 exact (minimum R:R to the target)
    'target_rr': 5.0,          # owner instruction, replaces 8.4 node target
    'partial_rr': 2.0,         # 9.3 exact
    'partial_fraction': 0.5,   # 9.3 "e.g. 50%"
    # Velocity Factor (3.2): the book's own optional measurement tool.
    'velocity_candles': 3,
    'velocity_atr_period': 20,
    'velocity_ratio': 1.5,
    # Displacement (2.6 "aggressive, large-bodied candle expansion"). The
    # book gives no number. Interpretation: the expansion threshold reuses the
    # book's only volatility number (1.5 x ATR20, from 3.2), and
    # "large-bodied" is read as body >= half the candle's range.
    'displacement_atr_ratio': 1.5,
    'displacement_body_frac': 0.5,
    # How long after the reclaim candle the 5M MSS may form. The book puts
    # MSS immediately after the sweep (4.3) but gives no clock. Interpretation:
    # within the 1H candle that follows the reclaim candle.
    'mss_window_bars': 1,
    # Sessions (2.4, 7.2). Book fixes NY open at 13:30 UTC; the rest are the
    # standard session clocks, UTC.
    'sessions': {'asian': ['00:00', '07:00'],
                 'london': ['07:00', '13:30'],
                 'new_york': ['13:30', '21:00']},
    'session_days': 2,         # completed sessions of today and yesterday
    # Psychological round numbers (3.1 "ending in .000 or .500"); step per
    # symbol in price units, 0 disables. Only the FX step is literal.
    'round_step': 0.0,
    # Pip (8.3, 5.2). Only FX (0.0001) and index points (1.0, 10.3) are in
    # the book; everything else is set per symbol.
    'pip': 0.0001,
    'rule3_enabled': True,
}

_TF_MIN = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1H': 60, '4H': 240, '1D': 1440,
           'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440}


# ═════════════════════════════════════════════════════════════════════════════
# Primitives
# ═════════════════════════════════════════════════════════════════════════════

def swing_points(df: pd.DataFrame, k: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """2.4: a swing high is a candle whose high is above the k candles on each
    side (k=2 gives the book's five-candle sequence). Only confirmed swings are
    returned: the last k bars cannot be swings yet."""
    h, l = df['high'].values, df['low'].values
    n = len(df)
    highs, lows = [], []
    for i in range(k, n - k):
        if all(h[i] > h[i - j] for j in range(1, k + 1)) and all(h[i] > h[i + j] for j in range(1, k + 1)):
            highs.append((i, float(h[i])))
        if all(l[i] < l[i - j] for j in range(1, k + 1)) and all(l[i] < l[i + j] for j in range(1, k + 1)):
            lows.append((i, float(l[i])))
    return highs, lows


def structure_bias(df: pd.DataFrame, k: int) -> Optional[str]:
    """4.2: HH + HL = bullish, LH + LL = bearish, anything else = no bias."""
    sh, sl = swing_points(df, k)
    if len(sh) < 2 or len(sl) < 2:
        return None
    hh, lh = sh[-1][1] > sh[-2][1], sh[-1][1] < sh[-2][1]
    hl, ll = sl[-1][1] > sl[-2][1], sl[-1][1] < sl[-2][1]
    if hh and hl:
        return 'bullish'
    if lh and ll:
        return 'bearish'
    return None


def atr(df: pd.DataFrame, period: int, end: int) -> float:
    """Simple ATR over the `period` bars ending at position `end` inclusive."""
    if end < 1:
        return 0.0
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    start = max(1, end - period + 1)
    tr = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(start, end + 1)]
    return float(np.mean(tr)) if tr else 0.0


def true_range(df: pd.DataFrame, i: int) -> float:
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    if i == 0:
        return float(h[0] - l[0])
    return float(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))


def _t(ts) -> str:
    return pd.Timestamp(ts).isoformat()


# ═════════════════════════════════════════════════════════════════════════════
# Engine
# ═════════════════════════════════════════════════════════════════════════════

class NLMMatrixEngine:
    """Nexus Liquidity Matrix, book v1.0. Same public API as StrategyEngine."""

    ENTRY_TYPE = 'nlm_matrix_fvg'

    def __init__(self, config: Dict):
        self.config = config or {}
        self.base_cfg = dict(DEFAULTS)
        self.base_cfg.update(self.config.get('nlm_matrix', {}) or {})

    # ── configuration ─────────────────────────────────────────────────────────
    def cfg(self, symbol: Optional[str]) -> Dict:
        c = dict(self.base_cfg)
        if symbol:
            sc = (self.config.get('symbols', {}) or {}).get(symbol, {}) or {}
            c.update(sc.get('nlm_matrix', {}) or {})
        return c

    @staticmethod
    def _tf_minutes(tf: str) -> int:
        return _TF_MIN.get(tf, 60)

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def analyze_market(self, symbol: str, multi_tf_data: Dict[str, pd.DataFrame],
                       symbol_config: Optional[Dict] = None) -> Dict:
        c = self.cfg(symbol)
        analysis = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now('UTC').tz_localize(None).isoformat(),
            'primary_timeframe': c['exec_tf'],
            'entry_signal': False, 'entry_reason': None, 'entry_type': None,
            'direction': None, 'order_type': 'limit', 'limit_price': None,
            'confidence_score': 0.0, 'confluence_score': 0.0, 'confluence_signals': [],
            'timeframe_snapshots': {}, 'market_structure': {}, 'indicators_state': {},
            'nlm_checklist': {},
        }
        try:
            decision = self.evaluate(symbol, multi_tf_data, c)
            analysis.update(decision)
        except Exception as e:                     # never let the engine kill the loop
            logger.error(f"[NLM-M] {symbol} evaluation error: {e}", exc_info=True)
            analysis['entry_reason'] = f'NLM-M error: {e}'
        analysis['market_structure'] = {'checklist': analysis.get('nlm_checklist', {})}
        logger.info(f"[NLM-M] {symbol} signal={analysis['entry_signal']} "
                    f"dir={analysis['direction']} reason={analysis['entry_reason']}")
        return analysis

    def calculate_entry_levels(self, analysis: Dict, multi_tf_data: Dict) -> Dict:
        """8.2 entry, 8.3 stop, owner-set 1:5 target, 9.3 partial as two legs."""
        s = analysis.get('nlm_setup')
        if not s:
            return {}
        return {
            'entry_price': s['last_close'],
            'order_price': s['entry'],
            'order_type': 'limit',
            'limit_price': s['entry'],
            'stop_loss': s['sl'],
            'take_profit_1': s['tp_partial'],
            'take_profit_2': s['tp'],
            'take_profit_3': None,
            'atr': s['exec_atr'],
            'risk_distance': abs(s['entry'] - s['sl']),
            # 9.3: close 50% at 1:2 and move the stop to entry. Executed as two
            # orders of equal size; the EA's BreakevenMinRR=2.0 moves the
            # surviving leg's stop to entry when the first leg's target fills.
            'legs': [
                {'fraction': s['partial_fraction'], 'take_profit': s['tp_partial'], 'label': 'P2R'},
                {'fraction': 1.0 - s['partial_fraction'], 'take_profit': s['tp'], 'label': 'T5R'},
            ],
            'nlm_setup': s,
        }

    def pending_order_valid(self, setup: Dict, multi_tf_data: Dict) -> Tuple[bool, str]:
        """Rules for a resting limit that has not filled yet.

        8.2  price moved past without retracing: it reached the 1:2 level
             before the limit filled -> the trade is missed.
        6.3 / 10.4  a sweep-TF candle body closes beyond the axis -> breakout,
             cancel all pending orders.
        6.1  a new HTF structural high/low against the trade voids it. For a
             resting short that event is a close above the swept axis, which
             the check above already cancels. (Recomputing the two-swing bias
             is not used here: the sweep itself prints a higher high, which
             4.2's second route counts as bearish, not as a bias change.)
        """
        c = self.cfg(setup.get('symbol'))
        ex = multi_tf_data.get(c['exec_tf'])
        sw = multi_tf_data.get(c['sweep_tf'])
        placed = pd.Timestamp(setup['signal_time'])
        short = setup['direction'] == 'short'
        if ex is not None and len(ex):
            after = ex[ex.index > placed]
            if len(after):
                if short and float(after['low'].min()) <= setup['tp_partial']:
                    return False, 'missed: price reached 1:2 without retracing to entry (8.2)'
                if (not short) and float(after['high'].max()) >= setup['tp_partial']:
                    return False, 'missed: price reached 1:2 without retracing to entry (8.2)'
        if sw is not None and len(sw):
            after = sw[sw.index >= pd.Timestamp(setup['reclaim_end'])]
            if len(after):
                if short and float(after['close'].max()) > setup['axis']:
                    return False, 'body close back above the axis: breakout (6.3, 10.4)'
                if (not short) and float(after['close'].min()) < setup['axis']:
                    return False, 'body close back below the axis: breakout (6.3, 10.4)'
        return True, ''

    # ══════════════════════════════════════════════════════════════════════════
    # CHECKLIST
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate(self, symbol: str, data: Dict[str, pd.DataFrame], c: Dict) -> Dict:
        out = {'entry_signal': False, 'direction': None, 'entry_reason': None,
               'nlm_checklist': {}}
        chk = out['nlm_checklist']
        k = int(c['swing_order'])

        H = data.get(c['bias_tf'])
        F = data.get(c['macro_tf'])
        S = data.get(c['sweep_tf'])
        Q = data.get(c['session_tf'])
        E = data.get(c['exec_tf'])
        for name, df in (('bias', H), ('macro', F), ('sweep', S), ('exec', E)):
            if df is None or len(df) < 30:
                out['entry_reason'] = f'NLM-M: insufficient {name} data'
                return out

        # ── G1 macro bias (4.2), HTF overrides LTF (4.4) ─────────────────────
        bias = structure_bias(H, k)
        macro = structure_bias(F, k)
        chk['G1_bias'] = f'1H={bias} 4H={macro}'
        if bias is None:
            return self._fail(out, 'G1: no 1H macro bias (HH+HL or LH+LL)')
        if macro is not None and macro != bias:
            return self._fail(out, f'G1: 1H {bias} conflicts with 4H {macro} (4.4)')
        short = bias == 'bearish'
        direction = 'short' if short else 'long'

        pip = float(c['pip'])
        nodes = self._nodes(symbol, H, F, Q, c, side='high' if short else 'low')
        opp_nodes = self._nodes(symbol, H, F, Q, c, side='low' if short else 'high')

        # Walk back through the recent sweep-TF candles, newest first, and
        # take the first complete chain that finishes on the last exec bar.
        n = len(S)
        window = int(c['mss_window_bars']) + 2
        last_fail = 'G5: no breach of an untouched Nexus Axis in the recent candles'
        for i in range(n - 1, max(n - 1 - window, 1), -1):
            res = self._chain(symbol, i, short, S, H, F, E, nodes, opp_nodes, c, pip)
            if res.get('ok'):
                chk.update(res['checklist'])
                s = res['setup']
                out.update({
                    'entry_signal': True,
                    'direction': direction,
                    'entry_type': self.ENTRY_TYPE,
                    'order_type': 'limit',
                    'limit_price': s['entry'],
                    'entry_reason': (f"NLM-M {direction.upper()}: sweep {s['axis_label']} "
                                     f"{s['axis']:.5f} -> 5M MSS -> FVG 50% {s['entry']:.5f}"),
                    'confidence_score': 1.0,       # binary checklist: all YES
                    'confluence_score': 0.0,
                    'expected_rr': float(c['target_rr']),
                    'nlm_setup': s,
                })
                return out
            if res.get('reason'):
                last_fail = res['reason']
                chk.update(res.get('checklist', {}))
        return self._fail(out, last_fail)

    @staticmethod
    def _fail(out: Dict, reason: str) -> Dict:
        out['entry_reason'] = 'NLM-M ' + reason
        out['nlm_checklist']['result'] = 'STAND DOWN: ' + reason
        return out

    # ── one sweep candidate through every remaining gate ─────────────────────
    def _chain(self, symbol, i, short, S, H, F, E, nodes, opp_nodes, c, pip) -> Dict:
        chk: Dict = {}
        o, h, l, cl = (S[x].values for x in ('open', 'high', 'low', 'close'))
        t_i = S.index[i]
        tf_min = self._tf_minutes(c['sweep_tf'])

        # ── G2 + G5: an untouched node breached by candle i ──────────────────
        breached = []
        for nd in nodes:
            if pd.Timestamp(nd['formed']) >= t_i:
                continue
            if not self._untouched(nd, short, S, H, F, until=t_i):
                continue
            if (short and h[i] > nd['price']) or ((not short) and l[i] < nd['price']):
                breached.append(nd)
        if not breached:
            return {}

        # ── G6: reclaim, same candle or the next (5.1, 5.3) ──────────────────
        two_candle = False
        valid = [nd for nd in breached
                 if (short and cl[i] < nd['price']) or ((not short) and cl[i] > nd['price'])]
        if not valid:
            if i + 1 >= len(S):
                return {'reason': 'G6: breach candle closed outside, waiting for next candle'}
            j = i + 1
            disp = self._is_displacement(S, j, short, c)
            valid = [nd for nd in breached
                     if disp and ((short and cl[j] < nd['price']) or ((not short) and cl[j] > nd['price']))]
            if not valid:
                return {'reason': 'G7: body closed outside the axis without a two-candle '
                                  'displacement reclaim: breakout, not a sweep (5.3, 6.3)'}
            two_candle = True
        # the axis is the breached node nearest the spike extreme
        axis = max(valid, key=lambda d: d['price']) if short else min(valid, key=lambda d: d['price'])
        reclaim_i = i + 1 if two_candle else i
        extreme = (max(h[i], h[reclaim_i]) if short else min(l[i], l[reclaim_i]))
        chk['G2_axis'] = f"{axis['label']} {axis['price']:.5f} formed {axis['formed']}"
        chk['G5_breach'] = f"{_t(t_i)} spike to {extreme:.5f}"
        chk['G6_reclaim'] = 'two-candle' if two_candle else 'immediate'

        # ── G7: no 3-candle acceptance is implied by G6; both-side sweep ─────
        opp_side_swept = any(
            pd.Timestamp(nd['formed']) < t_i and self._untouched(nd, not short, S, H, F, until=t_i)
            and any(((short and l[x] < nd['price']) or ((not short) and h[x] > nd['price']))
                    for x in range(i, reclaim_i + 1))
            for nd in opp_nodes)
        if opp_side_swept:
            return {'reason': 'G7: both-side liquidity sweep in the same window (10.6)', 'checklist': chk}

        # ── G4: Velocity Factor (3.2 measurement tool) ───────────────────────
        # "the final 3 candles approaching the axis": the candles before the
        # breach candle, against the 20-period ATR that precedes them.
        vn = int(c['velocity_candles'])
        base = atr(S, int(c['velocity_atr_period']), i - vn - 1)
        approach = np.mean([true_range(S, x) for x in range(i - vn, i)])
        vel = approach / base if base > 0 else 0.0
        chk['G4_velocity'] = f'{vel:.2f} (need > {c["velocity_ratio"]})'
        if vel <= float(c['velocity_ratio']):
            return {'reason': f'G4: approach velocity {vel:.2f} <= {c["velocity_ratio"]} '
                              f'(compressed drift favours a breakout, 3.2)', 'checklist': chk}

        # ── G3: HTF Mitigating Node beyond the axis (3.3) ────────────────────
        mn = self._mitigating_node(axis, short, H, F, until=t_i)
        chk['G3_mitigating_node'] = mn['desc'] if mn else 'none'
        if not mn:
            return {'reason': 'G3: no unmitigated 1H/4H FVG beyond the axis (3.3)', 'checklist': chk}

        # ── G9-G11 on the execution frame ────────────────────────────────────
        reclaim_end = S.index[reclaim_i] + timedelta(minutes=tf_min)
        mss_deadline = reclaim_end + timedelta(minutes=tf_min * int(c['mss_window_bars']))
        ltf = self._ltf_confirmation(E, short, S.index[i], reclaim_end, mss_deadline, c)
        chk.update(ltf.get('checklist', {}))
        if not ltf.get('ok'):
            return {'reason': ltf['reason'], 'checklist': chk}

        # ── G12-G14 risk ─────────────────────────────────────────────────────
        entry = ltf['entry']
        buf = float(c['sl_buffer_pips']) * pip
        sl = extreme + buf if short else extreme - buf
        risk = (sl - entry) if short else (entry - sl)
        last_close = float(E['close'].iloc[-1])
        if risk <= 0:
            return {'reason': 'G13: FVG equilibrium is beyond the stop', 'checklist': chk}
        if (short and entry <= last_close) or ((not short) and entry >= last_close):
            return {'reason': 'G12: FVG equilibrium already traded, limit would be on the wrong '
                              'side of market (do not market-order, 8.2)', 'checklist': chk}
        # 9.4 audits R:R to the trade's target. The owner fixed the target at
        # 1:target_rr (replacing the 8.4 node target), so the audit is applied
        # to that target and the opposing-node choice is not needed.
        sgn = -1 if short else 1
        tp = entry + sgn * float(c['target_rr']) * risk
        chk['G14_rr'] = f"1:{c['target_rr']} target {tp:.5f} (min 1:{c['min_rr_to_node']})"
        if float(c['target_rr']) < float(c['min_rr_to_node']):
            return {'reason': f"G14: target R:R {c['target_rr']} < {c['min_rr_to_node']} (9.4)",
                    'checklist': chk}

        # ── G8: Rule 3, deep target reached before the sweep (6.2) ───────────
        # The target destination is the trade's target; "deep" is the far
        # side of the 50% equilibrium (2.8) of the range from the axis to that
        # target. If price was already there between the axis forming
        # and the sweep, the expansion move has happened.
        if c.get('rule3_enabled', True):
            eq = (axis['price'] + tp) / 2.0
            t_axis = pd.Timestamp(axis['formed'])
            seg = S[(S.index >= t_axis) & (S.index < t_i)]
            deep = (len(seg) and ((short and float(seg['low'].min()) < eq)
                                  or ((not short) and float(seg['high'].max()) > eq)))
            chk['G8_rule3'] = f'equilibrium {eq:.5f}: ' + ('reached' if deep else 'clean')
            if deep:
                return {'reason': f'G8: Rule 3, price reached the deep half toward the '
                                  f'target (beyond {eq:.5f}) before the sweep (6.2)',
                        'checklist': chk}

        setup = {
            'symbol': symbol, 'direction': 'short' if short else 'long',
            'axis': float(axis['price']), 'axis_label': axis['label'],
            'sweep_time': _t(t_i), 'reclaim_end': _t(reclaim_end), 'extreme': float(extreme),
            'mss_time': ltf['mss_time'], 'fvg_top': ltf['fvg_top'], 'fvg_bottom': ltf['fvg_bottom'],
            'entry': float(entry), 'sl': float(sl),
            'tp_partial': float(entry + sgn * float(c['partial_rr']) * risk),
            'tp': float(tp),
            'partial_fraction': float(c['partial_fraction']),
            'signal_time': _t(E.index[-1]), 'last_close': last_close,
            'exec_atr': atr(E, 20, len(E) - 1),
            'mitigating_node': mn['desc'],
        }
        chk['G12_entry'] = f'{entry:.5f}'
        chk['G13_sl'] = f'{sl:.5f} ({c["sl_buffer_pips"]} pips beyond {extreme:.5f})'
        chk['result'] = 'ALL YES: EXECUTE LIMIT ORDER'
        return {'ok': True, 'setup': setup, 'checklist': chk}

    # ══════════════════════════════════════════════════════════════════════════
    # Building blocks
    # ══════════════════════════════════════════════════════════════════════════

    def _nodes(self, symbol, H, F, Q, c, side: str) -> List[Dict]:
        """3.1 Primary Node Classifications:
        1 EQH/EQL and 3 major HTF swing extremes: every confirmed 1H/4H swing
          point (a double top is two swing highs; the untouched one is the
          pool's edge).
        2 Session extremes: completed Asian/London/NY sessions (2.4, 7.2).
        4 Psychological round numbers (optional per symbol).
        """
        k = int(c['swing_order'])
        out: List[Dict] = []
        for tf, df in (('1H', H), ('4H', F)):
            sh, sl = swing_points(df, k)
            pts = sh if side == 'high' else sl
            for pos, px in pts:
                # a swing exists once its k right-hand candles have closed
                formed = df.index[min(pos + k, len(df) - 1)] + timedelta(minutes=self._tf_minutes(tf))
                out.append({'price': px, 'label': f'{tf} swing {side}', 'formed': _t(formed),
                            'major': tf == '4H'})
        if Q is not None and len(Q):
            out.extend(self._session_nodes(Q, c, side))
        step = float(c.get('round_step') or 0.0)
        if step > 0 and H is not None and len(H):
            lo, hi = float(H['low'].min()), float(H['high'].max())
            start = np.floor(lo / step) * step
            for lvl in np.arange(start, hi + step, step):
                out.append({'price': float(round(lvl, 10)), 'label': 'round number',
                            'formed': _t(H.index[0]), 'major': False})
        return out

    def _session_nodes(self, Q: pd.DataFrame, c: Dict, side: str) -> List[Dict]:
        out = []
        days = sorted({ts.normalize() for ts in Q.index})[-int(c['session_days']) - 1:]
        last_close = Q.index[-1] + timedelta(minutes=self._tf_minutes(c['session_tf']))
        for day in days:
            for name, (a, b) in (c['sessions'] or {}).items():
                start = day + pd.Timedelta(a + ':00')
                end = day + pd.Timedelta(b + ':00')
                if end > last_close:                      # session not complete
                    continue
                seg = Q[(Q.index >= start) & (Q.index < end)]
                if len(seg) < 2:
                    continue
                px = float(seg['high'].max()) if side == 'high' else float(seg['low'].min())
                out.append({'price': px, 'label': f'{name} session {side}', 'formed': _t(end),
                            'major': True})
        return out

    def _untouched(self, nd: Dict, high_side: bool, S, H, F, until) -> bool:
        """3.1 table: a strong node is untouched, not previously swept. True
        when no bar that opened at or after the node formed, and before
        `until`, traded through it. Uses the finest frame whose history
        reaches back to the node; the 4H frame is the last resort."""
        start, until = pd.Timestamp(nd['formed']), pd.Timestamp(until)
        if start >= until:
            return False
        frames = [df for df in (S, H, F) if df is not None and len(df)]
        use = next((df for df in frames if df.index[0] <= start), frames[-1])
        seg = use[(use.index >= start) & (use.index < until)]
        if not len(seg):
            return True
        if high_side:
            return float(seg['high'].max()) <= nd['price']
        return float(seg['low'].min()) >= nd['price']

    def _is_displacement(self, df: pd.DataFrame, i: int, short: bool, c: Dict) -> bool:
        o, cl = float(df['open'].iloc[i]), float(df['close'].iloc[i])
        rng = float(df['high'].iloc[i] - df['low'].iloc[i])
        if rng <= 0:
            return False
        if short and not cl < o:
            return False
        if (not short) and not cl > o:
            return False
        base = atr(df, 20, i - 1)
        return (true_range(df, i) >= float(c['displacement_atr_ratio']) * base
                and abs(cl - o) >= float(c['displacement_body_frac']) * rng)

    def _mitigating_node(self, axis: Dict, short: bool, H, F, until) -> Optional[Dict]:
        """3.3: an unmitigated 4H/1H FVG hidden behind the axis. For a short it
        sits above the axis; unmitigated means no price has traded into the
        gap since it formed. (The book also names order blocks but never
        defines them, so only FVGs are used.)"""
        until = pd.Timestamp(until)
        best = None
        for tf, df in (('1H', H), ('4H', F)):
            hh, ll = df['high'].values, df['low'].values
            tfm = self._tf_minutes(tf)
            for m in range(1, len(df) - 1):
                c3_end = df.index[m + 1] + timedelta(minutes=tfm)
                if c3_end > until:
                    break
                if short and ll[m - 1] > hh[m + 1]:            # bearish gap
                    bottom, top = float(hh[m + 1]), float(ll[m - 1])
                    if bottom <= axis['price']:
                        continue
                    later = df[(df.index >= c3_end) & (df.index < until)]
                    if len(later) and float(later['high'].max()) >= bottom:
                        continue
                elif (not short) and hh[m - 1] < ll[m + 1]:    # bullish gap
                    bottom, top = float(hh[m - 1]), float(ll[m + 1])
                    if top >= axis['price']:
                        continue
                    later = df[(df.index >= c3_end) & (df.index < until)]
                    if len(later) and float(later['low'].min()) <= top:
                        continue
                else:
                    continue
                edge = bottom if short else top
                if best is None or abs(edge - axis['price']) < abs(best[0] - axis['price']):
                    best = (edge, f'{tf} FVG {bottom:.5f}-{top:.5f} at {_t(df.index[m])}')
        return {'desc': best[1], 'price': best[0]} if best else None

    def _ltf_confirmation(self, E, short, sweep_open, reclaim_end, deadline, c) -> Dict:
        """4.3 on the 5M: MSS body close past the most recent opposing swing
        (2.6) by a displacement candle (6.4), then the FVG the displacement
        move created (2.7, 8.2). Fires only on the bar that completes the
        chain, so a setup is signalled once."""
        chk: Dict = {}
        k = int(c['swing_order'])
        seg = E[(E.index >= sweep_open) & (E.index < reclaim_end)]
        if not len(seg):
            return {'reason': 'G9: no 5M data for the sweep candle', 'checklist': chk}
        ext_time = seg['high'].idxmax() if short else seg['low'].idxmin()
        s_bar = E.index.get_loc(ext_time)
        highs, lows = swing_points(E, k)
        opp = lows if short else highs
        cl = E['close'].values
        mss = None
        for j in range(s_bar + 1, len(E)):
            if E.index[j] >= deadline:
                break
            confirmed = [p for p in opp if p[0] + k <= j - 1]
            if not confirmed:
                continue
            lvl = confirmed[-1][1]
            if (short and cl[j] < lvl) or ((not short) and cl[j] > lvl):
                mss = (j, lvl)
                break
        if mss is None:
            return {'reason': 'G9: no 5M MSS body close after the sweep (6.4, 10.5)', 'checklist': chk}
        j, lvl = mss
        chk['G9_mss'] = f'{_t(E.index[j])} close past {lvl:.5f}'
        if not self._is_displacement(E, j, short, c):
            return {'reason': 'G10: MSS candle is not a displacement candle (2.6, 6.4)',
                    'checklist': chk}
        chk['G10_displacement'] = 'yes'
        h, l = E['high'].values, E['low'].values
        fvg = None
        for m in range(j, s_bar, -1):                    # the displacement candle first
            if m + 1 >= len(E):
                continue
            if short and l[m - 1] > h[m + 1]:
                fvg = (m, float(h[m + 1]), float(l[m - 1]))
                break
            if (not short) and h[m - 1] < l[m + 1]:
                fvg = (m, float(h[m - 1]), float(l[m + 1]))
                break
        if fvg is None:
            if j + 1 >= len(E):
                return {'reason': 'G11: waiting for the candle after the displacement', 'checklist': chk}
            return {'reason': 'G11: displacement move left no FVG (8.2)', 'checklist': chk}
        m, bottom, top = fvg
        done_bar = max(j, m + 1)
        if done_bar != len(E) - 1:
            return {'reason': 'G11: chain completed on an earlier bar (already signalled)',
                    'checklist': chk}
        chk['G11_fvg'] = f'{bottom:.5f}-{top:.5f} middle candle {_t(E.index[m])}'
        return {'ok': True, 'entry': (top + bottom) / 2.0, 'fvg_top': top, 'fvg_bottom': bottom,
                'mss_time': _t(E.index[j]), 'checklist': chk}

