"""
PATCH 7 — strategy/engine.py + config/config.yaml
===================================================
The Original engine lost $2,950 in the Apr 7-10 test. Here's why and
what this patch fixes, based on analysis of professional gold/BTC
trading strategies used by prop firm traders.

ROOT CAUSES:
=============

1. TIMEFRAME HIERARCHY IS WRONG (config issue)
   Gold was configured: 15m / 5m / 1m (structure / primary / entry)
   Structure bias from 15m is too noisy — it flips bullish/bearish
   multiple times per day. Professional gold traders use 4H minimum
   for structure, 15m-1H for signal detection, 5m for entry precision.

   BTC was configured: 1H / 15m / 1m — slightly better but 1m entry
   on BTC with $70k price produces microscopic ATR for SL.

2. STOP LOSS USES ENTRY-TF ATR (code issue)
   calculate_entry_levels() computes ATR from the entry timeframe.
   On 1m gold, ATR ≈ $1.50. With conservative method (1.5×ATR) = $2.25 SL.
   That's 0.05% of price — normal spread noise wipes it out.
   
   Fix: Use the PRIMARY timeframe ATR for SL calculation, not entry TF.
   The entry TF should only be used for precise limit price placement.

3. NO SESSION AWARENESS (code issue)
   The Original engine has zero session filtering. ICT only trades
   during killzones (London/NY) — this is a HUGE edge. Professional
   gold strategies consistently show that Asian session generates the
   most false signals while London/NY overlap produces the cleanest
   directional moves.
   
   Fix: Add a session filter to _apply_filters() that rejects signals
   during low-liquidity hours (00:00-06:00 UTC for gold/forex).

4. pullback_to_sr USES EMA50, NOT ACTUAL S/R (code issue)
   The method checks proximity to EMA50 and calls it "support/resistance".
   Real S/R comes from swing highs/lows in the price structure. EMA50
   is a dynamic level that moves constantly — it's not S/R.
   
   Fix: Use last_swing_low (for longs) / last_swing_high (for shorts)
   from price_structure when available, EMA50 only as fallback.

5. CONFLUENCE THRESHOLD TOO HIGH FOR SIGNAL CAPACITY
   threshold=7 for volatile symbols, but most entry types can only
   generate 5-7 signals maximum. This means nearly every indicator
   must align perfectly — too restrictive for real markets. ICT uses
   signal weights of 2-4 per concept, making it easier to reach
   threshold with 2-3 strong confirmations.
   
   Fix: Not patching the weights (they work fine conceptually), but
   the config recommendation below lowers threshold from 7 to 6 for
   volatile symbols. One fewer signal required = significantly more
   trades, and the ones that fire will still have strong confluence.


HOW TO APPLY:
=============

PART A — config/config.yaml changes (CRITICAL):
  Update XAU/USD and BTC/USD symbol configs.

PART B — strategy/engine.py changes:
  1. Replace calculate_entry_levels() with the patched version.
  2. Replace _pullback_sr() with the patched version.
  3. Replace _apply_filters() with the patched version (adds session gate).
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART A — config/config.yaml changes
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_CHANGES = """
# Replace the symbol blocks in config.yaml with these:

symbols:
  XAU/USD:
    enabled: true
    platform: mt5
    mode: live
    timeframes:
    - 4H          # was 15m — MUST be 4H+ for structure bias
    - 15m         # was 5m  — signal detection
    - 5m          # was 1m  — entry precision
    primary_timeframe: 15m    # was 5m
    entry_timeframe: 5m       # was 1m — 1m ATR on gold is $1.50 = suicide
    confluence_threshold: 6   # was 7  — lets more qualified trades through

  BTC/USD:
    enabled: true
    platform: mt5
    mode: live
    timeframes:
    - 4H          # was 1H — give structure more room
    - 1H          # was 15m — BTC needs wider signal TF
    - 15m         # was 1m  — 1m ATR on BTC is ~$50 = too tight
    primary_timeframe: 1H     # was 15m
    entry_timeframe: 15m      # was 1m
    confluence_threshold: 6   # was 7

  # XAG/USD and EUR/USD configs are already reasonable, no changes needed.

# Also update confluence thresholds:
strategy:
  confluence_required: 6        # was 7 — for volatile symbols
  confluence_threshold_calm: 5  # unchanged — for calm symbols
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — strategy/engine.py method replacements
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE: calculate_entry_levels  (in class StrategyEngine)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_entry_levels(self, analysis, multi_tf_data):
    """
    PATCHED: Uses PRIMARY timeframe ATR for SL calculation, not entry TF.
    Entry TF is still used for limit price precision.

    The old code used entry_tf ATR (1m on gold = $1.50 ATR → $2.25 SL).
    Now uses primary_tf ATR (15m on gold ≈ $5-8 ATR → $7.50-12 SL).
    Combined with the conservative method (min of ATR and structure stop),
    this produces realistic stops that can survive normal market noise.
    """
    entry_tf = analysis.get('entry_tf', '5m')
    primary_tf = analysis.get('primary_timeframe', '15m')

    if entry_tf not in multi_tf_data:
        avail = sorted(multi_tf_data.keys(), key=lambda t: self._tf_minutes(t))
        entry_tf = avail[0] if avail else next(iter(multi_tf_data), None)
    if not entry_tf or entry_tf not in multi_tf_data:
        return {}

    df = multi_tf_data[entry_tf]
    current_price = float(df['close'].iloc[-1])

    # ── CHANGE: Use primary TF ATR for SL, not entry TF ATR ──────────────
    # Entry TF ATR is too small for SL on short timeframes (1m, 5m).
    # Primary TF ATR reflects the actual price movement the trade needs
    # to survive through.
    primary_df = multi_tf_data.get(primary_tf, df)
    atr = self.indicators.calculate_atr(primary_df)['current']

    # Also get entry TF ATR for limit price offset (still useful)
    entry_atr = self.indicators.calculate_atr(df)['current']

    order_type = analysis.get('order_type', 'market')
    limit_price = analysis.get('limit_price')
    entry_price = float(limit_price) if (order_type == 'limit' and limit_price) \
                  else current_price

    stop_loss = self._calc_sl(analysis, entry_price, atr)
    take_profits = self._calc_tps(analysis, entry_price, stop_loss)

    return {
        'entry_price':   current_price,
        'order_price':   entry_price,
        'order_type':    order_type,
        'limit_price':   entry_price if order_type == 'limit' else None,
        'stop_loss':     stop_loss,
        'take_profit_1': take_profits.get('tp1'),
        'take_profit_2': take_profits.get('tp2'),
        'take_profit_3': take_profits.get('tp3'),
        'atr':           atr,
        'risk_distance': abs(entry_price - stop_loss),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE: _pullback_sr  (in class StrategyEngine)
# ─────────────────────────────────────────────────────────────────────────────

def _pullback_sr(self, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
    """
    PATCHED: Uses actual swing points from price_structure for S/R,
    with EMA50 as fallback only when no swing points are available.

    Old code: used EMA50 proximity as "S/R" → not real support/resistance.
    New code: checks if price has pulled back to last_swing_low (for longs)
    or last_swing_high (for shorts). These are actual structural levels
    where price previously reversed.
    """
    d = self._e()
    if bias not in ('bullish', 'bearish'):
        return d

    pi = ptf['indicators']
    cp = ptf['ohlc']['close']
    atr = pi['atr']['value']

    # ── Try real structure S/R first ─────────────────────────────────────
    ps = pi.get('price_structure', {})
    swing_low = ps.get('last_swing_low')
    swing_high = ps.get('last_swing_high')

    sr_level = None
    dr = None

    if bias == 'bullish' and swing_low:
        # Long: price pulled back near the last swing low (support)
        distance = abs(cp - swing_low)
        if distance <= 0.5 * atr and cp > swing_low:
            sr_level = swing_low
            dr = 'long'
    elif bias == 'bearish' and swing_high:
        # Short: price pulled back near the last swing high (resistance)
        distance = abs(cp - swing_high)
        if distance <= 0.5 * atr and cp < swing_high:
            sr_level = swing_high
            dr = 'short'

    # ── Fallback to EMA50 if no swing points ─────────────────────────────
    if sr_level is None:
        e50 = pi['ema'].get(50, 0)
        if e50 <= 0:
            return d
        prox = abs(cp - e50) / e50 < 0.003
        if bias == 'bullish':
            dr, ok = 'long', prox and cp > e50
        else:
            dr, ok = 'short', prox and cp < e50
        if not ok:
            return d
        sr_level = e50

    # ── Build confluence signals ─────────────────────────────────────────
    sigs = ['pullback_to_sr']
    rsi = pi['rsi']['value']
    if dr == 'long' and (pi['rsi']['oversold'] or 30 < rsi < 45):
        sigs.append('rsi_oversold_recovery')
    elif dr == 'short' and (pi['rsi']['overbought'] or 55 < rsi < 70):
        sigs.append('rsi_overbought_rejection')

    if pi['supertrend']['trend'] == ('bullish' if dr == 'long' else 'bearish'):
        sigs.append('supertrend_aligned')

    if pi['adx'].get('trend_strength') == 'strong':
        sigs.append('adx_strong')

    for cpat, side in [('bullish_engulfing', 'long'), ('hammer', 'long'),
                       ('bearish_engulfing', 'short'), ('shooting_star', 'short')]:
        if dr == side and pi['candle_patterns'].get(cpat):
            sigs.append('candle_pattern')
            break

    e20, e50, e200 = pi['ema'].get(20, 0), pi['ema'].get(50, 0), pi['ema'].get(200, 0)
    if dr == 'long' and e20 > e50 > e200 > 0:
        sigs.append('ema_stack_full')
    elif dr == 'short' and e20 < e50 and e200 > 0 and e50 < e200:
        sigs.append('ema_stack_full')

    if htf['trend']['direction'] == ('bullish' if dr == 'long' else 'bearish'):
        sigs.append('price_structure_aligned')

    d.update({
        'signal': True, 'entry_signal': True, 'direction': dr,
        'confluence_signals': sigs,
        'entry_reason': f'Pullback to S/R {dr} @ {sr_level:.2f}',
        'limit_price': ltf['ohlc']['close'],
    })
    return d


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE: _apply_filters  (in class StrategyEngine)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filters(self, analysis, structure_tf, entry_tf):
    """
    PATCHED: Added session filter to avoid low-liquidity trading hours.

    Professional gold traders consistently report that the London/NY
    overlap (12:00-16:00 UTC) produces the highest-quality signals.
    Asian session (00:00-06:00 UTC) is the noisiest for gold/forex.

    The filter doesn't block crypto (BTC trades 24/7 with no clear
    session disadvantage).
    """
    filters = self.strategy_config.get('filters', {})
    snaps = analysis['timeframe_snapshots']

    # ── Existing: HTF bias filter ────────────────────────────────────────
    if filters.get('respect_higher_tf_bias', True) and structure_tf in snaps:
        htf_trend = snaps[structure_tf]['trend']['direction']
        if analysis['direction'] == 'long' and htf_trend == 'bearish':
            if not filters.get('allow_override', False):
                analysis['entry_signal'] = False
                analysis['entry_reason'] = 'Filtered: against HTF bearish'
                return analysis
        elif analysis['direction'] == 'short' and htf_trend == 'bullish':
            if not filters.get('allow_override', False):
                analysis['entry_signal'] = False
                analysis['entry_reason'] = 'Filtered: against HTF bullish'
                return analysis

    # ── Existing: ATR threshold filter ───────────────────────────────────
    tf_for_atr = entry_tf if entry_tf in snaps else (
        next(iter(snaps)) if snaps else None)
    if tf_for_atr and tf_for_atr in snaps:
        atr_pct = snaps[tf_for_atr]['indicators']['atr']['percent']
        mn = filters.get('min_atr_threshold', 0)
        mx = filters.get('max_atr_threshold', 100)
        if atr_pct < mn * 100:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'Filtered: ATR too low ({atr_pct:.2f}%)'
            return analysis
        if atr_pct > mx * 100:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'Filtered: ATR too high ({atr_pct:.2f}%)'
            return analysis

    # ── Existing: ADX chop filter ────────────────────────────────────────
    if structure_tf in snaps:
        htf_snap = snaps[structure_tf]
        adx = htf_snap['indicators'].get('adx', {})
        adx_val = adx.get('value')
        if isinstance(adx_val, pd.Series):
            adx_val = float(adx_val.iloc[-1])
        if adx_val is not None and adx_val < 20:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'Filtered: ADX chop ({adx_val:.1f})'
            return analysis

    # ── NEW: Session filter — avoid low-liquidity hours ──────────────────
    symbol = analysis.get('symbol', '')
    sym_upper = symbol.replace('/', '').upper()

    # Skip session filter for crypto (24/7 markets)
    _CRYPTO = {'BTCUSD', 'ETHUSD', 'BTCUSDT', 'ETHUSDT'}
    if sym_upper not in _CRYPTO:
        hour = datetime.utcnow().hour

        # Block signals during quiet hours (00:00-06:00 UTC)
        # This is when Asian session produces the most false breakouts
        # on gold and forex. London opens at 07:00 UTC.
        quiet_start = filters.get('quiet_hours_start', 0)
        quiet_end = filters.get('quiet_hours_end', 6)

        if quiet_start <= hour < quiet_end:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = (
                f'Filtered: Quiet hours ({hour:02d}:00 UTC) — '
                f'wait for London/NY session'
            )
            return analysis

    # ── Existing: Trend strength filter ──────────────────────────────────
    if structure_tf in snaps:
        trend_strength = snaps[structure_tf]['trend'].get('strength', 1.0)
        if trend_strength < 0.5:
            analysis['entry_signal'] = False
            analysis['entry_reason'] = f'Filtered: Trend too weak ({trend_strength:.2f})'
            return analysis

    return analysis