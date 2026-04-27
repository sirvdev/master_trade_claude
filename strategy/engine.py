"""
strategy/engine.py — Multi-timeframe strategy engine v2.
Three timeframe roles per symbol:
  structure_tf = timeframes[0] (longest) — macro bias
  primary_tf   = primary_timeframe       — signal pattern detection
  entry_tf     = entry_timeframe         — precise entry price + fine SL

BUG FIXED: self.timeframe_config was reading config.get('timeframes', {})
which is always empty (no top-level 'timeframes' key exists). Timeframes live
under each symbol block. Fixed by passing symbol_config into analyze_market().
"""

import logging
from datetime import datetime, time
from typing import Dict, List, Optional
import pandas as pd
from indicators.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

_SIGNAL_WEIGHTS: Dict[str, int] = {
    'supertrend_aligned': 3, 'ema_stack_full': 3, 'price_structure_aligned': 3,
    'rsi_divergence_confirmed': 3, 'breakout_held': 3,
    'macd_zero_cross': 2, 'rsi_confirmation': 2, 'bb_squeeze_active': 2,
    'ema_proximity_bounce': 2, 'pullback_to_ema50': 2, 'pullback_to_sr': 2, 'adx_strong': 2,
    'candle_pattern': 1, 'volume_confirmation': 1, 'rsi_oversold_recovery': 1,
    'rsi_overbought_rejection': 1, 'macd_bullish_cross': 1, 'macd_bearish_cross': 1,
    'bollinger_squeeze': 1, 'strong_trend': 1,
}

_LIMIT_ORDER_TYPES = {'ema_stack_pullback', 'pullback_to_sr', 'rsi_divergence'}
_VOLATILE_SYMBOLS  = {'XAUUSD','XAU/USD','BTCUSD','BTC/USD','NAS100USD','NAS100','US100'}


class StrategyEngine:

    def __init__(self, config: Dict):
        self.config          = config
        self.indicators      = TechnicalIndicators(config.get('indicators', {}))
        self.strategy_config = config.get('strategy', {})

    def _tf_minutes(self, tf: str) -> int:
        m = {'1m':1,'5m':5,'15m':15,'30m':30,'1H':60,'4H':240,'1D':1440,'1W':10080,
             'M1':1,'M5':5,'M15':15,'H1':60,'H4':240,'D1':1440}
        return m.get(tf, 60)

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze_market(self, symbol: str, multi_tf_data: Dict[str, pd.DataFrame],
                       symbol_config: Optional[Dict] = None) -> Dict:
        """
        Full 3-TF analysis. symbol_config must be passed for correct TF roles.
        Without it falls back to guessing from data keys.
        """
        if symbol_config:
            tfs          = symbol_config.get('timeframes', [])
            structure_tf = tfs[0] if tfs else '1H'
            primary_tf   = symbol_config.get('primary_timeframe',
                                             tfs[1] if len(tfs) > 1 else '15m')
            entry_tf     = symbol_config.get('entry_timeframe', tfs[-1] if tfs else '5m')
        else:
            avail        = sorted(multi_tf_data.keys(),
                                  key=lambda t: self._tf_minutes(t), reverse=True)
            structure_tf = avail[0]  if len(avail) > 0 else '1H'
            primary_tf   = avail[1]  if len(avail) > 1 else '15m'
            entry_tf     = avail[-1] if len(avail) > 0 else '5m'

        logger.info(f"Analyzing {symbol} structure={structure_tf} "
                    f"primary={primary_tf} entry={entry_tf}")

        analysis = {
            'symbol': symbol, 'timestamp': datetime.now().replace(tzinfo=None),
            'primary_timeframe': primary_tf, 'structure_tf': structure_tf,
            'entry_tf': entry_tf, 'timeframe_snapshots': {},
            'market_structure': {}, 'indicators_state': {},
            'entry_signal': False, 'entry_reason': None, 'entry_type': None,
            'confidence_score': 0.0, 'confluence_score': 0.0,
            'confluence_signals': [], 'direction': None,
            'order_type': 'market', 'limit_price': None,
        }

        try:
            for tf, df in multi_tf_data.items():
                if df is None or len(df) < 50:
                    logger.warning(f"Insufficient data {symbol}@{tf}")
                    continue
                analysis['timeframe_snapshots'][tf] = self._analyze_tf(df)

            snaps = analysis['timeframe_snapshots']
            if structure_tf in snaps:
                analysis['market_structure'] = self._determine_structure(snaps[structure_tf])
            elif snaps:
                analysis['market_structure'] = self._determine_structure(
                    snaps[next(iter(snaps))])

            decision = self._evaluate_entry(
                analysis, multi_tf_data, structure_tf, primary_tf, entry_tf)
            analysis.update(decision)

            if analysis['entry_signal']:
                analysis = self._apply_filters(analysis, structure_tf, entry_tf)

            logger.info(
                f"Analysis complete: {symbol} signal={analysis['entry_signal']} "
                f"dir={analysis['direction']} type={analysis.get('entry_type')} "
                f"order={analysis.get('order_type')} "
                f"score={analysis.get('confluence_score',0):.1f} "
                f"conf={analysis['confidence_score']:.2f}")

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)

        return analysis

    def calculate_entry_levels(self, analysis, multi_tf_data):
        """
        FIXED: Uses PRIMARY timeframe ATR for SL calculation instead of entry TF.
        
        The old code used entry_tf ATR (1m gold = $1.50 → $2.25 SL).
        Now uses primary_tf ATR (15m gold ≈ $5-8 → $7.50-12 SL).
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
    
        # ── Use primary TF ATR for SL ────────────────────────────────────
        primary_df = multi_tf_data.get(primary_tf, df)
        atr = self.indicators.calculate_atr(primary_df)['current']
    
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

    # ── TF analysis ────────────────────────────────────────────────────────────

    def _analyze_tf(self, df: pd.DataFrame) -> Dict:
        ind = self.indicators.calculate_all(df)
        macd_curr = float(ind['macd']['macd'].iloc[-1])
        signal_curr = float(ind['macd']['signal'].iloc[-1])
        macd_prev = float(ind['macd']['macd'].iloc[-2])
        signal_prev = float(ind['macd']['signal'].iloc[-2])
        snap = {
            'ohlc': {k: float(df[k].iloc[-1]) for k in ['open','high','low','close','volume']},
            'indicators': {
                'ema':         {p: float(v.iloc[-1]) for p, v in ind['ema'].items()},
                'rsi':         {'value': float(ind['rsi']['value'].iloc[-1]),
                                'overbought': bool(ind['rsi']['is_overbought']),
                                'oversold':   bool(ind['rsi']['is_oversold'])},
                'macd':        {'macd':     float(ind['macd']['macd'].iloc[-1]),
                                'signal':   float(ind['macd']['signal'].iloc[-1]),
                                'histogram':float(ind['macd']['histogram'].iloc[-1]),
                                'bullish_cross': (macd_prev <= signal_prev and macd_curr > signal_curr),
                                'bearish_cross': (macd_prev >= signal_prev and macd_curr < signal_curr)},
                'atr':         {'value':   float(ind['atr']['current']),
                                'percent': float(ind['atr']['percent_of_price'])},
                'bollinger':   {'upper':   float(ind['bollinger']['upper'].iloc[-1]),
                                'middle':  float(ind['bollinger']['middle'].iloc[-1]),
                                'lower':   float(ind['bollinger']['lower'].iloc[-1]),
                                'squeeze': bool(ind['bollinger']['squeeze'])},
                'supertrend':      ind['supertrend'],
                'adx':             ind['adx'],
                'price_structure': ind['price_structure'],
                'candle_patterns': ind['candle_patterns'],
            },
        }
        snap['trend'] = self._calc_trend(snap['indicators'])
        return snap

    def _calc_trend(self, ind: Dict) -> Dict:
        sigs = []
        ema20, ema50 = ind['ema'].get(20,0), ind['ema'].get(50,0)
        if ema20 > ema50: sigs.append(1)
        elif ema20 < ema50: sigs.append(-1)
        sigs.append(1 if ind['supertrend']['trend'] == 'bullish' else -1)
        sigs.append(1 if ind['adx']['direction'] == 'bullish' else -1)
        st = ind['price_structure']['structure']
        if st == 'uptrend': sigs.append(1)
        elif st == 'downtrend': sigs.append(-1)
        score = sum(sigs)/len(sigs) if sigs else 0
        trend = 'bullish' if score > 0.3 else ('bearish' if score < -0.3 else 'neutral')
        return {'direction': trend, 'strength': abs(score), 'signals': sigs}

    def _determine_structure(self, snap: Dict) -> Dict:
        ps = snap['indicators']['price_structure']
        tr = snap['trend']
        return {'type': ps['structure'], 'trend': tr['direction'],
                'strength': tr['strength'],
                'last_swing_high': ps.get('last_swing_high'),
                'last_swing_low':  ps.get('last_swing_low')}

    # ── Entry evaluation ───────────────────────────────────────────────────────

    def _evaluate_entry(self, analysis, multi_tf_data,
                        structure_tf, primary_tf, entry_tf) -> Dict:
        out = {'entry_signal':False,'entry_reason':None,'entry_type':None,
               'direction':None,'confidence_score':0.0,'confluence_score':0.0,
               'confluence_signals':[],'order_type':'market','limit_price':None}

        snaps = analysis['timeframe_snapshots']

        # ── Bug 3 fix: structure TF timeout fallback ──────────────────────────
        # If the structure TF fetch failed (e.g. 4H timeout), don't return
        # score=0 immediately. Fall back to the next available TF for structure
        # bias. The signal will still fire; it just uses a shorter TF for bias.
        if structure_tf not in snaps:
            # Try primary_tf, then any available TF, in descending timeframe order
            fallback_order = sorted(
                snaps.keys(),
                key=lambda t: self._tf_minutes(t),
                reverse=True
            )
            if not fallback_order:
                out['entry_reason'] = 'No timeframe data available'
                return out
            structure_tf = fallback_order[0]
            logger.warning(
                f"[ENGINE] structure TF not in snapshots — "
                f"falling back to {structure_tf} for bias"
            )

        # Use primary_tf for signal detection; fall back to structure_tf
        sig_tf = primary_tf if primary_tf in snaps else structure_tf
        # Use entry_tf for price precision; fall back to signal_tf
        prc_tf = entry_tf   if entry_tf   in snaps else sig_tf

        htf    = snaps[structure_tf]   # structure bias
        ptf    = snaps[sig_tf]         # signal pattern TF
        ltf    = snaps[prc_tf]         # entry precision TF

        htf_df = multi_tf_data.get(structure_tf)
        ptf_df = multi_tf_data.get(sig_tf)
        ltf_df = multi_tf_data.get(prc_tf, ptf_df)

        bias      = analysis.get('market_structure', {}).get('trend')
        threshold = self._get_threshold(analysis.get('symbol',''))

        # ── Bug 2 fix: track best score across all attempted entry types ──────
        # Even when no entry type meets the threshold, record the best score
        # found so analysts and the learning engine can see how close the system
        # was to signaling ("score=5, threshold=7" is far more useful than "0").
        best_score   = 0.0
        best_signals = []
        best_etype   = None
        best_dir     = None

        for etype in self.strategy_config.get('entry_types', ['breakout_retest']):
            res = self._check(etype, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias)
            if not res.get('signal'):
                continue

            score = self._score(res['confluence_signals'])

            # Track the highest score seen regardless of whether it passed
            if score > best_score:
                best_score   = score
                best_signals = res['confluence_signals']
                best_etype   = etype
                best_dir     = res.get('direction')

            if score < threshold:
                logger.debug(f"[{analysis.get('symbol')}] {etype}: "
                             f"score={score:.0f} < threshold={threshold} "
                             f"signals={res['confluence_signals']}")
                continue

            # ── Full match — signal fires ─────────────────────────────────────
            otype = 'limit' if etype in _LIMIT_ORDER_TYPES else 'market'
            out.update({
                'entry_signal': True, 'entry_reason': res.get('entry_reason', etype),
                'entry_type': etype, 'direction': res['direction'],
                'confidence_score': min(1.0, score / max(threshold * 1.5, 10.0)),
                'confluence_score': score,
                'confluence_signals': res['confluence_signals'],
                'order_type': otype, 'limit_price': res.get('limit_price'),
            })
            return out

        # ── No full match — populate best-attempt scores for logging/learning ─
        if best_score > 0:
            out.update({
                'confluence_score':   best_score,
                'confluence_signals': best_signals,
                'confidence_score':   min(1.0, best_score / max(threshold * 1.5, 10.0)),
                'entry_reason':       (
                    f'Below threshold ({best_score:.0f}/{threshold}) — '
                    f'best: {best_etype} {best_dir or ""}'
                ),
                'direction': best_dir,  # preserves direction for logging even without signal
            })

        return out

    def _check(self, etype, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
        dispatch = {
            'breakout_retest':    self._breakout_retest,
            'ema_stack_pullback': self._ema_stack_pullback,
            'pullback_to_sr':     self._pullback_sr,
            'bb_squeeze_breakout':self._bb_squeeze,
            'macd_zero_cross':    self._macd_zero,
            'momentum_breakout':  self._momentum,
            'rsi_divergence':     self._rsi_div,
        }
        fn = dispatch.get(etype)
        if not fn: return {'signal':False}
        try: return fn(htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias)
        except Exception as e:
            logger.debug(f"{etype} check error: {e}")
            return {'signal':False}

    def _e(self): # empty decision
        return {'signal':False,'entry_signal':False,'entry_reason':None,
                'direction':None,'confidence_score':0.0,'confluence_signals':[],
                'limit_price':None}

    # ── Entry check implementations ────────────────────────────────────────────

    def _breakout_retest(self, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
        d = self._e()
        if bias not in ('bullish','bearish') or ptf_df is None or len(ptf_df) < 10:
            return d
        pi = ptf['indicators']
        atr = pi['atr']['value']
        cfg = self.strategy_config.get('breakout_retest', {})
        lb, md = int(cfg.get('lookback_bars',5)), float(cfg.get('max_distance_atr',0.5))
        cc = float(ptf_df['close'].iloc[-1])
        prior = ptf_df.iloc[-(lb+2):-1]
        if len(prior) < 3: return d

        if bias == 'bullish':
            dr, lv = 'long', float(prior['high'].max())
            closes_above = [float(r['close']) > lv for _, r in prior.iterrows()]
            consecutive = any(closes_above[i] and closes_above[i+1] 
                            for i in range(len(closes_above)-1))
            if not (consecutive and abs(cc-lv) <= md*atr and cc >= lv):
                return d
        else:
            dr, lv = 'short', float(prior['low'].min())
            closes_above = [float(r['close']) < lv for _, r in prior.iterrows()]
            consecutive = any(closes_above[i] and closes_above[i+1] 
                            for i in range(len(closes_above)-1))
            if not (consecutive and abs(cc-lv) <= md*atr and cc <= lv):
                return d

        sigs = ['breakout_held']
        if pi['supertrend']['trend'] == ('bullish' if dr=='long' else 'bearish'):
            sigs.append('supertrend_aligned')
        rsi = pi['rsi']['value']
        if dr=='long' and 40<rsi<70: sigs.append('rsi_confirmation')
        elif dr=='short' and 30<rsi<60: sigs.append('rsi_confirmation')
        if pi['adx'].get('trend_strength')=='strong': sigs.append('adx_strong')
        if dr=='long' and pi['candle_patterns'].get('bullish_engulfing'):
            sigs.append('candle_pattern')
        elif dr=='short' and pi['candle_patterns'].get('bearish_engulfing'):
            sigs.append('candle_pattern')
        e20,e50 = pi['ema'].get(20,0),pi['ema'].get(50,0)
        if dr=='long' and e20>e50: sigs.append('ema_stack_full')
        elif dr=='short' and e20<e50: sigs.append('ema_stack_full')
        if htf['trend']['direction']==('bullish' if dr=='long' else 'bearish'):
            sigs.append('price_structure_aligned')

        d.update({'signal':True,'entry_signal':True,'direction':dr,
                  'confluence_signals':sigs,
                  'entry_reason':f'Breakout retest {dr} @ {lv:.2f}'})
        return d

    def _ema_stack_pullback(self, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
        d = self._e()
        if bias not in ('bullish','bearish'): return d
        pi,hi,li = ptf['indicators'],htf['indicators'],ltf['indicators']
        e20p,e50p = pi['ema'].get(20,0),pi['ema'].get(50,0)
        e200p     = pi['ema'].get(200,0)
        e20h,e50h = hi['ema'].get(20,0),hi['ema'].get(50,0)
        e20l      = li['ema'].get(20, e20p)  # entry TF EMA20 for limit price precision
        atr = pi['atr']['value']
        cp  = ptf['ohlc']['close']

        if bias=='bullish':
            dr = 'long'
            ok = e20p>e50p>0 and e20h>e50h>0 and abs(cp-e20p)<=0.5*atr and cp>e20p
            full = e200p>0 and e20p>e50p>e200p
        else:
            dr = 'short'
            ok = e20p<e50p and e20h<e50h and abs(cp-e20p)<=0.5*atr and cp<e20p
            full = e200p>0 and e20p<e50p<e200p
        if not ok: return d

        sigs = ['ema_proximity_bounce']
        if full: sigs.append('ema_stack_full')
        if pi['supertrend']['trend']==('bullish' if dr=='long' else 'bearish'):
            sigs.append('supertrend_aligned')
        if 35<pi['rsi']['value']<65: sigs.append('rsi_confirmation')
        if pi['adx'].get('trend_strength')=='strong': sigs.append('adx_strong')
        if htf['trend']['direction']==('bullish' if dr=='long' else 'bearish'):
            sigs.append('price_structure_aligned')

        d.update({'signal':True,'entry_signal':True,'direction':dr,
                  'confluence_signals':sigs,
                  'entry_reason':f'EMA stack pullback {dr} @ {e20l:.2f}',
                  'limit_price':e20l})
        return d

    def _pullback_sr(self, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
        """
        FIXED: Uses actual swing points from price_structure for S/R.
        Falls back to EMA50 only when no swing data exists.
        """
        d = self._e()
        if bias not in ('bullish', 'bearish'):
            return d
    
        pi = ptf['indicators']
        cp = ptf['ohlc']['close']
        atr = pi['atr']['value']
    
        # ── Try real structure S/R first ─────────────────────────────────
        ps = pi.get('price_structure', {})
        swing_low = ps.get('last_swing_low')
        swing_high = ps.get('last_swing_high')
    
        sr_level = None
        dr = None
    
        if bias == 'bullish' and swing_low:
            distance = abs(cp - swing_low)
            if distance <= 0.5 * atr and cp > swing_low:
                sr_level = swing_low
                dr = 'long'
        elif bias == 'bearish' and swing_high:
            distance = abs(cp - swing_high)
            if distance <= 0.5 * atr and cp < swing_high:
                sr_level = swing_high
                dr = 'short'
    
        # ── Fallback to EMA50 ────────────────────────────────────────────
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
    
        # ── Build confluence signals ─────────────────────────────────────
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
        e20, e50v, e200 = pi['ema'].get(20, 0), pi['ema'].get(50, 0), pi['ema'].get(200, 0)
        if dr == 'long' and e20 > e50v > e200 > 0:
            sigs.append('ema_stack_full')
        elif dr == 'short' and e20 < e50v and e200 > 0 and e50v < e200:
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

    def _bb_squeeze(self, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
        d = self._e()
        if bias not in ('bullish', 'bearish'):
            return d
        pi = ptf['indicators']
        bb = pi['bollinger']
        atr = pi['atr']['value']
        cp = ptf['ohlc']['close']

        # Check if we WERE in squeeze (prior bars) and are now breaking out
        was_squeeze = False
        if ptf_df is not None and len(ptf_df) >= 5:
            try:
                prev_bb = self.indicators.calculate_bollinger_bands(ptf_df.iloc[:-1])
                was_squeeze = prev_bb['squeeze']
            except:
                was_squeeze = bb.get('squeeze', False)
        else:
            was_squeeze = bb.get('squeeze', False)

        if not was_squeeze:
            return d

        # Now check breakout direction — price should be moving AWAY from middle
        if bias == 'bullish' and cp <= bb['middle']:
            return d  # Not breaking out upward
        if bias == 'bearish' and cp >= bb['middle']:
            return d  # Not breaking out downward

        dr = 'long' if bias == 'bullish' else 'short'

        # Check for ATR expansion (current ATR > recent ATR)
        atr_exp = True
        if ptf_df is not None and len(ptf_df) >= 6:
            try:
                ra = self.indicators.calculate_atr(ptf_df.iloc[-6:-1])['current']
                atr_exp = atr > ra * 1.1
            except:
                pass

        sigs = ['bb_squeeze_active', 'bollinger_squeeze']
        if atr_exp:
            sigs.append('strong_trend')
        if pi['supertrend']['trend'] == ('bullish' if dr == 'long' else 'bearish'):
            sigs.append('supertrend_aligned')
        if pi['adx'].get('trend_strength') == 'strong':
            sigs.append('adx_strong')
        if htf['trend']['direction'] == ('bullish' if dr == 'long' else 'bearish'):
            sigs.append('price_structure_aligned')

        d.update({
            'signal': True, 'entry_signal': True, 'direction': dr,
            'confluence_signals': sigs, 'entry_reason': f'BB squeeze breakout {dr}'
        })
        return d

    def _macd_zero(self, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
        d = self._e()
        if bias not in ('bullish','bearish') or ptf_df is None or len(ptf_df)<3:
            return d
        pi = ptf['indicators']
        cm = pi['macd']['macd']
        try:
            pi2 = self.indicators.calculate_all(ptf_df.iloc[:-1])
            pm  = float(pi2['macd']['macd'].iloc[-1])
        except: return d
        if bias=='bullish':
            dr,cross = 'long', pm<0 and cm>0
        else:
            dr,cross = 'short', pm>0 and cm<0
        if not cross: return d

        sigs = ['macd_zero_cross']
        if pi['supertrend']['trend']==('bullish' if dr=='long' else 'bearish'):
            sigs.append('supertrend_aligned')
        rsi = pi['rsi']['value']
        if dr=='long' and 40<rsi<70: sigs.append('rsi_confirmation')
        elif dr=='short' and 30<rsi<60: sigs.append('rsi_confirmation')
        if pi['adx'].get('trend_strength')=='strong': sigs.append('adx_strong')
        e20,e50 = pi['ema'].get(20,0),pi['ema'].get(50,0)
        if dr=='long' and e20>e50>0: sigs.append('ema_stack_full')
        elif dr=='short' and e20<e50: sigs.append('ema_stack_full')
        if htf['trend']['direction']==('bullish' if dr=='long' else 'bearish'):
            sigs.append('price_structure_aligned')

        d.update({'signal':True,'entry_signal':True,'direction':dr,
                  'confluence_signals':sigs,'entry_reason':f'MACD zero-cross {dr}'})
        return d

    def _momentum(self, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
        d = self._e()
        pi = ptf['indicators']
        dr = None
        if pi['bollinger']['squeeze']:
            d['confluence_signals'].append('bb_squeeze_active')
        if pi['macd']['bullish_cross']:
            d['confluence_signals'].append('macd_bullish_cross')
            dr = 'long'
        elif pi['macd']['bearish_cross']:
            # Now uses proper bearish cross detection (prev >= signal, curr < signal)
            d['confluence_signals'].append('macd_bearish_cross')
            dr = 'short'
        else:
            return d
        if pi['adx'].get('trend_strength') == 'strong':
            d['confluence_signals'].append('adx_strong')
        if pi['supertrend']['trend'] == ('bullish' if dr == 'long' else 'bearish'):
            d['confluence_signals'].append('supertrend_aligned')
        if htf['trend']['direction'] == ('bullish' if dr == 'long' else 'bearish'):
            d['confluence_signals'].append('price_structure_aligned')
        d.update({
            'signal': True, 'entry_signal': True, 'direction': dr,
            'entry_reason': f'Momentum breakout {dr}'
        })
        return d

    def _find_swing_lows(self, series, order=5):
            lows = []
            for i in range(order, len(series) - order):
                if series.iloc[i] == series.iloc[i-order:i+order+1].min():
                    lows.append((i, series.iloc[i]))
            return lows

    def _rsi_div(self, htf, ptf, ltf, htf_df, ptf_df, ltf_df, bias):
        d = self._e()
        if ptf_df is None or len(ptf_df) < 20:
            return d
        try:
            ai = self.indicators.calculate_all(ptf_df)
            rs = ai['rsi']['value']
        except:
            return d

        lb = 20
        ps = ptf_df['close'].iloc[-lb:]
        rss = rs.iloc[-lb:]
        div, dr = None, None

        if bias == 'bullish':
            swing_lows = self._find_swing_lows(ps, order=3)
            if len(swing_lows) >= 2:
                prev_idx, prev_price = swing_lows[-2]
                curr_idx, curr_price = swing_lows[-1]
                # Bullish divergence: lower price low, higher RSI low
                if curr_price < prev_price:
                    try:
                        prev_rsi = float(rss.iloc[prev_idx])
                        curr_rsi = float(rss.iloc[curr_idx])
                        if curr_rsi > prev_rsi:
                            div, dr = 'bullish', 'long'
                    except:
                        pass

        elif bias == 'bearish':
            swing_highs = self._find_swing_highs(ps, order=3)
            if len(swing_highs) >= 2:
                prev_idx, prev_price = swing_highs[-2]
                curr_idx, curr_price = swing_highs[-1]
                # Bearish divergence: higher price high, lower RSI high
                if curr_price > prev_price:
                    try:
                        prev_rsi = float(rss.iloc[prev_idx])
                        curr_rsi = float(rss.iloc[curr_idx])
                        if curr_rsi < prev_rsi:
                            div, dr = 'bearish', 'short'
                    except:
                        pass

        if not div:
            return d

        pi = ptf['indicators']
        sigs = ['rsi_divergence_confirmed']
        if pi['supertrend']['trend'] == ('bullish' if dr == 'long' else 'bearish'):
            sigs.append('supertrend_aligned')
        if htf['trend']['direction'] == ('bullish' if dr == 'long' else 'bearish'):
            sigs.append('price_structure_aligned')
        if pi['adx'].get('trend_strength') == 'strong':
            sigs.append('adx_strong')

        ep = ltf['ohlc']['close']
        d.update({
            'signal': True, 'entry_signal': True, 'direction': dr,
            'confluence_signals': sigs,
            'entry_reason': f'RSI {div} divergence {dr}',
            'limit_price': ep
        })
        return d

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score(self, signals: List[str]) -> float:
        return sum(_SIGNAL_WEIGHTS.get(s,1) for s in signals)

    def _get_threshold(self, symbol: str) -> int:
        sc = self.config.get('symbols',{}).get(symbol,{})
        if 'confluence_threshold' in sc: return int(sc['confluence_threshold'])
        ms = symbol.replace('/','').upper()
        if ms in _VOLATILE_SYMBOLS or symbol in _VOLATILE_SYMBOLS:
            return int(self.strategy_config.get('confluence_required',7))
        return int(self.strategy_config.get('confluence_threshold_calm',5))

    # ── Filters ───────────────────────────────────────────────────────────────

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
    
        # ── Get confluence score for counter-trend override check ─────────
        score = analysis.get('confluence_score', 0)
        symbol = analysis.get('symbol', '')
        threshold = self._get_threshold(symbol)
        strong_override = score >= (threshold + 2)
    
        # ── Existing: HTF bias filter ────────────────────────────────────────
        if filters.get('respect_higher_tf_bias', True) and structure_tf in snaps:
            htf_trend = snaps[structure_tf]['trend']['direction']
            against_trend = (
                (analysis['direction'] == 'long' and htf_trend == 'bearish') or
                (analysis['direction'] == 'short' and htf_trend == 'bullish')
            )
            if against_trend and not filters.get('allow_override', False):
                if strong_override:
                    logger.info(
                        f"[{symbol}] Counter-trend allowed: score {score:.0f} >= "
                        f"threshold+2 ({threshold + 2})"
                    )
                else:
                    analysis['entry_signal'] = False
                    analysis['entry_reason'] = (
                        f'Filtered: against HTF {htf_trend} '
                        f'(score {score:.0f} < override {threshold + 2})'
                    )
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

    def _news_blackout(self):
        cfg  = self.strategy_config.get('filters',{}).get('news_blackout',{})
        now  = datetime.now().replace(tzinfo=None)
        ct   = now.time()
        cd   = now.weekday()+1
        for w in cfg.get('windows',[]):
            if cd not in w.get('days',[]): continue
            s = datetime.strptime(str(w['start']),'%H:%M').time()
            e = datetime.strptime(str(w['end']),'%H:%M').time()
            if s<=ct<=e: return True
        return False

    # ── SL / TP ───────────────────────────────────────────────────────────────

    def _calc_sl(self, analysis, entry_price, atr):
        dr    = analysis['direction']
        snaps = analysis['timeframe_snapshots']
        et    = analysis.get('entry_tf','5m')
        ps    = snaps.get(et, snaps.get(analysis.get('primary_timeframe',''),{})\
                ).get('indicators',{}).get('price_structure',{})

        if dr=='long':
            struct_sl = ps.get('last_swing_low', entry_price-2*atr)
        else:
            struct_sl = ps.get('last_swing_high', entry_price+2*atr)

        sc   = self.strategy_config.get('session_atr_multipliers',{})
        h    = datetime.now().replace(tzinfo=None).hour
        mult = (float(sc.get('asian',1.5))   if 0<=h<7   else
                float(sc.get('london',2.5))  if 7<=h<12  else
                float(sc.get('ny',2.5))      if 12<=h<20 else
                float(sc.get('overlap',2.0)))

        if dr=='long':
            atr_sl = entry_price - atr*mult
            return min(struct_sl, atr_sl)
        else:
            atr_sl = entry_price + atr*mult
            return max(struct_sl, atr_sl)

    def _calc_tps(self, analysis, entry_price, stop_loss):
        risk = abs(entry_price-stop_loss)
        dr   = analysis['direction']
        raw  = (self.config.get('risk_management',{})
                           .get('take_profit',{})
                           .get('targets',[]))
        tgts = [t for t in raw if float(t.get('rr_ratio',999))<999]
        if not tgts:
            tgts = [{'rr_ratio':2.0},{'rr_ratio':4.0}]
        tps = {}
        for i,t in enumerate(tgts,1):
            rr = float(t['rr_ratio'])
            tps[f'tp{i}'] = entry_price + risk*rr if dr=='long' else entry_price - risk*rr
        return tps