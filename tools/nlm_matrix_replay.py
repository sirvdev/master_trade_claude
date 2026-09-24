"""
tools/nlm_matrix_replay.py - replay the Nexus Liquidity Matrix engine on MT5
CSV exports and compare its trades with the screenshot ledger.

Usage (from the master_trade_nlm folder):
    python tools/nlm_matrix_replay.py --data-dir data/replay \
        --server-offset-hours 0 --start 2026-07-13 --end 2026-07-18

Files in --data-dir, one per symbol and timeframe, named <SYMBOL>_<TF>.csv,
e.g. XAUUSDm_M5.csv, XAUUSDm_H1.csv, XAUUSDm_H4.csv (M15 optional, built
from M5 when missing; H1/H4 are built from M5 when missing but then need 6+
weeks of M5). Accepts MT5 "Bars > Export" and HistoryCsvExporter layouts.

The engine is called exactly as live: on every 5M close, with the last 250
CLOSED bars of each frame. Fills are simulated bar by bar with the book's
management: two legs, 1:2 and 1:5, stop to entry when 1:2 prints, and the
book's pending-order cancel rules. A bar touching both stop and target is
counted as a stop (conservative).
"""
import argparse
import os
import sys
from datetime import timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from strategy.nlm_matrix_engine import NLMMatrixEngine  # noqa: E402

TF_FILE = {'5m': 'M5', '15m': 'M15', '1H': 'H1', '4H': 'H4', '1m': 'M1'}
TF_MIN = {'1m': 1, '5m': 5, '15m': 15, '1H': 60, '4H': 240}
LIVE_BARS = 250


def load_csv(path: str, offset_h: float) -> pd.DataFrame:
    raw = open(path, 'r', encoding='utf-8-sig', errors='replace').read(4096)
    sep = '\t' if raw.count('\t') > raw.count(',') else (';' if raw.count(';') > raw.count(',') else ',')
    df = pd.read_csv(path, sep=sep, encoding='utf-8-sig')
    df.columns = [str(c).strip('<> ').lower() for c in df.columns]
    if 'date' in df.columns and 'time' in df.columns:
        ts = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str),
                            format='mixed', dayfirst=False)
    else:
        col = next(c for c in df.columns if c in ('time', 'datetime', 'date', 'timestamp'))
        s = df[col]
        ts = pd.to_datetime(s, unit='s') if pd.api.types.is_numeric_dtype(s) else pd.to_datetime(s, format='mixed')
    out = pd.DataFrame({'open': df['open'].astype(float), 'high': df['high'].astype(float),
                        'low': df['low'].astype(float), 'close': df['close'].astype(float)})
    vol = next((c for c in ('tickvol', 'tick_volume', 'volume', 'vol') if c in df.columns), None)
    out['volume'] = df[vol].astype(float) if vol else 1.0
    out.index = ts - pd.Timedelta(hours=offset_h)       # to UTC
    out.index.name = 'time'
    return out[~out.index.duplicated()].sort_index()


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min',
                                  'close': 'last', 'volume': 'sum'}).dropna()


def load_symbol(data_dir, broker_sym, offset_h):
    frames = {}
    for tf, tag in TF_FILE.items():
        p = os.path.join(data_dir, f'{broker_sym}_{tag}.csv')
        if os.path.exists(p):
            frames[tf] = load_csv(p, offset_h)
    if '5m' not in frames and '1m' in frames:
        frames['5m'] = resample(frames['1m'], '5min')
    if '5m' not in frames:
        raise FileNotFoundError(f'{broker_sym}: need an M5 or M1 file')
    for tf, rule in (('15m', '15min'), ('1H', '1h'), ('4H', '4h')):
        if tf not in frames:
            frames[tf] = resample(frames['5m'], rule)
    return frames


def closed_view(frames, now_close):
    """Frames as the live system sees them at `now_close`: closed bars only."""
    view = {}
    for tf, df in frames.items():
        if tf == '1m':
            continue
        end = df.index + pd.Timedelta(minutes=TF_MIN[tf])
        view[tf] = df[end <= now_close].iloc[-LIVE_BARS:]
    return view


def simulate(setup, e5, frames, eng):
    """Book management: two legs; returns dict with fill time and R result."""
    short = setup['direction'] == 'short'
    entry, sl = setup['entry'], setup['sl']
    tp2, tp5 = setup['tp_partial'], setup['tp']
    t0 = pd.Timestamp(setup['signal_time']) + pd.Timedelta(minutes=5)
    bars = e5[e5.index >= t0]
    filled_at = None
    for t, b in bars.iterrows():
        close_t = t + pd.Timedelta(minutes=5)
        if filled_at is None:
            view = closed_view(frames, close_t)
            ok, why = eng.pending_order_valid(setup, view)
            if not ok:
                return {'result': 'cancelled', 'why': why, 'R': 0.0, 'time': str(t)}
            if (short and b['high'] >= sl) or ((not short) and b['low'] <= sl):
                return {'result': 'cancelled', 'why': 'price through stop before fill', 'R': 0.0,
                        'time': str(t)}
            if (short and b['high'] >= entry) or ((not short) and b['low'] <= entry):
                filled_at = t
                legA, legB, be = None, None, False
            else:
                continue
        # position management (same bar as the fill included, stop first)
        hit_sl = (b['high'] >= (entry if be else sl)) if short else (b['low'] <= (entry if be else sl))
        hit2 = (b['low'] <= tp2) if short else (b['high'] >= tp2)
        hit5 = (b['low'] <= tp5) if short else (b['high'] >= tp5)
        if legA is None:
            if hit_sl and not be:
                return {'result': 'loss', 'R': -1.0, 'fill': str(filled_at), 'time': str(t)}
            if hit2:
                legA, be = 2.0, True
                if hit5:
                    return {'result': 'win_5R', 'R': 0.5 * 2 + 0.5 * 5, 'fill': str(filled_at), 'time': str(t)}
                continue
        else:
            if hit_sl:
                return {'result': 'partial_then_BE', 'R': 0.5 * 2 + 0.0, 'fill': str(filled_at), 'time': str(t)}
            if hit5:
                return {'result': 'win_5R', 'R': 0.5 * 2 + 0.5 * 5, 'fill': str(filled_at), 'time': str(t)}
    return {'result': 'open' if filled_at is not None else 'pending', 'R': 0.0,
            'fill': str(filled_at) if filled_at is not None else ''}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--server-offset-hours', type=float, default=0.0)
    ap.add_argument('--start', default='2026-07-13')
    ap.add_argument('--end', default='2026-07-18')
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--ledger', default=os.path.join(HERE, 'screenshot_ledger_2026-07-13.csv'))
    ap.add_argument('--match-hours', type=float, default=2.0)
    ap.add_argument('--out', default='data/replay/nlm_matrix_replay')
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    eng = NLMMatrixEngine(cfg)
    symbols = [s for s, c in cfg['symbols'].items() if c.get('enabled')]
    rows, reasons = [], []
    for sym in symbols:
        broker = sym.replace('/', '')
        try:
            frames = load_symbol(args.data_dir, broker, args.server_offset_hours)
        except FileNotFoundError as e:
            print('SKIP', e)
            continue
        e5 = frames['5m']
        span = e5[(e5.index >= args.start) & (e5.index < args.end)]
        print(f'{sym}: {len(span)} 5M bars {span.index[0]} .. {span.index[-1]}')
        for t in span.index:
            now_close = t + pd.Timedelta(minutes=5)
            a = eng.analyze_market(sym, closed_view(frames, now_close))
            reasons.append({'symbol': sym, 'bar_close': now_close, 'signal': a['entry_signal'],
                            'reason': a['entry_reason']})
            if a['entry_signal']:
                s = a['nlm_setup']
                sim = simulate(s, e5, frames, eng)
                rows.append({'symbol': sym, 'signal_bar_close': now_close, 'direction': s['direction'],
                             'axis': s['axis_label'] + f" {s['axis']}", 'sweep_time': s['sweep_time'],
                             'entry': s['entry'], 'sl': s['sl'], 'tp_2R': s['tp_partial'],
                             'tp_5R': s['tp'], **sim})
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    trades = pd.DataFrame(rows)
    trades.to_csv(args.out + '_trades.csv', index=False)
    rs = pd.DataFrame(reasons)
    rs.to_csv(args.out + '_gates.csv', index=False)
    print('\nGate that stopped each 5M evaluation (count):')
    print(rs['reason'].str.extract(r'(G\d+)')[0].value_counts().to_string())
    print(f'\nEngine trades: {len(trades)}')
    if len(trades):
        print(trades[['symbol', 'signal_bar_close', 'direction', 'result', 'R']].to_string(index=False))

    # ── compare with the screenshot ledger ────────────────────────────────
    led = pd.read_csv(args.ledger)
    led['time_utc'] = pd.to_datetime(led['time_utc'])
    tol = pd.Timedelta(hours=args.match_hours)
    out = []
    for _, r in led.iterrows():
        m = trades[(trades['symbol'] == r['symbol']) & (trades['direction'] == r['direction'])] if len(trades) else trades
        if len(m):
            d = (pd.to_datetime(m['signal_bar_close']) - r['time_utc']).abs()
            m = m[d <= tol]
        out.append({**r.to_dict(), 'engine_match': bool(len(m)),
                    'engine_time': str(m['signal_bar_close'].iloc[0]) if len(m) else '',
                    'engine_result': m['result'].iloc[0] if len(m) else ''})
    cmp_ = pd.DataFrame(out)
    cmp_.to_csv(args.out + '_vs_screenshots.csv', index=False)
    print('\nScreenshot trades matched (same symbol, direction, within '
          f'{args.match_hours}h):')
    print(cmp_.groupby('setup')['engine_match'].agg(['sum', 'count']).to_string())


if __name__ == '__main__':
    main()
