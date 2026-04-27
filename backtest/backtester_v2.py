"""
backtest/backtester_v2.py — Production-grade backtester
========================================================
Accurately simulates the live trading system + EA position management.

Live flow (what this simulates):
  1. _symbol_bar_close_loop waits for PRIMARY TF bar to close
  2. _analyze_symbol fetches multi-TF data, calls analyze_market(symbol, data, symbol_config)
  3. If signal → calculate_entry_levels → validate_trade → place order
  4. EA ManagePositions() runs every 100ms tick:
     - Divides entry→TP2 into (PartialCloseSteps+1) segments
     - Partial closes at each milestone
     - Breakeven SL at BreakevenMinRR (1.0R)
     - ATR trailing on runner after all milestones hit
     - SL always checked before TP

Backtest simulation:
  - Walk forward bar-by-bar on PRIMARY timeframe
  - At each bar close, build lookback windows (only closed bars, no lookahead)
  - Call analyze_market with symbol_config (exact same call as live)
  - Between primary bars, walk 1m bars for granular position management
  - Simulate EA partial closes, breakeven, and ATR trailing on 1m bars

Usage:
  python -m backtest.backtester_v2 --symbol XAUUSD --start 2026-04-12 --end 2026-04-17
"""

import asyncio
import logging
import sys
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── Simulated time for backtest ────────────────────────────────────────────────
# In live trading, datetime.utcnow() returns wall-clock time, which matches
# the bar close time. In backtest, all bars are processed instantly, so
# utcnow() returns whatever time the user runs the script — breaking every
# session/killzone check in every engine.
#
# Solution: temporarily replace `datetime` in the engine module with a mock
# that returns the simulated bar time for utcnow()/now(), while preserving
# all other datetime functionality (fromisoformat, strptime, etc.).

@contextmanager
def simulated_time(engine, bar_time):
    """
    Context manager that makes datetime.utcnow() return the simulated bar_time
    inside the strategy engine module. All other datetime methods work normally.
    
    Usage:
        with simulated_time(strategy_engine, bar_ts):
            analysis = strategy_engine.analyze_market(...)
    """
    engine_module = type(engine).__module__
    
    # Build a mock datetime class that forwards everything to real datetime
    # except utcnow() and now() which return the simulated time
    mock_dt = MagicMock(wraps=datetime)
    mock_dt.utcnow.return_value = bar_time
    mock_dt.now.return_value = bar_time
    # Preserve class methods that engines use
    mock_dt.fromisoformat = datetime.fromisoformat
    mock_dt.fromtimestamp = datetime.fromtimestamp
    mock_dt.strptime = datetime.strptime
    mock_dt.combine = datetime.combine
    # Preserve constructors (datetime(...) calls)
    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
    
    with patch(f'{engine_module}.datetime', mock_dt):
        yield

# ── Constants ──────────────────────────────────────────────────────────────────

TF_SECONDS = {
    '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '1H': 3600, '4h': 14400, '4H': 14400,
    '1d': 86400, '1D': 86400,
}

MAX_BARS_PER_REQUEST = 13000


def tf_to_seconds(tf: str) -> int:
    s = TF_SECONDS.get(tf)
    if s is None:
        raise ValueError(f"Unknown timeframe: {tf!r}")
    return s


def tf_normalize(tf: str) -> str:
    """Normalize TF strings: '1H' and '1h' → '1h' etc."""
    mapping = {'1H': '1h', '4H': '4h', '1D': '1d'}
    return mapping.get(tf, tf.lower() if len(tf) <= 3 else tf)


# ── Market hours per symbol class ──────────────────────────────────────────────
# Trading hours per day (in seconds) and weekly structure.
# Used to convert "N bars of trading time" into calendar days for data fetching.
#
# Gold (XAUUSD/XAGUSD):
#   - Trades ~23 hrs/day (closes ~21:00-22:00 UTC daily, varies by broker)
#   - Mon-Fri, opens Sunday ~22:00 UTC
#   - 5 trading days/week
#
# Forex (EURUSD, GBPUSD, etc.):
#   - Trades ~21 hrs/day (brief daily rollover close)
#   - Mon-Fri, opens Sunday ~22:00 UTC
#   - 5 trading days/week
#
# Crypto (BTC/USD):
#   - Trades ~23-24 hrs/day (some brokers have 1hr maintenance)
#   - 7 days/week
#   - Weekend maintenance windows vary by broker

_MARKET_HOURS = {
    # symbol_prefix: (trading_seconds_per_day, trading_days_per_week)
    'XAU': (82800, 5),    # 23 hrs/day, Mon-Fri
    'XAG': (82800, 5),    # 23 hrs/day, Mon-Fri
    'EUR': (75600, 5),    # 21 hrs/day, Mon-Fri
    'GBP': (75600, 5),    # 21 hrs/day, Mon-Fri
    'USD': (75600, 5),    # 21 hrs/day, Mon-Fri (for pairs like USDJPY)
    'AUD': (75600, 5),    # 21 hrs/day, Mon-Fri
    'NZD': (75600, 5),    # 21 hrs/day, Mon-Fri
    'CAD': (75600, 5),    # 21 hrs/day, Mon-Fri
    'CHF': (75600, 5),    # 21 hrs/day, Mon-Fri
    'JPY': (75600, 5),    # 21 hrs/day, Mon-Fri
    'BTC': (82800, 7),    # 23 hrs/day, 7 days (1hr broker maintenance)
    'ETH': (82800, 7),    # 23 hrs/day, 7 days
    'NAS': (23400, 5),    # 6.5 hrs/day, Mon-Fri (US market hours)
    'US1': (23400, 5),    # US100 etc.
    'US3': (23400, 5),    # US30 etc.
}


def _get_market_schedule(symbol: str):
    """Get (trading_secs_per_day, trading_days_per_week) for a symbol."""
    norm = symbol.replace('/', '').upper()
    for prefix, schedule in _MARKET_HOURS.items():
        if norm.startswith(prefix):
            return schedule
    return (75600, 5)  # Default: forex-like (21hrs, Mon-Fri)


# ── Position tracking (mirrors EA's PositionTrack struct) ──────────────────────

@dataclass
class SimPosition:
    """Mirrors EA's PositionTrack struct + Python trade metadata."""
    trade_id: str
    symbol: str
    direction: str            # 'long' | 'short'
    entry_time: datetime
    entry_price: float
    stop_loss: float          # current SL (moves with breakeven/trailing)
    original_sl: float        # initial SL (never changes)
    take_profit_2: float      # TP2 = EA's ceiling for milestone spacing
    position_size: float      # total lots at entry
    remaining_size: float     # current lots (decreases with partials)

    # EA milestone tracking
    total_milestones: int = 3
    next_milestone: int = 0
    step_distance: float = 0.0
    portion_lots: float = 0.0
    runner_lots: float = 0.0
    breakeven_done: bool = False
    runner_active: bool = False

    # Results
    realized_pnl: float = 0.0     # accumulated PnL from partial closes
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    max_favorable: float = 0.0
    max_adverse: float = 0.0


# ── Progress bar ───────────────────────────────────────────────────────────────

class ProgressBar:
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width

    def update(self, current: int, extra: str = ''):
        pct = current / max(self.total, 1)
        done = int(self.width * pct)
        bar = '#' * done + '-' * (self.width - done)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1%}  {current}/{self.total}  {extra}")
        sys.stdout.flush()

    def finish(self):
        self.update(self.total)
        print()


# ── Backtester ─────────────────────────────────────────────────────────────────

class BacktesterV2:
    """
    Production-grade backtester that mirrors the live trading system.
    """

    def __init__(self, config: Dict):
        bt_cfg = config.get('backtest', {})
        sim_cfg = bt_cfg.get('simulation', {})
        tf_cfg = bt_cfg.get('timeframes', {})

        self.lookback_bars = bt_cfg.get('lookback_bars', 250)
        self.slippage_pct = sim_cfg.get('slippage_percent', 0.05)
        self.commission_pct = sim_cfg.get('commission_percent', 0.1)
        self.latency_bars = sim_cfg.get('latency_bars', 1)
        self.max_concurrent = (
            config.get('risk_management', {})
                  .get('global_limits', {})
                  .get('max_concurrent_trades', 3)
        )

        # EA config
        ea_cfg = bt_cfg.get('ea_simulation', {})
        self.partial_close_steps = ea_cfg.get('partial_close_steps', 3)
        self.runner_percent = ea_cfg.get('runner_percent', 10.0)
        self.breakeven_min_rr = ea_cfg.get('breakeven_min_rr', 1.0)
        self.trail_atr_multiplier = ea_cfg.get('trail_atr_multiplier', 2.0)
        self.trail_atr_period = ea_cfg.get('trail_atr_period', 14)

        self._symbols_config = config.get('symbols', {})
        self._full_config = config

        # Runtime state
        self.open_positions: Dict[str, SimPosition] = {}
        self.closed_positions: List[SimPosition] = []
        self.balance: float = 0.0
        self.equity_curve: List[float] = []

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    async def run(
        self,
        strategy_engine,
        money_manager,
        stop_manager,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 100_000,
    ) -> Dict:
        """Run full backtest matching live system behavior."""

        # ── Resolve symbol config ────────────────────────────────────
        sym_key = next(
            (k for k in self._symbols_config
             if k.replace('/', '') == symbol.replace('/', '') or k == symbol),
            None
        )
        sym_cfg = self._symbols_config.get(sym_key, {}) if sym_key else {}

        # ── Determine timeframe roles (EXACTLY as live does) ─────────
        # IMPORTANT: Keep original case from config — MT5 bridge uses
        # '1H'/'4H'/'1D' not lowercase. tf_normalize() was converting
        # them to lowercase which the bridge couldn't map.
        tfs = sym_cfg.get('timeframes', ['4H', '15m', '5m'])
        primary_tf = sym_cfg.get('primary_timeframe', tfs[1] if len(tfs) > 1 else '15m')
        entry_tf = sym_cfg.get('entry_timeframe', tfs[-1] if tfs else '5m')
        structure_tf = tfs[0] if tfs else '4H'

        # Analysis TFs = the symbol's configured timeframes (for indicators)
        # These get passed to strategy_engine.analyze_market()
        analysis_tfs = list(dict.fromkeys(tfs))

        # Min TF = 1m for granular SL/TP/partial close simulation
        # NOT passed to strategy engine — only used for exit management
        min_tf = '1m'

        print()
        print("=" * 70)
        print(f"  BACKTEST v2  |  {symbol}  |  "
              f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        print(f"  Structure: {structure_tf}  |  Primary (walk): {primary_tf}  |  "
              f"Entry: {entry_tf}")
        print(f"  EA sim: {self.partial_close_steps} steps, "
              f"BE@{self.breakeven_min_rr}R, "
              f"trail {self.trail_atr_multiplier}×ATR")
        print("=" * 70)

        # ── Fetch data ───────────────────────────────────────────────
        raw_data = await self._fetch_data(
            symbol, analysis_tfs, min_tf, primary_tf, start_date, end_date
        )

        # ── Run simulation ───────────────────────────────────────────
        results = self._simulate(
            strategy_engine, money_manager, stop_manager,
            raw_data, symbol, sym_cfg, primary_tf, min_tf,
            start_date, end_date, initial_balance, analysis_tfs,
        )

        return results

    # ══════════════════════════════════════════════════════════════════════
    # DATA FETCHING (per-TF lookback windows)
    # ══════════════════════════════════════════════════════════════════════

    def _calc_lookback_date(self, symbol, start_date, lookback_trading_secs):
        """
        Calculate the fetch_from date by converting trading-time lookback
        into calendar days, accounting for:
        - Daily market close hours (gold ~1hr/day, forex ~3hrs/day)
        - Weekends (5-day vs 7-day markets)
        - Safety buffer for broker holidays

        Example for XAUUSD with 250 bars of 5m:
          lookback_trading_secs = 250 × 300 = 75,000 trading seconds
          Gold trades 23 hrs/day (82,800 secs), 5 days/week
          Trading days needed = ceil(75,000 / 82,800) = 1 day
          Calendar days = ceil(1 × 7/5) + 2 buffer = 4 calendar days
          fetch_from = start_date - 4 days
        """
        trading_secs_per_day, trading_days_per_week = _get_market_schedule(symbol)

        # How many full trading days does the lookback span?
        trading_days_needed = math.ceil(
            lookback_trading_secs / max(trading_secs_per_day, 1)
        )

        # Ensure at least 1 full trading day of lookback
        trading_days_needed = max(trading_days_needed, 1)

        # Convert trading days → calendar days
        if trading_days_per_week >= 7:
            # Crypto: trades every day (with possible brief maintenance)
            calendar_days = trading_days_needed + 1  # +1 for maintenance gaps
        else:
            # Forex/metals: 5 trading days per 7 calendar days
            full_weeks = trading_days_needed // trading_days_per_week
            remaining_days = trading_days_needed % trading_days_per_week
            calendar_days = (full_weeks * 7) + remaining_days

            # Add weekend bridge: if the lookback might land on a weekend,
            # add 2 extra days to ensure we reach Friday's data
            calendar_days += 2

        # Safety buffer for broker holidays (e.g., Easter, Christmas)
        calendar_days += 1

        fetch_from = start_date - timedelta(days=calendar_days)

        logger.debug(
            f"[LOOKBACK] {symbol}: {lookback_trading_secs}s trading time "
            f"= {trading_days_needed} trading days "
            f"= {calendar_days} calendar days "
            f"→ fetch from {fetch_from.strftime('%Y-%m-%d')}"
        )

        return fetch_from

    async def _fetch_data(self, symbol, analysis_tfs, min_tf, primary_tf,
                          start_date, end_date):
        from execution.mt5_file_bridge import MT5FileBridge

        bridge = MT5FileBridge(config={}, demo_mode=False)
        await bridge.connect()
        raw = {}

        print(f"\n  Fetching data from MT5...")

        # Analysis TFs — each with its own lookback
        for tf in analysis_tfs:
            tf_secs = tf_to_seconds(tf)
            lookback_trading_secs = self.lookback_bars * tf_secs

            # Calculate how many CALENDAR days of lookback we need,
            # accounting for market-closed hours and weekends.
            fetch_from = self._calc_lookback_date(
                symbol, start_date, lookback_trading_secs
            )
            est_bars = int((end_date - fetch_from).total_seconds() / tf_secs)

            print(f"  {tf:>4s}: from {fetch_from.strftime('%Y-%m-%d')} "
                  f"(~{est_bars:,} bars)...", end='', flush=True)

            try:
                if est_bars <= MAX_BARS_PER_REQUEST:
                    df = await bridge.fetch_historical_range(
                        symbol=symbol.replace('/', ''), timeframe=tf,
                        from_dt=fetch_from, to_dt=end_date,
                    )
                else:
                    df = await self._chunked_fetch(bridge, symbol.replace('/', ''),
                                                   tf, fetch_from, end_date, tf_secs)
                raw[tf] = df
                if len(df) > 0:
                    print(f" {len(df):>6,} bars OK")
                else:
                    print(f" 0 bars ⚠️")
            except Exception as e:
                print(f" FAILED: {e}")
                raise

        # 1m for granular exit checking — test period only
        # UNLESS 1m is also an analysis TF (then it already has lookback)
        if min_tf not in raw:
            tf_secs = tf_to_seconds(min_tf)
            est_bars = int((end_date - start_date).total_seconds() / tf_secs)
            print(f"  {min_tf:>4s}: test period only (~{est_bars:,} bars)...",
                  end='', flush=True)
            try:
                if est_bars <= MAX_BARS_PER_REQUEST:
                    df = await bridge.fetch_historical_range(
                        symbol=symbol.replace('/', ''), timeframe=min_tf,
                        from_dt=start_date, to_dt=end_date,
                    )
                else:
                    df = await self._chunked_fetch(bridge, symbol.replace('/', ''),
                                                   min_tf, start_date, end_date, tf_secs)
                raw[min_tf] = df
                print(f" {len(df):>6,} bars OK")
            except Exception as e:
                print(f" FAILED: {e}")
                raise

        await bridge.disconnect()

        # Validation
        self._validate(raw, analysis_tfs, min_tf, start_date, end_date)
        return raw

    async def _chunked_fetch(self, bridge, symbol, tf, from_date, to_date, tf_secs):
        chunk_dur = timedelta(seconds=MAX_BARS_PER_REQUEST * tf_secs * 0.9)
        chunks = []
        start = from_date
        while start < to_date:
            end = min(start + chunk_dur, to_date)
            try:
                df = await bridge.fetch_historical_range(
                    symbol=symbol, timeframe=tf, from_dt=start, to_dt=end,
                )
                if len(df) > 0:
                    chunks.append(df)
            except Exception as e:
                logger.warning(f"Chunk fetch failed {tf}: {e}")
            start = end - timedelta(seconds=tf_secs * 5)

        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks)
        return combined[~combined.index.duplicated(keep='first')].sort_index()

    def _validate(self, raw, analysis_tfs, min_tf, start_date, end_date):
        start_ts = pd.Timestamp(start_date, tz='UTC')
        print(f"\n  Data validation:")
        for tf in analysis_tfs + [min_tf]:
            df = raw.get(tf)
            if df is None or df.empty:
                print(f"    {tf:>4s}: ❌ NO DATA")
                continue
            test_bars = len(df[df.index >= start_ts])
            lookback = len(df[df.index < start_ts])
            if tf != min_tf:
                print(f"    {tf:>4s}: ✅ {lookback} lookback + {test_bars} test = {len(df)} total")
            else:
                print(f"    {tf:>4s}: ✅ {test_bars} test period bars")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # CORE SIMULATION
    # ══════════════════════════════════════════════════════════════════════

    def _simulate(self, strategy_engine, money_manager, stop_manager,
                  raw_data, symbol, sym_cfg, primary_tf, min_tf,
                  start_date, end_date, initial_balance, analysis_tfs):

        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.open_positions = {}
        self.closed_positions = []

        # Get the primary TF bars within the test period
        primary_df = raw_data[primary_tf]
        min_df = raw_data.get(min_tf, pd.DataFrame())

        start_ts = pd.Timestamp(start_date, tz='UTC')
        end_ts = pd.Timestamp(end_date, tz='UTC')

        test_bars = primary_df[
            (primary_df.index >= start_ts) & (primary_df.index <= end_ts)
        ]

        if test_bars.empty:
            raise ValueError("No primary-TF bars in test period")

        total = len(test_bars)
        primary_secs = tf_to_seconds(primary_tf)
        progress = ProgressBar(total)
        trade_counter = 0

        for idx, (bar_ts, bar) in enumerate(test_bars.iterrows()):

            # ── STEP 1: Manage open positions on 1m bars within this period ──
            # This happens BEFORE analysis (like EA running continuously).
            #
            # Use the PREVIOUS primary bar's timestamp as the window start,
            # NOT arithmetic subtraction. This correctly handles market close
            # gaps — e.g., gold's daily 1-hour close won't produce phantom
            # 1m bars, and we won't miss pre-close bars either.
            if idx > 0:
                prev_bar_ts = test_bars.index[idx - 1]
            else:
                prev_bar_ts = bar_ts - timedelta(seconds=primary_secs)

            one_min_bars = min_df[
                (min_df.index > prev_bar_ts) & (min_df.index <= bar_ts)
            ]
            self._manage_positions_granular(one_min_bars, raw_data, symbol)

            # ── STEP 2: At bar close, run strategy analysis ──────────────
            # Build lookback data for ANALYSIS timeframes only (not 1m).
            # 1m is only for SL/TP granular checking — the strategy engine
            # never sees it in live (live fetches only the symbol's configured TFs).
            tf_snapshot = {}
            for tf in analysis_tfs:
                df = raw_data.get(tf)
                if df is None:
                    continue
                available = df[df.index <= bar_ts]
                n = min(len(available), self.lookback_bars)
                if n > 0:
                    tf_snapshot[tf] = available.iloc[-n:]

            # Check: can we take new trades?
            if len(self.open_positions) < self.max_concurrent:
                try:
                    # Call analyze_market with symbol_config — SAME as live.
                    # Wrap in simulated_time so datetime.utcnow() returns the
                    # bar close time, not wall-clock time. This makes
                    # killzone/session checks work correctly in backtest.
                    bar_time = bar_ts.to_pydatetime().replace(tzinfo=None)
                    
                    with simulated_time(strategy_engine, bar_time):
                        analysis = strategy_engine.analyze_market(
                            symbol, tf_snapshot, symbol_config=sym_cfg
                        )

                    if analysis.get('entry_signal'):
                        levels = strategy_engine.calculate_entry_levels(
                            analysis, tf_snapshot
                        )

                        if levels and levels.get('stop_loss'):
                            sizing = money_manager.validate_trade(
                                account_equity=self.balance,
                                entry_price=levels['entry_price'],
                                stop_loss=levels['stop_loss'],
                                symbol=symbol,
                                direction=analysis['direction'],
                                daily_stats={},
                                recent_trades=[],
                            )

                            if sizing.get('approved'):
                                # Entry with 1-bar latency
                                entry_idx = idx + self.latency_bars
                                if entry_idx < total:
                                    entry_bar = test_bars.iloc[entry_idx]
                                    entry_ts = test_bars.index[entry_idx]

                                    pos = self._open_position(
                                        trade_id=f"bt_{trade_counter}",
                                        symbol=symbol,
                                        direction=analysis['direction'],
                                        entry_time=entry_ts.to_pydatetime(),
                                        entry_bar=entry_bar,
                                        levels=levels,
                                        position_size=sizing['position_size'],
                                    )
                                    if pos:
                                        self.open_positions[pos.trade_id] = pos
                                        trade_counter += 1

                except Exception as e:
                    logger.debug(f"Analysis error at {bar_ts}: {e}")

            # ── STEP 3: Track equity ─────────────────────────────────────
            current_price = float(bar['close'])
            unrealized = sum(
                self._unrealized_pnl(p, current_price)
                for p in self.open_positions.values()
            )
            self.equity_curve.append(self.balance + unrealized)

            if idx % max(1, total // 50) == 0 or idx == total - 1:
                progress.update(idx + 1,
                    f"bal:${self.balance:,.0f} open:{len(self.open_positions)} "
                    f"closed:{len(self.closed_positions)}")

        progress.finish()

        # Close remaining positions at last price
        last_price = float(test_bars.iloc[-1]['close'])
        last_time = test_bars.index[-1].to_pydatetime()
        for pos in list(self.open_positions.values()):
            self._close_position(pos, last_time, last_price, 'end_of_backtest')

        return self._build_results(symbol, initial_balance)

    # ══════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT (mirrors EA ManagePositions)
    # ══════════════════════════════════════════════════════════════════════

    def _manage_positions_granular(self, one_min_bars, raw_data, symbol):
        """
        Walk each 1m bar and manage all open positions.
        Mirrors EA's ManagePositions() running every 100ms.
        """
        if one_min_bars.empty:
            return

        for ts, bar in one_min_bars.iterrows():
            bar_high = float(bar['high'])
            bar_low = float(bar['low'])
            bar_close = float(bar['close'])

            for pos in list(self.open_positions.values()):
                # Update MFE/MAE
                if pos.direction == 'long':
                    pos.max_favorable = max(pos.max_favorable, bar_high - pos.entry_price)
                    pos.max_adverse = max(pos.max_adverse, pos.entry_price - bar_low)
                else:
                    pos.max_favorable = max(pos.max_favorable, pos.entry_price - bar_low)
                    pos.max_adverse = max(pos.max_adverse, bar_high - pos.entry_price)

                # ── 1. CHECK SL (always first priority) ──────────────────
                sl_hit = False
                if pos.direction == 'long' and bar_low <= pos.stop_loss:
                    sl_hit = True
                    exit_price = pos.stop_loss
                elif pos.direction == 'short' and bar_high >= pos.stop_loss:
                    sl_hit = True
                    exit_price = pos.stop_loss

                if sl_hit:
                    self._close_position(pos, ts.to_pydatetime(), exit_price, 'stop_loss')
                    continue

                # ── 2. CHECK BREAKEVEN (at BreakevenMinRR) ───────────────
                if not pos.breakeven_done:
                    risk = abs(pos.entry_price - pos.original_sl)
                    if risk > 0:
                        if pos.direction == 'long':
                            current_rr = (bar_high - pos.entry_price) / risk
                        else:
                            current_rr = (pos.entry_price - bar_low) / risk

                        if current_rr >= self.breakeven_min_rr:
                            # Move SL to entry (+ small buffer)
                            buffer = risk * 0.02  # 2% of risk as buffer
                            if pos.direction == 'long':
                                new_sl = pos.entry_price + buffer
                                if new_sl > pos.stop_loss:
                                    pos.stop_loss = new_sl
                            else:
                                new_sl = pos.entry_price - buffer
                                if new_sl < pos.stop_loss:
                                    pos.stop_loss = new_sl
                            pos.breakeven_done = True

                # ── 3. CHECK MILESTONES (partial closes) ─────────────────
                if not pos.runner_active and pos.total_milestones > 0:
                    next_m = pos.next_milestone
                    if next_m < pos.total_milestones:
                        if pos.direction == 'long':
                            milestone_price = pos.entry_price + (next_m + 1) * pos.step_distance
                            reached = bar_high >= milestone_price
                        else:
                            milestone_price = pos.entry_price - (next_m + 1) * pos.step_distance
                            reached = bar_low <= milestone_price

                        if reached:
                            # Partial close
                            is_last = (next_m == pos.total_milestones - 1)
                            if is_last:
                                close_lots = pos.remaining_size - pos.runner_lots
                            else:
                                close_lots = pos.portion_lots

                            close_lots = min(close_lots, pos.remaining_size - pos.runner_lots)
                            if close_lots > 0.001:
                                # Calculate PnL for this partial
                                if pos.direction == 'long':
                                    partial_pnl = (milestone_price - pos.entry_price) * close_lots
                                else:
                                    partial_pnl = (pos.entry_price - milestone_price) * close_lots

                                # Deduct commission
                                partial_pnl -= milestone_price * close_lots * (self.commission_pct / 100)

                                pos.realized_pnl += partial_pnl
                                self.balance += partial_pnl
                                pos.remaining_size -= close_lots

                            pos.next_milestone += 1

                            if pos.next_milestone >= pos.total_milestones:
                                pos.runner_active = True

                # ── 4. RUNNER TRAILING (ATR trail after all milestones) ───
                if pos.runner_active and pos.remaining_size > 0.001:
                    # Compute ATR from 1H data if available
                    atr = self._get_atr_for_trailing(raw_data, ts)
                    if atr > 0:
                        trail_dist = atr * self.trail_atr_multiplier
                        if pos.direction == 'long':
                            new_sl = bar_high - trail_dist
                            if new_sl > pos.stop_loss:
                                pos.stop_loss = new_sl
                        else:
                            new_sl = bar_low + trail_dist
                            if new_sl < pos.stop_loss:
                                pos.stop_loss = new_sl

    def _get_atr_for_trailing(self, raw_data, current_ts, period=14):
        """Compute ATR from 1H bars for runner trailing."""
        for tf in ['1h', '1H', '4h', '4H']:
            if tf in raw_data:
                df = raw_data[tf]
                available = df[df.index <= current_ts]
                if len(available) >= period + 1:
                    highs = available['high'].iloc[-(period+1):].values
                    lows = available['low'].iloc[-(period+1):].values
                    closes = available['close'].iloc[-(period+1):].values
                    tr = []
                    for i in range(1, len(highs)):
                        tr.append(max(
                            highs[i] - lows[i],
                            abs(highs[i] - closes[i-1]),
                            abs(lows[i] - closes[i-1])
                        ))
                    if tr:
                        return sum(tr) / len(tr)
        return 0.0

    # ══════════════════════════════════════════════════════════════════════
    # OPEN / CLOSE POSITIONS
    # ══════════════════════════════════════════════════════════════════════

    def _open_position(self, trade_id, symbol, direction, entry_time,
                       entry_bar, levels, position_size) -> Optional[SimPosition]:
        """Open a new position, setting up EA milestone tracking."""

        target_price = levels.get('order_price', levels['entry_price'])
        sl = levels['stop_loss']
        tp2 = levels.get('take_profit_2', levels.get('take_profit_1', 0))

        # Simulate slippage
        slip = abs(target_price * self.slippage_pct / 100)
        if direction == 'long':
            fill_price = target_price + slip
        else:
            fill_price = target_price - slip

        # Clamp to bar range
        fill_price = max(float(entry_bar['low']),
                         min(float(entry_bar['high']), fill_price))

        # Entry commission
        self.balance -= fill_price * position_size * (self.commission_pct / 100)

        # ── Set up EA milestone tracking ─────────────────────────────
        # EA divides entry→TP2 into (steps+1) segments
        if direction == 'long':
            distance = tp2 - fill_price
        else:
            distance = fill_price - tp2

        if distance <= 0:
            logger.debug(f"Invalid TP distance for {trade_id}, skipping")
            return None

        # Runner lots
        runner_lots = max(0.01, position_size * self.runner_percent / 100)
        if runner_lots >= position_size:
            runner_lots = 0.01
        working_lots = position_size - runner_lots

        # Effective steps (reduce if portion would be < min lot)
        eff_steps = self.partial_close_steps
        min_lot = 0.01
        while eff_steps > 0:
            portion = working_lots / eff_steps
            if portion >= min_lot:
                break
            eff_steps -= 1

        if eff_steps == 0:
            runner_lots = position_size
            portion = 0

        step_dist = distance / (eff_steps + 1) if eff_steps > 0 else distance

        pos = SimPosition(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            entry_price=fill_price,
            stop_loss=sl,
            original_sl=sl,
            take_profit_2=tp2,
            position_size=position_size,
            remaining_size=position_size,
            total_milestones=eff_steps,
            step_distance=step_dist,
            portion_lots=round(working_lots / eff_steps, 2) if eff_steps > 0 else 0,
            runner_lots=round(runner_lots, 2),
        )

        logger.info(
            f"  OPEN {direction.upper()} {symbol} @ {fill_price:.2f} "
            f"SL:{sl:.2f} TP2:{tp2:.2f} Size:{position_size:.2f} "
            f"Steps:{eff_steps} Runner:{runner_lots:.2f}"
        )
        return pos

    def _close_position(self, pos: SimPosition, exit_time, exit_price, reason):
        """Fully close remaining position."""
        # PnL on remaining lots
        if pos.direction == 'long':
            remaining_pnl = (exit_price - pos.entry_price) * pos.remaining_size
        else:
            remaining_pnl = (pos.entry_price - exit_price) * pos.remaining_size

        # Commission
        remaining_pnl -= exit_price * pos.remaining_size * (self.commission_pct / 100)

        pos.realized_pnl += remaining_pnl
        self.balance += remaining_pnl
        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.remaining_size = 0

        self.closed_positions.append(pos)
        self.open_positions.pop(pos.trade_id, None)

        logger.info(
            f"  CLOSE {pos.symbol} @ {exit_price:.2f} ({reason}) "
            f"Total P&L:${pos.realized_pnl:+.2f}"
        )

    def _unrealized_pnl(self, pos: SimPosition, current_price: float) -> float:
        if pos.direction == 'long':
            return (current_price - pos.entry_price) * pos.remaining_size
        else:
            return (pos.entry_price - current_price) * pos.remaining_size

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════

    def _build_results(self, symbol, initial_balance):
        positions = self.closed_positions
        n = len(positions)

        if n == 0:
            self._print_results({'total_trades': 0, 'win_rate': 0,
                'total_pnl': 0, 'final_balance': self.balance,
                'return_percent': (self.balance/initial_balance-1)*100,
                'max_drawdown': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
                'expectancy': 0, 'winning_trades': 0, 'losing_trades': 0,
                'avg_rr': 0, 'avg_win': 0, 'avg_loss': 0,
                'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
                'avg_duration_hours': 0, 'equity_curve': self.equity_curve,
                'trades': [], 'symbol': symbol})
            return {'total_trades': 0}

        pnls = [p.realized_pnl for p in positions]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        # Risk-reward
        rrs = []
        for p in positions:
            risk = abs(p.entry_price - p.original_sl) * p.position_size
            rrs.append(p.realized_pnl / risk if risk > 0 else 0)

        # Drawdown
        peak = initial_balance
        max_dd = 0.0
        running = initial_balance
        for p in pnls:
            running += p
            peak = max(peak, running)
            dd = (running - peak) / peak * 100
            max_dd = min(max_dd, dd)

        # Streaks
        max_cw = max_cl = cw = cl = 0
        for p in pnls:
            if p > 0:
                cw += 1; cl = 0; max_cw = max(max_cw, cw)
            else:
                cl += 1; cw = 0; max_cl = max(max_cl, cl)

        # Sharpe
        eq = pd.Series(self.equity_curve)
        rets = eq.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * (252**0.5)) if rets.std() > 0 else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        durations = [
            (p.exit_time - p.entry_time).total_seconds() / 3600
            for p in positions if p.exit_time
        ]

        results = {
            'symbol': symbol,
            'total_trades': n,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / n,
            'total_pnl': sum(pnls),
            'avg_win': sum(wins)/len(wins) if wins else 0,
            'avg_loss': sum(losses)/len(losses) if losses else 0,
            'avg_rr': sum(rrs)/n,
            'profit_factor': gross_profit/gross_loss if gross_loss > 0 else float('inf'),
            'expectancy': sum(pnls)/n,
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'max_consecutive_wins': max_cw,
            'max_consecutive_losses': max_cl,
            'avg_duration_hours': sum(durations)/len(durations) if durations else 0,
            'final_balance': self.balance,
            'return_percent': (self.balance/initial_balance - 1) * 100,
            'equity_curve': self.equity_curve,
            'trades': [
                {
                    'trade_id': p.trade_id, 'symbol': p.symbol,
                    'direction': p.direction, 'entry_time': p.entry_time,
                    'entry_price': p.entry_price, 'exit_time': p.exit_time,
                    'exit_price': p.exit_price, 'stop_loss': p.original_sl,
                    'pnl': p.realized_pnl, 'exit_reason': p.exit_reason,
                    'milestones_hit': p.next_milestone,
                    'runner_active': p.runner_active,
                    'breakeven_done': p.breakeven_done,
                }
                for p in positions
            ],
        }

        self._print_results(results)
        return results

    def _print_results(self, r):
        print()
        print("=" * 70)
        print("  RESULTS")
        print("=" * 70)
        print(f"  Total Trades        : {r['total_trades']}")
        if r['total_trades'] > 0:
            print(f"  Win Rate            : {r['win_rate']:.2%}  "
                  f"({r['winning_trades']}W / {r['losing_trades']}L)")
            print(f"  Avg R:R             : {r['avg_rr']:.2f}")
            print(f"  Profit Factor       : {r['profit_factor']:.2f}")
            print(f"  Expectancy          : ${r['expectancy']:+.2f} / trade")
            print()
            print(f"  Total P&L           : ${r['total_pnl']:+,.2f}")
        print(f"  Final Balance       : ${r['final_balance']:,.2f}")
        print(f"  Return              : {r['return_percent']:+.2f}%")
        if r['total_trades'] > 0:
            print()
            print(f"  Max Drawdown        : {r['max_drawdown']:.2f}%")
            print(f"  Sharpe Ratio        : {r['sharpe_ratio']:.2f}")
            print(f"  Max Consec. Wins    : {r['max_consecutive_wins']}")
            print(f"  Max Consec. Losses  : {r['max_consecutive_losses']}")
            print(f"  Avg Trade Duration  : {r['avg_duration_hours']:.1f} hours")
        print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════════

    def export_trades(self, filepath: str):
        import csv, os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        if not self.closed_positions:
            return
        fields = ['trade_id','symbol','direction','entry_time','entry_price',
                  'exit_time','exit_price','stop_loss','pnl','exit_reason',
                  'milestones_hit','breakeven_done','runner_active','duration_hours']
        with open(filepath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for p in self.closed_positions:
                w.writerow({
                    'trade_id': p.trade_id, 'symbol': p.symbol,
                    'direction': p.direction, 'entry_time': p.entry_time,
                    'entry_price': p.entry_price, 'exit_time': p.exit_time,
                    'exit_price': p.exit_price, 'stop_loss': p.original_sl,
                    'pnl': p.realized_pnl, 'exit_reason': p.exit_reason,
                    'milestones_hit': p.next_milestone,
                    'breakeven_done': p.breakeven_done,
                    'runner_active': p.runner_active,
                    'duration_hours': (p.exit_time-p.entry_time).total_seconds()/3600
                                       if p.exit_time else 0,
                })
        print(f"\n  Trades exported to {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    import yaml

    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from strategy.engine import StrategyEngine
    from risk_management.money_manager import MoneyManager
    from risk_management.stop_manager import StopManager

    parser = argparse.ArgumentParser(description='Backtest v2 — live-accurate simulation')
    parser.add_argument('--symbol', required=True, help='e.g. XAUUSD')
    parser.add_argument('--start', required=True, help='YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='YYYY-MM-DD')
    parser.add_argument('--balance', type=float, default=100000)
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'config' / 'config.yaml'))
    parser.add_argument('--engine', default='original',
                        choices=['original', 'ict', 'smc'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.engine == 'ict':
        from strategy.ict_engine import ICTStrategyEngine as EngineClass
    elif args.engine == 'smc':
        from strategy.smc_engine import SMCStrategyEngine as EngineClass
    else:
        EngineClass = StrategyEngine

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    async def main():
        bt = BacktesterV2(config)
        engine = EngineClass(config)
        mm = MoneyManager(config)
        sm = StopManager(config)

        results = await bt.run(engine, mm, sm, args.symbol,
                               start_date, end_date, args.balance)

        if results.get('total_trades', 0) > 0:
            out_dir = PROJECT_ROOT / 'data'
            out_dir.mkdir(exist_ok=True)
            bt.export_trades(str(
                out_dir / f"bt2_{args.symbol}_{args.start}_{args.end}.csv"
            ))

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())