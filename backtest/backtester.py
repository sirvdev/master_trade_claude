"""
Backtesting engine — MT5 live data edition.

Strategy:
  1. Bulk-fetch all data for the backtest period + lookback in 3 calls (one per TF).
  2. Walk forward bar-by-bar on the ENTRY timeframe.
  3. At each step, slice exactly `lookback_bars` from each TF (all time-aligned).
  4. Check SL/TP using 1m granularity so we know which hit first.
  5. Show a live progress bar.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Maximum bars per MT5 request (EA's MaxBarsPerRequest)
MAX_BARS_PER_REQUEST = 10000

# ── Timeframe helpers ─────────────────────────────────────────────────────────

TF_SECONDS: Dict[str, int] = {
    '1m':  60,
    '3m':  180,
    '5m':  300,
    '15m': 900,
    '30m': 1800,
    '1h':  3600,
    '4h':  14400,
    '1d':  86400,
}


def tf_to_seconds(tf: str) -> int:
    s = TF_SECONDS.get(tf.lower())
    if s is None:
        raise ValueError(f"Unknown timeframe: {tf!r}. Valid: {list(TF_SECONDS)}")
    return s


# ── Trade dataclass ────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    trade_id:      str
    symbol:        str
    direction:     str          # 'long' | 'short'
    entry_time:    datetime
    entry_price:   float
    stop_loss:     float
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    position_size: float
    exit_time:     Optional[datetime] = None
    exit_price:    Optional[float]    = None
    exit_reason:   Optional[str]      = None   # 'stop_loss' | 'take_profit' | 'end_of_backtest'
    pnl:           float = 0.0
    realized_rr:   float = 0.0
    max_favorable: float = 0.0
    max_adverse:   float = 0.0


# ── Progress bar ───────────────────────────────────────────────────────────────

class ProgressBar:
    """Simple terminal progress bar."""

    def __init__(self, total: int, width: int = 50):
        self.total   = total
        self.width   = width
        self.current = 0

    def update(self, current: int, extra: str = ''):
        self.current = current
        pct  = current / max(self.total, 1)
        done = int(self.width * pct)
        bar  = '#' * done + '-' * (self.width - done)
        line = f"\r  [{bar}] {pct:5.1%}  {current}/{self.total}  {extra}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self):
        self.update(self.total)
        print()


# ── Backtester ─────────────────────────────────────────────────────────────────

class Backtester:
    """
    MT5-powered backtester.

    Usage
    -----
    results = await backtester.run_from_mt5(
        strategy_engine  = ...,
        money_manager    = ...,
        stop_manager     = ...,
        symbol           = 'XAUUSD',
        start_date       = datetime(2024, 1, 1),
        end_date         = datetime(2024, 1, 31),
        initial_balance  = 10_000,
    )
    """

    def __init__(self, config: Dict):
        bt_cfg  = config.get('backtest', {})
        sim_cfg = bt_cfg.get('simulation', {})
        tf_cfg  = bt_cfg.get('timeframes', {})

        self.lookback_bars  = bt_cfg.get('lookback_bars', 250)
        self.entry_tf       = tf_cfg.get('entry_timeframe', '15m')
        self.primary_tf     = tf_cfg.get('primary_timeframe', self.entry_tf)
        self.min_tf         = tf_cfg.get('minimum_timeframe', '1m')
        self.slippage_pct   = sim_cfg.get('slippage_percent', 0.05)
        self.commission_pct = sim_cfg.get('commission_percent', 0.1)
        self.latency_bars   = sim_cfg.get('latency_bars', 1)
        self.max_concurrent = (
            config.get('risk_management', {})
                  .get('global_limits', {})
                  .get('max_concurrent_trades', 3)
        )

        self.timeframe_config = config.get('timeframes', {})
        self._symbols_config  = config.get('symbols', {})

        # Runtime state (reset each run)
        self.trades:       List[BacktestTrade]      = []
        self.open_trades:  Dict[str, BacktestTrade] = {}
        self.equity_curve: List[float]              = []
        self.balance:      float                    = 0.0
        self._platform:    str                      = 'mt5'   # resolved per symbol in run_from_mt5

    # ── Public entry point ─────────────────────────────────────────────────────

    async def run_from_mt5(
        self,
        strategy_engine,
        money_manager,
        stop_manager,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 100_000,
        ):
        """
        Full backtest driven by MT5 historical data.
        
        FIXED: Each TF gets its own lookback window instead of one shared
        window based on the highest TF.
        """
        print()
        print("=" * 70)
        print(f"  BACKTEST  |  {symbol}  |  "
            f"{start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
        print(f"  Lookback: {self.lookback_bars} bars  |  "
            f"Walk TF: {self.primary_tf}  |  Entry TF: {self.entry_tf}  |  "
            f"Min TF: {self.min_tf}")
        print("=" * 70)
    
        # Resolve platform and symbol config
        sym_key = next(
            (k for k in self._symbols_config
            if k.replace('/', '') == symbol or k == symbol),
            None
        )
        sym_cfg = self._symbols_config.get(sym_key, {}) if sym_key else {}
        self._platform = sym_cfg.get('platform', 'mt5')
    
        # Build TF list from symbol config
        sym_tfs_norm = sym_cfg.get('timeframes', [])
        # tf_normalise = {'1H': '1h', '4H': '4h', '1D': '1d',
        #                 '15m': '15m', '5m': '5m', '1m': '1m', '30m': '30m'}
        # sym_tfs_norm = [tf_normalise.get(t, t.lower()) for t in sym_tfs]
    
        # Always include entry and min TFs
        all_tf = list(dict.fromkeys(
            sym_tfs_norm + [self.entry_tf, self.min_tf]
        ))
    
        # ── Separate analysis TFs from the 1m SL/TP checking TF ──────────
        # Analysis TFs need lookback BEFORE the test period (for indicators)
        # The 1m TF only needs the test period itself (for SL/TP granularity)
        analysis_tfs = [tf for tf in all_tf if tf != self.min_tf]
        
        print(f"\n  Fetching data from MT5...")
        print(f"  Analysis TFs: {analysis_tfs} (each with {self.lookback_bars}-bar lookback)")
        print(f"  Min TF: {self.min_tf} (test period only, for SL/TP checks)")
        print(f"  Test period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}\n")
    
        # Fetch with per-TF lookback windows
        raw_data = await self._fetch_all_timeframes_v2(
            symbol, analysis_tfs, self.min_tf, start_date, end_date
        )
    
        # Validate
        # self._validate_data_v2(raw_data, all_tf, start_date, end_date)
    
        # Run simulation
        return await self._simulate(
            strategy_engine, money_manager, stop_manager,
            raw_data, symbol, start_date, end_date, initial_balance,
        )

    # ── Data fetching ──────────────────────────────────────────────────────────

    async def _fetch_all_timeframes_v2(
        self,
        symbol: str,
        analysis_tfs: List[str],
        min_tf: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch each TF with its OWN lookback window.
        
        Analysis TFs: fetch from (start_date - lookback_bars × tf_seconds) to end_date
        Min TF (1m):  fetch from start_date to end_date only (no lookback needed)
        
        If any single fetch would exceed MAX_BARS_PER_REQUEST, split into chunks.
        """
        from execution.mt5_file_bridge import MT5FileBridge
    
        bridge = MT5FileBridge(config={}, demo_mode=False)
        await bridge.connect()
    
        raw: Dict[str, pd.DataFrame] = {}
    
        # ── Fetch analysis TFs (4H, 1H, 15m, 5m, etc.) ──────────────────
        for tf in analysis_tfs:
            tf_secs = tf_to_seconds(tf)
            
            # Each TF gets its own lookback: lookback_bars × tf_seconds before start
            lookback_secs = self.lookback_bars * tf_secs
            tf_fetch_from = start_date - timedelta(seconds=lookback_secs)
            
            # Estimate total bars needed
            total_period_secs = (end_date - tf_fetch_from).total_seconds()
            estimated_bars = int(total_period_secs / tf_secs)
            
            print(f"  Fetching {tf:>4s}: "
                f"from {tf_fetch_from.strftime('%Y-%m-%d')} "
                f"(~{estimated_bars:,} bars)...", end='', flush=True)
            
            try:
                if estimated_bars <= MAX_BARS_PER_REQUEST:
                    # Single fetch is enough
                    df = await bridge.fetch_historical_range(
                        symbol=symbol, timeframe=tf,
                        from_dt=tf_fetch_from, to_dt=end_date,
                    )
                else:
                    # Need chunked fetch
                    df = await self._chunked_fetch(
                        bridge, symbol, tf, tf_fetch_from, end_date, tf_secs
                    )
                
                raw[tf] = df
                
                if len(df) > 0:
                    print(f" {len(df):>6,} bars "
                        f"({df.index[0].strftime('%Y-%m-%d')} → "
                        f"{df.index[-1].strftime('%Y-%m-%d')}) OK")
                else:
                    print(f" 0 bars — WARNING: no data returned")
                    
            except Exception as e:
                print(f" FAILED: {e}")
                raise
    
        # ── Fetch min TF (1m) — test period ONLY ─────────────────────────
        # 1m is only used for granular SL/TP checking within each bar
        # No lookback needed — indicators are computed on analysis TFs
        min_tf_secs = tf_to_seconds(min_tf)
        test_period_secs = (end_date - start_date).total_seconds()
        estimated_1m_bars = int(test_period_secs / min_tf_secs)
        
        print(f"  Fetching {min_tf:>4s}: "
            f"from {start_date.strftime('%Y-%m-%d')} "
            f"(test period only, ~{estimated_1m_bars:,} bars)...", end='', flush=True)
        
        try:
            if estimated_1m_bars <= MAX_BARS_PER_REQUEST:
                df = await bridge.fetch_historical_range(
                    symbol=symbol, timeframe=min_tf,
                    from_dt=start_date, to_dt=end_date,
                )
            else:
                df = await self._chunked_fetch(
                    bridge, symbol, min_tf, start_date, end_date, min_tf_secs
                )
            
            raw[min_tf] = df
            
            if len(df) > 0:
                print(f" {len(df):>6,} bars "
                    f"({df.index[0].strftime('%Y-%m-%d')} → "
                    f"{df.index[-1].strftime('%Y-%m-%d')}) OK")
            else:
                print(f" 0 bars — WARNING")
                
        except Exception as e:
            print(f" FAILED: {e}")
            raise
    
        await bridge.disconnect()
        return raw

    async def _chunked_fetch(
        self,
        bridge,
        symbol: str,
        tf: str,
        from_date: datetime,
        to_date: datetime,
        tf_secs: int,
    ) -> pd.DataFrame:
        """
        Fetch data in chunks of MAX_BARS_PER_REQUEST bars, then concatenate.
        Needed when the total period exceeds 5,000 bars at this timeframe.
        """
        chunk_duration = timedelta(seconds=MAX_BARS_PER_REQUEST * tf_secs * 0.9)
        # 0.9 factor leaves room for overlap to avoid gaps
        
        all_chunks = []
        chunk_start = from_date
        chunk_num = 0
        
        while chunk_start < to_date:
            chunk_end = min(chunk_start + chunk_duration, to_date)
            chunk_num += 1
            
            try:
                df = await bridge.fetch_historical_range(
                    symbol=symbol, timeframe=tf,
                    from_dt=chunk_start, to_dt=chunk_end,
                )
                if len(df) > 0:
                    all_chunks.append(df)
            except Exception as e:
                logger.warning(f"Chunk {chunk_num} fetch failed for {tf}: {e}")
            
            # Move start to end of this chunk (slight overlap to avoid gaps)
            chunk_start = chunk_end - timedelta(seconds=tf_secs * 10)
        
        if not all_chunks:
            return pd.DataFrame()
        
        # Concatenate and remove duplicates
        combined = pd.concat(all_chunks)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        
        return combined

    # ── Validation ─────────────────────────────────────────────────────────────

    
    def _validate_data_v2(
        self,
        raw_data: Dict[str, pd.DataFrame],
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
    ):
        """
        Validate that each TF has data covering the test period.
        Much better than the old validation which just checked bar count.
        """
        print("\n  Data validation:")
        
        start_ts = pd.Timestamp(start_date, tz='UTC')
        end_ts = pd.Timestamp(end_date, tz='UTC')
        
        all_ok = True
        
        for tf in timeframes:
            df = raw_data.get(tf)
            if df is None or df.empty:
                print(f"    {tf:>4s}: ❌ NO DATA")
                all_ok = False
                continue
            
            first = df.index[0]
            last = df.index[-1]
            
            # Check: does the data reach into the test period?
            if last < start_ts:
                print(f"    {tf:>4s}: ❌ Data ends at {last.strftime('%Y-%m-%d')} "
                    f"— BEFORE test period starts ({start_date.strftime('%Y-%m-%d')})")
                all_ok = False
                continue
            
            # Check: for analysis TFs, do we have lookback before test start?
            if tf != self.min_tf:
                bars_before_test = len(df[df.index < start_ts])
                min_needed = min(50, self.lookback_bars // 2)
                if bars_before_test < min_needed:
                    print(f"    {tf:>4s}: ⚠️  Only {bars_before_test} lookback bars "
                        f"(need {min_needed}+)")
                else:
                    # Count bars in test period
                    test_bars = len(df[(df.index >= start_ts) & (df.index <= end_ts)])
                    print(f"    {tf:>4s}: ✅ {bars_before_test} lookback + "
                        f"{test_bars} test bars = {len(df)} total")
            else:
                # For min_tf (1m), check coverage of test period
                test_bars = len(df[(df.index >= start_ts) & (df.index <= end_ts)])
                tf_secs = tf_to_seconds(tf)
                expected_bars = int((end_date - start_date).total_seconds() / tf_secs)
                coverage = test_bars / max(expected_bars, 1) * 100
                
                if coverage < 50:
                    print(f"    {tf:>4s}: ❌ Only {test_bars} test bars "
                        f"({coverage:.0f}% coverage, expected ~{expected_bars})")
                    all_ok = False
                else:
                    print(f"    {tf:>4s}: ✅ {test_bars} test bars "
                        f"({coverage:.0f}% of expected {expected_bars})")
        
        print()
        
        if not all_ok:
            raise ValueError(
                "Data validation failed — some timeframes don't cover the test period. "
                "See details above."
            )
    
    # ── Core simulation ────────────────────────────────────────────────────────

    async def _simulate(
        self,
        strategy_engine,
        money_manager,
        stop_manager,
        raw_data:        Dict[str, pd.DataFrame],
        symbol:          str,
        start_date:      datetime,
        end_date:        datetime,
        initial_balance: float,
    ) -> Dict:

        # Reset state
        self.balance      = initial_balance
        self.equity_curve = [initial_balance]
        self.trades       = []
        self.open_trades  = {}

        walk_tf = self.primary_tf
        if walk_tf not in raw_data:
            walk_tf = self.entry_tf
        entry_df = raw_data[walk_tf]
        min_df   = raw_data[self.min_tf]

        # Bars within the actual test period only
        start_ts = pd.Timestamp(start_date, tz='UTC') if start_date.tzinfo is None \
                   else pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date, tz='UTC') if end_date.tzinfo is None \
                   else pd.Timestamp(end_date)

        test_bars = entry_df[
            (entry_df.index >= start_ts) &
            (entry_df.index <= end_ts)
        ]

        if test_bars.empty:
            raise ValueError(
                "No entry-TF bars fall within the test period. "
                "Check that start_date / end_date match the fetched data."
            )

        total_iters   = len(test_bars)
        entry_tf_secs = tf_to_seconds(walk_tf)
        min_tf_secs   = tf_to_seconds(self.min_tf)
        bars_per_iter = entry_tf_secs // min_tf_secs   # e.g. 15m/1m = 15

        print(f"  Simulating {total_iters:,} iterations "
              f"({self.entry_tf} bars in test period)...\n")

        progress      = ProgressBar(total_iters)
        trade_counter = 0
        signal_counter = 0  # Track how many entry signals generated

        for idx, (current_ts, entry_bar) in enumerate(test_bars.iterrows()):

            current_price = entry_bar['close']

            # ── Build time-aligned lookback windows ────────────────────────────
            tf_snapshot: Dict[str, pd.DataFrame] = {}
            for tf, df in raw_data.items():
                available = df[df.index <= current_ts]
                # Take up to lookback_bars, but use whatever we have if less
                bars_to_take = min(len(available), self.lookback_bars)
                if bars_to_take > 0:
                    tf_snapshot[tf] = available.iloc[-bars_to_take:]
                else:
                    # Edge case: no data yet at this timestamp (shouldn't happen)
                    continue

            # ── Granular SL/TP check using minimum TF bars ─────────────────────
            # Get the 1m bars that belong to THIS entry_tf bar's period
            period_start = current_ts - timedelta(seconds=entry_tf_secs - min_tf_secs)
            min_period   = min_df[
                (min_df.index > period_start) &
                (min_df.index <= current_ts)
            ]
            self._check_exits_granular(min_period)

            # ── Strategy analysis ──────────────────────────────────────────────
            if len(self.open_trades) < self.max_concurrent:
                try:
                    analysis = strategy_engine.analyze_market(symbol, tf_snapshot)

                    if analysis.get('entry_signal'):
                        signal_counter += 1
                        logger.info(
                            f"Entry signal: {analysis['direction']} {symbol} "
                            f"(reason: {analysis.get('entry_reason')}, "
                            f"confidence: {analysis.get('confidence_score'):.2f})"
                        )
                        
                        levels = strategy_engine.calculate_entry_levels(
                            analysis, tf_snapshot
                        )
                        
                        logger.debug(
                            f"Entry levels: price={levels['entry_price']:.2f}, "
                            f"SL={levels['stop_loss']:.2f}, "
                            f"TP1={levels.get('take_profit_1')}"
                        )
                        
                        sizing = money_manager.calculate_position_size(
                            account_equity = self.balance,
                            entry_price    = levels['entry_price'],
                            stop_loss      = levels['stop_loss'],
                            symbol         = symbol,
                            direction      = analysis['direction'],
                            platform       = self._platform,
                        )
                        
                        logger.info(
                            f"Entry signal: {analysis['direction'].upper()} @ "
                            f"{levels['entry_price']:.2f}, SL={levels['stop_loss']:.2f}, "
                            f"size={sizing.get('position_size')}, "
                            f"risk={sizing.get('risk_percent', 0):.2f}%"
                        )

                        if sizing.get('approved') and sizing.get('position_size', 0) > 0:
                            trade_counter += 1
                            trade = self._open_trade(
                                trade_id      = f"bt_{trade_counter}",
                                symbol        = symbol,
                                direction     = analysis['direction'],
                                entry_time    = current_ts.to_pydatetime(),
                                target_price  = levels['entry_price'],
                                stop_loss     = levels['stop_loss'],
                                take_profit_1 = levels.get('take_profit_1'),
                                take_profit_2 = levels.get('take_profit_2'),
                                position_size = sizing['position_size'],
                                entry_bar     = entry_bar,
                            )
                            self.open_trades[trade.trade_id] = trade

                except Exception as e:
                    logger.debug(f"Strategy error at {current_ts}: {e}")

            # ── Equity curve ───────────────────────────────────────────────────
            unrealised = sum(
                self._unrealised_pnl(t, current_price)
                for t in self.open_trades.values()
            )
            self.equity_curve.append(self.balance + unrealised)

            # ── Progress ───────────────────────────────────────────────────────
            progress.update(
                idx + 1,
                f"Balance:${self.balance:,.0f} Open:{len(self.open_trades)} "
                f"Trades:{len(self.trades)}"
            )

        progress.finish()

        # Log signal vs trade summary
        logger.info(
            f"\n=== Signal Summary ===\n"
            f"Entry signals generated: {signal_counter}\n"
            f"Trades executed:         {len(self.trades)}\n"
            f"Signal-to-trade ratio:   {len(self.trades)/signal_counter if signal_counter > 0 else 0:.1%}"
        )

        # Close remaining open trades at final price
        if self.open_trades:
            final_price = float(test_bars.iloc[-1]['close'])
            final_time  = test_bars.index[-1].to_pydatetime()
            for trade in list(self.open_trades.values()):
                self._close_trade(trade, final_time, final_price, 'end_of_backtest')

        return self._build_results(symbol, initial_balance)

    # ── Trade lifecycle ────────────────────────────────────────────────────────

    def _open_trade(
        self,
        trade_id:      str,
        symbol:        str,
        direction:     str,
        entry_time:    datetime,
        target_price:  float,
        stop_loss:     float,
        take_profit_1: Optional[float],
        take_profit_2: Optional[float],
        position_size: float,
        entry_bar:     pd.Series,
    ) -> BacktestTrade:
        slip = abs(np.random.normal(0, self.slippage_pct / 100))
        fill = target_price * (1 + slip) if direction == 'long' \
               else target_price * (1 - slip)
        fill = max(float(entry_bar['low']), min(float(entry_bar['high']), fill))

        # Entry commission
        self.balance -= fill * position_size * (self.commission_pct / 100)

        trade = BacktestTrade(
            trade_id      = trade_id,
            symbol        = symbol,
            direction     = direction,
            entry_time    = entry_time,
            entry_price   = fill,
            stop_loss     = stop_loss,
            take_profit_1 = take_profit_1,
            take_profit_2 = take_profit_2,
            position_size = position_size,
        )
        logger.info(
            f"  OPEN  {direction.upper()} {symbol} @ {fill:.2f}  "
            f"SL:{stop_loss:.2f}  TP:{take_profit_1}  Size:{position_size:.4f}"
        )
        return trade

    def _close_trade(
        self,
        trade:      BacktestTrade,
        exit_time:  datetime,
        exit_price: float,
        reason:     str,
    ):
        trade.exit_time   = exit_time
        trade.exit_price  = exit_price
        trade.exit_reason = reason

        if trade.direction == 'long':
            gross = (exit_price - trade.entry_price) * trade.position_size
        else:
            gross = (trade.entry_price - exit_price) * trade.position_size

        commission = exit_price * trade.position_size * (self.commission_pct / 100)
        trade.pnl  = gross - commission

        risk = abs(trade.entry_price - trade.stop_loss) * trade.position_size
        trade.realized_rr = trade.pnl / risk if risk > 0 else 0.0

        self.balance += trade.pnl
        self.trades.append(trade)
        self.open_trades.pop(trade.trade_id, None)

        logger.info(
            f"  CLOSE {trade.symbol} @ {exit_price:.2f} ({reason})  "
            f"P&L:${trade.pnl:+.2f}  R:R:{trade.realized_rr:.2f}"
        )

    # ── Granular exit check ────────────────────────────────────────────────────

    def _check_exits_granular(self, min_period: pd.DataFrame):
        """
        Walk each 1m bar in this entry-TF period.
        SL always takes priority over TP if both would trigger on same bar.
        """
        if min_period.empty:
            return

        for ts, bar in min_period.iterrows():
            for trade in list(self.open_trades.values()):
                exit_price, reason = self._bar_exit(trade, bar)
                if exit_price is not None:
                    self._close_trade(trade, ts.to_pydatetime(), exit_price, reason)

    def _bar_exit(
        self,
        trade: BacktestTrade,
        bar:   pd.Series,
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Returns (exit_price, reason) or (None, None).

        SL checked BEFORE TP — if both trigger in same 1m bar, SL wins.
        For longs:  SL = low <= stop_loss   TP = high >= take_profit
        For shorts: SL = high >= stop_loss  TP = low  <= take_profit
        """
        low  = float(bar['low'])
        high = float(bar['high'])
        open_ = float(bar['open'])

        if trade.direction == 'long':
            if low <= trade.stop_loss:
                # Gap protection: if bar opened below SL, exit at open
                return min(trade.stop_loss, open_), 'stop_loss'
            if trade.take_profit_1 and high >= trade.take_profit_1:
                return trade.take_profit_1, 'take_profit'

        else:  # short
            if high >= trade.stop_loss:
                return max(trade.stop_loss, open_), 'stop_loss'
            if trade.take_profit_1 and low <= trade.take_profit_1:
                return trade.take_profit_1, 'take_profit'

        # Track MFE / MAE
        if trade.direction == 'long':
            trade.max_favorable = max(trade.max_favorable, high - trade.entry_price)
            trade.max_adverse   = min(trade.max_adverse,   low  - trade.entry_price)
        else:
            trade.max_favorable = max(trade.max_favorable, trade.entry_price - low)
            trade.max_adverse   = min(trade.max_adverse,   trade.entry_price - high)

        return None, None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _unrealised_pnl(self, trade: BacktestTrade, price: float) -> float:
        if trade.direction == 'long':
            return (price - trade.entry_price) * trade.position_size
        return (trade.entry_price - price) * trade.position_size

    # ── Results ────────────────────────────────────────────────────────────────

    def _build_results(self, symbol: str, initial_balance: float) -> Dict:
        trades = self.trades
        n      = len(trades)

        if n == 0:
            return {
                'symbol': symbol, 'total_trades': 0, 'winning_trades': 0,
                'losing_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'avg_rr': 0.0,
                'profit_factor': 0.0, 'expectancy': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'max_consecutive_wins': 0,
                'max_consecutive_losses': 0, 'avg_duration_hours': 0.0,
                'final_balance': self.balance,
                'return_percent': (self.balance / initial_balance - 1) * 100,
                'equity_curve': self.equity_curve, 'trades': [],
            }

        pnls   = [t.pnl for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        rrs    = [t.realized_rr for t in trades]

        # Max drawdown
        peak = initial_balance
        max_dd = 0.0
        running = initial_balance
        for p in pnls:
            running += p
            peak     = max(peak, running)
            dd       = (running - peak) / peak * 100
            max_dd   = min(max_dd, dd)

        # Consecutive streaks
        max_cw = max_cl = cw = cl = 0
        for p in pnls:
            if p > 0:
                cw += 1; cl = 0; max_cw = max(max_cw, cw)
            else:
                cl += 1; cw = 0; max_cl = max(max_cl, cl)

        # Sharpe ratio
        equity = pd.Series(self.equity_curve)
        rets   = equity.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0

        # Average duration
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in trades if t.exit_time
        ]

        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))

        trade_dicts = [
            {
                'trade_id':       t.trade_id,
                'symbol':         t.symbol,
                'direction':      t.direction,
                'entry_time':     t.entry_time,
                'entry_price':    t.entry_price,
                'exit_time':      t.exit_time,
                'exit_price':     t.exit_price,
                'stop_loss':      t.stop_loss,
                'take_profit_1':  t.take_profit_1,
                'pnl':            t.pnl,
                'realized_rr':    t.realized_rr,
                'exit_reason':    t.exit_reason,
                'max_favorable':  t.max_favorable,
                'max_adverse':    t.max_adverse,
                'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600
                                   if t.exit_time else 0,
            }
            for t in trades
        ]

        results = {
            'symbol':                 symbol,
            'total_trades':           n,
            'winning_trades':         len(wins),
            'losing_trades':          len(losses),
            'win_rate':               len(wins) / n,
            'total_pnl':              sum(pnls),
            'avg_win':                sum(wins) / len(wins) if wins else 0,
            'avg_loss':               sum(losses) / len(losses) if losses else 0,
            'avg_rr':                 sum(rrs) / n,
            'profit_factor':          gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'expectancy':             sum(pnls) / n,
            'sharpe_ratio':           round(sharpe, 2),
            'max_drawdown':           round(max_dd, 2),
            'max_consecutive_wins':   max_cw,
            'max_consecutive_losses': max_cl,
            'avg_duration_hours':     sum(durations) / len(durations) if durations else 0,
            'final_balance':          self.balance,
            'return_percent':         (self.balance / initial_balance - 1) * 100,
            'equity_curve':           self.equity_curve,
            'trades':                 trade_dicts,
        }

        self._print_results(results)
        return results

    def _print_results(self, r: Dict):
        print()
        print("=" * 70)
        print("  RESULTS")
        print("=" * 70)
        print(f"  Total Trades        : {r['total_trades']}")
        print(f"  Win Rate            : {r['win_rate']:.2%}  "
              f"({r['winning_trades']}W / {r['losing_trades']}L)")
        print(f"  Avg R:R             : {r['avg_rr']:.2f}")
        print(f"  Profit Factor       : {r['profit_factor']:.2f}")
        print(f"  Expectancy          : ${r['expectancy']:+.2f} / trade")
        print()
        print(f"  Total P&L           : ${r['total_pnl']:+,.2f}")
        print(f"  Final Balance       : ${r['final_balance']:,.2f}")
        print(f"  Return              : {r['return_percent']:+.2f}%")
        print()
        print(f"  Max Drawdown        : {r['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio        : {r['sharpe_ratio']:.2f}")
        print(f"  Max Consec. Wins    : {r['max_consecutive_wins']}")
        print(f"  Max Consec. Losses  : {r['max_consecutive_losses']}")
        print(f"  Avg Trade Duration  : {r['avg_duration_hours']:.1f} hours")
        print("=" * 70)

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_trades(self, filepath: str):
        """Export all closed trades to CSV."""
        import csv, os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        if not self.trades:
            logger.warning("No trades to export")
            return

        fields = [
            'trade_id', 'symbol', 'direction',
            'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'exit_reason',
            'stop_loss', 'take_profit_1',
            'pnl', 'realized_rr',
            'max_favorable', 'max_adverse', 'duration_hours',
        ]
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for t in self.trades:
                writer.writerow({
                    'trade_id':       t.trade_id,
                    'symbol':         t.symbol,
                    'direction':      t.direction,
                    'entry_time':     t.entry_time,
                    'entry_price':    t.entry_price,
                    'exit_time':      t.exit_time,
                    'exit_price':     t.exit_price,
                    'exit_reason':    t.exit_reason,
                    'stop_loss':      t.stop_loss,
                    'take_profit_1':  t.take_profit_1,
                    'pnl':            t.pnl,
                    'realized_rr':    t.realized_rr,
                    'max_favorable':  t.max_favorable,
                    'max_adverse':    t.max_adverse,
                    'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600
                                       if t.exit_time else 0,
                })
        logger.info(f"Exported {len(self.trades)} trades to {filepath}")
        print(f"\n  Trades exported to {filepath}")


# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import argparse
    from pathlib import Path
    import yaml

    # Ensure project root is importable
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from strategy.engine import StrategyEngine
    from risk_management.money_manager import MoneyManager
    from risk_management.stop_manager import StopManager

    # ── CLI arguments ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Run backtest from config/config.yaml')
    parser.add_argument('--symbol',  default=None,
                        help='Symbol to backtest (e.g. XAUUSD). '
                             'Defaults to first enabled MT5 symbol in config.')
    parser.add_argument('--start',   default=None,
                        help='Start date YYYY-MM-DD. Defaults to backtest.date_range.start in config.')
    parser.add_argument('--end',     default=None,
                        help='End date   YYYY-MM-DD. Defaults to backtest.date_range.end in config.')
    parser.add_argument('--balance', type=float, default=None,
                        help='Initial balance. Defaults to risk_management.initial_balance in config.')
    parser.add_argument('--config',  default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
                        help='Path to config file (default: config/config.yaml)')
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # ── Resolve symbol ─────────────────────────────────────────────────────────
    if args.symbol:
        symbol = args.symbol.replace('/', '')   # normalise XAUUSD / XAU/USD
    else:
        # Pick first enabled MT5 symbol from config
        symbol = None
        for sym, sym_cfg in config.get('symbols', {}).items():
            if sym_cfg.get('enabled') and sym_cfg.get('platform') == 'mt5':
                symbol = sym.replace('/', '')
                break
        if not symbol:
            print("ERROR: No enabled MT5 symbol found in config. "
                  "Use --symbol XAUUSD or enable a symbol in config.yaml")
            sys.exit(1)

    # ── Resolve dates ──────────────────────────────────────────────────────────
    bt_cfg     = config.get('backtest', {})
    date_range = bt_cfg.get('date_range', {})

    start_str = args.start or str(date_range.get('start', '2024-01-01'))
    end_str   = args.end   or str(date_range.get('end',   '2024-01-31'))

    start_date = datetime.strptime(start_str, '%Y-%m-%d')
    end_date   = datetime.strptime(end_str,   '%Y-%m-%d')

    # ── Resolve initial balance ────────────────────────────────────────────────
    initial_balance = (
        args.balance
        or config.get('risk_management', {}).get('initial_balance', 10_000)
    )

    # ── Inject backtest timeframe config if not already present ────────────────
    # The config uses 'timeframes' at top level; backtest needs entry + min TF.
    # Derive from symbol config if backtest.timeframes block is absent.
    if 'timeframes' not in bt_cfg:
        # Find the symbol's configured timeframes
        sym_key   = next((k for k in config.get('symbols', {}) if k.replace('/', '') == symbol), None)
        sym_cfg   = config['symbols'].get(sym_key, {}) if sym_key else {}
        entry_tf  = sym_cfg.get('entry_timeframe',
                                config.get('timeframes', {}).get('entry_timeframe', '5m'))
        # Minimum TF = lowest in the symbol's timeframe list, defaulting to 1m
        tf_list   = sym_cfg.get('timeframes', ['1H', '15m', '5m'])
        tf_order  = ['1m', '5m', '15m', '30m', '1H', '4H', '1D']
        min_tf    = next((t for t in tf_order if t in tf_list), '1m')

        config.setdefault('backtest', {})['timeframes'] = {
            'entry_timeframe':   entry_tf,
            'minimum_timeframe': min_tf,
        }

    # ── Run ────────────────────────────────────────────────────────────────────
    async def main():
        backtester      = Backtester(config)
        strategy_engine = StrategyEngine(config)
        money_manager   = MoneyManager(config)
        stop_manager    = StopManager(config)

        print(f"\n  Config : {config_path}")
        print(f"  Symbol : {symbol}")
        print(f"  Period : {start_date.date()} -> {end_date.date()}")
        print(f"  Balance: ${initial_balance:,.0f}")

        results = await backtester.run_from_mt5(
            strategy_engine = strategy_engine,
            money_manager   = money_manager,
            stop_manager    = stop_manager,
            symbol          = symbol,
            start_date      = start_date,
            end_date        = end_date,
            initial_balance = initial_balance,
        )

        if results['total_trades'] > 0:
            output_dir = PROJECT_ROOT / 'data'
            output_dir.mkdir(exist_ok=True)
            backtester.export_trades(str(output_dir / f'backtest_{symbol}_{start_str}_{end_str}.csv'))

    asyncio.run(main())