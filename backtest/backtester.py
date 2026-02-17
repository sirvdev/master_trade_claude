"""
Backtesting engine for strategy validation.
Implements walking-forward simulation with:
- Uniform lookback bars across all timeframes
- Iteration by entry timeframe
- 1m granularity for SL/TP execution
- Time-aligned multi-timeframe windows
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Data class for backtest trade."""
    trade_id: str
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    position_size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0
    realized_rr: float = 0
    max_favorable: float = 0
    max_adverse: float = 0
    

class Backtester:
    """
    Backtesting engine with realistic simulation.
    Loads real market data from CSV and resamples to configured timeframes.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtester.
        
        Args:
            config: Backtest configuration
        """
        self.config = config.get('backtest', {})
        self.simulation_config = self.config.get('simulation', {})
        
        # Simulation parameters
        self.slippage_percent = self.simulation_config.get('slippage_percent', 0.05)
        self.commission_percent = self.simulation_config.get('commission_percent', 0.1)
        self.latency_bars = self.simulation_config.get('latency_bars', 1)
        
        # Lookback configuration (uniform across all timeframes)
        self.lookback_bars = self.config.get('lookback_bars', 500)
        
        # State
        self.trades: List[BacktestTrade] = []
        self.open_trades: Dict[str, BacktestTrade] = {}
        self.equity_curve = []
        self.balance = 10000  # Starting balance
        
    def load_data_from_csv(
        self,
        csv_path: str,
        base_timeframe: str = '1m'
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Expected CSV format:
        timestamp,open,high,low,close,volume
        2024-01-01 00:00:00,2000.50,2001.00,2000.00,2000.75,1234
        
        Args:
            csv_path: Path to CSV file
            base_timeframe: Base timeframe of the data (1m, 5m, 15m)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading data from {csv_path}")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Parse timestamp column (try multiple formats)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
                df.drop('time', axis=1, inplace=True)
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df.drop('date', axis=1, inplace=True)
            else:
                # Assume first column is timestamp
                df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
                df = df.iloc[:, 1:]
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Ensure we have OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert to numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any NaN rows
            df.dropna(inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            logger.info(
                f"Loaded {len(df)} bars of {base_timeframe} data "
                f"from {df.index[0]} to {df.index[-1]}"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}", exc_info=True)
            raise
    
    def resample_to_timeframe(
        self,
        base_df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample base data to target timeframe.
        
        Args:
            base_df: Base OHLCV DataFrame
            target_timeframe: Target timeframe (e.g., '5m', '15m', '1h', '4h')
            
        Returns:
            Resampled DataFrame
        """
        # Map timeframe strings to pandas offset aliases
        tf_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D'
        }
        
        if target_timeframe.lower() not in tf_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        resample_rule = tf_map[target_timeframe.lower()]
        
        logger.info(f"Resampling to {target_timeframe}")
        
        # Resample using proper aggregation
        resampled = base_df.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Resampled to {len(resampled)} {target_timeframe} bars")
        
        return resampled
    
    def prepare_multi_timeframe_data(
        self,
        csv_path: str,
        timeframes: List[str],
        base_timeframe: str = '1m'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load base data and create all required timeframes.
        
        Args:
            csv_path: Path to CSV file with base timeframe data
            timeframes: List of timeframes to create (e.g., ['1h', '15m', '1m'])
            base_timeframe: Timeframe of the CSV data
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        logger.info("=" * 60)
        logger.info("Preparing Multi-Timeframe Data")
        logger.info("=" * 60)
        
        # Load base data
        base_df = self.load_data_from_csv(csv_path, base_timeframe)
        
        # Verify we have enough data for lookback
        # Calculate minimum base bars needed
        tf_multipliers = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        # Find highest timeframe to determine minimum data needed
        max_multiplier = max(tf_multipliers.get(tf.lower(), 1) for tf in timeframes)
        min_base_bars_needed = self.lookback_bars * max_multiplier
        
        if len(base_df) < min_base_bars_needed:
            raise ValueError(
                f"Insufficient base data: have {len(base_df)} {base_timeframe} bars, "
                f"need {min_base_bars_needed} for {self.lookback_bars} bars of highest timeframe"
            )
        
        logger.info(
            f"Base data sufficient: {len(base_df)} bars "
            f"(need {min_base_bars_needed} for {self.lookback_bars}-bar lookback)"
        )
        
        # Create dictionary for all timeframes
        multi_tf_data = {}
        
        for tf in timeframes:
            if tf.lower() == base_timeframe.lower():
                # Use base data directly
                multi_tf_data[tf] = base_df.copy()
                logger.info(f"{tf}: Using base data ({len(base_df)} bars)")
            else:
                # Resample to target timeframe
                multi_tf_data[tf] = self.resample_to_timeframe(base_df, tf)
        
        logger.info("=" * 60)
        logger.info("Multi-Timeframe Data Ready")
        logger.info("=" * 60)
        
        return multi_tf_data
        
    def run(
        self,
        strategy_engine,
        money_manager,
        stop_manager,
        multi_tf_data: Dict[str, pd.DataFrame],
        symbol: str,
        initial_balance: float = 10000
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            strategy_engine: Strategy engine instance
            money_manager: Money manager instance
            stop_manager: Stop manager instance
            multi_tf_data: Dictionary of timeframe to DataFrame
            symbol: Trading symbol
            initial_balance: Starting balance
            
        Returns:
            Backtest results dictionary
        """
        logger.info("=" * 60)
        logger.info(f"Starting Backtest: {symbol}")
        logger.info(f"Initial Balance: ${initial_balance:,.2f}")
        logger.info(f"Lookback Bars: {self.lookback_bars}")
        logger.info("=" * 60)
        
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades = []
        self.open_trades = {}
        
        # Get timeframe configuration
        timeframe_config = self.config.get('timeframes', {})
        entry_tf = timeframe_config.get('entry_timeframe', '15m')
        min_tf = timeframe_config.get('minimum_timeframe', '1m')
        
        # Verify required timeframes exist
        if entry_tf not in multi_tf_data:
            raise ValueError(f"Entry timeframe {entry_tf} not in data")
        if min_tf not in multi_tf_data:
            raise ValueError(f"Minimum timeframe {min_tf} not in data")
        
        # Get iteration data (entry timeframe)
        entry_data = multi_tf_data[entry_tf]
        min_data = multi_tf_data[min_tf]
        
        # Calculate bars ratio (how many min_tf bars per entry_tf bar)
        # E.g., 15m / 1m = 15 bars
        tf_multipliers = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440
        }
        bars_per_iteration = tf_multipliers[entry_tf.lower()] // tf_multipliers[min_tf.lower()]
        
        logger.info(f"Iteration timeframe: {entry_tf}")
        logger.info(f"Minimum timeframe: {min_tf} ({bars_per_iteration} bars per iteration)")
        logger.info(f"Total {entry_tf} bars to process: {len(entry_data)}")
        
        # Start after we have enough data for lookback
        start_idx = self.lookback_bars
        total_iterations = len(entry_data) - start_idx
        
        logger.info(f"Starting at bar {start_idx} (after {self.lookback_bars}-bar lookback)")
        logger.info(f"Total iterations: {total_iterations}\n")
        
        # Iterate through bars
        for i in range(start_idx, len(entry_data)):
            current_entry_bar = entry_data.iloc[i]
            current_time = current_entry_bar.name
            current_price = current_entry_bar['close']
            
            # Calculate corresponding indices in other timeframes
            # Find the index in each timeframe that matches current_time
            tf_indices = {}
            for tf, df in multi_tf_data.items():
                # Find index where timestamp <= current_time
                matching_idx = df.index.get_indexer([current_time], method='ffill')[0]
                if matching_idx >= 0:
                    tf_indices[tf] = matching_idx
            
            # Build multi-timeframe snapshot with exactly lookback_bars for each
            tf_snapshot = {}
            for tf, df in multi_tf_data.items():
                if tf not in tf_indices:
                    continue
                    
                current_idx = tf_indices[tf]
                
                # Get exactly lookback_bars ending at current index
                start = max(0, current_idx - self.lookback_bars + 1)
                end = current_idx + 1
                
                snapshot = df.iloc[start:end]
                
                # Ensure we have exactly lookback_bars (or close to it at start)
                if len(snapshot) >= min(self.lookback_bars // 2, 50):  # At least half or 50 bars
                    tf_snapshot[tf] = snapshot
            
            # Verify all timeframes are time-aligned
            if len(tf_snapshot) > 0:
                timestamps = [df.index[-1] for df in tf_snapshot.values()]
                if not all(t == timestamps[0] for t in timestamps):
                    logger.warning(f"Timeframes not aligned at {current_time}")
                    continue
            
            # Update existing positions using 1m bars for granular exit detection
            min_tf_idx = tf_indices.get(min_tf)
            if min_tf_idx is not None:
                # Get the min_tf bars that make up this entry_tf period
                min_bars_in_period = min_data.iloc[
                    max(0, min_tf_idx - bars_per_iteration + 1):min_tf_idx + 1
                ]
                
                self._update_open_positions_granular(
                    current_time,
                    min_bars_in_period,
                    stop_manager
                )
            
            # Skip if too many open positions
            max_concurrent = 3  # From config
            if len(self.open_trades) >= max_concurrent:
                continue
            
            # Run strategy analysis
            try:
                analysis = strategy_engine.analyze_market(symbol, tf_snapshot)
                
                if analysis['entry_signal']:
                    # Calculate entry levels
                    levels = strategy_engine.calculate_entry_levels(analysis, tf_snapshot)
                    
                    # Calculate position size
                    sizing = money_manager.calculate_position_size(
                        account_equity=self.balance,
                        entry_price=levels['entry_price'],
                        stop_loss=levels['stop_loss'],
                        symbol=symbol,
                        direction=analysis['direction']
                    )
                    
                    if sizing['approved'] and sizing['position_size'] > 0:
                        # Simulate order fill with latency and slippage
                        fill_bar_idx = min(i + self.latency_bars, len(entry_data) - 1)
                        fill_bar = entry_data.iloc[fill_bar_idx]
                        
                        fill_result = self._simulate_fill(
                            fill_bar,
                            levels['entry_price'],
                            analysis['direction']
                        )
                        
                        # Create trade
                        trade = BacktestTrade(
                            trade_id=f"bt_{i}_{symbol}",
                            symbol=symbol,
                            direction=analysis['direction'],
                            entry_time=fill_bar.name,
                            entry_price=fill_result['price'],
                            stop_loss=levels['stop_loss'],
                            take_profit_1=levels.get('take_profit_1'),
                            take_profit_2=levels.get('take_profit_2'),
                            position_size=sizing['position_size']
                        )
                        
                        self.open_trades[trade.trade_id] = trade
                        
                        # Deduct commission
                        commission = sizing['position_value'] * (self.commission_percent / 100)
                        self.balance -= commission
                        
                        logger.info(
                            f"[{trade.entry_time}] Entry: {trade.direction.upper()} {symbol} @ "
                            f"{trade.entry_price:.2f}, SL: {trade.stop_loss:.2f}, "
                            f"Size: {trade.position_size:.4f}"
                        )
                        
            except Exception as e:
                logger.error(f"Error in strategy analysis at bar {i}: {e}")
                continue
            
            # Update equity curve
            unrealized_pnl = sum(
                self._calculate_unrealized_pnl(trade, current_price)
                for trade in self.open_trades.values()
            )
            self.equity_curve.append(self.balance + unrealized_pnl)
            
            # Progress update
            if (i - start_idx) % 100 == 0:
                progress = ((i - start_idx) / total_iterations) * 100
                logger.info(
                    f"Progress: {progress:.1f}% [{i-start_idx}/{total_iterations}] - "
                    f"Balance: ${self.balance:,.2f}, "
                    f"Open: {len(self.open_trades)}, "
                    f"Closed: {len(self.trades)}"
                )
        
        # Close any remaining open positions
        logger.info("\nClosing remaining open positions...")
        for trade in list(self.open_trades.values()):
            final_bar = entry_data.iloc[-1]
            self._close_trade(
                trade,
                final_bar.name,
                final_bar['close'],
                'end_of_backtest'
            )
        
        # Generate results
        results = self._generate_results(symbol, initial_balance)
        
        logger.info("=" * 60)
        logger.info(f"Backtest Complete: {symbol}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Final Balance: ${results['final_balance']:,.2f}")
        logger.info(f"Return: {results['return_percent']:.2f}%")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        logger.info("=" * 60)
        
        return results
    
    def _update_open_positions_granular(
        self,
        current_time: datetime,
        min_bars: pd.DataFrame,
        stop_manager
    ):
        """
        Update open positions using 1m bars for granular exit detection.
        
        Args:
            current_time: Current timestamp
            min_bars: DataFrame of minimum timeframe bars in this period
            stop_manager: Stop manager instance
        """
        if len(min_bars) == 0:
            return
        
        for trade_id, trade in list(self.open_trades.items()):
            # Check each 1m bar in sequence for exits
            for idx, bar in min_bars.iterrows():
                bar_time = idx
                
                # Check stop loss first (priority)
                sl_hit = False
                exit_price = None
                
                if trade.direction == 'long':
                    if bar['low'] <= trade.stop_loss:
                        sl_hit = True
                        # Exit at stop or open if gap down
                        exit_price = min(trade.stop_loss, bar['open'])
                else:  # short
                    if bar['high'] >= trade.stop_loss:
                        sl_hit = True
                        # Exit at stop or open if gap up
                        exit_price = max(trade.stop_loss, bar['open'])
                
                if sl_hit:
                    self._close_trade(trade, bar_time, exit_price, 'stop_loss')
                    break  # Move to next trade
                
                # Check take profit (only if SL not hit)
                tp_hit = False
                
                if trade.take_profit_1:
                    if trade.direction == 'long':
                        if bar['high'] >= trade.take_profit_1:
                            tp_hit = True
                            exit_price = trade.take_profit_1
                    else:  # short
                        if bar['low'] <= trade.take_profit_1:
                            tp_hit = True
                            exit_price = trade.take_profit_1
                
                if tp_hit:
                    self._close_trade(trade, bar_time, exit_price, 'take_profit')
                    break  # Move to next trade
                
                # Update max favorable/adverse excursion
                if trade.direction == 'long':
                    excursion = bar['high'] - trade.entry_price
                    adverse = bar['low'] - trade.entry_price
                else:
                    excursion = trade.entry_price - bar['low']
                    adverse = trade.entry_price - bar['high']
                
                if excursion > 0:
                    trade.max_favorable = max(trade.max_favorable, excursion)
                if adverse < 0:
                    trade.max_adverse = min(trade.max_adverse, adverse)
    
    def _simulate_fill(
        self,
        bar: pd.Series,
        target_price: float,
        direction: str
    ) -> Dict:
        """Simulate order fill with slippage."""
        # Calculate slippage
        slippage = np.random.normal(0, self.slippage_percent / 100)
        
        # Apply slippage in unfavorable direction
        if direction == 'long':
            fill_price = target_price * (1 + abs(slippage))
        else:
            fill_price = target_price * (1 - abs(slippage))
        
        # Ensure fill price is within bar range
        fill_price = max(bar['low'], min(bar['high'], fill_price))
        
        return {
            'price': fill_price,
            'slippage': slippage,
            'filled': True
        }
    
    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_time: datetime,
        exit_price: float,
        reason: str
    ):
        """Close a trade and update balance."""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calculate P&L
        if trade.direction == 'long':
            pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.position_size
        
        # Deduct commission
        position_value = exit_price * trade.position_size
        commission = position_value * (self.commission_percent / 100)
        pnl -= commission
        
        trade.pnl = pnl
        
        # Calculate realized R:R
        risk = abs(trade.entry_price - trade.stop_loss) * trade.position_size
        trade.realized_rr = pnl / risk if risk > 0 else 0
        
        # Update balance
        self.balance += pnl
        
        # Move to closed trades
        self.trades.append(trade)
        del self.open_trades[trade.trade_id]
        
        logger.info(
            f"[{exit_time}] Exit: {trade.symbol} @ {exit_price:.2f} ({reason}), "
            f"P&L: ${pnl:.2f}, R:R: {trade.realized_rr:.2f}"
        )
    
    def _calculate_unrealized_pnl(self, trade: BacktestTrade, current_price: float) -> float:
        """Calculate unrealized P&L for open trade."""
        if trade.direction == 'long':
            return (current_price - trade.entry_price) * trade.position_size
        else:
            return (trade.entry_price - current_price) * trade.position_size
    
    def _generate_results(self, symbol: str, initial_balance: float) -> Dict:
        """Generate backtest results."""
        trade_dicts = [
            {
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'realized_rr': t.realized_rr,
                'exit_reason': t.exit_reason,
                'duration_minutes': (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else 0
            }
            for t in self.trades
        ]
        
        # Calculate metrics
        from learning.learner import StrategyLearner
        learner = StrategyLearner(None, {'learning': {}})
        metrics = learner.calculate_performance_metrics(trade_dicts, initial_balance)
        
        return {
            **metrics,
            'symbol': symbol,
            'trades': trade_dicts,
            'equity_curve': self.equity_curve,
            'configuration': {
                'lookback_bars': self.lookback_bars,
                'slippage_percent': self.slippage_percent,
                'commission_percent': self.commission_percent,
                'initial_balance': initial_balance
            }
        }
    
    def export_trades(self, filepath: str):
        """Export trades to CSV."""
        df = pd.DataFrame([
            {
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_time': t.entry_time,
                'entry_price': t.entry_price,
                'exit_time': t.exit_time,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'pnl': t.pnl,
                'realized_rr': t.realized_rr,
                'exit_reason': t.exit_reason
            }
            for t in self.trades
        ])
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(self.trades)} trades to {filepath}")
    
    def plot_equity_curve(self, filepath: Optional[str] = None):
        """Plot equity curve (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_curve)
            plt.title('Equity Curve')
            plt.xlabel('Bar')
            plt.ylabel('Balance ($)')
            plt.grid(True, alpha=0.3)
            
            if filepath:
                plt.savefig(filepath)
                logger.info(f"Saved equity curve to {filepath}")
            else:
                plt.show()
        
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


# Test harness
if __name__ == "__main__":
    import sys
    from strategy.engine import StrategyEngine
    from risk_management.money_manager import MoneyManager
    from risk_management.stop_manager import StopManager
    from indicators.indicators import TechnicalIndicators
    
    print("=" * 70)
    print(" BACKTESTER - Correct Implementation")
    print("=" * 70)
    
    # Check for CSV file
    csv_path = "data/XAUUSD_1m.csv"
    
    if not Path(csv_path).exists():
        print(f"\nERROR: CSV file not found: {csv_path}")
        print("\nPlease provide a CSV file with at least 2 months of 1m data")
        print("Format: timestamp,open,high,low,close,volume")
        sys.exit(1)
    
    # Configuration
    config = {
        'indicators': TechnicalIndicators._default_config(),
        'strategy': {
            'entry_types': ['breakout_retest'],
            'confluence_required': 2
        },
        'timeframes': {
            'structure_timeframe': '1h',
            'entry_timeframe': '15m',
            'minimum_timeframe': '1m'
        },
        'risk_management': {
            'max_risk_percent_per_trade': 1.0,
            'global_limits': {'max_concurrent_trades': 3}
        },
        'backtest': {
            'lookback_bars': 500,  # Single value for all timeframes
            'timeframes': {
                'entry_timeframe': '15m',
                'minimum_timeframe': '1m'
            },
            'simulation': {
                'slippage_percent': 0.05,
                'commission_percent': 0.1,
                'latency_bars': 1
            }
        }
    }
    
    # Initialize components
    strategy_engine = StrategyEngine(config)
    money_manager = MoneyManager(config)
    stop_manager = StopManager(config)
    backtester = Backtester(config)
    
    # Prepare data
    print("\nLoading and preparing data...")
    timeframes = ['1h', '15m', '1m']
    
    multi_tf_data = backtester.prepare_multi_timeframe_data(
        csv_path=csv_path,
        timeframes=timeframes,
        base_timeframe='1m'
    )
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtester.run(
        strategy_engine,
        money_manager,
        stop_manager,
        multi_tf_data,
        'XAUUSD',
        initial_balance=10000
    )
    
    # Display results
    print(f"\n{'='*70}")
    print(" RESULTS")
    print(f"{'='*70}")
    print(f"Total Trades:        {results['total_trades']}")
    print(f"Win Rate:            {results['win_rate']:.2%}")
    print(f"Total P&L:           ${results['total_pnl']:,.2f}")
    print(f"Return:              {results['return_percent']:.2f}%")
    print(f"Profit Factor:       {results['profit_factor']:.2f}")
    print(f"Max Drawdown:        {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
    print(f"{'='*70}")
    
    # Export
    if results['total_trades'] > 0:
        backtester.export_trades('data/backtest_trades.csv')
        print("\n✓ Trades exported to data/backtest_trades.csv")