"""
Backtesting engine for strategy validation.
Simulates trading on historical data with realistic slippage and fills.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import copy

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
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtester.
        
        Args:
            config: Backtest configuration
        """
        self.config = config.get('backtest', {})
        self.simulation_config = self.config.get('simulation', {})
        
        self.slippage_percent = self.simulation_config.get('slippage_percent', 0.05)
        self.commission_percent = self.simulation_config.get('commission_percent', 0.1)
        self.latency_bars = self.simulation_config.get('latency_bars', 1)
        
        # State
        self.trades: List[BacktestTrade] = []
        self.open_trades: Dict[str, BacktestTrade] = {}
        self.equity_curve = []
        self.balance = 10000  # Starting balance
        
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
        logger.info(f"Starting backtest for {symbol}")
        logger.info(f"Initial balance: ${initial_balance:,.2f}")
        
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades = []
        self.open_trades = {}
        
        # Get entry timeframe data for iteration
        entry_tf = '5m'  # From config
        if entry_tf not in multi_tf_data:
            raise ValueError(f"Entry timeframe {entry_tf} not in data")
            
        entry_data = multi_tf_data[entry_tf]
        total_bars = len(entry_data)
        
        logger.info(f"Processing {total_bars} bars")
        
        # Iterate through bars
        for i in range(100, total_bars):  # Start after warmup period
            current_bar = entry_data.iloc[i]
            current_time = current_bar.name
            current_price = current_bar['close']
            
            # Get multi-timeframe snapshot
            tf_snapshot = {}
            for tf, df in multi_tf_data.items():
                # Get data up to current bar
                tf_snapshot[tf] = df.iloc[:i+1].tail(200)  # Last 200 bars
                
            # Update existing positions
            self._update_open_positions(
                current_time,
                current_bar,
                stop_manager,
                tf_snapshot
            )
            
            # Skip if too many open positions
            if len(self.open_trades) >= 3:  # Max concurrent from config
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
                        fill_result = self._simulate_fill(
                            entry_data.iloc[i+self.latency_bars] if i+self.latency_bars < total_bars else current_bar,
                            levels['entry_price'],
                            analysis['direction']
                        )
                        
                        # Create trade
                        trade = BacktestTrade(
                            trade_id=f"bt_{i}_{symbol}",
                            symbol=symbol,
                            direction=analysis['direction'],
                            entry_time=current_time,
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
                        
                        logger.debug(
                            f"Entry: {trade.direction} {symbol} @ {trade.entry_price:.2f}, "
                            f"SL: {trade.stop_loss:.2f}, Size: {trade.position_size:.4f}"
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
            if i % 100 == 0:
                progress = (i / total_bars) * 100
                logger.info(
                    f"Progress: {progress:.1f}% - "
                    f"Balance: ${self.balance:,.2f}, "
                    f"Open: {len(self.open_trades)}, "
                    f"Closed: {len(self.trades)}"
                )
                
        # Close any remaining open positions
        logger.info("Closing remaining open positions...")
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
        
    def _update_open_positions(
        self,
        current_time: datetime,
        current_bar: pd.Series,
        stop_manager,
        tf_snapshot: Dict
    ):
        """Update open positions and check for exits."""
        current_price = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        
        for trade_id, trade in list(self.open_trades.items()):
            # Check stop loss hit
            sl_hit = False
            if trade.direction == 'long':
                if low <= trade.stop_loss:
                    sl_hit = True
                    exit_price = min(trade.stop_loss, current_bar['open'])
            else:
                if high >= trade.stop_loss:
                    sl_hit = True
                    exit_price = max(trade.stop_loss, current_bar['open'])
                    
            if sl_hit:
                self._close_trade(trade, current_time, exit_price, 'stop_loss')
                continue
                
            # Check take profit
            tp_hit = False
            if trade.take_profit_1:
                if trade.direction == 'long':
                    if high >= trade.take_profit_1:
                        tp_hit = True
                        exit_price = trade.take_profit_1
                else:
                    if low <= trade.take_profit_1:
                        tp_hit = True
                        exit_price = trade.take_profit_1
                        
            if tp_hit:
                self._close_trade(trade, current_time, exit_price, 'take_profit')
                continue
                
            # Update trailing stop if enabled
            if hasattr(stop_manager, 'update_trailing_stop'):
                # Get ATR from current data
                entry_tf = '5m'
                if entry_tf in tf_snapshot:
                    from indicators.indicators import TechnicalIndicators
                    indicators = TechnicalIndicators()
                    atr_result = indicators.calculate_atr(tf_snapshot[entry_tf])
                    current_atr = atr_result['current']
                    
                    # Track high/low since entry
                    if not hasattr(trade, 'high_since_entry'):
                        trade.high_since_entry = trade.entry_price
                        trade.low_since_entry = trade.entry_price
                        
                    trade.high_since_entry = max(trade.high_since_entry, high)
                    trade.low_since_entry = min(trade.low_since_entry, low)
                    
                    # Update trailing stop
                    update = stop_manager.update_trailing_stop(
                        trade={
                            'entry_price': trade.entry_price,
                            'stop_loss': trade.stop_loss,
                            'direction': trade.direction,
                            'position_size': trade.position_size
                        },
                        current_price=current_price,
                        atr=current_atr,
                        high_since_entry=trade.high_since_entry,
                        low_since_entry=trade.low_since_entry
                    )
                    
                    if update.get('update_required'):
                        trade.stop_loss = update['new_stop_loss']
                        
            # Track max favorable/adverse excursion
            if trade.direction == 'long':
                excursion = current_price - trade.entry_price
            else:
                excursion = trade.entry_price - current_price
                
            if excursion > 0:
                trade.max_favorable = max(trade.max_favorable, excursion)
            else:
                trade.max_adverse = min(trade.max_adverse, excursion)
                
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
        
        logger.debug(
            f"Exit: {trade.symbol} @ {exit_price:.2f} ({reason}), "
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


# Example usage
if __name__ == "__main__":
    from strategy.engine import StrategyEngine
    from risk_management.money_manager import MoneyManager
    from risk_management.stop_manager import StopManager
    from indicators.indicators import TechnicalIndicators
    
    # Generate sample data
    print("=== Backtester Test ===\n")
    
    dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')
    np.random.seed(42)
    
    # Create realistic price data with trend
    price = 50000
    prices = [price]
    for _ in range(499):
        change = np.random.normal(0, 100)
        price = max(price + change, 1000)  # Prevent negative
        prices.append(price)
        
    df_5m = pd.DataFrame({
        'open': prices,
        'high': [p * 1.002 for p in prices],
        'low': [p * 0.998 for p in prices],
        'close': prices,
        'volume': np.random.randint(100, 1000, 500)
    }, index=dates)
    
    # Create higher timeframe data (simplified)
    df_1h = df_5m.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    multi_tf_data = {'5m': df_5m, '1H': df_1h}
    
    # Initialize components
    config = {
        'indicators': TechnicalIndicators._default_config(),
        'strategy': {'entry_types': ['breakout_retest'], 'confluence_required': 2},
        'timeframes': {'structure_timeframe': '1H', 'entry_timeframe': '5m'},
        'risk_management': {
            'max_risk_percent_per_trade': 1.0,
            'global_limits': {'max_concurrent_trades': 3}
        },
        'backtest': {
            'simulation': {
                'slippage_percent': 0.05,
                'commission_percent': 0.1
            }
        }
    }
    
    strategy_engine = StrategyEngine(config)
    money_manager = MoneyManager(config)
    stop_manager = StopManager(config)
    backtester = Backtester(config)
    
    # Run backtest
    results = backtester.run(
        strategy_engine,
        money_manager,
        stop_manager,
        multi_tf_data,
        'BTC/USDT',
        initial_balance=10000
    )
    
    print(f"\n=== Backtest Results ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    print(f"Return: {results['return_percent']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    print("\nBacktester test completed!")