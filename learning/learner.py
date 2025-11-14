"""
Learning engine for automated parameter optimization.
Ingests trade logs, calculates performance metrics, and optimizes strategy parameters.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from itertools import product
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class StrategyLearner:
    """
    Automated learning and parameter optimization engine.
    """
    
    def __init__(self, db_manager, config: Dict):
        """
        Initialize learner.
        
        Args:
            db_manager: Database manager instance
            config: Learning configuration
        """
        self.db = db_manager
        self.config = config.get('learning', {})
        self.optimization_config = self.config.get('optimization', {})
        self.guardrails = self.config.get('guardrails', {})
        
    def calculate_performance_metrics(
        self,
        trades: List[Dict],
        initial_balance: float = 10000
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: List of closed trade dictionaries
            initial_balance: Starting balance for calculations
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            return self._empty_metrics()
            
        # Extract trade data
        pnls = [t.get('pnl', 0) for t in trades]
        rrs = [t.get('realized_rr', 0) for t in trades if t.get('realized_rr')]
        durations = [t.get('duration_minutes', 0) for t in trades if t.get('duration_minutes')]
        
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        total_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        # Basic metrics
        win_rate = num_wins / total_trades if total_trades > 0 else 0
        total_pnl = sum(pnls)
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        avg_rr = np.mean(rrs) if rrs else 0
        
        # Profit factor
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Calculate equity curve for drawdown and Sharpe
        equity_curve = [initial_balance]
        for pnl in pnls:
            equity_curve.append(equity_curve[-1] + pnl)
            
        equity_series = pd.Series(equity_curve)
        
        # Drawdown calculation
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 252 trading days)
        if len(pnls) > 1:
            returns = pd.Series(pnls) / initial_balance
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Consecutive wins/losses
        consecutive_wins = self._max_consecutive(pnls, positive=True)
        consecutive_losses = self._max_consecutive(pnls, positive=False)
        
        # Average duration
        avg_duration_hours = np.mean(durations) / 60 if durations else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_rr': avg_rr,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'avg_duration_hours': avg_duration_hours,
            'final_balance': equity_curve[-1],
            'return_percent': (equity_curve[-1] - initial_balance) / initial_balance * 100
        }
        
    def _max_consecutive(self, values: List[float], positive: bool = True) -> int:
        """Calculate maximum consecutive wins or losses."""
        max_consecutive = 0
        current_consecutive = 0
        
        for value in values:
            if (positive and value > 0) or (not positive and value < 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
        
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_rr': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_duration_hours': 0,
            'final_balance': 10000,
            'return_percent': 0
        }
        
    def analyze_by_symbol(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Calculate metrics per symbol."""
        symbols = {}
        for trade in trades:
            symbol = trade.get('symbol')
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(trade)
            
        return {
            symbol: self.calculate_performance_metrics(symbol_trades)
            for symbol, symbol_trades in symbols.items()
        }
        
    def analyze_by_timeframe(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Calculate metrics per entry timeframe."""
        # Would need to store timeframe in trade logs
        # Simplified version
        return {}
        
    def run_grid_search_optimization(
        self,
        param_ranges: Dict[str, List],
        backtest_function,
        historical_data: Dict
    ) -> Dict:
        """
        Run grid search parameter optimization.
        
        Args:
            param_ranges: Dictionary of parameter names to lists of values
            backtest_function: Function that runs backtest with params
            historical_data: Historical market data
            
        Returns:
            Best parameters and results
        """
        logger.info(f"Starting grid search with {len(param_ranges)} parameters")
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        combinations = list(product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        best_score = float('-inf')
        best_params = None
        best_metrics = None
        results = []
        
        optimization_metric = self.optimization_config.get('optimization_metric', 'expectancy')
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            # Run backtest with these parameters
            try:
                trades = backtest_function(historical_data, params)
                metrics = self.calculate_performance_metrics(trades)
                
                # Calculate score based on optimization metric
                score = metrics.get(optimization_metric, 0)
                
                # Apply multi-objective weighting if needed
                if self.optimization_config.get('multi_objective', False):
                    score = (
                        metrics['expectancy'] * 0.4 +
                        metrics['profit_factor'] * 0.3 +
                        metrics['sharpe_ratio'] * 0.2 +
                        metrics['win_rate'] * 0.1
                    )
                
                results.append({
                    'params': params,
                    'metrics': metrics,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
                    
                if (i + 1) % 10 == 0:
                    logger.info(f"Tested {i+1}/{len(combinations)} combinations")
                    
            except Exception as e:
                logger.error(f"Error testing params {params}: {e}")
                continue
                
        logger.info(
            f"Grid search complete. Best {optimization_metric}: {best_score:.4f}"
        )
        
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'best_score': best_score,
            'all_results': sorted(results, key=lambda x: x['score'], reverse=True)[:10],
            'total_combinations': len(combinations),
            'successful_tests': len(results)
        }
        
    def run_random_search_optimization(
        self,
        param_ranges: Dict[str, Tuple],
        backtest_function,
        historical_data: Dict,
        n_iterations: int = 50
    ) -> Dict:
        """
        Run random search parameter optimization.
        
        Args:
            param_ranges: Dict of param names to (min, max) tuples
            backtest_function: Backtest function
            historical_data: Historical data
            n_iterations: Number of random samples
            
        Returns:
            Best parameters and results
        """
        logger.info(f"Starting random search with {n_iterations} iterations")
        
        best_score = float('-inf')
        best_params = None
        best_metrics = None
        results = []
        
        optimization_metric = self.optimization_config.get('optimization_metric', 'expectancy')
        
        for i in range(n_iterations):
            # Generate random parameters
            params = {}
            for name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[name] = np.random.uniform(min_val, max_val)
                    
            try:
                trades = backtest_function(historical_data, params)
                metrics = self.calculate_performance_metrics(trades)
                score = metrics.get(optimization_metric, 0)
                
                results.append({
                    'params': params,
                    'metrics': metrics,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
                    
            except Exception as e:
                logger.error(f"Error testing params {params}: {e}")
                continue
                
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'best_score': best_score,
            'all_results': sorted(results, key=lambda x: x['score'], reverse=True)[:10]
        }
        
    def run_bandit_optimization(
        self,
        param_options: Dict[str, List],
        live_testing_function,
        n_rounds: int = 100,
        epsilon: float = 0.1
    ) -> Dict:
        """
        Run multi-armed bandit optimization (simplified RL approach).
        
        Args:
            param_options: Dict of parameters to option lists
            live_testing_function: Function to test params live (or on recent data)
            n_rounds: Number of test rounds
            epsilon: Exploration rate
            
        Returns:
            Best parameter combination
        """
        logger.info(f"Starting bandit optimization with {n_rounds} rounds")
        
        # Initialize arms (parameter combinations)
        param_names = list(param_options.keys())
        param_values = [param_options[name] for name in param_names]
        arms = [dict(zip(param_names, combo)) for combo in product(*param_values)]
        
        # Track rewards for each arm
        arm_rewards = {i: [] for i in range(len(arms))}
        arm_counts = {i: 0 for i in range(len(arms))}
        
        for round_num in range(n_rounds):
            # Epsilon-greedy selection
            if np.random.random() < epsilon:
                # Explore: random arm
                arm_idx = np.random.randint(len(arms))
            else:
                # Exploit: best average reward
                avg_rewards = {
                    i: np.mean(rewards) if rewards else 0
                    for i, rewards in arm_rewards.items()
                }
                arm_idx = max(avg_rewards, key=avg_rewards.get)
                
            # Test this arm
            params = arms[arm_idx]
            try:
                reward = live_testing_function(params)
                arm_rewards[arm_idx].append(reward)
                arm_counts[arm_idx] += 1
                
                logger.info(
                    f"Round {round_num+1}: Tested arm {arm_idx}, "
                    f"Reward: {reward:.4f}, Count: {arm_counts[arm_idx]}"
                )
            except Exception as e:
                logger.error(f"Error testing arm {arm_idx}: {e}")
                
        # Select best arm
        avg_rewards = {
            i: np.mean(rewards) if rewards else float('-inf')
            for i, rewards in arm_rewards.items()
        }
        best_arm_idx = max(avg_rewards, key=avg_rewards.get)
        
        return {
            'best_params': arms[best_arm_idx],
            'best_reward': avg_rewards[best_arm_idx],
            'arm_statistics': {
                i: {
                    'params': arms[i],
                    'avg_reward': np.mean(rewards) if rewards else 0,
                    'count': arm_counts[i],
                    'rewards': rewards
                }
                for i, rewards in arm_rewards.items()
            }
        }
        
    def validate_parameters(self, params: Dict, metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Validate parameters against guardrails.
        
        Args:
            params: Parameter dictionary
            metrics: Performance metrics
            
        Returns:
            Tuple of (is_valid, list of reasons)
        """
        issues = []
        
        # Check minimum sample size
        min_trades = self.guardrails.get('min_sample_size', 50)
        if metrics['total_trades'] < min_trades:
            issues.append(f"Insufficient trades: {metrics['total_trades']} < {min_trades}")
            
        # Check minimum expectancy
        min_expectancy = self.guardrails.get('min_expectancy', 0.5)
        if metrics['expectancy'] < min_expectancy:
            issues.append(f"Expectancy too low: {metrics['expectancy']:.2f} < {min_expectancy}")
            
        # Check max drawdown
        max_dd = self.guardrails.get('max_drawdown_threshold', 20.0)
        if abs(metrics['max_drawdown']) > max_dd:
            issues.append(f"Drawdown too high: {abs(metrics['max_drawdown']):.2f}% > {max_dd}%")
            
        # Check Sharpe ratio
        if self.guardrails.get('require_positive_sharpe', True):
            if metrics['sharpe_ratio'] <= 0:
                issues.append(f"Negative Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                
        is_valid = len(issues) == 0
        
        return is_valid, issues
        
    def create_parameter_version(
        self,
        params: Dict,
        metrics: Dict,
        source: str,
        notes: Optional[str] = None
    ) -> int:
        """
        Save parameter version to database.
        
        Args:
            params: Parameters dictionary
            metrics: Performance metrics
            source: Source of parameters (e.g., 'grid_search')
            notes: Additional notes
            
        Returns:
            Version ID
        """
        is_valid, issues = self.validate_parameters(params, metrics)
        
        status = 'pending' if is_valid else 'rejected'
        if issues:
            notes = (notes or '') + f" Validation issues: {'; '.join(issues)}"
            
        version_data = {
            'version_name': f"{source}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'parameters': params,
            'source': source,
            'backtest_metrics': metrics,
            'status': status,
            'notes': notes
        }
        
        version_id = self.db.save_parameter_version(version_data)
        logger.info(f"Saved parameter version {version_id} ({status})")
        
        return version_id
        
    def run_learning_cycle(
        self,
        backtest_function,
        historical_data: Dict,
        days_lookback: int = 30
    ) -> Dict:
        """
        Run complete learning cycle.
        
        Args:
            backtest_function: Backtest function
            historical_data: Historical market data
            days_lookback: Days of historical trades to analyze
            
        Returns:
            Learning results
        """
        logger.info("=" * 50)
        logger.info("Starting learning cycle")
        logger.info("=" * 50)
        
        # Get recent trades
        cutoff_date = datetime.utcnow() - timedelta(days=days_lookback)
        trades = self.db.get_trades(
            filters={'status': 'closed'},
            limit=1000
        )
        trades = [t for t in trades if t['entry_time'] >= cutoff_date]
        
        logger.info(f"Analyzing {len(trades)} trades from last {days_lookback} days")
        
        # Calculate current performance
        current_metrics = self.calculate_performance_metrics(trades)
        logger.info(f"Current performance: Expectancy={current_metrics['expectancy']:.2f}, "
                   f"Win Rate={current_metrics['win_rate']:.2%}, "
                   f"Sharpe={current_metrics['sharpe_ratio']:.2f}")
        
        # Check if optimization is needed
        if current_metrics['total_trades'] < self.guardrails.get('min_sample_size', 50):
            logger.info("Insufficient trades for optimization")
            return {'status': 'skipped', 'reason': 'insufficient_data'}
            
        # Run optimization
        optimization_method = self.optimization_config.get('method', 'grid_search')
        
        run_id = self.db.log_learning_run({
            'optimization_method': optimization_method,
            'trades_analyzed': len(trades)
        })
        
        try:
            if optimization_method == 'grid_search':
                param_ranges = self.optimization_config.get('param_ranges', {})
                results = self.run_grid_search_optimization(
                    param_ranges,
                    backtest_function,
                    historical_data
                )
            elif optimization_method == 'random_search':
                param_ranges = self.optimization_config.get('param_ranges', {})
                # Convert to min/max format
                param_ranges_tuple = {
                    k: (min(v), max(v))
                    for k, v in param_ranges.items()
                }
                results = self.run_random_search_optimization(
                    param_ranges_tuple,
                    backtest_function,
                    historical_data
                )
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
                
            # Create parameter version
            version_id = self.create_parameter_version(
                results['best_params'],
                results['best_metrics'],
                optimization_method,
                notes=f"Learning run {run_id}"
            )
            
            # Update learning run
            cursor = self.db.conn.cursor()
            cursor.execute("""
                UPDATE learning_runs
                SET completed_at = ?,
                    best_parameters = ?,
                    metrics = ?,
                    recommended_version_id = ?,
                    status = 'completed'
                WHERE run_id = ?
            """, (
                datetime.utcnow(),
                json.dumps(results['best_params']),
                json.dumps(results['best_metrics']),
                version_id,
                run_id
            ))
            self.db.conn.commit()
            
            logger.info("=" * 50)
            logger.info("Learning cycle complete")
            logger.info(f"Best parameters: {results['best_params']}")
            logger.info(f"Improvement: {results['best_metrics']['expectancy']:.2f} vs {current_metrics['expectancy']:.2f}")
            logger.info("=" * 50)
            
            return {
                'status': 'completed',
                'run_id': run_id,
                'version_id': version_id,
                'current_metrics': current_metrics,
                'optimized_metrics': results['best_metrics'],
                'best_params': results['best_params'],
                'improvement_percent': (
                    (results['best_metrics']['expectancy'] - current_metrics['expectancy']) /
                    abs(current_metrics['expectancy']) * 100
                    if current_metrics['expectancy'] != 0 else 0
                )
            }
            
        except Exception as e:
            logger.error(f"Learning cycle failed: {e}", exc_info=True)
            
            # Update learning run as failed
            cursor = self.db.conn.cursor()
            cursor.execute("""
                UPDATE learning_runs
                SET completed_at = ?, status = 'failed', error_message = ?
                WHERE run_id = ?
            """, (datetime.utcnow(), str(e), run_id))
            self.db.conn.commit()
            
            return {'status': 'failed', 'error': str(e)}


# Example usage
if __name__ == "__main__":
    from logger.db import DatabaseManager
    
    # Initialize
    db = DatabaseManager("test_trading.db")
    db.connect()
    
    config = {
        'learning': {
            'optimization': {
                'method': 'grid_search',
                'param_ranges': {
                    'atr_multiplier': [1.5, 2.0, 2.5],
                    'rsi_oversold': [25, 30, 35],
                    'confluence_required': [2, 3]
                },
                'optimization_metric': 'expectancy'
            },
            'guardrails': {
                'min_sample_size': 30,
                'min_expectancy': 0.3,
                'max_drawdown_threshold': 15.0
            }
        }
    }
    
    learner = StrategyLearner(db, config)
    
    # Test metrics calculation
    print("=== Testing Metrics Calculation ===")
    sample_trades = [
        {'pnl': 100, 'realized_rr': 2.0, 'duration_minutes': 120},
        {'pnl': -50, 'realized_rr': -1.0, 'duration_minutes': 60},
        {'pnl': 150, 'realized_rr': 3.0, 'duration_minutes': 180},
        {'pnl': 75, 'realized_rr': 1.5, 'duration_minutes': 90},
        {'pnl': -40, 'realized_rr': -0.8, 'duration_minutes': 45},
    ]
    
    metrics = learner.calculate_performance_metrics(sample_trades)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Expectancy: {metrics['expectancy']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Test parameter validation
    print("\n=== Testing Parameter Validation ===")
    is_valid, issues = learner.validate_parameters({'atr_mult': 2.0}, metrics)
    print(f"Valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    db.disconnect()
    print("\nLearner test completed!")