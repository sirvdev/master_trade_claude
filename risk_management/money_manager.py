"""
Position sizing and money management module.
Calculates dynamic position sizes based on account equity and risk parameters.
"""

import logging
from typing import Dict, Optional
import math

logger = logging.getLogger(__name__)


class MoneyManager:
    """
    Manages position sizing and risk allocation across trades.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize money manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config.get('risk_management', {})
        self.max_risk_percent = self.config.get('max_risk_percent_per_trade', 1.0)
        self.use_dynamic_sizing = self.config.get('use_dynamic_sizing', True)
        self.global_limits = self.config.get('global_limits', {})
        
    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
        symbol: str,
        direction: str,
        current_exposure: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            stop_loss: Stop loss price
            symbol: Trading symbol
            direction: 'long' or 'short'
            current_exposure: Optional dict of current positions
            
        Returns:
            Dictionary with position size and risk metrics
        """
        logger.info(
            f"Calculating position size for {symbol} {direction} - "
            f"Equity: {account_equity}, Entry: {entry_price}, SL: {stop_loss}"
        )
        
        # Calculate risk distance
        risk_distance = abs(entry_price - stop_loss)
        risk_distance_percent = (risk_distance / entry_price) * 100
        
        # Calculate max risk amount in currency
        max_risk_amount = account_equity * (self.max_risk_percent / 100)
        
        # Calculate base position size
        if risk_distance == 0:
            logger.error("Risk distance is zero - cannot calculate position size")
            return self._zero_position_response("Zero risk distance")
            
        # Position size = Risk Amount / Risk per Unit
        position_size = max_risk_amount / risk_distance
        
        # Apply dynamic sizing adjustments
        if self.use_dynamic_sizing:
            position_size = self._apply_dynamic_adjustments(
                position_size,
                account_equity,
                symbol,
                current_exposure
            )
            
        # Apply global limits
        position_size = self._apply_global_limits(
            position_size,
            account_equity,
            entry_price,
            symbol,
            current_exposure
        )
        
        # Calculate actual risk with final position size
        actual_risk = position_size * risk_distance
        actual_risk_percent = (actual_risk / account_equity) * 100
        
        # Calculate position value
        position_value = position_size * entry_price
        leverage_used = position_value / account_equity if account_equity > 0 else 0
        
        result = {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': actual_risk,
            'risk_percent': actual_risk_percent,
            'risk_distance': risk_distance,
            'risk_distance_percent': risk_distance_percent,
            'leverage_used': leverage_used,
            'max_risk_allowed': max_risk_amount,
            'sizing_method': 'dynamic' if self.use_dynamic_sizing else 'fixed',
            'approved': True
        }
        
        logger.info(
            f"Position size calculated: {position_size:.4f} units "
            f"(${position_value:.2f}, Risk: {actual_risk_percent:.2f}%)"
        )
        
        return result
        
    def _apply_dynamic_adjustments(
        self,
        base_size: float,
        equity: float,
        symbol: str,
        current_exposure: Optional[Dict]
    ) -> float:
        """
        Apply dynamic adjustments based on market conditions and performance.
        
        Args:
            base_size: Base position size
            equity: Account equity
            symbol: Trading symbol
            current_exposure: Current positions
            
        Returns:
            Adjusted position size
        """
        adjusted_size = base_size
        
        # Reduce size if multiple positions open
        if current_exposure:
            open_positions = current_exposure.get('open_count', 0)
            if open_positions > 0:
                # Scale down by 10% for each additional position
                scale_factor = 1.0 - (open_positions * 0.1)
                scale_factor = max(0.5, scale_factor)  # Minimum 50%
                adjusted_size *= scale_factor
                logger.info(f"Scaled position size by {scale_factor:.2f}x due to {open_positions} open positions")
                
        # Check if symbol already has exposure
        if current_exposure and symbol in current_exposure.get('symbols', {}):
            # Reduce size for same symbol
            adjusted_size *= 0.5
            logger.info(f"Halved position size - existing exposure in {symbol}")
            
        return adjusted_size
        
    def _apply_global_limits(
        self,
        position_size: float,
        equity: float,
        entry_price: float,
        symbol: str,
        current_exposure: Optional[Dict]
    ) -> float:
        """
        Apply global risk limits.
        
        Args:
            position_size: Calculated position size
            equity: Account equity
            entry_price: Entry price
            symbol: Trading symbol
            current_exposure: Current positions
            
        Returns:
            Position size after applying limits
        """
        # Max risk per symbol limit
        max_symbol_risk = self.global_limits.get('max_risk_per_symbol_percent', 2.0)
        
        if current_exposure and symbol in current_exposure.get('symbols', {}):
            current_symbol_risk = current_exposure['symbols'][symbol].get('risk_percent', 0)
            new_total_risk = current_symbol_risk + ((position_size * entry_price) / equity * 100)
            
            if new_total_risk > max_symbol_risk:
                # Scale down to stay within limit
                allowed_additional_risk = max_symbol_risk - current_symbol_risk
                if allowed_additional_risk <= 0:
                    logger.warning(f"Symbol {symbol} already at max risk - rejecting trade")
                    return 0
                    
                max_additional_value = equity * (allowed_additional_risk / 100)
                position_size = max_additional_value / entry_price
                logger.info(f"Scaled position to stay within symbol risk limit: {max_symbol_risk}%")
                
        # Max concurrent trades limit
        max_concurrent = self.global_limits.get('max_concurrent_trades', 3)
        if current_exposure:
            open_count = current_exposure.get('open_count', 0)
            if open_count >= max_concurrent:
                logger.warning(f"Max concurrent trades reached ({max_concurrent}) - rejecting trade")
                return 0
                
        return position_size
        
    def check_daily_limits(self, current_stats: Dict) -> Dict:
        """
        Check if daily risk limits have been exceeded.
        
        Args:
            current_stats: Dictionary with daily statistics
            
        Returns:
            Dictionary with limit check results
        """
        max_daily_dd = self.global_limits.get('daily_max_drawdown_percent', 5.0)
        max_trades_per_day = self.global_limits.get('max_trades_per_day', 10)
        
        current_dd = current_stats.get('daily_drawdown_percent', 0)
        trades_today = current_stats.get('trades_today', 0)
        
        limits_ok = True
        reasons = []
        
        if current_dd >= max_daily_dd:
            limits_ok = False
            reasons.append(f"Daily drawdown limit reached: {current_dd:.2f}% >= {max_daily_dd}%")
            logger.warning(reasons[-1])
            
        if trades_today >= max_trades_per_day:
            limits_ok = False
            reasons.append(f"Daily trade limit reached: {trades_today} >= {max_trades_per_day}")
            logger.warning(reasons[-1])
            
        return {
            'limits_ok': limits_ok,
            'reasons': reasons,
            'daily_drawdown': current_dd,
            'trades_today': trades_today,
            'max_daily_drawdown': max_daily_dd,
            'max_trades_per_day': max_trades_per_day
        }
        
    def check_consecutive_losses(self, recent_trades: list) -> Dict:
        """
        Check for consecutive losses and apply cooldown if needed.
        
        Args:
            recent_trades: List of recent trade results
            
        Returns:
            Dictionary with cooldown status
        """
        cooldown_config = self.global_limits.get('cooldown_after_losses', {})
        if not cooldown_config.get('enabled', True):
            return {'cooldown_active': False}
            
        max_consecutive = cooldown_config.get('consecutive_losses', 3)
        cooldown_seconds = cooldown_config.get('cooldown_seconds', 3600)
        
        # Count consecutive losses
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
                
        if consecutive_losses >= max_consecutive:
            logger.warning(
                f"Consecutive losses detected: {consecutive_losses} - "
                f"Applying {cooldown_seconds}s cooldown"
            )
            return {
                'cooldown_active': True,
                'consecutive_losses': consecutive_losses,
                'cooldown_seconds': cooldown_seconds,
                'reason': f"{consecutive_losses} consecutive losses"
            }
            
        return {
            'cooldown_active': False,
            'consecutive_losses': consecutive_losses
        }
        
    def calculate_portfolio_risk(self, open_positions: list) -> Dict:
        """
        Calculate total portfolio risk from open positions.
        
        Args:
            open_positions: List of open position dictionaries
            
        Returns:
            Portfolio risk metrics
        """
        total_risk = 0
        total_value = 0
        symbols = {}
        
        for position in open_positions:
            risk = position.get('risk_amount', 0)
            value = position.get('position_value', 0)
            symbol = position.get('symbol')
            
            total_risk += risk
            total_value += value
            
            if symbol not in symbols:
                symbols[symbol] = {'risk': 0, 'value': 0, 'count': 0}
                
            symbols[symbol]['risk'] += risk
            symbols[symbol]['value'] += value
            symbols[symbol]['count'] += 1
            
        return {
            'total_risk': total_risk,
            'total_value': total_value,
            'num_positions': len(open_positions),
            'symbols': symbols,
            'diversification_score': len(symbols) / max(len(open_positions), 1)
        }
        
    def _zero_position_response(self, reason: str) -> Dict:
        """Return zero position size response."""
        return {
            'position_size': 0,
            'position_value': 0,
            'risk_amount': 0,
            'risk_percent': 0,
            'approved': False,
            'reason': reason
        }
        
    def validate_trade(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
        symbol: str,
        direction: str,
        current_exposure: Optional[Dict] = None,
        daily_stats: Optional[Dict] = None,
        recent_trades: Optional[list] = None
    ) -> Dict:
        """
        Complete trade validation including all checks.
        
        Args:
            account_equity: Current equity
            entry_price: Entry price
            stop_loss: Stop loss price
            symbol: Trading symbol
            direction: Trade direction
            current_exposure: Current positions
            daily_stats: Daily statistics
            recent_trades: Recent trade history
            
        Returns:
            Validation result with position size or rejection reason
        """
        # Check daily limits
        if daily_stats:
            limit_check = self.check_daily_limits(daily_stats)
            if not limit_check['limits_ok']:
                logger.warning(f"Trade rejected: {limit_check['reasons']}")
                return {
                    'approved': False,
                    'reason': '; '.join(limit_check['reasons']),
                    'limit_check': limit_check
                }
                
        # Check consecutive losses cooldown
        if recent_trades:
            cooldown = self.check_consecutive_losses(recent_trades)
            if cooldown['cooldown_active']:
                logger.warning(f"Trade rejected: Cooldown active")
                return {
                    'approved': False,
                    'reason': f"Cooldown active: {cooldown['reason']}",
                    'cooldown': cooldown
                }
                
        # Calculate position size
        sizing = self.calculate_position_size(
            account_equity,
            entry_price,
            stop_loss,
            symbol,
            direction,
            current_exposure
        )
        
        if sizing['position_size'] == 0:
            sizing['approved'] = False
            sizing['reason'] = 'Position size calculated as zero'
            
        return sizing


# Example usage
if __name__ == "__main__":
    config = {
        'risk_management': {
            'max_risk_percent_per_trade': 1.0,
            'use_dynamic_sizing': True,
            'global_limits': {
                'daily_max_drawdown_percent': 5.0,
                'max_concurrent_trades': 3,
                'max_trades_per_day': 10,
                'max_risk_per_symbol_percent': 2.0,
                'cooldown_after_losses': {
                    'enabled': True,
                    'consecutive_losses': 3,
                    'cooldown_seconds': 3600
                }
            }
        }
    }
    
    manager = MoneyManager(config)
    
    # Test position sizing
    print("=== Position Sizing Test ===")
    result = manager.calculate_position_size(
        account_equity=10000,
        entry_price=50000,
        stop_loss=49500,
        symbol='BTC/USDT',
        direction='long'
    )
    
    print(f"Position Size: {result['position_size']:.6f} BTC")
    print(f"Position Value: ${result['position_value']:.2f}")
    print(f"Risk Amount: ${result['risk_amount']:.2f}")
    print(f"Risk Percent: {result['risk_percent']:.2f}%")
    print(f"Leverage: {result['leverage_used']:.2f}x")
    
    # Test daily limits
    print("\n=== Daily Limits Test ===")
    daily_stats = {
        'daily_drawdown_percent': 3.5,
        'trades_today': 5
    }
    limit_check = manager.check_daily_limits(daily_stats)
    print(f"Limits OK: {limit_check['limits_ok']}")
    print(f"Current DD: {limit_check['daily_drawdown']:.2f}%")
    print(f"Trades Today: {limit_check['trades_today']}")
    
    # Test consecutive losses
    print("\n=== Consecutive Losses Test ===")
    recent_trades = [
        {'pnl': -100},
        {'pnl': -50},
        {'pnl': -75}
    ]
    cooldown = manager.check_consecutive_losses(recent_trades)
    print(f"Cooldown Active: {cooldown['cooldown_active']}")
    print(f"Consecutive Losses: {cooldown['consecutive_losses']}")
    
    # Test full validation
    print("\n=== Full Validation Test ===")
    validation = manager.validate_trade(
        account_equity=10000,
        entry_price=2050,
        stop_loss=2045,
        symbol='XAU/USD',
        direction='long',
        daily_stats=daily_stats,
        recent_trades=recent_trades
    )
    
    print(f"Trade Approved: {validation['approved']}")
    if validation['approved']:
        print(f"Position Size: {validation['position_size']:.4f}")
    else:
        print(f"Rejection Reason: {validation.get('reason')}")
    
    print("\nMoney manager test completed!")