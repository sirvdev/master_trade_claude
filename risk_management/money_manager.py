"""
Fixed Money Manager with proper MT5 position sizing.
The issue: calculating position size in $ value instead of lots for MT5.
"""

import logging
from typing import Dict, Optional
import math

logger = logging.getLogger(__name__)


class MoneyManager:
    """
    Manages position sizing with proper MT5 lot calculations.
    """
    
    def __init__(self, config: Dict):
        """Initialize money manager."""
        self.config = config.get('risk_management', {})
        self.max_risk_percent = self.config.get('max_risk_percent_per_trade', 1.0)
        self.use_dynamic_sizing = self.config.get('use_dynamic_sizing', True)
        self.global_limits = self.config.get('global_limits', {})
        
        # MT5 specific limits
        self.mt5_min_lot = 0.01
        self.mt5_max_lot = 100.0  # Conservative default
        self.mt5_lot_step = 0.01
        
    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
        symbol: str,
        direction: str,
        platform: str = 'binance',  # ADD THIS PARAMETER
        current_exposure: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate position size with platform-specific logic.
        
        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            stop_loss: Stop loss price
            symbol: Trading symbol
            direction: 'long' or 'short'
            platform: 'mt5' or 'binance'
            current_exposure: Optional dict of current positions
            
        Returns:
            Dictionary with position size and risk metrics
        """
        logger.info(
            f"Calculating position size for {symbol} {direction} on {platform} - "
            f"Equity: ${account_equity:.2f}, Entry: {entry_price:.4f}, SL: {stop_loss:.4f}"
        )
        
        # Calculate risk distance
        risk_distance = abs(entry_price - stop_loss)
        risk_distance_percent = (risk_distance / entry_price) * 100
        
        if risk_distance == 0:
            logger.error("Risk distance is zero - cannot calculate position size")
            return self._zero_position_response("Zero risk distance")
        
        # Calculate max risk amount in currency
        max_risk_amount = account_equity * (self.max_risk_percent / 100)
        
        # Platform-specific position size calculation
        if platform == 'mt5':
            position_size = self._calculate_mt5_position_size(
                max_risk_amount,
                entry_price,
                risk_distance,
                symbol
            )
        else:  # binance or other
            # Binance: position size is in base currency (BTC, ETH, etc.)
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
        
        # Apply platform-specific constraints
        if platform == 'mt5':
            position_size = self._apply_mt5_constraints(position_size)
        
        # Calculate actual risk with final position size
        if platform == 'mt5':
            # MT5: risk = lots × contract size × pip value × pip distance
            # For XAUUSD: 1 lot = 100 oz, pip = $0.01, so risk = lots × 100 × distance
            actual_risk = position_size * 100 * risk_distance
        else:
            actual_risk = position_size * risk_distance
        
        actual_risk_percent = (actual_risk / account_equity) * 100
        
        # Calculate position value
        if platform == 'mt5':
            position_value = position_size * 100 * entry_price  # lots × contract size × price
        else:
            position_value = position_size * entry_price
        
        leverage_used = position_value / account_equity if account_equity > 0 else 0
        
        result = {
            'position_size': round(position_size, 2),  # Round to 2 decimals for MT5
            'position_value': position_value,
            'risk_amount': actual_risk,
            'risk_percent': actual_risk_percent,
            'risk_distance': risk_distance,
            'risk_distance_percent': risk_distance_percent,
            'leverage_used': leverage_used,
            'max_risk_allowed': max_risk_amount,
            'sizing_method': 'dynamic' if self.use_dynamic_sizing else 'fixed',
            'platform': platform,
            'approved': True
        }
        
        logger.info(
            f"Position size calculated: {position_size:.2f} {'lots' if platform == 'mt5' else 'units'} "
            f"(${position_value:.2f}, Risk: ${actual_risk:.2f} = {actual_risk_percent:.2f}%)"
        )
        
        return result
    
    def _calculate_mt5_position_size(
        self,
        max_risk_amount: float,
        entry_price: float,
        risk_distance: float,
        symbol: str
    ) -> float:
        """
        Calculate MT5 position size in lots.
        
        For XAUUSD (Gold):
        - 1 lot = 100 troy ounces
        - 1 pip = $0.01
        - Risk = lots × 100 × price_distance
        
        Example:
        - Risk $100, entry 2000, SL 1990 (10 point distance)
        - Lots = 100 / (100 × 10) = 0.10 lots
        """
        # Contract size (standard)
        if 'XAU' in symbol or 'GOLD' in symbol.upper():
            contract_size = 100  # 100 oz per lot
        elif 'XAG' in symbol or 'SILVER' in symbol.upper():
            contract_size = 5000  # 5000 oz per lot
        else:
            contract_size = 100000  # Standard forex lot
        
        # Calculate lots needed
        # Risk = lots × contract_size × risk_distance
        # Therefore: lots = risk / (contract_size × risk_distance)
        lots = max_risk_amount / (contract_size * risk_distance)
        
        logger.debug(
            f"MT5 calculation: ${max_risk_amount:.2f} risk / "
            f"({contract_size} × {risk_distance:.4f}) = {lots:.4f} lots"
        )
        
        return lots
    
    def _apply_mt5_constraints(self, lots: float) -> float:
        """Apply MT5 lot size constraints."""
        # Ensure within min/max bounds
        lots = max(self.mt5_min_lot, min(lots, self.mt5_max_lot))
        
        # Round to lot step (usually 0.01)
        lots = round(lots / self.mt5_lot_step) * self.mt5_lot_step
        
        # Final bounds check after rounding
        if lots < self.mt5_min_lot:
            logger.warning(f"Position size {lots} below minimum {self.mt5_min_lot}")
            return 0
        
        if lots > self.mt5_max_lot:
            logger.warning(f"Position size {lots} above maximum {self.mt5_max_lot}, capping")
            lots = self.mt5_max_lot
        
        return lots
    
    def _apply_dynamic_adjustments(
        self,
        base_size: float,
        equity: float,
        symbol: str,
        current_exposure: Optional[Dict]
    ) -> float:
        """Apply dynamic adjustments based on market conditions."""
        adjusted_size = base_size
        
        # Reduce size if multiple positions open
        if current_exposure:
            open_positions = current_exposure.get('open_count', 0)
            if open_positions > 0:
                scale_factor = 1.0 - (open_positions * 0.1)
                scale_factor = max(0.5, scale_factor)
                adjusted_size *= scale_factor
                logger.info(f"Scaled position by {scale_factor:.2f}x due to {open_positions} open positions")
        
        # Check if symbol already has exposure
        if current_exposure and symbol in current_exposure.get('symbols', {}):
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
        """Apply global risk limits."""
        # Max concurrent trades
        max_concurrent = self.global_limits.get('max_concurrent_trades', 3)
        if current_exposure:
            open_count = current_exposure.get('open_count', 0)
            if open_count >= max_concurrent:
                logger.warning(f"Max concurrent trades reached ({max_concurrent})")
                return 0
        
        return position_size
    
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
        platform: str = 'binance',  # ADD THIS
        current_exposure: Optional[Dict] = None,
        daily_stats: Optional[Dict] = None,
        recent_trades: Optional[list] = None
    ) -> Dict:
        """Complete trade validation with platform awareness."""
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
        
        # Calculate position size with platform parameter
        sizing = self.calculate_position_size(
            account_equity,
            entry_price,
            stop_loss,
            symbol,
            direction,
            platform,  # PASS PLATFORM HERE
            current_exposure
        )
        
        if sizing['position_size'] == 0:
            sizing['approved'] = False
            sizing['reason'] = 'Position size calculated as zero'
        
        return sizing
    
    def check_daily_limits(self, current_stats: Dict) -> Dict:
        """Check if daily risk limits have been exceeded."""
        max_daily_dd = self.global_limits.get('daily_max_drawdown_percent', 5.0)
        max_trades_per_day = self.global_limits.get('max_trades_per_day', 10)
        
        current_dd = current_stats.get('daily_drawdown_percent', 0)
        trades_today = current_stats.get('trades_today', 0)
        
        limits_ok = True
        reasons = []
        
        if current_dd >= max_daily_dd:
            limits_ok = False
            reasons.append(f"Daily drawdown limit reached: {current_dd:.2f}% >= {max_daily_dd}%")
        
        if trades_today >= max_trades_per_day:
            limits_ok = False
            reasons.append(f"Daily trade limit reached: {trades_today} >= {max_trades_per_day}")
        
        return {
            'limits_ok': limits_ok,
            'reasons': reasons,
            'daily_drawdown': current_dd,
            'trades_today': trades_today
        }
    
    def check_consecutive_losses(self, recent_trades: list) -> Dict:
        """Check for consecutive losses and apply cooldown if needed."""
        cooldown_config = self.global_limits.get('cooldown_after_losses', {})
        if not cooldown_config.get('enabled', True):
            return {'cooldown_active': False}
        
        max_consecutive = cooldown_config.get('consecutive_losses', 3)
        
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= max_consecutive:
            return {
                'cooldown_active': True,
                'consecutive_losses': consecutive_losses,
                'reason': f"{consecutive_losses} consecutive losses"
            }
        
        return {'cooldown_active': False, 'consecutive_losses': consecutive_losses}


# Test the fixed calculation
if __name__ == "__main__":
    config = {
        'risk_management': {
            'max_risk_percent_per_trade': 1.0,
            'global_limits': {'max_concurrent_trades': 3}
        }
    }
    
    manager = MoneyManager(config)
    
    # Test MT5 position sizing (the problematic case from logs)
    print("=== MT5 Position Sizing Test ===")
    result = manager.calculate_position_size(
        account_equity=10000,
        entry_price=4084.002,
        stop_loss=4093.250857142857,
        symbol='XAUUSD',
        direction='short',
        platform='mt5'
    )
    
    print(f"Position Size: {result['position_size']:.2f} lots")
    print(f"Position Value: ${result['position_value']:.2f}")
    print(f"Risk Amount: ${result['risk_amount']:.2f}")
    print(f"Risk Percent: {result['risk_percent']:.2f}%")
    print(f"Approved: {result['approved']}")
    
    # Expected result:
    # Risk distance = 9.248857142857
    # Max risk = $100 (1% of $10,000)
    # Lots = 100 / (100 × 9.248857142857) = 0.11 lots
    # This is much more reasonable than the 10.81 lots that was calculated before!