"""
Stop loss and take profit management including trailing stops.
Handles breakeven moves, partial profit-taking, and long runner logic.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class StopManager:
    """
    Manages stop loss, take profit, and trailing stop logic.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize stop manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config.get('risk_management', {})
        self.sl_config = self.config.get('stop_loss', {})
        self.tp_config = self.config.get('take_profit', {})
        self.trailing_config = self.config.get('trailing_stop', {})
        self.long_runner_config = self.config.get('long_runner', {})
        
    def calculate_initial_stops(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None
    ) -> Dict:
        """
        Calculate initial stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range value
            swing_high: Recent swing high for structure stop
            swing_low: Recent swing low for structure stop
            
        Returns:
            Dictionary with stop levels
        """
        logger.info(
            f"Calculating stops for {direction} @ {entry_price}, "
            f"ATR: {atr}, Swing H/L: {swing_high}/{swing_low}"
        )
        
        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(
            entry_price,
            direction,
            atr,
            swing_high,
            swing_low
        )
        
        # Calculate take profit targets
        risk = abs(entry_price - stop_loss)
        take_profits = self._calculate_take_profit_levels(
            entry_price,
            direction,
            risk
        )
        
        result = {
            'stop_loss': stop_loss,
            'take_profit_levels': take_profits,
            'risk_distance': risk,
            'stop_method': self.sl_config.get('method', 'conservative'),
            'trailing_enabled': self.trailing_config.get('enabled', True),
            'breakeven_enabled': self.trailing_config.get('breakeven', {}).get('enabled', True)
        }
        
        logger.info(f"Stops calculated - SL: {stop_loss:.4f}, TPs: {take_profits}")
        
        return result
        
    def _calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        swing_high: Optional[float],
        swing_low: Optional[float]
    ) -> float:
        """Calculate stop loss using configured method."""
        method = self.sl_config.get('method', 'conservative')
        atr_multiplier = self.sl_config.get('atr_multiplier', 2.0)
        structure_buffer_pips = self.sl_config.get('structure_buffer_pips', 2)
        
        # ATR-based stop
        if direction == 'long':
            atr_stop = entry_price - (atr * atr_multiplier)
        else:
            atr_stop = entry_price + (atr * atr_multiplier)
            
        # Structure-based stop
        structure_stop = None
        if direction == 'long' and swing_low is not None:
            # Place stop below swing low with buffer
            pip_value = entry_price * 0.0001  # Approximate pip value
            structure_stop = swing_low - (structure_buffer_pips * pip_value)
        elif direction == 'short' and swing_high is not None:
            pip_value = entry_price * 0.0001
            structure_stop = swing_high + (structure_buffer_pips * pip_value)
            
        # Choose stop based on method
        if method == 'atr':
            stop_loss = atr_stop
        elif method == 'structure' and structure_stop is not None:
            stop_loss = structure_stop
        else:  # conservative - use wider of the two
            if structure_stop is None:
                stop_loss = atr_stop
            else:
                if direction == 'long':
                    stop_loss = min(atr_stop, structure_stop)  # Wider stop
                else:
                    stop_loss = max(atr_stop, structure_stop)
                    
        return stop_loss
        
    def _calculate_take_profit_levels(
        self,
        entry_price: float,
        direction: str,
        risk: float
    ) -> List[Dict]:
        """Calculate take profit levels based on R:R ratios."""
        targets = self.tp_config.get('targets', [
            {'name': 'TP1', 'rr_ratio': 1.5, 'close_percent': 50},
            {'name': 'TP2', 'rr_ratio': 3.0, 'close_percent': 30},
            {'name': 'TP3', 'rr_ratio': 999, 'close_percent': 20}  # Trail remainder
        ])
        
        tp_levels = []
        
        for target in targets:
            rr_ratio = target['rr_ratio']
            
            if rr_ratio == 999:  # Trail indicator
                tp_price = None  # Will trail
            else:
                if direction == 'long':
                    tp_price = entry_price + (risk * rr_ratio)
                else:
                    tp_price = entry_price - (risk * rr_ratio)
                    
            tp_levels.append({
                'name': target['name'],
                'rr_ratio': rr_ratio,
                'price': tp_price,
                'close_percent': target['close_percent'],
                'hit': False
            })
            
        return tp_levels
        
    def update_trailing_stop(
        self,
        trade: Dict,
        current_price: float,
        atr: float,
        high_since_entry: float,
        low_since_entry: float
    ) -> Dict:
        """
        Update trailing stop if conditions are met.
        
        Args:
            trade: Current trade dictionary
            current_price: Current market price
            atr: Current ATR value
            high_since_entry: Highest price since entry
            low_since_entry: Lowest price since entry
            
        Returns:
            Update dictionary with new stop or None
        """
        if not self.trailing_config.get('enabled', True):
            return {'update_required': False}
            
        entry_price = trade['entry_price']
        current_sl = trade['stop_loss']
        direction = trade['direction']
        risk = abs(entry_price - current_sl)
        
        # Calculate current R:R
        if direction == 'long':
            current_rr = (current_price - entry_price) / risk if risk > 0 else 0
        else:
            current_rr = (entry_price - current_price) / risk if risk > 0 else 0
            
        # Check if trailing should be activated
        activation_rr = self.trailing_config.get('activation_rr', 1.0)
        
        if current_rr < activation_rr:
            # Not yet activated
            # But check for breakeven move
            return self._check_breakeven_move(trade, current_price, current_rr)
            
        # Trailing is active - calculate new stop
        trailing_method = self.trailing_config.get('method', 'atr')
        
        if trailing_method == 'atr':
            new_sl = self._calculate_atr_trailing_stop(
                direction,
                high_since_entry,
                low_since_entry,
                atr
            )
        else:  # percentage
            percentage = self.trailing_config.get('percentage', 0.5)
            new_sl = self._calculate_percentage_trailing_stop(
                direction,
                high_since_entry,
                low_since_entry,
                percentage
            )
            
        # Only move stop in favorable direction
        update_required = False
        if direction == 'long':
            if new_sl > current_sl:
                update_required = True
        else:
            if new_sl < current_sl:
                update_required = True
                
        if update_required:
            logger.info(
                f"Trailing stop update: {current_sl:.4f} -> {new_sl:.4f} "
                f"(R:R: {current_rr:.2f})"
            )
            
        return {
            'update_required': update_required,
            'new_stop_loss': new_sl if update_required else current_sl,
            'old_stop_loss': current_sl,
            'current_rr': current_rr,
            'trigger_reason': f'Trailing activated at {current_rr:.2f}R',
            'method': trailing_method
        }
        
    def _check_breakeven_move(
        self,
        trade: Dict,
        current_price: float,
        current_rr: float
    ) -> Dict:
        """Check if stop should be moved to breakeven."""
        breakeven_config = self.trailing_config.get('breakeven', {})
        if not breakeven_config.get('enabled', True):
            return {'update_required': False}
            
        trigger_rr = breakeven_config.get('trigger_rr', 1.0)
        buffer_pips = breakeven_config.get('buffer_pips', 1)
        
        if current_rr < trigger_rr:
            return {'update_required': False}
            
        # Move to breakeven + buffer
        entry_price = trade['entry_price']
        direction = trade['direction']
        current_sl = trade['stop_loss']
        
        pip_value = entry_price * 0.0001
        
        if direction == 'long':
            new_sl = entry_price + (buffer_pips * pip_value)
            update_required = new_sl > current_sl
        else:
            new_sl = entry_price - (buffer_pips * pip_value)
            update_required = new_sl < current_sl
            
        if update_required:
            logger.info(
                f"Moving stop to breakeven+{buffer_pips}: "
                f"{current_sl:.4f} -> {new_sl:.4f} (R:R: {current_rr:.2f})"
            )
            
        return {
            'update_required': update_required,
            'new_stop_loss': new_sl if update_required else current_sl,
            'old_stop_loss': current_sl,
            'current_rr': current_rr,
            'trigger_reason': f'Breakeven move at {current_rr:.2f}R',
            'method': 'breakeven'
        }
        
    def _calculate_atr_trailing_stop(
        self,
        direction: str,
        high: float,
        low: float,
        atr: float
    ) -> float:
        """Calculate ATR-based trailing stop."""
        multiplier = self.trailing_config.get('atr_multiplier', 1.5)
        
        if direction == 'long':
            # Trail below recent high
            return high - (atr * multiplier)
        else:
            # Trail above recent low
            return low + (atr * multiplier)
            
    def _calculate_percentage_trailing_stop(
        self,
        direction: str,
        high: float,
        low: float,
        percentage: float
    ) -> float:
        """Calculate percentage-based trailing stop."""
        if direction == 'long':
            return high * (1 - percentage / 100)
        else:
            return low * (1 + percentage / 100)
            
    def check_take_profit_hit(
        self,
        trade: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """
        Check if any take profit level has been hit.
        
        Args:
            trade: Trade dictionary
            current_price: Current market price
            
        Returns:
            TP level hit info or None
        """
        direction = trade['direction']
        tp_levels = trade.get('take_profit_levels', [])
        
        for tp in tp_levels:
            if tp.get('hit', False):
                continue  # Already hit
                
            tp_price = tp.get('price')
            if tp_price is None:
                continue  # Trailing target
                
            hit = False
            if direction == 'long':
                hit = current_price >= tp_price
            else:
                hit = current_price <= tp_price
                
            if hit:
                logger.info(
                    f"Take profit {tp['name']} hit @ {tp_price:.4f} "
                    f"(Close {tp['close_percent']}%)"
                )
                return {
                    'tp_hit': True,
                    'tp_name': tp['name'],
                    'tp_price': tp_price,
                    'close_percent': tp['close_percent'],
                    'rr_ratio': tp['rr_ratio']
                }
                
        return None
        
    def calculate_partial_close(
        self,
        trade: Dict,
        tp_hit: Dict
    ) -> Dict:
        """
        Calculate partial position close details.
        
        Args:
            trade: Trade dictionary
            tp_hit: TP hit information
            
        Returns:
            Partial close instructions
        """
        total_size = trade['position_size']
        remaining_size = trade.get('remaining_size', total_size)
        close_percent = tp_hit['close_percent']
        
        close_size = remaining_size * (close_percent / 100)
        new_remaining = remaining_size - close_size
        
        entry_price = trade['entry_price']
        exit_price = tp_hit['tp_price']
        direction = trade['direction']
        
        # Calculate P&L for partial
        if direction == 'long':
            pnl = (exit_price - entry_price) * close_size
        else:
            pnl = (entry_price - exit_price) * close_size
            
        return {
            'close_size': close_size,
            'remaining_size': new_remaining,
            'exit_price': exit_price,
            'partial_pnl': pnl,
            'tp_name': tp_hit['tp_name'],
            'remaining_percent': (new_remaining / total_size) * 100
        }
        
    def manage_long_runner(
        self,
        trade: Dict,
        current_price: float,
        atr: float,
        price_structure: Dict
    ) -> Dict:
        """
        Manage long runner position - extend TP or continue trailing.
        
        Args:
            trade: Trade dictionary
            current_price: Current price
            atr: Current ATR
            price_structure: Current price structure analysis
            
        Returns:
            Long runner management decision
        """
        if not self.long_runner_config.get('enabled', True):
            return {'action': 'none'}
            
        direction = trade['direction']
        entry_price = trade['entry_price']
        
        # Check if making new structural highs/lows
        structure_type = price_structure.get('structure')
        
        extend_tp = False
        if direction == 'long' and structure_type == 'uptrend':
            # Check for new higher high
            last_high = price_structure.get('last_swing_high')
            if last_high and current_price > last_high:
                extend_tp = True
        elif direction == 'short' and structure_type == 'downtrend':
            # Check for new lower low
            last_low = price_structure.get('last_swing_low')
            if last_low and current_price < last_low:
                extend_tp = True
                
        if extend_tp:
            # Continue trailing with wider distance
            max_trail_atr = self.long_runner_config.get('max_trail_distance_atr', 3.0)
            
            if direction == 'long':
                new_sl = current_price - (atr * max_trail_atr)
            else:
                new_sl = current_price + (atr * max_trail_atr)
                
            logger.info(
                f"Long runner extension: New structural extreme detected, "
                f"trailing with {max_trail_atr}x ATR"
            )
            
            return {
                'action': 'extend_trail',
                'new_stop_loss': new_sl,
                'reason': 'New structural extreme',
                'atr_multiplier': max_trail_atr
            }
            
        return {'action': 'none'}
        
    def calculate_realized_rr(self, trade: Dict, exit_price: float) -> float:
        """Calculate realized R:R ratio."""
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        direction = trade['direction']
        
        risk = abs(entry_price - stop_loss)
        
        if direction == 'long':
            reward = exit_price - entry_price
        else:
            reward = entry_price - exit_price
            
        return reward / risk if risk > 0 else 0


# Example usage
if __name__ == "__main__":
    config = {
        'risk_management': {
            'stop_loss': {
                'method': 'conservative',
                'atr_multiplier': 2.0,
                'structure_buffer_pips': 2
            },
            'take_profit': {
                'targets': [
                    {'name': 'TP1', 'rr_ratio': 1.5, 'close_percent': 50},
                    {'name': 'TP2', 'rr_ratio': 3.0, 'close_percent': 30},
                    {'name': 'TP3', 'rr_ratio': 999, 'close_percent': 20}
                ]
            },
            'trailing_stop': {
                'enabled': True,
                'activation_rr': 1.0,
                'method': 'atr',
                'atr_multiplier': 1.5,
                'breakeven': {
                    'enabled': True,
                    'trigger_rr': 1.0,
                    'buffer_pips': 1
                }
            },
            'long_runner': {
                'enabled': True,
                'max_trail_distance_atr': 3.0
            }
        }
    }
    
    manager = StopManager(config)
    
    # Test initial stops calculation
    print("=== Initial Stops Test ===")
    stops = manager.calculate_initial_stops(
        entry_price=50000,
        direction='long',
        atr=250,
        swing_high=None,
        swing_low=49200
    )
    
    print(f"Stop Loss: {stops['stop_loss']:.2f}")
    print(f"Risk Distance: {stops['risk_distance']:.2f}")
    print(f"Take Profit Levels:")
    for tp in stops['take_profit_levels']:
        print(f"  {tp['name']}: {tp['price']:.2f if tp['price'] else 'Trail'} "
              f"({tp['rr_ratio']}R, close {tp['close_percent']}%)")
    
    # Test breakeven move
    print("\n=== Breakeven Move Test ===")
    trade = {
        'entry_price': 50000,
        'stop_loss': 49500,
        'direction': 'long',
        'position_size': 0.1
    }
    
    update = manager.update_trailing_stop(
        trade=trade,
        current_price=50600,  # 1.2 R
        atr=250,
        high_since_entry=50700,
        low_since_entry=49900
    )
    
    print(f"Update Required: {update['update_required']}")
    if update['update_required']:
        print(f"Old SL: {update['old_stop_loss']:.2f}")
        print(f"New SL: {update['new_stop_loss']:.2f}")
        print(f"Reason: {update['trigger_reason']}")
    
    # Test TP hit
    print("\n=== Take Profit Check Test ===")
    trade['take_profit_levels'] = stops['take_profit_levels']
    
    tp_hit = manager.check_take_profit_hit(trade, 50750)  # TP1 price
    
    if tp_hit:
        print(f"TP Hit: {tp_hit['tp_name']} @ {tp_hit['tp_price']:.2f}")
        
        partial = manager.calculate_partial_close(trade, tp_hit)
        print(f"Close Size: {partial['close_size']:.4f}")
        print(f"Remaining: {partial['remaining_size']:.4f}")
        print(f"Partial PnL: ${partial['partial_pnl']:.2f}")
    
    # Test trailing stop activation
    print("\n=== Trailing Stop Test ===")
    trade['stop_loss'] = 50010  # After breakeven move
    
    update = manager.update_trailing_stop(
        trade=trade,
        current_price=51500,  # 3R
        atr=250,
        high_since_entry=51500,
        low_since_entry=49900
    )
    
    print(f"Update Required: {update['update_required']}")
    if update['update_required']:
        print(f"Old SL: {update['old_stop_loss']:.2f}")
        print(f"New SL: {update['new_stop_loss']:.2f}")
        print(f"Current R:R: {update['current_rr']:.2f}")
        print(f"Method: {update['method']}")
    
    print("\nStop manager test completed!")