"""
Unit tests for risk management components.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from risk_management.money_manager import MoneyManager
from risk_management.stop_manager import StopManager


@pytest.fixture
def money_manager_config():
    """Create money manager configuration."""
    return {
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


@pytest.fixture
def stop_manager_config():
    """Create stop manager configuration."""
    return {
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


class TestMoneyManager:
    """Test cases for MoneyManager."""
    
    def test_initialization(self, money_manager_config):
        """Test money manager initialization."""
        mm = MoneyManager(money_manager_config)
        assert mm.max_risk_percent == 1.0
        assert mm.use_dynamic_sizing is True
    
    def test_position_size_calculation(self, money_manager_config):
        """Test position size calculation."""
        mm = MoneyManager(money_manager_config)
        
        result = mm.calculate_position_size(
            account_equity=10000,
            entry_price=50000,
            stop_loss=49500,
            symbol='BTC/USDT',
            direction='long'
        )
        
        # Check result structure
        assert 'position_size' in result
        assert 'position_value' in result
        assert 'risk_amount' in result
        assert 'risk_percent' in result
        assert 'approved' in result
        
        # Check values are reasonable
        assert result['position_size'] > 0
        assert result['risk_percent'] <= 1.0
        assert result['approved'] is True
        
        # Risk amount should be ~1% of equity
        expected_risk = 10000 * 0.01
        assert abs(result['risk_amount'] - expected_risk) < 10
    
    def test_zero_risk_distance(self, money_manager_config):
        """Test handling of zero risk distance."""
        mm = MoneyManager(money_manager_config)
        
        result = mm.calculate_position_size(
            account_equity=10000,
            entry_price=50000,
            stop_loss=50000,  # Same as entry
            symbol='BTC/USDT',
            direction='long'
        )
        
        assert result['position_size'] == 0
        assert result['approved'] is False
    
    def test_daily_limit_check(self, money_manager_config):
        """Test daily drawdown limit check."""
        mm = MoneyManager(money_manager_config)
        
        # Within limits
        result = mm.check_daily_limits({
            'daily_drawdown_percent': 3.0,
            'trades_today': 5
        })
        
        assert result['limits_ok'] is True
        
        # Exceeding drawdown limit
        result = mm.check_daily_limits({
            'daily_drawdown_percent': 6.0,
            'trades_today': 5
        })
        
        assert result['limits_ok'] is False
        assert len(result['reasons']) > 0
    
    def test_consecutive_losses_check(self, money_manager_config):
        """Test consecutive losses cooldown."""
        mm = MoneyManager(money_manager_config)
        
        # No cooldown needed
        recent_trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 75}
        ]
        
        result = mm.check_consecutive_losses(recent_trades)
        assert result['cooldown_active'] is False
        
        # Cooldown triggered
        recent_trades = [
            {'pnl': -100},
            {'pnl': -50},
            {'pnl': -75}
        ]
        
        result = mm.check_consecutive_losses(recent_trades)
        assert result['cooldown_active'] is True
        assert result['consecutive_losses'] == 3
    
    def test_portfolio_risk_calculation(self, money_manager_config):
        """Test portfolio risk calculation."""
        mm = MoneyManager(money_manager_config)
        
        open_positions = [
            {
                'symbol': 'BTC/USDT',
                'risk_amount': 100,
                'position_value': 1000
            },
            {
                'symbol': 'ETH/USDT',
                'risk_amount': 80,
                'position_value': 800
            }
        ]
        
        result = mm.calculate_portfolio_risk(open_positions)
        
        assert result['total_risk'] == 180
        assert result['total_value'] == 1800
        assert result['num_positions'] == 2
        assert 'BTC/USDT' in result['symbols']
        assert 'ETH/USDT' in result['symbols']
    
    def test_validate_trade_full(self, money_manager_config):
        """Test complete trade validation."""
        mm = MoneyManager(money_manager_config)
        
        result = mm.validate_trade(
            account_equity=10000,
            entry_price=50000,
            stop_loss=49500,
            symbol='BTC/USDT',
            direction='long',
            daily_stats={
                'daily_drawdown_percent': 2.0,
                'trades_today': 3
            },
            recent_trades=[
                {'pnl': 100},
                {'pnl': -50}
            ]
        )
        
        assert 'approved' in result
        assert 'position_size' in result or 'reason' in result


class TestStopManager:
    """Test cases for StopManager."""
    
    def test_initialization(self, stop_manager_config):
        """Test stop manager initialization."""
        sm = StopManager(stop_manager_config)
        assert sm.sl_config['method'] == 'conservative'
        assert sm.trailing_config['enabled'] is True
    
    def test_calculate_initial_stops(self, stop_manager_config):
        """Test initial stop loss and take profit calculation."""
        sm = StopManager(stop_manager_config)
        
        stops = sm.calculate_initial_stops(
            entry_price=50000,
            direction='long',
            atr=250,
            swing_high=None,
            swing_low=49200
        )
        
        # Check structure
        assert 'stop_loss' in stops
        assert 'take_profit_levels' in stops
        assert 'risk_distance' in stops
        
        # Stop loss should be below entry for long
        assert stops['stop_loss'] < 50000
        
        # Should have multiple TP levels
        assert len(stops['take_profit_levels']) >= 2
        
        # Risk distance should be positive
        assert stops['risk_distance'] > 0
    
    def test_stop_loss_methods(self, stop_manager_config):
        """Test different stop loss calculation methods."""
        sm = StopManager(stop_manager_config)
        
        # ATR-based
        sm.sl_config['method'] = 'atr'
        stops_atr = sm.calculate_initial_stops(
            entry_price=50000,
            direction='long',
            atr=250,
            swing_low=49200
        )
        
        # Structure-based
        sm.sl_config['method'] = 'structure'
        stops_structure = sm.calculate_initial_stops(
            entry_price=50000,
            direction='long',
            atr=250,
            swing_low=49200
        )
        
        # Both should return valid stops
        assert stops_atr['stop_loss'] > 0
        assert stops_structure['stop_loss'] > 0
    
    def test_take_profit_levels(self, stop_manager_config):
        """Test take profit level calculation."""
        sm = StopManager(stop_manager_config)
        
        stops = sm.calculate_initial_stops(
            entry_price=50000,
            direction='long',
            atr=250,
            swing_low=49500
        )
        
        tp_levels = stops['take_profit_levels']
        
        # Check structure of TP levels
        for tp in tp_levels:
            assert 'name' in tp
            assert 'rr_ratio' in tp
            assert 'close_percent' in tp
            
            # Price should be above entry for long
            if tp['price'] is not None:
                assert tp['price'] > 50000
    
    def test_breakeven_move(self, stop_manager_config):
        """Test breakeven stop movement."""
        sm = StopManager(stop_manager_config)
        
        trade = {
            'entry_price': 50000,
            'stop_loss': 49500,
            'direction': 'long',
            'position_size': 0.1
        }
        
        # At 1:1 R:R, should move to breakeven
        update = sm.update_trailing_stop(
            trade=trade,
            current_price=50500,  # 1R gained
            atr=250,
            high_since_entry=50500,
            low_since_entry=49900
        )
        
        if update.get('update_required'):
            # Should move stop above entry (either breakeven or trailing)
            assert update['new_stop_loss'] >= 50000
            assert update['method'] in ['breakeven', 'atr']
    
    def test_trailing_stop_activation(self, stop_manager_config):
        """Test trailing stop activation."""
        sm = StopManager(stop_manager_config)
        
        trade = {
            'entry_price': 50000,
            'stop_loss': 50010,  # Already at breakeven
            'direction': 'long',
            'position_size': 0.1
        }
        
        # Well past 1:1 R:R, should activate trailing
        update = sm.update_trailing_stop(
            trade=trade,
            current_price=51500,  # 3R gained
            atr=250,
            high_since_entry=51500,
            low_since_entry=49900
        )
        
        # Should update stop
        assert update.get('current_rr', 0) > 1.0
    
    def test_tp_hit_detection(self, stop_manager_config):
        """Test take profit hit detection."""
        sm = StopManager(stop_manager_config)
        
        # Create trade with TP levels
        trade = {
            'entry_price': 50000,
            'direction': 'long',
            'take_profit_levels': [
                {'name': 'TP1', 'price': 50750, 'close_percent': 50, 'rr_ratio': 1.5, 'hit': False},
                {'name': 'TP2', 'price': 51500, 'close_percent': 30, 'rr_ratio': 3.0, 'hit': False}
            ]
        }
        
        # Price hits TP1
        tp_hit = sm.check_take_profit_hit(trade, 50800)
        
        assert tp_hit is not None
        assert tp_hit['tp_hit'] is True
        assert tp_hit['tp_name'] == 'TP1'
    
    def test_partial_close_calculation(self, stop_manager_config):
        """Test partial position close calculation."""
        sm = StopManager(stop_manager_config)
        
        trade = {
            'entry_price': 50000,
            'position_size': 1.0,
            'remaining_size': 1.0,
            'direction': 'long'
        }
        
        tp_hit = {
            'tp_name': 'TP1',
            'tp_price': 50750,
            'close_percent': 50,
            'rr_ratio': 1.5
        }
        
        result = sm.calculate_partial_close(trade, tp_hit)
        
        assert result['close_size'] == 0.5  # 50% of 1.0
        assert result['remaining_size'] == 0.5
        assert result['partial_pnl'] > 0
    
    def test_realized_rr_calculation(self, stop_manager_config):
        """Test realized R:R ratio calculation."""
        sm = StopManager(stop_manager_config)
        
        trade = {
            'entry_price': 50000,
            'stop_loss': 49500,
            'direction': 'long'
        }
        
        # Exit at 2R
        exit_price = 51000
        rr = sm.calculate_realized_rr(trade, exit_price)
        
        # Risk = 500, Reward = 1000, R:R = 2.0
        assert abs(rr - 2.0) < 0.1
    
    def test_short_trade_stops(self, stop_manager_config):
        """Test stops for short trades."""
        sm = StopManager(stop_manager_config)
        
        stops = sm.calculate_initial_stops(
            entry_price=50000,
            direction='short',
            atr=250,
            swing_high=50800
        )
        
        # Stop loss should be above entry for short
        assert stops['stop_loss'] > 50000
        
        # TP levels should be below entry for short
        for tp in stops['take_profit_levels']:
            if tp['price'] is not None:
                assert tp['price'] < 50000


class TestRiskIntegration:
    """Integration tests for risk management components."""
    
    def test_full_trade_workflow(self, money_manager_config, stop_manager_config):
        """Test complete trade workflow with both managers."""
        mm = MoneyManager(money_manager_config)
        sm = StopManager(stop_manager_config)
        
        # 1. Calculate position size
        sizing = mm.calculate_position_size(
            account_equity=10000,
            entry_price=50000,
            stop_loss=49500,
            symbol='BTC/USDT',
            direction='long'
        )
        
        assert sizing['approved']
        
        # 2. Calculate stops
        stops = sm.calculate_initial_stops(
            entry_price=50000,
            direction='long',
            atr=250,
            swing_low=49200
        )
        
        assert stops['stop_loss'] < 50000
        
        # 3. Check TP hit
        trade = {
            'entry_price': 50000,
            'direction': 'long',
            'take_profit_levels': stops['take_profit_levels'],
            'position_size': sizing['position_size']
        }
        
        tp_hit = sm.check_take_profit_hit(trade, 50750)
        
        if tp_hit:
            # 4. Calculate partial close
            partial = sm.calculate_partial_close(trade, tp_hit)
            assert partial['close_size'] > 0
            assert partial['remaining_size'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])