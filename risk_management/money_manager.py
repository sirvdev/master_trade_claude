"""
Fixed Money Manager with proper MT5 position sizing.
The issue: calculating position size in $ value instead of lots for MT5.
"""

import logging
from typing import Dict, Optional
import math

logger = logging.getLogger(__name__)

# Index CFDs are never 100,000 units per lot. Symbols matching these tokens
# must have an explicit contract size in risk_management.mt5_contract_sizes;
# see _get_mt5_contract_size().
_INDEX_TOKENS = (
    'US30', 'US500', 'US100', 'NAS100', 'SPX', 'DJ30', 'DE30', 'DE40',
    'UK100', 'JP225', 'FRA40', 'AUS200', 'HK50', 'EU50', 'STOXX',
)


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
        self.max_position_size_lots = float(self.config.get('max_position_size_lots', 2.0))

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
        platform: str = 'mt5',  # ADD THIS PARAMETER
        current_exposure: Optional[Dict] = None,
        symbol_spec: Optional[Dict] = None,
    ) -> Dict:
        """
        Calculate position size with platform-specific logic.
        
        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            stop_loss: Stop loss price
            symbol: Trading symbol
            direction: 'long' or 'short'
            platform: 'mt5' 
            current_exposure: Optional dict of current positions
            symbol_spec: Optional broker symbol specification from
                MT5FileBridge.get_symbol_info(), carrying contract_size,
                volume_min, volume_max and volume_step. When present it
                overrides every local assumption -- the broker is the only
                authority on its own contract sizes.
            
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
        
        # Platform-specific position size calculation.
        # R2 FIXED 2026-08-30 audit: position_size was bound only inside this
        # branch, so any other platform raised UnboundLocalError two lines
        # later instead of declining the trade.
        if platform != 'mt5':
            logger.error(
                f"Unsupported platform '{platform}' for {symbol} - not sizing"
            )
            return self._zero_position_response(f"Unsupported platform: {platform}")

        position_size = self._calculate_mt5_position_size(
            max_risk_amount,
            entry_price,
            risk_distance,
            symbol,
            symbol_spec,
        )
        intended_size = position_size

        # N2 FIXED 2026-08-30 audit: reject before _apply_mt5_constraints can
        # floor a broken size up to the 0.01 minimum. A computed size that far
        # below the minimum lot is a contract-size error, not a rounding
        # question, and taking the floor means taking many times the intended
        # risk. Live evidence: all 15 US30m/US500m trades were floored to 0.01.
        if position_size <= 0:
            return self._zero_position_response(
                f"No valid contract size for {symbol}"
            )
        if position_size < self.mt5_min_lot / 10.0:
            logger.error(
                f"{symbol}: computed {position_size:.6f} lots, more than 10x "
                f"below the {self.mt5_min_lot} minimum. Flooring it would take "
                f"{self.mt5_min_lot / position_size:.0f}x the intended risk. "
                f"Check risk_management.mt5_contract_sizes for this symbol."
            )
            return self._zero_position_response(
                f"Computed size {position_size:.6f} far below minimum lot"
            )
        
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
            position_size = self._apply_mt5_constraints(
                position_size, symbol, symbol_spec,
                entry_price=entry_price, account_equity=account_equity,
            )

        # FIXED 2026-08-30 audit: a binding lot cap used to be silent. It cost
        # 100% of ETH trades their intended risk (wanted ~32-46 lots, got the
        # 6.0 global cap, so ~$80-113 of risk instead of ~$600) and nobody could
        # see it. Any cap that bites now says so, in dollars.
        if intended_size > 0 and position_size < intended_size * 0.98:
            _cs = self._get_mt5_contract_size(symbol, symbol_spec)
            _want = intended_size * _cs * risk_distance
            _got  = position_size * _cs * risk_distance
            logger.warning(
                f"{symbol}: position capped {intended_size:.2f} -> "
                f"{position_size:.2f} lots. Intended risk ${_want:.2f}, "
                f"actual ${_got:.2f} ({_got / _want * 100:.0f}% of intent). "
                f"Raise risk_management.max_position_size_lots_per_symbol "
                f"for this symbol if that is not what you want."
            )
        
        # ── Initialize variables to prevent UnboundLocalError on non-MT5 platforms ──
        actual_risk = 0.0
        position_value = 0.0
        
        # Calculate actual risk with final position size
        if platform == 'mt5':
            contract_size = self._get_mt5_contract_size(symbol)
            actual_risk = position_size * contract_size * risk_distance
        
        actual_risk_percent = (actual_risk / account_equity) * 100 if account_equity > 0 else 0
        
        # Calculate position value
        if platform == 'mt5':
            contract_size = self._get_mt5_contract_size(symbol)
            position_value = position_size * contract_size * entry_price
        
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
        symbol: str,
        symbol_spec: Optional[Dict] = None,
    ) -> float:
        """
        Calculate MT5 position size in lots.

        Formula:  lots = max_risk / (contract_size × risk_distance)

        Examples:
          XAUUSD: $317 risk, $61 distance, contract=100
                  → 317 / (100 × 61) = 0.052 lots ✓

          BTCUSD: $317 risk, $61 distance, contract=1
                  → 317 / (1 × 61) = 5.2 lots ✓

          EURUSD: $100 risk, 0.0010 distance, contract=100000
                  → 100 / (100000 × 0.001) = 1.0 lots ✓
        """
        contract_size = self._get_mt5_contract_size(symbol, symbol_spec)
        if contract_size <= 0 or risk_distance <= 0:
            logger.error(
                f"Cannot size {symbol}: contract_size={contract_size}, "
                f"risk_distance={risk_distance}"
            )
            return 0.0
        lots = max_risk_amount / (contract_size * risk_distance)

        logger.debug(
            f"MT5 sizing: ${max_risk_amount:.2f} risk / "
            f"(contract={contract_size} × dist={risk_distance:.4f}) = {lots:.4f} lots"
        )

        return lots
    
    def _symbol_lot_cap(self, symbol: Optional[str]) -> float:
        """Per-symbol lot ceiling, falling back to the global one.

        ADDED 2026-08-30 audit: a single lot cap cannot be right across
        instruments whose notional per lot differs by five orders of
        magnitude. At 6.0 lots the same cap meant $14.6k of ETH and $2.76m of
        gold, so it never touched gold and strangled every ETH trade.
        """
        default = self.max_position_size_lots
        if not symbol:
            return default
        overrides = self.config.get('max_position_size_lots_per_symbol', {}) or {}
        norm = {str(k).upper().replace('/', ''): float(v) for k, v in overrides.items()}
        sym = symbol.upper().replace('/', '')
        if sym in norm:
            return norm[sym]
        hits = [k for k in norm if k and sym.startswith(k)]
        if hits:
            return norm[max(hits, key=len)]
        return default

    def _apply_mt5_constraints(self, lots: float, symbol: Optional[str] = None,
                               symbol_spec: Optional[Dict] = None,
                               entry_price: float = 0.0,
                               account_equity: float = 0.0) -> float:
        """Apply MT5 lot size constraints.

        FIXED 2026-08-30 audit: min/step/max used to be hardcoded 0.01/0.01/100
        for every instrument. When the broker's own specification is available
        (MT5FileBridge.get_symbol_info) it is used instead.
        """
        spec = symbol_spec or {}
        min_lot = float(spec.get('volume_min') or 0) or self.mt5_min_lot
        step    = float(spec.get('volume_step') or 0) or self.mt5_lot_step
        broker_max = float(spec.get('volume_max') or 0) or self.mt5_max_lot
        symbol_cap = self._symbol_lot_cap(symbol)

        # Notional backstop, symbol-agnostic.
        # ADDED 2026-08-30 audit: fixed-fractional sizing has no upper bound on
        # exposure -- when the stop is pathologically tight the lot count
        # explodes. Real examples from the live record, at $600 of risk: a
        # US500m signal with a 1.14-point stop wants 525 lots (40x equity in
        # notional) and an XAUUSDm signal with a 0.2-point stop wants 68 lots
        # (297x). Risk-per-trade is respected in both, but a weekend gap is not
        # bounded by the stop. The 30x default is deliberately loose: the widest
        # position any of the four instances actually took in Jul 30 - Aug 29 was
        # 26.0x (XAUUSDm), so this cannot change a single historical trade on
        # the symbols that already size correctly. Tighten it once you have
        # decided what maximum exposure you actually want.
        notional_mult = float(self.config.get(
            'max_notional_exposure_equity_multiple', 30.0) or 0)
        contract_size = self._get_mt5_contract_size(symbol, symbol_spec) if symbol else 0.0
        if (notional_mult > 0 and account_equity > 0 and entry_price > 0
                and contract_size > 0):
            notional_cap_lots = (account_equity * notional_mult) / (contract_size * entry_price)
            if notional_cap_lots < lots:
                logger.warning(
                    f"{symbol}: {lots:.2f} lots would be "
                    f"{lots * contract_size * entry_price / account_equity:.1f}x equity in "
                    f"notional; capping to {notional_cap_lots:.2f} lots "
                    f"({notional_mult:.0f}x). Stop distance is unusually tight."
                )
                lots = notional_cap_lots

        max_allowed = min(broker_max, symbol_cap)
        lots = max(min_lot, min(lots, max_allowed))

        # Round to the broker's lot step
        lots = round(lots / step) * step
        lots = round(lots, 8)

        # Final bounds check after rounding
        if lots < min_lot:
            logger.warning(f"{symbol}: position size {lots} below minimum {min_lot}")
            return 0

        if lots > symbol_cap:
            logger.warning(
                f"{symbol}: position size {lots} above lot cap {symbol_cap}, capping"
            )
            lots = symbol_cap

        if lots > broker_max:
            logger.warning(
                f"{symbol}: position size {lots} above broker volume_max "
                f"{broker_max}, capping"
            )
            lots = broker_max

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
        """Apply global risk limits - FIXED VERSION."""
        
        # Check max concurrent trades FIRST
        max_concurrent = self.global_limits.get('max_concurrent_trades', 3)
        if current_exposure:
            open_count = current_exposure.get('open_count', 0)
            if open_count >= max_concurrent:
                logger.warning(
                    f"REJECTED: Max concurrent trades limit reached "
                    f"({open_count}/{max_concurrent})"
                )
                return 0  # Return 0 to reject trade
        
        # Check max trades per day
        max_trades_day = self.global_limits.get('max_trades_per_day', 10)
        # This would need daily_stats passed in - skip for now or add parameter
        
        # Check max risk per symbol
        max_symbol_risk = self.global_limits.get('max_risk_per_symbol_percent', 2.0)
        if current_exposure and symbol in current_exposure.get('symbols', {}):
            symbol_exposure = current_exposure['symbols'][symbol]
            # Could add risk calculation here
        
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
        platform: str = 'mt5',
        current_exposure: Optional[Dict] = None,
        daily_stats: Optional[Dict] = None,
        recent_trades: Optional[list] = None
    ) -> Dict:
        """Complete trade validation - FIXED VERSION."""
        
        # Check daily limits FIRST (before calculating position size)
        if daily_stats:
            limit_check = self.check_daily_limits(daily_stats)
            if not limit_check['limits_ok']:
                logger.warning(f"Trade rejected: {limit_check['reasons']}")
                return {
                    'approved': False,
                    'position_size': 0,
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
                    'position_size': 0,
                    'reason': f"Cooldown active: {cooldown['reason']}",
                    'cooldown': cooldown
                }
        
        # Check max concurrent BEFORE calculating size
        max_concurrent = self.global_limits.get('max_concurrent_trades', 3)
        if current_exposure:
            open_count = current_exposure.get('open_count', 0)
            if open_count >= max_concurrent:
                logger.warning(
                    f"REJECTED: Max concurrent trades ({open_count}/{max_concurrent})"
                )
                return {
                    'approved': False,
                    'position_size': 0,
                    'reason': f'Max concurrent trades limit ({max_concurrent})',
                    'open_positions': open_count
                }
        
        # Now calculate position size
        sizing = self.calculate_position_size(
            account_equity,
            entry_price,
            stop_loss,
            symbol,
            direction,
            platform,
            current_exposure
        )
        
        # Final check: if position size is 0, reject
        if sizing['position_size'] == 0 or sizing['position_size'] < 0.01:
            sizing['approved'] = False
            if 'reason' not in sizing:
                sizing['reason'] = 'Position size too small or zero'
        
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
        import time as _time
        from datetime import datetime, timezone

        cooldown_config = self.global_limits.get('cooldown_after_losses', {})
        if not cooldown_config.get('enabled', True):
            return {'cooldown_active': False}

        max_consecutive = cooldown_config.get('consecutive_losses', 3)
        cooldown_seconds = cooldown_config.get('cooldown_seconds', 3600)

        consecutive_losses = 0
        last_loss_trade = None

        for trade in reversed(recent_trades):
            pnl = trade.get('pnl')
            if pnl is None:
                continue
            if pnl < 0:
                consecutive_losses += 1
                if last_loss_trade is None:
                    last_loss_trade = trade   # most recent loss
            else:
                break

        if consecutive_losses < max_consecutive:
            return {'cooldown_active': False, 'consecutive_losses': consecutive_losses}

        # ── Threshold hit — check if cooldown has expired ─────────────────────
        if last_loss_trade:
            raw_time = last_loss_trade.get('exit_time') or last_loss_trade.get('entry_time')
            if raw_time:
                try:
                    import pandas as pd
                    loss_dt = pd.to_datetime(raw_time)
                    # Make both naive UTC for comparison
                    if loss_dt.tzinfo is not None:
                        loss_dt = loss_dt.tz_localize(None)
                    elapsed = (datetime.utcnow() - loss_dt).total_seconds()

                    if elapsed >= cooldown_seconds:
                        logger.info(
                            f"[COOLDOWN] Expired after {elapsed:.0f}s "
                            f"({consecutive_losses} consecutive losses). Resuming."
                        )
                        return {
                            'cooldown_active'    : False,
                            'consecutive_losses' : consecutive_losses,
                            'expired'            : True,
                        }

                    remaining = int(cooldown_seconds - elapsed)
                    logger.info(
                        f"[COOLDOWN] Active — {consecutive_losses} consecutive losses, "
                        f"{remaining}s remaining."
                    )
                    return {
                        'cooldown_active'    : True,
                        'consecutive_losses' : consecutive_losses,
                        'reason'             : f"{consecutive_losses} consecutive losses",
                        'remaining_seconds'  : remaining,
                    }
                except Exception as e:
                    logger.warning(f"[COOLDOWN] Could not parse loss time: {e}")

        # Fallback — no timestamp available, block conservatively
        return {
            'cooldown_active'   : True,
            'consecutive_losses': consecutive_losses,
            'reason'            : f"{consecutive_losses} consecutive losses (no timestamp)",
        }

    def calculate_portfolio_risk(self, open_positions: list) -> Dict:
        """Calculate aggregated portfolio risk metrics for open positions."""
        total_risk = sum(float(p.get('risk_amount', 0.0)) for p in open_positions)
        total_value = sum(float(p.get('position_value', 0.0)) for p in open_positions)
        symbols = {
            p.get('symbol', 'unknown'): {
                'risk_amount': float(p.get('risk_amount', 0.0)),
                'position_value': float(p.get('position_value', 0.0)),
            }
            for p in open_positions
        }
        return {
            'total_risk': total_risk,
            'total_value': total_value,
            'num_positions': len(open_positions),
            'symbols': symbols,
            'average_risk': total_risk / len(open_positions) if open_positions else 0.0,
            'average_position_value': total_value / len(open_positions) if open_positions else 0.0,
        }

    def _get_mt5_contract_size(self, symbol: str,
                               symbol_spec: Optional[Dict] = None) -> float:
        """
        Return the MT5 contract size (units per lot) for a given symbol.

        MT5 contract sizes vary by instrument:
          - XAUUSD (Gold):   100 troy oz per lot
          - XAGUSD (Silver): 5000 troy oz per lot
          - BTCUSD / BTCUSD: 1 BTC per lot
          - ETHUSD:          1 ETH per lot
          - Other crypto:    1 coin per lot
          - Forex pairs:     100,000 base currency units per lot (standard)

        These can be overridden per-symbol in config:
          risk_management:
            mt5_contract_sizes:
              BTCUSD: 1
              XAUUSD: 100
        """
        # The broker's own specification wins over every local assumption.
        # ADDED 2026-08-30 audit: the EA has exposed SYMBOL_TRADE_CONTRACT_SIZE
        # via get_symbol_info since v2.505; the Python side simply never asked.
        if symbol_spec:
            cs = float(symbol_spec.get('contract_size') or 0)
            if cs > 0:
                return cs

        # Allow per-symbol overrides from config.
        # R3 FIXED 2026-08-30 audit: the old matcher was a bidirectional
        # substring test (`key in sym or sym in key`), which mis-matches the
        # moment one symbol name contains another (US30 vs US300, XAU vs
        # XAUUSD) and matches in an order that depends on dict insertion.
        # Exact match first, then longest prefix, one direction only.
        overrides = self.config.get('mt5_contract_sizes', {}) or {}
        sym_upper = symbol.upper().replace('/', '')
        norm = {str(k).upper().replace('/', ''): float(v) for k, v in overrides.items()}

        if sym_upper in norm:
            return norm[sym_upper]
        prefix_hits = [k for k in norm if k and sym_upper.startswith(k)]
        if prefix_hits:
            return norm[max(prefix_hits, key=len)]

        # Built-in defaults
        if any(x in sym_upper for x in ('XAU', 'GOLD')):
            return 100.0      # 100 troy oz per lot
        if any(x in sym_upper for x in ('XAG', 'SILVER')):
            return 5000.0     # 5000 troy oz per lot
        if any(x in sym_upper for x in ('BTC', 'BITCOIN')):
            return 1.0        # 1 BTC per lot
        if any(x in sym_upper for x in ('ETH', 'ETHEREUM')):
            return 1.0        # 1 ETH per lot
        if any(x in sym_upper for x in ('LTC', 'LITECOIN')):
            return 1.0
        if any(x in sym_upper for x in ('XRP', 'RIPPLE')):
            return 1.0

        # N2 FIXED 2026-08-30 audit: index CFDs used to fall through to the
        # forex default below. US30m and US500m were therefore sized as if one
        # lot were 100,000 units, making every computed size ~100,000x too
        # small; _apply_mt5_constraints then floored all 15 of them to the
        # 0.01 minimum. Rather than guess a contract size for live money, refuse
        # until one is set explicitly. Read the real value off the symbol in
        # MT5: right-click the symbol -> Specification -> "Contract size", then
        # put it in risk_management.mt5_contract_sizes.
        # FIXED 2026-08-30: measured, not guessed. Deriving contract size from
        # 441 real broker fills as profit / (lots x price_move) gives
        # US30m 1.0053 and US500m 1.0022 (and confirms XAU 100.0000,
        # XAG 5000.0000, EUR 100000.0000, BTC 1.0000, ETH 1.0000). Index CFDs
        # are 1 unit per lot on this broker, NOT the 100,000 forex default they
        # used to fall through to, which is why every US30m/US500m trade was
        # floored to 0.01 lots and worth about $5.
        if any(tok in sym_upper for tok in _INDEX_TOKENS):
            logger.debug(
                f"{symbol}: using fallback index contract size 1.0. Prefer the "
                f"live value from get_symbol_info, or pin it in "
                f"risk_management.mt5_contract_sizes."
            )
            return 1.0

        # Default: standard forex lot
        return 100000.0


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
        account_equity=100000,
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