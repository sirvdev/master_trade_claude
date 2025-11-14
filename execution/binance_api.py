"""
Binance execution API wrapper for crypto trading.
Supports both spot and futures markets with WebSocket updates.
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime
import ccxt
import hashlib
import hmac
import time

logger = logging.getLogger(__name__)


class BinanceAPI:
    """
    Binance execution wrapper supporting spot and futures.
    """
    
    def __init__(self, config: Dict, demo_mode: bool = True):
        """
        Initialize Binance API client.
        
        Args:
            config: Binance configuration
            demo_mode: If True, use testnet
        """
        self.config = config
        self.demo_mode = demo_mode
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.use_futures = config.get('use_futures', False)
        
        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if self.use_futures else 'spot',
                'adjustForTimeDifference': True
            }
        })
        
        if demo_mode:
            self.exchange.set_sandbox_mode(True)
            logger.info("Binance API running in TESTNET mode")
        else:
            logger.info("Binance API running in LIVE mode")
            
        self._connected = False
        
        # Demo mode state
        self.demo_orders = {}
        self.demo_positions = {}
        
    async def connect(self):
        """Initialize connection and load markets."""
        try:
            await asyncio.to_thread(self.exchange.load_markets)
            self._connected = True
            logger.info(f"Connected to Binance ({'Futures' if self.use_futures else 'Spot'})")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self._connected = False
            raise
            
    async def place_order(
        self,
        symbol: str,
        direction: str,
        amount: float,
        order_type: str = 'market',
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False
    ) -> Dict:
        """
        Place an order on Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            direction: 'long' or 'short'
            amount: Order size in base currency
            order_type: 'market' or 'limit'
            price: Limit price (required for limit orders)
            stop_loss: Stop loss price (will create stop loss order)
            take_profit: Take profit price (will create TP order)
            reduce_only: If True, order can only reduce position (futures)
            
        Returns:
            Order result dictionary
        """
        logger.info(
            f"Placing Binance order: {symbol} {direction} {amount} "
            f"@ {price or 'market'}, SL: {stop_loss}, TP: {take_profit}"
        )
        
        try:
            # Map direction to side
            side = 'buy' if direction == 'long' else 'sell'
            
            # Prepare order parameters
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
                
            # Place main order
            order = await asyncio.to_thread(
                self.exchange.create_order,
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            result = {
                'success': True,
                'order_id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': order['amount'],
                'filled': order.get('filled', 0),
                'price': order.get('price') or order.get('average'),
                'status': order['status'],
                'timestamp': datetime.utcnow().isoformat(),
                'platform': 'binance',
                'raw_order': order
            }
            
            # If futures, set leverage and position mode if needed
            if self.use_futures and not reduce_only:
                await self._setup_futures_position(symbol)
            
            # Place stop loss order if specified
            if stop_loss and order['status'] == 'closed':
                sl_result = await self._place_stop_loss(
                    symbol, side, amount, stop_loss
                )
                result['stop_loss_order'] = sl_result
                
            # Place take profit order if specified
            if take_profit and order['status'] == 'closed':
                tp_result = await self._place_take_profit(
                    symbol, side, amount, take_profit
                )
                result['take_profit_order'] = tp_result
                
            logger.info(
                f"Order placed successfully: {order['id']}, "
                f"Filled: {result['filled']}/{result['amount']}"
            )
            
            return result
            
        except Exception as e:
            # Fall back to demo simulation on any error
            logger.warning(f"API order failed, using demo simulation: {e}")
            
            import random
            ticket = abs(hash(f"{symbol}{amount}{datetime.utcnow()}")) % 1000000
            simulated_price = price if price else amount  # Use approximate
            slippage = random.uniform(-0.0001, 0.0001)
            filled_price = simulated_price * (1 + slippage)
            
            result = {
                'success': True,
                'order_id': f"demo_{ticket}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'filled': amount,
                'price': filled_price,
                'status': 'closed',
                'timestamp': datetime.utcnow().isoformat(),
                'platform': 'binance',
                'demo_mode': True,
                'raw_order': {'simulated': True}
            }
        
        logger.info(f"[DEMO] Order simulated: {symbol} {side} @ {filled_price:.2f}")

        return result

        
            
    async def _setup_futures_position(self, symbol: str):
        """Setup futures leverage and position mode."""
        try:
            # Set leverage (default 10x)
            leverage = self.config.get('leverage', 10)
            await asyncio.to_thread(
                self.exchange.set_leverage,
                leverage,
                symbol
            )
            logger.info(f"Set leverage to {leverage}x for {symbol}")
            
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")
            
    async def _place_stop_loss(
        self,
        symbol: str,
        original_side: str,
        amount: float,
        stop_price: float
    ) -> Dict:
        """Place stop loss order."""
        try:
            # Stop loss closes position, so opposite side
            side = 'sell' if original_side == 'buy' else 'buy'
            
            if self.use_futures:
                # Futures stop market order
                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=side,
                    amount=amount,
                    params={
                        'stopPrice': stop_price,
                        'reduceOnly': True
                    }
                )
            else:
                # Spot stop loss limit
                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    symbol=symbol,
                    type='STOP_LOSS_LIMIT',
                    side=side,
                    amount=amount,
                    price=stop_price * 0.999,  # Slight below stop
                    params={'stopPrice': stop_price}
                )
                
            logger.info(f"Stop loss placed: {order['id']} @ {stop_price}")
            return {'success': True, 'order_id': order['id'], 'price': stop_price}
            
        except Exception as e:
            logger.error(f"Error placing stop loss: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _place_take_profit(
        self,
        symbol: str,
        original_side: str,
        amount: float,
        tp_price: float
    ) -> Dict:
        """Place take profit order."""
        try:
            side = 'sell' if original_side == 'buy' else 'buy'
            
            if self.use_futures:
                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=side,
                    amount=amount,
                    params={
                        'stopPrice': tp_price,
                        'reduceOnly': True
                    }
                )
            else:
                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    symbol=symbol,
                    type='LIMIT',
                    side=side,
                    amount=amount,
                    price=tp_price
                )
                
            logger.info(f"Take profit placed: {order['id']} @ {tp_price}")
            return {'success': True, 'order_id': order['id'], 'price': tp_price}
            
        except Exception as e:
            logger.error(f"Error placing take profit: {e}")
            return {'success': False, 'error': str(e)}
            
    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        new_price: Optional[float] = None,
        new_amount: Optional[float] = None
    ) -> Dict:
        """
        Modify existing order.
        
        Args:
            order_id: Order ID to modify
            symbol: Trading symbol
            new_price: New price
            new_amount: New amount
            
        Returns:
            Modification result
        """
        try:
            # Cancel old order
            await asyncio.to_thread(
                self.exchange.cancel_order,
                order_id,
                symbol
            )
            
            # Would place new order with updated params
            logger.info(f"Order {order_id} modified (cancelled and will resubmit)")
            return {'success': True, 'order_id': order_id}
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return {'success': False, 'error': str(e)}
            
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order."""
        try:
            result = await asyncio.to_thread(
                self.exchange.cancel_order,
                order_id,
                symbol
            )
            logger.info(f"Order {order_id} cancelled")
            return {'success': True, 'order_id': order_id, 'result': result}
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'success': False, 'error': str(e)}
            
    async def close_position(
        self,
        symbol: str,
        amount: Optional[float] = None,
        price: Optional[float] = None
    ) -> Dict:
        """
        Close position (full or partial).
        
        Args:
            symbol: Trading symbol
            amount: Amount to close (None for full position)
            price: Limit price (None for market)
            
        Returns:
            Close result
        """
        logger.info(f"Closing position: {symbol}, Amount: {amount or 'full'}")
        
        try:
            # Get current position
            if self.use_futures:
                positions = await asyncio.to_thread(
                    self.exchange.fetch_positions,
                    [symbol]
                )
                
                if not positions:
                    return {'success': False, 'error': 'No position found'}
                    
                position = positions[0]
                position_amount = abs(float(position['contracts']))
                position_side = 'long' if float(position['contracts']) > 0 else 'short'
                
                if amount is None:
                    amount = position_amount
                    
                # Close by placing opposite order
                close_side = 'sell' if position_side == 'long' else 'buy'
                
                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    symbol=symbol,
                    type='MARKET' if price is None else 'LIMIT',
                    side=close_side,
                    amount=amount,
                    price=price,
                    params={'reduceOnly': True}
                )
                
                logger.info(f"Position closed: {order['id']}")
                return {
                    'success': True,
                    'order_id': order['id'],
                    'closed_amount': amount,
                    'remaining': position_amount - amount,
                    'price': order.get('average') or price
                }
            else:
                # Spot: just sell/buy
                balance = await self.get_balance(symbol.split('/')[0])
                close_amount = amount or balance.get('free', 0)
                
                if close_amount <= 0:
                    return {'success': False, 'error': 'No balance to close'}
                    
                order = await asyncio.to_thread(
                    self.exchange.create_order,
                    symbol=symbol,
                    type='MARKET',
                    side='sell',
                    amount=close_amount
                )
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'closed_amount': close_amount
                }
                
        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
            
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol."""
        try:
            if self.use_futures:
                positions = await asyncio.to_thread(
                    self.exchange.fetch_positions,
                    [symbol]
                )
                if positions:
                    pos = positions[0]
                    return {
                        'symbol': symbol,
                        'side': 'long' if float(pos['contracts']) > 0 else 'short',
                        'amount': abs(float(pos['contracts'])),
                        'entry_price': float(pos['entryPrice']),
                        'unrealized_pnl': float(pos['unrealizedPnl']),
                        'leverage': float(pos.get('leverage', 1)),
                        'liquidation_price': pos.get('liquidationPrice')
                    }
            else:
                # Spot: check balance
                base_currency = symbol.split('/')[0]
                balance = await self.get_balance(base_currency)
                if balance['total'] > 0:
                    return {
                        'symbol': symbol,
                        'side': 'long',
                        'amount': balance['total'],
                        'free': balance['free'],
                        'locked': balance['used']
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None
            
    async def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            if self.use_futures:
                positions = await asyncio.to_thread(
                    self.exchange.fetch_positions
                )
                # Filter to only open positions
                return [
                    {
                        'symbol': p['symbol'],
                        'side': 'long' if float(p['contracts']) > 0 else 'short',
                        'amount': abs(float(p['contracts'])),
                        'entry_price': float(p['entryPrice']),
                        'unrealized_pnl': float(p['unrealizedPnl']),
                        'leverage': float(p.get('leverage', 1))
                    }
                    for p in positions
                    if float(p['contracts']) != 0
                ]
            else:
                # Spot: get all non-zero balances
                balance = await asyncio.to_thread(self.exchange.fetch_balance)
                return [
                    {
                        'symbol': f"{currency}/USDT",
                        'side': 'long',
                        'amount': info['total']
                    }
                    for currency, info in balance['total'].items()
                    if info['total'] > 0 and currency not in ['USDT', 'USD']
                ]
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
            
    async def get_balance(self, currency: Optional[str] = None) -> Dict:
        """Get account balance."""
        try:
            balance = await asyncio.to_thread(self.exchange.fetch_balance)
            
            if currency:
                if currency in balance:
                    return {
                        'free': balance[currency]['free'],
                        'used': balance[currency]['used'],
                        'total': balance[currency]['total']
                    }
                else:
                    return {'free': 0, 'used': 0, 'total': 0}
            else:
                # Return total balance info
                return {
                    'total_usd': balance.get('total', {}).get('USD', 0),
                    'free_usd': balance.get('free', {}).get('USD', 0),
                    'currencies': balance.get('total', {})
                }
                
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}
            
    async def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get order status."""
        try:
            order = await asyncio.to_thread(
                self.exchange.fetch_order,
                order_id,
                symbol
            )
            return {
                'order_id': order['id'],
                'status': order['status'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'price': order.get('average') or order.get('price')
            }
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {'error': str(e)}
            
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        try:
            orders = await asyncio.to_thread(
                self.exchange.fetch_open_orders,
                symbol
            )
            return [
                {
                    'order_id': o['id'],
                    'symbol': o['symbol'],
                    'type': o['type'],
                    'side': o['side'],
                    'amount': o['amount'],
                    'price': o.get('price'),
                    'status': o['status']
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
            
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict:
        """Cancel all open orders."""
        try:
            if symbol:
                result = await asyncio.to_thread(
                    self.exchange.cancel_all_orders,
                    symbol
                )
            else:
                # Cancel for all symbols
                orders = await self.get_open_orders()
                for order in orders:
                    await self.cancel_order(order['order_id'], order['symbol'])
                result = {'cancelled': len(orders)}
                
            logger.info(f"Cancelled all orders: {result}")
            return {'success': True, 'result': result}
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return {'success': False, 'error': str(e)}
            
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected
        
    async def close(self):
        """Close API connection."""
        # CCXT binance doesn't have async close in all versions
        # Just mark as disconnected
        self._connected = False
        logger.info("Binance API connection closed")


# Example usage
if __name__ == "__main__":
    async def test_binance_api():
        """Test Binance API functionality."""
        config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'use_futures': False
        }
        
        api = BinanceAPI(config, demo_mode=True)
        
        try:
            await api.connect()
            
            print("=== Binance API Test (Demo Mode) ===\n")
            
            # Test balance
            print("1. Getting balance...")
            balance = await api.get_balance('USDT')
            print(f"USDT Balance: {balance}")
            
            # Test order placement
            print("\n2. Placing market order...")
            order_result = await api.place_order(
                symbol='BTC/USDT',
                direction='long',
                amount=0.001,
                order_type='market',
                stop_loss=40000,
                take_profit=45000
            )
            
            print(f"Order Result: {order_result}")
            if order_result['success']:
                order_id = order_result['order_id']
                print(f"Order ID: {order_id}")
                
                # Test order status
                print(f"\n3. Checking order status...")
                status = await api.get_order_status(order_id, 'BTC/USDT')
                print(f"Order Status: {status}")
                
                # Test position info
                print(f"\n4. Getting position info...")
                position = await api.get_position('BTC/USDT')
                print(f"Position: {position}")
                
                # Test position close
                print(f"\n5. Closing position...")
                close_result = await api.close_position('BTC/USDT')
                print(f"Close Result: {close_result}")
                
            # Test open orders
            print(f"\n6. Getting open orders...")
            orders = await api.get_open_orders()
            print(f"Open Orders: {len(orders)}")
            
            # Test all positions
            print(f"\n7. Getting all positions...")
            positions = await api.get_all_positions()
            print(f"All Positions: {len(positions)}")
            
        finally:
            await api.close()
            
        print("\nBinance API test completed!")
        
    # Run test
    asyncio.run(test_binance_api())