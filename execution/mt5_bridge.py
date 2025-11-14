"""
MT5 execution bridge via socket/EA connection.
Handles order placement, modification, and position management for metals trading.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MT5Bridge:
    """
    MT5 execution bridge for socket-based communication with EA.
    """
    
    def __init__(self, config: Dict, demo_mode: bool = True):
        """
        Initialize MT5 bridge.
        
        Args:
            config: MT5 configuration
            demo_mode: If True, simulate orders without actual execution
        """
        self.config = config
        self.demo_mode = demo_mode
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 9090)
        self.magic_number = config.get('magic_number', 123456)
        
        self.reader = None
        self.writer = None
        self._connected = False
        self._pending_orders = {}
        
        # Demo mode state
        self.demo_orders = {}
        self.demo_positions = {}
        
    async def connect(self):
        """Establish connection to MT5 EA bridge."""
        if self.demo_mode:
            logger.info("MT5 Bridge running in DEMO mode - no actual connection")
            self._connected = True
            return
            
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port
            )
            
            # Send authentication
            auth_msg = {
                'action': 'authenticate',
                'account': self.config.get('account'),
                'password': self.config.get('password'),
                'server': self.config.get('server'),
                'magic_number': self.magic_number
            }
            
            await self._send_command(auth_msg)
            response = await self._receive_response()
            
            if response.get('status') != 'success':
                raise ConnectionError(f"MT5 authentication failed: {response.get('error')}")
                
            self._connected = True
            logger.info(f"Connected to MT5 bridge at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MT5 bridge: {e}")
            self._connected = False
            raise
            
    async def _send_command(self, command: Dict):
        """Send command to MT5 EA."""
        data = json.dumps(command) + '\n'
        self.writer.write(data.encode())
        await self.writer.drain()
        
    async def _receive_response(self, timeout: float = 5.0) -> Dict:
        """Receive response from MT5 EA."""
        try:
            data = await asyncio.wait_for(
                self.reader.readline(),
                timeout=timeout
            )
            return json.loads(data.decode())
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for MT5 response")
            return {'status': 'error', 'error': 'timeout'}
            
    async def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        order_type: str = 'market',
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: Optional[str] = None
    ) -> Dict:
        """
        Place an order on MT5.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            direction: 'long' or 'short'
            volume: Position size in lots
            order_type: 'market' or 'limit'
            price: Entry price for limit orders
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            
        Returns:
            Order result dictionary
        """
        order_id = str(uuid.uuid4())
        
        logger.info(
            f"Placing MT5 order: {symbol} {direction} {volume} lots "
            f"@ {price or 'market'}, SL: {stop_loss}, TP: {take_profit}"
        )
        
        if self.demo_mode:
            return await self._simulate_order(
                order_id, symbol, direction, volume, order_type,
                price, stop_loss, take_profit, comment
            )
            
        try:
            # Map direction to MT5 order type
            if order_type == 'market':
                mt5_order_type = 'ORDER_TYPE_BUY' if direction == 'long' else 'ORDER_TYPE_SELL'
            else:  # limit
                mt5_order_type = 'ORDER_TYPE_BUY_LIMIT' if direction == 'long' else 'ORDER_TYPE_SELL_LIMIT'
                
            command = {
                'action': 'place_order',
                'order_id': order_id,
                'symbol': symbol,
                'order_type': mt5_order_type,
                'volume': volume,
                'price': price,
                'sl': stop_loss,
                'tp': take_profit,
                'magic': self.magic_number,
                'comment': comment or f"Trade_{order_id[:8]}",
                'deviation': self.config.get('deviation_points', 10),
                'filling': self.config.get('fill_policy', 'FOK')
            }
            
            await self._send_command(command)
            self._pending_orders[order_id] = command
            
            # Wait for response
            response = await self._receive_response(timeout=10.0)
            
            if response.get('status') == 'success':
                result = {
                    'success': True,
                    'order_id': order_id,
                    'ticket': response.get('ticket'),
                    'filled_price': response.get('price'),
                    'filled_volume': response.get('volume'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'platform': 'mt5',
                    'raw_response': response
                }
                logger.info(f"Order placed successfully: Ticket {result['ticket']}")
            else:
                result = {
                    'success': False,
                    'order_id': order_id,
                    'error': response.get('error', 'Unknown error'),
                    'error_code': response.get('error_code'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'platform': 'mt5',
                    'raw_response': response
                }
                logger.error(f"Order failed: {result['error']}")
                
            # Remove from pending
            self._pending_orders.pop(order_id, None)
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing MT5 order: {e}", exc_info=True)
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'platform': 'mt5'
            }
            
    async def _simulate_order(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        volume: float,
        order_type: str,
        price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        comment: Optional[str]
    ) -> Dict:
        """Simulate order execution in demo mode."""
        # Simulate network latency
        await asyncio.sleep(0.05)
        
        # Simulate slippage for market orders
        slippage = 0
        if order_type == 'market':
            import random
            slippage = random.uniform(-0.0002, 0.0002)  # ±0.02%
            
        filled_price = price or 2000.0  # Mock price if not provided
        filled_price *= (1 + slippage)
        
        ticket = abs(hash(order_id)) % 1000000
        
        # Store demo position
        self.demo_positions[order_id] = {
            'ticket': ticket,
            'symbol': symbol,
            'direction': direction,
            'volume': volume,
            'entry_price': filled_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'opened_at': datetime.utcnow(),
            'comment': comment
        }
        
        logger.info(
            f"[DEMO] Order simulated: Ticket {ticket}, "
            f"Filled @ {filled_price:.4f} (Slippage: {slippage*100:.3f}%)"
        )
        
        return {
            'success': True,
            'order_id': order_id,
            'ticket': ticket,
            'filled_price': filled_price,
            'filled_volume': volume,
            'slippage': slippage,
            'timestamp': datetime.utcnow().isoformat(),
            'platform': 'mt5',
            'demo_mode': True
        }
        
    async def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Modify stop loss or take profit for existing position.
        
        Args:
            ticket: MT5 position ticket
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            Modification result
        """
        logger.info(f"Modifying MT5 position {ticket}: SL={stop_loss}, TP={take_profit}")
        
        if self.demo_mode:
            # Find demo position
            for pos_id, pos in self.demo_positions.items():
                if pos['ticket'] == ticket:
                    if stop_loss is not None:
                        pos['stop_loss'] = stop_loss
                    if take_profit is not None:
                        pos['take_profit'] = take_profit
                    logger.info(f"[DEMO] Position {ticket} modified")
                    return {
                        'success': True,
                        'ticket': ticket,
                        'demo_mode': True
                    }
            return {'success': False, 'error': 'Position not found'}
            
        try:
            command = {
                'action': 'modify_position',
                'ticket': ticket,
                'sl': stop_loss,
                'tp': take_profit
            }
            
            await self._send_command(command)
            response = await self._receive_response()
            
            if response.get('status') == 'success':
                logger.info(f"Position {ticket} modified successfully")
                return {
                    'success': True,
                    'ticket': ticket,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                logger.error(f"Position modification failed: {response.get('error')}")
                return {
                    'success': False,
                    'ticket': ticket,
                    'error': response.get('error')
                }
                
        except Exception as e:
            logger.error(f"Error modifying position: {e}", exc_info=True)
            return {'success': False, 'ticket': ticket, 'error': str(e)}
            
    async def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: Optional[str] = None
    ) -> Dict:
        """
        Close position (full or partial).
        
        Args:
            ticket: MT5 position ticket
            volume: Volume to close (None for full close)
            comment: Close comment
            
        Returns:
            Close result
        """
        logger.info(f"Closing MT5 position {ticket}, Volume: {volume or 'full'}")
        
        if self.demo_mode:
            for pos_id, pos in list(self.demo_positions.items()):
                if pos['ticket'] == ticket:
                    close_volume = volume or pos['volume']
                    remaining = pos['volume'] - close_volume
                    
                    # Simulate P&L
                    import random
                    pnl = random.uniform(-100, 200)  # Mock P&L
                    
                    if remaining <= 0:
                        del self.demo_positions[pos_id]
                        logger.info(f"[DEMO] Position {ticket} fully closed, P&L: ${pnl:.2f}")
                    else:
                        pos['volume'] = remaining
                        logger.info(f"[DEMO] Position {ticket} partially closed: {close_volume}/{close_volume+remaining}")
                        
                    return {
                        'success': True,
                        'ticket': ticket,
                        'closed_volume': close_volume,
                        'remaining_volume': remaining,
                        'pnl': pnl,
                        'demo_mode': True
                    }
                    
            return {'success': False, 'error': 'Position not found'}
            
        try:
            command = {
                'action': 'close_position',
                'ticket': ticket,
                'volume': volume,
                'comment': comment or f"Close_{ticket}"
            }
            
            await self._send_command(command)
            response = await self._receive_response(timeout=10.0)
            
            if response.get('status') == 'success':
                logger.info(f"Position {ticket} closed successfully")
                return {
                    'success': True,
                    'ticket': ticket,
                    'closed_volume': response.get('volume'),
                    'close_price': response.get('price'),
                    'pnl': response.get('profit'),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                logger.error(f"Position close failed: {response.get('error')}")
                return {
                    'success': False,
                    'ticket': ticket,
                    'error': response.get('error')
                }
                
        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return {'success': False, 'ticket': ticket, 'error': str(e)}
            
    async def get_position_info(self, ticket: int) -> Optional[Dict]:
        """Get current position information."""
        if self.demo_mode:
            for pos in self.demo_positions.values():
                if pos['ticket'] == ticket:
                    return pos
            return None
            
        try:
            command = {
                'action': 'get_position',
                'ticket': ticket
            }
            
            await self._send_command(command)
            response = await self._receive_response()
            
            if response.get('status') == 'success':
                return response.get('position')
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return None
            
    async def get_all_positions(self) -> list:
        """Get all open positions."""
        if self.demo_mode:
            return list(self.demo_positions.values())
            
        try:
            command = {'action': 'get_all_positions', 'magic': self.magic_number}
            await self._send_command(command)
            response = await self._receive_response()
            
            if response.get('status') == 'success':
                return response.get('positions', [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
            
    async def disconnect(self):
        """Close connection to MT5 bridge."""
        if self.demo_mode:
            logger.info("MT5 Bridge disconnected (demo mode)")
            self._connected = False
            return
            
        if self.writer:
            try:
                await self._send_command({'action': 'disconnect'})
                self.writer.close()
                await self.writer.wait_closed()
                logger.info("Disconnected from MT5 bridge")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
                
        self._connected = False
        
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected


# Example usage
if __name__ == "__main__":
    async def test_mt5_bridge():
        """Test MT5 bridge functionality."""
        config = {
            'host': 'localhost',
            'port': 9090,
            'account': 12345678,
            'password': 'test_password',
            'server': 'MetaQuotes-Demo',
            'magic_number': 123456
        }
        
        # Initialize in demo mode
        bridge = MT5Bridge(config, demo_mode=True)
        await bridge.connect()
        
        print("=== MT5 Bridge Test (Demo Mode) ===\n")
        
        # Test order placement
        print("1. Placing market order...")
        order_result = await bridge.place_order(
            symbol='XAUUSD',
            direction='long',
            volume=0.1,
            order_type='market',
            stop_loss=2040.0,
            take_profit=2060.0,
            comment='Test order'
        )
        
        print(f"Order Result: {order_result}")
        print(f"Success: {order_result['success']}")
        if order_result['success']:
            ticket = order_result['ticket']
            print(f"Ticket: {ticket}")
            print(f"Filled @ {order_result['filled_price']:.2f}")
            
            # Test position modification
            print(f"\n2. Modifying position {ticket}...")
            mod_result = await bridge.modify_position(
                ticket=ticket,
                stop_loss=2045.0,
                take_profit=2065.0
            )
            print(f"Modification Result: {mod_result}")
            
            # Test position info
            print(f"\n3. Getting position info...")
            pos_info = await bridge.get_position_info(ticket)
            print(f"Position Info: {pos_info}")
            
            # Test partial close
            print(f"\n4. Partially closing position...")
            close_result = await bridge.close_position(
                ticket=ticket,
                volume=0.05
            )
            print(f"Close Result: {close_result}")
            
            # Test full close
            print(f"\n5. Fully closing position...")
            close_result = await bridge.close_position(ticket=ticket)
            print(f"Close Result: {close_result}")
            
        # Test get all positions
        print(f"\n6. Getting all positions...")
        positions = await bridge.get_all_positions()
        print(f"Open Positions: {len(positions)}")
        
        await bridge.disconnect()
        print("\nMT5 Bridge test completed!")
        
    # Run test
    asyncio.run(test_mt5_bridge())