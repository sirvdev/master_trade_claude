"""
MT5 File-based bridge - Works with ALL MT5 versions (no sockets needed).
Communicates via files in MT5 Common Files directory.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Optional
from datetime import datetime
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class MT5FileBridge:
    """
    MT5 execution bridge using file-based communication.
    Works on ALL MT5 versions (no socket support needed).
    """
    
    def __init__(self, config: Dict, demo_mode: bool = True):
        """
        Initialize MT5 file bridge.
        
        Args:
            config: MT5 configuration
            demo_mode: If True, simulate orders
        """
        self.config = config
        self.demo_mode = demo_mode
        self.magic_number = config.get('magic_number', 123456)
        
        # Find MT5 Common Files directory
        self.common_path = self._find_mt5_common_path()
        
        self.command_file = self.common_path / "python_command.txt"
        self.response_file = self.common_path / "python_response.txt"
        self.status_file = self.common_path / "mt5_status.txt"
        
        self._connected = False
        
        # Demo mode state
        self.demo_orders = {}
        self.demo_positions = {}
        
    def _find_mt5_common_path(self) -> Path:
        """Find MT5 Common Files directory."""
        # Common locations for MT5 Common Files
        possible_paths = [
            Path(os.environ.get('APPDATA', '')) / "MetaQuotes" / "Terminal" / "Common" / "Files",
            Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal" / "Common" / "Files",
            Path("C:/Users/Public/Documents/MetaQuotes/Terminal/Common/Files"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found MT5 Common Files at: {path}")
                return path
                
        # Default fallback - create if doesn't exist
        default_path = possible_paths[0]
        default_path.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Using default path: {default_path}")
        return default_path
        
    async def connect(self):
        """Check if EA is running."""
        if self.demo_mode:
            logger.info("MT5 File Bridge running in DEMO mode")
            self._connected = True
            return
            
        # Check if status file exists and is recent
        if self.status_file.exists():
            try:
                try:
                    status = self.status_file.read_text(encoding='utf-16')
                except:
                    status = self.status_file.read_text(encoding='utf-8')
                    
                if 'ready' in status or 'stopped' in status:
                    logger.info(f"MT5 EA status: {status}")
                    self._connected = True
                else:
                    logger.warning("MT5 EA may not be running")
                    self._connected = False
            except Exception as e:
                logger.error(f"Error reading status file: {e}")
                self._connected = False
        else:
            logger.warning("MT5 status file not found - EA may not be running")
            self._connected = False
            
    async def _send_command(self, command: Dict, timeout: float = 5.0) -> Dict:
        """
        Send command to MT5 EA and wait for response.
        
        Args:
            command: Command dictionary
            timeout: Response timeout in seconds
            
        Returns:
            Response dictionary
        """
        if self.demo_mode:
            return await self._simulate_command(command)
            
        try:
            # Delete old response file
            if self.response_file.exists():
                self.response_file.unlink()
                
            # Write command file
            self.command_file.write_text(json.dumps(command))
            logger.debug(f"Sent command: {command.get('action')}")
            
            # Wait for response with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.response_file.exists():
                    # Small delay to ensure file is fully written
                    await asyncio.sleep(0.05)
                    
                    try:
                        response_text = self.response_file.read_text()
                        response = json.loads(response_text)
                        
                        # Delete response file
                        self.response_file.unlink()
                        
                        logger.debug(f"Received response: {response.get('status')}")
                        return response
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON response: {e}")
                        await asyncio.sleep(0.1)
                        continue
                    except Exception as e:
                        logger.error(f"Error reading response: {e}")
                        await asyncio.sleep(0.1)
                        continue
                        
                await asyncio.sleep(0.1)
                
            logger.error(f"Command timeout after {timeout}s")
            return {'status': 'error', 'error': 'timeout'}
            
        except Exception as e:
            logger.error(f"Error sending command: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}
            
    async def _simulate_command(self, command: Dict) -> Dict:
        """Simulate command in demo mode."""
        await asyncio.sleep(0.05)  # Simulate latency
        
        action = command.get('action')
        
        if action == 'place_order':
            ticket = abs(hash(str(command))) % 1000000
            return {
                'status': 'success',
                'ticket': ticket,
                'price': command.get('price', 2000.0),
                'volume': command.get('volume'),
                'demo_mode': True
            }
        elif action == 'authenticate':
            return {
                'status': 'success',
                'account': 12345678,
                'balance': 10000.0,
                'equity': 10000.0,
                'demo_mode': True
            }
        else:
            return {'status': 'success', 'demo_mode': True}
            
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
        logger.info(
            f"Placing MT5 order: {symbol} {direction} {volume} lots, "
            f"SL: {stop_loss}, TP: {take_profit}"
        )
        
        # Map direction to MT5 order type
        if order_type == 'market':
            mt5_order_type = 'ORDER_TYPE_BUY' if direction == 'long' else 'ORDER_TYPE_SELL'
        else:
            mt5_order_type = 'ORDER_TYPE_BUY_LIMIT' if direction == 'long' else 'ORDER_TYPE_SELL_LIMIT'
            
        command = {
            'action': 'place_order',
            'symbol': symbol,
            'order_type': mt5_order_type,
            'volume': volume,
            'price': price,
            'sl': stop_loss or 0,
            'tp': take_profit or 0,
            'comment': comment or 'Python'
        }
        
        response = await self._send_command(command)
        
        if response.get('status') == 'success':
            result = {
                'success': True,
                'order_id': f"mt5_{response.get('ticket')}",
                'ticket': response.get('ticket'),
                'filled_price': response.get('price'),
                'filled_volume': volume,
                'timestamp': datetime.utcnow().isoformat(),
                'platform': 'mt5',
                'demo_mode': self.demo_mode
            }
            logger.info(f"Order placed: Ticket {result['ticket']}")
        else:
            result = {
                'success': False,
                'error': response.get('error', 'Unknown error'),
                'timestamp': datetime.utcnow().isoformat(),
                'platform': 'mt5'
            }
            logger.error(f"Order failed: {result['error']}")
            
        return result
        
    async def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """Modify position SL/TP."""
        logger.info(f"Modifying position {ticket}: SL={stop_loss}, TP={take_profit}")
        
        command = {
            'action': 'modify_position',
            'ticket': ticket,
            'sl': stop_loss or 0,
            'tp': take_profit or 0
        }
        
        response = await self._send_command(command)
        
        if response.get('status') == 'success':
            logger.info(f"Position {ticket} modified")
            return {'success': True, 'ticket': ticket}
        else:
            logger.error(f"Modify failed: {response.get('error')}")
            return {'success': False, 'error': response.get('error')}
            
    async def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: Optional[str] = None
    ) -> Dict:
        """Close position (full or partial)."""
        logger.info(f"Closing position {ticket}")
        
        command = {
            'action': 'close_position',
            'ticket': ticket,
            'volume': volume or 0,
            'comment': comment or 'Python'
        }
        
        response = await self._send_command(command)
        
        if response.get('status') == 'success':
            logger.info(f"Position {ticket} closed")
            return {
                'success': True,
                'ticket': ticket,
                'closed_volume': volume or 0,
                'profit': response.get('profit', 0)
            }
        else:
            logger.error(f"Close failed: {response.get('error')}")
            return {'success': False, 'error': response.get('error')}
            
    async def get_position_info(self, ticket: int) -> Optional[Dict]:
        """Get position information."""
        command = {
            'action': 'get_position',
            'ticket': ticket
        }
        
        response = await self._send_command(command)
        
        if response.get('status') == 'success':
            return response
        else:
            return None
            
    async def get_all_positions(self) -> list:
        """Get all open positions."""
        command = {
            'action': 'get_all_positions',
            'magic': self.magic_number
        }
        
        response = await self._send_command(command)
        
        if response.get('status') == 'success':
            return response.get('positions', [])
        else:
            return []
            
    async def disconnect(self):
        """Close connection."""
        logger.info("MT5 File Bridge disconnected")
        self._connected = False
        
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected


# Example usage
if __name__ == "__main__":
    async def test_file_bridge():
        """Test file-based bridge."""
        config = {
            'magic_number': 123456
        }
        
        bridge = MT5FileBridge(config, demo_mode=True)
        await bridge.connect()
        
        print("=== MT5 File Bridge Test ===\n")
        
        # Test order
        print("1. Placing order...")
        result = await bridge.place_order(
            symbol='XAUUSD',
            direction='long',
            volume=0.01,
            stop_loss=2000.0,
            take_profit=2050.0
        )
        print(f"Result: {result}")
        
        if result['success']:
            ticket = result['ticket']
            
            # Test modify
            print(f"\n2. Modifying position {ticket}...")
            mod_result = await bridge.modify_position(
                ticket=ticket,
                stop_loss=2005.0
            )
            print(f"Result: {mod_result}")
            
            # Test close
            print(f"\n3. Closing position {ticket}...")
            close_result = await bridge.close_position(ticket=ticket)
            print(f"Result: {close_result}")
            
        await bridge.disconnect()
        print("\nTest completed!")
        
    asyncio.run(test_file_bridge())