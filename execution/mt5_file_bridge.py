"""
MT5 File-based bridge - Fixed encoding issues.
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
                # Try different encodings
                status = None
                for encoding in ['utf-8', 'utf-16-le', 'utf-16', 'ansi']:
                    try:
                        status = self.status_file.read_text(encoding=encoding)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                
                if status is None:
                    # Try with no encoding specified (binary)
                    status = self.status_file.read_bytes().decode('utf-8', errors='ignore')
                
                # Remove BOM and whitespace
                status = status.lstrip('\ufeff').strip()
                
                logger.info(f"Status file content: '{status}'")
                
                if 'ready' in status.lower():
                    logger.info(f"✓ MT5 EA is ready")
                    self._connected = True
                elif 'stopped' in status.lower():
                    logger.warning("MT5 EA is stopped")
                    self._connected = False
                else:
                    logger.warning(f"Unexpected status: {status}")
                    # Still try to connect
                    self._connected = True
                    
            except Exception as e:
                logger.error(f"Error reading status file: {e}")
                self._connected = False
        else:
            logger.warning("MT5 status file not found - EA may not be running")
            self._connected = False
            
    async def _send_command(self, command: Dict, timeout: float = 10.0) -> Dict:
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
            # Delete old response file if exists
            if self.response_file.exists():
                try:
                    self.response_file.unlink()
                except:
                    pass
                
            # Write command file as UTF-8 (simple ASCII JSON)
            command_json = json.dumps(command, ensure_ascii=True)
            
            # Write as ANSI/UTF-8 (what MQL5 expects)
            with open(self.command_file, 'w', encoding='utf-8') as f:
                f.write(command_json)
            
            logger.debug(f"Sent command: {command.get('action')}")
            
            # Wait for response with timeout
            start_time = time.time()
            response_data = None
            
            while time.time() - start_time < timeout:
                if self.response_file.exists():
                    # Small delay to ensure file is fully written
                    await asyncio.sleep(0.05)
                    
                    try:
                        # Try UTF-16 first (MT5 default), then UTF-8
                        try:
                            response_text = self.response_file.read_text(encoding='utf-16-le')
                        except UnicodeDecodeError:
                            try:
                                response_text = self.response_file.read_text(encoding='utf-16')
                            except UnicodeDecodeError:
                                response_text = self.response_file.read_text(encoding='utf-8')
                        
                        # Remove BOM if present
                        response_text = response_text.lstrip('\ufeff').strip()
                        
                        if response_text:
                            response_data = json.loads(response_text)
                            
                            # Delete response file after reading
                            try:
                                self.response_file.unlink()
                            except:
                                pass
                            
                            logger.debug(f"Received response: {response_data.get('status')}")
                            return response_data
                            
                    except json.JSONDecodeError as e:
                        logger.debug(f"Waiting for complete JSON response...")
                        logger.debug(f"Response text: {response_text[:200]}")
                        await asyncio.sleep(0.1)
                        continue
                    except Exception as e:
                        logger.error(f"Error reading response: {e}")
                        await asyncio.sleep(0.1)
                        continue
                        
                await asyncio.sleep(0.05)  # Check every 50ms
                
            logger.error(f"Command timeout after {timeout}s")
            logger.debug(f"Command was: {command}")
            logger.debug(f"Command file exists: {self.command_file.exists()}")
            logger.debug(f"Response file exists: {self.response_file.exists()}")
            
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
        elif action == 'ping':
            return {
                'status': 'success',
                'message': 'pong',
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