"""
MT5 File-based bridge with single concatenated response file.
Replace execution/mt5_file_bridge.py with this entire file.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Optional
from datetime import datetime
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class MT5FileBridge:
    """
    MT5 execution bridge using single-file response system.
    Each request gets a unique ID, all responses concatenated in one file.
    """
    
    def __init__(self, config: Dict, demo_mode: bool = True):
        """Initialize MT5 bridge."""
        self.config = config
        self.demo_mode = demo_mode
        self.magic_number = config.get('magic_number', 123456)
        
        # Generate session ID (resets on each program start)
        self.session_id = str(uuid.uuid4())[:8]
        
        # Find MT5 Common Files directory
        self.common_path = self._find_mt5_common_path()
        
        # File paths
        self.command_file = self.common_path / "python_command.txt"
        self.response_file = self.common_path / "python_responses.txt"
        self.status_file = self.common_path / "mt5_status.txt"
        self.session_file = self.common_path / "python_session.txt"
        
        self._connected = False
        self.request_counter = 0
        self.last_read_position = 0
        
        # Demo mode state
        self.demo_orders = {}
        self.demo_positions = {}
        
    def _find_mt5_common_path(self) -> Path:
        """Find MT5 Common Files directory."""
        possible_paths = [
            Path(os.environ.get('APPDATA', '')) / "MetaQuotes" / "Terminal" / "Common" / "Files",
            Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal" / "Common" / "Files",
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found MT5 Common Files at: {path}")
                return path
                
        default_path = possible_paths[0]
        default_path.mkdir(parents=True, exist_ok=True)
        return default_path
        
    async def connect(self):
        """Check if EA is running."""
        if self.demo_mode:
            logger.info("MT5 File Bridge running in DEMO mode")
            self._connected = True
            return
            
        # Write session ID so EA knows to reset
        try:
            self.session_file.write_text(self.session_id, encoding='utf-8')
            logger.info(f"Session ID: {self.session_id}")
        except Exception as e:
            logger.error(f"Could not write session file: {e}")
        
        # Check status file
        if self.status_file.exists():
            try:
                status = self.status_file.read_text(encoding='utf-8', errors='ignore')
                status = status.lstrip('\ufeff').strip()
                
                if 'ready' in status.lower():
                    logger.info(f" MT5 EA is ready")
                    self._connected = True
                else:
                    logger.warning(f"MT5 EA status: {status}")
                    self._connected = True
            except Exception as e:
                logger.error(f"Error reading status: {e}")
                self._connected = False
        else:
            logger.warning("MT5 status file not found")
            self._connected = False
            
    async def _send_command(self, command: Dict, timeout: float = 0.5) -> Dict:
        """Send command with unique ID and wait for response."""
        if self.demo_mode:
            return await self._simulate_command(command)
        
        # Generate unique request ID
        self.request_counter += 1
        request_id = f"{self.session_id}_{self.request_counter}"
        
        # Add request ID to command
        command['request_id'] = request_id
        
        try:
            # Write command
            command_json = json.dumps(command, ensure_ascii=True)
            self.command_file.write_text(command_json, encoding='utf-8')
            
            logger.debug(f"Sent command {request_id}: {command.get('action')}")
            
            # Wait for response in the concatenated file
            start_time = time.time()
            while time.time() - start_time < timeout:
                response = await self._read_response_for_id(request_id)
                if response:
                    logger.debug(f"Received response {request_id}: {response.get('status')}")
                    return response
                
                await asyncio.sleep(0.05)
            
            logger.error(f"Command {request_id} timeout after {timeout}s")
            return {'status': 'error', 'error': 'timeout'}
            
        except Exception as e:
            logger.error(f"Error sending command: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}
    
    async def _read_response_for_id(self, request_id: str) -> Optional[Dict]:
        """Read response file and find response for specific request ID."""
        if not self.response_file.exists():
            return None
        
        try:
            # Read file from last position
            with open(self.response_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_read_position)
                new_content = f.read()
                
                if not new_content:
                    return None
                
                # Split into lines
                lines = new_content.strip().split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    try:
                        response = json.loads(line)
                        if response.get('request_id') == request_id:
                            # Update read position to end of file
                            self.last_read_position = f.tell()
                            return response
                    except json.JSONDecodeError:
                        continue
                
                # Update position even if not found
                self.last_read_position = f.tell()
                
        except Exception as e:
            logger.debug(f"Error reading response file: {e}")
        
        return None
    
    async def _simulate_command(self, command: Dict) -> Dict:
        """Simulate command in demo mode."""
        await asyncio.sleep(0.05)
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
            return {'status': 'success', 'message': 'pong', 'demo_mode': True}
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
        """Place an order on MT5."""
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
                'price': response.get('price'),
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
        """Close connection and cleanup response file."""
        # Delete the response file on shutdown
        try:
            if self.response_file.exists():
                self.response_file.unlink()
                logger.info("Cleaned up response file")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
        
        logger.info("MT5 File Bridge disconnected")
        self._connected = False
        
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected