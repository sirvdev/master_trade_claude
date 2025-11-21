"""
MT5 File Bridge - Fixed to get current price and validate stops.
Implements an execution bridge to MetaTrader 5 using file-based communication.
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
    """MT5 execution bridge with current price fetching."""
    
    def __init__(self, config: Dict, demo_mode: bool = True):
        """Initialize MT5 bridge."""
        self.config = config
        self.demo_mode = demo_mode
        self.magic_number = config.get('magic_number', 123456)
        
        # Generate session ID
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
                    logger.info(f"MT5 EA is ready")
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
        """Send command and wait for response."""
        if self.demo_mode:
            return await self._simulate_command(command)
        
        self.request_counter += 1
        request_id = f"{self.session_id}_{self.request_counter}"
        command['request_id'] = request_id
        
        try:
            command_json = json.dumps(command, ensure_ascii=True)
            self.command_file.write_text(command_json, encoding='utf-8')
            
            logger.debug(f"Sent command {request_id}: {command.get('action')}")
            
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
            with open(self.response_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_read_position)
                new_content = f.read()
                
                if not new_content:
                    return None
                
                lines = new_content.strip().split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    try:
                        response = json.loads(line)
                        if response.get('request_id') == request_id:
                            self.last_read_position = f.tell()
                            return response
                    except json.JSONDecodeError:
                        continue
                
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
        elif action == 'get_current_price':
            return {
                'status': 'success',
                'bid': 4084.5,
                'ask': 4084.7,
                'demo_mode': True
            }
        elif action == 'get_symbol_info':
            return {
                'status': 'success',
                'stops_level': 10.0,
                'freeze_level': 5.0,
                'demo_mode': True
            }
        else:
            return {'status': 'success', 'demo_mode': True}
    
    async def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask prices."""
        logger.debug(f"Getting current price for {symbol}")
        
        command = {
            'action': 'get_current_price',
            'symbol': symbol
        }
        
        response = await self._send_command(command)
        
        if response.get('status') == 'success':
            return {
                'bid': response.get('bid'),
                'ask': response.get('ask'),
                'spread': response.get('ask', 0) - response.get('bid', 0)
            }
        else:
            logger.error(f"Failed to get price: {response.get('error')}")
            return None
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading info (stops level, etc)."""
        command = {
            'action': 'get_symbol_info',
            'symbol': symbol
        }
        
        response = await self._send_command(command)
        
        if response.get('status') == 'success':
            return {
                'stops_level': response.get('stops_level', 0),
                'freeze_level': response.get('freeze_level', 0),
                'min_lot': response.get('min_lot', 0.01),
                'max_lot': response.get('max_lot', 100),
                'lot_step': response.get('lot_step', 0.01),
                'point': response.get('point', 0.01)
            }
        else:
            return None
    
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
        """Place an order with validated stops."""
        logger.info(
            f"Placing MT5 order: {symbol} {direction} {volume} lots, "
            f"SL: {stop_loss}, TP: {take_profit}"
        )
        
        # Get current price
        current_price_data = await self.get_current_price(symbol)
        if not current_price_data:
            return {
                'success': False,
                'error': 'Could not get current price',
                'platform': 'mt5'
            }
        
        bid = current_price_data['bid']
        ask = current_price_data['ask']
        
        logger.info(f"Current prices - Bid: {bid}, Ask: {ask}")
        
        # Get symbol info for stops validation
        symbol_info = await self.get_symbol_info(symbol)
        
        if symbol_info:
            stops_level = symbol_info['stops_level']
            point = symbol_info['point']
            
            logger.info(f"Symbol info - Stops level: {stops_level}, Point: {point}")
            
            # Validate and adjust stops
            if direction == 'long':
                entry_price = ask
                if stop_loss and (entry_price - stop_loss) < stops_level * point:
                    old_sl = stop_loss
                    stop_loss = entry_price - (stops_level * point * 1.5)  # Add 50% buffer
                    logger.warning(
                        f"Adjusted SL from {old_sl} to {stop_loss} "
                        f"(min distance: {stops_level * point})"
                    )
                    
                if take_profit and (take_profit - entry_price) < stops_level * point:
                    old_tp = take_profit
                    take_profit = entry_price + (stops_level * point * 1.5)
                    logger.warning(f"Adjusted TP from {old_tp} to {take_profit}")
            else:  # short
                entry_price = bid
                if stop_loss and (stop_loss - entry_price) < stops_level * point:
                    old_sl = stop_loss
                    stop_loss = entry_price + (stops_level * point * 1.5)
                    logger.warning(
                        f"Adjusted SL from {old_sl} to {stop_loss} "
                        f"(min distance: {stops_level * point})"
                    )
                    
                if take_profit and (entry_price - take_profit) < stops_level * point:
                    old_tp = take_profit
                    take_profit = entry_price - (stops_level * point * 1.5)
                    logger.warning(f"Adjusted TP from {old_tp} to {take_profit}")
        
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
        
        response = await self._send_command(command, timeout=2.0)
        
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
                'error_code': response.get('code'),
                'timestamp': datetime.utcnow().isoformat(),
                'platform': 'mt5'
            }
            logger.error(f" Order failed: {result['error']}")
            
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
        """Close position."""
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