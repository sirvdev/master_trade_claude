"""
MT5 File-based bridge with single concatenated response file.
v2.1: Added fetch_historical_range for backtest date-range fetching.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Optional
from datetime import datetime, timezone
import time
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class MT5FileBridge:
    """
    MT5 execution bridge using single-file response system.
    Each request gets a unique ID, all responses concatenated in one file.
    """

    def __init__(self, config: Dict, demo_mode: bool = True):
        self.config = config
        self.demo_mode = demo_mode
        self.magic_number = config.get('magic_number', 123456)

        self.session_id = str(uuid.uuid4())[:8]
        self.common_path = self._find_mt5_common_path()

        self.command_file  = self.common_path / "python_command.txt"
        self.response_file = self.common_path / "python_responses.txt"
        self.status_file   = self.common_path / "mt5_status.txt"
        self.session_file  = self.common_path / "python_session.txt"

        self._connected = False
        self.request_counter = 0
        self.last_read_position = 0

        self.demo_orders    = {}
        self.demo_positions = {}

    def _find_mt5_common_path(self) -> Path:
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
        if self.demo_mode:
            logger.info("MT5 File Bridge running in DEMO mode")
            self._connected = True
            return

        try:
            self.session_file.write_text(self.session_id, encoding='utf-8')
            logger.info(f"Session ID: {self.session_id}")
        except Exception as e:
            logger.error(f"Could not write session file: {e}")

        if self.status_file.exists():
            try:
                status = self.status_file.read_text(encoding='utf-8', errors='ignore')
                status = status.lstrip('\ufeff').strip()
                if 'ready' in status.lower():
                    logger.info("MT5 EA is ready")
                    self._connected = True
                else:
                    logger.warning(f"MT5 EA status: {status}")
                    self._connected = True
            except Exception as e:
                logger.error(f"Error reading status: {e}")
                self._connected = False
        else:
            logger.warning("MT5 status file not found - is the EA running?")
            self._connected = False

    async def disconnect(self):
        self._connected = False
        logger.info("MT5 File Bridge disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Core command/response layer
    # ------------------------------------------------------------------

    async def _send_command(self, command: Dict, timeout: float = 30.0) -> Dict:
        """Send command with unique ID and wait for response."""
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

            logger.error(f"Command {request_id} timed out after {timeout}s")
            return {'status': 'error', 'error': 'timeout'}

        except Exception as e:
            logger.error(f"Error sending command: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    async def _read_response_for_id(self, request_id: str) -> Optional[Dict]:
        try:
            if not self.response_file.exists():
                return None

            content = self.response_file.read_text(encoding='utf-8', errors='ignore')
            content = content.lstrip('\ufeff')

            for line in reversed(content.strip().split('\n')):
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                    if resp.get('request_id') == request_id:
                        return resp
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.debug(f"Error reading response: {e}")

        return None

    # ------------------------------------------------------------------
    # Historical data methods
    # ------------------------------------------------------------------

    async def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500
    ) -> pd.DataFrame:
        """
        Fetch the last N closed bars for a symbol/timeframe (live mode use).

        Args:
            symbol:    e.g. 'XAUUSD'
            timeframe: e.g. '1m', '15m', '1h'
            count:     number of bars to fetch

        Returns:
            DataFrame indexed by timestamp (UTC), columns: open high low close volume
        """
        tf_map = {
            '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
            '1h': 'H1', '4h': 'H4', '1d': 'D1'
        }
        mt5_tf = tf_map.get(timeframe.lower(), 'H1')

        command = {
            'action':    'get_historical',
            'symbol':    symbol.replace('/', ''),
            'timeframe': mt5_tf,
            'count':     count + 1   # +1 to drop current forming bar
        }

        response = await self._send_command(command)

        if response.get('status') != 'success':
            raise ValueError(
                f"fetch_historical failed for {symbol} {timeframe}: "
                f"{response.get('error')}"
            )

        return self._bars_to_df(response.get('data', []), drop_last=True)

    async def fetch_historical_range(
        self,
        symbol: str,
        timeframe: str,
        from_date: datetime,
        to_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch ALL bars between from_date and to_date (bulk backtest fetch).

        The EA uses CopyRates(symbol, tf, from_date, to_date) which returns
        the full range in one call — no bar-count limit.

        Args:
            symbol:    e.g. 'XAUUSD'
            timeframe: e.g. '1m', '15m', '1h'
            from_date: Start datetime (UTC)
            to_date:   End datetime (UTC)

        Returns:
            DataFrame indexed by timestamp (UTC), columns: open high low close volume
        """
        tf_map = {
            '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
            '1h': 'H1', '4h': 'H4', '1d': 'D1'
        }
        mt5_tf = tf_map.get(timeframe.lower(), 'H1')

        # Convert to UTC unix timestamps
        from_ts = int(from_date.replace(tzinfo=timezone.utc).timestamp()) \
                  if from_date.tzinfo is None \
                  else int(from_date.timestamp())
        to_ts   = int(to_date.replace(tzinfo=timezone.utc).timestamp()) \
                  if to_date.tzinfo is None \
                  else int(to_date.timestamp())

        logger.info(
            f"Fetching {symbol} {timeframe} range: "
            f"{from_date.strftime('%Y-%m-%d')} → {to_date.strftime('%Y-%m-%d')}"
        )

        command = {
            'action':    'get_historical_range',
            'symbol':    symbol.replace('/', ''),
            'timeframe': mt5_tf,
            'from_date': from_ts,
            'to_date':   to_ts
        }

        # Large date ranges can take a few seconds — use longer timeout
        response = await self._send_command(command, timeout=60.0)

        if response.get('status') != 'success':
            raise ValueError(
                f"fetch_historical_range failed for {symbol} {timeframe} "
                f"{from_date} → {to_date}: {response.get('error')}"
            )

        df = self._bars_to_df(response.get('data', []), drop_last=False)

        logger.info(
            f"  ✓ {symbol} {timeframe}: {len(df)} bars fetched "
            f"({df.index[0]} → {df.index[-1]})"
        )
        return df

    # ------------------------------------------------------------------
    # Live trading methods (order execution & position management)
    # ------------------------------------------------------------------

    async def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current bid/ask prices for a symbol.
        
        Args:
            symbol: Symbol (e.g. 'XAUUSD')
            
        Returns:
            Dict with 'bid', 'ask', 'spread' or None on error
        """
        # Get the last bar to extract current price
        try:
            df = await self.fetch_historical(symbol, '1m', count=1)
            if df.empty:
                return None
            
            last_bar = df.iloc[-1]
            # Approximate bid/ask from close (MT5 doesn't provide tick data via file bridge)
            close = float(last_bar['close'])
            # Typical gold spread is ~0.30, use that as approximation
            spread = 0.30
            
            return {
                'symbol': symbol,
                'bid': close - spread / 2,
                'ask': close + spread / 2,
                'spread': spread,
                'time': last_bar.name
            }
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
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
        """
        Place an order on MT5.
        
        Args:
            symbol:       e.g. 'XAUUSD'
            direction:    'long' or 'short'
            volume:       Lot size (e.g. 0.1)
            order_type:   'market' or 'limit'
            price:        Limit price (required for limit orders)
            stop_loss:    SL price
            take_profit:  TP price
            comment:      Order comment
            
        Returns:
            Result dict with 'success', 'ticket', 'filled_price', etc.
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
            'action':     'place_order',
            'symbol':     symbol,
            'order_type': mt5_order_type,
            'volume':     volume,
            'price':      price or 0,
            'sl':         stop_loss or 0,
            'tp':         take_profit or 0,
            'comment':    comment or 'Python'
        }
        
        response = await self._send_command(command, timeout=10.0)
        
        if response.get('status') == 'success':
            result = {
                'success':       True,
                'order_id':      f"mt5_{response.get('ticket')}",
                'ticket':        response.get('ticket'),
                'filled_price':  response.get('price'),
                'price':         response.get('price'),
                'filled_volume': volume,
                'timestamp':     datetime.utcnow().isoformat(),
                'platform':      'mt5',
                'demo_mode':     self.demo_mode
            }
            logger.info(f"Order placed: Ticket {result['ticket']}")
        else:
            result = {
                'success':   False,
                'error':     response.get('error', 'Unknown error'),
                'timestamp': datetime.utcnow().isoformat(),
                'platform':  'mt5'
            }
            logger.error(f"Order failed: {result['error']}")
        
        return result

    async def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Modify position SL/TP.
        
        Args:
            ticket:      MT5 position ticket
            stop_loss:   New SL price
            take_profit: New TP price
            
        Returns:
            Modification result
        """
        logger.info(f"Modifying position {ticket}: SL={stop_loss}, TP={take_profit}")
        
        command = {
            'action': 'modify_position',
            'ticket': ticket,
            'sl':     stop_loss or 0,
            'tp':     take_profit or 0
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
        """
        Close position (full or partial).
        
        Args:
            ticket:  MT5 position ticket
            volume:  Volume to close (None = full close)
            comment: Close comment
            
        Returns:
            Close result
        """
        logger.info(f"Closing position {ticket}, volume: {volume or 'full'}")
        
        command = {
            'action':  'close_position',
            'ticket':  ticket,
            'volume':  volume or 0,
            'comment': comment or 'Python'
        }
        
        response = await self._send_command(command, timeout=10.0)
        
        if response.get('status') == 'success':
            logger.info(f"Position {ticket} closed")
            return {
                'success':       True,
                'ticket':        ticket,
                'closed_volume': volume or 0,
                'profit':        response.get('profit', 0)
            }
        else:
            logger.error(f"Close failed: {response.get('error')}")
            return {'success': False, 'error': response.get('error')}

    async def get_position_info(self, ticket: int) -> Optional[Dict]:
        """
        Get position information.
        
        Args:
            ticket: MT5 position ticket
            
        Returns:
            Position dict or None
        """
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
        """
        Get all open positions.
        
        Returns:
            List of position dicts
        """
        command = {
            'action': 'get_all_positions',
            'magic':  self.magic_number
        }
        
        response = await self._send_command(command)
        
        if response.get('status') == 'success':
            return response.get('positions', [])
        else:
            return []

    # ------------------------------------------------------------------
    # Helper: convert raw bar list to DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_df(bars: list, drop_last: bool = False) -> pd.DataFrame:
        """
        Convert list of [time, open, high, low, close, volume] to DataFrame.

        Args:
            bars:      Raw bar data from EA
            drop_last: If True, drop the last (currently forming) bar

        Returns:
            DataFrame with UTC timestamp index
        """
        if not bars:
            return pd.DataFrame(
                columns=['open', 'high', 'low', 'close', 'volume']
            )

        df = pd.DataFrame(
            bars,
            columns=['time', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
        df.drop('time', axis=1, inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        if drop_last and len(df) > 1:
            df = df.iloc[:-1]

        return df

    # ------------------------------------------------------------------
    # Demo-mode simulation
    # ------------------------------------------------------------------

    async def _simulate_command(self, command: Dict) -> Dict:
        """Simulate EA responses for demo/test mode."""
        action = command.get('action')
        await asyncio.sleep(0.01)

        if action == 'ping':
            return {'status': 'success', 'message': 'pong'}

        if action == 'authenticate':
            return {'status': 'success', 'account': 99999, 'balance': 10000.0, 'equity': 10000.0}

        if action in ('get_historical', 'get_historical_range'):
            # Generate synthetic bars for demo
            import numpy as np
            count = command.get('count', 500)
            if action == 'get_historical_range':
                from_ts = command.get('from_date', 0)
                to_ts   = command.get('to_date', 0)
                tf_seconds = {'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
                              'H1': 3600, 'H4': 14400, 'D1': 86400}
                tf_secs = tf_seconds.get(command.get('timeframe', 'H1'), 3600)
                count = max(1, (to_ts - from_ts) // tf_secs)
                start_ts = from_ts
            else:
                tf_seconds = {'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
                              'H1': 3600, 'H4': 14400, 'D1': 86400}
                tf_secs  = tf_seconds.get(command.get('timeframe', 'H1'), 3600)
                start_ts = int(time.time()) - count * tf_secs

            np.random.seed(42)
            price = 2000.0
            bars  = []
            for i in range(count):
                ts    = start_ts + i * tf_secs
                chg   = np.random.normal(0, 0.5)
                price = max(price + chg, 1000.0)
                rng   = abs(np.random.normal(0, 0.3))
                bars.append([ts, round(price, 2), round(price + rng, 2),
                              round(price - rng, 2), round(price, 2), 1000])

            return {'status': 'success', 'count': count, 'data': bars}

        if action == 'place_order':
            ticket = len(self.demo_orders) + 1
            price  = 2000.0
            self.demo_orders[ticket] = command
            self.demo_positions[ticket] = {'ticket': ticket, 'price': price, **command}
            return {'status': 'success', 'ticket': ticket, 'price': price, 'volume': command.get('volume', 0.1)}

        if action == 'modify_position':
            ticket = command.get('ticket')
            if ticket in self.demo_positions:
                self.demo_positions[ticket].update({'sl': command.get('sl'), 'tp': command.get('tp')})
                return {'status': 'success', 'ticket': ticket}
            return {'status': 'error', 'error': 'Position not found'}

        if action == 'close_position':
            ticket = command.get('ticket')
            if ticket in self.demo_positions:
                del self.demo_positions[ticket]
                return {'status': 'success', 'ticket': ticket}
            return {'status': 'error', 'error': 'Position not found'}

        if action == 'get_all_positions':
            positions = list(self.demo_positions.values())
            return {'status': 'success', 'positions': positions}

        return {'status': 'error', 'error': f'Unknown action: {action}'}