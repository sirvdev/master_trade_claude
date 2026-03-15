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
from click import command
import pandas as pd

logger = logging.getLogger(__name__)

# ── Module-level singleton — shared across ALL MT5FileBridge instances ─────────
# This is critical: both the execution bridge and the market data bridge
# create separate instances but use the same physical command/response files.
# Without a shared lock they corrupt each other's commands under concurrency.
_MT5_GLOBAL_LOCK: Optional[asyncio.Lock] = None

_MT5_PREFIX   = "main"
# MT5_CMD_FILE_PATTERN = f"python_command_{MT5_SESSION_PREFIX}_"   # + request_id + ".txt"
# MT5_RESP_FILE_PREFIX = f"python_response_{MT5_SESSION_PREFIX}_"   # + request_id + ".txt"
# MT5_STATUS_FILE      = f"mt5_status_{MT5_SESSION_PREFIX}.txt"
# MT5_SESSION_FILE     = f"python_session_{MT5_SESSION_PREFIX}.txt"

def _get_mt5_global_lock() -> asyncio.Lock:
    """Lazy-init the shared lock (must be called from inside a running event loop)."""
    global _MT5_GLOBAL_LOCK
    if _MT5_GLOBAL_LOCK is None:
        _MT5_GLOBAL_LOCK = asyncio.Lock()
    return _MT5_GLOBAL_LOCK

class MT5FileBridge:
    """
    MT5 execution bridge using single-file response system.
    Each request gets a unique ID, all responses concatenated in one file.
    """

    def __init__(self, config: Dict, demo_mode: bool = True):
        self.config = config
        self.demo_mode = (config.get('mode') == 'demo')
        self.magic_number = config.get('magic_number', 654321)

        # self.session_id = str(uuid.uuid4())[:8]
        self.common_path = self._find_mt5_common_path()

        self.session_id   = f"{_MT5_PREFIX}_{str(uuid.uuid4())[:8]}"
        # command_file removed — each request writes its own file in _send_command
        self.status_file  = self.common_path / f"mt5_status_{_MT5_PREFIX}.txt"
        self.session_file = self.common_path / f"python_session_{_MT5_PREFIX}.txt"

        self._connected = False
        self.request_counter = 0


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

        await asyncio.sleep(1.0)

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
        """
        Send command to EA and wait for response.

        Lock design — critical section is very narrow (just the write):
          The EA's ProcessCommandFile() does:
            1. read command file
            2. FileDelete(commandFile)   ← happens immediately, before any processing
            3. process command (slow: CopyRates, order placement, etc.)
            4. AppendResponse()          ← appends tagged line to response file

          Once the file is deleted (step 2) the EA holds the command in memory.
          It's therefore safe for Python to issue another command immediately;
          we only need a lock to prevent two coroutines from writing at the same
          instant.  We no longer wait for EA pickup because per-request-ID
          responses avoid collisions.

          The response file is append‑only and every line is tagged with
          request_id, so overlapping commands never collide.

        Lock hold time: <5ms (just the file write) instead of up to 30s.
        """
        if self.demo_mode:
            return await self._simulate_command(command)

        # ── Assign request_id and write to command file (under lock).  We no
        # longer wait for the EA to delete the command file; per-request-ID
        # responses make overlapping commands safe.  Lock only protects the
        # write itself so two coroutines don't stomp each other.
        async with _get_mt5_global_lock():
            self.request_counter += 1
            request_id = f"{self.session_id}_{self.request_counter}"
            command['request_id'] = request_id

            try:
                cmd_file = self.common_path / f"python_command_{request_id}.txt"
                cmd_file.write_text(json.dumps(command, ensure_ascii=True), encoding='utf-8')
                logger.debug(f"[BRIDGE] → {request_id}: {command.get('action')}")
            except Exception as e:
                logger.error(f"[BRIDGE] Error writing command {request_id}: {e}")
                return {'status': 'error', 'error': str(e)}

        # ── Lock released immediately after write.  EA pickup no longer blocks
        # further commands.

        # ── Lock released — poll for response outside the lock ────────────────
        # The response file accumulates tagged lines; we find ours by request_id.
        # Other commands can now proceed in parallel through the lock.
        poll_start = time.time()
        while time.time() - poll_start < timeout:
            response = await self._read_response_for_id(request_id)
            if response:
                logger.debug(
                    f"[BRIDGE] ← {request_id}: {response.get('status')} "
                    f"({time.time() - poll_start:.2f}s)"
                )
                return response
            await asyncio.sleep(0.05)

        logger.error(
            f"[BRIDGE] {request_id} ({command.get('action')}) "
            f"timed out after {timeout}s"
        )
        return {'status': 'error', 'error': 'timeout'}

    async def _read_response_for_id(self, request_id: str) -> Optional[Dict]:
        response_file = self.common_path / f"python_response_{request_id}.txt"
        try:
            if not response_file.exists():
                return None
            content = response_file.read_text(encoding='utf-8', errors='ignore').strip()
            response_file.unlink(missing_ok=True)   # self-cleaning
            return json.loads(content)
        except (json.JSONDecodeError, PermissionError, FileNotFoundError):
            return None
        except Exception as e:
            logger.debug(f"Error reading response for {request_id}: {e}")
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
            'ticket': ticket
        }
        if stop_loss is not None:
            command['sl'] = stop_loss
        if take_profit is not None:
            command['tp'] = take_profit
        
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

    async def get_all_positions(self) -> list | None:
        """
        Get all open positions.

        Returns:
            List of position dicts if the bridge call succeeded (may be empty
            if there are genuinely no open positions).
            None if the bridge call failed (timeout, EA unresponsive, etc.).
            Callers MUST check for None before processing closes — an empty
            list from a failed call is indistinguishable from "no positions"
            otherwise, causing false external-close events.
        """
        command = {
            'action': 'get_all_positions',
            'magic':  self.magic_number
        }

        response = await self._send_command(command)

        if response.get('status') == 'success':
            return response.get('positions', [])
        else:
            logger.warning(
                f"get_all_positions failed: {response.get('error', 'unknown')} "
                f"— returning None to prevent false close detection"
            )
            return None

    async def get_balance(self) -> Dict:
            """
            Fetch live account balance and equity from MT5 via the authenticate action.
            The EA's HandleAuthenticate() returns account, balance, and equity.

            Returns:
                Dict with keys: success (bool), balance (float), equity (float), account (int)
            """
            response = await self._send_command({'action': 'authenticate'})

            if response.get('status') == 'success':
                return {
                    'success': True,
                    'balance': response.get('balance', 0.0),
                    'equity' : response.get('equity',  0.0),
                    'account': response.get('account', 0),
                }
            else:
                logger.warning(f"get_balance failed: {response.get('error')}")
                return {'success': False, 'balance': 0.0, 'equity': 0.0}

    async def get_deal_history(self, ticket: int, lookback_hours: int = 48) -> dict:
        """
        Fetch all closing deal records for a given position ticket from MT5.
        Fixed EA returns a 'deals' array covering all partial closes.
        """
        if self.demo_mode:
            return {
                'status'              : 'success',
                'ticket'              : ticket,
                'deal_count'          : 1,
                'total_profit'        : 0.0,
                'total_volume_closed' : 0.0,
                'exit_price'          : 0.0,   # convenience field
                'profit'              : 0.0,
                'exit_reason'         : 'demo_close',
                'deals'               : []
            }

        now      = int(time.time())
        from_ts  = now - (lookback_hours * 3600)

        response = await self._send_command({
            'action'    : 'get_deal_history',
            'ticket'    : ticket,
            'from_time' : from_ts,
            'to_time'   : now + 3600,
        })

        if response.get('status') != 'success':
            logger.warning(f"get_deal_history failed for ticket {ticket}: {response.get('error')}")
            return {
                'status'  : 'error',
                'ticket'  : ticket,
                'profit'  : 0.0,
                'deals'   : [],
                'error'   : response.get('error', 'unknown')
            }

        deals = response.get('deals', [])

        # Convenience scalars derived from the deals array
        # Use the LAST closing deal's price as the representative exit price
        last_deal      = deals[-1] if deals else {}
        total_profit   = response.get('total_profit', 0.0)
        exit_price     = last_deal.get('exit_price', 0.0)
        exit_time      = last_deal.get('exit_time', 0)

        logger.info(
            f"get_deal_history ticket={ticket}: "
            f"{response.get('deal_count', 0)} deal(s), "
            f"total_profit={total_profit:.2f}, "
            f"exit_price={exit_price}"
        )

        return {
            'status'              : 'success',
            'ticket'              : ticket,
            'deal_count'          : response.get('deal_count', len(deals)),
            'total_profit'        : total_profit,
            'total_volume_closed' : response.get('total_volume_closed', 0.0),
            'exit_price'          : exit_price,       # last partial close price
            'profit'              : total_profit,     # alias for callers expecting 'profit'
            'exit_time'           : exit_time,
            'exit_reason'         : 'closed',
            'deals'               : deals,            # full array for audit logging
        }
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
            return {'status': 'success', 'account': 99999, 'balance': 100000.0, 'equity': 100000.0}

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

        if action == 'get_deal_history':
            ticket = command.get('ticket', 0)
            return {
                'status'     : 'success',
                'ticket'     : ticket,
                'entry_price': 0.0,
                'exit_price' : 0.0,
                'volume'     : 0.0,
                'profit'     : 0.0,
                'swap'       : 0.0,
                'commission' : 0.0,
                'net_profit' : 0.0,
                'close_time' : 0,
                'exit_reason': 'demo_close',
            }

        if action == 'authenticate':
            return {'status': 'success', 'account': 99999, 'balance': 10000.0, 'equity': 10000.0}

        return {'status': 'error', 'error': f'Unknown action: {action}'}