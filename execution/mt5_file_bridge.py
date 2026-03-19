"""
execution/mt5_file_bridge.py
============================
MT5 File-based bridge — Per-Request-ID response system.

Changes v2.1+:
  - Retry decorator applied to place_order (3×,2s), close_position (5×,1s),
    cancel_order (3×,1s), get_deal_history (3×,5s).
  - Added get_all_orders() and cancel_order() — were missing from main bridge,
    causing AttributeError in _check_pending_limit_orders every 15s.
  - Retry only on genuine failure (status!=success or exception).
    Never retries on success.
"""

import asyncio
import json
import logging
import os
import uuid
import time
import functools
from typing import Dict, Optional
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Module-level singleton lock shared across ALL MT5FileBridge instances ──────
# Both the execution bridge and the market data bridge share the same physical
# command/response files. Without a shared lock they corrupt each other.
_MT5_GLOBAL_LOCK: Optional[asyncio.Lock] = None
_MT5_PREFIX = "main"


def _get_mt5_global_lock() -> asyncio.Lock:
    global _MT5_GLOBAL_LOCK
    if _MT5_GLOBAL_LOCK is None:
        _MT5_GLOBAL_LOCK = asyncio.Lock()
    return _MT5_GLOBAL_LOCK


# ── Retry decorator ───────────────────────────────────────────────────────────

def _with_retry(max_attempts: int, backoff_seconds: float):
    """
    Async retry decorator for bridge methods.

    Retries only when the response has status != 'success' OR an exception
    is raised. Never retries on success. Uses fixed backoff between attempts.

    Usage:
        @_with_retry(max_attempts=3, backoff_seconds=2.0)
        async def place_order(self, ...): ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_result = None
            last_exc    = None
            for attempt in range(1, max_attempts + 1):
                try:
                    result = await fn(*args, **kwargs)
                    if result.get('success') or result.get('status') == 'success':
                        return result
                    # Non-success — record and retry
                    last_result = result
                    err = result.get('error', 'unknown error')
                    logger.warning(
                        f"[RETRY] {fn.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {err}"
                    )
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        f"[RETRY] {fn.__name__} attempt {attempt}/{max_attempts} "
                        f"raised: {exc}"
                    )

                if attempt < max_attempts:
                    await asyncio.sleep(backoff_seconds)

            # All attempts exhausted
            if last_exc is not None:
                logger.error(
                    f"[RETRY] {fn.__name__} failed after {max_attempts} attempts: "
                    f"{last_exc}"
                )
                return {'success': False, 'error': str(last_exc)}

            logger.error(
                f"[RETRY] {fn.__name__} failed after {max_attempts} attempts: "
                f"{last_result.get('error', 'unknown')}"
            )
            return last_result or {'success': False, 'error': 'max retries exceeded'}

        return wrapper
    return decorator


# ── Bridge class ──────────────────────────────────────────────────────────────

class MT5FileBridge:
    """
    MT5 execution bridge using per-request command/response files.
    Each request gets a unique ID; response file is self-cleaning.
    """

    def __init__(self, config: Dict, demo_mode: bool = True):
        self.config       = config
        self.demo_mode    = (config.get('mode') == 'demo')
        self.magic_number = config.get('magic_number', 654321)

        self.common_path  = self._find_mt5_common_path()
        self.session_id   = f"{_MT5_PREFIX}_{str(uuid.uuid4())[:8]}"
        self.status_file  = self.common_path / f"mt5_status_{_MT5_PREFIX}.txt"
        self.session_file = self.common_path / f"python_session_{_MT5_PREFIX}.txt"

        self._connected      = False
        self.request_counter = 0

        # Demo state
        self.demo_orders    = {}
        self.demo_positions = {}

    def _find_mt5_common_path(self) -> Path:
        possible = [
            Path(os.environ.get('APPDATA', '')) / "MetaQuotes" / "Terminal" / "Common" / "Files",
            Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal" / "Common" / "Files",
        ]
        for p in possible:
            if p.exists():
                logger.info(f"Found MT5 Common Files at: {p}")
                return p
        default = possible[0]
        default.mkdir(parents=True, exist_ok=True)
        return default

    # ── Connection ─────────────────────────────────────────────────────────────

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
            logger.warning("MT5 status file not found — is the EA running?")

    async def disconnect(self):
        self._connected = False
        logger.info("MT5 File Bridge disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ── Core bridge transport ──────────────────────────────────────────────────

    async def _send_command(self, command: Dict, timeout: float = 30.0) -> Dict:
        if self.demo_mode:
            return await self._handle_demo_command(command)

        self.request_counter += 1
        request_id = f"{self.session_id}_{self.request_counter}"
        command['request_id'] = request_id
        command['magic']      = self.magic_number

        cmd_file = self.common_path / f"python_command_{request_id}.txt"

        lock = _get_mt5_global_lock()
        async with lock:
            try:
                cmd_file.write_text(
                    json.dumps(command, ensure_ascii=True),
                    encoding='utf-8'
                )
            except Exception as e:
                logger.error(f"[BRIDGE] Failed to write command file: {e}")
                return {'status': 'error', 'error': str(e)}

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
            response_file.unlink(missing_ok=True)
            return json.loads(content)
        except (json.JSONDecodeError, PermissionError, FileNotFoundError):
            return None
        except Exception as e:
            logger.debug(f"Error reading response for {request_id}: {e}")
            return None

    # ── Demo handler ───────────────────────────────────────────────────────────

    async def _handle_demo_command(self, command: Dict) -> Dict:
        import numpy as np
        action = command.get('action', '')

        if action == 'get_historical':
            symbol    = command.get('symbol', 'DEMO')
            timeframe = command.get('timeframe', 'M15')
            count     = int(command.get('count', 100))

            tf_map = {'M1':60,'M5':300,'M15':900,'M30':1800,
                      'H1':3600,'H4':14400,'D1':86400}
            tf_secs   = tf_map.get(timeframe, 900)
            start_ts  = int(time.time()) - count * tf_secs

            np.random.seed(42)
            price = 2000.0
            bars  = []
            for i in range(count):
                ts    = start_ts + i * tf_secs
                chg   = np.random.normal(0, 0.5)
                price = max(price + chg, 1000.0)
                rng   = abs(np.random.normal(0, 0.3))
                bars.append([ts, round(price,2), round(price+rng,2),
                              round(price-rng,2), round(price,2), 1000])
            return {'status':'success','count':count,'data':bars}

        if action == 'place_order':
            ticket = len(self.demo_orders) + 1
            price  = 2000.0
            self.demo_orders[ticket]    = command
            self.demo_positions[ticket] = {'ticket':ticket,'price':price,**command}
            return {'status':'success','ticket':ticket,'price':price,
                    'volume':command.get('volume',0.1)}

        if action == 'modify_position':
            ticket = command.get('ticket')
            if ticket in self.demo_positions:
                self.demo_positions[ticket].update(
                    {'sl': command.get('sl'), 'tp': command.get('tp')}
                )
                return {'status':'success','ticket':ticket}
            return {'status':'error','error':'Position not found'}

        if action == 'close_position':
            ticket = command.get('ticket')
            if ticket in self.demo_positions:
                del self.demo_positions[ticket]
                return {'status':'success','ticket':ticket}
            return {'status':'error','error':'Position not found'}

        if action == 'get_all_positions':
            return {'status':'success','positions':list(self.demo_positions.values())}

        if action == 'get_all_orders':
            return {'status':'success','count':0,'orders':[]}

        if action == 'cancel_order':
            ticket = command.get('ticket')
            self.demo_orders.pop(ticket, None)
            return {'status':'success','ticket':ticket}

        if action == 'get_deal_history':
            ticket = command.get('ticket', 0)
            return {
                'status':'success','ticket':ticket,
                'entry_price':0.0,'exit_price':0.0,
                'volume':0.0,'profit':0.0,'swap':0.0,
                'commission':0.0,'net_profit':0.0,
                'close_time':0,'exit_reason':'demo_close',
            }

        if action == 'authenticate':
            return {'status':'success','account':99999,
                    'balance':10000.0,'equity':10000.0}

        return {'status':'error','error':f'Unknown action: {action}'}

    # ── Historical data ────────────────────────────────────────────────────────

    async def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        count: int = 250,
    ) -> pd.DataFrame:
        """Fetch the last N closed bars for a symbol/timeframe.

        Timeout scales with timeframe size — 4H/1D bars produce much larger
        response payloads than 1m/5m bars, causing timeouts at the default 30s.
        """
        tf_map = {
            '1m':'M1','5m':'M5','15m':'M15','30m':'M30',
            '1H':'H1','4H':'H4','1D':'D1','1W':'W1',
            'M1':'M1','M5':'M5','M15':'M15','H1':'H1','H4':'H4','D1':'D1',
        }
        mt5_tf = tf_map.get(timeframe, timeframe)

        # Longer TFs have larger payloads — scale timeout accordingly
        timeout_map = {
            'M1': 30.0, 'M5': 30.0, 'M15': 30.0, 'M30': 30.0,
            'H1': 45.0, 'H4': 60.0, 'D1': 60.0, 'W1': 60.0,
        }
        fetch_timeout = timeout_map.get(mt5_tf, 45.0)

        command = {
            'action':    'get_historical',
            'symbol':    symbol,
            'timeframe': mt5_tf,
            'count':     count,
        }

        response = await self._send_command(command, timeout=fetch_timeout)

        if response.get('status') != 'success':
            raise ValueError(f"Failed to fetch data: {response.get('error','unknown')}")

        bars = response.get('data', [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.set_index('timestamp')
        df = df.astype(float)
        logger.info(f"Fetched {len(df)} candles for {symbol} from MT5")
        return df

    async def fetch_historical_range(
        self,
        symbol: str,
        timeframe: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> pd.DataFrame:
        tf_map = {
            '1m':'M1','5m':'M5','15m':'M15','30m':'M30',
            '1H':'H1','4H':'H4','1D':'D1',
        }
        mt5_tf = tf_map.get(timeframe, timeframe)

        command = {
            'action':    'get_historical_range',
            'symbol':    symbol,
            'timeframe': mt5_tf,
            'from':      int(from_dt.timestamp()),
            'to':        int(to_dt.timestamp()),
        }

        response = await self._send_command(command, timeout=60.0)

        if response.get('status') != 'success':
            raise ValueError(f"Failed to fetch range: {response.get('error','unknown')}")

        bars = response.get('data', [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.set_index('timestamp')
        df = df.astype(float)
        return df

    # ── Order / position operations ────────────────────────────────────────────

    @_with_retry(max_attempts=3, backoff_seconds=2.0)
    async def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        order_type: str = 'market',
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> Dict:
        """
        Place a market or limit order on MT5.
        Retries up to 3 times with 2-second backoff on failure.
        Covers both market and limit orders (limit = place_order with price set).
        """
        logger.info(
            f"Placing MT5 order: {symbol} {direction} {volume} lots "
            f"type={order_type} SL={stop_loss} TP={take_profit}"
        )

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
            'comment':    comment or 'Python',
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
                'timestamp':     datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                'platform':      'mt5',
                'demo_mode':     self.demo_mode,
            }
            logger.info(f"Order placed: Ticket {result['ticket']}")
        else:
            result = {
                'success':   False,
                'error':     response.get('error', 'Unknown error'),
                'timestamp': datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                'platform':  'mt5',
            }
            logger.error(f"Order failed: {result['error']}")

        return result

    async def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict:
        """
        Modify position SL/TP.
        No retry — only used for emergency manual SL adjustments.
        Trailing and breakeven are now fully managed by the EA.
        """
        logger.info(f"Modifying position {ticket}: SL={stop_loss}, TP={take_profit}")

        command = {'action': 'modify_position', 'ticket': ticket}
        if stop_loss is not None:
            command['sl'] = stop_loss
        if take_profit is not None:
            command['tp'] = take_profit

        response = await self._send_command(command)

        if response.get('status') == 'success':
            return {'success': True, 'ticket': ticket}
        else:
            logger.error(f"Modify failed: {response.get('error')}")
            return {'success': False, 'error': response.get('error')}

    @_with_retry(max_attempts=5, backoff_seconds=1.0)
    async def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> Dict:
        """
        Close position (full or partial).
        Retries up to 5 times with 1-second backoff — missed close = exposure risk.
        """
        logger.info(f"Closing position {ticket}, volume: {volume or 'full'}")

        command = {
            'action':  'close_position',
            'ticket':  ticket,
            'volume':  volume or 0,
            'comment': comment or 'Python',
        }

        response = await self._send_command(command, timeout=10.0)

        if response.get('status') == 'success':
            logger.info(f"Position {ticket} closed")
            return {
                'success':       True,
                'ticket':        ticket,
                'closed_volume': volume or 0,
                'profit':        response.get('profit', 0),
            }
        else:
            logger.error(f"Close failed: {response.get('error')}")
            return {'success': False, 'error': response.get('error')}

    async def get_position_info(self, ticket: int) -> Optional[Dict]:
        response = await self._send_command({'action': 'get_position', 'ticket': ticket})
        return response if response.get('status') == 'success' else None

    async def get_all_positions(self) -> list | None:
        """
        Get all open positions managed by this EA (filtered by magic number).

        Returns:
            List of position dicts on success (may be empty = no positions).
            None if the bridge call failed — callers must check for None to
            avoid false external-close detection.
        """
        command = {'action': 'get_all_positions', 'magic': self.magic_number}
        response = await self._send_command(command)

        if response.get('status') == 'success':
            return response.get('positions', [])
        else:
            logger.warning(
                f"get_all_positions failed: {response.get('error', 'unknown')} "
                f"— returning None to prevent false close detection"
            )
            return None

    async def get_all_orders(self) -> list | None:
        """
        Get all pending limit/stop orders managed by this EA.

        Returns:
            List of order dicts on success, None on bridge failure.
        """
        command = {'action': 'get_all_orders', 'magic': self.magic_number}
        response = await self._send_command(command)

        if response.get('status') == 'success':
            return response.get('orders', [])
        else:
            logger.warning(
                f"get_all_orders failed: {response.get('error', 'unknown')}"
            )
            return None

    @_with_retry(max_attempts=3, backoff_seconds=1.0)
    async def cancel_order(self, ticket: int) -> Dict:
        """
        Cancel a pending limit or stop order.
        Retries up to 3 times with 1-second backoff.
        Returns dict with 'success' key.
        """
        logger.info(f"Cancelling order ticket={ticket}")

        command = {'action': 'cancel_order', 'ticket': ticket}
        response = await self._send_command(command)

        if response.get('status') == 'success':
            logger.info(f"Order {ticket} cancelled")
            return {'success': True, 'ticket': ticket}
        else:
            logger.error(f"Cancel failed ticket={ticket}: {response.get('error')}")
            return {'success': False, 'error': response.get('error')}

    async def get_deal_history(
        self,
        ticket: int,
        lookback_hours: int = 48,
    ) -> Dict:
        """
        Fetch all closing deal records for a given position ticket.
        No retry decorator — main.py._handle_external_close already runs its own
        3-attempt loop with growing lookback windows (48h → 96h → 192h).
        Adding a decorator here caused 9 EA calls instead of 3 and 30+ seconds
        of startup delay every time a trade was closed while the system was down.
        """
        now_ts   = int(time.time())
        from_ts  = now_ts - lookback_hours * 3600
        to_ts    = now_ts + 3600

        command = {
            'action':    'get_deal_history',
            'ticket':    ticket,
            'from_time': from_ts,
            'to_time':   to_ts,
        }

        response = await self._send_command(command, timeout=30.0)

        if response.get('status') == 'success':
            deals        = response.get('deals', [])
            total_profit = response.get('total_profit', 0.0)
            total_vol    = response.get('total_volume_closed', 0.0)
            deal_count   = response.get('deal_count', len(deals))

            logger.info(
                f"get_deal_history ticket={ticket}: {deal_count} deal(s), "
                f"total_profit={total_profit:.2f}, "
                f"exit_price={deals[-1].get('exit_price', 0) if deals else 0}"
            )

            last_deal = deals[-1] if deals else {}

            # Sum swap and commission across all partial deals (EA returns per-deal)
            total_swap       = sum(float(d.get('swap', 0.0))       for d in deals)
            total_commission = sum(float(d.get('commission', 0.0)) for d in deals)
            net_profit       = total_profit + total_swap + total_commission

            return {
                'status':       'success',   # main.py checks deal.get('status') == 'success'
                'success':      True,
                'ticket':       ticket,
                'deal_count':   deal_count,
                # Aggregated fields (new names)
                'total_profit': total_profit,
                # Backward-compat names that _compute_close_fields reads via deal.get(...)
                'profit':       total_profit,   # deal.get('profit', 0.0)
                'net_profit':   net_profit,     # deal.get('net_profit', gross+swap+commission)
                'swap':         total_swap,     # deal.get('swap', 0.0)
                'commission':   total_commission, # deal.get('commission', 0.0)
                'exit_reason':  last_deal.get('exit_reason', 'external_close'),
                'close_time':   last_deal.get('exit_time', 0),  # deal.get('close_time')
                # Price and time
                'exit_price':   last_deal.get('exit_price', 0.0),
                'exit_time':    last_deal.get('exit_time', 0),
                'deals':        deals,
            }
        else:
            logger.warning(
                f"get_deal_history ticket={ticket}: "
                f"{response.get('error', 'unknown error')}"
            )
            return {
                'status':  'error',
                'success': False,
                'error':   response.get('error', 'unknown'),
            }

    async def get_balance(self) -> Dict:
        """Fetch live account balance/equity via the authenticate action."""
        response = await self._send_command({'action': 'authenticate'})

        if response.get('status') == 'success':
            return {
                'success': True,
                'balance': response.get('balance', 0.0),
                'equity':  response.get('equity',  0.0),
                'account': response.get('account', 0),
            }
        else:
            logger.warning(f"get_balance failed: {response.get('error')}")
            return {'success': False, 'balance': 0.0, 'equity': 0.0}

    async def get_symbol_sessions(self, symbol: str) -> Dict:
        return await self._send_command(
            {'action': 'get_symbol_sessions', 'symbol': symbol},
            timeout=15.0,
        )

    async def get_current_price(self, symbol: str) -> Optional[Dict]:
        try:
            df = await self.fetch_historical(symbol, '1m', count=1)
            if df.empty:
                return None
            close  = float(df['close'].iloc[-1])
            spread = 0.30
            return {
                'symbol': symbol,
                'bid':    close - spread / 2,
                'ask':    close + spread / 2,
                'spread': spread,
                'time':   df.index[-1],
            }
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def subscribe_live(self, symbol: str, callback) -> None:
        logger.warning("Live subscriptions not supported with file bridge")

    async def close(self):
        await self.disconnect()