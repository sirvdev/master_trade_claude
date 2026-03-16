"""
Main orchestrator for the trading system.
Coordinates all components and manages the trading loop.
"""

import asyncio
import logging
import signal
import sys
import time as _monotime
import yaml
import os
import math
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import all modules
from logger.db import DatabaseManager
from logger.audit_logger import AuditLogger
from data_feed.market_client import MultiMarketClient
from indicators.indicators import TechnicalIndicators
from strategy.engine import StrategyEngine
from risk_management.money_manager import MoneyManager
from risk_management.stop_manager import StopManager
from execution.mt5_file_bridge import MT5FileBridge as MT5Bridge
from learning.learner import StrategyLearner
from utils.market_hours import MarketHoursChecker
from telegram_signal.channel_signal_bot import ChannelSignalBot, _limit_order_alert_watcher

# Setup logging
Path('logs').mkdir(exist_ok=True)
_file_handler   = logging.FileHandler('logs/trading_system.log', encoding='utf-8')
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[_file_handler, _stream_handler]
)
logger = logging.getLogger(__name__)


# ── Helper functions ───────────────────────────────────────────────────────────────


def _tf_to_seconds(tf: str) -> int:
    """Convert timeframe string to seconds."""
    mapping = {
        '1m': 60,    '5m': 300,   '15m': 900,  '30m': 1800,
        '1H': 3600,  '4H': 14400, '1D': 86400,
        # Aliases
        '1h': 3600,  '4h': 14400, '1d': 86400,
    }
    return mapping.get(tf, 900)   # default 15m
 
 
def _next_bar_close_utc(tf_secs: int) -> datetime:
    import time as _t
    now_ts        = _t.time()
    next_close_ts = math.ceil(now_ts / tf_secs) * tf_secs
    return datetime.fromtimestamp(next_close_ts, timezone.utc).replace(tzinfo=None)
 
 
def _limit_expiry_times(placed_at, primary_tf, entry_type, strategy_config):
    import time as _t
    tf_secs = _tf_to_seconds(primary_tf)
    exp_cfg = strategy_config.get("limit_order_expiry", {})
    bars_map = {
        "ema_stack_pullback":  exp_cfg.get("ema_stack_pullback_bars",  2),
        "pullback_to_sr":      exp_cfg.get("pullback_to_sr_bars",      5),
        "rsi_divergence":      exp_cfg.get("rsi_divergence_bars",      3),
        "bb_squeeze_breakout": exp_cfg.get("bb_squeeze_breakout_bars", 1),
    }
    expiry_bars = int(bars_map.get(entry_type, 3))
    floor_secs  = int(exp_cfg.get("min_cancel_floor_seconds", 60))
    placed_ts      = _t.time()
    next_bar_close = math.ceil(placed_ts / tf_secs) * tf_secs
    expiry_ts      = next_bar_close + expiry_bars * tf_secs
    expiry_time      = datetime.fromtimestamp(expiry_ts, timezone.utc).replace(tzinfo=None)
    min_cancel_floor = placed_at + timedelta(seconds=floor_secs)
    return expiry_time, min_cancel_floor


class TradingSystem:
    """TradingSystem Class Documentation
    A comprehensive trading system orchestrator that manages multi-platform trading operations,
    including market data fetching, strategy analysis, position management, and risk control.
    Attributes:
        config (dict): Configuration dictionary loaded from YAML file with environment overrides
        db (DatabaseManager): Database connection manager for persistent storage
        audit_logger (AuditLogger): Logger for trade and analysis audit trails
        market_client (MultiMarketClient): Client for fetching market data from multiple sources
        indicators (TechnicalIndicators): Technical analysis indicators calculator
        strategy_engine (StrategyEngine): Main strategy analysis engine
        money_manager (MoneyManager): Position sizing and risk validation engine
        stop_manager (StopManager): Stop loss and take profit management
        mt5_client (MT5Bridge): MetaTrader 5 execution and data client
        learner (StrategyLearner): Machine learning engine for strategy optimization
        running (bool): System execution state flag
        open_positions (dict): Dictionary of currently open trades tracked by trade_id
        daily_stats (dict): Daily performance metrics including trade count and drawdown
    Methods:
        __init__(config_path: str) -> None:
            Initialize trading system with configuration and all components.
        _apply_env_overrides() -> None:
            Apply environment variable overrides to configuration settings.
        _init_database() -> None:
            Initialize database connection and audit logging.
        _init_market_clients() -> None:
            Initialize multi-market data clients (MT5).
        _init_strategy_components() -> None:
            Initialize technical indicators, strategy engine, money manager, and stop manager.
        _init_execution_clients() -> None:
            Initialize order execution clients for both MT5 platforms.
        _init_learning_engine() -> None:
            Initialize machine learning engine if enabled in configuration.
        start() -> Coroutine:
            Main entry point - starts all async trading loops concurrently.
        _load_open_positions_from_db() -> Coroutine:
            Load positions from database on startup and verify with brokers.
        _trading_loop() -> Coroutine:
            Main trading loop that analyzes symbols and processes entry signals.
            Enforces position limits before entry validation.
        _process_entry_signal(symbol: str, symbol_config: dict, analysis: dict, 
                             multi_tf_data: dict) -> Coroutine:
            Process entry signal with final validation, order placement, and position tracking.
        _position_monitor_loop() -> Coroutine:
            Monitor open positions with batched API calls for efficiency.
        _batch_update_mt5_positions(positions: dict) -> Coroutine:
            Update all MT5 positions in a single API call.
        _update_trailing_stop_if_needed(trade_id: str, position: dict, 
                                       current_price: float) -> Coroutine:
            Update trailing stop loss for a position if conditions are met.
        _handle_external_close(trade_id: str, position: dict) -> Coroutine:
            Handle positions closed outside the system and update database.
        _update_position(trade_id: str, position: dict) -> Coroutine:
            Update a single position's current status and market data.
        _learning_loop() -> Coroutine:
            Run strategy learning engine periodically on configured schedule.
        _daily_summary_loop() -> Coroutine:
            Generate daily performance summary and reset daily statistics.
        _get_total_balance() -> Coroutine[float]:
            Get total account balance across all connected platforms.
        _get_current_exposure() -> dict:
            Get current exposure summary including open position count and symbol breakdown.
        _get_recent_trades(n: int = 10) -> list[dict]:
            Get N most recent closed trades from database.
        shutdown() -> Coroutine:
            Gracefully shutdown system, close positions, and disconnect clients.
    
    Main trading system orchestrator.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize trading system.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("=" * 60)
        logger.info("Initializing Trading System")
        logger.info("=" * 60)
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Override with environment variables
        self._apply_env_overrides()
        
        # Initialize components
        self._init_database()
        self._init_market_clients()
        self._init_strategy_components()
        self._init_execution_clients()
        self._init_learning_engine()
        self._init_notifier_and_signal_bot()
        
        # State
        self.running = False
        self.kill_switch = False  # Emergency shutdown flag
        self.open_positions = {}
        self.current_equity = 0.0
        self.daily_stats = {
            'trades_today': 0,
            'daily_drawdown_percent': 0,
            'starting_balance': 0
        }
        self.consecutive_losses = 0
        self.last_loss_time     = 0.0
        self.halt_new_trades = False
        
        logger.info("Trading System initialized successfully")
        
    def _apply_env_overrides(self):
        """
        Apply environment variable overrides to the configuration.
        Reads environment variables and updates the config dictionary with their values:
        - ENVIRONMENT: Sets the general mode (defaults to 'demo' if not set)
        - DATABASE_PATH: Sets the database path if provided
        Environment variables take precedence over existing config values.
        """
        """Apply environment variable overrides to config."""
        env_mode = os.getenv('ENVIRONMENT', 'demo')
        if env_mode:
            self.config['general']['mode'] = env_mode
            
        # Database
        db_path = os.getenv('DATABASE_PATH')
        if db_path:
            self.config.setdefault('database', {})['path'] = db_path
            
    def _init_database(self):
        """Initialize database and logging."""
        db_path = self.config.get('database', {}).get('path', 'data/trading.db')
        self.db = DatabaseManager(db_path)
        self.db.connect()  # Ensure connection is established
                
        self.audit_logger = AuditLogger(self.db)
        logger.info(f"Database initialized: {db_path}")
        
    def _init_market_clients(self):
        """Initialize market data clients."""
        market_config = {
            'mt5': {
                'host': os.getenv('MT5_BRIDGE_HOST', 'localhost'),
                'port': int(os.getenv('MT5_BRIDGE_PORT', 9090)),
                'account': os.getenv('MT5_ACCOUNT'),
                'password': os.getenv('MT5_PASSWORD'),
                'server': os.getenv('MT5_SERVER')
            }
        }
        
        self.market_client = MultiMarketClient(market_config)
        logger.info("Market clients initialized")
        
    def _init_strategy_components(self):
        """Initialize strategy and risk management."""
        # ── Note: StrategyEngine creates its own TechnicalIndicators instance ──
        # Don't create a separate one here to avoid duplication and sync issues.
        # Access indicators via: self.strategy_engine.indicators if needed.
        self.strategy_engine = StrategyEngine(self.config)
        self.money_manager = MoneyManager(self.config)
        self.stop_manager = StopManager(self.config)
        
        logger.info("Strategy components initialized")
        
    def _init_execution_clients(self):
        """Initialize execution clients."""
        demo_mode = self.config['general']['mode'] != 'live'
        
        # MT5
        mt5_config = {
            'host': os.getenv('MT5_BRIDGE_HOST', 'localhost'),
            'port': int(os.getenv('MT5_BRIDGE_PORT', 9090)),
            'account': os.getenv('MT5_ACCOUNT'),
            'password': os.getenv('MT5_PASSWORD'),
            'server': os.getenv('MT5_SERVER'),
            'magic_number': 123456
        }
        self.mt5_client = MT5Bridge(mt5_config, demo_mode=demo_mode)

        # ── MarketHoursChecker needs MT5 client, but it's not yet connected ──
        # The actual connection happens in start() before prefetch_all() is called.
        self.market_hours = MarketHoursChecker(self.mt5_client)
        
        logger.info(f"Execution clients initialized ({'DEMO' if demo_mode else 'LIVE'} mode)")
        
    def _init_learning_engine(self):
        """Initialize learning engine."""
        if self.config.get('learning', {}).get('enabled', True):
            self.learner = StrategyLearner(self.db, self.config)
            logger.info("Learning engine initialized")
        else:
            self.learner = None
            logger.info("Learning engine disabled")
    
    def _init_notifier_and_signal_bot(self):
        """Initialize Telegram notifier and channel signal bot."""
        from notification.telegram_notifier import TelegramNotifier
        try:
            self.notifier = TelegramNotifier()
            logger.info("Telegram notifier initialized")
        except Exception as e:
            logger.warning(f"Telegram notifier not available (check .env): {e}")
            self.notifier = None

        db_path = self.config.get('database', {}).get('path', 'data/trading.db')
        try:
            if self.config.get('telegram_signals', {}).get('enabled', True):
                self.signal_bot = ChannelSignalBot(
                    mt5_bridge = self.mt5_client,   # MT5Bridge IS the bridge directly
                    notifier   = self.notifier,
                    db_path    = db_path,
                )
                logger.info("Channel signal bot initialized")
            else:
                logger.info("Channel signal bot disabled in config")
        except Exception as e:
            logger.warning(f"Signal bot init failed: {e}")
            self.signal_bot = None

    async def _balance_monitor_loop(self):
        """Monitor account balance and update equity."""
        logger.info("Balance monitor loop started")
        
        while self.running:
            try:
                # Update equity every 30 seconds
                await asyncio.sleep(30)
                
                current_equity = await self._get_total_balance()
                
                # Calculate drawdown
                if self.daily_stats['starting_balance'] > 0:
                    drawdown = (
                        (self.daily_stats['starting_balance'] - current_equity) / 
                        self.daily_stats['starting_balance'] * 100
                    )
                    self.daily_stats['daily_drawdown_percent'] = drawdown
                    
                    # Check drawdown limits
                    max_dd = self.config.get('risk_management', {}).get(
                        'global_limits', {}
                    ).get('daily_max_drawdown_percent', 5.0)
                    
                    if drawdown >= max_dd:
                        if not self.halt_new_trades:
                            logger.critical(
                                f"[DRAWDOWN] Daily drawdown {drawdown:.2f}% >= limit {max_dd}% "
                                f"— halting all new trade entries immediately."
                            )
                            self.halt_new_trades = True
                            self.audit_logger.log_risk_event({
                                'event_type'      : 'daily_drawdown_limit_hit',
                                'drawdown_percent': drawdown,
                                'max_allowed'     : max_dd,
                                'message'         : f'New entries halted at {drawdown:.2f}% drawdown',
                            })
                            if hasattr(self, 'notifier'):
                                asyncio.ensure_future(self.notifier.send(
                                    f"⚠️ Daily drawdown limit hit: {drawdown:.2f}% — trading paused."
                                )) # assumed dd_pct mean drawdown percent

                        # Emergency shutdown (close positions) is separate from halting entries
                        emergency_config = self.config.get('risk_management', {}).get(
                            'global_limits', {}
                        ).get('emergency_shutdown', {})
                        if emergency_config.get('enabled', True):
                            if emergency_config.get('close_positions_on_shutdown', False):
                                await self._emergency_close_all()
                    
            except Exception as e:
                logger.error(f"Error in balance monitor: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _emergency_close_all(self):
        """
        Close all open positions.  Only runs when close_positions_on_shutdown=true.
        """
        close_cfg = (
            self.config
                .get('risk_management', {})
                .get('global_limits', {})
                .get('emergency_shutdown', {})
        )
        if not close_cfg.get('close_positions_on_shutdown', False):
            logger.info(
                "[EMERGENCY_CLOSE] Skipped — close_positions_on_shutdown=false. "
                "Positions will be reconciled on next startup."
            )
            return

        logger.warning(f"EMERGENCY: Closing all {len(self.open_positions)} positions!")

        for trade_id, position in list(self.open_positions.items()):
            try:
                platform = position.get('platform')
                ticket   = position.get('ticket')

                if platform == 'mt5' and ticket:
                    result = await self.mt5_client.close_position(ticket)
                    logger.info(f"Emergency closed MT5 position {ticket}: {result}")

                self.audit_logger.log_trade_exit(trade_id, {
                    'exit_price' : 0,
                    'reason'     : 'emergency_shutdown',
                    'pnl'        : 0,
                    'pnl_percent': 0,
                    'realized_rr': 0,
                })
            except Exception as e:
                logger.error(f"Error closing position {trade_id}: {e}")

    async def start(self):
        """Start the trading system."""
        logger.info("=" * 60)
        logger.info("Starting Trading System")
        logger.info("=" * 60)
        
        self.running = True
        
        # Connect execution clients
        await self.mt5_client.connect()
        
        # Get starting balance
        self.daily_stats['starting_balance'] = await self._get_total_balance()

        await self._load_open_positions_from_db()

        # cLEANING UP EMPTY OR NULL TRADES ON STARTUP
        asyncio.ensure_future(self._startup_pnl_backfill())

        enabled_symbols = [s for s, c in self.config.get('symbols', {}).items() if c.get('enabled')]
        await self.market_hours.prefetch_all(enabled_symbols)
        # this fires one get_symbol_sessions per symbol at startup, logs the full schedule

        
        # Start main trading loop
        # ── Use return_exceptions=True to prevent cascading failures ──────────
        # If one loop crashes, others continue running so trading/monitoring isn't
        # completely interrupted by a transient error in learning or summary loops.
        try:
            _coros = [
                self._trading_loop(),
                self._position_monitor_loop(),
                self._balance_monitor_loop(),
                self._learning_loop(),
                self._daily_summary_loop(),
            ]
            if getattr(self, 'signal_bot', None):
                _coros.append(self.signal_bot.start())
                _coros.append(_limit_order_alert_watcher(self.signal_bot.executor))

            results = await asyncio.gather(*_coros, return_exceptions=True)
            # Log any exceptions that occurred in individual loops
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    loop_names = [
                        'trading', 'position_monitor', 'balance_monitor',
                        'learning', 'daily_summary', 'signal_bot', 'limit_alert_watcher'
                    ]
                    logger.error(f"Loop '{loop_names[i]}' failed: {result}", exc_info=result)
        except asyncio.CancelledError:
            logger.info("Trading loops cancelled - shutting down")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def _load_open_positions_from_db(self):
        """
        Load open positions from database on startup and verify each one
        still exists on the broker.

        - If the ticket is missing from DB (wasn't saved): log a warning and
          mark closed — we cannot verify without a ticket.
        - If the position is not found on MT5: call _handle_external_close so
          deal history is fetched and P&L/exit data is populated properly.
        - If found: restore the full position dict into open_positions.
        """
        try:
            open_trades = self.db.get_open_trades()
            logger.info(f"Loading {len(open_trades)} open positions from database")

            for trade in open_trades:
                trade_id = trade['trade_id']
                platform = trade.get('platform', 'mt5')
                ticket   = trade.get('ticket')

                # ── Guard: ticket must be present to verify ───────────────────
                if not ticket:
                    logger.warning(
                        f"[STARTUP] trade_id={trade_id} has no ticket saved — "
                        f"cannot verify on broker, marking closed."
                    )
                    self.db.update_trade(trade_id, {
                        'status'     : 'closed',
                        'exit_reason': 'missing_ticket_on_restart',
                        'exit_time'  : datetime.now(timezone.utc).replace(tzinfo=None),
                    })
                    continue

                # ── MT5 verification ──────────────────────────────────────────
                if platform == 'mt5':
                    pos_info = await self.mt5_client.get_position_info(int(ticket))

                    if not pos_info:
                        # ── Before assuming the position was closed, check if the market
                        #    is currently closed. Some brokers don't serve position data
                        #    during the weekend — the ticket will appear "missing" even
                        #    though the trade is still open and will resume on Monday.
                        symbol = trade.get('symbol', '')
                        market_is_open = False

                        try:
                            # Primary check: broker session schedule (fetched at startup).
                            # During startup position verification we trust the schedule
                            # and skip the is_open_by_price secondary check — that check
                            # sends a get_historical command which times out when the EA
                            # is in a reduced weekend state. The trading loop handles the
                            # price-freshness check at analysis time.
                            market_is_open = self.market_hours.is_open(symbol)
                            
                        except Exception as e:
                            logger.warning(
                                f"[STARTUP] Market-open check failed for {symbol}: {e} — "
                                f"assuming market closed to protect position."
                            )
                            market_is_open = False  # fail-safe: don't destroy the trade on uncertainty

                        if not market_is_open:
                            logger.info(
                                f"[STARTUP] ticket={ticket} ({symbol}) not found on MT5 "
                                f"but market is currently closed — restoring as active. "
                                f"Position monitor will reconcile when market reopens."
                            )
                            # Restore into tracking exactly like a confirmed live position
                            self.open_positions[trade_id] = {
                                'trade_id'               : trade_id,
                                'ticket'                 : ticket,
                                'symbol'                 : trade.get('symbol'),
                                'platform'               : platform,
                                'direction'              : trade.get('direction'),
                                'entry_price'            : trade.get('entry_price', 0.0),
                                'current_price'          : trade.get('entry_price', 0.0),
                                'stop_loss'              : trade.get('stop_loss', 0.0),
                                'original_stop_loss'     : trade.get('original_stop_loss') or trade.get('stop_loss', 0.0),
                                'take_profit_1'          : trade.get('take_profit_1'),
                                'take_profit_2'          : trade.get('take_profit_2'),
                                'position_size'          : trade.get('position_size', 0.0),
                                'volume'                 : trade.get('position_size', 0.0),
                                'entry_time'             : trade.get('entry_time'),
                                'analysis_id'            : trade.get('analysis_id'),
                                'trailing_active'        : False,
                                'tp1_hit'                : False,
                                'tp2_hit'                : False,
                                'last_sl_update'         : 0.0,
                                'high_since_entry'       : trade.get('entry_price', 0.0),
                                'low_since_entry'        : trade.get('entry_price', 0.0),
                                'max_favorable_excursion': 0.0,
                                'max_adverse_excursion'  : 0.0,
                                'source'                 : 'market_closed_restore',
                            }
                            continue

                        # Market is open and position not found → truly closed externally
                        logger.warning(
                            f"[STARTUP] trade_id={trade_id} ticket={ticket} "
                            f"not found on MT5 (market is open) — was closed while system was down."
                        )
                        ghost_position = {
                            'ticket'             : ticket,
                            'symbol'             : trade.get('symbol'),
                            'direction'          : trade.get('direction'),
                            'entry_price'        : trade.get('entry_price', 0.0),
                            'stop_loss'          : trade.get('stop_loss', 0.0),
                            'original_stop_loss' : trade.get('original_stop_loss') or trade.get('stop_loss', 0.0),
                            'take_profit_1'      : trade.get('take_profit_1', 0.0),
                            'volume'             : trade.get('position_size', 0.0),
                            'position_size'      : trade.get('position_size', 0.0),
                            'entry_time'         : trade.get('entry_time'),
                            'platform'           : platform,
                            'max_favorable_excursion': None,
                            'max_adverse_excursion'  : None,
                        }
                        await self._handle_external_close(trade_id, ghost_position)
                        continue
                # ── Position confirmed alive — restore into tracking ───────────
                self.open_positions[trade_id] = {
                    'trade_id'           : trade_id,
                    'ticket'             : ticket,
                    'symbol'             : trade.get('symbol'),
                    'platform'           : platform,
                    'direction'          : trade.get('direction'),
                    'entry_price'        : trade.get('entry_price', 0.0),
                    'current_price'      : trade.get('entry_price', 0.0),
                    'stop_loss'          : trade.get('stop_loss', 0.0),
                    'original_stop_loss' : trade.get('stop_loss', 0.0),
                    'take_profit_1'      : trade.get('take_profit_1'),
                    'take_profit_2'      : trade.get('take_profit_2'),
                    'position_size'      : trade.get('position_size', 0.0),
                    'volume'             : trade.get('position_size', 0.0),
                    'entry_time'         : trade.get('entry_time'),
                    'analysis_id'        : trade.get('analysis_id'),
                    'trailing_active'    : False,
                    'tp1_hit'            : False,
                    'tp2_hit'            : False,
                    'last_sl_update'     : 0.0,
                    'high_since_entry'   : trade.get('entry_price', 0.0),
                    'low_since_entry'    : trade.get('entry_price', 0.0),
                    'max_favorable_excursion': 0.0,
                    'max_adverse_excursion'  : 0.0,
                }

            logger.info(f"Loaded {len(self.open_positions)} active positions")

        except Exception as e:
            logger.error(f"Error loading open positions: {e}", exc_info=True)

            
    async def _trading_loop(self):
        """
        Spawn one bar-close aligned analysis task per enabled symbol.
        Each symbol wakes independently at its own primary_timeframe boundary.
        The outer while-loop monitors the tasks and restarts any that die.
        """
        import asyncio as _asyncio
 
        symbols_config = self.config.get('symbols', {})
        enabled = [
            (sym, cfg)
            for sym, cfg in symbols_config.items()
            if cfg.get('enabled', False)
        ]
 
        if not enabled:
            logger.warning("[TRADING] No enabled symbols — trading loop idle.")
            while self.running:
                await asyncio.sleep(60)
            return
 
        sym_names = [s for s, _ in enabled]
        logger.info(
            f"[TRADING] Bar-close sync started for {len(enabled)} symbol(s): "
            f"{', '.join(sym_names)}"
        )
 
        tasks = {
            sym: asyncio.create_task(
                self._symbol_bar_close_loop(sym, cfg),
                name=f"bar_close_{sym.replace('/', '')}"
            )
            for sym, cfg in enabled
        }
 
        try:
            while self.running:
                await asyncio.sleep(15)
                # Detect and restart dead tasks
                for sym, cfg in enabled:
                    task = tasks[sym]
                    if task.done() and not task.cancelled():
                        exc = task.exception()
                        if exc:
                            logger.error(
                                f"[TRADING] Task for {sym} crashed: {exc} — restarting",
                                exc_info=exc,
                            )
                        else:
                            logger.warning(f"[TRADING] Task for {sym} ended — restarting")
                        tasks[sym] = asyncio.create_task(
                            self._symbol_bar_close_loop(sym, cfg),
                            name=f"bar_close_{sym.replace('/', '')}",
                        )
        finally:
            for t in tasks.values():
                t.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            logger.info("[TRADING] All bar-close tasks stopped.")
    
    async def _symbol_bar_close_loop(self, symbol: str, symbol_config: dict):
        """
        Per-symbol loop: sleep until the next primary_timeframe bar closes,
        then run the full analysis and entry cycle for this symbol.
        3-second post-close buffer lets the broker finalize the bar.
        """
        primary_tf  = symbol_config.get('primary_timeframe', '15m')
        tf_secs     = _tf_to_seconds(primary_tf)
        buffer_secs = 3    # broker finalization buffer
 
        logger.info(
            f"[{symbol}] Bar-close loop started "
            f"(primary_tf={primary_tf}, tf_secs={tf_secs})"
        )
 
        while self.running:
            try:
                # Calculate sleep duration to next bar close
                next_close   = _next_bar_close_utc(tf_secs)
                sleep_target = next_close + timedelta(seconds=buffer_secs)
                wait_secs    = (sleep_target - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
 
                if wait_secs > 0:
                    await asyncio.sleep(wait_secs)
 
                # ── Guard: global limits before doing any work ────────────────
                if self.halt_new_trades:
                    logger.debug(
                        f"[{symbol}] Daily drawdown halt active — skipping cycle."
                    )
                    continue
 
                max_concurrent = (
                    self.config.get('risk_management', {})
                               .get('global_limits', {})
                               .get('max_concurrent_trades', 3)
                )
                if len(self.open_positions) >= max_concurrent:
                    logger.debug(
                        f"[{symbol}] At max positions "
                        f"({len(self.open_positions)}/{max_concurrent}) — skipping."
                    )
                    continue
 
                if not self.market_hours.is_open(symbol):
                    logger.debug(
                        f"[{symbol}] Market closed — "
                        f"next open: {self.market_hours.next_open_str(symbol)}"
                    )
                    continue
 
                # Skip if already holding a position in this symbol
                norm_sym = symbol.replace('/', '')
                if any(
                    p.get('symbol', '').replace('/', '') == norm_sym
                    for p in self.open_positions.values()
                ):
                    logger.debug(f"[{symbol}] Already have open position — skip.")
                    continue
 
                # ── Run analysis ──────────────────────────────────────────────
                await self._analyze_symbol(symbol, symbol_config)
 
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[{symbol}] Error in bar-close loop: {e}", exc_info=True
                )
                await asyncio.sleep(60)
 
        logger.info(f"[{symbol}] Bar-close loop stopped.")

    async def _analyze_symbol(self, symbol: str, symbol_config: dict):
        """
        Fetch multi-timeframe data, run analysis, and fire entry signal
        if conditions are met. Called after confirmed bar close.
        """
        try:
            platform   = symbol_config.get('platform', 'mt5')
            timeframes = symbol_config.get('timeframes', [])
 
            multi_tf_data = await self.market_client.fetch_multiple_timeframes(
                symbol, platform, timeframes
            )
            if not multi_tf_data:
                logger.warning(f"[{symbol}] No data fetched — skipping cycle.")
                return
 
            analysis = self.strategy_engine.analyze_market(symbol, multi_tf_data)
 
            try:
                analysis_id = self.audit_logger.log_analysis(analysis)
                analysis['analysis_id'] = analysis_id
            except Exception as e:
                logger.error(f"Error logging analysis: {e}", exc_info=True)
                analysis['analysis_id'] = None
 
            if analysis['entry_signal']:
                # Final position count guard before processing
                max_concurrent = (
                    self.config.get('risk_management', {})
                               .get('global_limits', {})
                               .get('max_concurrent_trades', 3)
                )
                if len(self.open_positions) >= max_concurrent:
                    logger.warning(
                        f"[{symbol}] Signal detected but max positions reached "
                        f"({len(self.open_positions)}/{max_concurrent}) — skipped."
                    )
                    return
 
                await self._process_entry_signal(
                    symbol, symbol_config, analysis, multi_tf_data
                )
 
        except Exception as e:
            logger.error(f"[{symbol}] _analyze_symbol error: {e}", exc_info=True)
                
    async def _process_entry_signal(
        self,
        symbol:        str,
        symbol_config: dict,
        analysis:      dict,
        multi_tf_data: dict,
    ):
        """
        Place order after a confirmed entry signal.
 
        Market orders: registered in open_positions immediately.
        Limit  orders: saved to pending_limit_orders; NOT in open_positions
                       until the order fills (detected by position monitor).
        """
        try:
            analysis_id = analysis.get('analysis_id', 'unknown')
            platform    = symbol_config.get('platform', 'mt5')
 
            # ── Absolute final guard ──────────────────────────────────────────
            max_concurrent = (
                self.config.get('risk_management', {})
                           .get('global_limits', {})
                           .get('max_concurrent_trades', 3)
            )
            if len(self.open_positions) >= max_concurrent:
                logger.error(
                    f"[{symbol}] Attempted entry at max positions — aborting."
                )
                return
 
            # ── Calculate levels ──────────────────────────────────────────────
            levels = self.strategy_engine.calculate_entry_levels(analysis, multi_tf_data)
            if not levels:
                logger.warning(f"[{symbol}] Could not calculate entry levels — skipped.")
                return
 
            current_price  = levels.get('entry_price', 0.0)
            order_price    = levels.get('order_price', current_price)
            order_type     = levels.get('order_type', 'market')
            limit_price    = levels.get('limit_price')
            stop_loss      = levels.get('stop_loss', 0.0)
            take_profit_1  = levels.get('take_profit_1', 0.0)
            take_profit_2  = levels.get('take_profit_2', 0.0)
            atr            = levels.get('atr', 0.0)
            entry_type     = analysis.get('entry_type', 'unknown')
 
            # ── Risk checks ───────────────────────────────────────────────────
            if not self._check_cooldown_after_losses():
                return
 
            balance = await self._get_total_balance()
            if balance <= 0:
                logger.error(f"[{symbol}] Cannot size position — balance=0.")
                return
 
            sizing = self.money_manager.calculate_position_size(
                account_equity = balance,
                entry_price    = order_price,
                stop_loss      = stop_loss,
                symbol         = symbol,
                direction      = analysis['direction'],
                platform       = platform,
            )
            if not sizing.get('approved') or sizing.get('position_size', 0) <= 0:
                logger.info(f"[{symbol}] Position sizing rejected: {sizing}")
                return
 
            # ── Per-symbol drawdown guard ────────────────────────────────────
            symbol_dd_limit = (
                self.config.get('risk_management', {})
                           .get('global_limits', {})
                           .get('max_drawdown_per_symbol_percent', 3.0)
            )
            symbol_pnl   = self.daily_stats.get('symbol_pnl', {}).get(symbol, 0.0)
            symbol_dd_pct = (-symbol_pnl / balance * 100) if symbol_pnl < 0 else 0.0
            if symbol_dd_pct >= symbol_dd_limit:
                logger.warning(
                    f"[{symbol}] Per-symbol DD {symbol_dd_pct:.2f}% >= "
                    f"limit {symbol_dd_limit}% — rejecting."
                )
                return
 
            # ── Place order ───────────────────────────────────────────────────
            if platform == 'mt5':
                result = await self.mt5_client.place_order(
                    symbol      = symbol.replace('/', ''),
                    direction   = analysis['direction'],
                    volume      = sizing['position_size'],
                    order_type  = order_type,
                    price       = limit_price if order_type == 'limit' else None,
                    stop_loss   = stop_loss,
                    take_profit = take_profit_2,
                    comment     = f"Analysis_{analysis_id[:8]}"
                                  if analysis_id != 'unknown' else "Python",
                )
            else:
                logger.warning(f"[{symbol}] Unsupported platform: {platform}")
                return
 
            if not result.get('success'):
                logger.error(f"[{symbol}] Order failed: {result.get('error')}")
                return
 
            ticket         = result.get('ticket') or result.get('order_id')
            executed_price = result.get('filled_price') or result.get('price') or order_price
            placed_at      = datetime.now(timezone.utc).replace(tzinfo=None)
 
            # ── Log trade ─────────────────────────────────────────────────────
            trade_status = 'pending_limit' if order_type == 'limit' else 'open'
            trade_id = self.audit_logger.log_trade_entry({
                'analysis_id'  : analysis_id,
                'symbol'       : symbol,
                'platform'     : platform,
                'direction'    : analysis['direction'],
                'entry_price'  : executed_price,
                'stop_loss'    : stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'position_size': sizing['position_size'],
                'status'       : trade_status,
            })
            self.db.update_trade(trade_id, {'ticket': ticket})
 
            # ── Route: limit vs market ─────────────────────────────────────────
            if order_type == 'limit':
                # Limit order is PENDING — save to tracker, do NOT open position
                primary_tf = symbol_config.get('primary_timeframe', '15m')
                expiry_time, min_floor = _limit_expiry_times(
                    placed_at, primary_tf, entry_type,
                    self.config.get('strategy', {})
                )
 
                invalidation_atr_mult = float(
                    self.config.get('strategy', {})
                               .get('limit_order_expiry', {})
                               .get('price_invalidation_atr_multiplier', 1.0)
                )
                if analysis['direction'] == 'long':
                    invalidation_price = float(limit_price) - invalidation_atr_mult * atr
                else:
                    invalidation_price = float(limit_price) + invalidation_atr_mult * atr
 
                self.db.save_pending_limit_order({
                    'trade_id':           trade_id,
                    'ticket':             ticket,
                    'symbol':             symbol,
                    'direction':          analysis['direction'],
                    'entry_type':         entry_type,
                    'limit_price':        limit_price or executed_price,
                    'placed_at':          placed_at.isoformat(),
                    'expiry_time':        expiry_time.isoformat(),
                    'min_cancel_floor':   min_floor.isoformat(),
                    'invalidation_price': invalidation_price,
                    'atr_at_placement':   atr,
                })
 
                logger.info(
                    f"[{symbol}] LIMIT ORDER placed — ticket={ticket} "
                    f"type={entry_type} price={limit_price:.5f} "
                    f"expiry={expiry_time.strftime('%H:%M:%S')} UTC"
                )
                if hasattr(self, 'notifier'):
                    asyncio.ensure_future(self.notifier.send(
                        f"⏳ <b>Limit Order Placed</b> — {symbol}\n"
                        f"<b>{analysis['direction'].upper()}</b> "
                        f"@ <code>{limit_price:.5f}</code> [{entry_type}]\n"
                        f"SL: <code>{stop_loss:.5f}</code>  "
                        f"TP1: <code>{take_profit_1:.5f}</code>\n"
                        f"Expires: <code>{expiry_time.strftime('%Y-%m-%d %H:%M UTC')}</code>\n"
                        f"Ticket: <code>{ticket}</code>"
                    ))
 
            else:
                # Market order — register in open_positions immediately
                self._register_new_position(
                    trade_id    = trade_id,
                    ticket      = ticket,
                    symbol      = symbol,
                    direction   = analysis['direction'],
                    entry_price = executed_price,
                    volume      = sizing['position_size'],
                    sl          = stop_loss,
                    tp1         = take_profit_1,
                    tp2         = take_profit_2,
                    tp1_fraction = 0.5,
                    platform    = platform,
                    analysis_id = analysis_id,
                )
                self.daily_stats['trades_today'] += 1
                logger.info(
                    f"[{symbol}] MARKET ORDER filled — ticket={ticket} "
                    f"price={executed_price:.5f} "
                    f"dir={analysis['direction']} type={entry_type}"
                )
 
        except Exception as e:
            logger.error(
                f"[{symbol}] Error processing entry signal: {e}", exc_info=True
            )

    async def _check_pending_limit_orders(self):
        """
        Check all pending limit orders for fill, expiry, or price invalidation.
        Called every position-monitor cycle (every ~15s).
 
        Logic per order:
          - Not past min_cancel_floor → skip (never cancel within 60s of placement)
          - Ticket not in EA's order list → filled → promote to open position
          - Past expiry_time → cancel and mark expired
          - Price past invalidation_price → cancel and mark invalidated
        """
        try:
            pending = self.db.get_pending_limit_orders()
            if not pending:
                return
 
            # Single EA call for all pending orders
            ea_orders = await self.mt5_client.get_all_orders() or []
            ea_order_tickets = {int(o.get('ticket', 0)) for o in ea_orders}
 
            # Also get all positions to detect fills
            ea_positions = await self.mt5_client.get_all_positions() or []
            ea_pos_by_comment = {}
            for p in ea_positions:
                comment = str(p.get('comment', ''))
                ea_pos_by_comment[comment] = p
 
            now = datetime.now(timezone.utc).replace(tzinfo=None)
 
            for order in pending:
                ticket              = int(order['ticket'])
                trade_id            = order['trade_id']
                symbol              = order['symbol']
                direction           = order['direction']
                entry_type          = order['entry_type']
                limit_price         = float(order['limit_price'])
                atr                 = float(order.get('atr_at_placement') or 0.0)
                invalidation_price  = order.get('invalidation_price')
                expiry_time         = datetime.fromisoformat(order['expiry_time'])
                min_floor           = datetime.fromisoformat(order['min_cancel_floor'])
 
                # Never cancel before the minimum floor
                if now < min_floor:
                    continue
 
                # ── Filled: ticket no longer in pending orders ─────────────────
                if ticket not in ea_order_tickets:
                    # Find the resulting position by matching our comment
                    analysis_id_short = (trade_id.split('_')[2][:8]
                                         if '_' in trade_id else trade_id[:8])
                    comment_key = f"Analysis_{analysis_id_short}"
                    filled_pos  = ea_pos_by_comment.get(comment_key)
 
                    if filled_pos:
                        pos_ticket     = int(filled_pos.get('ticket', 0))
                        executed_price = float(filled_pos.get('price', limit_price))
                        volume         = float(filled_pos.get('volume', 0.0))
 
                        # Update trade record with real position ticket
                        self.db.update_trade(trade_id, {'ticket': pos_ticket, 'status': 'open'})
                        self.db.update_pending_limit_order_status(ticket, 'filled')
 
                        # Get original trade data
                        trade_rows = self.db.get_open_trades()
                        trade_data = next(
                            (t for t in trade_rows if t['trade_id'] == trade_id), {}
                        )
 
                        self._register_new_position(
                            trade_id     = trade_id,
                            ticket       = pos_ticket,
                            symbol       = symbol,
                            direction    = direction,
                            entry_price  = executed_price,
                            volume       = volume,
                            sl           = float(trade_data.get('stop_loss', 0.0)),
                            tp1          = float(trade_data.get('take_profit_1') or 0.0),
                            tp2          = float(trade_data.get('take_profit_2') or 0.0),
                            tp1_fraction = 0.5,
                            platform     = 'mt5',
                            analysis_id  = trade_data.get('analysis_id', ''),
                        )
                        self.daily_stats['trades_today'] += 1
                        logger.info(
                            f"[LIMIT] Filled — symbol={symbol} "
                            f"order_ticket={ticket} pos_ticket={pos_ticket} "
                            f"price={executed_price:.5f}"
                        )
                        if hasattr(self, 'notifier'):
                            asyncio.ensure_future(self.notifier.send(
                                f"✅ <b>Limit Order Filled</b> — {symbol}\n"
                                f"<b>{direction.upper()}</b> "
                                f"@ <code>{executed_price:.5f}</code>\n"
                                f"Ticket: <code>{pos_ticket}</code>"
                            ))
                    else:
                        # Order gone but no matching position — cancelled externally
                        logger.info(
                            f"[LIMIT] Order {ticket} for {symbol} gone with no "
                            f"matching position — likely manually cancelled."
                        )
                        self.db.update_pending_limit_order_status(
                            ticket, 'cancelled', 'external_cancel'
                        )
                    continue
 
                # ── Expired ───────────────────────────────────────────────────
                if now >= expiry_time:
                    await self._cancel_limit_order(
                        ticket, trade_id, symbol, 'expired',
                        f"Expired at {expiry_time.strftime('%H:%M UTC')} "
                        f"({entry_type})"
                    )
                    continue
 
                # ── Price invalidation ─────────────────────────────────────────
                if invalidation_price and atr > 0:
                    # Get a quick current price estimate from open positions
                    # or from a recent bar (avoid extra bridge call if possible)
                    current_price = self._estimate_current_price(symbol, ea_positions)
                    if current_price:
                        if direction == 'long' and current_price < float(invalidation_price):
                            await self._cancel_limit_order(
                                ticket, trade_id, symbol, 'invalidated',
                                f"Price {current_price:.5f} broke below "
                                f"invalidation {invalidation_price:.5f}"
                            )
                        elif direction == 'short' and current_price > float(invalidation_price):
                            await self._cancel_limit_order(
                                ticket, trade_id, symbol, 'invalidated',
                                f"Price {current_price:.5f} broke above "
                                f"invalidation {invalidation_price:.5f}"
                            )
 
        except Exception as e:
            logger.error(f"[LIMIT] _check_pending_limit_orders error: {e}", exc_info=True)

    async def _cancel_limit_order(
        self,
        ticket:   int,
        trade_id: str,
        symbol:   str,
        reason:   str,
        detail:   str = '',
    ) -> None:
        """Cancel a pending limit order on the EA and update the DB."""
        try:
            success = await self.mt5_client.cancel_order(ticket)
            if success:
                self.db.update_pending_limit_order_status(ticket, reason, detail)
                self.db.update_trade(trade_id, {'status': 'cancelled'})
                logger.info(
                    f"[LIMIT] Cancelled ticket={ticket} symbol={symbol} "
                    f"reason={reason} {detail}"
                )
                if hasattr(self, 'notifier'):
                    asyncio.ensure_future(self.notifier.send(
                        f"🗑 <b>Limit Order Cancelled</b> — {symbol}\n"
                        f"Reason: <code>{reason}</code>\n"
                        f"{detail}\nTicket: <code>{ticket}</code>"
                    ))
            else:
                logger.warning(
                    f"[LIMIT] EA cancel failed for ticket={ticket} — "
                    f"may already be filled or gone."
                )
                # Mark as unknown state so it doesn't keep retrying
                self.db.update_pending_limit_order_status(
                    ticket, 'cancelled', 'cancel_failed_on_ea'
                )
        except Exception as e:
            logger.error(f"[LIMIT] _cancel_limit_order error: {e}", exc_info=True)

    def _estimate_current_price(
        self, symbol: str, ea_positions: list
    ):
        """
        Try to infer current price from live position data.
        Returns None if no data available (caller skips invalidation check).
        """
        norm = symbol.replace('/', '').upper()
        for p in ea_positions:
            if str(p.get('symbol', '')).upper() == norm:
                cur = p.get('current_price')
                if cur:
                    return float(cur)
        return None

    async def _position_monitor_loop(self):
        """Monitor open positions - BATCHED VERSION."""
        logger.info("Position monitor loop started (batched)")
        
        # ── Enforce minimum monitor interval to prevent bridge saturation ──────
        monitor_interval = max(
            5,  # Minimum 5 seconds
            self.config.get('monitor', {}).get('interval_seconds', 10)
        )
        
        while self.running:
            try:
                if len(self.open_positions) == 0:
                    # No positions to monitor, wait longer
                    await asyncio.sleep(30)
                    continue
                
                # ── Market hours guard: don't poll the EA when market is closed ──
                # When closed, SL/TP cannot be hit, price cannot move, and the EA
                # often doesn't respond to commands reliably in a weekend state.
                # Sleep until the next open rather than hammering the bridge.
                tracked_symbols = {
                    pos.get('symbol') for pos in self.open_positions.values()
                    if pos.get('symbol')
                        }
                any_symbol_open = any(
                    self.market_hours.is_open(sym) for sym in tracked_symbols
                        ) if tracked_symbols else True  # no positions = don't gate

                if not any_symbol_open:
                    # Find soonest market open across all tracked symbols
                    secs_until_open = min(
                        (self.market_hours.seconds_until_open(sym) for sym in tracked_symbols),
                        default=300
                    )
                    sleep_secs = max(60, min(secs_until_open, 300))  # check every 5min max
                    logger.info(
                        f"[MONITOR] All markets closed — sleeping {sleep_secs//60}m "
                        f"(next open in {self.market_hours.next_open_str(next(iter(tracked_symbols)))})"
                    )
                    await asyncio.sleep(sleep_secs)
                    continue

                # Group positions by platform for batch checking
                mt5_positions = {}
                
                for trade_id, position in list(self.open_positions.items()):
                    if position.get('platform', 'mt5') == 'mt5':
                        mt5_positions[trade_id] = position
                
                # Batch check MT5 positions (single API call)
                if mt5_positions:
                    await self._batch_update_mt5_positions(mt5_positions)
                    await self._check_pending_limit_orders()
                
                # Wait before next check (respecting minimum interval)
                await asyncio.sleep(monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}", exc_info=True)
                await asyncio.sleep(monitor_interval)


    async def _batch_update_mt5_positions(self, positions: dict):
        """
        Update all MT5 positions in one bridge call.
        Relies on the updated MQ5 bridge that now returns:
        ticket, symbol, volume, price (entry), current_price (live bid/ask),
        sl, tp, type (0=BUY/1=SELL), profit
        """
        try:
            all_mt5_positions = await self.mt5_client.get_all_positions()

            # GUARD: None means the bridge call failed (timeout, EA unresponsive).
            # An empty list returned from an error is indistinguishable from
            # "genuinely no open positions", which would trigger false external-close
            # events on every live trade. Abort this monitoring cycle entirely.
            if all_mt5_positions is None:
                logger.warning(
                    "[MONITOR] get_all_positions returned None (bridge error) "
                    "— skipping close detection this cycle to prevent false closes."
                )
                return

            # Build lookup by ticket
            mt5_by_ticket = {
                int(pos['ticket']): pos
                for pos in all_mt5_positions
            }

            for trade_id, position in list(positions.items()):
                ticket = position.get('ticket')
                if ticket is None:
                    continue

                if int(ticket) not in mt5_by_ticket:
                    # Position closed externally (SL/TP hit on broker side)
                    logger.warning(
                        f"[MONITOR] trade_id={trade_id} ticket={ticket} "
                        f"not found on MT5 — closed externally"
                    )
                    await self._handle_external_close(trade_id, position)
                    continue

                current = mt5_by_ticket[int(ticket)]

                # ── Use current_price (live bid/ask) — NOT price (entry price) ──
                current_price = current.get('current_price') or current.get('price', 0)

                # Refresh tracked position with live data from bridge
                position['stop_loss']     = current.get('sl', position.get('stop_loss', 0))
                position['take_profit_1'] = current.get('tp', position.get('take_profit_1', 0))
                position['profit']        = current.get('profit', 0)
                position['volume']        = current.get('volume', position.get('volume'))

                # ── Track Max Favorable / Adverse Excursion ───────────────────
                entry_price   = position.get('entry_price', 0.0)
                direction     = position.get('direction', 'long')

                if entry_price and current_price:
                    if direction == 'long':
                        favorable = current_price - entry_price
                        adverse   = entry_price - current_price
                    else:
                        favorable = entry_price - current_price
                        adverse   = current_price - entry_price

                    # MFE: best the trade has looked (highest favorable move)
                    position['max_favorable_excursion'] = max(
                        position.get('max_favorable_excursion', 0.0),
                        favorable or 0.0
                    )
                    # MAE: worst the trade has looked (highest adverse move, stored positive)
                    position['max_adverse_excursion'] = max(
                        position.get('max_adverse_excursion', 0.0),
                        adverse or 0.0
                    )
                    # Persist MFE/MAE to DB on every update cycle so deferred
                    # closes and crash-restarts have real values to work with.
                    try:
                        self.db.update_trade(trade_id, {
                            'max_favorable_excursion': position['max_favorable_excursion'],
                            'max_adverse_excursion'  : position['max_adverse_excursion'],
                        })
                    except Exception as _mfe_err:
                        logger.debug(f"[MONITOR] MFE/MAE persist failed for {trade_id}: {_mfe_err}")

                # ── TP level checks ───────────────────────────────────────────────
                position['current_price'] = current_price
                self._check_and_handle_tp_levels_sync(trade_id, position, current_price)

                # ── Trailing stop ─────────────────────────────────────────────────
                await self._update_trailing_stop_if_needed(trade_id, position, current_price)

        except Exception as e:
            logger.error(f"[MONITOR] Error batch updating MT5 positions: {e}", exc_info=True)


    async def _update_trailing_stop_if_needed(
        self,
        trade_id: str,
        position: dict,
        current_price: float,
        ):
        """
        Check whether the trailing stop should be moved and, if so, send the
        modify call to MT5.

        Uses StopManager.compute_trailing_sl() which wraps the existing
        _calculate_atr_trailing_stop / _calculate_percentage_trailing_stop.
        """
        cfg_trail   = self.config.get('risk_management', {}).get('trailing_stop', {})
        if not cfg_trail.get('enabled', True):
            return

        rr_activate         = cfg_trail.get('activation_rr', 1.0)
        min_update_interval = cfg_trail.get('min_update_interval_seconds', 30)

        ticket      = position.get('ticket')
        symbol      = position.get('symbol', '')
        direction   = position.get('direction', 'long')   # 'long' | 'short'
        entry_price = position.get('entry_price', 0.0)
        # Use original SL for RR calc — current_sl may have moved to breakeven
        original_sl = position.get('original_stop_loss') or position.get('stop_loss', 0.0)
        current_sl  = position.get('stop_loss', 0.0)
        current_tp  = position.get('take_profit_1', 0.0) or 0.0

        if not entry_price or not current_sl:
            return

        # Rate-limit: avoid hammering MT5 with modify calls
        last_update = position.get('last_sl_update', 0.0)
        if (_monotime.time() - last_update) < min_update_interval:
            return

        # Initial risk uses ORIGINAL SL so RR is always relative to entry risk
        initial_risk = abs(entry_price - original_sl)
        if initial_risk == 0:
            return

        price_move = (current_price - entry_price) if direction == 'long' \
                     else (entry_price - current_price)
        achieved_rr = price_move / initial_risk

        # ── Activation check ─────────────────────────────────────────────────
        if not position.get('trailing_active', False):
            if achieved_rr < rr_activate:
                return  # Not yet profitable enough to trail

            # Break-even move on first activation
            be_cfg     = cfg_trail.get('breakeven', {})
            buffer_pts = be_cfg.get('buffer_pips', 1) * self._pip_size(symbol)

            if direction == 'long':
                breakeven_sl = entry_price + buffer_pts
                if current_sl < breakeven_sl:
                    try:
                        await self._send_sl_modify(
                            trade_id, position, breakeven_sl, current_tp, label='breakeven'
                        )
                    except Exception as e:
                        logger.error(
                            f"[TRAIL] Breakeven SL modify failed for trade_id={trade_id} "
                            f"ticket={ticket}: {e}"
                        )
                        # Don't propagate — let position monitoring continue
            else:
                breakeven_sl = entry_price - buffer_pts
                if current_sl > breakeven_sl:
                    try:
                        await self._send_sl_modify(
                            trade_id, position, breakeven_sl, current_tp, label='breakeven'
                        )
                    except Exception as e:
                        logger.error(
                            f"[TRAIL] Breakeven SL modify failed for trade_id={trade_id} "
                            f"ticket={ticket}: {e}"
                        )
                        # Don't propagate — let position monitoring continue

            position['trailing_active'] = True
            logger.info(
                f"[TRAIL] trade_id={trade_id} ticket={ticket} ({symbol}) "
                f"trailing ACTIVATED at RR={achieved_rr:.2f}"
            )

        # ── Fetch ATR for trail calculation ───────────────────────────────────
        # Use the ATR already computed during the last analysis pass if cached,
        # otherwise fall back to a simple price-based estimate.
        atr = position.get('last_atr')
        if not atr or atr <= 0:
            # Rough fallback: 0.1% of price — keeps trailing functional if
            # no ATR is cached yet
            atr = current_price * 0.001
            logger.debug(
                f"[TRAIL] No cached ATR for {symbol}, using fallback {atr:.4f}"
            )

        # Track high/low since entry on the position dict
        if direction == 'long':
            position['high_since_entry'] = max(
                position.get('high_since_entry', entry_price), current_price
            )
            position['low_since_entry'] = min(
                position.get('low_since_entry', entry_price), current_price
            )
        else:
            position['high_since_entry'] = max(
                position.get('high_since_entry', entry_price), current_price
            )
            position['low_since_entry'] = min(
                position.get('low_since_entry', entry_price), current_price
            )

        # ── Call the real StopManager method ─────────────────────────────────
        try:
            new_sl = self.stop_manager.compute_trailing_sl(
                direction        = direction,
                current_price    = current_price,
                current_sl       = current_sl,
                high_since_entry = position['high_since_entry'],
                low_since_entry  = position['low_since_entry'],
                atr              = atr,
            )
        except Exception as exc:
            logger.error(f"[TRAIL] StopManager error for ticket={ticket}: {exc}")
            return

        if new_sl is None:
            return

        try:
            await self._send_sl_modify(trade_id, position, new_sl, current_tp, label='trail')
        except Exception as e:
            logger.error(
                f"[TRAIL] Trailing SL modify failed for trade_id={trade_id} "
                f"ticket={ticket}: {e}"
            )
            # Don't propagate — let position monitoring continue



    async def _send_sl_modify(
        self,
        trade_id: str,
        position: dict,
        new_sl: float,
        current_tp: float,
        label: str = 'modify',
    ) -> bool:
        """
        Send modify_position to MT5 and update the local position record on success.
        Uses mt5_file_bridge.modify_position(ticket, stop_loss, take_profit).
        """
        import time as _time

        ticket = position.get('ticket')
        symbol = position.get('symbol')

        try:
            result = await self.mt5_client.modify_position(
                ticket      = ticket,
                stop_loss   = round(new_sl, self._price_digits(symbol)),
                take_profit = current_tp,
            )
        except Exception as exc:
            logger.error(f"[MODIFY] Exception modifying ticket={ticket}: {exc}")
            return False

        if not result.get('success', False):
            logger.warning(
                f"[MODIFY] MT5 rejected modify for ticket={ticket}: {result.get('error')}"
            )
            return False

        old_sl = position.get('stop_loss', 0.0)
        position['stop_loss']      = new_sl
        position['last_sl_update'] = _monotime.time()

        # Persist SL change to database
        try:
            self.db.update_trade(trade_id, {'stop_loss': new_sl})
        except Exception as exc:
            logger.warning(f"[MODIFY] DB update failed for trade_id={trade_id}: {exc}")

        logger.info(
            f"[{label.upper()}] trade_id={trade_id} ticket={ticket} ({symbol}) "
            f"SL moved {old_sl:.5f} → {new_sl:.5f}"
        )
        # Record every SL move to sl_tp_adjustments for learning/analytics
        try:
            entry_price  = position.get('entry_price', 0.0)
            original_sl  = position.get('original_stop_loss') or old_sl
            initial_risk = abs(entry_price - original_sl)
            price_move   = abs(position.get('current_price', entry_price) - entry_price)
            current_rr   = (price_move / initial_risk) if initial_risk > 0 else 0.0

            self.audit_logger.log_sl_tp_adjustment(trade_id, {
                'adjustment_type': label,          # 'breakeven' or 'trail'
                'old_value'      : old_sl,
                'new_value'      : new_sl,
                'trigger_reason' : f"{label} at RR={current_rr:.2f}",
                'current_price'  : position.get('current_price', 0.0),
                'current_rr'     : current_rr,
            })
        except Exception as _log_err:
            logger.debug(f"[TRAIL] sl_tp_adjustments log failed: {_log_err}")
        try:
            self.audit_logger.log_trade_event(
                trade_id   = trade_id,
                event_type = 'sl_adjusted',
                details    = {
                    'old_sl': old_sl,
                    'new_sl': new_sl,
                    'reason': label,
                    'ticket': ticket,
                }
            )
        except Exception:
            pass  # Audit log failure must not block trading

        return True


    def _pip_size(self, symbol: str) -> float:
        """One pip/point for a symbol. Configurable via config pip_sizes map."""
        return self.config.get('pip_sizes', {}).get(symbol, 0.01)


    def _price_digits(self, symbol: str) -> int:
        """Decimal precision for a symbol price. Configurable via price_digits map."""
        return self.config.get('price_digits', {}).get(symbol, 5)


    async def _partial_close_async(
        self,
        trade_id: str,
        position: dict,
        close_fraction: float,
        reason: str = 'tp1',
    ) -> bool:
        """
        Partially close an MT5 position.
        close_fraction=1.0 closes the entire remaining volume.
        """
        ticket         = position.get('ticket')
        symbol         = position.get('symbol')
        current_volume = position.get('position_size') or position.get('volume', 0)
        close_volume   = round(current_volume * close_fraction, 2)

        min_lot = self.config.get('execution', {}).get('min_lot_size', 0.01)
        if close_volume < min_lot:
            logger.warning(
                f"[PARTIAL] trade_id={trade_id}: close volume {close_volume} "
                f"< min lot {min_lot}, skipping"
            )
            return False

        remaining = round(current_volume - close_volume, 2)

        try:
            result = await self.mt5_client.close_position(
                ticket = ticket,
                volume = close_volume,
            )
        except Exception as exc:
            logger.error(f"[PARTIAL] Exception closing ticket={ticket}: {exc}")
            return False

        if not result.get('success', False):
            logger.warning(
                f"[PARTIAL] MT5 rejected partial close for ticket={ticket}: "
                f"{result.get('error')}"
            )
            return False

        position['position_size'] = remaining
        position['volume']        = remaining

        # If full close, remove from tracking
        if remaining <= min_lot:
            self.open_positions.pop(trade_id, None)
            self.db.update_trade(trade_id, {
                'status':       'closed',
                'exit_reason':  reason,
                'exit_time':    datetime.now(timezone.utc).replace(tzinfo=None),
            })
            logger.info(
                f"[PARTIAL] trade_id={trade_id} ticket={ticket} ({symbol}) "
                f"fully closed via {reason}"
            )
        else:
            self.db.update_trade(trade_id, {'position_size': remaining})
            logger.info(
                f"[PARTIAL] trade_id={trade_id} ticket={ticket} ({symbol}): "
                f"closed {close_volume} lots ({reason}), {remaining} lots remain"
            )

        try:
            self.audit_logger.log_trade_event(
                trade_id   = trade_id,
                event_type = 'partial_close',
                details    = {
                    'closed_volume':   close_volume,
                    'remaining_volume': remaining,
                    'reason':          reason,
                }
            )
        except Exception:
            pass

        return True


    def _check_and_handle_tp_levels_sync(
        self,
        trade_id: str,
        position: dict,
        current_price: float,
    ) -> None:
        """
        Check TP1/TP2 targets and trigger partial closes synchronously.
        (Actual close calls are fire-and-forget via asyncio.ensure_future.)
        """
        direction = position.get('direction', 'long')
        ticket    = position.get('ticket')

        tp1 = position.get('take_profit_1') or position.get('tp1_price')
        tp2 = position.get('take_profit_2') or position.get('tp2_price')

        # ── TP1 ──────────────────────────────────────────────────────────────
        if tp1 and not position.get('tp1_hit', False):
            hit = (
                (direction == 'long'  and current_price >= tp1) or
                (direction == 'short' and current_price <= tp1)
            )
            if hit:
                fraction = position.get(
                    'tp1_fraction',
                    self.config.get('risk_management', {}).get(
                        'take_profit', {}
                    ).get('tp1_close_fraction', 0.5),
                )
                logger.info(
                    f"[TP1] trade_id={trade_id} ticket={ticket} hit TP1={tp1:.5f} "
                    f"at price={current_price:.5f}, closing {fraction*100:.0f}%"
                )
                asyncio.ensure_future(
                    self._partial_close_async(trade_id, position, fraction, reason='tp1')
                )
                position['tp1_hit']        = True
                position['trailing_active'] = True  # activate trailing on the remainder

        # ── TP2 ──────────────────────────────────────────────────────────────
        if tp2 and position.get('tp1_hit', False) and not position.get('tp2_hit', False):
            hit = (
                (direction == 'long'  and current_price >= tp2) or
                (direction == 'short' and current_price <= tp2)
            )
            if hit:
                logger.info(
                    f"[TP2] trade_id={trade_id} ticket={ticket} hit TP2={tp2:.5f} "
                    f"— closing remainder"
                )
                asyncio.ensure_future(
                    self._partial_close_async(trade_id, position, 1.0, reason='tp2')
                )
                position['tp2_hit'] = True


    def _sync_positions_with_mt5(self) -> None:
        """
        Pull live positions from MT5 and reconcile with self.open_positions.

        - Adds any MT5 positions not yet tracked (manual trades or restarts).
        - Removes ghost positions (tracked locally but no longer on MT5).
        - Updates current_price, sl, tp, profit for each surviving position.
        """
        try:
            response = self.mt5_client.get_all_positions()
        except Exception as exc:
            logger.error(f"[SYNC] Failed to fetch MT5 positions: {exc}")
            return

        if response.get("status") != "success":
            logger.warning(f"[SYNC] get_all_positions error: {response.get('error')}")
            return

        mt5_positions: list[dict] = response.get("positions", [])
        mt5_tickets = {int(p["ticket"]) for p in mt5_positions}

        # ── Remove ghost positions ───────────────────────────────────────────
        ghost_tickets = [t for t in list(self.open_positions) if t not in mt5_tickets]
        for ticket in ghost_tickets:
            pos = self.open_positions.pop(ticket)
            logger.warning(
                f"[SYNC] Ghost position removed: ticket {ticket} ({pos.get('symbol')}). "
                f"Likely closed externally."
            )
            self.audit_logger.log_event(
                event_type = "position_closed_external",
                ticket     = ticket,
                symbol     = pos.get("symbol"),
            )

        # ── Update surviving + register new positions ────────────────────────
        for mt5_pos in mt5_positions:
            ticket = int(mt5_pos["ticket"])

            if ticket in self.open_positions:
                # Refresh live fields; preserve our strategy metadata
                self.open_positions[ticket].update({
                    "current_price": mt5_pos["current_price"],
                    "sl"           : mt5_pos["sl"],
                    "tp"           : mt5_pos["tp"],
                    "profit"       : mt5_pos["profit"],
                    "volume"       : mt5_pos["volume"],
                    "type"         : mt5_pos["type"],
                    "type"         : mt5_pos["platform"],
                })
                if "stop_loss" not in self.open_positions[ticket]:
                    self.open_positions[ticket]["stop_loss"] = mt5_pos["sl"]
            else:
                # Unknown position — register it so we can manage it going forward.
                # open_time from MT5 is a unix timestamp; convert to datetime so
                # _handle_external_close can compute duration correctly.
                open_time_raw = mt5_pos.get("open_time") or mt5_pos.get("time")
                if open_time_raw:
                    try:
                        entry_time = datetime.fromtimestamp(int(open_time_raw), timezone.utc).replace(tzinfo=None)
                    except Exception:
                        entry_time = None
                else:
                    entry_time = None

                sl = float(mt5_pos.get("sl") or 0.0)

                logger.info(
                    f"[SYNC] Discovered untracked position: ticket {ticket} "
                    f"({mt5_pos['symbol']}). Registering."
                )
                self.open_positions[ticket] = {
                    "ticket"                 : ticket,
                    "symbol"                 : mt5_pos["symbol"],
                    "volume"                 : mt5_pos["volume"],
                    "price"                  : mt5_pos["price"],
                    "entry_price"            : mt5_pos["price"],
                    "current_price"          : mt5_pos["current_price"],
                    "sl"                     : sl,
                    "stop_loss"              : sl,
                    "original_stop_loss"     : sl,   # best guess — breakeven may have already moved it
                    "tp"                     : mt5_pos["tp"],
                    "type"                   : mt5_pos["type"],
                    "profit"                 : mt5_pos["profit"],
                    "entry_time"             : entry_time,   # datetime or None
                    "direction"              : "long" if mt5_pos.get("type") in ("0", 0, "buy") else "short",
                    "trailing_active"        : False,
                    "tp1_hit"                : False,
                    "tp2_hit"                : False,
                    "last_sl_update"         : 0.0,
                    "max_favorable_excursion": 0.0,
                    "max_adverse_excursion"  : 0.0,
                    "source"                 : "sync_discovered",
                    "platform"               : mt5_pos["platform"],
                }


    def _compute_close_fields(self, position: dict, deal: dict) -> dict:
        """
        Compute every derived field for a closed trade from a filled deal dict
        and the position snapshot.  Used by both _handle_external_close and
        _deferred_deal_lookup so neither can drift from the other.

        Args:
            position: Position dict as stored in open_positions (or a snapshot
                      of it). Must contain: entry_price, stop_loss,
                      original_stop_loss, direction, take_profit_1, entry_time,
                      max_favorable_excursion, max_adverse_excursion.
            deal:     Successful get_deal_history response dict. Must contain:
                      exit_price, profit, swap, commission, net_profit,
                      exit_reason, volume, close_time.

        Returns:
            dict with every field needed for log_trade_exit.
        """
        import time as _time

        entry_price  = float(position.get('entry_price', 0.0))
        direction    = position.get('direction', 'long')
        sl           = float(position.get('stop_loss', 0.0))
        tp1          = float(position.get('take_profit_1') or 0.0)
        original_sl  = float(position.get('original_stop_loss') or sl)

        # ── Raw numbers from the broker deal ─────────────────────────────────
        exit_price   = float(deal.get('exit_price', 0.0))
        gross_profit = float(deal.get('profit', 0.0))
        swap         = float(deal.get('swap', 0.0))
        commission   = float(deal.get('commission', 0.0))
        net_pnl      = float(deal.get('net_profit', gross_profit + swap + commission))
        exit_reason  = deal.get('exit_reason', 'external_close')
        close_time   = deal.get('close_time')   # unix epoch int or None

        # ── Duration ─────────────────────────────────────────────────────────
        entry_time = position.get('entry_time')
        # Normalise: DB returns ISO string, live dict holds datetime object
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                entry_time = entry_time.replace(tzinfo=None)   # work in naïve UTC
            except Exception:
                entry_time = None

        duration_minutes = None
        if entry_time:
            if close_time and int(close_time) > 0:
                exit_dt = datetime.fromtimestamp(int(close_time), timezone.utc).replace(tzinfo=None)
            else:
                exit_dt = datetime.now(timezone.utc).replace(tzinfo=None)
            if isinstance(entry_time, datetime):
                delta = exit_dt - entry_time
                duration_minutes = round(delta.total_seconds() / 60, 1)

        # ── Slippage — difference between intended and actual fill ────────────
        # SL hit → compare to stop_loss level
        # TP hit → compare to take_profit_1 level
        # All other reasons → 0
        slippage = 0.0
        if exit_price > 0:
            if exit_reason == 'stop_loss' and sl:
                slippage = round(abs(exit_price - sl), 5)
            elif exit_reason == 'take_profit' and tp1:
                slippage = round(abs(exit_price - tp1), 5)

        # ── Realised R:R — MUST use original SL, not post-breakeven SL ───────
        initial_risk = abs(entry_price - original_sl) if original_sl and entry_price else 0.0
        realized_rr  = 0.0
        if initial_risk > 0 and exit_price > 0 and entry_price > 0:
            price_move  = (exit_price - entry_price) if direction == 'long' \
                          else (entry_price - exit_price)
            realized_rr = round(price_move / initial_risk, 4)

        # ── P&L as % of live equity — never divide by near-zero ──────────────
        equity      = self.current_equity if self.current_equity > 0 else 10_000.0
        pnl_percent = round((net_pnl / equity) * 100, 4)

        return {
            'exit_price'             : exit_price,
            'exit_time'              : exit_dt if exit_dt else datetime.now(timezone.utc).replace(tzinfo=None),
            'exit_reason'            : exit_reason,
            'net_pnl'                : round(net_pnl, 2),
            'pnl_percent'            : pnl_percent,
            'realized_rr'            : realized_rr,
            'duration_minutes'       : duration_minutes,
            'commission'             : round(commission, 2),
            'slippage'               : slippage,
            'max_favorable_excursion': position.get('max_favorable_excursion'),
            'max_adverse_excursion'  : position.get('max_adverse_excursion'),
        }

    def _apply_close_side_effects(self, net_pnl: float, symbol: str = '') -> None:
        """
        Update in-memory stats that are side-effects of closing a trade.
        Called by both _handle_external_close and _deferred_deal_lookup.
        Separated so the deferred path can't forget to run it.
        """
        import time as _time

        if net_pnl < 0:
            self.consecutive_losses = getattr(self, 'consecutive_losses', 0) + 1
            self.last_loss_time     = _monotime.time()
            logger.info(f"[CLOSE] Loss — consecutive_losses={self.consecutive_losses}")
        else:
            self.consecutive_losses = 0

        # Track per-symbol daily P&L for per-symbol drawdown enforcement
        if symbol:
            sym_pnl = self.daily_stats.setdefault('symbol_pnl', {})
            sym_pnl[symbol] = sym_pnl.get(symbol, 0.0) + net_pnl

    async def _handle_external_close(self, trade_id: str, position: dict):
        """
        Handle a position closed outside our system (broker SL/TP, manual close,
        margin call, etc.).

        Fetches the actual closing deal from MT5 history (with up to 3 retries
        using growing lookback windows) so the database record contains real
        exit_price, P&L, duration, commission, slippage, MFE/MAE and
        exit_reason.  Also captures account equity immediately after the close
        so the analytics tab can plot a real equity curve.

        If all retries fail, marks the trade as 'pending_exit' and fires a
        background _deferred_deal_lookup that retries at 30 s / 120 s / 300 s.
        Never writes zeros to the database.
        """
        ticket      = position.get('ticket')
        symbol      = position.get('symbol', '')

        logger.info(f"Recording external close for {trade_id} (ticket={ticket})")

        # ── Fetch real deal data from MT5 history (up to 3 attempts) ─────────
        base_lookback = self.config.get('monitor', {}).get('history_lookback_hours', 48)
        deal = {'status': 'failed'}

        if ticket:
            for attempt, lookback in enumerate(
                [base_lookback, base_lookback * 2, base_lookback * 4], start=1
            ):
                try:
                    deal = await self.mt5_client.get_deal_history(
                        ticket        = int(ticket),
                        lookback_hours= lookback,
                    )
                    if deal.get('status') == 'success' and float(deal.get('exit_price', 0)) > 0:
                        break
                    logger.warning(
                        f"[CLOSE] get_deal_history attempt {attempt}/3 "
                        f"ticket={ticket} lookback={lookback}h: "
                        f"{deal.get('error', 'no exit_price')}"
                    )
                except Exception as e:
                    logger.error(
                        f"[CLOSE] Exception on attempt {attempt}/3 for ticket={ticket}: {e}"
                    )
                if attempt < 3:
                    await asyncio.sleep(2.0 * attempt)

        # ── CRITICAL: Always remove from open_positions first BEFORE any DB updates ──
        self.open_positions.pop(trade_id, None)

        # ── All retries failed — mark as pending and retry in background ─────
        if deal.get('status') != 'success' or float(deal.get('exit_price', 0)) <= 0:
            logger.warning(
                f"[CLOSE] All deal history attempts failed for ticket={ticket}. "
                f"Marking as pending_exit — retrying in background."
            )
            self.db.update_trade(trade_id, {
                'status'     : 'pending_exit',
                'exit_reason': 'pending_deal_lookup',
                'exit_time'  : datetime.now(timezone.utc).replace(tzinfo=None),
            })
            asyncio.ensure_future(
                self._deferred_deal_lookup(trade_id, ticket, position)
            )
            return

        # ── Log each partial close deal separately to the audit trail ─────────
        for deal_item in deal.get('deals', []):
            self.audit_logger.log_order_event({
                'trade_id'    : trade_id,
                'event_type'  : 'partial_close' if len(deal.get('deals', [])) > 1 else 'close',
                'price'       : deal_item.get('exit_price'),
                'quantity'    : deal_item.get('volume'),
                'api_response': deal_item,
                'notes'       : f"deal_ticket={deal_item.get('deal_ticket')}"
            })

        # ── Got real data — compute all derived fields via shared helper ───────
        # _compute_close_fields expects the aggregated deal info at the top level
        fields = self._compute_close_fields(position, deal)

        logger.info(
            f"[CLOSE] ticket={ticket} exit={fields['exit_price']:.5f} "
            f"net_pnl={fields['net_pnl']:.2f} commission={fields['commission']:.2f} "
            f"reason={fields['exit_reason']} rr={fields['realized_rr']:.2f} "
            f"duration={fields['duration_minutes']}m slippage={fields['slippage']}"
        )

        # ── Side-effects: consecutive losses, daily drawdown ──────────────────
        self._apply_close_side_effects(fields['net_pnl'], position.get('symbol', ''))

        # ── Snapshot equity immediately after close ────────────────────────────
        # This is stored per-trade so the dashboard can plot a real equity curve
        # instead of extrapolating from a starting balance.
        equity_after_close = None
        try:
            balance_result = await self.mt5_client.get_balance()
            if balance_result.get('success'):
                equity_after_close = balance_result.get('equity')
                # Keep our cached equity fresh
                self.current_equity = equity_after_close
        except Exception as e:
            logger.debug(f"[CLOSE] Could not fetch equity after close: {e}")

        # ── Write to audit log + database ─────────────────────────────────────
        self.audit_logger.log_trade_exit(trade_id, {
            'exit_price'             : fields['exit_price'],
            'reason'                 : fields['exit_reason'],
            'pnl'                    : fields['net_pnl'],
            'pnl_percent'            : fields['pnl_percent'],
            'realized_rr'            : fields['realized_rr'],
            'duration_minutes'       : fields['duration_minutes'],
            'commission'             : fields['commission'],
            'slippage'               : fields['slippage'],
            'max_favorable_excursion': fields['max_favorable_excursion'],
            'max_adverse_excursion'  : fields['max_adverse_excursion'],
            'equity_after_close'     : equity_after_close,
        })

        # ── Notification ──────────────────────────────────────────────────────
        if hasattr(self, 'notifier'):
            emoji   = '✅' if fields['net_pnl'] >= 0 else '❌'
            dur_str = f"{fields['duration_minutes']:.0f}m" if fields['duration_minutes'] else "?"
            eq_str  = f" | Equity: ${equity_after_close:,.2f}" if equity_after_close else ""
            await self.notifier.send(
                f"{emoji} {symbol} closed ({fields['exit_reason']})\n"
                f"Exit: {fields['exit_price']:.5f} | Net P&L: {fields['net_pnl']:.2f} "
                f"| RR: {fields['realized_rr']:.2f} | Duration: {dur_str}{eq_str}"
            )

        # ── Update database with final close information ──────────────────────
        self.db.update_trade(trade_id, {
            'status'     : 'closed',
            'exit_price' : fields['exit_price'],
            'exit_time'  : fields['exit_time'],
            'exit_reason': fields['exit_reason'],
            'pnl'        : fields['net_pnl'],
            'pnl_percent': fields['pnl_percent'],
            'duration_minutes': fields['duration_minutes'],
            'commission' : fields['commission'],
            'slippage'   : fields['slippage'],
            'max_favorable_excursion': fields['max_favorable_excursion'],
            'max_adverse_excursion'  : fields['max_adverse_excursion'],
            'realized_rr': fields['realized_rr'],
        })

    async def _deferred_deal_lookup(
        self,
        trade_id: str,
        ticket,
        position: dict,
        min_lookback_hours: int = 0,   # ← new param, 0 = use config default
    ) -> None:
        base_lookback = max(
            min_lookback_hours,
            self.config.get('monitor', {}).get('history_lookback_hours', 48)
        )

        # Deferred retries use a genuinely progressive lookback window.
        # The sync path in _handle_external_close already tried base, base*2, base*4
        # (48h / 96h / 192h). Start the deferred path beyond that:
        #   30s  delay → 192h  (same as last sync attempt — covers EA being briefly busy)
        #   120s delay → 288h  (12 days — weekend/broker holiday gap)
        #   300s delay → 384h  (16 days — max reasonable history window)
        for delay, lookback in [
            (30,  base_lookback * 4),
            (120, base_lookback * 6),
            (300, base_lookback * 8),
        ]:
            await asyncio.sleep(delay)

            if not self.running:
                logger.info(
                    f"[DEFERRED_CLOSE] System shutting down — "
                    f"aborting deferred lookup for trade_id={trade_id}"
                )
                return

            # ── RE-ADOPTION CHECK ──────────────────────────────────────────────
            # Before searching deal history, verify the position is actually
            # closed on MT5. If it is still open, the false-close that triggered
            # this deferred lookup was caused by a transient bridge error (e.g.
            # Bug 2: get_all_positions timeout returning []). Re-adopt the
            # position back into open_positions and cancel this deferred lookup.
            try:
                pos_info = await self.mt5_client.get_position_info(int(ticket))
                if pos_info and pos_info.get('status') == 'success':
                    logger.warning(
                        f"[DEFERRED_CLOSE] ticket={ticket} is STILL OPEN on MT5 — "
                        f"false close detected. Re-adopting trade_id={trade_id} "
                        f"back into open_positions and cancelling deferred lookup."
                    )
                    # Restore DB status to open
                    self.db.update_trade(trade_id, {
                        'status'     : 'open',
                        'exit_reason': None,
                        'exit_time'  : None,
                    })
                    # Re-adopt into live tracking so the monitor resumes managing it
                    if trade_id not in self.open_positions:
                        self.open_positions[trade_id] = {
                            **position,
                            'trade_id'       : trade_id,
                            'ticket'         : ticket,
                            'current_price'  : float(pos_info.get('current_price', position.get('entry_price', 0.0))),
                            'stop_loss'      : float(pos_info.get('sl', position.get('stop_loss', 0.0))),
                            'trailing_active': position.get('trailing_active', False),
                            'tp1_hit'        : position.get('tp1_hit', False),
                            'tp2_hit'        : position.get('tp2_hit', False),
                        }
                    return  # ← cancel the deferred close entirely

            except Exception as e:
                logger.debug(
                    f"[DEFERRED_CLOSE] Re-adoption check failed for ticket={ticket}: {e} "
                    f"— proceeding with deal history lookup"
                )

            logger.info(
                f"[DEFERRED_CLOSE] Attempting deal history for "
                f"trade_id={trade_id} ticket={ticket} lookback={lookback}h"
            )

            try:
                deal = await self.mt5_client.get_deal_history(
                    ticket        = int(ticket),
                    lookback_hours= lookback,
                )
            except Exception as e:
                logger.warning(
                    f"[DEFERRED_CLOSE] Exception fetching deal for ticket={ticket}: {e}"
                )
                continue

            if deal.get('status') != 'success' or float(deal.get('exit_price', 0)) <= 0:
                logger.warning(
                    f"[DEFERRED_CLOSE] No usable data yet for ticket={ticket}: "
                    f"{deal.get('error', 'no exit_price')}"
                )
                continue

            # ── Got real data — compute all fields the same way as _handle_external_close ──
            try:
                fields = self._compute_close_fields(position, deal)
            except Exception as e:
                logger.error(
                    f"[DEFERRED_CLOSE] _compute_close_fields failed for "
                    f"trade_id={trade_id}: {e}"
                )
                continue

            # ── Log each partial close deal separately to the audit trail ────────
            for deal_item in deal.get('deals', []):
                self.audit_logger.log_order_event({
                    'trade_id'    : trade_id,
                    'event_type'  : 'partial_close' if len(deal.get('deals', [])) > 1 else 'close',
                    'price'       : deal_item.get('exit_price'),
                    'quantity'    : deal_item.get('volume'),
                    'api_response': deal_item,
                    'notes'       : f"deal_ticket={deal_item.get('deal_ticket')} [deferred]"
                })

            logger.info(
                f"[DEFERRED_CLOSE] ✓ trade_id={trade_id} ticket={ticket} "
                f"exit={fields['exit_price']:.5f} net_pnl={fields['net_pnl']:.2f} "
                f"rr={fields['realized_rr']:.2f} duration={fields['duration_minutes']}m "
                f"commission={fields['commission']:.2f} slippage={fields['slippage']}"
            )

            # Apply in-memory side-effects (consecutive losses, daily drawdown)
            self._apply_close_side_effects(fields['net_pnl'], position.get('symbol', ''))

            # Write to audit log + DB
            self.audit_logger.log_trade_exit(trade_id, {
                'exit_price'             : fields['exit_price'],
                'reason'                 : fields['exit_reason'],
                'pnl'                    : fields['net_pnl'],
                'pnl_percent'            : fields['pnl_percent'],
                'realized_rr'            : fields['realized_rr'],
                'duration_minutes'       : fields['duration_minutes'],
                'commission'             : fields['commission'],
                'slippage'               : fields['slippage'],
                'max_favorable_excursion': fields['max_favorable_excursion'],
                'max_adverse_excursion'  : fields['max_adverse_excursion'],
            })

            # Update database with final close information
            self.db.update_trade(trade_id, {
                'status'     : 'closed',
                'exit_price' : fields['exit_price'],
                'exit_time'  : fields['exit_time'],
                'exit_reason': fields['exit_reason'],
                'pnl'        : fields['net_pnl'],
                'pnl_percent': fields['pnl_percent'],
                'duration_minutes': fields['duration_minutes'],
                'commission' : fields['commission'],
                'slippage'   : fields['slippage'],
                'max_favorable_excursion': fields['max_favorable_excursion'],
                'max_adverse_excursion'  : fields['max_adverse_excursion'],
                'realized_rr': fields['realized_rr'],
            })

            if hasattr(self, 'notifier') and self.notifier:
                await self.notifier.notify_trade_close(
                    symbol           = position.get('symbol', ''),
                    direction        = position.get('direction', ''),
                    entry_price      = position.get('entry_price', 0.0),
                    exit_price       = fields['exit_price'],
                    pnl              = fields['net_pnl'],
                    realized_rr      = fields['realized_rr'],
                    exit_reason      = fields['exit_reason'],
                    duration_minutes = fields['duration_minutes'],
                    trade_id         = trade_id,
                    ticket           = ticket,
                )

            return   # ── Done — record fully and correctly populated ──────────

        # ── All retries exhausted ────────────────────────────────────────────
        logger.error(
            f"[DEFERRED_CLOSE] All retries exhausted for trade_id={trade_id} "
            f"ticket={ticket}. Marking closed without deal data."
        )
        self.db.update_trade(trade_id, {
            'status'     : 'closed',
            'exit_reason': 'deal_history_unavailable',
        })


    async def _startup_pnl_backfill(self):
        await asyncio.sleep(10)

        try:
            trades = self.db.get_trades_missing_pnl()
        except Exception as e:
            logger.error(f"[PNL_BACKFILL] DB query failed: {e}")
            return

        if not trades:
            logger.info("[PNL_BACKFILL] No trades missing P&L — nothing to backfill.")
            return

        seen = set()
        unique = []
        for t in trades:
            tid = t.get('trade_id')
            if tid and tid not in seen:
                seen.add(tid)
                unique.append(t)

        logger.info(f"[PNL_BACKFILL] {len(unique)} trade(s) need P&L data — backfilling in background.")

        base_lookback = self.config.get('monitor', {}).get('history_lookback_hours', 48)

        for i, trade in enumerate(unique):
            if not self.running:
                logger.info("[PNL_BACKFILL] System stopping — aborting backfill.")
                break

            trade_id = trade.get('trade_id')
            ticket   = trade.get('ticket')

            if not ticket:
                logger.debug(f"[PNL_BACKFILL] {trade_id} has no ticket — skipping.")
                continue

            # ── Calculate lookback from actual trade age ──────────────────────
            # Always add a 24h buffer beyond the trade age so the first
            # attempt covers the full window without needing a retry cycle.
            entry_time_raw = trade.get('entry_time') or trade.get('exit_time')
            lookback = base_lookback  # fallback

            if entry_time_raw:
                try:
                    entry_dt = datetime.fromisoformat(
                        str(entry_time_raw).replace('Z', '+00:00')
                    ).replace(tzinfo=None)
                    age_hours = (datetime.now(timezone.utc).replace(tzinfo=None) - entry_dt).total_seconds() / 3600
                    # Cover full age + 24h buffer, minimum base_lookback
                    lookback = max(base_lookback, int(age_hours) + 24)
                except Exception:
                    lookback = base_lookback * 4  # safe fallback if parsing fails

            logger.info(
                f"[PNL_BACKFILL] Queuing {trade_id} ticket={ticket} "
                f"(lookback={lookback}h, {i+1}/{len(unique)})"
            )

            position = {
                'entry_price'            : trade.get('entry_price', 0.0),
                'direction'              : trade.get('direction', 'long'),
                'entry_time'             : trade.get('entry_time'),
                'stop_loss'              : trade.get('stop_loss', 0.0),
                'original_stop_loss'     : trade.get('original_stop_loss') or trade.get('stop_loss', 0.0),
                'take_profit_1'          : trade.get('take_profit_1', 0.0),
                'volume'                 : trade.get('position_size', 0.0),
                'position_size'          : trade.get('position_size', 0.0),
                'max_favorable_excursion': trade.get('max_favorable_excursion'),
                'max_adverse_excursion'  : trade.get('max_adverse_excursion'),
                # Required by re-adoption path in _deferred_deal_lookup:
                # if the position is found to still be alive on MT5, these fields
                # are spread into open_positions via **position. Without them,
                # the monitor loop crashes on position['platform'] / position['symbol'].
                'platform'               : trade.get('platform', 'mt5'),
                'symbol'                 : trade.get('symbol', ''),
                'ticket'                 : ticket,
            }

            asyncio.create_task(
                self._deferred_deal_lookup(trade_id, ticket, position,
                                        min_lookback_hours=lookback)
            )

            await asyncio.sleep(1.0)

        logger.info(f"[PNL_BACKFILL] All {len(unique)} backfill tasks queued.")


    async def _update_position(self, trade_id: str, position: dict):
        """Update single position."""
        try:
            symbol = position['symbol']
            platform = position['platform']
            
            # Get current price
            if platform == 'mt5':
                pos_info = await self.mt5_client.get_position_info(position['ticket'])
            
                
            if not pos_info:
                # Position may have been closed
                del self.open_positions[trade_id]
                return
                
            # Get current market data for trailing stop logic
            # (Simplified - would fetch actual data)
            
            # For now, just check if SL/TP hit via position info
            # In real implementation, would update trailing stops here
            
        except Exception as e:
            logger.error(f"Error updating position {trade_id}: {e}")

    def _check_cooldown_after_losses(self) -> bool:
        """
        Return True if the system is allowed to enter new trades.
        Blocks entries if consecutive_losses >= threshold and cooldown has not expired.
        """
        cfg_cd     = self.config.get("risk", {}).get("cooldown", {})
        max_losses = cfg_cd.get("consecutive_loss_count", 3)
        cooldown_s = cfg_cd.get("cooldown_seconds", 1800)   # 30 min default

        if self.consecutive_losses < max_losses:
            return True

        elapsed = _monotime.time() - self.last_loss_time
        if elapsed < cooldown_s:
            remaining = int(cooldown_s - elapsed)
            logger.info(
                f"[COOLDOWN] {self.consecutive_losses} consecutive losses. "
                f"Cooldown active — {remaining}s remaining."
            )
            return False

        # Cooldown expired — reset counter
        logger.info("[COOLDOWN] Cooldown expired, resuming entries.")
        self.consecutive_losses = 0
        return True

    def _emergency_shutdown(self, reason: str = "manual") -> None:
        """
        Halt all new activity and set kill-switch.
        Whether positions are closed is controlled by config:
        close_positions_on_shutdown: false  → leave open, reconcile on restart
        close_positions_on_shutdown: true   → close all immediately
        """
        logger.critical(f"[SHUTDOWN] Emergency shutdown triggered: {reason}")
        self.kill_switch = True

        self.audit_logger.log_event(event_type="emergency_shutdown", reason=reason)

        close_cfg = (
            self.config
                .get('risk_management', {})
                .get('global_limits', {})
                .get('emergency_shutdown', {})
        )
        close_on_shutdown = close_cfg.get('close_positions_on_shutdown', False)

        if hasattr(self, "notifier"):
            action = "Closing all positions." if close_on_shutdown else "Positions left open — will reconcile on restart."
            asyncio.ensure_future(self.notifier.send(f"🚨 Emergency shutdown: {reason}. {action}"))

        if close_on_shutdown:
            # Fire async close from the sync context via the running event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._emergency_close_all())
            logger.critical("[SHUTDOWN] Position close scheduled.")
        else:
            logger.critical(
                f"[SHUTDOWN] close_positions_on_shutdown=false — "
                f"{len(self.open_positions)} position(s) left open. "
                f"They will be reconciled on next startup."
            )

        logger.critical("[SHUTDOWN] Emergency shutdown complete.")

    def _register_new_position(
        self,
        trade_id: str,
        ticket: int,
        symbol: str,
        direction: str,         # 'long' | 'short'  (matches rest of codebase)
        entry_price: float,
        volume: float,
        sl: float,
        tp1: float,
        tp2: float | None = None,
        tp1_fraction: float = 0.5,
        platform: str = 'mt5',
        analysis_id: str | None = None,
    ) -> dict:
        """
        Register a freshly filled order in self.open_positions.
        Keyed by trade_id (string UUID) to match the rest of the codebase.
        """
        import time as _time

        position = {
            'trade_id'       : trade_id,
            'ticket'         : ticket,
            'symbol'         : symbol,
            'platform'       : platform,
            'direction'      : direction,
            'entry_price'    : entry_price,
            'current_price'  : entry_price,
            'position_size'  : volume,
            'volume'         : volume,
            'stop_loss'      : sl,
            'original_stop_loss': sl,
            'take_profit_1'  : tp1,
            'take_profit_2'  : tp2,
            'tp1_fraction'   : tp1_fraction,
            'tp1_hit'        : False,
            'tp2_hit'        : False,
            'trailing_active': False,
            'last_sl_update' : 0.0,
            'profit'         : 0.0,
            'analysis_id'    : analysis_id,
            'entry_time'     : datetime.now(timezone.utc).replace(tzinfo=None),
            'open_time'      : _monotime.time(),
        }
        self.open_positions[trade_id] = position

        logger.info(
            f"[REGISTER] trade_id={trade_id} ticket={ticket} {symbol} "
            f"{direction.upper()} {volume} lots @ {entry_price:.5f} "
            f"| SL={sl:.5f} TP1={tp1:.5f}"
        )
        try:
            self.audit_logger.log_trade_event(
                trade_id   = trade_id,
                event_type = 'order_placed',
                details    = {
                    'ticket'     : ticket,
                    'symbol'     : symbol,
                    'direction'  : direction,
                    'entry_price': entry_price,
                    'volume'     : volume,
                    'sl'         : sl,
                    'tp1'        : tp1,
                    'tp2'        : tp2,
                    'analysis_id': analysis_id,
                }
            )
        except Exception:
            pass

        if hasattr(self, 'notifier') and self.notifier:
            asyncio.ensure_future(self.notifier.notify_trade_entry(
                symbol             = symbol,
                direction          = direction,
                entry_price        = entry_price,
                stop_loss          = sl,
                take_profit_1      = tp1,
                take_profit_2      = tp2,
                position_size      = volume,
                expected_rr        = 0.0,
                confluence_reasons = [],
                trade_id           = trade_id,
                ticket             = ticket,
            ))

        return position

    def _close_and_unregister(self, position: dict, reason: str = "strategy") -> bool:
        """
        Close an entire position on MT5 and remove it from self.open_positions.
        Updates consecutive_losses counter.
        """
        ticket = position["ticket"]

        try:
            result = self.mt5_client.close_position(ticket=ticket)
        except Exception as exc:
            logger.error(f"[CLOSE] Exception for ticket {ticket}: {exc}")
            return False

        if result.get("status") != "success":
            logger.warning(
                f"[CLOSE] MT5 rejected close for {ticket}: {result.get('error')}"
            )
            return False

        profit = position.get("profit", 0.0)
        self.open_positions.pop(ticket, None)

        if profit < 0:
            self.consecutive_losses += 1
            self.last_loss_time = _monotime.time()
        else:
            self.consecutive_losses = 0

        logger.info(
            f"[CLOSE] Ticket {ticket} ({position.get('symbol')}) closed. "
            f"P&L={profit:.2f} | reason={reason}"
        )
        self.audit_logger.log_event(
            event_type = "position_closed",
            ticket     = ticket,
            symbol     = position.get("symbol"),
            profit     = profit,
            reason     = reason,
        )

        if hasattr(self, "notifier"):
            emoji = "✅" if profit >= 0 else "❌"
            asyncio.ensure_future(self.notifier.send(
                f"{emoji} Position closed: {position.get('symbol')} "
                f"| P&L: {profit:.2f} | Reason: {reason}"
            ))
        return True

    async def _learning_loop(self):
        """Run learning engine periodically."""
        if not self.learner:
            return
            
        logger.info("Learning loop started")
        
        schedule_hours = self.config.get('learning', {}).get('learning_schedule_hours', 24)
        
        while self.running:
            try:
                await asyncio.sleep(schedule_hours * 3600)
                
                logger.info("Running learning cycle...")
                
                # Would run actual learning here
                result = self.learner.run_grid_search(days_lookback=90)
                logger.info(f"Learning cycle result: {result.get('message', result.get('status'))}")
                self.audit_logger.log_learning_event({
                    'event_type': 'learning_cycle_complete',
                    'message'   : result.get('message', ''),
                    'status'    : result.get('status'),
                    'run_id'    : result.get('run_id'),
                })
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}", exc_info=True)
                
    async def _daily_summary_loop(self):
        """Generate daily performance summary."""
        logger.info("Daily summary loop started")
        
        while self.running:
            try:
                # Wait until end of day
                await asyncio.sleep(86400)  # 24 hours
                
                # Generate summary
                summary = self.audit_logger.generate_daily_summary()
                logger.info(f"Daily Summary: {summary}")
                
                # Reset daily stats
                self.daily_stats['trades_today'] = 0
                self.daily_stats['symbol_pnl'] = {}   # ← add this line
                self.daily_stats['starting_balance'] = await self._get_total_balance()
                self.halt_new_trades = False   # ← add this line
                logger.info("[DAILY] Daily drawdown halt cleared for new trading day.")
                
            except Exception as e:
                logger.error(f"Error in daily summary: {e}", exc_info=True)
                
    async def _get_total_balance(self) -> float:
            """
            Fetch live account equity across all connected platforms.

            MT5  → authenticate action → returns balance + equity (we use equity
                so unrealised P&L is included in risk calculations).

            Falls back to the last cached self.current_equity if a platform call
            fails, so a temporary disconnect doesn't zero-out the risk engine.
            In demo mode the MT5 bridge returns the simulated equity from
            _simulate_command, so no special-casing is needed here.
            """
            total_equity = 0.0
            any_platform_succeeded = False

            # ── MT5 ──────────────────────────────────────────────────────────────
            if self.mt5_client and self.mt5_client.is_connected():
                try:
                    mt5_balance = await self.mt5_client.get_balance()
                    if mt5_balance.get('success'):
                        mt5_equity = mt5_balance.get('equity', 0.0)
                        total_equity += mt5_equity
                        any_platform_succeeded = True
                        # logger.info(f"MT5 equity: ${mt5_equity:,.2f}")
                    else:
                        logger.warning(
                            f"MT5 get_balance returned failure: {mt5_balance}"
                        )
                except Exception as e:
                    logger.error(f"Error fetching MT5 balance: {e}")
            else:
                logger.debug("MT5 client not connected — skipping MT5 balance fetch")

            # ── Fallback ──────────────────────────────────────────────────────────
            if not any_platform_succeeded:
                if self.current_equity > 0:
                    logger.warning(
                        f"All balance fetches failed — using last known equity: "
                        f"${self.current_equity:,.2f}"
                    )
                    return self.current_equity
                else:
                    logger.error(
                        "All balance fetches failed and no cached equity available. "
                        "Returning 0.0 — risk calculations will be blocked."
                    )
                    return 0.0

            # ── Cache and return ──────────────────────────────────────────────────
            self.current_equity = total_equity
            # logger.info(f"Total equity: ${total_equity:,.2f}")
            return total_equity
        
    def _get_current_exposure(self) -> dict:
        """Get current exposure summary."""
        symbols = {}
        for position in self.open_positions.values():
            symbol = position['symbol']
            if symbol not in symbols:
                symbols[symbol] = {'risk_percent': 0, 'count': 0}
            symbols[symbol]['count'] += 1

        exposure = {
            'open_count': len(self.open_positions),  # Actual count
            'symbols': symbols
        }

        logger.debug(f"Current exposure: {exposure['open_count']} positions")
            
        return exposure
        
    def _get_recent_trades(self, n: int = 10) -> list[dict]:
        """Get recent closed trades."""
        return self.db.get_trades(filters={'status': 'closed'}, limit=n)
        
    async def shutdown(self):
        if not self.running:
            return

        logger.info("=" * 60)
        logger.info("Shutting down Trading System")
        logger.info("=" * 60)

        self.running = False
        await asyncio.sleep(1)

        close_cfg = (
            self.config
                .get('risk_management', {})
                .get('global_limits', {})
                .get('emergency_shutdown', {})
        )
        close_on_shutdown = close_cfg.get('close_positions_on_shutdown', False)

        if close_on_shutdown and self.open_positions:
            logger.warning(f"Shutdown — closing {len(self.open_positions)} position(s) as configured.")
            for trade_id, position in list(self.open_positions.items()):
                try:
                    if position.get('platform') == 'mt5':
                        await self.mt5_client.close_position(position['ticket'])
                except Exception as e:
                    logger.error(f"Error closing position on shutdown: {e}")
        elif self.open_positions:
            logger.info(
                f"[SHUTDOWN] Leaving {len(self.open_positions)} position(s) open "
                f"(close_positions_on_shutdown=false). "
                f"They will be reconciled on next startup."
            )

        # disconnect clients / DB as before
        try:
            await self.mt5_client.disconnect()
        except Exception:
            pass
        try:
            await self.market_client.close_all()
        except Exception:
            pass
        try:
            self.db.disconnect()
        except Exception:
            pass

        logger.info("Trading System shutdown complete")


async def main():
    """Main entry point."""
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Initialize system
    system = TradingSystem()
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}")
        # Cancel all tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
    
    # Register signal handlers
    if sys.platform != 'win32':
        # Unix signals
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
    
    # Start system
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt - shutting down gracefully...")
        await system.shutdown()
    except asyncio.CancelledError:
        logger.info("Tasks cancelled - shutting down...")
        await system.shutdown()
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        await system.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)