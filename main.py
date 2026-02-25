"""
Main orchestrator for the trading system.
Coordinates all components and manages the trading loop.
"""

import asyncio
import logging
import signal
import sys
import time as _time
from datetime import datetime, time
from pathlib import Path
import yaml
from dotenv import load_dotenv
import os

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

# Setup logging
_file_handler   = logging.FileHandler('logs/trading_system.log', encoding='utf-8')
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[_file_handler, _stream_handler]
)
logger = logging.getLogger(__name__)


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
        
        # State
        self.running = False
        self.open_positions = {}
        self.current_equity = 0.0
        self.daily_stats = {
            'trades_today': 0,
            'daily_drawdown_percent': 0,
            'starting_balance': 0
        }
        self.consecutive_losses = 0
        self.last_loss_time     = 0.0
        
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
        self.db.connect()
        
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
        self.indicators = TechnicalIndicators(self.config.get('indicators', {}))
        self.strategy_engine = StrategyEngine(self.config)
        self.money_manager = MoneyManager(self.config)
        self.stop_manager = StopManager(self.config)
        
        logger.info("Strategy components initialized")
        
    def _init_execution_clients(self):
        """Initialize execution clients."""
        demo_mode = self.config['general']['mode'] == 'live'
        
        # MT5
        mt5_config = {
            'host': os.getenv('MT5_BRIDGE_HOST', 'localhost'),
            'port': int(os.getenv('MT5_BRIDGE_PORT', 9090)),
            'account': os.getenv('MT5_ACCOUNT'),
            'password': os.getenv('MT5_PASSWORD'),
            'server': os.getenv('MT5_SERVER'),
            'magic_number': 123456
        }
        self.mt5_client = MT5Bridge(mt5_config, demo_mode=False)

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
                        logger.error(
                            f"DRAWDOWN LIMIT HIT: {drawdown:.2f}% >= {max_dd}%"
                        )
                        
                        # Check if emergency shutdown enabled
                        emergency_config = self.config.get('risk_management', {}).get(
                            'global_limits', {}
                        ).get('emergency_shutdown', {})
                        
                        if emergency_config.get('enabled', True):
                            logger.error("EMERGENCY SHUTDOWN TRIGGERED!")
                            self.audit_logger.log_risk_event({
                                'event_type': 'emergency_shutdown',
                                'drawdown_percent': drawdown,
                                'max_allowed': max_dd,
                                'message': 'Emergency shutdown triggered by drawdown limit'
                            })
                            
                            # Stop trading
                            self.running = False
                            
                            # Close all positions if configured
                            if emergency_config.get('auto_close_all', True):
                                await self._emergency_close_all()
                    
            except Exception as e:
                logger.error(f"Error in balance monitor: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _emergency_close_all(self):
        """Emergency close all positions."""
        logger.warning("EMERGENCY: Closing all positions!")
        
        for trade_id, position in list(self.open_positions.items()):
            try:
                platform = position['platform']
                symbol = position['symbol']
                ticket = position.get('ticket')
                
                if platform == 'mt5' and ticket:
                    result = await self.mt5_client.close_position(ticket)
                    logger.info(f"Emergency closed MT5 position {ticket}: {result}")
                
                # Log the closure
                self.audit_logger.log_trade_exit(trade_id, {
                    'exit_price': 0,  # Will be filled from actual result
                    'reason': 'emergency_shutdown',
                    'pnl': 0,
                    'pnl_percent': 0,
                    'realized_rr': 0
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

        enabled_symbols = [s for s, c in self.config.get('symbols', {}).items() if c.get('enabled')]
        await self.market_hours.prefetch_all(enabled_symbols)
        # this fires one get_symbol_sessions per symbol at startup, logs the full schedule

        
        # Start main trading loop
        try:
            await asyncio.gather(
                self._trading_loop(),
                self._position_monitor_loop(),
                self._balance_monitor_loop(),
                self._learning_loop(),
                self._daily_summary_loop()
            )
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
                        'exit_time'  : datetime.utcnow(),
                    })
                    continue

                # ── MT5 verification ──────────────────────────────────────────
                if platform == 'mt5':
                    pos_info = await self.mt5_client.get_position_info(int(ticket))

                    if not pos_info:
                        logger.warning(
                            f"[STARTUP] trade_id={trade_id} ticket={ticket} "
                            f"not found on MT5 — was closed while system was down."
                        )
                        # Build a minimal position dict so _handle_external_close
                        # has the data it needs for the deal history lookup
                        ghost_position = {
                            'ticket'             : ticket,
                            'symbol'             : trade.get('symbol'),
                            'direction'          : trade.get('direction'),
                            'entry_price'        : trade.get('entry_price', 0.0),
                            'stop_loss'          : trade.get('stop_loss', 0.0),
                            'original_stop_loss' : trade.get('stop_loss', 0.0),
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
        """Main trading loop - WITH POSITION COUNT CHECK."""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                # ✅ CHECK: Verify we haven't exceeded limits BEFORE analyzing
                current_positions = len(self.open_positions)
                max_concurrent = self.config.get('risk_management', {}).get(
                    'global_limits', {}
                ).get('max_concurrent_trades', 3)
                
                if current_positions >= max_concurrent:
                    logger.debug(
                        f"Skipping analysis - at max positions "
                        f"({current_positions}/{max_concurrent})"
                    )
                    await asyncio.sleep(903)
                    continue
                
                # Get enabled symbols — shuffle each cycle so no single symbol
                import random as _random
                symbols_config = self.config.get('symbols', {})
                enabled_symbols = [
                    (symbol, cfg)
                    for symbol, cfg in symbols_config.items()
                    if cfg.get('enabled', False)
                ]
                _random.shuffle(enabled_symbols)
                
                for symbol, symbol_config in enabled_symbols:
                    # ✅ DOUBLE CHECK before each symbol
                    if len(self.open_positions) >= max_concurrent:
                        logger.info(
                            f"Reached max positions ({max_concurrent}), "
                            f"stopping new entries this cycle"
                        )
                        break
                    # ── Market hours guard ────────────────────────────────────
                    if not self.market_hours.is_open(symbol):
                        # Double-check with live price freshness — works even
                        # without EA v2.2 (uses existing get_historical action).
                        price_fresh = await self.market_hours.is_open_by_price(
                            symbol, self.mt5_client, timeframe='1H'
                        )
                        if not price_fresh:
                            logger.info(
                                f"[MARKET_HOURS] {symbol} closed — "
                                f"next open: {self.market_hours.next_open_str(symbol)}"
                            )
                            continue
                        # Schedule says closed but live price data is fresh →
                        # trust the price, proceed with analysis
                        logger.debug(
                            f"[MARKET_HOURS] {symbol}: schedule says closed "
                            f"but recent price data found — proceeding"
                        )

                    # ✅ CHECK: Don't analyze if already have position in this symbol
                    symbol_has_position = any(
                        pos['symbol'] == symbol 
                        for pos in self.open_positions.values()
                    )
                    
                    if symbol_has_position:
                        logger.debug(f"Skipping {symbol} - already have open position")
                        continue
                    
                    platform = symbol_config['platform']
                    timeframes = symbol_config['timeframes']
                    
                    # Fetch multi-timeframe data
                    multi_tf_data = await self.market_client.fetch_multiple_timeframes(
                        symbol,
                        platform,
                        timeframes
                    )
                    
                    if not multi_tf_data:
                        logger.warning(f"No data fetched for {symbol}")
                        continue
                    
                    # Run strategy analysis
                    try:
                        analysis = self.strategy_engine.analyze_market(symbol, multi_tf_data)
                        
                        # Log analysis
                        try:
                            analysis_id = self.audit_logger.log_analysis(analysis)
                            analysis['analysis_id'] = analysis_id
                        except Exception as e:
                            logger.error(f"Error logging analysis: {e}", exc_info=True)
                            analysis['analysis_id'] = None
                        
                        # ✅ FINAL CHECK before processing entry
                        if analysis['entry_signal']:
                            current_count = len(self.open_positions)
                            if current_count >= max_concurrent:
                                logger.warning(
                                    f"Signal detected but max positions reached "
                                    f"({current_count}/{max_concurrent}), skipping"
                                )
                                continue
                            
                            await self._process_entry_signal(
                                symbol,
                                symbol_config,
                                analysis,
                                multi_tf_data
                            )
                        
                    except Exception as e:
                        logger.error(f"Error in strategy analysis: {e}", exc_info=True)
                        continue
                
                # Wait before next iteration
                await asyncio.sleep(303)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)
                
    async def _process_entry_signal(self, symbol: str, symbol_config: dict,
                                analysis: dict, multi_tf_data: dict):
        """Process entry signal - FINAL VALIDATION BEFORE ORDER."""
        try:
            analysis_id = analysis.get('analysis_id', 'unknown')
            platform = symbol_config['platform']
            
            # ✅ ABSOLUTE FINAL CHECK (defensive programming)
            max_concurrent = self.config.get('risk_management', {}).get(
                'global_limits', {}
            ).get('max_concurrent_trades', 3)
            
            current_count = len(self.open_positions)
            
            if current_count >= max_concurrent:
                logger.error(
                    f"CRITICAL: Attempted to place order with {current_count} positions "
                    f"(max: {max_concurrent}). This should never happen!"
                )
                return
            
            # Get current price from broker
            if platform == 'mt5':
                price_data = await self.mt5_client.get_current_price(symbol.replace('/', ''))
                if not price_data:
                    logger.error("Could not get current price from MT5")
                    return
                
                if analysis['direction'] == 'long':
                    current_price = price_data['ask']
                else:
                    current_price = price_data['bid']
                
                logger.info(f"Current price from MT5: {current_price:.4f}")
            
            # Calculate stops with current price
            entry_tf = symbol_config['timeframes'][-1]
            df = multi_tf_data[entry_tf]
            
            from indicators.indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            atr_result = indicators.calculate_atr(df)
            atr = atr_result['current']
            
            atr_multiplier = self.config.get('risk_management', {}).get(
                'stop_loss', {}
            ).get('atr_multiplier', 2.0)
            
            # ── TP levels read from config, not hardcoded ──────────────────
            _tp_targets = self.config.get('risk_management', {}).get(
                'take_profit', {}
            ).get('targets', [])
            # Filter out the trailing-only entry (rr_ratio: 999)
            _real_tps = [t for t in _tp_targets if float(t.get('rr_ratio', 999)) < 999]
            # Fall back to 2R / 3R if config section missing
            _tp1_rr       = float(_real_tps[0]['rr_ratio']) if len(_real_tps) >= 1 else 2.0
            _tp2_rr       = float(_real_tps[1]['rr_ratio']) if len(_real_tps) >= 2 else 3.0
            _tp1_fraction = float(_real_tps[0].get('close_percent', 50)) / 100.0 if _real_tps else 0.5
                            

            if analysis['direction'] == 'long':
                stop_loss    = current_price - (atr * atr_multiplier)
                risk         = current_price - stop_loss
                take_profit_1 = current_price + (risk * _tp1_rr)
                take_profit_2 = current_price + (risk * _tp2_rr)
            else:
                stop_loss    = current_price + (atr * atr_multiplier)
                risk         = stop_loss - current_price
                take_profit_1 = current_price - (risk * _tp1_rr)
                take_profit_2 = current_price - (risk * _tp2_rr)

            logger.info(
                f"[TP] {symbol} {analysis['direction']}: "
                f"SL={stop_loss:.5f}  TP1={take_profit_1:.5f} ({_tp1_rr}R, "
                f"{_tp1_fraction*100:.0f}% close)  TP2={take_profit_2:.5f} ({_tp2_rr}R)"
            )
            
            # Get balance
            balance = await self._get_total_balance()
            
            # ✅ Validate trade with current exposure
            sizing = self.money_manager.validate_trade(
                account_equity=balance,
                entry_price=current_price,
                stop_loss=stop_loss,
                symbol=symbol,
                direction=analysis['direction'],
                platform=platform,
                current_exposure=self._get_current_exposure(),  # Gets actual count
                daily_stats=self.daily_stats,
                recent_trades=self._get_recent_trades()
            )
            
            if not sizing['approved']:
                logger.info(f"Trade rejected: {sizing.get('reason')}")
                return
            
            # Place order
            if platform == 'mt5':
                result = await self.mt5_client.place_order(
                    symbol=symbol.replace('/', ''),
                    direction=analysis['direction'],
                    volume=sizing['position_size'],
                    order_type='market',
                    price=None,
                    stop_loss=stop_loss,
                    take_profit=take_profit_2,
                    comment=f"Analysis_{analysis_id[:8]}" if analysis_id != 'unknown' else "Python"
                )
            
            
            if result['success']:
                executed_price = result.get('filled_price') or result.get('price') or current_price
                ticket = result.get('ticket') or result.get('order_id')

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
                })

                self.db.update_trade(trade_id, {'ticket': ticket, 'status': 'open'})

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
                    tp1_fraction= _tp1_fraction,   
                    platform    = platform,
                    analysis_id = analysis_id,
                )

                self.daily_stats['trades_today'] += 1
                logger.info(
                    f"Trade #{len(self.open_positions)}: {symbol} {analysis['direction']} "
                    f"@ {executed_price:.4f}, Ticket: {ticket}"
                )
            else:
                logger.error(f" Order failed: {result.get('error')}")
        
        except Exception as e:
            logger.error(f"Error processing entry signal: {e}", exc_info=True)
            
    async def _position_monitor_loop(self):
        """Monitor open positions - BATCHED VERSION."""
        logger.info("Position monitor loop started (batched)")
        
        while self.running:
            try:
                if len(self.open_positions) == 0:
                    # No positions to monitor, wait longer
                    await asyncio.sleep(30)
                    continue
                
                # Group positions by platform for batch checking
                mt5_positions = {}
                
                for trade_id, position in list(self.open_positions.items()):
                    if position['platform'] == 'mt5':
                        mt5_positions[trade_id] = position
                
                # Batch check MT5 positions (single API call)
                if mt5_positions:
                    await self._batch_update_mt5_positions(mt5_positions)
                
                # Wait before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def _batch_update_mt5_positions(self, positions: dict):
        """
        Update all MT5 positions in one bridge call.
        Relies on the updated MQ5 bridge that now returns:
        ticket, symbol, volume, price (entry), current_price (live bid/ask),
        sl, tp, type (0=BUY/1=SELL), profit
        """
        try:
            all_mt5_positions = await self.mt5_client.get_all_positions()

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
                        favorable
                    )
                    # MAE: worst the trade has looked (highest adverse move, stored positive)
                    position['max_adverse_excursion'] = max(
                        position.get('max_adverse_excursion', 0.0),
                        adverse
                    )

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
        if (_time.time() - last_update) < min_update_interval:
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
                    await self._send_sl_modify(
                        trade_id, position, breakeven_sl, current_tp, label='breakeven'
                    )
            else:
                breakeven_sl = entry_price - buffer_pts
                if current_sl > breakeven_sl:
                    await self._send_sl_modify(
                        trade_id, position, breakeven_sl, current_tp, label='breakeven'
                    )

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

        await self._send_sl_modify(trade_id, position, new_sl, current_tp, label='trail')


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
        position['last_sl_update'] = _time.time()

        # Persist SL change to database
        try:
            self.db.update_trade(trade_id, {'stop_loss': new_sl})
        except Exception as exc:
            logger.warning(f"[MODIFY] DB update failed for trade_id={trade_id}: {exc}")

        logger.info(
            f"[{label.upper()}] trade_id={trade_id} ticket={ticket} ({symbol}) "
            f"SL moved {old_sl:.5f} → {new_sl:.5f}"
        )
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
                'exit_time':    datetime.utcnow(),
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

        - Adds any MT5 positions not yet tracked (e.g. manual trades or restarts).
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

        # ── Update surviving positions ───────────────────────────────────────
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
                })
                # Keep our internal stop_loss in sync with MT5 unless we have a newer value
                if "stop_loss" not in self.open_positions[ticket]:
                    self.open_positions[ticket]["stop_loss"] = mt5_pos["sl"]
            else:
                # Unknown position — register it so we can manage it going forward
                logger.info(
                    f"[SYNC] Discovered untracked position: ticket {ticket} "
                    f"({mt5_pos['symbol']}). Registering."
                )
                self.open_positions[ticket] = {
                    "ticket"         : ticket,
                    "symbol"         : mt5_pos["symbol"],
                    "volume"         : mt5_pos["volume"],
                    "price"          : mt5_pos["price"],        # entry price
                    "entry_price"    : mt5_pos["price"],
                    "current_price"  : mt5_pos["current_price"],
                    "sl"             : mt5_pos["sl"],
                    "stop_loss"      : mt5_pos["sl"],
                    "tp"             : mt5_pos["tp"],
                    "type"           : mt5_pos["type"],
                    "profit"         : mt5_pos["profit"],
                    "trailing_active": False,
                    "tp1_hit"        : False,
                    "tp2_hit"        : False,
                    "last_sl_update" : 0.0,
                    "source"         : "sync_discovered",
                }

    async def _handle_external_close(self, trade_id: str, position: dict):
        """
        Handle a position that was closed outside our system (SL/TP hit by broker,
        manual close on MT5 terminal, margin call, etc.).

        Fetches the actual closing deal from MT5 history so the database record
        contains real exit_price, P&L, duration, commission, slippage, MFE/MAE
        and exit_reason instead of zeros.
        """
        ticket      = position.get('ticket')
        symbol      = position.get('symbol', '')
        entry_price = position.get('entry_price', 0.0)
        direction   = position.get('direction', 'long')
        sl          = position.get('stop_loss', 0.0)
        tp1         = position.get('take_profit_1', 0.0)
        entry_time  = position.get('entry_time')      # datetime stored at registration

        logger.info(f"Recording external close for {trade_id} (ticket={ticket})")

        # ── Fetch real deal data from MT5 history ────────────────────────────
        exit_price   = 0.0
        net_pnl      = 0.0
        gross_profit = 0.0
        swap         = 0.0
        commission   = 0.0
        exit_reason  = 'external_close'
        volume       = position.get('volume', position.get('position_size', 0.0))
        close_time   = None     # will be set from deal if available

        if ticket:
            try:
                lookback_hours = self.config.get('monitor', {}).get(
                    'history_lookback_hours', 48
                )
                deal = await self.mt5_client.get_deal_history(
                    ticket        = int(ticket),
                    lookback_hours= lookback_hours,
                )

                if deal.get('status') == 'success':
                    exit_price   = deal.get('exit_price',  0.0)
                    gross_profit = deal.get('profit',      0.0)
                    swap         = deal.get('swap',        0.0)
                    commission   = deal.get('commission',  0.0)
                    net_pnl      = deal.get('net_profit',  gross_profit + swap + commission)
                    exit_reason  = deal.get('exit_reason', 'external_close')
                    volume       = deal.get('volume',      volume)
                    close_time   = deal.get('close_time',  None)   # unix epoch int

                    if entry_price == 0.0:
                        entry_price = deal.get('entry_price', 0.0)

                    logger.info(
                        f"[CLOSE] ticket={ticket} exit={exit_price:.5f} "
                        f"net_pnl={net_pnl:.2f} commission={commission:.2f} "
                        f"reason={exit_reason}"
                    )
                else:
                    logger.warning(
                        f"[CLOSE] get_deal_history failed for ticket={ticket}: "
                        f"{deal.get('error')} — writing with available data"
                    )

            except Exception as e:
                logger.error(
                    f"[CLOSE] Exception fetching deal history for ticket={ticket}: {e}"
                )

        # ── Duration ─────────────────────────────────────────────────────────
        duration_minutes = None
        if entry_time:
            # entry_time is a datetime; close_time from deal is a unix epoch int
            if close_time and close_time > 0:
                exit_dt = datetime.utcfromtimestamp(close_time)
            else:
                exit_dt = datetime.utcnow()

            if isinstance(entry_time, datetime):
                delta = exit_dt - entry_time
                duration_minutes = round(delta.total_seconds() / 60, 1)

        # ── Slippage (exit): difference between the reference price and actual fill ──
        # For SL hits: reference = stop_loss level
        # For TP hits: reference = take_profit_1 level
        # For everything else: 0
        slippage = 0.0
        if exit_price > 0:
            if exit_reason == 'stop_loss' and sl:
                slippage = round(abs(exit_price - sl), 5)
            elif exit_reason == 'take_profit' and tp1:
                slippage = round(abs(exit_price - tp1), 5)

        # ── Realised R:R — use ORIGINAL SL, not the current (post-breakeven) SL ──
        original_sl  = position.get('original_stop_loss') or sl
        initial_risk = abs(entry_price - original_sl) if original_sl and entry_price else 0.0
        realized_rr  = 0.0
        if initial_risk > 0 and exit_price > 0 and entry_price > 0:
            price_move  = (exit_price - entry_price) if direction == 'long' \
                          else (entry_price - exit_price)
            realized_rr = price_move / initial_risk

        # ── P&L as % of current live equity (not hardcoded config value) ─────
        equity      = self.current_equity if self.current_equity > 0 else 10_000.0
        pnl_percent = round((net_pnl / equity * 100), 4) if equity else 0.0

        # ── Update daily drawdown stats ──────────────────────────────────────
        if net_pnl < 0 and self.daily_stats.get('starting_balance', 0) > 0:
            self.daily_stats['daily_drawdown_percent'] = (
                self.daily_stats.get('daily_drawdown_percent', 0.0)
                + abs(pnl_percent)
            )

        # ── Consecutive loss tracking ────────────────────────────────────────
        if net_pnl < 0:
            self.consecutive_losses = getattr(self, 'consecutive_losses', 0) + 1
            self.last_loss_time     = _time.time()
            logger.info(
                f"[CLOSE] Loss — consecutive_losses={self.consecutive_losses}"
            )
        else:
            self.consecutive_losses = 0

        # ── Write everything to audit log + database ──────────────────────────
        # audit_logger.log_trade_exit now passes all fields through to db.update_trade
        self.audit_logger.log_trade_exit(trade_id, {
            'exit_price'             : exit_price,
            'reason'                 : exit_reason,
            'pnl'                    : round(net_pnl, 2),
            'pnl_percent'            : pnl_percent,
            'realized_rr'            : round(realized_rr, 4),
            'duration_minutes'       : duration_minutes,
            'commission'             : round(commission, 2),
            'slippage'               : slippage,
            'max_favorable_excursion': position.get('max_favorable_excursion'),
            'max_adverse_excursion'  : position.get('max_adverse_excursion'),
        })

        # ── Notification ─────────────────────────────────────────────────────
        if hasattr(self, 'notifier'):
            emoji = '✅' if net_pnl >= 0 else '❌'
            dur_str = f"{duration_minutes:.0f}m" if duration_minutes else "?"
            self.notifier.send(
                f"{emoji} {symbol} closed ({exit_reason})\n"
                f"Exit: {exit_price:.5f} | Net P&L: {net_pnl:.2f} "
                f"| RR: {realized_rr:.2f} | Duration: {dur_str}"
            )

        # ── Remove from open positions ────────────────────────────────────────
        if trade_id in self.open_positions:
            del self.open_positions[trade_id]

                
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

    def _check_daily_drawdown(self) -> bool:
        """
        Return True if trading is allowed, False if daily drawdown limit is hit.
        Reads today's realised P&L from SQLite via the audit logger.
        """
        max_dd_pct = self.config.get("risk", {}).get("daily_max_drawdown_percent", 5.0)

        try:
            account_info = self.mt5_client.get_account_info()
            balance      = account_info.get("balance", 0)
            equity       = account_info.get("equity", balance)
        except Exception as exc:
            logger.error(f"[DRAWDOWN] Cannot fetch account info: {exc}")
            return True   # fail-open: don't block trading on connection error

        if balance <= 0:
            return True

        daily_pnl  = self.audit_logger.get_daily_realised_pnl()     # returns float
        dd_pct     = (-daily_pnl / balance) * 100 if daily_pnl < 0 else 0.0

        if dd_pct >= max_dd_pct:
            logger.critical(
                f"[DRAWDOWN] Daily drawdown {dd_pct:.2f}% ≥ limit {max_dd_pct}%. "
                f"Halting new entries."
            )
            self.audit_logger.log_event(
                event_type   = "daily_drawdown_limit_hit",
                drawdown_pct = dd_pct,
                limit_pct    = max_dd_pct,
                balance      = balance,
            )
            # Optionally fire a notification
            if hasattr(self, "notifier"):
                self.notifier.send(
                    f"⚠️ Daily drawdown limit hit: {dd_pct:.2f}% — trading paused."
                )
            return False

        return True

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

        elapsed = _time.time() - self.last_loss_time
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
        Immediately halt all new activity, close every open position, and
        set the kill-switch flag so the main loop exits cleanly.
        """
        logger.critical(f"[SHUTDOWN] Emergency shutdown triggered: {reason}")
        self.kill_switch = True   # checked by run() loop

        self.audit_logger.log_event(
            event_type = "emergency_shutdown",
            reason     = reason,
        )

        if hasattr(self, "notifier"):
            self.notifier.send(f"🚨 Emergency shutdown: {reason}. Closing all positions.")

        # Close all MT5 positions
        for ticket, position in list(self.open_positions.items()):
            try:
                result = self.mt5_client.close_position(ticket=ticket)
                if result.get("status") == "success":
                    logger.info(f"[SHUTDOWN] Closed MT5 position {ticket}.")
                    self.audit_logger.log_event(
                        event_type = "position_closed",
                        ticket     = ticket,
                        symbol     = position.get("symbol"),
                        reason     = "emergency_shutdown",
                    )
                else:
                    logger.error(
                        f"[SHUTDOWN] Failed to close {ticket}: {result.get('error')}"
                    )
            except Exception as exc:
                logger.error(f"[SHUTDOWN] Exception closing {ticket}: {exc}")

        self.open_positions.clear()
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
            'entry_time'     : datetime.utcnow(),
            'open_time'      : _time.time(),
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
            self.last_loss_time = _time.time()
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
            self.notifier.send(
                f"{emoji} Position closed: {position.get('symbol')} "
                f"| P&L: {profit:.2f} | Reason: {reason}"
            )
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
                # result = self.learner.run_learning_cycle(...)
                
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
                self.daily_stats['starting_balance'] = await self._get_total_balance()
                
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
        """Shutdown trading system gracefully."""
        if not self.running:
            return  # Already shut down
            
        logger.info("=" * 60)
        logger.info("Shutting down Trading System")
        logger.info("=" * 60)
        
        self.running = False
        
        # Give loops time to finish current iteration
        await asyncio.sleep(1)
        
        # Close open positions if emergency shutdown enabled
        if self.config.get('risk_management', {}).get('global_limits', {}).get('emergency_shutdown', {}).get('auto_close_all', False):
            logger.warning("Emergency shutdown - closing all positions")
            for position in self.open_positions.values():
                try:
                    if position['platform'] == 'mt5':
                        await self.mt5_client.close_position(position['ticket'])
                except Exception as e:
                    logger.error(f"Error closing position: {e}")
                    
        # Close connections
        try:
            await self.mt5_client.disconnect()
        except:
            pass
            
        try:
            await self.market_client.close_all()
        except:
            pass
        
        # Close database
        try:
            self.db.disconnect()
        except:
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