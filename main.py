"""
Main orchestrator for the trading system.
Coordinates all components and manages the trading loop.
"""

import asyncio
import logging
import signal
import sys
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
from execution.binance_api import BinanceAPI
from learning.learner import StrategyLearner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
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
        binance_client (BinanceAPI): Binance exchange execution and data client
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
            Initialize multi-market data clients (Binance, MT5).
        _init_strategy_components() -> None:
            Initialize technical indicators, strategy engine, money manager, and stop manager.
        _init_execution_clients() -> None:
            Initialize order execution clients for both MT5 and Binance platforms.
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
        _batch_update_binance_positions(positions: dict) -> Coroutine:
            Update all Binance positions in a single API call.
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
            'binance': {
                'api_key': os.getenv('BINANCE_API_KEY'),
                'api_secret': os.getenv('BINANCE_API_SECRET'),
                'mode': os.getenv('BINANCE_MODE', 'testnet'),
                'use_futures': os.getenv('BINANCE_USE_FUTURES', 'false').lower() == 'true'
            },
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
        demo_mode = self.config['general']['mode'] == 'demo'
        
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
        
        # Binance
        binance_config = {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'api_secret': os.getenv('BINANCE_API_SECRET'),
            'use_futures': os.getenv('BINANCE_USE_FUTURES', 'false').lower() == 'true'
        }
        self.binance_client = BinanceAPI(binance_config, demo_mode=demo_mode)
        
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

    async def _get_total_balance(self) -> float:
        """Get total account balance across all platforms."""
        total_equity = 0.0
        
        try:
            # Get MT5 balance
            if self.mt5_client and self.mt5_client.is_connected():
                mt5_balance = await self.mt5_client.get_balance()
                if mt5_balance.get('success'):
                    total_equity += mt5_balance.get('equity', 0.0)
                    logger.info(f"MT5 Equity: ${mt5_balance.get('equity', 0):.2f}")
            
            # Get Binance balance
            if self.binance_client and self.binance_client.is_connected():
                try:
                    binance_balance = await self.binance_client.get_balance()
                    # Binance returns different format
                    total_equity += binance_balance.get('total_usd', 0.0)
                    logger.info(f"Binance Equity: ${binance_balance.get('total_usd', 0):.2f}")
                except Exception as e:
                    logger.warning(f"Could not get Binance balance: {e}")
            
            # Update equity variable
            self.current_equity = total_equity
            
            logger.info(f"Total Equity: ${total_equity:.2f}")
            return total_equity
            
        except Exception as e:
            logger.error(f"Error getting total balance: {e}")
            # Return last known equity or default
            return self.current_equity if self.current_equity > 0 else 10000.0

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
                elif platform == 'binance':
                    result = await self.binance_client.close_position(symbol)
                    logger.info(f"Emergency closed Binance position {symbol}: {result}")
                
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
        await self.binance_client.connect()
        
        # Get starting balance
        self.daily_stats['starting_balance'] = await self._get_total_balance()

        await self._load_open_positions_from_db()
        
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
        """Load open positions from database on startup."""
        try:
            open_trades = self.db.get_open_trades()
            
            logger.info(f"Loading {len(open_trades)} open positions from database")
            
            for trade in open_trades:
                trade_id = trade['trade_id']
                
                # Verify position still exists on broker
                platform = trade['platform']
                ticket = trade.get('ticket')
                
                if platform == 'mt5' and ticket:
                    pos_info = await self.mt5_client.get_position_info(ticket)
                    if not pos_info:
                        logger.warning(f"Position {trade_id} not found on MT5, marking closed")
                        self.db.update_trade(trade_id, {
                            'status': 'closed',
                            'exit_reason': 'not_found_on_broker',
                            'exit_time': datetime.utcnow()
                        })
                        continue
                
                # Add to tracking
                self.open_positions[trade_id] = {
                    'trade_id': trade_id,
                    'symbol': trade['symbol'],
                    'platform': trade['platform'],
                    'direction': trade['direction'],
                    'entry_price': trade['entry_price'],
                    'stop_loss': trade['stop_loss'],
                    'take_profit_1': trade.get('take_profit_1'),
                    'take_profit_2': trade.get('take_profit_2'),
                    'position_size': trade['position_size'],
                    'ticket': ticket,
                    'entry_time': trade['entry_time'],
                    'analysis_id': trade.get('analysis_id')
                }
            
            logger.info(f" Loaded {len(self.open_positions)} active positions")
            
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
                    await asyncio.sleep(960)
                    continue
                
                # Get enabled symbols
                symbols_config = self.config.get('symbols', {})
                enabled_symbols = [
                    (symbol, cfg)
                    for symbol, cfg in symbols_config.items()
                    if cfg.get('enabled', False)
                ]
                
                for symbol, symbol_config in enabled_symbols:
                    # ✅ DOUBLE CHECK before each symbol
                    if len(self.open_positions) >= max_concurrent:
                        logger.info(
                            f"Reached max positions ({max_concurrent}), "
                            f"stopping new entries this cycle"
                        )
                        break
                    
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
                await asyncio.sleep(960)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(960)
                
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
            else:
                ticker = await self.binance_client.get_ticker(symbol)
                current_price = ticker['last']
                logger.info(f"Current price from Binance: {current_price:.4f}")
            
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
            
            if analysis['direction'] == 'long':
                stop_loss = current_price - (atr * atr_multiplier)
                risk = current_price - stop_loss
                take_profit_1 = current_price + (risk * 1.5)
                take_profit_2 = current_price + (risk * 3.0)
            else:
                stop_loss = current_price + (atr * atr_multiplier)
                risk = stop_loss - current_price
                take_profit_1 = current_price - (risk * 1.5)
                take_profit_2 = current_price - (risk * 3.0)
            
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
                    take_profit=take_profit_1,
                    comment=f"Analysis_{analysis_id[:8]}" if analysis_id != 'unknown' else "Python"
                )
            else:
                result = await self.binance_client.place_order(
                    symbol=symbol,
                    direction=analysis['direction'],
                    amount=sizing['position_size'],
                    order_type='market',
                    stop_loss=stop_loss,
                    take_profit=take_profit_1
                )
            
            if result['success']:
                executed_price = result.get('filled_price') or result.get('price') or current_price
                ticket = result.get('ticket') or result.get('order_id')
                
                trade_data = {
                    'analysis_id': analysis_id,
                    'symbol': symbol,
                    'platform': platform,
                    'direction': analysis['direction'],
                    'entry_price': executed_price,
                    'stop_loss': stop_loss,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'position_size': sizing['position_size']
                }
                
                # Log to database
                trade_id = self.audit_logger.log_trade_entry(trade_data)
                
                # Update with ticket
                self.db.update_trade(trade_id, {
                    # 'ticket': ticket,
                    'status': 'open'
                })
                
                # Add to tracking
                self.open_positions[trade_id] = {
                    **trade_data,
                    'trade_id': trade_id,
                    'ticket': ticket,
                    'entry_time': datetime.utcnow()
                }
                
                self.daily_stats['trades_today'] += 1
                
                logger.info(
                    f" Trade #{len(self.open_positions)}: {symbol} {analysis['direction']} "
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
                binance_positions = {}
                
                for trade_id, position in list(self.open_positions.items()):
                    if position['platform'] == 'mt5':
                        mt5_positions[trade_id] = position
                    else:
                        binance_positions[trade_id] = position
                
                # Batch check MT5 positions (single API call)
                if mt5_positions:
                    await self._batch_update_mt5_positions(mt5_positions)
                
                # Batch check Binance positions (single API call)
                if binance_positions:
                    await self._batch_update_binance_positions(binance_positions)
                
                # Wait before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def _batch_update_mt5_positions(self, positions: dict):
        """Update all MT5 positions in one call."""
        try:
            # Get ALL positions from MT5 in single call
            all_mt5_positions = await self.mt5_client.get_all_positions()
            
            # Create lookup by ticket
            mt5_by_ticket = {
                pos['ticket']: pos 
                for pos in all_mt5_positions
            }
            
            # Update each tracked position
            for trade_id, position in list(positions.items()):
                ticket = position.get('ticket')
                
                # Check if position still exists
                if ticket not in mt5_by_ticket:
                    # Position was closed externally
                    logger.warning(f"Position {trade_id} closed externally")
                    await self._handle_external_close(trade_id, position)
                    continue
                
                # Get current position data
                current = mt5_by_ticket[ticket]
                current_price = current.get('price', 0)
                
                # Check for stop loss hit (simplified check)
                # In production, MT5 handles SL/TP automatically
                # This is just for our tracking
                
                # Update trailing stop if needed
                await self._update_trailing_stop_if_needed(
                    trade_id, 
                    position, 
                    current_price
                )
                
        except Exception as e:
            logger.error(f"Error batch updating MT5 positions: {e}", exc_info=True)


    async def _batch_update_binance_positions(self, positions: dict):
        """Update all Binance positions in one call."""
        try:
            # Get all positions from Binance
            all_binance_positions = await self.binance_client.get_all_positions()
            
            # Create lookup by symbol
            binance_by_symbol = {
                pos['symbol']: pos
                for pos in all_binance_positions
            }
            
            for trade_id, position in list(positions.items()):
                symbol = position['symbol']
                
                if symbol not in binance_by_symbol:
                    logger.warning(f"Position {trade_id} not found on Binance")
                    await self._handle_external_close(trade_id, position)
                    continue
                
                current = binance_by_symbol[symbol]
                # Update trailing stops etc.
                
        except Exception as e:
            logger.error(f"Error batch updating Binance positions: {e}", exc_info=True)


    def _update_trailing_stop_if_needed(self, position: dict) -> None:
        """
        Check whether the trailing stop should be moved for a live position and,
        if so, compute the new SL with StopManager and send the modify call to MT5.

        Expects position keys (now provided by the updated MQ5 bridge):
            ticket        – MT5 position ticket (int)
            symbol        – e.g. "XAUUSD"
            type          – 0 = BUY, 1 = SELL
            price         – entry price  (POSITION_PRICE_OPEN)
            current_price – live bid (BUY) or ask (SELL) from MT5
            sl            – current stop-loss value on MT5
            tp            – current take-profit value on MT5
            stop_loss     – our internal SL record (kept in sync after each modify)
            entry_price   – copied from price on registration
            trailing_active – bool, set True once activation threshold is crossed
            last_sl_update  – timestamp of last SL modify (float, UTC epoch)
        """
        cfg_trail  = self.config.get("risk", {}).get("trailing_stop", {})
        trail_mode = cfg_trail.get("mode", "atr")          # "atr" | "percent"
        rr_activate= cfg_trail.get("activation_rr", 1.0)   # activate at 1:1 by default
        min_update_interval = cfg_trail.get("min_update_interval_seconds", 30)

        ticket        = position["ticket"]
        symbol        = position["symbol"]
        direction     = position["type"]          # 0=BUY, 1=SELL
        entry_price   = position.get("entry_price", position["price"])
        current_price = position["current_price"]
        current_sl    = position.get("stop_loss") or position["sl"]
        current_tp    = position["tp"]

        if current_sl == 0:
            self.logger.warning(f"[TRAIL] Ticket {ticket} has SL=0, skipping trail check.")
            return

        # ── Rate-limit: don't hammer MT5 with modify calls ──────────────────
        last_update = position.get("last_sl_update", 0.0)
        if (time.time() - last_update) < min_update_interval:
            return

        # ── Determine initial risk distance ─────────────────────────────────
        initial_risk = abs(entry_price - current_sl)
        if initial_risk == 0:
            return

        # ── Check trailing activation threshold (RR-based) ──────────────────
        if direction == 0:  # BUY
            price_move = current_price - entry_price
        else:               # SELL
            price_move = entry_price - current_price

        achieved_rr = price_move / initial_risk if initial_risk > 0 else 0.0

        if not position.get("trailing_active", False):
            if achieved_rr < rr_activate:
                return  # haven't reached activation threshold yet
            # ── Break-even + buffer on first activation ──────────────────────
            be_buffer = cfg_trail.get("breakeven_buffer_pips", 2) * self._pip_size(symbol)
            if direction == 0:
                breakeven_sl = entry_price + be_buffer
                if current_sl < breakeven_sl:
                    self._send_sl_modify(position, breakeven_sl, current_tp,
                                        label="breakeven")
            else:
                breakeven_sl = entry_price - be_buffer
                if current_sl > breakeven_sl:
                    self._send_sl_modify(position, breakeven_sl, current_tp,
                                        label="breakeven")
            position["trailing_active"] = True
            self.logger.info(
                f"[TRAIL] Ticket {ticket} ({symbol}) trailing ACTIVATED at RR={achieved_rr:.2f}"
            )

        # ── Compute new SL via StopManager ──────────────────────────────────
        try:
            new_sl = self.stop_manager.compute_trailing_sl(
                symbol        = symbol,
                direction     = direction,
                current_price = current_price,
                current_sl    = current_sl,
                mode          = trail_mode,
                config        = cfg_trail,
            )
        except Exception as exc:
            self.logger.error(f"[TRAIL] StopManager error for {ticket}: {exc}")
            return

        if new_sl is None:
            return

        # ── Only move SL in the favourable direction (never widen it) ───────
        if direction == 0 and new_sl <= current_sl:
            return
        if direction == 1 and new_sl >= current_sl:
            return

        self._send_sl_modify(position, new_sl, current_tp, label="trail")


    def _send_sl_modify(
        self,
        position: dict,
        new_sl: float,
        current_tp: float,
        label: str = "modify",
    ) -> bool:
        """
        Send a modify_position request to MT5 and update the local position record.
        Returns True on success.
        """
        ticket = position["ticket"]
        symbol = position["symbol"]

        try:
            result = self.mt5_client.modify_position(
                ticket = ticket,
                sl     = round(new_sl, self._price_digits(symbol)),
                tp     = current_tp,
            )
        except Exception as exc:
            self.logger.error(f"[MODIFY] Exception modifying ticket {ticket}: {exc}")
            return False

        if result.get("status") != "success":
            self.logger.warning(
                f"[MODIFY] MT5 rejected modify for {ticket}: {result.get('error')}"
            )
            return False

        old_sl = position.get("stop_loss", position.get("sl"))
        position["stop_loss"]     = new_sl
        position["sl"]            = new_sl          # keep bridge mirror in sync
        position["last_sl_update"] = time.time()

        self.logger.info(
            f"[{label.upper()}] Ticket {ticket} ({symbol}) SL moved "
            f"{old_sl:.5f} → {new_sl:.5f}"
        )
        self.audit_logger.log_event(
            event_type = "sl_adjusted",
            ticket     = ticket,
            symbol     = symbol,
            old_sl     = old_sl,
            new_sl     = new_sl,
            reason     = label,
        )
        return True


    def _pip_size(self, symbol: str) -> float:
        """Return one pip/point size for a symbol (configurable fallback)."""
        pip_map = self.config.get("pip_sizes", {})
        return pip_map.get(symbol, 0.01)   # XAUUSD = 0.01, FX pairs = 0.0001


    def _price_digits(self, symbol: str) -> int:
        """Return decimal precision for a symbol price."""
        digits_map = self.config.get("price_digits", {})
        return digits_map.get(symbol, 5)


    def _handle_partial_close(
        self,
        position: dict,
        close_fraction: float,
        reason: str = "tp1",
    ) -> bool:
        """
        Partially close a position by sending a close for `close_fraction` of its volume.
        Updates internal position volume on success.

        Args:
            position:       live position dict (from self.open_positions)
            close_fraction: e.g. 0.5 to close half the position
            reason:         label for audit log ("tp1", "manual", etc.)
        """
        ticket        = position["ticket"]
        symbol        = position["symbol"]
        current_volume= position["volume"]
        close_volume  = round(current_volume * close_fraction, 2)

        min_lot = self.config.get("execution", {}).get("min_lot_size", 0.01)
        if close_volume < min_lot:
            self.logger.warning(
                f"[PARTIAL] Ticket {ticket}: close volume {close_volume} < min lot {min_lot}, skipping."
            )
            return False

        remaining = round(current_volume - close_volume, 2)

        try:
            result = self.mt5_client.close_position(ticket=ticket, volume=close_volume)
        except Exception as exc:
            self.logger.error(f"[PARTIAL] Exception closing ticket {ticket}: {exc}")
            return False

        if result.get("status") != "success":
            self.logger.warning(
                f"[PARTIAL] MT5 rejected partial close for {ticket}: {result.get('error')}"
            )
            return False

        position["volume"] = remaining
        self.logger.info(
            f"[PARTIAL] Ticket {ticket} ({symbol}): closed {close_volume} lots "
            f"({reason}), {remaining} lots remain."
        )
        self.audit_logger.log_event(
            event_type     = "partial_close",
            ticket         = ticket,
            symbol         = symbol,
            closed_volume  = close_volume,
            remaining_volume = remaining,
            reason         = reason,
        )
        return True

    def _check_and_handle_tp_levels(self, position: dict) -> None:
        """
        Evaluate TP1 partial-close and TP2 trail-remainder logic for a position.

        Expects position fields:
            tp1_price    – first target price (set at registration)
            tp1_hit      – bool flag
            tp1_fraction – fraction to close at TP1 (default 0.5 from config)
            tp2_price    – second target (optional; if absent, trail handles it)
            tp2_hit      – bool flag
            type         – 0=BUY, 1=SELL
            current_price
        """
        direction     = position["type"]
        current_price = position["current_price"]
        ticket        = position["ticket"]

        tp1 = position.get("tp1_price")
        tp2 = position.get("tp2_price")

        # ── TP1 ──────────────────────────────────────────────────────────────
        if tp1 and not position.get("tp1_hit", False):
            tp1_reached = (
                (direction == 0 and current_price >= tp1) or
                (direction == 1 and current_price <= tp1)
            )
            if tp1_reached:
                fraction = position.get(
                    "tp1_fraction",
                    self.config.get("risk", {}).get("tp1_close_fraction", 0.5),
                )
                success = self._handle_partial_close(position, fraction, reason="tp1")
                if success:
                    position["tp1_hit"] = True
                    # After TP1, activate trailing on the remainder immediately
                    position["trailing_active"] = True
                    self.logger.info(
                        f"[TP1] Ticket {ticket} TP1 hit at {current_price:.5f}. "
                        f"Trailing activated on remainder."
                    )

        # ── TP2 ──────────────────────────────────────────────────────────────
        if tp2 and not position.get("tp2_hit", False) and position.get("tp1_hit", False):
            tp2_reached = (
                (direction == 0 and current_price >= tp2) or
                (direction == 1 and current_price <= tp2)
            )
            if tp2_reached:
                # Close the rest of the position at TP2
                success = self._handle_partial_close(position, 1.0, reason="tp2")
                if success:
                    position["tp2_hit"] = True
                    self._close_and_unregister(position, reason="tp2_full_close")

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
            self.logger.error(f"[SYNC] Failed to fetch MT5 positions: {exc}")
            return

        if response.get("status") != "success":
            self.logger.warning(f"[SYNC] get_all_positions error: {response.get('error')}")
            return

        mt5_positions: list[dict] = response.get("positions", [])
        mt5_tickets = {int(p["ticket"]) for p in mt5_positions}

        # ── Remove ghost positions ───────────────────────────────────────────
        ghost_tickets = [t for t in list(self.open_positions) if t not in mt5_tickets]
        for ticket in ghost_tickets:
            pos = self.open_positions.pop(ticket)
            self.logger.warning(
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
                self.logger.info(
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
        """Handle position closed outside our system."""
        try:
            logger.info(f"Recording external close for {trade_id}")

            # Update database
            self.db.update_trade(trade_id, {
            'status': 'closed',
            'exit_time': datetime.utcnow(),
            'exit_reason': 'external_close',
            'exit_price': 0  # Unknown
        })
            
            # Log the close
            self.audit_logger.log_trade_exit(trade_id, {
                'exit_price': 0,  # Unknown
                'reason': 'external_close',
                'pnl': 0,  # Would need to calculate
                'pnl_percent': 0,
                'realized_rr': 0
            })
            
            # Remove from tracking
            if trade_id in self.open_positions:
                del self.open_positions[trade_id]
            
        except Exception as e:
            logger.error(f"Error handling external close: {e}")
                
    async def _update_position(self, trade_id: str, position: dict):
        """Update single position."""
        try:
            symbol = position['symbol']
            platform = position['platform']
            
            # Get current price
            if platform == 'mt5':
                pos_info = await self.mt5_client.get_position_info(position['ticket'])
            else:
                pos_info = await self.binance_client.get_position(symbol)
                
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
            self.logger.error(f"[DRAWDOWN] Cannot fetch account info: {exc}")
            return True   # fail-open: don't block trading on connection error

        if balance <= 0:
            return True

        daily_pnl  = self.audit_logger.get_daily_realised_pnl()     # returns float
        dd_pct     = (-daily_pnl / balance) * 100 if daily_pnl < 0 else 0.0

        if dd_pct >= max_dd_pct:
            self.logger.critical(
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

        elapsed = time.time() - self.last_loss_time
        if elapsed < cooldown_s:
            remaining = int(cooldown_s - elapsed)
            self.logger.info(
                f"[COOLDOWN] {self.consecutive_losses} consecutive losses. "
                f"Cooldown active — {remaining}s remaining."
            )
            return False

        # Cooldown expired — reset counter
        self.logger.info("[COOLDOWN] Cooldown expired, resuming entries.")
        self.consecutive_losses = 0
        return True

    def _emergency_shutdown(self, reason: str = "manual") -> None:
        """
        Immediately halt all new activity, close every open position, and
        set the kill-switch flag so the main loop exits cleanly.
        """
        self.logger.critical(f"[SHUTDOWN] Emergency shutdown triggered: {reason}")
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
                    self.logger.info(f"[SHUTDOWN] Closed MT5 position {ticket}.")
                    self.audit_logger.log_event(
                        event_type = "position_closed",
                        ticket     = ticket,
                        symbol     = position.get("symbol"),
                        reason     = "emergency_shutdown",
                    )
                else:
                    self.logger.error(
                        f"[SHUTDOWN] Failed to close {ticket}: {result.get('error')}"
                    )
            except Exception as exc:
                self.logger.error(f"[SHUTDOWN] Exception closing {ticket}: {exc}")

        # Close all Binance positions if applicable
        if hasattr(self, "binance_client"):
            try:
                self.binance_client.close_all_positions(reason="emergency_shutdown")
            except Exception as exc:
                self.logger.error(f"[SHUTDOWN] Binance close-all failed: {exc}")

        self.open_positions.clear()
        self.logger.critical("[SHUTDOWN] Emergency shutdown complete.")

    def _register_new_position(
        self,
        ticket: int,
        symbol: str,
        direction: int,         # 0=BUY 1=SELL
        entry_price: float,
        volume: float,
        sl: float,
        tp: float,
        tp1_price: float | None = None,
        tp2_price: float | None = None,
        tp1_fraction: float     = 0.5,
        analysis_id: str | None = None,
    ) -> dict:
        """
        Register a freshly filled order in self.open_positions with all
        strategy metadata needed for trailing-stop and TP management.
        """
        position = {
            "ticket"         : ticket,
            "symbol"         : symbol,
            "type"           : direction,
            "price"          : entry_price,
            "entry_price"    : entry_price,
            "current_price"  : entry_price,   # will be updated on first sync
            "volume"         : volume,
            "sl"             : sl,
            "stop_loss"      : sl,
            "tp"             : tp,
            "tp1_price"      : tp1_price,
            "tp2_price"      : tp2_price,
            "tp1_fraction"   : tp1_fraction,
            "tp1_hit"        : False,
            "tp2_hit"        : False,
            "trailing_active": False,
            "last_sl_update" : 0.0,
            "profit"         : 0.0,
            "analysis_id"    : analysis_id,
            "open_time"      : time.time(),
            "source"         : "strategy",
        }
        self.open_positions[ticket] = position

        self.logger.info(
            f"[REGISTER] Ticket {ticket} {symbol} "
            f"{'BUY' if direction == 0 else 'SELL'} "
            f"{volume} lots @ {entry_price:.5f} | SL={sl:.5f} TP={tp:.5f}"
        )
        self.audit_logger.log_event(
            event_type   = "order_placed",
            ticket       = ticket,
            symbol       = symbol,
            direction    = "BUY" if direction == 0 else "SELL",
            entry_price  = entry_price,
            volume       = volume,
            sl           = sl,
            tp           = tp,
            analysis_id  = analysis_id,
        )
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
            self.logger.error(f"[CLOSE] Exception for ticket {ticket}: {exc}")
            return False

        if result.get("status") != "success":
            self.logger.warning(
                f"[CLOSE] MT5 rejected close for {ticket}: {result.get('error')}"
            )
            return False

        profit = position.get("profit", 0.0)
        self.open_positions.pop(ticket, None)

        if profit < 0:
            self.consecutive_losses += 1
            self.last_loss_time = time.time()
        else:
            self.consecutive_losses = 0

        self.logger.info(
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
        """Get total account balance across platforms."""
        total = 10000.0  # Default/demo
        
        try:
            # Would get real balances here
            pass
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            
        return total
        
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
                    else:
                        await self.binance_client.close_position(position['symbol'])
                except Exception as e:
                    logger.error(f"Error closing position: {e}")
                    
        # Close connections
        try:
            await self.mt5_client.disconnect()
        except:
            pass
            
        try:
            await self.binance_client.close()
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