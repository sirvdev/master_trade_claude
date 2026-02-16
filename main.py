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
    """
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
        self.daily_stats = {
            'trades_today': 0,
            'daily_drawdown_percent': 0,
            'starting_balance': 0
        }
        
        logger.info("Trading System initialized successfully")
        
    def _apply_env_overrides(self):
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
                    await asyncio.sleep(300)
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
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(300)
                
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


    async def _update_trailing_stop_if_needed(
        self, 
        trade_id: str, 
        position: dict, 
        current_price: float
    ):
        """Update trailing stop for a single position (only if needed)."""
        try:
            # Only update if position has been running for a while
            # and price has moved favorably
            
            direction = position['direction']
            entry_price = position['entry_price']
            current_sl = position['stop_loss']
            
            # Calculate current R:R
            risk = abs(entry_price - current_sl)
            if direction == 'long':
                current_rr = (current_price - entry_price) / risk if risk > 0 else 0
            else:
                current_rr = (entry_price - current_price) / risk if risk > 0 else 0
            
            # Only trail if RR >= 1.0 (configurable)
            trail_activation = 1.0
            if current_rr < trail_activation:
                return
            
            # Calculate new trailing stop
            # This would use stop_manager logic
            # For now, simplified version
            
            # Only update if significantly moved (reduce API calls)
            # Update at most every 180 seconds per position
            last_update = position.get('last_sl_update', 0)
            if time.time() - last_update < 180:
                return
            
            # # Would call MT5 modify_position here if needed
            # position['last_sl_update'] = time.time()
            
        except Exception as e:
            logger.error(f"Error updating trailing stop for {trade_id}: {e}")


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