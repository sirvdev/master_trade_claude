"""
Main orchestrator for the trading system.
Coordinates all components and manages the trading loop.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
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
            
    async def _trading_loop(self):
        """Main trading loop - analyzes markets and places trades."""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                # Get enabled symbols
                symbols_config = self.config.get('symbols', {})
                enabled_symbols = [
                    (symbol, cfg)
                    for symbol, cfg in symbols_config.items()
                    if cfg.get('enabled', False)
                ]
                
                for symbol, symbol_config in enabled_symbols:
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
                    analysis = self.strategy_engine.analyze_market(symbol, multi_tf_data)
                    
                    # Log analysis - CAPTURE THE ID HERE
                    try:
                        analysis_id = self.audit_logger.log_analysis(analysis)
                        # Add the ID to the analysis dict for later use
                        analysis['analysis_id'] = analysis_id
                    except Exception as e:
                        logger.error(f"Error logging analysis: {e}", exc_info=True)
                        # Continue even if logging fails
                        analysis['analysis_id'] = None
                    
                    # Check for entry signal
                    if analysis['entry_signal']:
                        await self._process_entry_signal(
                            symbol,
                            symbol_config,
                            analysis,
                            multi_tf_data
                        )
                        
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)
                
    async def _process_entry_signal(
        self,
        symbol: str,
        symbol_config: dict,
        analysis: dict,
        multi_tf_data: dict
    ):
        """Process entry signal and place trade."""
        try:
            # Get analysis_id from the analysis dict
            analysis_id = analysis.get('analysis_id', 'unknown')
            
            # Calculate entry levels
            levels = self.strategy_engine.calculate_entry_levels(analysis, multi_tf_data)
            
            # Get current balance
            balance = await self._get_total_balance()
            
            # Calculate position size
            sizing = self.money_manager.validate_trade(
                    account_equity=balance,
                    entry_price=levels['entry_price'],
                    stop_loss=levels['stop_loss'],
                    symbol=symbol,
                    direction=analysis['direction'],
                    platform=symbol_config['platform'],  # <-- ADD THIS LINE
                    current_exposure=self._get_current_exposure(),
                    daily_stats=self.daily_stats,
                    recent_trades=self._get_recent_trades()
                )
            
            if not sizing['approved']:
                logger.info(f"Trade rejected: {sizing.get('reason')}")
                return
                
            # Place order
            platform = symbol_config['platform']
            
            if platform == 'mt5':
                result = await self.mt5_client.place_order(
                    symbol=symbol.replace('/', ''),
                    direction=analysis['direction'],
                    volume=sizing['position_size'],
                    order_type='market',
                    stop_loss=levels['stop_loss'],
                    take_profit=levels.get('take_profit_1'),
                    comment=f"Analysis_{analysis_id[:8]}" if analysis_id != 'unknown' else "Python"
                )
            else:  # binance
                result = await self.binance_client.place_order(
                    symbol=symbol,
                    direction=analysis['direction'],
                    amount=sizing['position_size'],
                    order_type='market',
                    stop_loss=levels['stop_loss'],
                    take_profit=levels.get('take_profit_1')
                )
                
            if result['success']:
                # Log trade
                trade_data = {
                    'analysis_id': analysis_id,
                    'symbol': symbol,
                    'platform': platform,
                    'direction': analysis['direction'],
                    'entry_price': result.get('filled_price') or result.get('price') or levels['entry_price'],
                    'stop_loss': levels['stop_loss'],
                    'take_profit_1': levels.get('take_profit_1'),
                    'take_profit_2': levels.get('take_profit_2'),
                    'position_size': sizing['position_size']
                }
                
                trade_id = self.audit_logger.log_trade_entry(trade_data)
                
                # Store in open positions
                self.open_positions[trade_id] = {
                    **trade_data,
                    'trade_id': trade_id,
                    'ticket': result.get('ticket') or result.get('order_id'),
                    'entry_time': datetime.utcnow()
                }
                
                self.daily_stats['trades_today'] += 1
                
                logger.info(f"Trade opened: {symbol} {analysis['direction']}")
                
            else:
                logger.error(f"Order failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error processing entry signal: {e}", exc_info=True)
            
    async def _position_monitor_loop(self):
        """Monitor open positions and manage exits."""
        logger.info("Position monitor loop started")
        
        while self.running:
            try:
                for trade_id, position in list(self.open_positions.items()):
                    await self._update_position(trade_id, position)
                    
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}", exc_info=True)
                await asyncio.sleep(10)
                
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
            
        return {
            'open_count': len(self.open_positions),
            'symbols': symbols
        }
        
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