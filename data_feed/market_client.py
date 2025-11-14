"""
Market data client for fetching historical and live data.
Supports both MT5 (via bridge) and Binance APIs with WebSocket streaming.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import ccxt
import websockets
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MarketDataClient(ABC):
    """Abstract base class for market data clients."""
    
    @abstractmethod
    async def fetch_historical(self, symbol: str, timeframe: str, 
                               limit: int = 500) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        pass
        
    @abstractmethod
    async def subscribe_live(self, symbol: str, callback: Callable):
        """Subscribe to live price updates."""
        pass
        
    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status."""
        pass


class BinanceDataClient(MarketDataClient):
    """Binance market data client with REST and WebSocket support."""
    
    def __init__(self, config: Dict):
        """
        Initialize Binance client.
        
        Args:
            config: Configuration dictionary with API credentials
        """
        self.config = config
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.testnet = config.get('mode') == 'testnet'
        self.use_futures = config.get('use_futures', False)
        
        # Initialize CCXT exchange
        exchange_class = ccxt.binance if not self.testnet else ccxt.binance
        self.exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if self.use_futures else 'spot',
                'adjustForTimeDifference': True
            }
        })
        
        if self.testnet:
            self.exchange.set_sandbox_mode(True)
            
        self.ws_connections = {}
        self._connected = False
        
    async def fetch_historical(self, symbol: str, timeframe: str, 
                               limit: int = 500) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '15m', '5m')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert timeframe format: 1H -> 1h, 15m -> 15m, 5m -> 5m
            timeframe = timeframe.lower()

            logger.info(f"Fetching {limit} candles for {symbol} @ {timeframe}")
            
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
            
    async def subscribe_live(self, symbol: str, callback: Callable):
        """
        Subscribe to live price updates via WebSocket.
        
        Args:
            symbol: Trading pair
            callback: Async function to call with new data
        """
        # Convert symbol format: BTC/USDT -> btcusdt
        ws_symbol = symbol.replace('/', '').lower()
        
        stream = f"wss://stream.binance.com:9443/ws/{ws_symbol}@kline_1m"
        
        if self.testnet:
            stream = f"wss://testnet.binance.vision/ws/{ws_symbol}@kline_1m"
            
        logger.info(f"Subscribing to live stream: {symbol}")
        
        try:
            async with websockets.connect(stream) as websocket:
                self.ws_connections[symbol] = websocket
                self._connected = True
                
                async for message in websocket:
                    data = json.loads(message)
                    
                    if 'k' in data:  # Kline data
                        kline = data['k']
                        
                        tick_data = {
                            'symbol': symbol,
                            'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                            'open': float(kline['o']),
                            'high': float(kline['h']),
                            'low': float(kline['l']),
                            'close': float(kline['c']),
                            'volume': float(kline['v']),
                            'is_closed': kline['x']  # Is candle closed
                        }
                        
                        await callback(tick_data)
                        
        except Exception as e:
            logger.error(f"WebSocket error for {symbol}: {e}")
            self._connected = False
            # Attempt reconnection
            await asyncio.sleep(5)
            await self.subscribe_live(symbol, callback)
            
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data."""
        try:
            ticker = await asyncio.to_thread(
                self.exchange.fetch_ticker,
                symbol
            )
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
            
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected
        
    async def close(self):
        """Close all WebSocket connections."""
        for symbol, ws in self.ws_connections.items():
            await ws.close()
            logger.info(f"Closed WebSocket for {symbol}")
        self.ws_connections.clear()
        self._connected = False


class MT5DataClient(MarketDataClient):
    """MT5 market data client via file bridge."""
    
    def __init__(self, config: Dict):
        """
        Initialize MT5 client.
        
        Args:
            config: MT5 configuration
        """
        self.config = config
        # Import here to avoid circular dependency
        from execution.mt5_file_bridge import MT5FileBridge
        self.bridge = MT5FileBridge(config, demo_mode=config.get('mode') == 'demo')
        self._connected = False
        
    async def connect(self):
        """Initialize connection."""
        await self.bridge.connect()
        self._connected = self.bridge.is_connected()
        
    async def fetch_historical(self, symbol: str, timeframe: str, 
                               limit: int = 500) -> pd.DataFrame:
        """
        Fetch historical data via MT5 file bridge.
        
        Args:
            symbol: Symbol (e.g., 'XAUUSD')
            timeframe: Timeframe (e.g., '1H', '15m')
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframe to MT5 format
            tf_map = {
                '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
                '1h': 'H1', '4h': 'H4', '1d': 'D1'
            }
            mt5_timeframe = tf_map.get(timeframe.lower(), 'H1')
            
            # Get historical data via file bridge
            command = {
                'action': 'get_historical',
                'symbol': symbol.replace('/', ''),  # Remove slash for MT5
                'timeframe': mt5_timeframe,
                'count': limit
            }
            
            response = await self.bridge._send_command(command)
            
            if response.get('status') != 'success':
                raise ValueError(f"Failed to fetch data: {response.get('error')}")
                
            bars = response.get('data', [])
            
            if not bars:
                raise ValueError("No data returned")
            
            # Convert to DataFrame
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            df.drop('time', axis=1, inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} from MT5")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching MT5 historical data: {e}")
            raise
            
    async def subscribe_live(self, symbol: str, callback):
        """MT5 file bridge doesn't support live subscriptions yet."""
        logger.warning("Live subscriptions not supported with file bridge")
        pass
        
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected
        
    async def close(self):
        """Close connection."""
        await self.bridge.disconnect()
        self._connected = False


class MultiMarketClient:
    """
    Unified client for multiple market data sources.
    Routes requests to appropriate client based on symbol configuration.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize multi-market client.
        
        Args:
            config: Full system configuration
        """
        self.config = config
        self.clients = {}
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize all configured clients."""
        # Initialize Binance client
        if 'binance' in self.config:
            self.clients['binance'] = BinanceDataClient(self.config['binance'])
            logger.info("Initialized Binance client")
            
        # Initialize MT5 client
        if 'mt5' in self.config:
            self.clients['mt5'] = MT5DataClient(self.config['mt5'])
            logger.info("Initialized MT5 client")
            
    def get_client(self, platform: str) -> MarketDataClient:
        """Get client for specific platform."""
        if platform not in self.clients:
            raise ValueError(f"Unknown platform: {platform}")
        return self.clients[platform]
        
    async def fetch_historical(self, symbol: str, platform: str, 
                               timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetch historical data from appropriate platform.
        
        Args:
            symbol: Trading symbol
            platform: 'binance' or 'mt5'
            timeframe: Timeframe string
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        client = self.get_client(platform)
        return await client.fetch_historical(symbol, timeframe, limit)
        
    async def fetch_multiple_timeframes(self, symbol: str, platform: str,
                                        timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes simultaneously.
        
        Args:
            symbol: Trading symbol
            platform: Platform name
            timeframes: List of timeframes
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        tasks = [
            self.fetch_historical(symbol, platform, tf)
            for tf in timeframes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {tf} for {symbol}: {result}")
                continue
            data[tf] = result
            
        return data
        
    async def subscribe_live(self, symbol: str, platform: str, callback: Callable):
        """Subscribe to live updates."""
        client = self.get_client(platform)
        await client.subscribe_live(symbol, callback)
        
    async def close_all(self):
        """Close all client connections."""
        for platform, client in self.clients.items():
            try:
                await client.close()
                logger.info(f"Closed {platform} client")
            except Exception as e:
                logger.error(f"Error closing {platform} client: {e}")


class DataBuffer:
    """
    Buffer for managing streaming data across multiple timeframes.
    Maintains synchronized OHLCV data for analysis.
    """
    
    def __init__(self, max_bars: int = 1000):
        """
        Initialize data buffer.
        
        Args:
            max_bars: Maximum bars to keep per timeframe
        """
        self.max_bars = max_bars
        self.buffers = {}  # {symbol: {timeframe: DataFrame}}
        self.last_update = {}
        
    def update(self, symbol: str, timeframe: str, bar: Dict):
        """
        Update buffer with new bar data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            bar: Bar data dictionary
        """
        if symbol not in self.buffers:
            self.buffers[symbol] = {}
            
        if timeframe not in self.buffers[symbol]:
            self.buffers[symbol][timeframe] = pd.DataFrame()
            
        df = self.buffers[symbol][timeframe]
        
        # Create new row
        new_row = pd.DataFrame([{
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume']
        }], index=[bar['timestamp']])
        
        # Append or update
        if len(df) == 0 or bar['timestamp'] not in df.index:
            df = pd.concat([df, new_row])
        else:
            df.loc[bar['timestamp']] = new_row.iloc[0]
            
        # Trim to max size
        if len(df) > self.max_bars:
            df = df.tail(self.max_bars)
            
        self.buffers[symbol][timeframe] = df
        self.last_update[f"{symbol}_{timeframe}"] = datetime.utcnow()
        
    def get_data(self, symbol: str, timeframe: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Get data from buffer.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            bars: Number of bars to return
            
        Returns:
            DataFrame or None if not available
        """
        if symbol not in self.buffers or timeframe not in self.buffers[symbol]:
            return None
            
        df = self.buffers[symbol][timeframe]
        return df.tail(bars) if len(df) > 0 else None
        
    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get the most recent bar."""
        df = self.get_data(symbol, timeframe, bars=1)
        if df is None or len(df) == 0:
            return None
            
        return df.iloc[-1].to_dict()
        
    def is_stale(self, symbol: str, timeframe: str, max_age_seconds: int = 300) -> bool:
        """Check if data is stale."""
        key = f"{symbol}_{timeframe}"
        if key not in self.last_update:
            return True
            
        age = (datetime.utcnow() - self.last_update[key]).total_seconds()
        return age > max_age_seconds


# Example usage
if __name__ == "__main__":
    async def test_binance():
        """Test Binance client."""
        config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'mode': 'testnet',
            'use_futures': False
        }
        
        client = BinanceDataClient(config)
        
        # Fetch historical data
        df = await client.fetch_historical('BTC/USDT', '1h', limit=100)
        print(f"Fetched {len(df)} candles")
        print(df.head())
        print(df.tail())
        
        # Test live subscription
        async def handle_tick(data):
            print(f"Live tick: {data['symbol']} @ {data['close']}")
            
        # Would start WebSocket in production
        # await client.subscribe_live('BTC/USDT', handle_tick)
        
    async def test_multi_client():
        """Test multi-market client."""
        config = {
            'binance': {
                'api_key': 'test',
                'api_secret': 'test',
                'mode': 'testnet'
            }
        }
        
        client = MultiMarketClient(config)
        
        # Fetch multiple timeframes
        data = await client.fetch_multiple_timeframes(
            'BTC/USDT',
            'binance',
            ['1h', '15m', '5m']
        )
        
        for tf, df in data.items():
            print(f"\n{tf}: {len(df)} bars")
            print(df.tail(3))
            
        await client.close_all()
        
    # Run tests
    print("Testing Binance client...")
    asyncio.run(test_binance())
    
    print("\n\nTesting multi-market client...")
    asyncio.run(test_multi_client())
    
    print("\nData client tests completed!")