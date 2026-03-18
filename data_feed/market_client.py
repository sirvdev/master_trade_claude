"""
Market data client for fetching historical and live data.
Supports both MT5 (via bridge).
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
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
                               limit: int = 250) -> pd.DataFrame:
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
                'count': limit + 1 # +1 to account for current forming candle
            }
            
            response = await self.bridge._send_command(command)

            # print(response)
            
            if response.get('status') != 'success':
                raise ValueError(f"Failed to fetch data: {response.get('error')}")
                
            bars = response.get('data', [])

            # print(bars)
            
            if not bars:
                raise ValueError("No data returned")
            
            # Convert to DataFrame
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            df.drop('time', axis=1, inplace=True)
            
            # Drop the last (current-forming) candle since we requested count+1
            # This ensures only *closed* candles are used in technical indicator calculations
            if len(df) > 1:
                df = df.iloc[:-1]
            
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
                               timeframe: str, limit: int = 250) -> pd.DataFrame:
        """
        Fetch historical data from appropriate platform.
        
        Args:
            symbol: Trading symbol
            platform: 'mt5'
            timeframe: Timeframe string
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        client = self.get_client(platform)
        return await client.fetch_historical(symbol, timeframe, limit)
        
    """
    Fixed MT5DataClient in data_feed/market_client.py
    Replace the fetch_multiple_timeframes method in MultiMarketClient class
    """

    async def fetch_multiple_timeframes(self, symbol: str, platform: str,
                                    timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.
        
        For MT5 file bridge, fetches sequentially to avoid file conflicts.
        For other platforms, fetches in parallel.
        
        Args:
            symbol: Trading symbol
            platform: Platform name
            timeframes: List of timeframes
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        
        if platform == 'mt5':
            # Fetch sequentially for MT5 to avoid file conflicts
            for tf in timeframes:
                try:
                    df = await self.fetch_historical(symbol, platform, tf)
                    data[tf] = df
                    # Small delay to enure fil\\\
                except Exception as e:
                    logger.error(f"Error fetching {tf} for {symbol}: {e}")
                    continue
        
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
        
    def get_data(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
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
        
    def is_stale(self, symbol: str, timeframe: str, max_age_seconds: int = 960) -> bool:
        """Check if data is stale."""
        key = f"{symbol}_{timeframe}"
        if key not in self.last_update:
            return True
            
        age = (datetime.utcnow() - self.last_update[key]).total_seconds()
        return age > max_age_seconds


