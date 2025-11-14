"""
SQLite database manager for trading system.
Handles schema creation, migrations, and CRUD operations.
Postgres-ready design for future migration.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations with migration support."""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.current_version = 1
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._initialize_schema()
        
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def _initialize_schema(self):
        """Create all tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Schema version tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis logs - every market analysis pass
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_logs (
                analysis_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                primary_timeframe TEXT NOT NULL,
                timeframe_snapshots TEXT,  -- JSON: {TF: {ohlc, indicators}}
                market_structure TEXT,  -- JSON: HH/HL/LH/LL detection
                indicators_state TEXT,  -- JSON: all indicator values
                entry_signal BOOLEAN,
                entry_reason TEXT,  -- Code/description of entry logic
                entry_price REAL,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                take_profit_3 REAL,
                position_size REAL,
                expected_rr REAL,
                confidence_score REAL,
                notes TEXT
            )
        """)
        
        # Trade execution logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                analysis_id TEXT,
                symbol TEXT NOT NULL,
                platform TEXT NOT NULL,  -- mt5 or binance
                direction TEXT NOT NULL,  -- long or short
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL NOT NULL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                take_profit_3 REAL,
                position_size REAL NOT NULL,
                status TEXT NOT NULL,  -- open, closed, partial
                exit_reason TEXT,  -- tp1, tp2, tp3, sl, manual, trailing
                pnl REAL,
                pnl_percent REAL,
                realized_rr REAL,
                duration_minutes INTEGER,
                commission REAL,
                slippage REAL,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL,
                FOREIGN KEY (analysis_id) REFERENCES analysis_logs(analysis_id)
            )
        """)
        
        # Order lifecycle events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,  -- placed, filled, partial, modified, cancelled
                order_type TEXT,  -- market, limit, stop
                price REAL,
                quantity REAL,
                api_response TEXT,  -- JSON: raw API response
                notes TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)
        
        # Stop loss/take profit adjustments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sl_tp_adjustments (
                adjustment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                adjustment_type TEXT NOT NULL,  -- sl_moved, tp_moved, trailing_activated
                old_value REAL,
                new_value REAL,
                trigger_reason TEXT,
                current_price REAL,
                current_rr REAL,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)
        
        # Strategy parameter versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameter_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version_name TEXT NOT NULL,
                parameters TEXT NOT NULL,  -- JSON: full parameter dict
                source TEXT,  -- manual, grid_search, rl_bandit
                backtest_metrics TEXT,  -- JSON: performance on backtest
                status TEXT DEFAULT 'pending',  -- pending, active, archived
                notes TEXT
            )
        """)
        
        # Learning metrics and results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                optimization_method TEXT,
                trades_analyzed INTEGER,
                best_parameters TEXT,  -- JSON
                metrics TEXT,  -- JSON: win_rate, expectancy, etc.
                recommended_version_id INTEGER,
                status TEXT,  -- running, completed, failed
                error_message TEXT,
                FOREIGN KEY (recommended_version_id) REFERENCES parameter_versions(version_id)
            )
        """)
        
        # System events and errors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,  -- info, warning, error, critical
                component TEXT,  -- module name
                message TEXT,
                details TEXT,  -- JSON: additional context
                resolved BOOLEAN DEFAULT 0
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_symbol ON analysis_logs(symbol)")
        
        self.conn.commit()
        logger.info("Database schema initialized")
        
    def log_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """
        Log a market analysis pass.
        
        Args:
            analysis_data: Dictionary containing analysis results
            
        Returns:
            analysis_id of the logged entry
        """
        cursor = self.conn.cursor()
        
        
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

        analysis_id = f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        cursor.execute("""
            INSERT INTO analysis_logs (
                analysis_id, timestamp, symbol, primary_timeframe,
                timeframe_snapshots, market_structure, indicators_state,
                entry_signal, entry_reason, entry_price, stop_loss,
                take_profit_1, take_profit_2, take_profit_3,
                position_size, expected_rr, confidence_score, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id,
            datetime.utcnow(),
            analysis_data.get('symbol'),
            analysis_data.get('primary_timeframe'),
            json.dumps(convert_numpy_types(analysis_data.get('timeframe_snapshots', {}))),
            json.dumps(convert_numpy_types(analysis_data.get('market_structure', {}))),
            json.dumps(convert_numpy_types(analysis_data.get('indicators_state', {}))),
            analysis_data.get('entry_signal', False),
            analysis_data.get('entry_reason'),
            analysis_data.get('entry_price'),
            analysis_data.get('stop_loss'),
            analysis_data.get('take_profit_1'),
            analysis_data.get('take_profit_2'),
            analysis_data.get('take_profit_3'),
            analysis_data.get('position_size'),
            analysis_data.get('expected_rr'),
            analysis_data.get('confidence_score'),
            analysis_data.get('notes')
        ))
        
        self.conn.commit()
        return analysis_id
        
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Log a trade execution.
        
        Args:
            trade_data: Dictionary containing trade details
            
        Returns:
            trade_id of the logged trade
        """
        cursor = self.conn.cursor()
        
        trade_id = trade_data.get('trade_id') or f"trade_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        cursor.execute("""
            INSERT INTO trades (
                trade_id, analysis_id, symbol, platform, direction,
                entry_time, entry_price, stop_loss, take_profit_1,
                take_profit_2, take_profit_3, position_size, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            trade_data.get('analysis_id'),
            trade_data.get('symbol'),
            trade_data.get('platform'),
            trade_data.get('direction'),
            datetime.utcnow(),
            trade_data.get('entry_price'),
            trade_data.get('stop_loss'),
            trade_data.get('take_profit_1'),
            trade_data.get('take_profit_2'),
            trade_data.get('take_profit_3'),
            trade_data.get('position_size'),
            'open'
        ))
        
        self.conn.commit()
        return trade_id
        
    def update_trade(self, trade_id: str, updates: Dict[str, Any]):
        """Update trade fields."""
        cursor = self.conn.cursor()
        
        set_clauses = ', '.join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [trade_id]
        
        cursor.execute(f"""
            UPDATE trades SET {set_clauses} WHERE trade_id = ?
        """, values)
        
        self.conn.commit()
        
    def log_order_event(self, event_data: Dict[str, Any]):
        """Log an order lifecycle event."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO order_events (
                trade_id, timestamp, event_type, order_type,
                price, quantity, api_response, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_data.get('trade_id'),
            datetime.utcnow(),
            event_data.get('event_type'),
            event_data.get('order_type'),
            event_data.get('price'),
            event_data.get('quantity'),
            json.dumps(event_data.get('api_response', {})),
            event_data.get('notes')
        ))
        
        self.conn.commit()
        
    def log_sl_tp_adjustment(self, adjustment_data: Dict[str, Any]):
        """Log stop loss or take profit adjustment."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO sl_tp_adjustments (
                trade_id, timestamp, adjustment_type, old_value,
                new_value, trigger_reason, current_price, current_rr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            adjustment_data.get('trade_id'),
            datetime.utcnow(),
            adjustment_data.get('adjustment_type'),
            adjustment_data.get('old_value'),
            adjustment_data.get('new_value'),
            adjustment_data.get('trigger_reason'),
            adjustment_data.get('current_price'),
            adjustment_data.get('current_rr')
        ))
        
        self.conn.commit()
        
    def save_parameter_version(self, version_data: Dict[str, Any]) -> int:
        """Save a new parameter version."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO parameter_versions (
                version_name, parameters, source, backtest_metrics, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            version_data.get('version_name'),
            json.dumps(version_data.get('parameters')),
            version_data.get('source'),
            json.dumps(version_data.get('backtest_metrics', {})),
            version_data.get('status', 'pending'),
            version_data.get('notes')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
        
    def log_learning_run(self, run_data: Dict[str, Any]) -> int:
        """Log a learning engine run."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO learning_runs (
                started_at, optimization_method, trades_analyzed,
                best_parameters, metrics, status
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow(),
            run_data.get('optimization_method'),
            run_data.get('trades_analyzed'),
            json.dumps(run_data.get('best_parameters', {})),
            json.dumps(run_data.get('metrics', {})),
            'running'
        ))
        
        self.conn.commit()
        return cursor.lastrowid
        
    def get_trades(self, filters: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
        """
        Retrieve trades with optional filters.
        
        Args:
            filters: Optional dict with keys like symbol, status, platform
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM trades"
        params = []
        
        if filters:
            where_clauses = []
            for key, value in filters.items():
                where_clauses.append(f"{key} = ?")
                params.append(value)
            query += " WHERE " + " AND ".join(where_clauses)
            
        query += f" ORDER BY entry_time DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    def get_open_trades(self) -> List[Dict]:
        """Get all currently open trades."""
        return self.get_trades(filters={'status': 'open'})
        
    def get_trade_statistics(self, symbol: Optional[str] = None, days: int = 30) -> Dict:
        """
        Calculate trade statistics.
        
        Args:
            symbol: Optional symbol filter
            days: Number of days to include
            
        Returns:
            Dictionary of statistics
        """
        cursor = self.conn.cursor()
        
        where_clause = "WHERE status = 'closed'"
        params = []
        
        if symbol:
            where_clause += " AND symbol = ?"
            params.append(symbol)
            
        where_clause += " AND entry_time >= datetime('now', '-' || ? || ' days')"
        params.append(days)
        
        query = f"""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl,
                AVG(realized_rr) as avg_rr,
                MAX(pnl) as max_win,
                MIN(pnl) as max_loss,
                AVG(duration_minutes) as avg_duration_minutes
            FROM trades
            {where_clause}
        """
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        stats = dict(row) if row else {}
        
        # Calculate win rate
        if stats.get('total_trades', 0) > 0:
            stats['win_rate'] = stats.get('winning_trades', 0) / stats['total_trades']
        else:
            stats['win_rate'] = 0
            
        return stats
        
    def export_to_csv(self, table_name: str, output_path: str):
        """Export a table to CSV."""
        import csv
        
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([desc[0] for desc in cursor.description])
            writer.writerows(cursor.fetchall())
            
        logger.info(f"Exported {table_name} to {output_path}")
        
    def backup_database(self, backup_path: Optional[str] = None):
        """Create a backup of the database."""
        import shutil
        
        if not backup_path:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.db_path}.backup_{timestamp}"
            
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
        
        return backup_path


# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager("test_trading.db")
    db.connect()
    
    # Test analysis log
    analysis_data = {
        'symbol': 'XAU/USD',
        'primary_timeframe': '1H',
        'entry_signal': True,
        'entry_reason': 'Breakout + retest with RSI confirmation',
        'entry_price': 2050.50,
        'stop_loss': 2045.00,
        'take_profit_1': 2058.00,
        'position_size': 0.1,
        'expected_rr': 1.5,
        'confidence_score': 0.85
    }
    
    analysis_id = db.log_analysis(analysis_data)
    print(f"Logged analysis: {analysis_id}")
    
    # Test trade log
    trade_data = {
        'analysis_id': analysis_id,
        'symbol': 'XAU/USD',
        'platform': 'mt5',
        'direction': 'long',
        'entry_price': 2050.50,
        'stop_loss': 2045.00,
        'take_profit_1': 2058.00,
        'position_size': 0.1
    }
    
    trade_id = db.log_trade(trade_data)
    print(f"Logged trade: {trade_id}")
    
    # Get statistics
    stats = db.get_trade_statistics()
    print(f"Statistics: {stats}")
    
    db.disconnect()
    print("Database test completed successfully")