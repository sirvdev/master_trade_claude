"""
SQLite database manager for trading system.
Handles schema creation, migrations, and CRUD operations.
Postgres-ready design for future migration.
"""

import sqlite3
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ── Whitelist of allowed columns for update_trade() to prevent SQL injection ──
ALLOWED_TRADE_COLUMNS = {
    'status', 'exit_price', 'exit_time', 'exit_reason', 'pnl', 'pnl_percent',
    'realized_rr', 'duration_minutes', 'commission', 'slippage', 'ticket',
    'stop_loss', 'original_stop_loss', 'take_profit_1', 'take_profit_2',
    'take_profit_3', 'max_favorable_excursion', 'max_adverse_excursion',
    'equity_after_close'
}


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
        # ── Thread-safe lock for concurrent async/sync database operations ────────
        self._db_lock = threading.Lock()
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._initialize_schema()
        self._run_migrations()
        
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def _initialize_schema(self):
        """Create all tables if they don't exist."""
        with self._db_lock:
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
                    platform TEXT NOT NULL,  -- mt5
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
                    ticket INTEGER,  -- MT5 broker position ticket
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

            # Pending limit orders — tracks unfilled limit orders until
            # they are filled, expired, or price-invalidated.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_limit_orders (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id            TEXT NOT NULL,
                    ticket              INTEGER NOT NULL UNIQUE,
                    symbol              TEXT NOT NULL,
                    direction           TEXT NOT NULL,
                    entry_type          TEXT NOT NULL,
                    limit_price         REAL NOT NULL,
                    placed_at           TEXT NOT NULL,
                    expiry_time         TEXT NOT NULL,
                    min_cancel_floor    TEXT NOT NULL,
                    invalidation_price  REAL,
                    atr_at_placement    REAL,
                    status              TEXT NOT NULL DEFAULT 'pending',
                    cancelled_reason    TEXT,
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plo_status "
                "ON pending_limit_orders(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plo_ticket "
                "ON pending_limit_orders(ticket)"
            )
            
            # Create indexes for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_symbol ON analysis_logs(symbol)")
            # Stamp current schema version (idempotent — INSERT OR IGNORE)
            cursor.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)",(self.current_version,))
            
            self.conn.commit()
            logger.info("Database schema initialized")

    
    def _run_migrations(self):
        """
        Apply any schema migrations needed for existing databases.
        Safe to run every startup — each migration checks before applying.
        """
        with self._db_lock:
            cursor = self.conn.cursor()

            existing = {
                row[1] for row in cursor.execute("PRAGMA table_info(trades)")
            }

            # ── Migration 1: ticket column ────────────────────────────────────────
            if 'ticket' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN ticket INTEGER")
                logger.info("Migration: added 'ticket' column to trades")

            # ── Migration 2: original_stop_loss column ────────────────────────────
            if 'original_stop_loss' not in existing:
                cursor.execute(
                    "ALTER TABLE trades ADD COLUMN original_stop_loss REAL"
                )
                logger.info("Migration: added 'original_stop_loss' column to trades")

            # ── Migration 3: equity_after_close column ────────────────────────────
            # Stores live account equity snapshot taken immediately after each close.
            # Used by the analytics dashboard to plot a real equity curve.
            if 'equity_after_close' not in existing:
                cursor.execute(
                    "ALTER TABLE trades ADD COLUMN equity_after_close REAL"
                )
                logger.info("Migration: added 'equity_after_close' column to trades")

            # ── Migration 4: pending_limit_orders table ───────────────────────
            # Safe — CREATE TABLE IF NOT EXISTS handles fresh DBs.
            # Re-running on an existing DB that already has the table is a no-op.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_limit_orders (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id            TEXT NOT NULL,
                    ticket              INTEGER NOT NULL UNIQUE,
                    symbol              TEXT NOT NULL,
                    direction           TEXT NOT NULL,
                    entry_type          TEXT NOT NULL,
                    limit_price         REAL NOT NULL,
                    placed_at           TEXT NOT NULL,
                    expiry_time         TEXT NOT NULL,
                    min_cancel_floor    TEXT NOT NULL,
                    invalidation_price  REAL,
                    atr_at_placement    REAL,
                    status              TEXT NOT NULL DEFAULT 'pending',
                    cancelled_reason    TEXT,
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plo_status "
                "ON pending_limit_orders(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plo_ticket "
                "ON pending_limit_orders(ticket)"
            )

            self.conn.commit()

    def log_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """
        Log a market analysis pass.
        
        Args:
            analysis_data: Dictionary containing analysis results
            
        Returns:
            analysis_id of the logged entry
        """
        with self._db_lock:
            cursor = self.conn.cursor()
            
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
                json.dumps(self._convert_numpy_types(analysis_data.get('timeframe_snapshots', {}))),
                json.dumps(self._convert_numpy_types(analysis_data.get('market_structure', {}))),
                json.dumps(self._convert_numpy_types(analysis_data.get('indicators_state', {}))),
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


    def _convert_numpy_types(self, obj):
        """Convert numpy and pandas types to native Python types for JSON serialization."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, pd.Series):
            # Convert Series to list of values
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to dict of lists
            return obj.to_dict('list')
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            # ── Safe check for NaN values (can raise on non-scalar) ──────────
            try:
                if pd.isna(obj):
                    return None
            except (TypeError, ValueError):
                # pd.isna() failed on this type — just return as-is
                pass
            return obj
        
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Log a trade execution.

        Args:
            trade_data: Dictionary containing trade details.
                        Now includes 'original_stop_loss' so RR is always
                        computed against the entry-time SL, not the
                        post-breakeven level.

        Returns:
            trade_id of the logged trade
        """
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Log a trade execution.

        Args:
            trade_data: Dictionary containing trade details.
                        Now includes 'original_stop_loss' so RR is always
                        computed against the entry-time SL, not the
                        post-breakeven level.

        Returns:
            trade_id of the logged trade
        """
        with self._db_lock:
            cursor = self.conn.cursor()

            trade_id = (
                trade_data.get('trade_id')
                or f"trade_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            )

            cursor.execute("""
                INSERT INTO trades (
                    trade_id, analysis_id, symbol, platform, direction,
                    entry_time, entry_price, stop_loss, original_stop_loss,
                    take_profit_1, take_profit_2, take_profit_3,
                    position_size, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                trade_data.get('analysis_id'),
                trade_data.get('symbol'),
                trade_data.get('platform'),
                trade_data.get('direction'),
                datetime.utcnow(),
                trade_data.get('entry_price'),
                trade_data.get('stop_loss'),
                trade_data.get('original_stop_loss') or trade_data.get('stop_loss'),
                trade_data.get('take_profit_1'),
                trade_data.get('take_profit_2'),
                trade_data.get('take_profit_3'),
                trade_data.get('position_size'),
                trade_data.get('status', 'open'),
            ))

            self.conn.commit()
            return trade_id
        
    def update_trade(self, trade_id: str, updates: Dict[str, Any]):
        """
        Update trade fields.
        
        Args:
            trade_id: Trade ID to update
            updates: Dictionary of column -> value pairs
            
        Raises:
            ValueError: If any column name is not in the whitelist (prevents SQL injection)
        """
        # ── Validate column names against whitelist ────────────────────────────
        for col in updates.keys():
            if col not in ALLOWED_TRADE_COLUMNS:
                raise ValueError(
                    f"Invalid column '{col}' — not in allowed columns: "
                    f"{', '.join(sorted(ALLOWED_TRADE_COLUMNS))}"
                )
        
        with self._db_lock:
            cursor = self.conn.cursor()
            
            set_clauses = ', '.join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [trade_id]
            
            cursor.execute(f"""
                UPDATE trades SET {set_clauses} WHERE trade_id = ?
            """, values)
            
            self.conn.commit()
        
    def log_order_event(self, event_data: Dict[str, Any]):
        """Log an order lifecycle event."""
        with self._db_lock:
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
        with self._db_lock:
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
        with self._db_lock:
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
        with self._db_lock:
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
        with self._db_lock:
            cursor = self.conn.cursor()

            query  = "SELECT * FROM trades"
            params = []

            if filters:
                where_clauses = []
                for key, value in filters.items():
                    if value is None:
                        # SQL requires IS NULL — '= NULL' always returns no rows
                        where_clauses.append(f"{key} IS NULL")
                        # No param appended — IS NULL takes no bind parameter
                    else:
                        where_clauses.append(f"{key} = ?")
                        params.append(value)
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

            query += f" ORDER BY entry_time DESC LIMIT {limit}"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_trades_missing_pnl(self) -> List[Dict]:
        """
        Return all closed/pending trades that have no real P&L data.

        Matches:
        - pnl IS NULL                          (never written)
        - pnl = 0 AND status != 'open'         (written as zero placeholder)
        - exit_reason = 'deal_history_unavailable'
        - status = 'pending_exit'

        Only returns rows that have a ticket (required for deal history lookup).
        Excludes status='open' so live positions are never queued for backfill.
        """
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM trades
                WHERE  status IN ('closed', 'pending_exit')
                AND  ticket IS NOT NULL
                AND  (
                        pnl IS NULL
                    OR pnl = 0
                    OR exit_reason IN ('deal_history_unavailable', 'pending_deal_lookup')
                )
                ORDER  BY entry_time DESC
            """)
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
        with self._db_lock:
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
        
        with self._db_lock:
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
    
    def save_pending_limit_order(self, data: Dict[str, Any]) -> None:
        """
        Save a newly placed limit order to the pending tracker.
 
        Args:
            data: dict with keys: trade_id, ticket, symbol, direction,
                  entry_type, limit_price, placed_at, expiry_time,
                  min_cancel_floor, invalidation_price, atr_at_placement
        """
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO pending_limit_orders
                    (trade_id, ticket, symbol, direction, entry_type,
                     limit_price, placed_at, expiry_time, min_cancel_floor,
                     invalidation_price, atr_at_placement, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (
                data['trade_id'],
                int(data['ticket']),
                data['symbol'],
                data['direction'],
                data['entry_type'],
                float(data['limit_price']),
                str(data['placed_at']),
                str(data['expiry_time']),
                str(data['min_cancel_floor']),
                float(data['invalidation_price']) if data.get('invalidation_price') else None,
                float(data['atr_at_placement']) if data.get('atr_at_placement') else None,
            ))
            self.conn.commit()
 
    def get_pending_limit_orders(self) -> List[Dict[str, Any]]:
        """Return all limit orders with status='pending'."""
        with self._db_lock:
            cursor = self.conn.cursor()
            rows = cursor.execute("""
                SELECT id, trade_id, ticket, symbol, direction, entry_type,
                       limit_price, placed_at, expiry_time, min_cancel_floor,
                       invalidation_price, atr_at_placement, status
                FROM   pending_limit_orders
                WHERE  status = 'pending'
                ORDER  BY placed_at ASC
            """).fetchall()
            return [dict(r) for r in rows]
 
    def update_pending_limit_order_status(
        self,
        ticket: int,
        status: str,
        reason: str = None,
    ) -> None:
        """
        Update the status of a pending limit order.
 
        Args:
            ticket:  MT5 order ticket
            status:  New status — 'filled' | 'expired' | 'invalidated' | 'cancelled'
            reason:  Optional human-readable cancellation reason
        """
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE pending_limit_orders
                SET    status = ?, cancelled_reason = ?
                WHERE  ticket = ?
            """, (status, reason, int(ticket)))
            self.conn.commit()


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