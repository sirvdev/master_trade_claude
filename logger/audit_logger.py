"""
Audit logger for structured logging of decisions, trades, and system events.
Integrates with database and provides JSON-structured logs.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
from .db import DatabaseManager


class AuditLogger:
    """Structured logger for trading decisions and system events."""
    
    def __init__(self, db_manager: DatabaseManager, log_dir: str = "logs"):
        """
        Initialize audit logger.
        
        Args:
            db_manager: Database manager instance
            log_dir: Directory for JSON log files
        """
        self.db = db_manager
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # JSON file handler
        log_file = self.log_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Console handler for important events
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console)
        
    def _create_log_entry(self, event_type: str, data: Dict[str, Any]) -> Dict:
        """Create standardized log entry."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            **data
        }
        
    def _write_json_log(self, entry: Dict):
        """Write JSON log entry."""
        self.logger.info(json.dumps(entry))
        
    def log_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Log market analysis decision.
        
        Args:
            analysis: Analysis data dictionary
            
        Returns:
            analysis_id
        """
        # Store in database
        analysis_id = self.db.log_analysis(analysis)
        
        # Create JSON log
        entry = self._create_log_entry('analysis', {
            'analysis_id': analysis_id,
            'symbol': analysis.get('symbol'),
            'timeframe': analysis.get('primary_timeframe'),
            'entry_signal': analysis.get('entry_signal'),
            'entry_reason': analysis.get('entry_reason'),
            'confidence': analysis.get('confidence_score'),
            'expected_rr': analysis.get('expected_rr')
        })
        self._write_json_log(entry)
        
        return analysis_id
        
    def log_trade_entry(self, trade: Dict[str, Any]) -> str:
        """
        Log trade entry.
        
        Args:
            trade: Trade data dictionary
            
        Returns:
            trade_id
        """
        # Store in database
        trade_id = self.db.log_trade(trade)
        
        # Create JSON log
        entry = self._create_log_entry('trade_entry', {
            'trade_id': trade_id,
            'symbol': trade.get('symbol'),
            'platform': trade.get('platform'),
            'direction': trade.get('direction'),
            'entry_price': trade.get('entry_price'),
            'stop_loss': trade.get('stop_loss'),
            'position_size': trade.get('position_size'),
            'analysis_id': trade.get('analysis_id')
        })
        self._write_json_log(entry)
        
        self.logger.warning(
            f"TRADE ENTRY: {trade.get('symbol')} {trade.get('direction')} "
            f"@ {trade.get('entry_price')} (ID: {trade_id})"
        )
        
        return trade_id
        
    def log_trade_exit(self, trade_id: str, exit_data: Dict[str, Any]):
        """
        Log trade exit.
        
        Args:
            trade_id: Trade identifier
            exit_data: Exit data including price, reason, PnL
        """
        # Update database
        self.db.update_trade(trade_id, {
            'exit_time': datetime.utcnow(),
            'exit_price': exit_data.get('exit_price'),
            'exit_reason': exit_data.get('reason'),
            'pnl': exit_data.get('pnl'),
            'pnl_percent': exit_data.get('pnl_percent'),
            'realized_rr': exit_data.get('realized_rr'),
            'status': 'closed'
        })
        
        # Create JSON log
        entry = self._create_log_entry('trade_exit', {
            'trade_id': trade_id,
            'exit_price': exit_data.get('exit_price'),
            'exit_reason': exit_data.get('reason'),
            'pnl': exit_data.get('pnl'),
            'pnl_percent': exit_data.get('pnl_percent'),
            'realized_rr': exit_data.get('realized_rr')
        })
        self._write_json_log(entry)
        
        self.logger.warning(
            f"TRADE EXIT: {trade_id} - {exit_data.get('reason')} "
            f"PnL: {exit_data.get('pnl'):.2f} ({exit_data.get('pnl_percent'):.2f}%)"
        )
        
    def log_partial_exit(self, trade_id: str, partial_data: Dict[str, Any]):
        """
        Log partial position close.
        
        Args:
            trade_id: Trade identifier
            partial_data: Partial close data
        """
        # Update trade status
        self.db.update_trade(trade_id, {'status': 'partial'})
        
        # Log order event
        self.db.log_order_event({
            'trade_id': trade_id,
            'event_type': 'partial',
            'price': partial_data.get('price'),
            'quantity': partial_data.get('quantity'),
            'notes': partial_data.get('reason')
        })
        
        # Create JSON log
        entry = self._create_log_entry('partial_exit', {
            'trade_id': trade_id,
            'price': partial_data.get('price'),
            'quantity_closed': partial_data.get('quantity'),
            'quantity_remaining': partial_data.get('quantity_remaining'),
            'reason': partial_data.get('reason'),
            'partial_pnl': partial_data.get('pnl')
        })
        self._write_json_log(entry)
        
        self.logger.info(
            f"PARTIAL EXIT: {trade_id} - {partial_data.get('quantity')} closed "
            f"@ {partial_data.get('price')} ({partial_data.get('reason')})"
        )
        
    def log_sl_tp_adjustment(self, trade_id: str, adjustment: Dict[str, Any]):
        """
        Log stop loss or take profit adjustment.
        
        Args:
            trade_id: Trade identifier
            adjustment: Adjustment data
        """
        # Store in database
        self.db.log_sl_tp_adjustment({
            'trade_id': trade_id,
            **adjustment
        })
        
        # Create JSON log
        entry = self._create_log_entry('sl_tp_adjustment', {
            'trade_id': trade_id,
            'adjustment_type': adjustment.get('adjustment_type'),
            'old_value': adjustment.get('old_value'),
            'new_value': adjustment.get('new_value'),
            'trigger_reason': adjustment.get('trigger_reason'),
            'current_rr': adjustment.get('current_rr')
        })
        self._write_json_log(entry)
        
        self.logger.info(
            f"SL/TP ADJUSTMENT: {trade_id} - {adjustment.get('adjustment_type')} "
            f"from {adjustment.get('old_value')} to {adjustment.get('new_value')} "
            f"({adjustment.get('trigger_reason')})"
        )
        
    def log_order_event(self, event: Dict[str, Any]):
        """
        Log order lifecycle event.
        
        Args:
            event: Order event data
        """
        # Store in database
        self.db.log_order_event(event)
        
        # Create JSON log
        entry = self._create_log_entry('order_event', {
            'trade_id': event.get('trade_id'),
            'event_type': event.get('event_type'),
            'order_type': event.get('order_type'),
            'price': event.get('price'),
            'quantity': event.get('quantity')
        })
        self._write_json_log(entry)
        
    def log_error(self, component: str, error: Exception, context: Optional[Dict] = None):
        """
        Log system error.
        
        Args:
            component: Component/module name
            error: Exception object
            context: Additional context
        """
        # Store in database
        cursor = self.db.conn.cursor()
        cursor.execute("""
            INSERT INTO system_events (event_type, component, message, details)
            VALUES (?, ?, ?, ?)
        """, ('error', component, str(error), json.dumps(context or {})))
        self.db.conn.commit()
        
        # Create JSON log
        entry = self._create_log_entry('error', {
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        })
        self._write_json_log(entry)
        
        self.logger.error(f"ERROR in {component}: {error}", exc_info=True)
        
    def log_risk_event(self, event: Dict[str, Any]):
        """
        Log risk management event (drawdown, limits hit, etc.).
        
        Args:
            event: Risk event data
        """
        entry = self._create_log_entry('risk_event', event)
        self._write_json_log(entry)
        
        self.logger.warning(
            f"RISK EVENT: {event.get('event_type')} - {event.get('message')}"
        )
        
    def log_learning_event(self, event: Dict[str, Any]):
        """
        Log learning engine event.
        
        Args:
            event: Learning event data
        """
        entry = self._create_log_entry('learning', event)
        self._write_json_log(entry)
        
        self.logger.info(
            f"LEARNING: {event.get('event_type')} - {event.get('message')}"
        )
        
    def log_manual_intervention(self, intervention: Dict[str, Any]):
        """
        Log manual override or intervention.
        
        Args:
            intervention: Intervention data with reason
        """
        # Store in database
        cursor = self.db.conn.cursor()
        cursor.execute("""
            INSERT INTO system_events (event_type, component, message, details)
            VALUES (?, ?, ?, ?)
        """, (
            'manual_intervention',
            'user',
            intervention.get('action'),
            json.dumps(intervention)
        ))
        self.db.conn.commit()
        
        # Create JSON log
        entry = self._create_log_entry('manual_intervention', intervention)
        self._write_json_log(entry)
        
        self.logger.warning(
            f"MANUAL INTERVENTION: {intervention.get('action')} - "
            f"Reason: {intervention.get('reason')}"
        )
        
    def get_recent_logs(self, hours: int = 24, event_types: Optional[list] = None) -> list:
        """
        Retrieve recent log entries from database.
        
        Args:
            hours: Number of hours to look back
            event_types: Optional filter by event types
            
        Returns:
            List of log entries
        """
        cursor = self.db.conn.cursor()
        
        query = """
            SELECT * FROM system_events
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
        """
        params = [hours]
        
        if event_types:
            placeholders = ','.join('?' * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
            
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
        
    def generate_daily_summary(self) -> Dict[str, Any]:
        """
        Generate daily performance summary.
        
        Returns:
            Summary statistics
        """
        stats = self.db.get_trade_statistics(days=1)
        
        # Log summary
        entry = self._create_log_entry('daily_summary', {
            'total_trades': stats.get('total_trades', 0),
            'winning_trades': stats.get('winning_trades', 0),
            'losing_trades': stats.get('losing_trades', 0),
            'win_rate': stats.get('win_rate', 0),
            'total_pnl': stats.get('total_pnl', 0),
            'avg_rr': stats.get('avg_rr', 0)
        })
        self._write_json_log(entry)
        
        return stats


# Example usage
if __name__ == "__main__":
    from .db import DatabaseManager
    
    db = DatabaseManager("test_trading.db")
    db.connect()
    
    audit = AuditLogger(db)
    
    # Test analysis log
    analysis_id = audit.log_analysis({
        'symbol': 'BTC/USDT',
        'primary_timeframe': '1H',
        'entry_signal': True,
        'entry_reason': 'EMA crossover + RSI oversold recovery',
        'confidence_score': 0.78,
        'expected_rr': 2.0
    })
    
    # Test trade entry
    trade_id = audit.log_trade_entry({
        'analysis_id': analysis_id,
        'symbol': 'BTC/USDT',
        'platform': 'binance',
        'direction': 'long',
        'entry_price': 42000.0,
        'stop_loss': 41500.0,
        'position_size': 0.01
    })
    
    # Test SL adjustment
    audit.log_sl_tp_adjustment(trade_id, {
        'adjustment_type': 'trailing_activated',
        'old_value': 41500.0,
        'new_value': 41800.0,
        'trigger_reason': 'Reached 1:1 R:R',
        'current_rr': 1.0
    })
    
    # Test trade exit
    audit.log_trade_exit(trade_id, {
        'exit_price': 43000.0,
        'reason': 'tp1',
        'pnl': 1000.0,
        'pnl_percent': 2.38,
        'realized_rr': 2.0
    })
    
    # Generate summary
    summary = audit.generate_daily_summary()
    print(f"Daily Summary: {summary}")
    
    db.disconnect()
    print("Audit logger test completed successfully")