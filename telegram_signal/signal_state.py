"""
signal_state.py
===============
Tracks the lifecycle of every channel signal from entry through all TPs.

Each channel signal (one Telegram message) maps to N MT5 positions
(one per TP). This module keeps that mapping persistent in a SQLite
table alongside the main trading DB, so the state survives restarts.

State machine per signal:
    pending  → open → partially_closed → closed
                   ↘ cancelled (before any fill)

State machine per position within a signal:
    pending_limit → open → tp_hit | sl_hit | closed_manual
"""

import sqlite3
import logging
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SignalPosition:
    """One MT5 position opened for a specific TP level."""
    signal_id:    str           # parent signal ID
    tp_index:     int           # 1-based (TP1, TP2, ...)
    tp_price:     float
    ticket:       Optional[int] = None    # MT5 ticket, None until filled
    lot_size:     float         = 0.01
    entry_price:  float         = 0.0
    stop_loss:    float         = 0.0
    order_type:   str           = "market"  # 'market' | 'limit'
    status:       str           = "pending"  # pending | open | tp_hit | sl_hit | closed
    opened_at:    Optional[str] = None
    closed_at:    Optional[str] = None
    pnl:          Optional[float] = None


@dataclass
class SignalRecord:
    """Full record for one channel signal and all its child positions."""
    signal_id:      str
    message_id:     int           # Telegram message ID
    channel:        str
    raw_text:       str
    symbol:         str
    direction:      str           # 'buy' | 'sell'
    entry_type:     str           # 'market' | 'limit' | 'range'
    entry_price:    Optional[float]
    stop_loss:      Optional[float]
    take_profits:   list[float]   # ordered list
    positions:      list[SignalPosition] = field(default_factory=list)
    status:         str           = "pending"   # pending|open|partially_closed|closed|cancelled
    created_at:     str           = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes:          str           = ""


class SignalStateManager:
    """
    Persists signal state in a dedicated SQLite table.
    Thread-safe for use from async context (calls are short, serialised by GIL).
    """

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channel_signals (
                    signal_id     TEXT PRIMARY KEY,
                    message_id    INTEGER,
                    channel       TEXT,
                    raw_text      TEXT,
                    symbol        TEXT,
                    direction     TEXT,
                    entry_type    TEXT,
                    entry_price   REAL,
                    stop_loss     REAL,
                    take_profits  TEXT,   -- JSON array
                    status        TEXT DEFAULT 'pending',
                    created_at    TEXT,
                    notes         TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channel_positions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id    TEXT REFERENCES channel_signals(signal_id),
                    tp_index     INTEGER,
                    tp_price     REAL,
                    ticket       INTEGER,
                    lot_size     REAL,
                    entry_price  REAL,
                    stop_loss    REAL,
                    order_type   TEXT,
                    status       TEXT DEFAULT 'pending',
                    opened_at    TEXT,
                    closed_at    TEXT,
                    pnl          REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_csig_symbol
                ON channel_signals(symbol, status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cpos_ticket
                ON channel_positions(ticket)
            """)

    # ── Write ──────────────────────────────────────────────────────────────────

    def save_signal(self, record: SignalRecord):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO channel_signals
                (signal_id, message_id, channel, raw_text, symbol, direction,
                 entry_type, entry_price, stop_loss, take_profits, status,
                 created_at, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                record.signal_id, record.message_id, record.channel,
                record.raw_text, record.symbol, record.direction,
                record.entry_type, record.entry_price, record.stop_loss,
                json.dumps(record.take_profits), record.status,
                record.created_at, record.notes,
            ))
            for pos in record.positions:
                self._upsert_position(conn, pos)

    def _upsert_position(self, conn: sqlite3.Connection, pos: SignalPosition):
        existing = conn.execute(
            "SELECT id FROM channel_positions WHERE signal_id=? AND tp_index=?",
            (pos.signal_id, pos.tp_index)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE channel_positions
                SET ticket=?, entry_price=?, stop_loss=?, order_type=?,
                    status=?, opened_at=?, closed_at=?, pnl=?, lot_size=?
                WHERE signal_id=? AND tp_index=?
            """, (
                pos.ticket, pos.entry_price, pos.stop_loss, pos.order_type,
                pos.status, pos.opened_at, pos.closed_at, pos.pnl, pos.lot_size,
                pos.signal_id, pos.tp_index,
            ))
        else:
            conn.execute("""
                INSERT INTO channel_positions
                (signal_id, tp_index, tp_price, ticket, lot_size, entry_price,
                 stop_loss, order_type, status, opened_at, closed_at, pnl)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                pos.signal_id, pos.tp_index, pos.tp_price, pos.ticket,
                pos.lot_size, pos.entry_price, pos.stop_loss, pos.order_type,
                pos.status, pos.opened_at, pos.closed_at, pos.pnl,
            ))

    def update_position_ticket(self, signal_id: str, tp_index: int,
                                ticket: int, entry_price: float,
                                order_type: str = "market"):
        with self._conn() as conn:
            conn.execute("""
                UPDATE channel_positions
                SET ticket=?, entry_price=?, order_type=?, status='open', opened_at=?
                WHERE signal_id=? AND tp_index=?
            """, (ticket, entry_price, order_type,
                  datetime.utcnow().isoformat(),
                  signal_id, tp_index))

    def update_position_closed(self, ticket: int, status: str, pnl: float = 0.0):
        with self._conn() as conn:
            conn.execute("""
                UPDATE channel_positions
                SET status=?, closed_at=?, pnl=?
                WHERE ticket=?
            """, (status, datetime.utcnow().isoformat(), pnl, ticket))
            # Check if all positions in the signal are closed
            row = conn.execute(
                "SELECT signal_id FROM channel_positions WHERE ticket=?", (ticket,)
            ).fetchone()
            if row:
                self._maybe_close_signal(conn, row["signal_id"])

    def update_signal_status(self, signal_id: str, status: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE channel_signals SET status=? WHERE signal_id=?",
                (status, signal_id)
            )

    def update_position_sl(self, signal_id: str, new_sl: float,
                           only_open: bool = True):
        """Update SL for all positions in a signal (or only open ones)."""
        with self._conn() as conn:
            if only_open:
                conn.execute("""
                    UPDATE channel_positions SET stop_loss=?
                    WHERE signal_id=? AND status='open'
                """, (new_sl, signal_id))
            else:
                conn.execute(
                    "UPDATE channel_positions SET stop_loss=? WHERE signal_id=?",
                    (new_sl, signal_id)
                )

    def _maybe_close_signal(self, conn: sqlite3.Connection, signal_id: str):
        rows = conn.execute(
            "SELECT status FROM channel_positions WHERE signal_id=?",
            (signal_id,)
        ).fetchall()
        statuses = {r["status"] for r in rows}
        open_statuses = {"open", "pending"}
        if not statuses & open_statuses:
            conn.execute(
                "UPDATE channel_signals SET status='closed' WHERE signal_id=?",
                (signal_id,)
            )
            logger.info(f"[STATE] Signal {signal_id} fully closed.")

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_signal(self, signal_id: str) -> Optional[SignalRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM channel_signals WHERE signal_id=?", (signal_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_record(conn, row)

    def get_open_signals(self, symbol: Optional[str] = None) -> list[SignalRecord]:
        with self._conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM channel_signals WHERE status IN ('pending','open','partially_closed') AND symbol=?",
                    (symbol,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM channel_signals WHERE status IN ('pending','open','partially_closed')"
                ).fetchall()
            return [self._row_to_record(conn, r) for r in rows]

    def get_open_positions(self, signal_id: str) -> list[SignalPosition]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM channel_positions WHERE signal_id=? AND status='open'",
                (signal_id,)
            ).fetchall()
            return [self._row_to_position(r) for r in rows]

    def get_position_by_ticket(self, ticket: int) -> Optional[SignalPosition]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM channel_positions WHERE ticket=?", (ticket,)
            ).fetchone()
            return self._row_to_position(row) if row else None

    def get_signal_by_message_id(self, message_id: int) -> Optional[SignalRecord]:
        """Find signal created from a specific Telegram message ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM channel_signals WHERE message_id=?", (message_id,)
            ).fetchone()
            return self._row_to_record(conn, row) if row else None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _row_to_record(self, conn: sqlite3.Connection, row: sqlite3.Row) -> SignalRecord:
        pos_rows = conn.execute(
            "SELECT * FROM channel_positions WHERE signal_id=? ORDER BY tp_index",
            (row["signal_id"],)
        ).fetchall()
        return SignalRecord(
            signal_id   = row["signal_id"],
            message_id  = row["message_id"],
            channel     = row["channel"],
            raw_text    = row["raw_text"],
            symbol      = row["symbol"],
            direction   = row["direction"],
            entry_type  = row["entry_type"],
            entry_price = row["entry_price"],
            stop_loss   = row["stop_loss"],
            take_profits= json.loads(row["take_profits"] or "[]"),
            positions   = [self._row_to_position(r) for r in pos_rows],
            status      = row["status"],
            created_at  = row["created_at"],
            notes       = row["notes"] or "",
        )

    def _row_to_position(self, row: sqlite3.Row) -> SignalPosition:
        return SignalPosition(
            signal_id   = row["signal_id"],
            tp_index    = row["tp_index"],
            tp_price    = row["tp_price"],
            ticket      = row["ticket"],
            lot_size    = row["lot_size"],
            entry_price = row["entry_price"],
            stop_loss   = row["stop_loss"],
            order_type  = row["order_type"],
            status      = row["status"],
            opened_at   = row["opened_at"],
            closed_at   = row["closed_at"],
            pnl         = row["pnl"],
        )