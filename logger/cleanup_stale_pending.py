#!/usr/bin/env python3
"""
scripts/cleanup_stale_pending.py
=================================
One-time script to clean up the 242 stale pending_limit records left over
from the Apr 7-10 test week across all three engine databases.

Usage:
    python scripts/cleanup_stale_pending.py --db-path path/to/trades.db
    python scripts/cleanup_stale_pending.py --db-path path/to/trades.db --dry-run

Run this ONCE for each of the 3 databases (main, t1-ict, t2-smc) before
starting the next week's test.
"""

import argparse
import sqlite3
from datetime import datetime


def cleanup_stale_pending(db_path: str, dry_run: bool = False):
    """Mark all pending_limit trades as cancelled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find all pending_limit trades
    cursor.execute("""
        SELECT trade_id, ticket, symbol, entry_time, status
        FROM trades
        WHERE status = 'pending_limit'
    """)
    stale = cursor.fetchall()

    print(f"Found {len(stale)} stale pending_limit records in {db_path}")

    if not stale:
        conn.close()
        return

    # Show breakdown
    by_symbol = {}
    for row in stale:
        sym = row['symbol']
        by_symbol[sym] = by_symbol.get(sym, 0) + 1
    for sym, count in sorted(by_symbol.items()):
        print(f"  {sym}: {count}")

    if dry_run:
        print("\n[DRY RUN] No changes made. Remove --dry-run to apply.")
        conn.close()
        return

    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # Update trades table
    cursor.execute("""
        UPDATE trades
        SET status = 'cancelled',
            exit_reason = 'stale_cleanup_post_test',
            exit_time = ?
        WHERE status = 'pending_limit'
    """, (now,))
    trades_updated = cursor.rowcount

    # Update pending_limit_orders table (if it exists)
    try:
        cursor.execute("""
            UPDATE pending_limit_orders
            SET status = 'cancelled',
                cancel_reason = 'stale_cleanup_post_test'
            WHERE status = 'pending'
        """)
        pending_updated = cursor.rowcount
    except sqlite3.OperationalError:
        pending_updated = 0  # Table doesn't exist

    conn.commit()
    conn.close()

    print(f"\nCleaned up:")
    print(f"  trades table:               {trades_updated} rows → cancelled")
    print(f"  pending_limit_orders table:  {pending_updated} rows → cancelled")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean up stale pending_limit records after test week'
    )
    parser.add_argument(
        '--db-path', required=True,
        help='Path to the SQLite trades database'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be cleaned without making changes'
    )
    args = parser.parse_args()
    cleanup_stale_pending(args.db_path, args.dry_run)
    -