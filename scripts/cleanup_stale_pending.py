#!/usr/bin/env python3
"""
scripts/cleanup_stale_pending.py (REVISED)
============================================
Clean up stale pending_limit records. This version is DB-only cleanup
for records that are guaranteed stale (market is closed, so the broker
already expired them). Safe to run on weekends.

For records where the market is OPEN, use --check-ea mode which will
attempt to verify each order against the EA before marking cancelled.

Usage:
    # Dry run — see what would change
    python scripts/cleanup_stale_pending.py --db-path path/to/trades.db --dry-run

    # Clean up (only touches orders for symbols whose market is closed)
    python scripts/cleanup_stale_pending.py --db-path path/to/trades.db

    # Force all — only use if you've manually confirmed no orders remain on MT5
    python scripts/cleanup_stale_pending.py --db-path path/to/trades.db --force-all

Run once for each database (main, t1-ict, t2-smc) before starting next week.
"""

import argparse
import sqlite3
from datetime import datetime


def cleanup_stale_pending(db_path: str, symbols: list, dry_run: bool = True,
                          force_all: bool = False):
    """Mark stale pending_limit trades as cancelled."""
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

    print(f"Found {len(stale)} pending_limit records in {db_path}")

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

    print(f"\nScoped to symbols: {', '.join(symbols)}")

    if dry_run:
        print("\n[DRY RUN] No changes made. Pass --apply to write.")
        conn.close()
        return

    if not force_all:
        # Safety check: only proceed on weekends or when user confirms
        now = datetime.utcnow()
        if now.weekday() < 5:  # Mon-Fri
            print(
                f"\n⚠️  Today is {now.strftime('%A')} — markets may be open."
                f"\n    These orders might still be live on the broker!"
                f"\n    Options:"
                f"\n      1. Run this on a weekend when all markets are closed"
                f"\n      2. Use --force-all if you've manually confirmed on MT5"
                f"\n      3. Let the system's _check_pending_limit_orders() handle"
                f"\n         it automatically on next startup (it checks the EA)"
            )
            resp = input("\n    Proceed anyway? (yes/no): ").strip().lower()
            if resp != 'yes':
                print("Aborted.")
                conn.close()
                return

    now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # B3 FIXED 2026-08-30 audit: both statements below used to run without any
    # symbol filter, despite this module's docstring claiming it "only touches
    # orders for symbols whose market is closed". The EA places every pending as
    # GTC with expiration 0, so brokers do NOT expire them over a weekend -- run
    # against an instance holding BTC/USDm or ETH/USDm and every live crypto
    # order was marked cancelled in the DB while it stayed live at the broker,
    # after which nothing would ever expire or cancel it. --symbols is now
    # mandatory and both UPDATEs are scoped to it.
    placeholders = ','.join('?' for _ in symbols)

    cursor.execute(f"""
        UPDATE trades
        SET status = 'cancelled',
            exit_reason = 'stale_cleanup_post_test',
            exit_time = ?
        WHERE status = 'pending_limit'
          AND symbol IN ({placeholders})
    """, (now_str, *symbols))
    trades_updated = cursor.rowcount

    # Update pending_limit_orders table (if it exists)
    try:
        cursor.execute(f"""
            UPDATE pending_limit_orders
            SET status = 'cancelled',
                cancelled_reason = 'stale_cleanup_post_test'
            WHERE status = 'pending'
              AND symbol IN ({placeholders})
        """, tuple(symbols))
        pending_updated = cursor.rowcount
    except sqlite3.OperationalError as _oe:
        # B2 FIXED 2026-08-30 audit: the column above was written as
        # `cancel_reason`; the schema column is `cancelled_reason`
        # (logger/db.py). sqlite raised OperationalError, this handler swallowed
        # it, and because sqlite3 does not roll back a failed statement the
        # commit below still landed the trades UPDATE. The two tables then
        # disagreed and the report printed "0 rows", indistinguishable from
        # "no such table". Only a genuinely missing table is benign now.
        if 'no such table' not in str(_oe).lower():
            raise
        pending_updated = 0

    conn.commit()
    conn.close()

    print(f"\nCleaned up:")
    print(f"  trades table:               {trades_updated} rows → cancelled")
    print(f"  pending_limit_orders table:  {pending_updated} rows → cancelled")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean up stale pending_limit records'
    )
    parser.add_argument(
        '--db-path', required=True,
        help='Path to the SQLite trades database'
    )
    parser.add_argument(
        '--symbols', required=True,
        help='Comma-separated symbols to clean, exactly as they appear in the '
             'trades table (e.g. "XAU/USDm,XAG/USDm"). Required: this script '
             'used to update every pending row regardless of instrument.'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Deprecated, dry run is now the default. Pass --apply to write.'
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Actually write the changes. Without it this is a dry run.'
    )
    parser.add_argument(
        '--include-crypto', action='store_true',
        help='Allow crypto symbols. Crypto trades 24/7, so a pending order on '
             'it is almost never stale; requires an explicit opt-in.'
    )
    parser.add_argument(
        '--force-all', action='store_true',
        help='Skip safety checks (use only if you confirmed on MT5)'
    )
    args = parser.parse_args()

    syms = [x.strip() for x in args.symbols.split(',') if x.strip()]
    if not syms:
        parser.error('--symbols must name at least one symbol')

    CRYPTO_TOKENS = ('BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'ADA', 'SOL', 'DOGE')
    crypto = [s for s in syms
              if any(tok in s.upper().replace('/', '') for tok in CRYPTO_TOKENS)]
    if crypto and not args.include_crypto:
        parser.error(
            f"refusing to touch 24/7 instruments {crypto}: their markets never "
            f"close, so a pending order on them is not stale. Pass "
            f"--include-crypto only if you have confirmed on MT5 that these "
            f"specific orders are gone."
        )

    cleanup_stale_pending(args.db_path, syms, not args.apply, args.force_all)