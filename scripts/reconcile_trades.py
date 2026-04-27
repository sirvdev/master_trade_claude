"""
reconcile_trades.py v3 — Full Broker History Reconciliation
============================================================
Run from PROJECT ROOT:   python reconcile_trades.py
"""

import asyncio
import sys
import yaml
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.mt5_file_bridge import MT5FileBridge
from logger.db import DatabaseManager


async def reconcile():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db = DatabaseManager(Path(__file__).parent.parent / "data" / "trading.db")
    db.connect()

    mt5_config = config.get('execution', {}).get('mt5', {})
    mt5_config['mode'] = config.get('general', {}).get('mode', 'live')
    bridge = MT5FileBridge(mt5_config, demo_mode=False)
    await bridge.connect()

    if not bridge._connected:
        print("ERROR: Cannot connect to MT5 EA. Is it running?")
        return

    # ── Get ALL tickets from DB ────────────────────────────────────────
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT trade_id, ticket, symbol, direction, entry_price, exit_price,
               stop_loss, original_stop_loss, pnl, realized_rr, exit_reason,
               status, position_size, entry_time
        FROM trades
        WHERE ticket IS NOT NULL
        ORDER BY entry_time ASC
    """)
    all_rows = [dict(row) for row in cursor.fetchall()]

    # Group by ticket — some tickets may have duplicate rows
    db_by_ticket = defaultdict(list)
    for row in all_rows:
        db_by_ticket[int(row['ticket'])].append(row)

    status_counts = defaultdict(int)
    for row in all_rows:
        status_counts[row['status']] += 1

    print(f"\n{'='*70}")
    print(f"  FULL BROKER RECONCILIATION v3")
    print(f"{'='*70}")
    print(f"  DB rows: {len(all_rows)}")
    print(f"  Unique tickets: {len(db_by_ticket)}")
    print(f"  Statuses: {dict(status_counts)}")

    ea_positions = await bridge.get_all_positions() or []
    open_tickets = {int(p.get('ticket', 0)) for p in ea_positions}
    print(f"  Currently open on MT5: {len(open_tickets)}")
    print(f"\n{'─'*70}\n")

    updated = 0
    fixed = 0
    already_ok = 0
    truly_unfilled = 0
    no_data = 0
    skipped_open = 0

    for ticket, rows in sorted(db_by_ticket.items()):
        # Pick the "best" row for this ticket — prefer closed > pending_limit > cancelled
        priority = {'closed': 0, 'pending_exit': 1, 'open': 2, 'pending_limit': 3, 'cancelled': 4}
        rows.sort(key=lambda r: priority.get(r['status'], 99))
        trade = rows[0]

        trade_id = trade['trade_id']
        status = trade['status']
        symbol = trade['symbol']
        direction = trade['direction']
        old_pnl = float(trade['pnl'] or 0)
        entry_price = float(trade['entry_price'] or 0)
        original_sl = float(trade['original_stop_loss'] or trade['stop_loss'] or 0)

        # Skip currently open positions
        if ticket in open_tickets:
            print(f"  ticket={ticket} {symbol:8s} [{status:14s}] — still open, skip")
            skipped_open += 1
            continue

        # ── Fetch deal history from EA ─────────────────────────────────
        try:
            deal = await bridge.get_deal_history(ticket, lookback_hours=336)
        except Exception as e:
            print(f"  ticket={ticket} {symbol:8s} [{status:14s}] — ERROR: {e}")
            no_data += 1
            await asyncio.sleep(0.3)
            continue

        has_deals = (deal.get('status') == 'success'
                     and deal.get('deals')
                     and len(deal.get('deals', [])) > 0)

        # ── No deals found ─────────────────────────────────────────────
        if not has_deals:
            if status in ('cancelled', 'pending_limit'):
                print(f"  ticket={ticket} {symbol:8s} [{status:14s}] — truly unfilled, no deals")
                # Mark pending_limit as cancelled since it never filled
                if status == 'pending_limit':
                    db.update_trade(trade_id, {
                        'status': 'cancelled',
                        'exit_reason': 'reconcile_unfilled',
                    })
                truly_unfilled += 1
            elif status == 'closed' and old_pnl != 0:
                print(f"  ticket={ticket} {symbol:8s} [{status:14s}] — closed with PnL=${old_pnl:.2f}, no deals from EA (history expired?)")
                already_ok += 1
            else:
                print(f"  ticket={ticket} {symbol:8s} [{status:14s}] — no deal data")
                no_data += 1
            await asyncio.sleep(0.3)
            continue

        # ── Has deals — compute correct values ─────────────────────────
        deals_list = deal.get('deals', [])
        deal_count = len(deals_list)
        new_pnl = float(deal.get('net_profit', deal.get('profit', 0)))

        # Volume-weighted exit price
        total_vol = sum(float(d.get('volume', 0)) for d in deals_list)
        if total_vol > 0:
            new_exit = sum(
                float(d.get('exit_price', 0)) * float(d.get('volume', 0))
                for d in deals_list
            ) / total_vol
        else:
            new_exit = float(deals_list[-1].get('exit_price', 0))

        # R:R
        initial_risk = abs(entry_price - original_sl) if original_sl and entry_price else 0
        new_rr = 0.0
        if initial_risk > 0 and new_exit > 0 and entry_price > 0:
            price_move = (new_exit - entry_price) if direction == 'long' else (entry_price - new_exit)
            new_rr = round(price_move / initial_risk, 4)

        exit_reason = deal.get('exit_reason', 'external_close')
        close_time = deal.get('close_time')
        exit_dt = None
        if close_time and int(close_time) > 0:
            exit_dt = datetime.fromtimestamp(int(close_time), timezone.utc).replace(tzinfo=None)

        # Duration
        duration_minutes = None
        if trade.get('entry_time') and exit_dt:
            try:
                et = trade['entry_time']
                if isinstance(et, str):
                    et = datetime.fromisoformat(et.replace('Z', '+00:00')).replace(tzinfo=None)
                duration_minutes = round((exit_dt - et).total_seconds() / 60, 1)
            except:
                pass

        # ── Decide what to do ──────────────────────────────────────────
        needs_update = False
        action = ""

        if status in ('pending_limit', 'cancelled', 'pending_exit'):
            # This trade was never properly closed in the DB — fix it
            needs_update = True
            action = f"STATUS FIX {status} → closed"
        elif status == 'closed':
            pnl_diff = abs(new_pnl - old_pnl)
            exit_diff = abs(new_exit - float(trade['exit_price'] or 0))
            if pnl_diff > 0.01 or exit_diff > 0.001:
                needs_update = True
                action = f"PNL FIX ${old_pnl:.2f} → ${new_pnl:.2f}"
            else:
                print(f"  ticket={ticket} {symbol:8s} [{status:14s}] — OK ({deal_count} deals, ${new_pnl:.2f})")
                already_ok += 1
                await asyncio.sleep(0.3)
                continue

        if needs_update:
            update_data = {
                'status': 'closed',
                'pnl': round(new_pnl, 2),
                'exit_price': round(new_exit, 5),
                'realized_rr': new_rr,
                'exit_reason': exit_reason,
            }
            if exit_dt:
                update_data['exit_time'] = exit_dt
            if duration_minutes:
                update_data['duration_minutes'] = duration_minutes

            # Sum commission/swap across all deals
            total_commission = sum(float(d.get('commission', 0)) for d in deals_list)
            total_swap = sum(float(d.get('swap', 0)) for d in deals_list)
            if total_commission != 0:
                update_data['commission'] = round(total_commission, 2)

            db.update_trade(trade_id, update_data)

            # Also fix pending_limit_orders table
            try:
                cursor.execute(
                    "UPDATE pending_limit_orders SET status='filled' "
                    "WHERE ticket=? AND status IN ('pending', 'cancelled')",
                    (ticket,)
                )
                db.conn.commit()
            except:
                pass

            pnl_indicator = "✅" if new_pnl > 0 else "❌"
            print(f"  ticket={ticket} {symbol:8s} [{status:14s}] — {action} "
                  f"| {deal_count} deals | {pnl_indicator} ${new_pnl:+.2f} | RR={new_rr:.2f}")

            if status in ('pending_limit', 'cancelled', 'pending_exit'):
                fixed += 1
            else:
                updated += 1

        await asyncio.sleep(0.3)

    # ── Final summary ──────────────────────────────────────────────────
    cursor.execute("""
        SELECT status, COUNT(*) as cnt, SUM(COALESCE(pnl, 0)) as total_pnl
        FROM trades
        GROUP BY status
        ORDER BY status
    """)
    final_stats = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT SUM(pnl) FROM trades WHERE status='closed' AND pnl IS NOT NULL")
    final_pnl = float(cursor.fetchone()[0] or 0)

    cursor.execute("SELECT COUNT(*) FROM trades WHERE status='closed'")
    final_count = int(cursor.fetchone()[0] or 0)

    broker_pnl = -1685.14

    print(f"\n{'='*70}")
    print(f"  RECONCILIATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Fixed (pending_limit/cancelled → closed): {fixed}")
    print(f"  Updated (PnL corrected):                  {updated}")
    print(f"  Already correct:                          {already_ok}")
    print(f"  Truly unfilled (no deals):                {truly_unfilled}")
    print(f"  No data / errors:                         {no_data}")
    print(f"  Skipped (still open):                     {skipped_open}")
    print()
    print(f"  Final DB status breakdown:")
    for s in final_stats:
        print(f"    {s['status']:15s}: {s['cnt']:3d} trades  P&L: ${float(s['total_pnl'] or 0):+,.2f}")
    print()
    print(f"  DB closed trades: {final_count}")
    print(f"  DB total P&L:     ${final_pnl:+,.2f}")
    print(f"  Broker P&L:       ${broker_pnl:+,.2f}")
    print(f"  Gap:              ${(final_pnl - broker_pnl):+,.2f}")

    gap = abs(final_pnl - broker_pnl)
    if gap < 50:
        print(f"\n  ✅ Reconciled! Gap is only ${gap:.2f}")
    else:
        print(f"\n  ⚠️  Gap of ${gap:.2f} remains — some positions may have")
        print(f"      different tickets on MT5 than what Python stored.")

    print(f"{'='*70}\n")

    db.disconnect()
    await bridge.disconnect()


if __name__ == '__main__':
    asyncio.run(reconcile())