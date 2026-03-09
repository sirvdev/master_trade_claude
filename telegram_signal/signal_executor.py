"""
signal_executor.py
==================
Executes parsed channel signals via the MT5 file bridge.

Recursive position model:
    One signal with N TPs → up to N simultaneous MT5 positions, each with:
      - Individual TP (TP1 for pos 1, TP2 for pos 2, etc.)
      - Equal lot size derived from risk % split across positions
      - Same SL

    Sizing logic:
      - Total risk budget  = equity × SIGNAL_RISK_PERCENT / 100
      - Budget per position = total_budget / num_positions
      - Lot per position    = floor(budget_per_pos / (|entry−SL| × contract_size), step)
      - If lot < MIN_LOT, drop that TP and retry with N-1
      - Always keep TP1..TPN in order, dropping from the END

Entry when price has moved:
      - TPs already passed          → skip entirely
      - First MAX_MARKET valid TPs  → market order at current price
      - Remaining valid TPs         → limit order at original entry price
      - No original entry price     → all market

Limit order expiry:
      - NOT auto-cancelled
      - After SIGNAL_LIMIT_ALERT_HOURS the user is alerted with ticket
        numbers to decide manually. No automatic action.

All configuration via .env (full reference at bottom of file).
"""

import asyncio
import logging
import math
import os
import uuid
from datetime import datetime
from typing import Optional

from signal_parser import ParsedSignal
from signal_state import SignalStateManager, SignalRecord, SignalPosition

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def _floor_lot(lot: float, step: float) -> float:
    """Round lot DOWN to the nearest lot step (e.g. 0.01)."""
    if step <= 0:
        return lot
    return math.floor(lot / step) * step


# ── Executor ───────────────────────────────────────────────────────────────────

class SignalExecutor:
    """
    Receives ParsedSignal objects and executes them against MT5.

    .env keys and defaults
    ──────────────────────
    SIGNAL_RISK_PERCENT         1.0    total % of equity risked for the whole signal
    SIGNAL_MAGIC_NUMBER         234567 separate from main system magic number
    SIGNAL_MAX_MARKET_ORDERS    2      max positions opened at market when price moved
    SIGNAL_PRICE_TOLERANCE_PIPS 50     market order allowed if within N pips of entry
    SIGNAL_LIMIT_ALERT_HOURS    6      alert user when limit orders this old are unfilled
    SIGNAL_DEFAULT_SYMBOL       XAUUSD fallback symbol when parser finds none
    SIGNAL_CONTRACT_SIZE        100    oz per lot for XAUUSD (use 1 for FX pairs)
    SIGNAL_MIN_LOT              0.01   broker minimum lot size
    SIGNAL_MAX_LOT              10.0   per-position ceiling
    SIGNAL_LOT_STEP             0.01   broker lot increment
    """

    def __init__(
        self,
        mt5_bridge,
        state_manager: SignalStateManager,
        notifier=None,
        channel: str = "",
    ):
        self.bridge   = mt5_bridge
        self.state    = state_manager
        self.notifier = notifier
        self.channel  = channel

        # Risk & sizing
        self.risk_pct        = _env_float("SIGNAL_RISK_PERCENT", 1.0)
        self.contract_size   = _env_float("SIGNAL_CONTRACT_SIZE", 100.0)
        self.min_lot         = _env_float("SIGNAL_MIN_LOT", 0.01)
        self.max_lot         = _env_float("SIGNAL_MAX_LOT", 10.0)
        self.lot_step        = _env_float("SIGNAL_LOT_STEP", 0.01)

        # Execution
        self.magic_number    = _env_int("SIGNAL_MAGIC_NUMBER", 234567)
        self.max_market      = _env_int("SIGNAL_MAX_MARKET_ORDERS", 2)
        self.price_tol_pips  = _env_float("SIGNAL_PRICE_TOLERANCE_PIPS", 50)
        self.limit_alert_h   = _env_float("SIGNAL_LIMIT_ALERT_HOURS", 6)
        self.default_symbol  = os.getenv("SIGNAL_DEFAULT_SYMBOL", "XAUUSD")

        # When budget only covers 1 position, use this TP index instead of TP1
        # Default=3: bigger reward target since risk is fixed at min_lot anyway
        self.single_pos_tp_index = _env_int("SIGNAL_SINGLE_POS_TP_INDEX", 3)

        logger.info(
            f"[EXECUTOR] Init — risk={self.risk_pct}% magic={self.magic_number} "
            f"max_market={self.max_market} contract={self.contract_size} "
            f"min_lot={self.min_lot} step={self.lot_step} "
            f"limit_alert={self.limit_alert_h}h"
        )

    # ── Position sizing ────────────────────────────────────────────────────────

    def _calculate_lot_sizes(
        self,
        equity:        float,
        entry_price:   float,
        stop_loss:     float,
        num_positions: int,
    ) -> tuple[float, int]:
        """
        Calculate lot size per position and how many TPs the budget can cover.

        Returns:
            (lot_per_position, affordable_count)

        Algorithm:
            total_risk_usd   = equity × risk_pct / 100
            budget_per_pos   = total_risk_usd / num_positions
            risk_per_1_lot   = |entry − SL| × contract_size
            lot_per_pos      = floor(budget_per_pos / risk_per_1_lot, lot_step)

        If the result is below min_lot we reduce num_positions by 1 and retry,
        always keeping the first N TPs (closest to entry = most likely to fill).
        """
        total_risk_usd = equity * self.risk_pct / 100.0
        risk_per_1_lot = abs(entry_price - stop_loss) * self.contract_size

        if risk_per_1_lot <= 0:
            logger.error("[SIZER] risk_per_1_lot is 0 — bad entry/SL values, using min_lot")
            return self.min_lot, 1

        for count in range(num_positions, 0, -1):
            budget_per_pos = total_risk_usd / count
            raw_lot        = budget_per_pos / risk_per_1_lot
            lot            = _floor_lot(raw_lot, self.lot_step)
            lot            = min(lot, self.max_lot)

            if lot >= self.min_lot:
                dropped = num_positions - count
                if dropped:
                    logger.info(
                        f"[SIZER] Budget covers {count}/{num_positions} positions "
                        f"— dropped {dropped} TP(s) from the end"
                    )
                actual_risk = risk_per_1_lot * lot * count
                logger.info(
                    f"[SIZER] equity={equity:.0f}  risk_budget={total_risk_usd:.2f}  "
                    f"count={count}  lot_each={lot:.2f}  "
                    f"actual_risk={actual_risk:.2f}"
                )
                return lot, count

        # Absolute fallback — single position at min_lot
        logger.warning("[SIZER] Could not afford even 1 position at min_lot — using min_lot x1")
        return self.min_lot, 1

    # ── Main dispatch ──────────────────────────────────────────────────────────

    async def execute(self, signal: ParsedSignal, message_id: int = 0):
        """Route signal to handler. Never raises."""
        try:
            logger.info(
                f"[EXECUTOR] {signal.signal_type.upper()} "
                f"symbol={signal.symbol} conf={signal.confidence:.1f}"
            )
            if signal.signal_type == "entry":
                await self._handle_entry(signal, message_id)
            elif signal.signal_type == "tp_hit":
                await self._handle_tp_hit(signal)
            elif signal.signal_type == "breakeven":
                await self._handle_breakeven(signal)
            elif signal.signal_type == "move_sl":
                await self._handle_move_sl(signal)
            elif signal.signal_type == "close":
                await self._handle_close(signal)
            elif signal.signal_type == "close_all":
                await self._handle_close_all(signal)
            elif signal.signal_type == "cancel":
                await self._handle_cancel(signal)
            elif signal.signal_type == "pre_announcement":
                await self._handle_pre_announcement(signal)
            elif signal.signal_type == "unknown":
                logger.debug(f"[EXECUTOR] Unknown — skipping: {signal.raw_text[:60]!r}")

        except Exception as e:
            logger.error(f"[EXECUTOR] Error on {signal.signal_type}: {e}", exc_info=True)
            await self._notify(f"⚠️ Executor error ({signal.signal_type}): {e}")

    # ── Entry ──────────────────────────────────────────────────────────────────

    async def _handle_entry(self, signal: ParsedSignal, message_id: int):
        symbol    = signal.symbol or self.default_symbol
        direction = signal.direction
        tps       = signal.take_profits
        sl        = signal.stop_loss

        if not tps:
            await self._notify(f"⚠️ Entry for {symbol} skipped — no TPs found.")
            return
        if not sl:
            await self._notify(f"⚠️ Entry for {symbol} skipped — no SL found.")
            return

        current_price = await self._get_current_price(symbol, direction)
        if not current_price:
            await self._notify(f"⚠️ Cannot get price for {symbol} — entry skipped.")
            return

        original_entry  = signal.entry_price       # None for pure "buy now"
        effective_entry = original_entry or current_price

        # ── Classify each TP ──────────────────────────────────────────────────
        # Order: skip passed → market (up to max_market) → limit → rest market
        valid_tps: list[tuple[int, float, str]] = []
        market_used = 0

        for i, tp in enumerate(tps, start=1):
            if self._tp_already_passed(direction, current_price, tp):
                logger.info(f"[EXECUTOR] TP{i} ({tp}) already passed — skipping")
                continue

            if market_used < self.max_market:
                valid_tps.append((i, tp, "market"))
                market_used += 1
            elif original_entry:
                valid_tps.append((i, tp, "limit"))
            else:
                valid_tps.append((i, tp, "market"))

        if not valid_tps:
            await self._notify(f"⚠️ {symbol} — all TPs already passed, signal skipped.")
            return

        # ── Size positions ─────────────────────────────────────────────────────
        equity = await self._get_equity()
        if not equity:
            await self._notify(f"⚠️ Cannot read equity — entry for {symbol} skipped.")
            return

        lot_per_pos, affordable = self._calculate_lot_sizes(
            equity        = equity,
            entry_price   = effective_entry,
            stop_loss     = sl,
            num_positions = len(valid_tps),
        )

        # Keep only as many TPs as we can afford (TP1 first)
        dropped_budget = len(valid_tps) - affordable
        valid_tps      = valid_tps[:affordable]

        # Small account: single position — aim for a deeper TP instead of TP1
        # (risk is identical regardless of which TP we target with 1 position)
        if affordable == 1 and len(valid_tps) == 1:
            target_idx = self.single_pos_tp_index - 1   # 0-based
            all_valid_tps = [vt for vt in enumerate(signal.take_profits, start=1)
                             if vt[1] > 0 and not self._tp_already_passed(direction, current_price, vt[1])]
            # Rebuild as (tp_index, tp_price, order_type)
            all_classified = []
            for i, tp in all_valid_tps:
                ot = "market" if i <= self.max_market else ("limit" if original_entry else "market")
                all_classified.append((i, tp, ot))
            if target_idx < len(all_classified):
                valid_tps = [all_classified[target_idx]]
                logger.info(
                    f"[EXECUTOR] Single-position budget — using TP{self.single_pos_tp_index} "
                    f"({valid_tps[0][1]}) instead of TP1 for better reward"
                )
            else:
                logger.info(
                    f"[EXECUTOR] Single-position: TP{self.single_pos_tp_index} not available, "
                    f"using deepest available TP"
                )
                valid_tps = [all_classified[-1]] if all_classified else valid_tps

        signal_id = f"CH-{message_id}-{uuid.uuid4().hex[:6].upper()}"
        record = SignalRecord(
            signal_id   = signal_id,
            message_id  = message_id,
            channel     = self.channel,
            raw_text    = signal.raw_text,
            symbol      = symbol,
            direction   = direction,
            entry_type  = signal.entry_type or "market",
            entry_price = effective_entry,
            stop_loss   = sl,
            take_profits= [tp for _, tp, _ in valid_tps],
        )

        # ── Place orders ───────────────────────────────────────────────────────
        placed: list[SignalPosition] = []
        failed = 0

        for tp_index, tp_price, order_type in valid_tps:
            pos = SignalPosition(
                signal_id   = signal_id,
                tp_index    = tp_index,
                tp_price    = tp_price,
                lot_size    = lot_per_pos,
                stop_loss   = sl,
                order_type  = order_type,
            )

            if order_type == "market":
                result = await self._place_market_order(
                    symbol, direction, lot_per_pos, sl, tp_price
                )
            else:
                result = await self._place_limit_order(
                    symbol, direction, lot_per_pos, original_entry, sl, tp_price
                )

            if result and result.get("ticket"):
                pos.ticket      = result["ticket"]
                pos.entry_price = result.get("price", current_price)
                pos.status      = "open"
                pos.opened_at   = datetime.utcnow().isoformat()
                logger.info(
                    f"[EXECUTOR] ✅ TP{tp_index} {order_type} "
                    f"ticket={pos.ticket} lot={lot_per_pos} tp={tp_price}"
                )
            else:
                logger.error(f"[EXECUTOR] ❌ TP{tp_index} order failed: {result}")
                pos.status = "failed"
                failed += 1

            placed.append(pos)

        record.positions = placed
        record.status    = "open" if any(p.status == "open" for p in placed) else "failed"
        self.state.save_signal(record)

        # ── Notification ───────────────────────────────────────────────────────
        open_pos    = [p for p in placed if p.status == "open"]
        market_open = sum(1 for p in open_pos if p.order_type == "market")
        limit_open  = sum(1 for p in open_pos if p.order_type == "limit")
        total_risk  = equity * self.risk_pct / 100.0
        risk_per    = total_risk / max(len(open_pos), 1)

        notes = []
        if dropped_budget:
            notes.append(f"⚠️ {dropped_budget} TP(s) dropped — budget insufficient")
        if failed:
            notes.append(f"⚠️ {failed} order(s) failed — check MT5")
        notes_str = "\n" + "\n".join(notes) if notes else ""

        await self._notify(
            f"📡 <b>CHANNEL SIGNAL EXECUTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{'🟢 BUY' if direction == 'buy' else '🔴 SELL'}  <b>{symbol}</b>\n\n"
            f"<b>📊 Positions: {len(open_pos)} opened</b>\n"
            f"  Market: {market_open}  |  Limit: {limit_open}  |  Failed: {failed}\n\n"
            f"<b>💰 Risk</b>\n"
            f"  Total budget:  <code>${total_risk:.2f}</code>  ({self.risk_pct}% of ${equity:,.0f})\n"
            f"  Per position:  <code>${risk_per:.2f}</code>\n"
            f"  Lot each:      <code>{lot_per_pos:.2f}</code>\n\n"
            f"<b>SL:</b> <code>{sl}</code>\n"
            f"<b>TPs:</b>  {' → '.join(str(p.tp_price) for p in open_pos)}\n"
            f"<b>Signal:</b> <code>{signal_id}</code>"
            f"{notes_str}"
        )

    # ── TP Hit ─────────────────────────────────────────────────────────────────

    async def _handle_tp_hit(self, signal: ParsedSignal):
        tp_num  = signal.tp_number
        records = self.state.get_open_signals(signal.symbol)
        if not records:
            logger.warning(f"[EXECUTOR] TP{tp_num} for {signal.symbol} — no open signal")
            return

        record = records[-1]
        for pos in record.positions:
            if pos.tp_index == tp_num and pos.status == "open":
                self.state.update_position_closed(pos.ticket, "tp_hit")
                break

        remaining = sum(1 for p in record.positions if p.status == "open")

        # Some messages combine "TP hit + move to BE" in one line
        if any("breakeven" in w for w in signal.warnings):
            await self._move_all_sl_to_be(record)

        await self._notify(
            f"🎯 <b>TP{tp_num} Hit</b> — {record.symbol}\n"
            f"Signal: <code>{record.signal_id}</code>\n"
            f"Remaining open: {remaining}"
        )

    # ── Breakeven ──────────────────────────────────────────────────────────────

    async def _handle_breakeven(self, signal: ParsedSignal):
        records = self.state.get_open_signals(signal.symbol)
        if not records:
            logger.warning(f"[EXECUTOR] BE for {signal.symbol} — no open signal")
            return
        await self._move_all_sl_to_be(records[-1])

    async def _move_all_sl_to_be(self, record: SignalRecord):
        """
        Move SL of every open position to its own individual entry price.
        This is "individual breakeven" — not the aggregate signal breakeven.
        """
        modified = 0
        for pos in record.positions:
            if pos.status != "open" or not pos.ticket or not pos.entry_price:
                continue
            success = await self._modify_sl(pos.ticket, pos.entry_price, pos.tp_price)
            if success:
                modified += 1
                logger.info(
                    f"[EXECUTOR] BE ✅ ticket={pos.ticket} "
                    f"SL → entry {pos.entry_price}"
                )
            else:
                logger.error(f"[EXECUTOR] BE ❌ ticket={pos.ticket}")

        await self._notify(
            f"⚖️ <b>Breakeven Set</b> — {record.symbol}\n"
            f"Signal: <code>{record.signal_id}</code>\n"
            f"Positions updated: {modified}\n"
            f"<i>Each position's SL → its own entry price</i>"
        )

    # ── Move SL ────────────────────────────────────────────────────────────────

    async def _handle_move_sl(self, signal: ParsedSignal):
        records = self.state.get_open_signals(signal.symbol)
        if not records or not signal.new_sl:
            return
        record   = records[-1]
        modified = 0
        for pos in record.positions:
            if pos.status != "open" or not pos.ticket:
                continue
            if await self._modify_sl(pos.ticket, signal.new_sl, pos.tp_price):
                modified += 1
        self.state.update_position_sl(record.signal_id, signal.new_sl)
        await self._notify(
            f"📐 <b>SL Moved</b> — {record.symbol}\n"
            f"New SL: <code>{signal.new_sl}</code>  |  Updated: {modified}"
        )

    # ── Close ──────────────────────────────────────────────────────────────────

    async def _handle_close(self, signal: ParsedSignal):
        records = self.state.get_open_signals(signal.symbol)
        if not records:
            logger.warning(f"[EXECUTOR] Close for {signal.symbol} — no open signal")
            return
        await self._close_all_in_signal(records[-1], "channel_close")

    async def _handle_close_all(self, signal: ParsedSignal):
        records = self.state.get_open_signals(signal.symbol)
        for record in records:
            await self._close_all_in_signal(record, "channel_close_all")
        await self._notify(f"🔴 <b>Close All</b> — {len(records)} signal(s) processed")

    async def _close_all_in_signal(self, record: SignalRecord, reason: str):
        closed = 0
        for pos in record.positions:
            if pos.status != "open" or not pos.ticket:
                continue
            result = await self._close_position(pos.ticket, pos.lot_size)
            if result:
                self.state.update_position_closed(pos.ticket, "closed", result.get("profit", 0))
                closed += 1
            else:
                logger.error(f"[EXECUTOR] Failed to close ticket={pos.ticket}")
        self.state.update_signal_status(record.signal_id, "closed")
        await self._notify(
            f"🔴 <b>Signal Closed</b> — {record.symbol}\n"
            f"Positions closed: {closed}  |  Reason: {reason}\n"
            f"Signal: <code>{record.signal_id}</code>"
        )

    # ── Cancel pending limits ──────────────────────────────────────────────────

    async def _handle_cancel(self, signal: ParsedSignal):
        records   = self.state.get_open_signals(signal.symbol)
        cancelled = 0
        for record in records:
            for pos in record.positions:
                if pos.status == "open" and pos.order_type == "limit" and pos.ticket:
                    if await self._delete_pending_order(pos.ticket):
                        self.state.update_position_closed(pos.ticket, "cancelled")
                        cancelled += 1
        if cancelled:
            await self._notify(
                f"❌ <b>Cancelled</b> {cancelled} pending limit(s) for {signal.symbol}"
            )

    # ── Stale limit order ALERT (NOT auto-cancel) ──────────────────────────────

    async def alert_stale_limits(self):
        """
        Called by the background watcher every check_interval_seconds.
        Sends an alert for limit orders older than SIGNAL_LIMIT_ALERT_HOURS.
        Does NOT cancel them — user decides.
        """
        now     = datetime.utcnow()
        alerted = []

        for record in self.state.get_open_signals():
            for pos in record.positions:
                if pos.status != "open" or pos.order_type != "limit" or not pos.opened_at:
                    continue
                try:
                    age_h = (now - datetime.fromisoformat(pos.opened_at)).total_seconds() / 3600
                    if age_h >= self.limit_alert_h:
                        alerted.append((record, pos, age_h))
                except Exception:
                    pass

        if not alerted:
            return

        lines = [
            f"  • <b>{r.symbol}</b> TP{p.tp_index}  "
            f"ticket=<code>{p.ticket}</code>  "
            f"tp=<code>{p.tp_price}</code>  "
            f"age=<b>{h:.1f}h</b>"
            for r, p, h in alerted
        ]

        await self._notify(
            f"⏰ <b>Unfilled Limit Orders — Your Decision Needed</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"These limit orders have been open ≥{self.limit_alert_h:.0f}h:\n\n"
            + "\n".join(lines)
            + "\n\n<b>No automatic action taken.</b>\n"
            "Options:\n"
            "  • Leave open — they'll keep waiting\n"
            "  • Cancel manually in MT5\n"
            "  • Reply with <code>cancel XAUUSD</code> to cancel by symbol"
        )

    # ── SL correction (standalone 🛑 SL XXXX message) ────────────────────────────

    async def _handle_sl_correction(self, signal):
        """Provider sent a corrected SL after a typo in the entry signal."""
        new_sl = signal.new_sl
        if not new_sl:
            return
        records = self.state.get_open_signals(signal.symbol)
        if not records:
            await self._notify(f"⚠️ SL correction received ({new_sl}) but no open signal found.")
            return
        record   = records[-1]
        modified = 0
        for pos in record.positions:
            if pos.status != "open" or not pos.ticket:
                continue
            if await self._modify_sl(pos.ticket, new_sl, pos.tp_price):
                modified += 1
        self.state.update_position_sl(record.signal_id, new_sl)
        await self._notify(
            f"🔧 <b>SL Corrected</b> — {record.symbol}\n"
            f"New SL: <code>{new_sl}</code>  |  Positions updated: {modified}\n"
            f"Signal: <code>{record.signal_id}</code>"
        )

    # ── BE hit (market announcement — info only, no action) ───────────────────

    async def _handle_be_hit(self, signal):
        """Provider announcing the market hit breakeven — informational only."""
        records = self.state.get_open_signals(signal.symbol)
        sig_id  = records[-1].signal_id if records else "n/a"
        remaining = sum(1 for p in records[-1].positions if p.status == "open") if records else 0
        await self._notify(
            f"ℹ️ <b>Breakeven Hit</b> (market event) — {signal.symbol}\n"
            f"Signal: <code>{sig_id}</code>  |  Remaining open: {remaining}\n"
            f"<i>No action taken — SL was already at entry</i>"
        )

    # ── Pre-announcement ("buy gold", "sell now", etc.) ──────────────────────────

    async def _handle_pre_announcement(self, signal):
        """
        Provider announced direction before posting the full signal.
        No trade is opened — we just log and notify so you know one is coming.
        The full RiskY traDE signal will arrive shortly and execute normally.
        """
        emoji = "📢🟢" if signal.direction == "buy" else "📢🔴"
        logger.info(f"[EXECUTOR] Pre-announcement: {signal.direction} — full signal expected shortly")
        await self._notify(
            f"{emoji} <b>Incoming signal</b> — provider announced <b>{signal.direction.upper()}</b>\n"
            f"Waiting for full RiskY traDE signal with SL & TPs...\n"
            f"<i>No trade placed yet.</i>"
        )

        # ── Incomplete entry (ghost template) — skip ──────────────────────────────

    async def _handle_entry_incomplete(self, signal):
        logger.info("[EXECUTOR] Skipping incomplete/ghost entry signal")

        # ── MT5 bridge wrappers ────────────────────────────────────────────────────

    async def _get_equity(self) -> Optional[float]:
        try:
            resp = await self.bridge._send_command({"action": "authenticate"})
            if resp.get("status") == "success":
                return float(resp.get("equity", 0))
        except Exception as e:
            logger.error(f"[EXECUTOR] get_equity failed: {e}")
        return None

    # Since get current price is not in EA we used get_historical with 1 bar to fetch the last price.
    async def _get_current_price(self, symbol: str, direction: str) -> Optional[float]:
        """
        Get current price via get_historical (1 bar).
        The EA does NOT support a 'get_price' action — it only supports
        get_historical, so we fetch the last 1m bar and use its close price.
        """
        try:
            resp = await self.bridge._send_command({
                "action":    "get_historical",
                "symbol":    symbol,
                "timeframe": "M1",
                "count":     1,
            })
            if resp.get("status") == "success" and resp.get("data"):
                close = float(resp["data"][-1][4])  # index 4 = close
                spread = 0.30  # typical XAUUSD spread
                if direction == "buy":
                    return round(close + spread / 2, 2)
                else:
                    return round(close - spread / 2, 2)
        except Exception as e:
            logger.error(f"[EXECUTOR] get_current_price failed: {e}")
        return None

    async def _place_market_order(
        self, symbol: str, direction: str, lot: float, sl: float, tp: float
    ) -> Optional[dict]:
        try:
            return await self.bridge._send_command({
                "action":       "place_order",
                "symbol":       symbol,
                "order_type":   "buy" if direction == "buy" else "sell",
                "lot_size":     lot,
                "stop_loss":    sl,
                "take_profit":  tp,
                "magic_number": self.magic_number,
                "comment":      "ch_signal",
            })
        except Exception as e:
            logger.error(f"[EXECUTOR] market order failed: {e}")
            return None

    async def _place_limit_order(
        self, symbol: str, direction: str, lot: float,
        entry: float, sl: float, tp: float
    ) -> Optional[dict]:
        try:
            return await self.bridge._send_command({
                "action":       "place_order",
                "symbol":       symbol,
                "order_type":   "buy_limit" if direction == "buy" else "sell_limit",
                "price":        entry,
                "lot_size":     lot,
                "stop_loss":    sl,
                "take_profit":  tp,
                "magic_number": self.magic_number,
                "comment":      "ch_signal_limit",
            })
        except Exception as e:
            logger.error(f"[EXECUTOR] limit order failed: {e}")
            return None

    async def _modify_sl(self, ticket: int, new_sl: float, tp: float) -> bool:
        try:
            resp = await self.bridge._send_command({
                "action":      "modify_order",
                "ticket":      ticket,
                "stop_loss":   new_sl,
                "take_profit": tp,
            })
            return resp.get("status") == "success"
        except Exception as e:
            logger.error(f"[EXECUTOR] modify_sl ticket={ticket}: {e}")
            return False

    async def _close_position(self, ticket: int, lot: float) -> Optional[dict]:
        try:
            resp = await self.bridge._send_command({
                "action":   "close_position",
                "ticket":   ticket,
                "lot_size": lot,
            })
            return resp if resp.get("status") == "success" else None
        except Exception as e:
            logger.error(f"[EXECUTOR] close_position ticket={ticket}: {e}")
            return None

    async def _delete_pending_order(self, ticket: int) -> bool:
        try:
            resp = await self.bridge._send_command({"action": "delete_order", "ticket": ticket})
            return resp.get("status") == "success"
        except Exception as e:
            logger.error(f"[EXECUTOR] delete_order ticket={ticket}: {e}")
            return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _tp_already_passed(self, direction: str, price: float, tp: float) -> bool:
        return price >= tp if direction == "buy" else price <= tp

    async def _notify(self, message: str):
        if self.notifier:
            try:
                await self.notifier.send(message)
            except Exception as e:
                logger.error(f"[EXECUTOR] notify failed: {e}")


# ── Standalone sizing test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ.setdefault("SIGNAL_RISK_PERCENT",  "1.0")
    os.environ.setdefault("SIGNAL_CONTRACT_SIZE", "100")
    os.environ.setdefault("SIGNAL_MIN_LOT",       "0.01")
    os.environ.setdefault("SIGNAL_LOT_STEP",      "0.01")
    os.environ.setdefault("SIGNAL_MAX_LOT",       "10.0")

    class _FakeBridge:
        async def _send_command(self, cmd):
            return {}

    ex = SignalExecutor(_FakeBridge(), None)

    print("=== Lot sizing scenarios ===\n")
    scenarios = [
        (200_000, 2345.5, 2330.0, 7, "Rich account, 7 TPs"),
        (50_000,  2345.5, 2330.0, 7, "Mid account, 7 TPs"),
        (10_000,  2345.5, 2330.0, 7, "Small account, 7 TPs — some will be dropped"),
        (5_000,   2345.5, 2330.0, 4, "Small account, 4 TPs"),
        (1_000,   2345.5, 2330.0, 3, "Tiny account, 3 TPs"),
    ]
    for equity, entry, sl, n, label in scenarios:
        lot, count = ex._calculate_lot_sizes(equity, entry, sl, n)
        total_risk     = equity * ex.risk_pct / 100
        actual_per_pos = abs(entry - sl) * ex.contract_size * lot
        print(f"  {label}")
        print(
            f"    equity=${equity:>8,.0f}  budget=${total_risk:.2f}  "
            f"tps_req={n}  tps_affordable={count}  "
            f"lot={lot:.2f}  risk/pos=${actual_per_pos:.2f}\n"
        )