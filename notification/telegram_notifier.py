"""
Telegram Trade Notifier
=======================
Sends trade notifications to your private Telegram account via a bot.
Uses python-telegram-bot (async). Configured entirely via environment variables.

Environment variables required (.env):
    TELEGRAM_BOT_TOKEN   — your bot token from @BotFather
    TELEGRAM_CHAT_ID     — your personal chat ID (see instructions below)

How to get your CHAT_ID:
    1. Start a chat with your bot (send it /start)
    2. Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
    3. Look for "chat": {"id": <number>} — that number is your CHAT_ID
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
import os

from telegram import Bot
from telegram.error import TelegramError
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends formatted trade notifications to a specific private Telegram account.
    Thread-safe async implementation. Never broadcasts — only sends to the
    configured TELEGRAM_CHAT_ID.
    """

    def __init__(self):
        self.token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self._bot: Optional[Bot] = None

        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set in environment variables.")
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is not set in environment variables.")

    def _get_bot(self) -> Bot:
        if self._bot is None:
            self._bot = Bot(token=self.token)
        return self._bot

    # ── Core send ──────────────────────────────────────────────────────────────

    async def send(self, message: str) -> bool:
        """Send a raw message. Returns True on success."""
        try:
            await self._get_bot().send_message(
                chat_id    = self.chat_id,
                text       = message,
                parse_mode = ParseMode.HTML,
            )
            return True
        except TelegramError as e:
            logger.error(f"[TELEGRAM] Failed to send message: {e}")
            return False

    # ── Trade entry ────────────────────────────────────────────────────────────

    async def notify_trade_entry(
        self,
        symbol:       str,
        direction:    str,        # 'long' or 'short'
        entry_price:  float,
        stop_loss:    float,
        take_profit_1: float,
        take_profit_2: float,
        position_size: float,
        expected_rr:  float,
        confluence_reasons: list[str],
        platform:     str = "MT5",
        trade_id:     str = "",
        ticket:       int = 0,
    ) -> bool:
        arrow    = "🟢 LONG  📈" if direction.lower() == "long" else "🔴 SHORT 📉"
        reasons  = "\n".join(f"  • {r}" for r in confluence_reasons) if confluence_reasons else "  • Signal detected"
        risk_pips = abs(entry_price - stop_loss)

        msg = (
            f"<b>🚀 TRADE OPENED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{arrow}  <b>{symbol}</b>\n\n"
            f"<b>📊 Entry Details</b>\n"
            f"  Entry:   <code>{entry_price:.5f}</code>\n"
            f"  SL:      <code>{stop_loss:.5f}</code>  ({risk_pips:.5f} risk)\n"
            f"  TP1:     <code>{take_profit_1:.5f}</code>\n"
            f"  TP2:     <code>{take_profit_2:.5f}</code>\n"
            f"  Size:    <code>{position_size:.2f} lots</code>\n"
            f"  R:R:     <code>{expected_rr:.1f}R</code>\n\n"
            f"<b>🧠 Why This Trade</b>\n"
            f"{reasons}\n\n"
            f"<b>⚙️ System</b>\n"
            f"  Platform: {platform}\n"
            f"  Ticket:   <code>{ticket}</code>\n"
            f"  Trade ID: <code>{trade_id}</code>\n"
            f"  Time:     {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        return await self.send(msg)

    # ── Trade close ────────────────────────────────────────────────────────────

    async def notify_trade_close(
        self,
        symbol:        str,
        direction:     str,
        entry_price:   float,
        exit_price:    float,
        pnl:           float,
        realized_rr:   float,
        exit_reason:   str,
        duration_minutes: Optional[float] = None,
        trade_id:      str = "",
        ticket:        int = 0,
    ) -> bool:
        is_win = pnl >= 0
        result_emoji = "✅ WIN" if is_win else "❌ LOSS"
        pnl_sign     = "+" if pnl >= 0 else ""
        arrow        = "📈" if direction.lower() == "long" else "📉"

        reason_labels = {
            "stop_loss":    "🛑 Stop Loss hit",
            "take_profit":  "🎯 Take Profit hit",
            "tp1":          "🎯 TP1 hit",
            "tp2":          "🎯 TP2 hit",
            "trailing":     "🔄 Trailing stop",
            "manual":       "👤 Manual close",
            "breakeven":    "⚖️  Breakeven",
            "external":     "📡 Closed externally",
        }
        reason_display = reason_labels.get(exit_reason, f"📌 {exit_reason}")

        duration_str = ""
        if duration_minutes:
            h, m = divmod(int(duration_minutes), 60)
            duration_str = f"\n  Duration: <code>{h}h {m}m</code>"

        msg = (
            f"<b>{result_emoji}  TRADE CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{arrow}  <b>{symbol}</b>  ({direction.upper()})\n\n"
            f"<b>💰 Result</b>\n"
            f"  P&L:     <code>{pnl_sign}{pnl:.2f} USD</code>\n"
            f"  R:R:     <code>{realized_rr:.2f}R</code>\n"
            f"  Reason:  {reason_display}\n\n"
            f"<b>📊 Price</b>\n"
            f"  Entry:   <code>{entry_price:.5f}</code>\n"
            f"  Exit:    <code>{exit_price:.5f}</code>{duration_str}\n\n"
            f"<b>⚙️ Reference</b>\n"
            f"  Ticket:   <code>{ticket}</code>\n"
            f"  Trade ID: <code>{trade_id}</code>\n"
            f"  Time:     {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        return await self.send(msg)

    # ── Risk events ────────────────────────────────────────────────────────────

    async def notify_drawdown_limit(
        self,
        drawdown_pct: float,
        limit_pct:    float,
        equity:       float,
    ) -> bool:
        msg = (
            f"<b>🚨 DRAWDOWN LIMIT HIT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"  Drawdown: <code>{drawdown_pct:.2f}%</code>  (limit: {limit_pct:.1f}%)\n"
            f"  Equity:   <code>${equity:,.2f}</code>\n"
            f"  Action:   Trading HALTED for today\n"
            f"  Time:     {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
            f"⚠️ Review your positions and check the dashboard."
        )
        return await self.send(msg)

    async def notify_system_event(self, event: str, detail: str = "") -> bool:
        icons = {
            "startup":  "🟢",
            "shutdown": "🔴",
            "error":    "⚠️",
            "warning":  "⚡",
        }
        icon = icons.get(event.lower(), "📌")
        msg = (
            f"<b>{icon} SYSTEM: {event.upper()}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{detail}\n"
            f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        return await self.send(msg)

    # ── Convenience sync wrapper (for non-async callers) ──────────────────────

    def send_sync(self, message: str) -> bool:
        """Blocking send for use outside async context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as task if already in an event loop
                asyncio.ensure_future(self.send(message))
                return True
            else:
                return loop.run_until_complete(self.send(message))
        except Exception as e:
            logger.error(f"[TELEGRAM] send_sync failed: {e}")
            return False


# ── Standalone test ────────────────────────────────────────────────────────────

async def _test():
    """Run this directly to verify your bot token and chat ID work."""
    from dotenv import load_dotenv
    load_dotenv()

    notifier = TelegramNotifier()

    print("Sending test trade entry notification...")
    await notifier.notify_trade_entry(
        symbol         = "XAUUSD",
        direction      = "long",
        entry_price    = 2345.500,
        stop_loss      = 2338.200,
        take_profit_1  = 2360.100,
        take_profit_2  = 2378.500,
        position_size  = 0.10,
        expected_rr    = 2.0,
        confluence_reasons = [
            "1H bullish market structure (HH/HL confirmed)",
            "RSI oversold recovery (28 → 42)",
            "Price bounced off 200 EMA support",
            "SuperTrend bullish on 1H and 15m",
            "High-volume bullish engulfing candle",
        ],
        trade_id = "TRD-20260308-001",
        ticket   = 123456789,
    )

    await asyncio.sleep(1)

    print("Sending test trade close notification...")
    await notifier.notify_trade_close(
        symbol           = "XAUUSD",
        direction        = "long",
        entry_price      = 2345.500,
        exit_price       = 2360.100,
        pnl              = 146.00,
        realized_rr      = 2.0,
        exit_reason      = "tp1",
        duration_minutes = 87,
        trade_id         = "TRD-20260308-001",
        ticket           = 123456789,
    )

    print("Done. Check your Telegram.")


if __name__ == "__main__":
    asyncio.run(_test())