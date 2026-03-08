"""
channel_signal_bot.py
=====================
Main entry point for the channel signal follower.

Listens to the configured Telegram channel using your USER account (Telethon),
parses every message, and routes to SignalExecutor for MT5 execution.

Run standalone:
    python channel_signal_bot.py

Or integrate into the main trading system:
    from notifications.channel_signal_bot import ChannelSignalBot
    bot = ChannelSignalBot(mt5_bridge, notifier)
    asyncio.create_task(bot.start())

Environment variables (.env):
    TELEGRAM_API_ID              — from https://my.telegram.org/apps
    TELEGRAM_API_HASH            — from https://my.telegram.org/apps
    TELEGRAM_SIGNAL_CHANNEL      — channel username/ID (e.g. @mysignals)
    TELEGRAM_SESSION_NAME        channel_session  (optional override)

    SIGNAL_LOT_SIZE              0.01
    SIGNAL_MAGIC_NUMBER          234567
    SIGNAL_MAX_MARKET_ORDERS     2
    SIGNAL_PRICE_TOLERANCE_PIPS  50
    SIGNAL_LIMIT_TIMEOUT_HOURS   24
    SIGNAL_DEFAULT_SYMBOL        XAUUSD
    SIGNAL_MIN_CONFIDENCE        0.4   — reject signals below this threshold
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from telethon import TelegramClient, events
    from telethon.tl.types import Message
except ImportError:
    print("ERROR: telethon not installed. Run: pip install telethon")
    sys.exit(1)

# Allow running from project root or notifications/ folder
sys.path.insert(0, str(Path(__file__).parent))

from signal_parser import parse_signal, ParsedSignal
from signal_state import SignalStateManager
from signal_executor import SignalExecutor

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_ID        = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH      = os.getenv("TELEGRAM_API_HASH", "")
def _resolve_channel(raw: str):
    raw = raw.strip()
    if not raw:
        return raw
    if raw.startswith("@") or "t.me" in raw or "https" in raw:
        return raw
    if raw.lstrip("-").isdigit():
        n = int(raw)
        if n > 0:
            return int(f"-100{n}")
        if str(n).startswith("-100"):
            return n
        return int(f"-100{abs(n)}")
    return raw

CHANNEL = _resolve_channel(os.getenv("TELEGRAM_SIGNAL_CHANNEL", ""))
SESSION_NAME  = os.getenv("TELEGRAM_SESSION_NAME", "channel_session")
MIN_CONFIDENCE = float(os.getenv("SIGNAL_MIN_CONFIDENCE", "0.4"))


class ChannelSignalBot:
    """
    Listens to the signal channel and executes trades via MT5.

    Handles:
      - New entry messages (full recursive position set)
      - Reply messages ("close this", "move sl to be")
      - Inline modify messages ("TP1 hit", "breakeven", "close all")
      - Reconnection on Telethon disconnect
    """

    def __init__(
        self,
        mt5_bridge=None,
        notifier=None,
        db_path: str = "data/trading.db",
    ):
        self.mt5_bridge = mt5_bridge
        self.notifier   = notifier
        self.state      = SignalStateManager(db_path)
        self.executor   = SignalExecutor(
            mt5_bridge    = mt5_bridge,
            state_manager = self.state,
            notifier      = notifier,
            channel       = CHANNEL,
        )
        self._client: TelegramClient | None = None
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        """Start the channel listener. Runs until stop() is called."""
        if not API_ID or not API_HASH:
            logger.error("[BOT] TELEGRAM_API_ID / TELEGRAM_API_HASH not set.")
            return
        if not CHANNEL:
            logger.error("[BOT] TELEGRAM_SIGNAL_CHANNEL not set.")
            return

        self._running = True
        logger.info(f"[BOT] Starting channel listener on {CHANNEL}")

        while self._running:
            try:
                await self._run_client()
            except Exception as e:
                logger.error(f"[BOT] Client error: {e} — reconnecting in 30s")
                await asyncio.sleep(30)

    async def stop(self):
        self._running = False
        if self._client:
            await self._client.disconnect()
        logger.info("[BOT] Stopped.")

    # ── Client ─────────────────────────────────────────────────────────────────

    async def _run_client(self):
        self._client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

        @self._client.on(events.NewMessage(chats=CHANNEL))
        async def on_message(event: events.NewMessage.Event):
            await self._handle_message(event)

        async with self._client:
            me = await self._client.get_me()
            logger.info(f"[BOT] ✅ Connected as @{me.username} — watching {CHANNEL}")

            if self.notifier:
                await self.notifier.send(
                    f"🟢 <b>Channel Bot Active</b>\n"
                    f"Watching: <code>{CHANNEL}</code>\n"
                    f"Account: @{me.username}"
                )

            await self._client.run_until_disconnected()

    # ── Message handler ────────────────────────────────────────────────────────

    async def _handle_message(self, event: events.NewMessage.Event):
        msg: Message = event.message
        if not msg.text:
            return

        text = msg.text.strip()
        if len(text) < 3:
            return

        now = datetime.utcnow().strftime("%H:%M:%S")
        logger.info(f"[BOT] [{now}] New message (id={msg.id}): {text[:80]!r}")

        # ── Resolve reply context ──────────────────────────────────────────────
        is_reply   = False
        reply_text = ""

        if msg.reply_to_msg_id:
            is_reply = True
            try:
                replied_msg = await event.get_reply_message()
                if replied_msg and replied_msg.text:
                    reply_text = replied_msg.text.strip()
                    logger.debug(f"[BOT] Reply to: {reply_text[:60]!r}")
            except Exception as e:
                logger.warning(f"[BOT] Could not fetch replied-to message: {e}")

        # ── Parse ──────────────────────────────────────────────────────────────
        signal = parse_signal(text, is_reply=is_reply, reply_text=reply_text)

        logger.info(
            f"[BOT] Parsed → type={signal.signal_type} "
            f"symbol={signal.symbol} direction={signal.direction} "
            f"confidence={signal.confidence:.1f}"
            + (f" WARNINGS={signal.warnings}" if signal.warnings else "")
        )

        # ── Confidence gate ────────────────────────────────────────────────────
        if signal.signal_type == "entry" and signal.confidence < MIN_CONFIDENCE:
            logger.warning(
                f"[BOT] Entry signal below confidence threshold "
                f"({signal.confidence:.1f} < {MIN_CONFIDENCE}) — skipping"
            )
            if self.notifier:
                await self.notifier.send(
                    f"⚠️ Low-confidence signal skipped\n"
                    f"Text: <code>{text[:120]}</code>\n"
                    f"Confidence: {signal.confidence:.1f}"
                )
            return

        # ── Execute ────────────────────────────────────────────────────────────
        if self.mt5_bridge:
            await self.executor.execute(signal, message_id=msg.id)
        else:
            # Dry-run mode: just log and notify
            logger.info(f"[BOT] DRY RUN — would execute: {signal}")
            if self.notifier:
                await self.notifier.send(
                    f"🔍 <b>DRY RUN — Signal detected</b>\n"
                    f"Type: <code>{signal.signal_type}</code>\n"
                    f"Symbol: <code>{signal.symbol}</code>\n"
                    f"Direction: <code>{signal.direction}</code>\n"
                    f"TPs: <code>{signal.take_profits}</code>\n"
                    f"SL: <code>{signal.stop_loss}</code>\n"
                    f"Confidence: <code>{signal.confidence:.1f}</code>"
                )


# ── Limit order expiry watcher ─────────────────────────────────────────────────

async def _limit_order_alert_watcher(
    executor: SignalExecutor,
    check_interval_seconds: int = 600,
):
    """
    Background task: alerts the user about unfilled limit orders older than
    SIGNAL_LIMIT_ALERT_HOURS. Does NOT auto-cancel — user decides.
    Each ticket is alerted only once to avoid notification spam.
    """
    alerted_tickets: set[int] = set()

    while True:
        await asyncio.sleep(check_interval_seconds)
        try:
            now   = datetime.utcnow()
            fresh = []
            for record in executor.state.get_open_signals():
                for pos in record.positions:
                    if (pos.status != "open"
                            or pos.order_type != "limit"
                            or not pos.opened_at
                            or pos.ticket in alerted_tickets):
                        continue
                    try:
                        age_h = (now - datetime.fromisoformat(pos.opened_at)).total_seconds() / 3600
                        if age_h >= executor.limit_alert_h:
                            fresh.append((record, pos, age_h))
                            alerted_tickets.add(pos.ticket)
                    except Exception:
                        pass

            if not fresh:
                continue

            lines = [
                f"  • <b>{r.symbol}</b> TP{p.tp_index}  "
                f"ticket=<code>{p.ticket}</code>  "
                f"tp=<code>{p.tp_price}</code>  age=<b>{h:.1f}h</b>"
                for r, p, h in fresh
            ]
            await executor._notify(
                f"⏰ <b>Unfilled Limit Orders — Your Decision Needed</b>\n"
                f"━" * 20 + "\n"
                f"These limit orders have been open ≥{executor.limit_alert_h:.0f}h:\n\n"
                + "\n".join(lines)
                + "\n\n<b>No automatic action.</b>\n"
                "  • Leave open — they will keep waiting\n"
                "  • Cancel manually in MT5\n"
                "  • Reply <code>cancel XAUUSD</code> to cancel by symbol"
            )
        except Exception as e:
            logger.error(f"[ALERT_WATCHER] Error: {e}")


# ── Standalone entry point ─────────────────────────────────────────────────────

async def _standalone():
    """
    Run the bot in dry-run mode without MT5 (for testing the listener & parser).
    Prints all parsed signals to console and sends Telegram notifications.
    """
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    # Try to load notifier if token is available
    notifier = None
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from notification.telegram_notifier import TelegramNotifier
            notifier = TelegramNotifier()
            logger.info("[MAIN] Telegram notifier loaded.")
        except Exception as e:
            logger.warning(f"[MAIN] Could not load notifier: {e}")

    bot = ChannelSignalBot(
        mt5_bridge = None,      # dry-run — no execution
        notifier   = notifier,
        db_path    = "data/trading.db",
    )

    state = bot.state
    expiry_task = asyncio.create_task(
        _limit_order_alert_watcher(bot.executor)
    )

    try:
        await bot.start()
    finally:
        expiry_task.cancel()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "listen"

    if mode == "test-parser":
        # Quick parser test without Telegram connection
        from signal_parser import parse_signal
        test_msgs = [
            "Buy XAUUSD now\nSL: 2320\nTP1: 2345\nTP2: 2350\nTP3: 2355\nTP4: 2360",
            "TP2 hit, move sl to breakeven",
            "close this",
            "close all",
            "move sl to brakeeven",  # typo
        ]
        for m in test_msgs:
            s = parse_signal(m)
            print(f"\n{m!r}\n  → {s.signal_type} | {s.symbol} | {s.direction} | TPs={s.take_profits}")
    else:
        print(f"Starting channel listener in {'DRY RUN' if not os.getenv('MT5_BRIDGE_HOST') else 'LIVE'} mode")
        asyncio.run(_standalone())