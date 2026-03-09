"""
Telegram Channel Message Fetcher
=================================
Reads recent messages from a private Telegram channel using YOUR USER account
(not a bot). Uses Telethon — the most reliable library for user-account access.

This is the DIAGNOSTIC TOOL — run it first to see exactly how the signal
provider formats their messages so we can build the parser correctly.

Setup:
    pip install telethon

Environment variables required (.env):
    TELEGRAM_API_ID      — from https://my.telegram.org/apps
    TELEGRAM_API_HASH    — from https://my.telegram.org/apps
    TELEGRAM_CHANNEL     — channel username (e.g. @mysignals) or numeric ID

How to get API_ID and API_HASH:
    1. Go to https://my.telegram.org
    2. Log in with your phone number
    3. Click "API development tools"
    4. Create an app (name/platform don't matter)
    5. Copy api_id and api_hash into your .env

First run: Telethon will ask for your phone number and a verification code.
It then saves a session file (telegram_session.session) so you won't be
asked again. Keep this file safe — it's equivalent to being logged in.
"""

import asyncio
import os
import json
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

try:
    from telethon import TelegramClient
    from telethon.tl.types import MessageEntityBold, MessageEntityCode
except ImportError:
    print("ERROR: telethon is not installed. Run: pip install telethon")
    exit(1)

load_dotenv()

API_ID      = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH    = os.getenv("TELEGRAM_API_HASH", "")
SESSION     = "telegram_session"


def _resolve_channel(raw: str):
    """
    Resolve channel value from .env to what Telethon accepts.

    - @username or t.me/...  → string, Telethon resolves it
    - bare positive int like 3516918293 → prepend -100 → -1003516918293
    - already -100xxxxxxxxx  → return as int unchanged
    """
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


CHANNEL = _resolve_channel(os.getenv("TELEGRAM_CHANNEL", ""))


async def fetch_channel_messages(
    limit:       int  = 50,
    save_to_file: bool = True,
    min_length:  int  = 10,     # ignore very short messages (reactions, etc.)
) -> list[dict]:
    """
    Fetch the last `limit` messages from the configured channel.

    Returns a list of dicts, each containing:
        id, date, text, raw_text, has_entities, entity_types
    """
    if not API_ID or not API_HASH:
        print("ERROR: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        print("Get them from: https://my.telegram.org/apps")
        return []

    if not CHANNEL:
        print("ERROR: TELEGRAM_CHANNEL must be set in .env (e.g. @mysignals)")
        return []

    client = TelegramClient(SESSION, API_ID, API_HASH)

    print(f"\nConnecting to Telegram...")
    print("(First run will ask for your phone number and a verification code)\n")

    async with client:
        print(f"✅ Connected as: {(await client.get_me()).username}\n")
        print(f"Fetching last {limit} messages from: {CHANNEL}\n")
        print("=" * 70)

        messages = []
        async for msg in client.iter_messages(CHANNEL, limit=limit):
            if not msg.text:
                continue
            text = msg.text.strip()
            if len(text) < min_length:
                continue

            entity_types = []
            if msg.entities:
                entity_types = list({type(e).__name__ for e in msg.entities})

            entry = {
                "id"          : msg.id,
                "date_utc"    : msg.date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "text"        : text,
                "has_entities": bool(msg.entities),
                "entity_types": entity_types,
                "char_count"  : len(text),
            }
            messages.append(entry)

        print(f"\nFetched {len(messages)} messages with text content.\n")
        print("=" * 70)

        # ── Pretty print to terminal ───────────────────────────────────────────
        for i, m in enumerate(messages, 1):
            print(f"\n{'─' * 60}")
            print(f"MSG #{i}  |  ID: {m['id']}  |  {m['date_utc']} UTC")
            if m['entity_types']:
                print(f"Formatting: {', '.join(m['entity_types'])}")
            print(f"{'─' * 60}")
            print(m['text'])

        print(f"\n{'=' * 70}")
        print(f"Total messages shown: {len(messages)}")

        # ── Save to JSON for analysis ──────────────────────────────────────────
        if save_to_file:
            filename = f"channel_messages_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{CHANNEL}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            print(f"\n📁 All messages saved to: {filename}")
            print("Share this file so the signal parser can be built correctly.\n")

        return messages


async def watch_channel_live(callback=None):
    """
    Live listener — prints every new message as it arrives.
    Use this to see real-time signals without waiting.

    Args:
        callback: Optional async function(message_text: str) to call on each message
    """
    from telethon import events

    client = TelegramClient(SESSION, API_ID, API_HASH)

    print(f"\nStarting live listener on: {CHANNEL}")
    print("Press Ctrl+C to stop.\n")
    print("=" * 70)

    @client.on(events.NewMessage(chats=CHANNEL))
    async def handler(event):
        msg = event.message
        if not msg.text:
            return

        text = msg.text.strip()
        now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n[{now} UTC]  NEW MESSAGE (ID: {msg.id})")
        print("─" * 60)
        print(text)
        print("─" * 60)

        if callback:
            await callback(text)

    async with client:
        await client.run_until_disconnected()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "fetch"

    if mode == "live":
        print("Mode: LIVE — watching for new messages in real time")
        asyncio.run(watch_channel_live())
    else:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
        print(f"Mode: FETCH — retrieving last {limit} messages")
        asyncio.run(fetch_channel_messages(limit=limit))