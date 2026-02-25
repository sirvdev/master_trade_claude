"""
utils/market_hours.py

Fetches real trade session schedules directly from the MT5 broker via the
file bridge, then caches them in memory for the lifetime of the process.

MT5 provides SymbolInfoSessionTrade() per day-of-week which returns the
exact windows the broker has configured — no hardcoding needed.

Day-of-week mapping:
  MT5:    SUNDAY=0, MONDAY=1, TUESDAY=2 ... SATURDAY=6
  Python: Monday=0, Tuesday=1 ...           Sunday=6

The conversion is handled internally. All public methods use Python weekday
convention (Monday=0 ... Sunday=6) so callers never have to think about it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Fallback schedule (used only if broker fetch fails) ───────────────────────
# Represents the most common broker schedule for each asset class.
# Stored in MT5 day format (SUNDAY=0 .. SATURDAY=6).
_FALLBACK_SESSIONS: Dict[str, List[dict]] = {
    # Gold / Silver: Mon–Fri nearly 24h, closed Sat, opens Sun 22:00 UTC
    "XAUUSD": [
        {"day": 0, "from": 79200, "to": 86399},  # Sun  22:00–23:59
        {"day": 1, "from":     0, "to": 86399},  # Mon  00:00–23:59
        {"day": 2, "from":     0, "to": 86399},  # Tue
        {"day": 3, "from":     0, "to": 86399},  # Wed
        {"day": 4, "from":     0, "to": 86399},  # Thu
        {"day": 5, "from":     0, "to": 79200},  # Fri  00:00–22:00
    ],
    "XAGUSD": [
        {"day": 0, "from": 79200, "to": 86399},
        {"day": 1, "from":     0, "to": 86399},
        {"day": 2, "from":     0, "to": 86399},
        {"day": 3, "from":     0, "to": 86399},
        {"day": 4, "from":     0, "to": 86399},
        {"day": 5, "from":     0, "to": 79200},
    ],
    # Crypto: 24/7
    "BTCUSD":  [{"day": d, "from": 0, "to": 86399} for d in range(7)],
    "ETHUSD":  [{"day": d, "from": 0, "to": 86399} for d in range(7)],
    "BTCUSDT": [{"day": d, "from": 0, "to": 86399} for d in range(7)],
    "ETHUSDT": [{"day": d, "from": 0, "to": 86399} for d in range(7)],
}

# MT5 Sunday=0 → Python Sunday=6 conversion table
# mt5_day -> python_weekday
_MT5_TO_PYTHON = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
# Python weekday -> MT5 day
_PYTHON_TO_MT5 = {v: k for k, v in _MT5_TO_PYTHON.items()}


def _mt5_symbol(symbol: str) -> str:
    """Normalise symbol to MT5 format: 'XAU/USD' → 'XAUUSD'."""
    return symbol.replace("/", "").upper()


class MarketHoursChecker:
    """
    Checks whether a symbol's market is open, fetching session data from
    the MT5 broker on first use and caching it for the process lifetime.

    Usage:
        checker = MarketHoursChecker(mt5_bridge)
        await checker.prefetch_all(["XAU/USD", "BTC/USD"])

        if not checker.is_open("XAU/USD"):
            secs = checker.seconds_until_open("XAU/USD")
    """

    def __init__(self, mt5_bridge):
        """
        Args:
            mt5_bridge: MT5FileBridge instance (must already be connected).
        """
        self._bridge = mt5_bridge
        # {mt5_symbol -> list of {day (python weekday 0=Mon), from_sec, to_sec}}
        self._sessions: Dict[str, List[dict]] = {}
        self._fetch_errors: Dict[str, str] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def prefetch_all(self, symbols: List[str]) -> None:
        """
        Fetch and cache session schedules for all given symbols at startup.
        Call this once from TradingSystem.start() before the trading loop.
        """
        for symbol in symbols:
            await self._ensure_fetched(symbol)

    def is_open(self, symbol: str, now: datetime = None) -> bool:
        """
        Return True if the market for this symbol is currently open.
        Uses UTC time. Fetches session data synchronously if not yet cached
        (only happens if prefetch_all was not called first).
        """
        if now is None:
            now = datetime.utcnow()

        sessions = self._get_cached_sessions(symbol)
        if not sessions:
            return True  # fail open — don't block trading on missing data

        python_weekday = now.weekday()  # 0=Mon, 6=Sun
        current_sec = now.hour * 3600 + now.minute * 60 + now.second

        for s in sessions:
            if s["day"] == python_weekday:
                if s["from_sec"] <= current_sec <= s["to_sec"]:
                    return True

        return False

    def seconds_until_open(self, symbol: str, now: datetime = None) -> int:
        """
        Return seconds until next session open. Returns 0 if already open.
        Searches up to 8 days ahead.
        """
        if now is None:
            now = datetime.utcnow()

        if self.is_open(symbol, now):
            return 0

        sessions = self._get_cached_sessions(symbol)
        if not sessions:
            return 0  # fail open

        # Index sessions by python weekday for fast lookup
        by_day: Dict[int, List[Tuple[int, int]]] = {}
        for s in sessions:
            by_day.setdefault(s["day"], []).append((s["from_sec"], s["to_sec"]))

        best: Optional[int] = None

        for day_offset in range(1, 9):
            candidate_dt = now + timedelta(days=day_offset)
            candidate_dt = candidate_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            candidate_weekday = candidate_dt.weekday()

            for (from_sec, _) in by_day.get(candidate_weekday, []):
                open_dt = candidate_dt + timedelta(seconds=from_sec)
                if open_dt <= now:
                    continue
                secs = int((open_dt - now).total_seconds())
                if best is None or secs < best:
                    best = secs

        # Also check later today (session starts after current time same day)
        today_weekday = now.weekday()
        current_sec = now.hour * 3600 + now.minute * 60 + now.second
        for (from_sec, _) in by_day.get(today_weekday, []):
            if from_sec > current_sec:
                secs = from_sec - current_sec
                if best is None or secs < best:
                    best = secs

        return best if best is not None else 24 * 3600

    def next_open_str(self, symbol: str, now: datetime = None) -> str:
        """Human-readable string of when the next session opens (UTC)."""
        if now is None:
            now = datetime.utcnow()
        secs = self.seconds_until_open(symbol, now)
        if secs == 0:
            return "now"
        hours, rem = divmod(secs, 3600)
        mins = rem // 60
        opens_at = now + timedelta(seconds=secs)
        return f"{opens_at.strftime('%Y-%m-%d %H:%M')} UTC ({hours}h {mins}m)"

    def session_summary(self, symbol: str) -> str:
        """Return a one-line human-readable summary of the symbol's sessions."""
        sessions = self._get_cached_sessions(symbol)
        if not sessions:
            return f"{symbol}: session data unavailable (assumed 24/7)"

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        parts = []
        for s in sessions:
            d = day_names[s["day"]]
            fh, fm = divmod(s["from_sec"] // 60, 60)
            th, tm = divmod(s["to_sec"]  // 60, 60)
            parts.append(f"{d} {fh:02d}:{fm:02d}–{th:02d}:{tm:02d}")
        return f"{symbol}: " + ", ".join(parts)

    async def is_open_by_price(
        self,
        symbol: str,
        mt5_bridge,
        timeframe: str = "1H",
        stale_multiplier: float = 2.5,
    ) -> bool:
        """
        Secondary market-open check using price data freshness.

        Asks MT5 for the last 2 bars and checks whether the most recent bar's
        timestamp is recent enough. If the last bar is older than
        (timeframe_seconds × stale_multiplier), the market is very likely closed
        (no new bars are being formed by the broker).

        This works with ANY EA version — it only uses 'get_historical' which
        every version of PythonFileBridge supports. It does NOT require the
        new get_symbol_sessions action.

        Args:
            symbol:           Trading symbol, e.g. 'XAU/USD'
            mt5_bridge:       Connected MT5FileBridge instance
            timeframe:        Timeframe to inspect (default '1H')
            stale_multiplier: How many full periods old = "closed"
                              2.5 means: if the last 1H bar is >2.5 hours old,
                              market is considered closed.

        Returns:
            True  — market appears open (price is fresh), or on any error.
            False — last bar is stale, market is likely closed.
        """
        _TF_SECONDS = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1H': 3600, '4H': 14400, '1D': 86400,
            # MT5 style aliases
            'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
            'H1': 3600, 'H4': 14400, 'D1': 86400,
        }
        tf_secs = _TF_SECONDS.get(timeframe, 3600)
        mt5_sym = _mt5_symbol(symbol)

        try:
            response = await mt5_bridge._send_command(
                {
                    "action":    "get_historical",
                    "symbol":    mt5_sym,
                    "timeframe": timeframe,
                    "count":     2,
                },
                timeout=10.0,
            )

            if response.get("status") != "success":
                logger.debug(
                    f"[MarketHours] is_open_by_price: get_historical failed for "
                    f"{symbol}: {response.get('error')} — assuming open"
                )
                return True  # fail open

            data = response.get("data", [])
            if not data:
                return True  # fail open — no data means bridge issue, not closed market

            # data rows: [timestamp_unix, open, high, low, close, volume]
            last_bar_ts = int(data[-1][0])
            now_ts = int(datetime.utcnow().timestamp())
            age_secs = now_ts - last_bar_ts

            stale_threshold = tf_secs * stale_multiplier
            is_fresh = age_secs < stale_threshold

            if not is_fresh:
                h, m = divmod(age_secs // 60, 60) if age_secs < 3600 else (age_secs // 3600, (age_secs % 3600) // 60)
                logger.info(
                    f"[MarketHours] {symbol}: last {timeframe} bar is "
                    f"{age_secs // 3600}h {(age_secs % 3600) // 60}m old "
                    f"(threshold {stale_threshold / 3600:.1f}h) — market likely closed"
                )
            return is_fresh

        except Exception as e:
            logger.debug(
                f"[MarketHours] is_open_by_price error for {symbol}: {e} — assuming open"
            )
            return True  # fail open

    # ── Internal fetch / cache ─────────────────────────────────────────────────

    async def _ensure_fetched(self, symbol: str) -> None:
        """Fetch sessions from broker if not already in cache."""
        mt5_sym = _mt5_symbol(symbol)
        if mt5_sym in self._sessions:
            return
        if mt5_sym in self._fetch_errors:
            return  # already tried and failed, don't spam the EA

        await self._fetch_from_broker(symbol, mt5_sym)

    async def _fetch_from_broker(self, symbol: str, mt5_sym: str) -> None:
        """
        Call get_symbol_sessions on the MT5 EA and parse the response.
        Falls back to _FALLBACK_SESSIONS if the call fails.
        """
        try:
            response = await self._bridge._send_command(
                {"action": "get_symbol_sessions", "symbol": mt5_sym},
                timeout=15.0
            )

            if response.get("status") != "success":
                raise ValueError(response.get("error", "unknown error"))

            raw_sessions = response.get("sessions", [])
            if not raw_sessions:
                raise ValueError("empty sessions list in response")

            # Convert MT5 day (Sun=0) → Python weekday (Mon=0)
            parsed = []
            for s in raw_sessions:
                py_day = _MT5_TO_PYTHON[int(s["day"])]
                parsed.append({
                    "day"     : py_day,
                    "from_sec": int(s["from"]),
                    "to_sec"  : int(s["to"]),
                })

            self._sessions[mt5_sym] = parsed
            logger.info(
                f"[MarketHours] {symbol}: fetched {len(parsed)} session windows "
                f"from broker. {self.session_summary(symbol)}"
            )

        except Exception as e:
            logger.warning(
                f"[MarketHours] Could not fetch sessions for {symbol} from broker: {e}. "
                f"Falling back to built-in schedule."
            )
            self._fetch_errors[mt5_sym] = str(e)
            self._load_fallback(symbol, mt5_sym)

    def _load_fallback(self, symbol: str, mt5_sym: str) -> None:
        """Load hardcoded fallback sessions and convert to Python weekday format."""
        raw = _FALLBACK_SESSIONS.get(mt5_sym)
        if not raw:
            logger.warning(
                f"[MarketHours] No fallback schedule for {symbol}. "
                f"Assuming 24/7 (will never block trading)."
            )
            self._sessions[mt5_sym] = [
                {"day": d, "from_sec": 0, "to_sec": 86399} for d in range(7)
            ]
            return

        parsed = []
        for s in raw:
            py_day = _MT5_TO_PYTHON[s["day"]]
            parsed.append({
                "day"     : py_day,
                "from_sec": s["from"],
                "to_sec"  : s["to"],
            })
        self._sessions[mt5_sym] = parsed
        logger.info(
            f"[MarketHours] {symbol}: loaded fallback schedule. "
            f"{self.session_summary(symbol)}"
        )

    def _get_cached_sessions(self, symbol: str) -> List[dict]:
        """Return cached sessions, or empty list if unavailable."""
        mt5_sym = _mt5_symbol(symbol)
        return self._sessions.get(mt5_sym, [])