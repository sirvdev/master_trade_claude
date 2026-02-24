"""
Market hours checker — all times in UTC.
Determines if a symbol's market is currently open and calculates
seconds until the next open session.
"""

from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ── Default schedules (UTC) ───────────────────────────────────────────────────
# Format: list of session dicts with:
#   days:  list of weekday ints (0=Mon, 6=Sun)
#   open:  "HH:MM"  (UTC)
#   close: "HH:MM"  (UTC)  — "24:00" means midnight end-of-day
#
# Gold/Silver trade nearly 24h on weekdays with a small daily break and
# a full weekend closure (Friday close to Sunday open).
#
# Weekday sessions: Mon-Thu open Sunday 22:00 → Friday 22:00 continuously,
# with a daily maintenance gap 21:55–22:05 on most brokers. We model it as
# five separate same-day sessions to keep the logic simple; the tiny gap
# is not worth complicating the schedule.

DEFAULT_SCHEDULES = {
    # Gold: 24h Mon–Fri, closed Sat, opens ~22:00 UTC Sun
    "XAU/USD": [
        {"days": [0, 1, 2, 3], "open": "00:00", "close": "23:59"},  # Mon–Thu full day
        {"days": [4],          "open": "00:00", "close": "22:00"},  # Fri closes 22:00
        {"days": [6],          "open": "22:00", "close": "23:59"},  # Sun opens 22:00
    ],
    "XAUUSD": [
        {"days": [0, 1, 2, 3], "open": "00:00", "close": "23:59"},
        {"days": [4],          "open": "00:00", "close": "22:00"},
        {"days": [6],          "open": "22:00", "close": "23:59"},
    ],
    # Silver: same as gold
    "XAG/USD": [
        {"days": [0, 1, 2, 3], "open": "00:00", "close": "23:59"},
        {"days": [4],          "open": "00:00", "close": "22:00"},
        {"days": [6],          "open": "22:00", "close": "23:59"},
    ],
    "XAGUSD": [
        {"days": [0, 1, 2, 3], "open": "00:00", "close": "23:59"},
        {"days": [4],          "open": "00:00", "close": "22:00"},
        {"days": [6],          "open": "22:00", "close": "23:59"},
    ],
    # Crypto: 24/7
    "BTC/USD":  [{"days": [0,1,2,3,4,5,6], "open": "00:00", "close": "23:59"}],
    "BTC/USDT": [{"days": [0,1,2,3,4,5,6], "open": "00:00", "close": "23:59"}],
    "ETH/USD":  [{"days": [0,1,2,3,4,5,6], "open": "00:00", "close": "23:59"}],
    "ETH/USDT": [{"days": [0,1,2,3,4,5,6], "open": "00:00", "close": "23:59"}],
    # Forex (default 24h weekdays)
    "EUR/USD": [
        {"days": [0,1,2,3],  "open": "00:00", "close": "23:59"},
        {"days": [4],        "open": "00:00", "close": "22:00"},
        {"days": [6],        "open": "22:00", "close": "23:59"},
    ],
}


def _parse_hm(hm: str):
    """Parse 'HH:MM' into (hour, minute)."""
    h, m = hm.split(":")
    return int(h), int(m)


class MarketHoursChecker:
    """
    Checks whether a symbol's market is open at a given UTC datetime.
    Sessions are defined in config per symbol or fall back to DEFAULT_SCHEDULES.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full system config dict. Per-symbol sessions can be defined under:
              symbols:
                XAU/USD:
                  trading_hours:
                    - {days: [0,1,2,3,4], open: "00:00", close: "23:59"}
        """
        self.config = config
        self._cache: dict = {}   # symbol -> parsed sessions list

    def _get_sessions(self, symbol: str) -> list:
        """Return session list for symbol (config overrides defaults)."""
        if symbol in self._cache:
            return self._cache[symbol]

        sym_cfg = self.config.get("symbols", {}).get(symbol, {})
        sessions = sym_cfg.get("trading_hours")

        if not sessions:
            # Try stripping slash
            bare = symbol.replace("/", "")
            sessions = DEFAULT_SCHEDULES.get(symbol) or DEFAULT_SCHEDULES.get(bare)

        if not sessions:
            # Unknown symbol — assume 24/7 (crypto-like, fail open)
            logger.warning(
                f"[MarketHours] No schedule for '{symbol}' — assuming 24/7."
            )
            sessions = [{"days": [0,1,2,3,4,5,6], "open": "00:00", "close": "23:59"}]

        self._cache[symbol] = sessions
        return sessions

    def is_open(self, symbol: str, now: datetime = None) -> bool:
        """Return True if symbol's market is currently open (UTC)."""
        if now is None:
            now = datetime.utcnow()

        weekday = now.weekday()   # 0=Mon, 6=Sun
        current_minutes = now.hour * 60 + now.minute

        for session in self._get_sessions(symbol):
            if weekday not in session["days"]:
                continue
            oh, om = _parse_hm(session["open"])
            ch, cm = _parse_hm(session["close"])
            open_min  = oh * 60 + om
            close_min = ch * 60 + cm
            if open_min <= current_minutes <= close_min:
                return True

        return False

    def seconds_until_open(self, symbol: str, now: datetime = None) -> int:
        """
        Return seconds until the next session open for symbol.
        Returns 0 if market is already open.
        Searches up to 7 days ahead.
        """
        if now is None:
            now = datetime.utcnow()

        if self.is_open(symbol, now):
            return 0

        # Walk minute-by-minute is too slow — instead walk forward by day
        # and check each session's next open time.
        sessions = self._get_sessions(symbol)

        best_seconds = None

        for day_offset in range(8):   # search next 7 days
            candidate_date = (now + timedelta(days=day_offset)).date()
            candidate_weekday = candidate_date.weekday()

            for session in sessions:
                if candidate_weekday not in session["days"]:
                    continue

                oh, om = _parse_hm(session["open"])
                candidate_open = datetime(
                    candidate_date.year, candidate_date.month, candidate_date.day,
                    oh, om, 0
                )

                if candidate_open <= now:
                    continue   # this open is in the past

                secs = int((candidate_open - now).total_seconds())
                if best_seconds is None or secs < best_seconds:
                    best_seconds = secs

        return best_seconds if best_seconds is not None else 24 * 3600

    def next_open_str(self, symbol: str, now: datetime = None) -> str:
        """Human-readable string of when the next session opens."""
        if now is None:
            now = datetime.utcnow()
        secs = self.seconds_until_open(symbol, now)
        if secs == 0:
            return "now"
        hours, rem = divmod(secs, 3600)
        mins = rem // 60
        opens_at = now + timedelta(seconds=secs)
        return f"{opens_at.strftime('%Y-%m-%d %H:%M')} UTC ({hours}h {mins}m from now)"