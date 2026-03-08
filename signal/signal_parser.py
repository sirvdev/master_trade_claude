"""
signal_parser.py
================
Parser built from real channel messages (Lion Trading Academy / AB Marshal).

Entry format (ALWAYS starts with 'RiskY traDE'):
    RiskY traDE ☠️
    👉🏾BUY/SELL/buy/sell XAUUSD now
    🛑 SL 5340
    ✅ TP 5362
    ✅ TP 5368
    ...
    ✅ TP open         ← last position: runner with no TP (tp_price=0)
    disclaimer: ...

signal_type values:
    'entry'            — full signal with direction/SL/TPs — EXECUTE
    'entry_incomplete' — RiskY traDE but blank SL or no valid TPs (skip)
    'pre_announcement' — "buy gold", "sell now" etc. before full signal
                         STORE direction, wait for full RiskY traDE to follow
    'tp_hit'           — TP hit announcement
    'breakeven'        — instruction to move SL to entry
    'be_hit'           — market announcing BE was hit (info only, no action)
    'sl_correction'    — standalone 🛑 SL XXXX correcting last signal's SL
    'close'            — close most recent / last position
    'close_all'        — close every open position
    'unknown'          — nothing matched (chatter/admin messages)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── TP ordinal words → numbers ─────────────────────────────────────────────────
_TP_ORDINALS = {
    'first': 1, 'second': 2,  'third': 3,
    '2nd':   2, '3rd':    3,  '4th':  4,
    '4rd':   4,  # his typo
    '5th':   5,  '6th':    6,  '7th':  7,
    '8th':   8,
}

# ── Pre-announcement patterns ──────────────────────────────────────────────────
# Matches: "buy", "buy now", "buy gold", "buy gold now", "BUY NOW GOLD",
#          "sell", "sell now", "sell gold", "sell gold now", "Sell again",
#          "Looking buys on gold", "Looking sells on GOLD", "am looking buys"
_PRE_ANN_BUY = re.compile(
    r'^(?:(?:am\s+|i\'?m?\s+)?looking\s+(?:for\s+)?(?:buys|buy)|'
    r'buy(?:\s+(?:now|gold|xauusd|xau|again))*|'
    r'(?:gold|xauusd|xau)\s+buy|'
    r'buy\s+now\s+(?:gold|xauusd|xau)|'
    r'(?:gold|xauusd|xau)\s+buy\s*now)$',
    re.IGNORECASE
)
_PRE_ANN_SELL = re.compile(
    r'^(?:(?:am\s+|i\'?m?\s+)?looking\s+(?:for\s+)?(?:sells|sell)|'
    r'sell(?:\s+(?:now|gold|xauusd|xau|again))*|'
    r'(?:gold|xauusd|xau)\s+sell|'
    r'sell\s+now\s+(?:gold|xauusd|xau)|'
    r'(?:gold|xauusd|xau)\s+sell\s*now)$',
    re.IGNORECASE
)


@dataclass
class ParsedSignal:
    signal_type:   str   = 'unknown'
    raw_text:      str   = ''
    symbol:        str   = 'XAUUSD'
    direction:     Optional[str]  = None   # 'buy' | 'sell'
    stop_loss:     Optional[float] = None
    # TPs in order; 0.0 = open runner (no TP price)
    take_profits:  list  = field(default_factory=list)
    tp_number:     Optional[int]   = None   # for tp_hit
    new_sl:        Optional[float] = None   # for sl_correction
    confidence:    float = 1.0
    warnings:      list  = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _price(s: str) -> Optional[float]:
    """Extract first decimal number ≥ 1000 (valid XAUUSD price)."""
    for m in re.finditer(r'\d{4,5}(?:\.\d{1,2})?', s):
        v = float(m.group())
        if v >= 1000:
            return v
    return None

def _clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()

def _tp_ordinal_to_num(text: str) -> Optional[int]:
    """Convert ordinal word or number in a TP hit message to int."""
    low = text.lower()
    # Direct digit: "TP 7" or "take the TP 7"
    m = re.search(r'tp\s+(\d)', low)
    if m:
        return int(m.group(1))
    # Ordinal word
    for word, num in _TP_ORDINALS.items():
        if re.search(r'\b' + re.escape(word) + r'\b', low):
            return num
    return None

def _is_breakeven_instruction(text: str) -> bool:
    low = _clean(text)
    if re.match(r'^breakeven hit', low):
        return False
    if 'last touched breakeven' in low:
        return False
    has_sl  = bool(re.search(r'\bstop\s*loss\b|\bsl\b', low))
    has_be  = bool(re.search(
        r'\bbreak\s*even\b|\bbreakeven\b|\bbeal\s*even\b'
        r'|\bbreak\s*now\b|\bto\s+be\b|\bsl\s+to\s+be\b', low))
    return has_sl and has_be

def _is_be_hit_announcement(text: str) -> bool:
    low = _clean(text)
    return (
        re.match(r'^breakeven hit', low) is not None
        or 'last touched breakeven' in low
    )

def _is_close_all(text: str) -> bool:
    low = _clean(text)
    return bool(re.search(r'close\s+all|close\s+all\s+profit|close\s+all\s+position', low))

def _is_close(text: str) -> bool:
    low = _clean(text)
    if _is_close_all(text):
        return False
    close_patterns = [
        r'\bclose\s+(it|this|last|your|the)',
        r'\bclose\s+(trade|position|profit)',
        r"i\s+don'?t\s+like\s+it",
        r'not\s+good\s+anymore',
        r'close\s+with\s+breakeven',
        r'close\s+last\s+position',
        r'close\s+your\s+(last|position)',
        r'\bomg.*close\b',
        r'\bclose\s+here\b',
    ]
    return any(re.search(p, low) for p in close_patterns)


# ── Pre-announcement detector ─────────────────────────────────────────────────

def _detect_pre_announcement(text: str) -> Optional[str]:
    """
    Detect bare directional pre-announcement messages.

    Returns 'buy', 'sell', or None.

    Handles all observed variants:
        "buy", "buy now", "buy gold", "buy gold now", "BUY NOW GOLD",
        "buy again", "sell", "sell gold", "sell now", "sell again",
        "Sell again\\n\\nSmall this time",
        "Looking buys on gold", "looking sells on gold",
        "am looking buys on gold", "I'm looking buys on gold"
    """
    # Strip noise lines (e.g. "Sell again\n\nSmall this time" → "sell again")
    first_line = text.strip().split('\n')[0].strip()
    low = _clean(first_line)

    if _PRE_ANN_BUY.match(low):
        return 'buy'
    if _PRE_ANN_SELL.match(low):
        return 'sell'

    # Extra patterns that span a few words with context
    if re.search(r'\blooking\b.*\bbuys?\b', low):
        return 'buy'
    if re.search(r'\blooking\b.*\bsells?\b', low):
        return 'sell'

    return None


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_signal(text: str, is_reply: bool = False, reply_text: str = '') -> ParsedSignal:
    sig = ParsedSignal(raw_text=text)
    low = _clean(text)

    # ── 1. ENTRY SIGNAL ────────────────────────────────────────────────────────
    if 'risky trade' in low:
        sig.symbol = 'XAUUSD'
        dir_match = re.search(r'(buy|sell)', text, re.IGNORECASE)
        if dir_match:
            sig.direction = dir_match.group(1).lower()

        sl_line = next((l for l in text.split('\n') if '🛑' in l or 'SL' in l.upper()), '')
        sig.stop_loss = _price(sl_line)

        tps = []
        for line in text.split('\n'):
            if not line.strip().startswith('✅'):
                continue
            if 'TP' not in line.upper():
                continue
            if 'disclaimer' in line.lower() or 'copy' in line.lower():
                continue
            if re.search(r'TP\s+open', line, re.IGNORECASE):
                tps.append(0.0)
                continue
            price = _price(line)
            if price:
                tps.append(price)

        sig.take_profits = tps
        real_tps = [t for t in tps if t > 0]

        if not sig.direction or not sig.stop_loss or not real_tps:
            sig.signal_type = 'entry_incomplete'
            sig.warnings.append('Missing direction/SL/TPs — ghost/template signal, skip')
            sig.confidence = 0.0
        else:
            sig.signal_type = 'entry'
            if any(t == 0.0 for t in tps):
                sig.warnings.append('Last position is runner (✅ TP open) — no TP set')
            logger.info('[PARSER] ENTRY %s %s  SL=%s  TPs=%s' % (
                sig.direction.upper(), sig.symbol, sig.stop_loss, sig.take_profits))
        return sig

    # ── 2. STANDALONE SL CORRECTION ───────────────────────────────────────────
    if text.strip().startswith('🛑') and re.search(r'SL\s+\d{4}', text):
        price = _price(text)
        if price:
            sig.signal_type = 'sl_correction'
            sig.new_sl = price
            sig.symbol = 'XAUUSD'
            logger.info('[PARSER] SL correction → %s' % price)
            return sig

    # ── 3. TP HIT ANNOUNCEMENT ────────────────────────────────────────────────
    if re.search(r'(TP successfully hit|take\s+the\s+TP\s+\d)', text, re.IGNORECASE):
        sig.signal_type = 'tp_hit'
        sig.symbol = 'XAUUSD'
        sig.tp_number = _tp_ordinal_to_num(text)
        if not sig.tp_number:
            sig.warnings.append('Could not determine TP number')
        logger.info('[PARSER] TP_HIT  TP#=%s' % sig.tp_number)
        return sig

    # ── 4. BREAKEVEN INSTRUCTION ──────────────────────────────────────────────
    if _is_breakeven_instruction(text):
        sig.signal_type = 'breakeven'
        sig.symbol = 'XAUUSD'
        logger.info('[PARSER] BREAKEVEN instruction')
        return sig

    # ── 5. BREAKEVEN HIT (info only) ──────────────────────────────────────────
    if _is_be_hit_announcement(text):
        sig.signal_type = 'be_hit'
        sig.symbol = 'XAUUSD'
        logger.info('[PARSER] BE_HIT announcement (info only)')
        return sig

    # ── 6. CLOSE ALL ──────────────────────────────────────────────────────────
    if _is_close_all(text):
        sig.signal_type = 'close_all'
        sig.symbol = 'XAUUSD'
        logger.info('[PARSER] CLOSE_ALL')
        return sig

    # ── 7. CLOSE ──────────────────────────────────────────────────────────────
    if _is_close(text):
        sig.signal_type = 'close'
        sig.symbol = 'XAUUSD'
        logger.info('[PARSER] CLOSE')
        return sig

    # ── 8. PRE-ANNOUNCEMENT ───────────────────────────────────────────────────
    # Must be LAST before unknown — only match if nothing else matched
    direction = _detect_pre_announcement(text)
    if direction:
        sig.signal_type = 'pre_announcement'
        sig.direction   = direction
        sig.symbol      = 'XAUUSD'
        sig.confidence  = 0.7  # not tradeable alone, just advance notice
        logger.info('[PARSER] PRE_ANNOUNCEMENT direction=%s — waiting for full signal' % direction)
        return sig

    # ── 9. UNKNOWN ────────────────────────────────────────────────────────────
    sig.signal_type = 'unknown'
    sig.confidence  = 0.0
    logger.debug('[PARSER] unknown: %r' % text[:60])
    return sig


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import json, sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Test pre-announcement variants first
    pre_ann_tests = [
        "buy", "buy now", "buy gold", "buy gold now", "BUY NOW GOLD",
        "buy again", "Buy now again",
        "sell", "sell now", "sell gold", "sell gold now", "Sell again",
        "Sell again\n\nSmall this time",
        "Looking buys on gold", "Looking sells on GOLD",
        "am looking buys on gold", "looking sells again on GOLD",
        "Am looking another entry",
        # should NOT match
        "RiskY traDE ☠️\n👉🏾buy XAUUSD now\n🛑 SL 5340\n✅ TP 5362\n✅ TP open\ndisclaimer:",
        "Good morning", "We eat!!!", "Congratulations",
    ]

    print("=== PRE-ANNOUNCEMENT TESTS ===")
    for t in pre_ann_tests:
        r = parse_signal(t)
        if r.signal_type in ('pre_announcement', 'entry'):
            marker = '✅' if r.signal_type in ('pre_announcement', 'entry') else '❌'
        else:
            marker = '  '
        print("%s %-20s dir=%-5s  %r" % (marker, r.signal_type, str(r.direction), t[:50]))

    # Test against real channel data
    try:
        with open('channel_messages_20260308_163556.json', encoding='utf-8') as f:
            msgs = json.load(f)
    except FileNotFoundError:
        print("\nNo JSON file found — pre-announcement tests above are the full test.")
        sys.exit(0)

    counts = {}
    print("\n=== FULL CHANNEL PARSE ===")
    print("%-6s %-20s %-6s %-6s %-50s %s" % ("ID","TYPE","DIR","SL","TPs","WARNINGS"))
    print("-" * 120)
    for m in reversed(msgs):
        r = parse_signal(m['text'])
        counts[r.signal_type] = counts.get(r.signal_type, 0) + 1
        if r.signal_type == 'unknown':
            continue
        tps_str = str(r.take_profits)[:50] if r.take_profits else str(r.tp_number)
        print("%-6s %-20s %-6s %-6s %-50s %s" % (
            m['id'], r.signal_type, str(r.direction or ''), str(r.stop_loss or ''),
            tps_str or '', '; '.join(r.warnings)
        ))

    print("\n=== SUMMARY ===")
    for k, v in sorted(counts.items()):
        print("  %-25s %d" % (k, v))