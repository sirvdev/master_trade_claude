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
    'entry'       — new trade, has direction/SL/TPs
    'entry_incomplete' — RiskY traDE but blank SL or no valid TPs (sent by mistake, skip)
    'tp_hit'      — TP hit announcement
    'breakeven'   — instruction to move SL to BE
    'be_hit'      — market announcement that BE was hit (info only, no action)
    'sl_correction' — standalone 🛑 SL XXXX message correcting last signal's SL
    'close'       — close most recent / last position
    'close_all'   — close every open position
    'unknown'     — nothing matched
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


@dataclass
class ParsedSignal:
    signal_type:   str   = 'unknown'
    raw_text:      str   = ''
    symbol:        str   = 'XAUUSD'   # always XAUUSD from this channel
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
        # match as whole word (allow mixed case)
        if re.search(r'\b' + re.escape(word) + r'\b', low):
            return num
    return None

def _is_breakeven_instruction(text: str) -> bool:
    """
    Detect breakeven MOVE instructions (not 'breakeven hit' announcements).
    Pattern: message contains "stop loss" or "sl" AND "break|beal|be"
    AND is NOT a "breakeven hit" announcement.
    """
    low = _clean(text)

    # 'Breakeven hit' announcements → NOT an instruction
    if re.match(r'^breakeven hit', low):
        return False
    if 'last touched breakeven' in low:
        return False

    has_sl  = bool(re.search(r'\bstop\s*loss\b|\bsl\b', low))
    has_be  = bool(re.search(r'\bbreak\s*even\b|\bbreakeven\b|\bbeal\s*even\b|\bbreak\s*now\b|\bto\s+be\b|\bsl\s+to\s+be\b', low))
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
    """Casual close instructions from this provider."""
    low = _clean(text)
    if _is_close_all(text):
        return False  # handled separately

    close_patterns = [
        r'\bclose\s+(it|this|last|your|the)',
        r'\bclose\s+(trade|position|profit)',
        r"i\s+don'?t\s+like\s+it",
        r'not\s+good\s+anymore',
        r'close\s+with\s+breakeven',   # "not good anymore close with breakeven"
        r'close\s+last\s+position',
        r'close\s+your\s+(last|position)',
        r'\bomg.*close\b',
        r'\bclose\s+here\b',
    ]
    return any(re.search(p, low) for p in close_patterns)


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_signal(text: str, is_reply: bool = False, reply_text: str = '') -> ParsedSignal:
    sig = ParsedSignal(raw_text=text)
    low = _clean(text)

    # ── 1. ENTRY SIGNAL ────────────────────────────────────────────────────────
    if 'risky trade' in low:
        sig.symbol = 'XAUUSD'

        # Direction from the 👉🏾 line
        dir_match = re.search(r'(buy|sell)', text, re.IGNORECASE)
        if dir_match:
            sig.direction = dir_match.group(1).lower()

        # SL from 🛑 SL line
        sl_line = next((l for l in text.split('\n') if '🛑' in l or 'SL' in l.upper()), '')
        sig.stop_loss = _price(sl_line)

        # TPs: lines starting with ✅ TP
        tps = []
        for line in text.split('\n'):
            if not line.strip().startswith('✅'):
                continue
            if 'TP' not in line.upper():
                continue
            if 'disclaimer' in line.lower() or 'copy' in line.lower():
                continue

            # Runner position: "✅ TP open"
            if re.search(r'TP\s+open', line, re.IGNORECASE):
                tps.append(0.0)
                continue

            # Numbered TP with price
            price = _price(line)
            if price:
                tps.append(price)
            # else: blank TP line (ghost signal) — skip

        sig.take_profits = tps

        # Validate: need direction + SL + at least 1 real TP
        real_tps = [t for t in tps if t > 0]
        if not sig.direction or not sig.stop_loss or not real_tps:
            sig.signal_type = 'entry_incomplete'
            sig.warnings.append('Missing direction/SL/TPs — ghost/template signal, skip')
            sig.confidence = 0.0
            logger.warning('[PARSER] entry_incomplete (ghost signal id likely sent before editing)')
        else:
            sig.signal_type = 'entry'
            runner = sum(1 for t in tps if t == 0.0)
            if runner:
                sig.warnings.append('Last position is runner (✅ TP open) — no TP set')
            logger.info('[PARSER] ENTRY %s %s  SL=%s  TPs=%s' % (
                sig.direction.upper(), sig.symbol, sig.stop_loss, sig.take_profits))
        return sig

    # ── 2. STANDALONE SL CORRECTION ───────────────────────────────────────────
    # "🛑 SL 5097 \nSorry I was texting fast correct the stop loss"
    if text.strip().startswith('🛑') and re.search(r'SL\s+\d{4}', text):
        price = _price(text)
        if price:
            sig.signal_type = 'sl_correction'
            sig.new_sl = price
            sig.symbol = 'XAUUSD'
            logger.info('[PARSER] SL correction → %s' % price)
            return sig

    # ── 3. TP HIT ANNOUNCEMENT ────────────────────────────────────────────────
    # "✅ Our Nth TP successfully hit ✅"
    # "✅ Take the TP 7 5098"
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

    # ── 5. BREAKEVEN HIT (market announcement, info only) ─────────────────────
    if _is_be_hit_announcement(text):
        sig.signal_type = 'be_hit'
        sig.symbol = 'XAUUSD'
        logger.info('[PARSER] BE_HIT announcement (info only, no action)')
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

    # ── 8. UNKNOWN ────────────────────────────────────────────────────────────
    sig.signal_type = 'unknown'
    sig.confidence = 0.0
    logger.debug('[PARSER] unknown: %r' % text[:60])
    return sig


# ── Test against real messages ─────────────────────────────────────────────────

if __name__ == '__main__':
    import json, sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        with open('channel_messages_20260308_163556.json') as f:
            msgs = json.load(f)
    except FileNotFoundError:
        print("Run from the folder containing channel_messages_20260308_163556.json")
        sys.exit(1)

    counts = {}
    print("%-6s %-20s %-6s %-6s %-50s %s" % ("ID","TYPE","DIR","SL","TPs","WARNINGS"))
    print("-" * 120)
    for m in reversed(msgs):
        r = parse_signal(m['text'])
        counts[r.signal_type] = counts.get(r.signal_type, 0) + 1
        if r.signal_type == 'unknown':
            continue   # skip chatter
        tps_str = str(r.take_profits)[:50] if r.take_profits else str(r.tp_number)
        print("%-6s %-20s %-6s %-6s %-50s %s" % (
            m['id'], r.signal_type, str(r.direction or ''), str(r.stop_loss or ''),
            tps_str or '', '; '.join(r.warnings)
        ))

    print("\n=== SUMMARY ===")
    for k, v in sorted(counts.items()):
        print("  %-25s %d" % (k, v))