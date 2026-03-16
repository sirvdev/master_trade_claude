"""
dashboard/app.py — Trading Terminal v4
=======================================
A professional dark trading terminal dashboard.

Design direction: Bloomberg Terminal meets modern SaaS.
Dark zinc background, amber/gold accent for profits,
red for losses, cyan for pending/neutral states.
Monospaced data fields, clean table rows, no decorative noise.

Tabs:
  📊 Overview   — live positions, pending limits, KPIs, recent activity
  📈 Trades     — full trade history with filters and detail drill-down
  ⏳ Orders     — pending limit orders with countdown, status, cancel
  📉 Analytics  — equity curve, entry-type breakdown, session heatmap, RR dist
  ⚙️ Config     — risk and strategy parameters
  🧠 Learning   — optimizer controls and parameter versions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import json
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, date, timezone

sys.path.append(str(Path(__file__).parent.parent))
from logger.db import DatabaseManager


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — dark terminal aesthetic ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Hide Streamlit's default header/toolbar so nav sits at top ── */
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
.main .block-container {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
}
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0c10;
    color: #c9d1d9;
}

/* ── Mono data text ── */
code, .mono {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85em;
}

/* ── Colour helpers ── */
.profit   { color: #f0b429; font-weight: 600; }
.loss     { color: #f87171; font-weight: 600; }
.neutral  { color: #6b7280; }
.pending  { color: #38bdf8; font-weight: 600; }
.long-dir { color: #34d399; font-weight: 600; }
.short-dir{ color: #f87171; font-weight: 600; }

/* ── Mode badges ── */
.badge-live { background: #7f1d1d; color: #fca5a5; padding: 3px 10px;
              border-radius: 3px; font-weight: 600; font-size: 12px;
              letter-spacing: 1px; font-family: 'IBM Plex Mono', monospace; }
.badge-demo { background: #1e3a5f; color: #93c5fd; padding: 3px 10px;
              border-radius: 3px; font-weight: 600; font-size: 12px;
              letter-spacing: 1px; font-family: 'IBM Plex Mono', monospace; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 12px 16px !important;
}
div[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 11px !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
    color: #f0f6fc !important;
}

/* ── Section headers ── */
h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; }
h2 { color: #e2e8f0; border-bottom: 1px solid #1f2937; padding-bottom: 6px; }
h3 { color: #94a3b8; font-size: 14px !important; text-transform: uppercase;
     letter-spacing: 1px; }

/* ── Trade row cards ── */
.trade-row {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.trade-row:hover { border-color: #374151; }

/* ── Pending order cards ── */
.order-card {
    background: #0f1f2e;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.order-card.expiring { border-left-color: #f59e0b; }
.order-card.critical { border-left-color: #f87171; }

/* ── Session indicator ── */
.session-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 1px;
}
.session-asian  { background: #1e293b; color: #94a3b8; border: 1px solid #334155; }
.session-london { background: #1e3a5f; color: #93c5fd; border: 1px solid #1d4ed8; }
.session-ny     { background: #2d1b69; color: #a78bfa; border: 1px solid #7c3aed; }
.session-overlap{ background: #1c1917; color: #d6d3d1; border: 1px solid #44403c; }

/* ── Nav bar ── */
.stButton > button {
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #f0b429 !important;
    font-size: 16px !important;
    letter-spacing: 2px;
}

/* ── Dataframe ── */
.dataframe { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px; }

/* ── Dividers ── */
hr { border-color: #1f2937 !important; margin: 12px 0 !important; }

/* ── Info/success boxes ── */
.stAlert { border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_PATH  = Path("config/config.yaml")
CONTROL_DIR  = Path("data")
CLOSE_CMD_FILE = CONTROL_DIR / "close_commands.json"

NAV_PAGES = [
    "📊 Overview", "📈 Trades", "⏳ Orders",
    "📉 Analytics", "⚙️ Config", "🧠 Learning",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default

def clamp(v, lo, hi):
    return max(lo, min(hi, v if v is not None else lo))

def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def save_config(cfg: dict) -> bool:
    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False

def issue_close_command(ticket, symbol, trade_id):
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    cmds = []
    if CLOSE_CMD_FILE.exists():
        try:
            cmds = json.loads(CLOSE_CMD_FILE.read_text())
        except Exception:
            cmds = []
    cmds.append({
        "action": "close_position", "ticket": ticket,
        "symbol": symbol, "trade_id": trade_id,
        "issued_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    })
    CLOSE_CMD_FILE.write_text(json.dumps(cmds, indent=2))

def get_current_session() -> tuple[str, str]:
    """Return (session_name, css_class) based on UTC hour."""
    h = datetime.now(timezone.utc).replace(tzinfo=None).hour
    if 0 <= h < 7:
        return "ASIAN", "session-asian"
    elif 7 <= h < 12:
        return "LONDON", "session-london"
    elif 12 <= h < 20:
        return "NEW YORK", "session-ny"
    else:
        return "OVERLAP", "session-overlap"

def get_time_bounds(time_range: str, custom_start=None, custom_end=None):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if time_range == "Today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0), now
    elif time_range == "Last 7 Days":
        return now - timedelta(days=7), now
    elif time_range == "Last 30 Days":
        return now - timedelta(days=30), now
    elif time_range == "Custom" and custom_start and custom_end:
        return (datetime.combine(custom_start, datetime.min.time()),
                datetime.combine(custom_end,   datetime.max.time()))
    return datetime(2000, 1, 1), now

def filter_trades_by_time(trades, start_dt, end_dt):
    result = []
    for t in trades:
        raw = t.get("entry_time")
        if raw is None:
            result.append(t)
            continue
        try:
            dt = pd.to_datetime(raw)
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)
            if start_dt <= dt <= end_dt:
                result.append(t)
        except Exception:
            result.append(t)
    return result

def format_countdown(expiry_str: str) -> tuple[str, str]:
    """Return (countdown_text, severity) for a limit order expiry."""
    try:
        expiry = datetime.fromisoformat(expiry_str)
        remaining = (expiry - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        if remaining <= 0:
            return "EXPIRED", "critical"
        h, rem = divmod(int(remaining), 3600)
        m, s   = divmod(rem, 60)
        text = f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s"
        if remaining < 300:
            return text, "critical"
        elif remaining < 900:
            return text, "expiring"
        return text, "normal"
    except Exception:
        return "—", "normal"


# ── DB helpers ────────────────────────────────────────────────────────────────

@st.cache_resource
def init_database():
    db = DatabaseManager("data/trading.db")
    db.connect()
    return db

@st.cache_data(ttl=15)
def load_all_trades(_db, limit=2000):
    return _db.get_trades(limit=limit)

@st.cache_data(ttl=10)
def load_open_trades(_db):
    return _db.get_open_trades()

@st.cache_data(ttl=10)
def load_pending_limit_orders(_db):
    try:
        return _db.get_pending_limit_orders()
    except Exception:
        return []

@st.cache_data(ttl=60)
def load_parameter_versions(_db):
    try:
        cur = _db.conn.cursor()
        cur.execute("SELECT * FROM parameter_versions ORDER BY created_at DESC LIMIT 20")
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


# ── Equity backfill ───────────────────────────────────────────────────────────

def backfill_equity_once(db_path: str = "data/trading.db") -> None:
    try:
        conn   = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT trade_id, pnl, equity_after_close
            FROM   trades
            WHERE  status = 'closed' AND exit_time IS NOT NULL
            ORDER  BY exit_time ASC
        """)
        rows = [dict(r) for r in cursor.fetchall()]
        if not rows:
            conn.close()
            return

        # Find anchor — earliest row with a real equity value
        anchor_idx = next(
            (i for i, r in enumerate(rows) if r.get("equity_after_close")),
            None
        )
        if anchor_idx is None:
            conn.close()
            return

        # Walk backwards from anchor
        anchor_eq = float(rows[anchor_idx]["equity_after_close"])
        for i in range(anchor_idx - 1, -1, -1):
            if rows[i].get("equity_after_close"):
                break
            rows[i]["equity_after_close"] = anchor_eq - sum(
                safe_float(rows[j]["pnl"]) for j in range(i + 1, anchor_idx + 1)
            )

        # Walk forwards from anchor
        for i in range(anchor_idx + 1, len(rows)):
            if rows[i].get("equity_after_close"):
                anchor_eq = float(rows[i]["equity_after_close"])
                continue
            prev_eq = float(rows[i - 1].get("equity_after_close") or anchor_eq)
            rows[i]["equity_after_close"] = prev_eq + safe_float(rows[i]["pnl"])

        updates = {
            r["trade_id"]: r["equity_after_close"]
            for r in rows
            if r.get("equity_after_close")
        }
        for trade_id, eq in updates.items():
            conn.execute(
                "UPDATE trades SET equity_after_close = ? WHERE trade_id = ?",
                (eq, trade_id),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        st.toast(f"Equity backfill skipped: {e}", icon="⚠️")


# ── Nav bar ───────────────────────────────────────────────────────────────────

def render_top_nav() -> str:
    if "active_page" not in st.session_state:
        from_url = st.query_params.get("page", NAV_PAGES[0])
        st.session_state["active_page"] = (
            from_url if from_url in NAV_PAGES else NAV_PAGES[0]
        )

    active = st.session_state["active_page"]
    cols   = st.columns(len(NAV_PAGES))

    for i, page in enumerate(NAV_PAGES):
        btn_type = "primary" if page == active else "secondary"
        if cols[i].button(page, key=f"nav_{i}", type=btn_type, use_container_width=True):
            st.session_state["active_page"] = page
            st.query_params["page"] = page
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    return st.session_state["active_page"]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def show_overview_tab(db, open_trades, pending_orders, closed_filt, time_range):
    # ── KPI row ───────────────────────────────────────────────────────────────
    total_pnl   = sum(safe_float(t.get("pnl")) for t in closed_filt)
    win_trades  = [t for t in closed_filt if safe_float(t.get("pnl")) > 0]
    loss_trades = [t for t in closed_filt if safe_float(t.get("pnl")) < 0]
    n_trades    = len(closed_filt)
    win_rate    = len(win_trades) / n_trades * 100 if n_trades else 0
    avg_rr      = (sum(safe_float(t.get("realized_rr")) for t in closed_filt) / n_trades
                   if n_trades else 0)
    gross_profit = sum(safe_float(t.get("pnl")) for t in win_trades)
    gross_loss   = sum(safe_float(t.get("pnl")) for t in loss_trades)
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("NET P&L", f"${total_pnl:,.2f}",
                        delta=f"{total_pnl:+,.2f}")
    with c2: st.metric("WIN RATE", f"{win_rate:.1f}%",
                        delta=f"{win_rate-50:.1f}% vs 50%" if n_trades else None)
    with c3: st.metric("TRADES", f"{n_trades}",
                        delta=f"{len(win_trades)}W / {len(loss_trades)}L")
    with c4: st.metric("AVG R:R", f"{avg_rr:.2f}R",
                        delta=f"{avg_rr-1:.2f}" if n_trades else None)
    with c5: st.metric("PROFIT FACTOR",
                        f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
    with c6: st.metric("OPEN + PENDING", f"{len(open_trades)} + {len(pending_orders)}")

    st.markdown("<hr>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        # ── Open positions ────────────────────────────────────────────────────
        st.markdown(f"### 🟢 OPEN POSITIONS ({len(open_trades)})")
        if open_trades:
            for idx, trade in enumerate(open_trades):
                ticket    = trade.get("ticket")
                symbol    = trade.get("symbol", "?")
                direction = trade.get("direction", "?")
                ep        = safe_float(trade.get("entry_price"))
                sl        = safe_float(trade.get("stop_loss"))
                pnl       = safe_float(trade.get("profit", trade.get("pnl")))
                pnl_sign  = "profit" if pnl >= 0 else "loss"
                dir_cls   = "long-dir" if direction == "long" else "short-dir"
                dir_sym   = "▲" if direction == "long" else "▼"

                r1, r2 = st.columns([6, 1])
                with r1:
                    st.markdown(
                        f"<div class='trade-row'>"
                        f"<span style='color:#e2e8f0;font-weight:600'>{symbol}</span>"
                        f"&nbsp;&nbsp;<span class='{dir_cls}'>{dir_sym} {direction.upper()}</span>"
                        f"&nbsp;&nbsp;Entry: <span class='mono'>{ep:,.4f}</span>"
                        f"&nbsp;&nbsp;SL: <span class='mono'>{sl:,.4f}</span>"
                        f"&nbsp;&nbsp;<span class='{pnl_sign}'>"
                        f"{'+'if pnl>=0 else ''}${pnl:,.2f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with r2:
                    if st.button("✕", key=f"close_{idx}_{ticket}", help="Close position"):
                        if ticket:
                            issue_close_command(ticket, symbol, trade.get("trade_id"))
                            st.success(f"Close queued: {symbol}")
                        else:
                            st.warning("No ticket")
        else:
            st.markdown("<div class='trade-row neutral'>No open positions</div>",
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Pending limit orders (overview summary) ────────────────────────
        st.markdown(f"### ⏳ PENDING LIMITS ({len(pending_orders)})")
        if pending_orders:
            for order in pending_orders[:5]:   # show top 5 in overview
                symbol     = order.get("symbol", "?")
                direction  = order.get("direction", "?")
                limit_px   = safe_float(order.get("limit_price"))
                entry_type = order.get("entry_type", "?")
                expiry_str = order.get("expiry_time", "")
                countdown, severity = format_countdown(expiry_str)
                dir_cls = "long-dir" if direction == "long" else "short-dir"
                dir_sym = "▲" if direction == "long" else "▼"

                st.markdown(
                    f"<div class='order-card {severity}'>"
                    f"<span style='color:#e2e8f0;font-weight:600'>{symbol}</span>"
                    f"&nbsp;&nbsp;<span class='{dir_cls}'>{dir_sym} {direction.upper()}</span>"
                    f"&nbsp;&nbsp;@ <span class='mono pending'>{limit_px:,.5f}</span>"
                    f"&nbsp;&nbsp;<span style='color:#6b7280;font-size:11px'>[{entry_type}]</span>"
                    f"&nbsp;&nbsp;expires: <span class='mono' style='color:#f59e0b'>{countdown}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if len(pending_orders) > 5:
                st.caption(f"+{len(pending_orders)-5} more — see ⏳ Orders tab")
        else:
            st.markdown("<div class='trade-row neutral'>No pending limit orders</div>",
                        unsafe_allow_html=True)

    with right:
        # ── P&L gauge ─────────────────────────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=round(total_pnl, 2),
            delta={
                "reference": 0,
                "valueformat": "$.2f",
                "increasing": {"color": "#f0b429"},
                "decreasing": {"color": "#f87171"},
            },
            number={
                "prefix": "$",
                "valueformat": ",.2f",
                "font": {"size": 32, "color": "#f0f6fc", "family": "IBM Plex Mono"},
            },
            title={"text": f"Net P&L — {time_range}",
                   "font": {"size": 13, "color": "#6b7280"}},
        ))
        fig.update_layout(
            height=150,
            margin=dict(t=35, b=5, l=5, r=5),
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Win/loss bar ──────────────────────────────────────────────────────
        if n_trades:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Wins",
                x=["Wins"],
                y=[len(win_trades)],
                marker_color="#f0b429",
                text=[f"{len(win_trades)}"],
                textposition="inside",
                textfont={"family": "IBM Plex Mono", "color": "#000"},
            ))
            fig2.add_trace(go.Bar(
                name="Losses",
                x=["Losses"],
                y=[len(loss_trades)],
                marker_color="#f87171",
                text=[f"{len(loss_trades)}"],
                textposition="inside",
                textfont={"family": "IBM Plex Mono", "color": "#fff"},
            ))
            fig2.update_layout(
                height=160,
                showlegend=False,
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                margin=dict(t=10, b=30, l=10, r=10),
                xaxis=dict(showgrid=False, color="#6b7280",
                           tickfont={"family": "IBM Plex Mono", "size": 11}),
                yaxis=dict(showgrid=False, color="#6b7280"),
                font={"family": "IBM Plex Mono"},
                barmode="group",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Quick stats table ─────────────────────────────────────────────────
        avg_dur = (sum(safe_float(t.get("duration_minutes")) for t in closed_filt) / n_trades
                   if n_trades else 0)
        pf_str = f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"
        st.markdown(f"""
<div style='background:#111827;border:1px solid #1f2937;border-radius:6px;
     padding:14px;font-family:"IBM Plex Mono",monospace;font-size:13px;
     line-height:2'>
<span style='color:#6b7280'>GROSS PROFIT </span>
<span class='profit'>${gross_profit:,.2f}</span><br>
<span style='color:#6b7280'>GROSS LOSS   </span>
<span class='loss'>${gross_loss:,.2f}</span><br>
<span style='color:#6b7280'>PROFIT FACTOR </span>
<span style='color:#e2e8f0'>{pf_str}</span><br>
<span style='color:#6b7280'>AVG DURATION  </span>
<span style='color:#e2e8f0'>{avg_dur:.0f} min</span>
</div>
""", unsafe_allow_html=True)

    # ── Recent closed trades ──────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📝 RECENT CLOSED TRADES")
    recent = sorted(closed_filt, key=lambda t: t.get("entry_time") or "", reverse=True)[:10]
    if recent:
        cols = st.columns([2, 1, 2, 2, 2, 2, 2])
        headers = ["Symbol", "Dir", "Entry", "Exit", "P&L", "R:R", "Reason"]
        for col, h in zip(cols, headers):
            col.markdown(f"<span style='color:#6b7280;font-size:11px;letter-spacing:1px'>"
                         f"{h}</span>", unsafe_allow_html=True)

        for t in recent:
            pnl = safe_float(t.get("pnl"))
            rr  = safe_float(t.get("realized_rr"))
            pnl_cls = "profit" if pnl > 0 else "loss" if pnl < 0 else "neutral"
            dir_cls = "long-dir" if t.get("direction") == "long" else "short-dir"
            cols = st.columns([2, 1, 2, 2, 2, 2, 2])
            cols[0].markdown(f"<span class='mono' style='color:#e2e8f0'>"
                             f"{t.get('symbol','?')}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"<span class='{dir_cls}'>"
                             f"{'▲' if t.get('direction')=='long' else '▼'}</span>",
                             unsafe_allow_html=True)
            cols[2].markdown(f"<span class='mono'>{safe_float(t.get('entry_price')):,.4f}</span>",
                             unsafe_allow_html=True)
            cols[3].markdown(f"<span class='mono'>{safe_float(t.get('exit_price')):,.4f}</span>",
                             unsafe_allow_html=True)
            cols[4].markdown(f"<span class='{pnl_cls}'>${pnl:,.2f}</span>",
                             unsafe_allow_html=True)
            cols[5].markdown(f"<span class='mono'>{rr:.2f}R</span>", unsafe_allow_html=True)
            cols[6].markdown(f"<span class='mono' style='color:#6b7280;font-size:11px'>"
                             f"{t.get('exit_reason','?')}</span>", unsafe_allow_html=True)
    else:
        st.info("No closed trades in this period.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRADES
# ══════════════════════════════════════════════════════════════════════════════

def show_trades_tab(db, closed_filt, open_trades):
    st.header("📈 Trade History")

    c1, c2, c3 = st.columns(3)
    with c1:
        syms = sorted(set(t.get("symbol", "?") for t in closed_filt))
        sym_filter = st.selectbox("Symbol", ["All"] + syms)
    with c2:
        dir_filter = st.selectbox("Direction", ["All", "long", "short"])
    with c3:
        reason_filter = st.selectbox("Exit Reason",
            ["All", "stop_loss", "take_profit", "trailing_stop",
             "manual", "external_close"])

    display = closed_filt[:]
    if sym_filter    != "All": display = [t for t in display if t.get("symbol")      == sym_filter]
    if dir_filter    != "All": display = [t for t in display if t.get("direction")   == dir_filter]
    if reason_filter != "All": display = [t for t in display if t.get("exit_reason") == reason_filter]

    st.caption(f"{len(display)} trades")
    if not display:
        st.info("No trades match.")
        return

    df = pd.DataFrame(display)
    for col in ["pnl", "realized_rr", "entry_price", "exit_price",
                "duration_minutes", "commission", "slippage",
                "max_favorable_excursion", "max_adverse_excursion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    preferred = ["symbol", "direction", "entry_time", "entry_price",
                 "exit_price", "pnl", "realized_rr", "exit_reason",
                 "duration_minutes", "status"]
    show_cols  = [c for c in preferred if c in df.columns]
    df_display = df[show_cols].copy()

    fmt = {
        "pnl":              lambda x: f"${x:,.2f}" if pd.notna(x) else "—",
        "realized_rr":      lambda x: f"{x:.2f}R"  if pd.notna(x) else "—",
        "entry_price":      lambda x: f"{x:,.4f}"  if pd.notna(x) else "—",
        "exit_price":       lambda x: f"{x:,.4f}"  if pd.notna(x) else "—",
        "duration_minutes": lambda x: f"{x:.0f}m"  if pd.notna(x) else "—",
        "entry_time":       lambda x: str(x)[:16],
    }
    for col, fn in fmt.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(fn)

    selection = st.dataframe(
        df_display,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="trade_table",
    )

    selected_rows = (selection.selection.rows
                     if hasattr(selection, "selection") else [])
    if selected_rows:
        orig = display[selected_rows[0]]
        st.markdown("<hr>", unsafe_allow_html=True)
        dir_sym = "▲ LONG" if orig.get("direction") == "long" else "▼ SHORT"
        st.subheader(
            f"{orig.get('symbol','?')} {dir_sym} — "
            f"{str(orig.get('entry_time',''))[:16]}"
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Entry",    f"{safe_float(orig.get('entry_price')):,.4f}")
            st.metric("Stop Loss",f"{safe_float(orig.get('stop_loss')):,.4f}")
            st.metric("TP1",      f"{safe_float(orig.get('take_profit_1')):,.4f}")
        with c2:
            st.metric("Exit",     f"{safe_float(orig.get('exit_price')):,.4f}")
            st.metric("Net P&L",  f"${safe_float(orig.get('pnl')):,.2f}")
            st.metric("P&L %",    f"{safe_float(orig.get('pnl_percent')):.3f}%")
        with c3:
            st.metric("R:R",      f"{safe_float(orig.get('realized_rr')):.2f}R")
            st.metric("Duration", f"{safe_float(orig.get('duration_minutes')):.0f} min")
            st.metric("Reason",   str(orig.get("exit_reason") or "—"))
        with c4:
            st.metric("MFE",      f"{safe_float(orig.get('max_favorable_excursion')):.4f}")
            st.metric("MAE",      f"{safe_float(orig.get('max_adverse_excursion')):.4f}")
            eq = orig.get("equity_after_close")
            st.metric("Equity ↗", f"${safe_float(eq):,.2f}" if eq else "—")
        st.caption(
            f"Trade: `{orig.get('trade_id','?')}` | "
            f"Ticket: `{orig.get('ticket','?')}` | "
            f"Analysis: `{orig.get('analysis_id','?')}`"
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    csv = pd.DataFrame(display).to_csv(index=False)
    st.download_button("📥 Export CSV", csv, "trades.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PENDING LIMIT ORDERS
# ══════════════════════════════════════════════════════════════════════════════

def show_orders_tab(db, pending_orders):
    st.header("⏳ Pending Limit Orders")

    if not pending_orders:
        st.info("No pending limit orders. The system places limit orders when entry type "
                "is ema_stack_pullback, pullback_to_sr, or rsi_divergence.")
        return

    # Summary row
    c1, c2, c3, c4 = st.columns(4)
    expiring_soon = [o for o in pending_orders
                     if format_countdown(o.get("expiry_time",""))[1] in ("critical","expiring")]
    c1.metric("PENDING ORDERS", len(pending_orders))
    c2.metric("EXPIRING SOON (<15m)", len(expiring_soon))
    long_orders  = [o for o in pending_orders if o.get("direction") == "long"]
    short_orders = [o for o in pending_orders if o.get("direction") == "short"]
    c3.metric("LONG / SHORT", f"{len(long_orders)} / {len(short_orders)}")
    c4.metric("ENTRY TYPES",
              len(set(o.get("entry_type","?") for o in pending_orders)))

    st.markdown("<hr>", unsafe_allow_html=True)

    for order in pending_orders:
        ticket      = order.get("ticket")
        trade_id    = order.get("trade_id", "?")
        symbol      = order.get("symbol", "?")
        direction   = order.get("direction", "?")
        entry_type  = order.get("entry_type", "?")
        limit_px    = safe_float(order.get("limit_price"))
        inv_px      = order.get("invalidation_price")
        placed_at   = order.get("placed_at", "?")[:16]
        expiry_str  = order.get("expiry_time", "")
        countdown, severity = format_countdown(expiry_str)
        atr         = safe_float(order.get("atr_at_placement"))
        dir_sym     = "▲" if direction == "long" else "▼"
        dir_cls     = "long-dir" if direction == "long" else "short-dir"

        col_info, col_action = st.columns([8, 1])
        with col_info:
            inv_str = f"{safe_float(inv_px):,.5f}" if inv_px else "—"
            color_count = ("#f87171" if severity == "critical"
                           else "#f59e0b" if severity == "expiring"
                           else "#38bdf8")
            st.markdown(
                f"<div class='order-card {severity}'>"
                f"<span style='color:#e2e8f0;font-weight:600;font-size:14px'>"
                f"{symbol}</span>"
                f"&nbsp;&nbsp;<span class='{dir_cls}'>{dir_sym} {direction.upper()}</span>"
                f"&nbsp;&nbsp;"
                f"<span style='color:#6b7280;font-size:11px'>[{entry_type}]</span>"
                f"<br>"
                f"<span style='color:#6b7280'>LIMIT  </span>"
                f"<span class='pending' style='font-size:15px'>{limit_px:,.5f}</span>"
                f"&nbsp;&nbsp;"
                f"<span style='color:#6b7280'>INVALID IF &lt; </span>"
                f"<span class='mono loss'>{inv_str}</span>"
                f"&nbsp;&nbsp;"
                f"<span style='color:#6b7280'>ATR  </span>"
                f"<span class='mono'>{atr:.4f}</span>"
                f"<br>"
                f"<span style='color:#6b7280'>PLACED  </span>"
                f"<span class='mono'>{placed_at}</span>"
                f"&nbsp;&nbsp;"
                f"<span style='color:#6b7280'>EXPIRES IN  </span>"
                f"<span class='mono' style='color:{color_count}'>{countdown}</span>"
                f"&nbsp;&nbsp;"
                f"<span style='color:#374151;font-size:11px'>ticket={ticket}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_action:
            if st.button("✕", key=f"cancel_order_{ticket}",
                         help="Cancel this limit order"):
                issue_close_command(ticket, symbol, trade_id)
                st.warning(f"Cancel queued for {symbol} ticket {ticket}. "
                           f"main.py will execute on next cycle.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(
        "⚠️ Cancel button queues a close command — main.py's position monitor "
        "executes it on the next 15-second cycle. Refresh the page to see updated status."
    )

    # Historical cancelled / expired orders
    with st.expander("📋 Order History (cancelled / expired / filled)"):
        try:
            cur = db.conn.cursor()
            cur.execute("""
                SELECT symbol, direction, entry_type, limit_price,
                       placed_at, expiry_time, status, cancelled_reason
                FROM   pending_limit_orders
                WHERE  status != 'pending'
                ORDER  BY placed_at DESC
                LIMIT  50
            """)
            rows = [dict(r) for r in cur.fetchall()]
            if rows:
                df_hist = pd.DataFrame(rows)
                st.dataframe(df_hist, use_container_width=True)
            else:
                st.info("No historical order records yet.")
        except Exception as e:
            st.info(f"Order history unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def show_analytics_tab(closed_filt):
    st.header("📉 Performance Analytics")

    if not closed_filt:
        st.info("No closed trades in the selected period.")
        return

    df = pd.DataFrame(closed_filt)
    df["entry_time"]         = pd.to_datetime(df.get("entry_time"),  errors="coerce")
    df["exit_time"]          = pd.to_datetime(df.get("exit_time"),   errors="coerce")
    df["pnl"]                = pd.to_numeric(df.get("pnl"),                errors="coerce").fillna(0.0)
    df["realized_rr"]        = pd.to_numeric(df.get("realized_rr"),        errors="coerce").fillna(0.0)
    df["equity_after_close"] = pd.to_numeric(df.get("equity_after_close"), errors="coerce")
    df_sorted = df.dropna(subset=["exit_time"]).sort_values("exit_time").copy()

    CHART_COLORS = {
        "bg":     "#111827",
        "grid":   "#1f2937",
        "text":   "#6b7280",
        "profit": "#f0b429",
        "loss":   "#f87171",
        "line":   "#38bdf8",
    }

    def _base_layout(height=300, title=""):
        return dict(
            height=height,
            title=dict(text=title, font=dict(size=12, color="#6b7280"), x=0.01),
            paper_bgcolor=CHART_COLORS["bg"],
            plot_bgcolor=CHART_COLORS["bg"],
            margin=dict(t=35, b=40, l=50, r=20),
            font=dict(family="IBM Plex Mono", color=CHART_COLORS["text"], size=11),
            xaxis=dict(showgrid=False, color=CHART_COLORS["text"],
                       zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=CHART_COLORS["grid"],
                       color=CHART_COLORS["text"], zeroline=True,
                       zerolinecolor=CHART_COLORS["grid"], tickfont=dict(size=10)),
        )

    # ── Row 1: equity curve + drawdown ────────────────────────────────────────
    st.markdown("### EQUITY CURVE")

    has_real = df_sorted["equity_after_close"].notna().any()
    if has_real:
        df_sorted["equity_plot"] = df_sorted["equity_after_close"].ffill().bfill()
        eq_caption = "Real equity from MT5 balance snapshots"
    else:
        try:
            cfg     = load_config()
            start_eq = float(cfg.get("general", {}).get("starting_equity", 100_000.0))
        except Exception:
            start_eq = 100_000.0
        df_sorted["equity_plot"] = start_eq + df_sorted["pnl"].cumsum()
        eq_caption = "⚠️ Estimated equity (P&L cumsum — run system to get real values)"

    df_sorted["peak"]     = df_sorted["equity_plot"].cummax()
    df_sorted["drawdown"] = (df_sorted["equity_plot"] - df_sorted["peak"])

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=df_sorted["exit_time"], y=df_sorted["equity_plot"],
        name="Equity",
        line=dict(color=CHART_COLORS["profit"], width=2),
        fill="tozeroy",
        fillcolor="rgba(240,180,41,0.06)",
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    fig_eq.add_trace(go.Scatter(
        x=df_sorted["exit_time"], y=df_sorted["peak"],
        name="Peak",
        line=dict(color=CHART_COLORS["line"], width=1, dash="dot"),
        hovertemplate="Peak: $%{y:,.2f}<extra></extra>",
    ))
    fig_eq.update_layout(**_base_layout(280, "EQUITY"))
    fig_eq.update_layout(showlegend=True,
                         legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                                     font=dict(size=10)))
    st.plotly_chart(fig_eq, use_container_width=True)
    st.caption(eq_caption)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Bar(
        x=df_sorted["exit_time"], y=df_sorted["drawdown"],
        marker_color=CHART_COLORS["loss"],
        opacity=0.7,
        name="Drawdown",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>DD: $%{y:,.2f}<extra></extra>",
    ))
    fig_dd.update_layout(**_base_layout(160, "DRAWDOWN"))
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 2: entry type breakdown + session heatmap ─────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### ENTRY TYPE BREAKDOWN")
        if "entry_reason" in df.columns or "exit_reason" in df.columns:
            # Use entry_reason from analysis_logs if we had it; use exit_reason as proxy
            entry_col = "entry_reason" if "entry_reason" in df.columns else "exit_reason"
            type_stats = (
                df.groupby(entry_col)
                  .agg(count=("pnl", "count"), total_pnl=("pnl", "sum"),
                       win_rate=("pnl", lambda x: (x > 0).mean() * 100))
                  .reset_index()
                  .sort_values("total_pnl", ascending=False)
            )
            fig_et = go.Figure()
            colors = [CHART_COLORS["profit"] if p >= 0 else CHART_COLORS["loss"]
                      for p in type_stats["total_pnl"]]
            fig_et.add_trace(go.Bar(
                x=type_stats[entry_col],
                y=type_stats["total_pnl"],
                marker_color=colors,
                text=[f"${v:,.0f}" for v in type_stats["total_pnl"]],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10),
                hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
            ))
            fig_et.update_layout(**_base_layout(260, "P&L BY EXIT TYPE"))
            st.plotly_chart(fig_et, use_container_width=True)
        else:
            st.info("Entry type data not available.")

    with col_b:
        st.markdown("### TRADE HOUR DISTRIBUTION (UTC)")
        if df_sorted["entry_time"].notna().any():
            df_sorted["entry_hour"] = df_sorted["entry_time"].dt.hour
            hour_stats = (
                df_sorted.groupby("entry_hour")
                         .agg(count=("pnl", "count"),
                              total_pnl=("pnl", "sum"))
                         .reindex(range(24), fill_value=0)
                         .reset_index()
            )
            # Colour by session
            def _session_color(h):
                if 0 <= h < 7:   return "#334155"
                elif 7 <= h < 12: return "#1d4ed8"
                elif 12 <= h < 20:return "#7c3aed"
                return "#44403c"

            fig_hr = go.Figure()
            fig_hr.add_trace(go.Bar(
                x=hour_stats["entry_hour"],
                y=hour_stats["count"],
                marker_color=[_session_color(h) for h in hour_stats["entry_hour"]],
                hovertemplate="Hour %{x}:00 UTC<br>Trades: %{y}<extra></extra>",
            ))
            fig_hr.update_layout(**_base_layout(260, "TRADES BY HOUR (UTC)"))
            fig_hr.update_layout(xaxis=dict(
                tickvals=list(range(0, 24, 3)),
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
            ))
            st.plotly_chart(fig_hr, use_container_width=True)
        else:
            st.info("Entry time data not available.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 3: R:R histogram + win streak ─────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("### R:R DISTRIBUTION")
        rr_data = df_sorted["realized_rr"].dropna()
        if len(rr_data) > 2:
            fig_rr = go.Figure()
            fig_rr.add_trace(go.Histogram(
                x=rr_data,
                nbinsx=20,
                marker_color=CHART_COLORS["line"],
                opacity=0.8,
                hovertemplate="R:R: %{x:.1f}<br>Count: %{y}<extra></extra>",
            ))
            fig_rr.add_vline(x=1.0, line_dash="dash", line_color="#6b7280",
                             annotation_text="1R", annotation_font_size=10)
            avg_rr = rr_data.mean()
            fig_rr.add_vline(x=avg_rr, line_dash="dash", line_color=CHART_COLORS["profit"],
                             annotation_text=f"avg {avg_rr:.2f}R",
                             annotation_font_size=10)
            fig_rr.update_layout(**_base_layout(260, "R:R DISTRIBUTION"))
            st.plotly_chart(fig_rr, use_container_width=True)
        else:
            st.info("Need more trades for R:R distribution.")

    with col_d:
        st.markdown("### CUMULATIVE P&L WATERFALL")
        if len(df_sorted) > 1:
            fig_wf = go.Figure()
            colors_wf = [CHART_COLORS["profit"] if p >= 0 else CHART_COLORS["loss"]
                         for p in df_sorted["pnl"]]
            fig_wf.add_trace(go.Bar(
                x=list(range(1, len(df_sorted) + 1)),
                y=df_sorted["pnl"],
                marker_color=colors_wf,
                opacity=0.85,
                hovertemplate="Trade %{x}<br>P&L: $%{y:,.2f}<extra></extra>",
            ))
            fig_wf.update_layout(**_base_layout(260, "P&L PER TRADE"))
            fig_wf.add_hline(y=0, line_color="#374151", line_width=1)
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("Need more trades for waterfall chart.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Symbol breakdown table ─────────────────────────────────────────────────
    st.markdown("### SYMBOL BREAKDOWN")
    if "symbol" in df.columns:
        sym_stats = (
            df.groupby("symbol")
              .agg(
                  trades    = ("pnl", "count"),
                  total_pnl = ("pnl", "sum"),
                  avg_pnl   = ("pnl", "mean"),
                  win_rate  = ("pnl", lambda x: f"{(x > 0).mean()*100:.1f}%"),
                  avg_rr    = ("realized_rr", "mean"),
              )
              .reset_index()
              .sort_values("total_pnl", ascending=False)
        )
        sym_stats["total_pnl"] = sym_stats["total_pnl"].apply(lambda x: f"${x:,.2f}")
        sym_stats["avg_pnl"]   = sym_stats["avg_pnl"].apply(lambda x: f"${x:,.2f}")
        sym_stats["avg_rr"]    = sym_stats["avg_rr"].apply(lambda x: f"{x:.2f}R")
        st.dataframe(sym_stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def show_configuration_tab(config):
    st.header("⚙️ Configuration")

    rm   = config.get("risk_management", {})
    gl   = rm.get("global_limits", {})
    sl   = rm.get("stop_loss", {})
    tr   = rm.get("trailing_stop", {})
    stg  = config.get("strategy", {})
    sess = stg.get("session_atr_multipliers", {})

    with st.form("config_form"):
        st.subheader("Risk Management")
        r1, r2, r3 = st.columns(3)
        with r1:
            max_risk  = st.number_input("Max Risk % / Trade", 0.01, 10.0, step=0.01,
                value=float(clamp(rm.get("max_risk_percent_per_trade", 1.0), 0.01, 10.0)))
            atr_mult  = st.number_input("ATR Multiplier (SL)", 0.5, 10.0, step=0.1,
                value=float(clamp(sl.get("atr_multiplier", 2.0), 0.5, 10.0)))
        with r2:
            max_dd_pct = st.number_input("Daily Max DD %", 0.1, 50.0, step=0.1,
                value=float(clamp(gl.get("daily_max_drawdown_percent", 5.0), 0.1, 50.0)))
            max_conc   = st.number_input("Max Concurrent Trades", 1, 20, step=1,
                value=int(clamp(gl.get("max_concurrent_trades", 3), 1, 20)))
        with r3:
            max_day    = st.number_input("Max Trades / Day", 1, 100, step=1,
                value=int(clamp(gl.get("max_trades_per_day", 10), 1, 100)))
            trail_rr   = st.number_input("Trailing Activation R:R", 0.1, 5.0, step=0.1,
                value=float(clamp(tr.get("activation_rr", 1.0), 0.1, 5.0)))

        sl_methods = ["conservative", "atr", "structure"]
        cur_method = sl.get("method", "conservative")
        sl_method  = st.selectbox("Stop Loss Method", sl_methods,
            index=sl_methods.index(cur_method) if cur_method in sl_methods else 0)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Session ATR Multipliers")
        s1, s2, s3, s4 = st.columns(4)
        with s1: asian_m  = st.number_input("Asian (00-07 UTC)",   0.5, 5.0, step=0.1,
                                value=float(clamp(sess.get("asian",   1.5), 0.5, 5.0)))
        with s2: london_m = st.number_input("London (07-12 UTC)",  0.5, 5.0, step=0.1,
                                value=float(clamp(sess.get("london",  2.5), 0.5, 5.0)))
        with s3: ny_m     = st.number_input("New York (12-20 UTC)",0.5, 5.0, step=0.1,
                                value=float(clamp(sess.get("ny",      2.5), 0.5, 5.0)))
        with s4: ov_m     = st.number_input("Overlap (20-24 UTC)", 0.5, 5.0, step=0.1,
                                value=float(clamp(sess.get("overlap",  2.0), 0.5, 5.0)))

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Confluence Threshold")
        cc1, cc2 = st.columns(2)
        with cc1: conf_volatile = st.number_input("Volatile symbols (XAUUSD, BTC, NAS)", 1, 20, step=1,
                                      value=int(clamp(stg.get("confluence_required", 7), 1, 20)))
        with cc2: conf_calm     = st.number_input("Calm symbols (EURUSD, GBPUSD, XAGUSD)", 1, 20, step=1,
                                      value=int(clamp(stg.get("confluence_threshold_calm", 5), 1, 20)))

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Cooldown After Losses")
        cd1, cd2 = st.columns(2)
        cooldown = gl.get("cooldown_after_losses", {})
        with cd1: cd_losses = st.number_input("Consecutive Losses Trigger", 1, 10, step=1,
                                  value=int(clamp(cooldown.get("consecutive_losses", 3), 1, 10)))
        with cd2: cd_secs   = st.number_input("Cooldown Duration (seconds)", 60, 86400, step=60,
                                  value=int(clamp(cooldown.get("cooldown_seconds", 3600), 60, 86400)))

        submitted = st.form_submit_button("💾 Save Configuration", type="primary")

    if submitted:
        config.setdefault("risk_management", {})["max_risk_percent_per_trade"] = max_risk
        config["risk_management"].setdefault("stop_loss", {}).update(
            {"atr_multiplier": atr_mult, "method": sl_method})
        config["risk_management"].setdefault("trailing_stop", {})["activation_rr"] = trail_rr
        gl_block = config["risk_management"].setdefault("global_limits", {})
        gl_block.update({
            "daily_max_drawdown_percent": max_dd_pct,
            "max_concurrent_trades": max_conc,
            "max_trades_per_day": max_day,
        })
        gl_block.setdefault("cooldown_after_losses", {}).update({
            "consecutive_losses": cd_losses,
            "cooldown_seconds": cd_secs,
        })
        config.setdefault("strategy", {})["confluence_required"]        = conf_volatile
        config["strategy"]["confluence_threshold_calm"]                  = conf_calm
        config["strategy"].setdefault("session_atr_multipliers", {}).update({
            "asian": asian_m, "london": london_m, "ny": ny_m, "overlap": ov_m
        })
        if save_config(config):
            st.success("✅ Saved. Restart main.py to apply.")
            st.cache_data.clear()

    # ── Symbols ────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Symbol Configuration")
    symbols = config.get("symbols", {})
    if symbols:
        rows = []
        for sym, cfg in symbols.items():
            rows.append({
                "Symbol":    sym,
                "Enabled":   "✅" if cfg.get("enabled") else "❌",
                "Platform":  cfg.get("platform", "—"),
                "Primary TF": cfg.get("primary_timeframe", "—"),
                "Entry TF":  cfg.get("entry_timeframe", "—"),
                "Confluence": cfg.get("confluence_threshold", "default"),
                "Timeframes": ", ".join(cfg.get("timeframes", [])),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Edit config.yaml directly to enable/disable symbols or change timeframes.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LEARNING
# ══════════════════════════════════════════════════════════════════════════════

def show_learning_tab(db):
    st.header("🧠 Learning Engine")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("30-Day Performance")
        try:
            stats = db.get_trade_statistics(days=30)
            st.metric("Total Trades", stats.get("total_trades", 0))
            st.metric("Win Rate",     f"{safe_float(stats.get('win_rate'))*100:.1f}%")
            st.metric("Avg P&L",      f"${safe_float(stats.get('avg_pnl')):.2f}")
            st.metric("Total P&L",    f"${safe_float(stats.get('total_pnl')):.2f}")
            st.metric("Avg R:R",      f"{safe_float(stats.get('avg_rr')):.2f}R")
            st.metric("Avg Duration", f"{safe_float(stats.get('avg_duration_minutes')):.0f} min")
        except Exception as e:
            st.error(f"Stats unavailable: {e}")

    with col2:
        st.subheader("Optimizer Controls")
        st.info("The learning engine analyses closed trades and suggests better parameters.")
        try:
            from learning.learner import StrategyLearner as Learner
            cfg     = load_config()
            learner = Learner(db, cfg)
            ca, cb  = st.columns(2)
            with ca:
                if st.button("🔍 Grid Search"):
                    with st.spinner("Running..."):
                        result = learner.run_grid_search()
                    st.success(result.get("message", "Done"))
                    st.cache_data.clear()
            with cb:
                if st.button("🎰 RL Bandit"):
                    with st.spinner("Running..."):
                        result = learner.run_rl_bandit()
                    st.success(result.get("message", "Done"))
                    st.cache_data.clear()
        except Exception as e:
            st.warning(f"Learner unavailable: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Parameter Versions")
    versions = load_parameter_versions(db)
    if versions:
        rows = []
        for v in versions:
            m = {}
            try:
                m = json.loads(v.get("backtest_metrics") or "{}")
            except Exception:
                pass
            rows.append({
                "ID":         v.get("version_id"),
                "Name":       v.get("version_name"),
                "Created":    str(v.get("created_at",""))[:16],
                "Source":     v.get("source"),
                "Status":     v.get("status"),
                "Win Rate":   f"{safe_float(m.get('win_rate'))*100:.1f}%" if m.get("win_rate") else "—",
                "Expectancy": f"${safe_float(m.get('expectancy')):.2f}" if m.get("expectancy") else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No parameter versions yet.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    db     = init_database()
    config = load_config()

    if not st.session_state.get("equity_backfill_done", False):
        backfill_equity_once("data/trading.db")
        st.session_state["equity_backfill_done"] = True

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚡ TERMINAL")
        st.markdown("<hr>", unsafe_allow_html=True)

        current_mode = config.get("general", {}).get("mode", "demo")
        is_live      = current_mode == "live"
        badge = (f'<span class="badge-live">● LIVE</span>' if is_live
                 else f'<span class="badge-demo">● DEMO</span>')
        st.markdown(badge, unsafe_allow_html=True)

        # Session indicator
        session_name, session_cls = get_current_session()
        st.markdown(
            f"<br><span class='session-badge {session_cls}'>{session_name} SESSION</span>"
            f"<br><span style='color:#6b7280;font-size:11px;font-family:\"IBM Plex Mono\"'>"
            f"{datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M')} UTC</span>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        new_mode = st.radio("Mode", ["demo", "live"], index=1 if is_live else 0,
                            horizontal=True)
        if new_mode != current_mode:
            if new_mode == "live":
                if st.checkbox("⚠️ Confirm LIVE trading", key="live_confirm"):
                    config.setdefault("general", {})["mode"] = "live"
                    if save_config(config):
                        st.success("Set to LIVE. Restart main.py.")
                    st.cache_data.clear()
            else:
                config.setdefault("general", {})["mode"] = "demo"
                if save_config(config):
                    st.success("Set to DEMO.")
                st.cache_data.clear()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Time Filter**")
        time_range = st.selectbox("Period", ["Today", "Last 7 Days", "Last 30 Days",
                                              "All Time", "Custom"],
                                   index=0, label_visibility="collapsed")
        custom_start = custom_end = None
        if time_range == "Custom":
            custom_start = st.date_input("From", value=date.today() - timedelta(days=7))
            custom_end   = st.date_input("To",   value=date.today())

        start_dt, end_dt = get_time_bounds(time_range, custom_start, custom_end)
        st.caption(f"{start_dt.strftime('%Y-%m-%d %H:%M')} →\n"
                   f"{end_dt.strftime('%Y-%m-%d %H:%M')} UTC")

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.checkbox("Auto-refresh (30s)"):
            import time as _t
            _t.sleep(0.5)
            st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    all_trades     = load_all_trades(db)
    open_trades    = load_open_trades(db)
    pending_orders = load_pending_limit_orders(db)
    filtered       = filter_trades_by_time(all_trades, start_dt, end_dt)
    closed_filt    = [t for t in filtered if t.get("status") == "closed"]

    # ── Nav ───────────────────────────────────────────────────────────────────
    active_page = render_top_nav()

    # ── Route ─────────────────────────────────────────────────────────────────
    if active_page == "📊 Overview":
        show_overview_tab(db, open_trades, pending_orders, closed_filt, time_range)
    elif active_page == "📈 Trades":
        show_trades_tab(db, closed_filt, open_trades)
    elif active_page == "⏳ Orders":
        show_orders_tab(db, pending_orders)
    elif active_page == "📉 Analytics":
        show_analytics_tab(closed_filt)
    elif active_page == "⚙️ Config":
        show_configuration_tab(config)
    elif active_page == "🧠 Learning":
        show_learning_tab(db)


if __name__ == "__main__":
    main()