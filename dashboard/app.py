"""
Trading System Dashboard — v3
Changes from v2:
  - Top horizontal navigation bar replaces sidebar radio
  - Nav bar persists across F5 reloads via st.query_params
  - Equity curve backfills equity_after_close for old trades (one-time per session)
  - ALL use_container_width replaced: width='stretch' for dataframes + plotly charts
  - Sidebar kept lean: mode, quick actions, time filter only
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
from datetime import datetime, timedelta, date

sys.path.append(str(Path(__file__).parent.parent))
from logger.db import DatabaseManager

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Colour helpers ── */
.positive { color: #00cc44; font-weight: bold; }
.negative { color: #ff4444; font-weight: bold; }
.neutral  { color: #888888; }

/* ── Mode badges ── */
.mode-live { background: #d32f2f; color: white; padding: 3px 10px;
             border-radius: 4px; font-weight: bold; font-size: 13px; }
.mode-demo { background: #1565c0; color: white; padding: 3px 10px;
             border-radius: 4px; font-weight: bold; font-size: 13px; }

/* ── Metric card styling ── */
div[data-testid="metric-container"] {
    background: #0e1a2b;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 10px 14px;
}
</style>
""", unsafe_allow_html=True)

CONFIG_PATH    = Path("config/config.yaml")
CONTROL_DIR    = Path("data")
CLOSE_CMD_FILE = CONTROL_DIR / "close_commands.json"

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
        "issued_at": datetime.utcnow().isoformat(),
    })
    CLOSE_CMD_FILE.write_text(json.dumps(cmds, indent=2))

# ── Time helpers ──────────────────────────────────────────────────────────────

def get_time_bounds(time_range: str, custom_start=None, custom_end=None):
    now = datetime.utcnow()
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

# ── DB helpers ────────────────────────────────────────────────────────────────

@st.cache_resource
def init_database():
    db = DatabaseManager("data/trading.db")
    db.connect()
    return db

@st.cache_data(ttl=20)
def load_all_trades(_db, limit=2000):
    return _db.get_trades(limit=limit)

@st.cache_data(ttl=20)
def load_open_trades(_db):
    return _db.get_open_trades()

@st.cache_data(ttl=60)
def load_parameter_versions(_db):
    try:
        cur = _db.conn.cursor()
        cur.execute("SELECT * FROM parameter_versions ORDER BY created_at DESC LIMIT 20")
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# EQUITY BACKFILL
# Runs once per browser session (stored in session_state).
# Finds every closed trade that is missing equity_after_close and fills it
# by working backwards and forwards from the earliest known real equity value.
# ══════════════════════════════════════════════════════════════════════════════

def backfill_equity_once(db_path: str = "data/trading.db") -> None:
    """
    One-time backfill of equity_after_close for historical trades.

    Strategy:
      1. Load all closed trades ordered by exit_time ASC.
      2. Find the earliest row that already has a real equity_after_close.
      3. Walk backwards: equity before trade N = equity_after_close[N] - pnl[N]
      4. Walk forwards:  equity after trade N  = equity_after_close[N-1] + pnl[N]
         (only fills rows that still have NULL — real values are never overwritten)
      5. UPDATE those rows in the database.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cols = [r[1] for r in cur.execute("PRAGMA table_info(trades)")]
        if "equity_after_close" not in cols:
            conn.close()
            return

        rows = cur.execute("""
            SELECT trade_id, exit_time, pnl, equity_after_close
            FROM   trades
            WHERE  status IN ('closed', 'pending_exit')
              AND  exit_time IS NOT NULL
            ORDER  BY exit_time ASC
        """).fetchall()

        if not rows:
            conn.close()
            return

        # Find first row with a real equity snapshot
        anchor_idx = next(
            (i for i, r in enumerate(rows) if r["equity_after_close"] is not None),
            None,
        )
        if anchor_idx is None:
            conn.close()
            return

        anchor_equity = float(rows[anchor_idx]["equity_after_close"])
        updates = []

        # Backwards: reconstruct equity for rows before the anchor
        running = anchor_equity
        for i in range(anchor_idx - 1, -1, -1):
            pnl     = float(rows[i]["pnl"] or 0.0)
            running = running - pnl          # what equity was AFTER this earlier close
            if rows[i]["equity_after_close"] is None:
                updates.append((round(running, 2), rows[i]["trade_id"]))

        # Forwards: fill gaps after the anchor
        running = anchor_equity
        for i in range(anchor_idx + 1, len(rows)):
            if rows[i]["equity_after_close"] is not None:
                running = float(rows[i]["equity_after_close"])   # re-anchor on real value
            else:
                pnl     = float(rows[i]["pnl"] or 0.0)
                running = running + pnl
                updates.append((round(running, 2), rows[i]["trade_id"]))

        if updates:
            cur.executemany(
                "UPDATE trades SET equity_after_close = ? WHERE trade_id = ?",
                updates,
            )
            conn.commit()

        conn.close()

    except Exception as e:
        # Non-fatal — just log, don't crash the dashboard
        st.toast(f"Equity backfill skipped: {e}", icon="⚠️")


# ══════════════════════════════════════════════════════════════════════════════
# TOP NAVIGATION BAR
# ══════════════════════════════════════════════════════════════════════════════

NAV_PAGES = ["📊 Overview", "📈 Trades", "📉 Analytics", "⚙️ Config", "🧠 Learning"]

def render_top_nav() -> str:
    """
    Render the horizontal nav bar and return the active page name.

    Active page is stored in BOTH session_state (fast within-session) and
    st.query_params so it survives F5 / hard reload — the browser URL keeps
    ?page=<name> and Streamlit reads it back on the next load.
    """
    # On first load of a session: restore from URL query param if present
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
            st.query_params["page"] = page   # persist in URL — survives F5
            st.rerun()

    st.markdown("<hr style='margin:0 0 16px 0; border-color:#1e3a5f'>",
                unsafe_allow_html=True)
    return st.session_state["active_page"]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def show_overview_tab(db, open_trades, closed_filt, time_range, start_dt, end_dt):
    st.header(f"📊 Overview — {time_range}")

    total_pnl   = sum(safe_float(t.get("pnl")) for t in closed_filt)
    win_trades  = [t for t in closed_filt if safe_float(t.get("pnl")) > 0]
    loss_trades = [t for t in closed_filt if safe_float(t.get("pnl")) < 0]
    n_trades    = len(closed_filt)
    win_rate    = len(win_trades) / n_trades * 100 if n_trades else 0
    avg_rr      = (sum(safe_float(t.get("realized_rr")) for t in closed_filt) / n_trades
                   if n_trades else 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Closed Trades",  n_trades)
    with c2: st.metric("Win Rate",       f"{win_rate:.1f}%",
                        delta=f"{win_rate-50:.1f}%" if n_trades else None)
    with c3: st.metric("Period P&L",     f"${total_pnl:,.2f}",
                        delta=f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"${total_pnl:,.2f}")
    with c4: st.metric("Avg R:R",        f"{avg_rr:.2f}",
                        delta=f"{avg_rr-1:.2f}" if n_trades else None)
    with c5: st.metric("Open Positions", len(open_trades))

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📍 Open Positions")
        if open_trades:
            for idx, trade in enumerate(open_trades):
                ticket    = trade.get("ticket")
                symbol    = trade.get("symbol", "?")
                direction = trade.get("direction", "?")
                ep        = safe_float(trade.get("entry_price"))
                sl        = safe_float(trade.get("stop_loss"))
                pnl       = safe_float(trade.get("profit", trade.get("pnl")))
                color     = "positive" if pnl >= 0 else "negative"
                sign      = "+" if pnl >= 0 else ""

                r1, r2 = st.columns([5, 1])
                with r1:
                    st.markdown(
                        f"**{symbol}** {direction.upper()} &nbsp;|&nbsp; "
                        f"Entry: `{ep:,.4f}` &nbsp;|&nbsp; SL: `{sl:,.4f}` &nbsp;|&nbsp; "
                        f"<span class='{color}'>{sign}${pnl:,.2f}</span>",
                        unsafe_allow_html=True,
                    )
                with r2:
                    if st.button("✕ Close", key=f"close_{idx}_{ticket}"):
                        if ticket:
                            issue_close_command(ticket, symbol, trade.get("trade_id"))
                            st.success(f"Close command issued for {symbol}")
                        else:
                            st.warning("No ticket — cannot close from dashboard.")
            st.caption("ℹ️ Close button writes a command file; main.py executes the MT5 close.")
        else:
            st.info("No open positions.")

    with col_right:
        st.subheader("💰 Period P&L")
        gross_profit = sum(safe_float(t.get("pnl")) for t in win_trades)
        gross_loss   = sum(safe_float(t.get("pnl")) for t in loss_trades)
        avg_dur      = (sum(safe_float(t.get("duration_minutes")) for t in closed_filt) / n_trades
                        if n_trades else 0)

        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=round(total_pnl, 2),
            delta={"reference": 0, "valueformat": "$.2f",
                   "increasing": {"color": "#00cc44"}, "decreasing": {"color": "#ff4444"}},
            number={"prefix": "$", "valueformat": ",.2f", "font": {"size": 36}},
            title={"text": f"Net P&L ({time_range})"},
        ))
        fig.update_layout(height=160, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

        st.markdown(f"""
| | |
|---|---|
| ✅ Gross Profit | `${gross_profit:,.2f}` |
| ❌ Gross Loss   | `${gross_loss:,.2f}` |
| 🏆 Win / Loss  | `{len(win_trades)} / {len(loss_trades)}` |
| ⏱ Avg Duration | `{avg_dur:.0f} min` |
""")

    st.markdown("---")
    st.subheader("📝 Recent Closed Trades")
    recent = sorted(closed_filt, key=lambda t: t.get("entry_time") or "", reverse=True)[:8]
    if recent:
        for t in recent:
            pnl   = safe_float(t.get("pnl"))
            color = "positive" if pnl > 0 else ("negative" if pnl < 0 else "neutral")
            sign  = "+" if pnl > 0 else ""
            st.markdown(
                f"**{t.get('symbol','?')}** {t.get('direction','')} &nbsp;|&nbsp; "
                f"<span class='{color}'>{sign}${pnl:,.2f}</span> &nbsp;|&nbsp; "
                f"RR: {safe_float(t.get('realized_rr')):.2f} &nbsp;|&nbsp; "
                f"{t.get('exit_reason','?')} &nbsp;|&nbsp; "
                f"{str(t.get('entry_time',''))[:16]}",
                unsafe_allow_html=True,
            )
    else:
        st.info("No closed trades in this period.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRADES
# ══════════════════════════════════════════════════════════════════════════════

def show_trades_tab(db, closed_filt, open_trades):
    st.header("📈 Trade History")

    c1, c2, c3 = st.columns(3)
    with c1:
        syms       = sorted(set(t.get("symbol", "?") for t in closed_filt))
        sym_filter = st.selectbox("Symbol", ["All"] + syms)
    with c2:
        dir_filter = st.selectbox("Direction", ["All", "long", "short"])
    with c3:
        reason_filter = st.selectbox("Exit Reason",
            ["All", "stop_loss", "take_profit", "trailing_stop",
             "manual", "external_close", "ea"])

    display = closed_filt[:]
    if sym_filter    != "All": display = [t for t in display if t.get("symbol")      == sym_filter]
    if dir_filter    != "All": display = [t for t in display if t.get("direction")   == dir_filter]
    if reason_filter != "All": display = [t for t in display if t.get("exit_reason") == reason_filter]

    st.caption(f"{len(display)} trades shown")
    if not display:
        st.info("No trades match the current filters.")
        return

    df = pd.DataFrame(display)
    for col in ["pnl", "realized_rr", "entry_price", "exit_price",
                "duration_minutes", "commission", "slippage",
                "max_favorable_excursion", "max_adverse_excursion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    preferred  = ["symbol", "direction", "entry_time", "entry_price",
                  "exit_price", "pnl", "realized_rr", "exit_reason",
                  "duration_minutes", "status"]
    show_cols  = [c for c in preferred if c in df.columns]
    df_display = df[show_cols].copy()

    fmt = {
        "pnl":               lambda x: f"${x:,.2f}" if pd.notna(x) else "—",
        "realized_rr":       lambda x: f"{x:.2f}"   if pd.notna(x) else "—",
        "entry_price":       lambda x: f"{x:,.4f}"  if pd.notna(x) else "—",
        "exit_price":        lambda x: f"{x:,.4f}"  if pd.notna(x) else "—",
        "duration_minutes":  lambda x: f"{x:.0f}m"  if pd.notna(x) else "—",
        "entry_time":        lambda x: str(x)[:16],
    }
    for col, fn in fmt.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(fn)

    st.markdown("**Click a row to view full details.**")
    selection = st.dataframe(
        df_display,
        width="stretch",
        selection_mode="single-row",
        on_select="rerun",
        key="trade_table",
    )

    selected_rows = selection.selection.rows if hasattr(selection, "selection") else []
    if selected_rows:
        orig = display[selected_rows[0]]
        st.markdown("---")
        st.subheader(
            f"🔍 {orig.get('symbol','?')} {orig.get('direction','').upper()} — "
            f"{str(orig.get('entry_time',''))[:16]}"
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Entry Price",   f"{safe_float(orig.get('entry_price')):,.4f}")
            st.metric("Stop Loss",     f"{safe_float(orig.get('stop_loss')):,.4f}")
            st.metric("Take Profit 1", f"{safe_float(orig.get('take_profit_1')):,.4f}")
        with c2:
            st.metric("Exit Price",    f"{safe_float(orig.get('exit_price')):,.4f}")
            st.metric("Net P&L",       f"${safe_float(orig.get('pnl')):,.2f}")
            st.metric("P&L %",         f"{safe_float(orig.get('pnl_percent')):.3f}%")
        with c3:
            st.metric("Realized R:R",  f"{safe_float(orig.get('realized_rr')):.2f}")
            st.metric("Duration",      f"{safe_float(orig.get('duration_minutes')):.0f} min")
            st.metric("Exit Reason",   str(orig.get("exit_reason") or "—"))
        with c4:
            st.metric("Commission",    f"${safe_float(orig.get('commission')):.2f}")
            st.metric("Slippage",      f"{safe_float(orig.get('slippage')):.5f}")
            eq = orig.get("equity_after_close")
            st.metric("Equity After",  f"${safe_float(eq):,.2f}" if eq else "—")
        c1, c2 = st.columns(2)
        with c1: st.metric("Max Favorable Excursion",
                            f"{safe_float(orig.get('max_favorable_excursion')):.4f}")
        with c2: st.metric("Max Adverse Excursion",
                            f"{safe_float(orig.get('max_adverse_excursion')):.4f}")
        st.caption(
            f"Trade ID: `{orig.get('trade_id','?')}` | "
            f"Analysis ID: `{orig.get('analysis_id','?')}`"
        )

    st.markdown("---")
    csv = pd.DataFrame(display).to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "trades_filtered.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
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

    # ── Build equity series ───────────────────────────────────────────────────
    # After backfill_equity_once() all rows should have equity_after_close.
    # ffill + bfill handles any remaining edge-case gaps.
    has_real = df_sorted["equity_after_close"].notna().any()

    if has_real:
        df_sorted["equity_plot"] = (
            df_sorted["equity_after_close"].ffill().bfill()
        )
        equity_caption = "📡 Real account equity — sourced from MT5 balance snapshots"
        equity_warning = False
    else:
        # No snapshots yet — use cumsum anchored to config starting equity
        try:
            cfg      = load_config()
            start_eq = float(cfg.get("general", {}).get("starting_equity", 100_000.0))
        except Exception:
            start_eq = 100_000.0
        df_sorted["equity_plot"] = start_eq + df_sorted["pnl"].cumsum()
        equity_caption = ("⚠️ Estimated equity (P&L cumsum from configured starting equity — "
                          "run the updated system to get real values)")
        equity_warning = True

    df_sorted["peak"]     = df_sorted["equity_plot"].cummax()
    df_sorted["drawdown"] = df_sorted["equity_plot"] - df_sorted["peak"]

    # ── Equity Curve chart ────────────────────────────────────────────────────
    st.subheader("Equity Curve")
    if equity_warning:
        st.warning(equity_caption)
    else:
        st.caption(equity_caption)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["exit_time"], y=df_sorted["equity_plot"],
        mode="lines", name="Equity",
        line=dict(color="#2196f3", width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.07)",
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Equity: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_sorted["exit_time"], y=df_sorted["peak"],
        mode="lines", name="Peak",
        line=dict(color="rgba(0,200,0,0.5)", width=1, dash="dash"),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Peak: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_sorted["exit_time"], y=df_sorted["drawdown"],
        mode="lines", name="Drawdown",
        line=dict(color="rgba(220,50,50,0.6)", width=1),
        fill="tozeroy", fillcolor="rgba(220,50,50,0.06)",
        yaxis="y2",
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>DD: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Date (UTC)", yaxis_title="Equity ($)",
        yaxis2=dict(title="Drawdown ($)", overlaying="y", side="right", showgrid=False),
        hovermode="x unified", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Equity summary ────────────────────────────────────────────────────────
    max_dd     = df_sorted["drawdown"].min()
    max_dd_pct = (max_dd / df_sorted["peak"].max() * 100) if df_sorted["peak"].max() > 0 else 0
    current_eq = df_sorted["equity_plot"].iloc[-1] if not df_sorted.empty else 0

    d1, d2, d3 = st.columns(3)
    with d1: st.metric("Current Equity",   f"${current_eq:,.2f}")
    with d2: st.metric("Max Drawdown",     f"${max_dd:,.2f}",
                        delta=f"{max_dd_pct:.2f}%", delta_color="inverse")
    with d3: st.metric("Net P&L (period)", f"${df_sorted['pnl'].sum():,.2f}")

    st.markdown("---")

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.subheader("Summary Statistics")
    wins   = df[df["pnl"] > 0]["pnl"]
    losses = df[df["pnl"] < 0]["pnl"]
    n      = len(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Trades", n)
        st.metric("Win Rate", f"{len(wins)/n*100:.1f}%" if n else "—")
    with c2:
        st.metric("Net P&L",   f"${df['pnl'].sum():,.2f}")
        st.metric("Avg Trade", f"${df['pnl'].mean():,.2f}")
    with c3:
        st.metric("Best Trade",  f"${df['pnl'].max():,.2f}")
        st.metric("Worst Trade", f"${df['pnl'].min():,.2f}")
    with c4:
        st.metric("Avg Win",  f"${wins.mean():,.2f}"  if not wins.empty  else "—")
        st.metric("Avg Loss", f"${losses.mean():,.2f}" if not losses.empty else "—")

    st.markdown("---")

    # ── Per-symbol + distribution ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("By Symbol")
        if "symbol" in df.columns:
            sym_stats = df.groupby("symbol").agg(
                trades    = ("pnl", "count"),
                total_pnl = ("pnl", "sum"),
                avg_rr    = ("realized_rr", "mean"),
            ).round(2)
            st.dataframe(sym_stats, width="stretch")
    with col2:
        st.subheader("Win / Loss Distribution")
        fig2 = go.Figure()
        if not wins.empty:
            fig2.add_trace(go.Histogram(x=wins,   name="Wins",   marker_color="#00cc44", opacity=0.7))
        if not losses.empty:
            fig2.add_trace(go.Histogram(x=losses, name="Losses", marker_color="#ff4444", opacity=0.7))
        fig2.update_layout(barmode="overlay", height=280,
                           xaxis_title="P&L ($)", yaxis_title="Count")
        st.plotly_chart(fig2, width="stretch")

    # ── Time-based ────────────────────────────────────────────────────────────
    st.subheader("Time-Based Analysis")
    df_t = df.dropna(subset=["entry_time"]).copy()
    df_t["hour"]        = df_t["entry_time"].dt.hour
    df_t["day_of_week"] = df_t["entry_time"].dt.day_name()

    if not df_t.empty:
        col1, col2 = st.columns(2)
        with col1:
            hour_pnl = df_t.groupby("hour")["pnl"].sum()
            fig3 = px.bar(x=hour_pnl.index, y=hour_pnl.values,
                          labels={"x": "Hour (UTC)", "y": "P&L ($)"},
                          title="P&L by Hour",
                          color=hour_pnl.values,
                          color_continuous_scale=["red", "yellow", "green"])
            st.plotly_chart(fig3, width="stretch")
        with col2:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            day_pnl   = df_t.groupby("day_of_week")["pnl"].sum()
            day_pnl   = day_pnl.reindex([d for d in day_order if d in day_pnl.index])
            fig4 = px.bar(x=day_pnl.index, y=day_pnl.values,
                          labels={"x": "Day", "y": "P&L ($)"},
                          title="P&L by Day of Week",
                          color=day_pnl.values,
                          color_continuous_scale=["red", "yellow", "green"])
            st.plotly_chart(fig4, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def show_configuration_tab(config: dict):
    st.header("⚙️ System Configuration")
    st.info("Changes write to config/config.yaml. **Restart main.py to apply.**")

    rm  = config.get("risk_management", {})
    gl  = rm.get("global_limits", {})
    sl  = rm.get("stop_loss", {})
    tr  = rm.get("trailing_stop", {})
    stg = config.get("strategy", {})

    with st.form("config_form"):
        st.subheader("⚠️ Risk Management")
        c1, c2 = st.columns(2)
        with c1:
            max_risk   = st.number_input("Max Risk Per Trade (%)", 0.1, 5.0, step=0.1,
                value=float(clamp(rm.get("max_risk_percent_per_trade", 1.0), 0.1, 5.0)))
            max_dd_pct = st.number_input("Max Daily Drawdown (%)", 1.0, 20.0, step=0.5,
                value=float(clamp(gl.get("daily_max_drawdown_percent", 5.0), 1.0, 20.0)))
        with c2:
            max_conc = st.number_input("Max Concurrent Trades", 1, 20, step=1,
                value=int(clamp(gl.get("max_concurrent_trades", 3), 1, 20)))
            max_day  = st.number_input("Max Trades Per Day", 1, 999, step=1,
                value=int(clamp(gl.get("max_trades_per_day", 10), 1, 999)))

        st.subheader("🎯 Strategy Parameters")
        c1, c2 = st.columns(2)
        with c1:
            atr_mult   = st.number_input("ATR Multiplier (SL)", 0.5, 10.0, step=0.1,
                value=float(clamp(sl.get("atr_multiplier", 2.0), 0.5, 10.0)))
            confluence = st.number_input("Confluence Required", 1, 5, step=1,
                value=int(clamp(stg.get("confluence_required", 2), 1, 5)))
        with c2:
            trail_rr  = st.number_input("Trailing Activation R:R", 0.1, 5.0, step=0.1,
                value=float(clamp(tr.get("activation_rr", 1.0), 0.1, 5.0)))
            sl_methods = ["conservative", "atr", "structure"]
            cur_method = sl.get("method", "conservative")
            sl_method  = st.selectbox("Stop Loss Method", sl_methods,
                index=sl_methods.index(cur_method) if cur_method in sl_methods else 0)

        st.subheader("⏱️ Cooldown Settings")
        cooldown = gl.get("cooldown_after_losses", {})
        c1, c2 = st.columns(2)
        with c1:
            cd_losses = st.number_input("Losses Before Cooldown", 1, 10, step=1,
                value=int(clamp(cooldown.get("consecutive_losses", 3), 1, 10)))
        with c2:
            cd_secs = st.number_input("Cooldown Duration (seconds)", 60, 86400, step=60,
                value=int(clamp(cooldown.get("cooldown_seconds", 3600), 60, 86400)))

        submitted = st.form_submit_button("💾 Save Configuration", type="primary")

    if submitted:
        config.setdefault("risk_management", {})["max_risk_percent_per_trade"] = max_risk
        config["risk_management"].setdefault("stop_loss", {}).update(
            {"atr_multiplier": atr_mult, "method": sl_method})
        config["risk_management"].setdefault("trailing_stop", {})["activation_rr"] = trail_rr
        config.setdefault("strategy", {})["confluence_required"] = confluence
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
        if save_config(config):
            st.success("✅ Configuration saved. Restart main.py to apply.")
            st.cache_data.clear()

    st.markdown("---")
    st.subheader("📚 Parameter Versions")
    db = init_database()
    versions = load_parameter_versions(db)
    if versions:
        rows = []
        for v in versions:
            metrics = {}
            try:
                metrics = json.loads(v.get("backtest_metrics") or "{}")
            except Exception:
                pass
            rows.append({
                "ID":         v.get("version_id"),
                "Name":       v.get("version_name"),
                "Created":    str(v.get("created_at",""))[:16],
                "Source":     v.get("source"),
                "Status":     v.get("status"),
                "Win Rate":   f"{safe_float(metrics.get('win_rate'))*100:.1f}%"
                              if metrics.get("win_rate") else "—",
                "Expectancy": f"{safe_float(metrics.get('expectancy')):.2f}"
                              if metrics.get("expectancy") else "—",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No parameter versions yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LEARNING
# ══════════════════════════════════════════════════════════════════════════════

def show_learning_tab(db):
    st.header("🧠 Learning Engine")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("30-Day Performance Metrics")
        try:
            stats = db.get_trade_statistics(days=30)
            st.metric("Total Trades", stats.get("total_trades", 0))
            st.metric("Win Rate",     f"{safe_float(stats.get('win_rate'))*100:.1f}%")
            st.metric("Avg P&L",      f"${safe_float(stats.get('avg_pnl')):.2f}")
            st.metric("Total P&L",    f"${safe_float(stats.get('total_pnl')):.2f}")
            st.metric("Avg R:R",      f"{safe_float(stats.get('avg_rr')):.2f}")
            st.metric("Avg Duration", f"{safe_float(stats.get('avg_duration_minutes')):.0f} min")
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.subheader("Optimization Controls")
        st.info("The learning engine analyses closed trades and suggests better parameters.")
        try:
            from learning.learner import StrategyLearner as Learner
            cfg     = load_config()
            learner = Learner(db, cfg)
            ca, cb  = st.columns(2)
            with ca:
                if st.button("🔍 Run Grid Search"):
                    with st.spinner("Running..."):
                        result = learner.run_grid_search()
                    st.success(result.get("message", "Done"))
                    st.cache_data.clear()
            with cb:
                if st.button("🎰 Run RL Bandit"):
                    with st.spinner("Running..."):
                        result = learner.run_rl_bandit()
                    st.success(result.get("message", "Done"))
                    st.cache_data.clear()
        except Exception as e:
            st.warning(f"Learner unavailable: {e}")

    st.markdown("---")
    st.subheader("Recent Learning Runs")
    try:
        cur = db.conn.cursor()
        cur.execute("SELECT * FROM learning_runs ORDER BY started_at DESC LIMIT 10")
        runs = [dict(r) for r in cur.fetchall()]
        if runs:
            st.dataframe(pd.DataFrame(runs), width="stretch")
        else:
            st.info("No learning runs recorded yet.")
    except Exception as e:
        st.info(f"Learning runs table not available: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    db     = init_database()
    config = load_config()

    # One-time equity backfill per browser session
    if not st.session_state.get("equity_backfill_done", False):
        backfill_equity_once("data/trading.db")
        st.session_state["equity_backfill_done"] = True

    # ── Sidebar: controls + time filter ───────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.markdown("---")

        current_mode = config.get("general", {}).get("mode", "demo")
        is_live      = current_mode == "live"
        badge = ('<span class="mode-live">⚡ LIVE TRADING</span>' if is_live
                 else '<span class="mode-demo">🧪 DEMO MODE</span>')
        st.markdown(badge, unsafe_allow_html=True)

        new_mode = st.radio("Mode", ["demo", "live"], index=1 if is_live else 0,
                            horizontal=True,
                            help="Writes to config.yaml — restart main.py to apply.")
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
                    st.success("Set to DEMO. Restart main.py.")
                    st.cache_data.clear()

        st.markdown("---")
        st.markdown("**Quick Actions**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⏸️ Pause"):
                (CONTROL_DIR / "pause.flag").touch()
                st.warning("Paused.")
        with c2:
            if st.button("▶️ Resume"):
                flag = CONTROL_DIR / "pause.flag"
                if flag.exists():
                    flag.unlink()
                st.success("Resumed.")
        if st.button("🚨 Emergency Close All", type="primary"):
            (CONTROL_DIR / "emergency_close.flag").touch()
            st.error("Emergency close flag set!")

        st.markdown("---")
        st.markdown("**Time Filter**")
        time_range = st.selectbox(
            "Period",
            ["Today", "Last 7 Days", "Last 30 Days", "All Time", "Custom"],
            index=0, label_visibility="collapsed",
        )
        custom_start = custom_end = None
        if time_range == "Custom":
            custom_start = st.date_input("From", value=date.today() - timedelta(days=7))
            custom_end   = st.date_input("To",   value=date.today())
            if custom_start > custom_end:
                st.error("Start must be before end.")

        start_dt, end_dt = get_time_bounds(time_range, custom_start, custom_end)
        st.caption(
            f"{start_dt.strftime('%Y-%m-%d %H:%M')} →\n"
            f"{end_dt.strftime('%Y-%m-%d %H:%M')} UTC"
        )

        st.markdown("---")
        if st.checkbox("Auto-refresh (30s)", value=False):
            import time as _t; _t.sleep(0.5)
            st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    all_trades  = load_all_trades(db)
    open_trades = load_open_trades(db)
    filtered    = filter_trades_by_time(all_trades, start_dt, end_dt)
    closed_filt = [t for t in filtered if t.get("status") == "closed"]

    # ── Top nav bar ───────────────────────────────────────────────────────────
    active_page = render_top_nav()

    # ── Route ─────────────────────────────────────────────────────────────────
    if active_page == "📊 Overview":
        show_overview_tab(db, open_trades, closed_filt, time_range, start_dt, end_dt)
    elif active_page == "📈 Trades":
        show_trades_tab(db, closed_filt, open_trades)
    elif active_page == "📉 Analytics":
        show_analytics_tab(closed_filt)
    elif active_page == "⚙️ Config":
        show_configuration_tab(config)
    elif active_page == "🧠 Learning":
        show_learning_tab(db)


if __name__ == "__main__":
    main()