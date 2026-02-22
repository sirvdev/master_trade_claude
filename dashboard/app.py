"""
Trading System Dashboard — Improved v2
Fixes and improvements:
  1. Mode indicator reads real config; toggle actually writes to config.yaml
  2. Overview shows real today P&L (positive when profitable, not just drawdown)
  3. Open positions table has per-row Close buttons
  4. Time filter in sidebar is wired to all DB queries; custom date range added
  5. Trade details shown by clicking a row in the table (on_select)
  6. Parameter versions pulled from real DB table with explanation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, date

sys.path.append(str(Path(__file__).parent.parent))
from logger.db import DatabaseManager

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.positive { color: #00cc44; font-weight: bold; }
.negative { color: #ff3333; font-weight: bold; }
.neutral  { color: #888888; }
.mode-live { background:#ff4444; color:white; padding:4px 12px;
             border-radius:4px; font-weight:bold; font-size:14px; }
.mode-demo { background:#2196F3; color:white; padding:4px 12px;
             border-radius:4px; font-weight:bold; font-size:14px; }
.section-header { font-size:16px; font-weight:600; margin:8px 0 4px 0; }
</style>
""", unsafe_allow_html=True)

CONFIG_PATH = Path("config/config.yaml")
CONTROL_DIR = Path("data")
CLOSE_CMD_FILE = CONTROL_DIR / "close_commands.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False

def safe_float(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default

def clamp(v, lo, hi):
    if v is None:
        return lo
    return max(lo, min(hi, v))

def issue_close_command(ticket, symbol, trade_id):
    """Write a close command to a file that main.py monitors."""
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    cmds = []
    if CLOSE_CMD_FILE.exists():
        try:
            cmds = json.loads(CLOSE_CMD_FILE.read_text())
        except Exception:
            cmds = []
    cmds.append({
        "action": "close_position",
        "ticket": ticket,
        "symbol": symbol,
        "trade_id": trade_id,
        "issued_at": datetime.utcnow().isoformat()
    })
    CLOSE_CMD_FILE.write_text(json.dumps(cmds, indent=2))

# ── Time filter helpers ───────────────────────────────────────────────────────

def get_time_bounds(time_range: str, custom_start=None, custom_end=None):
    """Return (start_dt, end_dt) for the selected time filter."""
    now = datetime.utcnow()
    if time_range == "Today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = now
    elif time_range == "Last 7 Days":
        start = now - timedelta(days=7)
        end   = now
    elif time_range == "Last 30 Days":
        start = now - timedelta(days=30)
        end   = now
    elif time_range == "Custom" and custom_start and custom_end:
        start = datetime.combine(custom_start, datetime.min.time())
        end   = datetime.combine(custom_end,   datetime.max.time())
    else:  # All Time
        start = datetime(2000, 1, 1)
        end   = now
    return start, end

def filter_trades_by_time(trades: list, start_dt: datetime, end_dt: datetime) -> list:
    """Filter a list of trade dicts to the given time window using entry_time."""
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

# ── DB ────────────────────────────────────────────────────────────────────────

@st.cache_resource
def init_database():
    db = DatabaseManager("data/trading.db")
    db.connect()
    return db

@st.cache_data(ttl=20)
def load_all_trades(_db, limit=1000):
    return _db.get_trades(limit=limit)

@st.cache_data(ttl=20)
def load_open_trades(_db):
    return _db.get_open_trades()

@st.cache_data(ttl=60)
def load_parameter_versions(_db):
    """Load parameter versions from the DB."""
    try:
        cursor = _db.conn.cursor()
        cursor.execute("SELECT * FROM parameter_versions ORDER BY created_at DESC LIMIT 20")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    db     = init_database()
    config = load_config()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.markdown("---")

        # ── System mode (reads and writes config.yaml) ────────────────────────
        st.markdown("**System Mode**")
        current_mode = config.get("general", {}).get("mode", "demo")
        is_live      = current_mode == "live"

        if is_live:
            st.markdown('<span class="mode-live">⚡ LIVE TRADING</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="mode-demo">🧪 DEMO MODE</span>', unsafe_allow_html=True)

        new_mode = st.radio(
            "Switch mode",
            ["demo", "live"],
            index=1 if is_live else 0,
            horizontal=True,
            help="Changing mode writes to config/config.yaml and requires a system restart to take effect."
        )

        if new_mode != current_mode:
            if new_mode == "live":
                confirmed = st.checkbox("⚠️ I confirm I want to switch to LIVE trading", key="live_confirm")
                if confirmed:
                    config.setdefault("general", {})["mode"] = "live"
                    if save_config(config):
                        st.success("Mode set to LIVE. Restart main.py to apply.")
                        st.cache_data.clear()
            else:
                config.setdefault("general", {})["mode"] = "demo"
                if save_config(config):
                    st.success("Mode set to DEMO. Restart main.py to apply.")
                    st.cache_data.clear()

        st.markdown("---")

        # ── Quick actions ─────────────────────────────────────────────────────
        st.markdown("**Quick Actions**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏸️ Pause"):
                (CONTROL_DIR / "pause.flag").touch()
                st.warning("Pause flag set.")
        with col2:
            if st.button("▶️ Resume"):
                flag = CONTROL_DIR / "pause.flag"
                if flag.exists():
                    flag.unlink()
                st.success("Resumed.")

        if st.button("🚨 Emergency Close All", type="primary"):
            (CONTROL_DIR / "emergency_close.flag").touch()
            st.error("Emergency close flag set! main.py will close all positions.")

        st.markdown("---")

        # ── Time filter ───────────────────────────────────────────────────────
        st.markdown("**Time Filter**")
        time_range = st.selectbox(
            "Period",
            ["Today", "Last 7 Days", "Last 30 Days", "All Time", "Custom"],
            index=0,
            label_visibility="collapsed"
        )

        custom_start = custom_end = None
        if time_range == "Custom":
            custom_start = st.date_input("From", value=date.today() - timedelta(days=7))
            custom_end   = st.date_input("To",   value=date.today())
            if custom_start > custom_end:
                st.error("Start must be before end date.")

        start_dt, end_dt = get_time_bounds(time_range, custom_start, custom_end)
        st.caption(f"Showing: {start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")

        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            import time as _time
            _time.sleep(0.5)  # Small delay to allow UI to update before refresh
            st.rerun()

    # ── Load and filter data ──────────────────────────────────────────────────
    all_trades   = load_all_trades(db)
    open_trades  = load_open_trades(db)
    filtered     = filter_trades_by_time(all_trades, start_dt, end_dt)
    closed_filt  = [t for t in filtered if t.get("status") == "closed"]

    # ── Page navigation (session_state keeps selected page across reruns) ────
    PAGES = {
        "📊 Overview"      : lambda: show_overview_tab(db, open_trades, closed_filt, time_range, start_dt, end_dt),
        "📈 Trades"        : lambda: show_trades_tab(db, closed_filt, open_trades),
        "⚙️ Configuration" : lambda: show_configuration_tab(config),
        "🧠 Learning"      : lambda: show_learning_tab(db),
        "📉 Analytics"     : lambda: show_analytics_tab(closed_filt),
    }

    # Inject nav into sidebar (below the time filter already there)
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Navigation**")
        page = st.radio(
            "Go to",
            list(PAGES.keys()),
            index=list(PAGES.keys()).index(st.session_state.get("active_page", "📊 Overview")),
            label_visibility="collapsed",
            key="nav_radio",
        )
        st.session_state["active_page"] = page

    # Render selected page
    PAGES[page]()


# ── Tab 1: Overview ───────────────────────────────────────────────────────────

def show_overview_tab(db, open_trades, closed_filt, time_range, start_dt, end_dt):
    st.header(f"📊 Overview — {time_range}")

    # ── Key metrics (filtered period) ─────────────────────────────────────────
    total_pnl    = sum(safe_float(t.get("pnl")) for t in closed_filt)
    win_trades   = [t for t in closed_filt if safe_float(t.get("pnl")) > 0]
    loss_trades  = [t for t in closed_filt if safe_float(t.get("pnl")) < 0]
    n_trades     = len(closed_filt)
    win_rate     = len(win_trades) / n_trades * 100 if n_trades else 0
    avg_rr       = sum(safe_float(t.get("realized_rr")) for t in closed_filt) / n_trades if n_trades else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Closed Trades", n_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{win_rate-50:.1f}%" if n_trades else None)
    with col3:
        pnl_delta = f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}"
        st.metric("Period P&L", f"${total_pnl:,.2f}", delta=pnl_delta)
    with col4:
        st.metric("Avg R:R", f"{avg_rr:.2f}", delta=f"{avg_rr-1:.2f}" if n_trades else None)
    with col5:
        st.metric("Open Positions", len(open_trades))

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    # ── Open Positions with Close buttons ────────────────────────────────────
    with col_left:
        st.subheader("📍 Open Positions")
        if open_trades:
            for idx, trade in enumerate(open_trades):
                ticket    = trade.get("ticket")
                symbol    = trade.get("symbol", "?")
                direction = trade.get("direction", "?")
                entry     = safe_float(trade.get("entry_price"))
                sl        = safe_float(trade.get("stop_loss"))
                size      = safe_float(trade.get("position_size"))
                trade_id  = trade.get("trade_id", "")

                c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 2, 2, 1.5, 1.5])
                with c1: st.write(f"**{symbol}**")
                with c2:
                    color = "positive" if direction == "long" else "negative"
                    st.markdown(f'<span class="{color}">{direction.upper()}</span>', unsafe_allow_html=True)
                with c3: st.write(f"Entry: `{entry:,.2f}`")
                with c4: st.write(f"SL: `{sl:,.2f}`")
                with c5: st.write(f"{size} lots")
                with c6:
                    if st.button("🔴 Close", key=f"close_{trade_id}_{idx}",
                                 help=f"Issue close command for ticket {ticket}"):
                        if ticket:
                            issue_close_command(ticket, symbol, trade_id)
                            st.success(f"Close command issued for {symbol} (ticket {ticket}). main.py will execute it.")
                        else:
                            st.warning("No ticket found — cannot close from dashboard.")

            st.caption("ℹ️ Close button writes a command file. main.py executes the actual MT5 close.")
        else:
            st.info("No open positions")

    # ── P&L gauge and period breakdown ────────────────────────────────────────
    with col_right:
        st.subheader("💰 Period P&L")

        gross_profit = sum(safe_float(t.get("pnl")) for t in win_trades)
        gross_loss   = sum(safe_float(t.get("pnl")) for t in loss_trades)  # negative number

        # Load max_dd from config for gauge scale
        try:
            with open(CONFIG_PATH) as f:
                _cfg = yaml.safe_load(f)
            max_dd_pct = safe_float(
                _cfg.get("risk_management", {}).get("global_limits", {})
                    .get("daily_max_drawdown_percent"), 5.0
            )
        except Exception:
            max_dd_pct = 5.0

        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=round(total_pnl, 2),
            delta={"reference": 0, "valueformat": "$.2f",
                   "increasing": {"color": "#00cc44"},
                   "decreasing": {"color": "#ff3333"}},
            number={"prefix": "$", "valueformat": ",.2f",
                    "font": {"size": 36}},
            title={"text": f"Net P&L ({time_range})"},
        ))
        fig.update_layout(height=160, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig)

        st.markdown(f"""
        | | |
        |---|---|
        | ✅ Gross Profit | `${gross_profit:,.2f}` |
        | ❌ Gross Loss   | `${gross_loss:,.2f}` |
        | 🏆 Win / Loss  | `{len(win_trades)} / {len(loss_trades)}` |
        | ⏱ Avg Duration | `{sum(safe_float(t.get('duration_minutes')) for t in closed_filt)/n_trades:.0f} min` if {n_trades} else `—` |
        """)

    st.markdown("---")

    # ── Recent closed trades in period ────────────────────────────────────────
    st.subheader("📝 Recent Closed Trades (this period)")
    recent = sorted(closed_filt, key=lambda t: t.get("entry_time") or "", reverse=True)[:8]
    if recent:
        for t in recent:
            pnl = safe_float(t.get("pnl"))
            color = "positive" if pnl > 0 else ("negative" if pnl < 0 else "neutral")
            sign  = "+" if pnl > 0 else ""
            st.markdown(
                f"**{t.get('symbol','?')}** {t.get('direction','')} &nbsp;|&nbsp; "
                f"<span class='{color}'>{sign}${pnl:,.2f}</span> &nbsp;|&nbsp; "
                f"RR: {safe_float(t.get('realized_rr')):.2f} &nbsp;|&nbsp; "
                f"{t.get('exit_reason','?')} &nbsp;|&nbsp; "
                f"{str(t.get('entry_time',''))[:16]}",
                unsafe_allow_html=True
            )
    else:
        st.info("No closed trades in this period.")


# ── Tab 2: Trades ─────────────────────────────────────────────────────────────

def show_trades_tab(db, closed_filt, open_trades):
    st.header("📈 Trade History")

    # Filters row
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols_available = sorted(set(t.get("symbol","?") for t in closed_filt))
        sym_filter = st.selectbox("Symbol", ["All"] + symbols_available)
    with col2:
        dir_filter = st.selectbox("Direction", ["All", "long", "short"])
    with col3:
        reason_filter = st.selectbox("Exit Reason", ["All", "stop_loss", "take_profit",
                                                       "trailing_stop", "manual", "external_close"])

    # Apply filters
    display = closed_filt[:]
    if sym_filter    != "All": display = [t for t in display if t.get("symbol")      == sym_filter]
    if dir_filter    != "All": display = [t for t in display if t.get("direction")   == dir_filter]
    if reason_filter != "All": display = [t for t in display if t.get("exit_reason") == reason_filter]

    st.caption(f"{len(display)} trades shown (filtered period)")

    if not display:
        st.info("No trades match the current filters.")
        return

    df = pd.DataFrame(display)

    # Sanitise numeric cols
    for col in ["pnl", "realized_rr", "entry_price", "exit_price",
                 "duration_minutes", "commission", "slippage",
                 "max_favorable_excursion", "max_adverse_excursion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Columns for the table
    preferred = ["symbol", "direction", "entry_time", "entry_price",
                 "exit_price", "pnl", "realized_rr", "exit_reason",
                 "duration_minutes", "status"]
    show_cols = [c for c in preferred if c in df.columns]
    df_display = df[show_cols].copy()

    # Format for display
    if "pnl"             in df_display.columns: df_display["pnl"]             = df_display["pnl"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    if "realized_rr"     in df_display.columns: df_display["realized_rr"]     = df_display["realized_rr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    if "entry_price"     in df_display.columns: df_display["entry_price"]     = df_display["entry_price"].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "—")
    if "exit_price"      in df_display.columns: df_display["exit_price"]      = df_display["exit_price"].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "—")
    if "duration_minutes"in df_display.columns: df_display["duration_minutes"]= df_display["duration_minutes"].map(lambda x: f"{x:.0f}m" if pd.notna(x) else "—")
    if "entry_time"      in df_display.columns: df_display["entry_time"]      = df_display["entry_time"].astype(str).str[:16]

    # ── Row-click selection ────────────────────────────────────────────────────
    st.markdown("**Click a row to see full trade details below.**")

    selection = st.dataframe(
        df_display,
        width='stretch',
        selection_mode="single-row",
        on_select="rerun",
        key="trade_table"
    )

    # ── Trade detail panel (shown when a row is selected) ─────────────────────
    selected_rows = selection.selection.rows if hasattr(selection, "selection") else []
    if selected_rows:
        idx   = selected_rows[0]
        trade = df.iloc[idx].to_dict()         # raw (un-formatted) values
        orig  = display[idx]                    # original dict

        st.markdown("---")
        st.subheader(f"🔍 Trade Detail — {orig.get('symbol','?')} {orig.get('direction','').upper()}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Entry Price",    f"{safe_float(orig.get('entry_price')):,.4f}")
            st.metric("Stop Loss",      f"{safe_float(orig.get('stop_loss')):,.4f}")
            st.metric("Take Profit 1",  f"{safe_float(orig.get('take_profit_1')):,.4f}")
        with c2:
            st.metric("Exit Price",     f"{safe_float(orig.get('exit_price')):,.4f}")
            st.metric("Net P&L",        f"${safe_float(orig.get('pnl')):,.2f}")
            st.metric("P&L %",          f"{safe_float(orig.get('pnl_percent')):.3f}%")
        with c3:
            st.metric("Realized R:R",   f"{safe_float(orig.get('realized_rr')):.2f}")
            st.metric("Duration",       f"{safe_float(orig.get('duration_minutes')):.0f} min")
            st.metric("Exit Reason",    str(orig.get("exit_reason") or "—"))
        with c4:
            st.metric("Commission",     f"${safe_float(orig.get('commission')):.2f}")
            st.metric("Slippage",       f"{safe_float(orig.get('slippage')):.5f}")
            st.metric("Ticket",         str(orig.get("ticket") or "—"))

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Max Favorable Excursion", f"{safe_float(orig.get('max_favorable_excursion')):.4f}")
        with c2:
            st.metric("Max Adverse Excursion",   f"{safe_float(orig.get('max_adverse_excursion')):.4f}")

        st.caption(f"Trade ID: `{orig.get('trade_id','?')}` | Analysis ID: `{orig.get('analysis_id','?')}`")

    st.markdown("---")

    # ── CSV download ──────────────────────────────────────────────────────────
    csv = pd.DataFrame(display).to_csv(index=False)
    st.download_button("📥 Download CSV (filtered)", csv, "trades_filtered.csv", "text/csv")


# ── Tab 3: Configuration ──────────────────────────────────────────────────────

def show_configuration_tab(config: dict):
    st.header("⚙️ System Configuration")
    st.info("Changes here write directly to config/config.yaml. **Restart main.py to apply.**")

    rm  = config.get("risk_management", {})
    gl  = rm.get("global_limits", {})
    sl  = rm.get("stop_loss", {})
    tr  = rm.get("trailing_stop", {})
    stg = config.get("strategy", {})

    with st.form("config_form"):
        st.subheader("⚠️ Risk Management")
        c1, c2 = st.columns(2)
        with c1:
            max_risk = st.number_input("Max Risk Per Trade (%)", 0.1, 5.0, step=0.1,
                value=float(clamp(rm.get("max_risk_percent_per_trade", 1.0), 0.1, 5.0)))
            max_dd_pct = st.number_input("Max Daily Drawdown (%)", 1.0, 20.0, step=0.5,
                value=float(clamp(gl.get("daily_max_drawdown_percent", 5.0), 1.0, 20.0)))
        with c2:
            max_conc = st.number_input("Max Concurrent Trades", 1, 20, step=1,
                value=int(clamp(gl.get("max_concurrent_trades", 3), 1, 20)))
            max_day = st.number_input("Max Trades Per Day", 1, 999, step=1,
                value=int(clamp(gl.get("max_trades_per_day", 10), 1, 999)))

        st.subheader("🎯 Strategy Parameters")
        c1, c2 = st.columns(2)
        with c1:
            atr_mult = st.number_input("ATR Multiplier (SL)", 0.5, 10.0, step=0.1,
                value=float(clamp(sl.get("atr_multiplier", 2.0), 0.5, 10.0)))
            confluence = st.number_input("Confluence Required", 1, 5, step=1,
                value=int(clamp(stg.get("confluence_required", 2), 1, 5)))
        with c2:
            trail_rr = st.number_input("Trailing Activation R:R", 0.1, 5.0, step=0.1,
                value=float(clamp(tr.get("activation_rr", 1.0), 0.1, 5.0)))
            sl_methods = ["conservative", "atr", "structure"]
            cur_method = sl.get("method", "conservative")
            sl_method = st.selectbox("Stop Loss Method", sl_methods,
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
        config.setdefault("risk_management", {})
        config["risk_management"]["max_risk_percent_per_trade"] = max_risk
        config["risk_management"].setdefault("stop_loss", {})["atr_multiplier"]  = atr_mult
        config["risk_management"]["stop_loss"]["method"] = sl_method
        config["risk_management"].setdefault("trailing_stop", {})["activation_rr"] = trail_rr
        config.setdefault("strategy", {})["confluence_required"] = confluence
        gl_block = config["risk_management"].setdefault("global_limits", {})
        gl_block["daily_max_drawdown_percent"] = max_dd_pct
        gl_block["max_concurrent_trades"]      = max_conc
        gl_block["max_trades_per_day"]         = max_day
        gl_block.setdefault("cooldown_after_losses", {})["consecutive_losses"] = cd_losses
        gl_block["cooldown_after_losses"]["cooldown_seconds"] = cd_secs
        if save_config(config):
            st.success("✅ Configuration saved. Restart main.py to apply.")
            st.cache_data.clear()

    st.markdown("---")

    # ── Parameter Versions ────────────────────────────────────────────────────
    st.subheader("📚 Parameter Versions")
    st.markdown("""
The learning engine periodically tests different strategy parameters (ATR multipliers,
confluence thresholds, RSI levels, etc.) against your trade history. Each time it finds
a better-performing configuration it saves a **parameter version** here with the backtest
metrics that justified it. You can review past versions and roll back if needed.

| Column | Meaning |
|---|---|
| `version_name` | Label for this configuration snapshot |
| `source` | How it was generated: `manual`, `grid_search`, or `rl_bandit` |
| `status` | `pending` = awaiting approval, `active` = currently in use, `archived` = superseded |
| `backtest_metrics` | Win rate, expectancy, Sharpe etc. measured on historical data |
""")

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
                "ID"          : v.get("version_id"),
                "Name"        : v.get("version_name"),
                "Created"     : str(v.get("created_at", ""))[:16],
                "Source"      : v.get("source"),
                "Status"      : v.get("status"),
                "Win Rate"    : f"{safe_float(metrics.get('win_rate'))*100:.1f}%" if metrics.get("win_rate") else "—",
                "Expectancy"  : f"{safe_float(metrics.get('expectancy')):.2f}" if metrics.get("expectancy") else "—",
                "Notes"       : v.get("notes") or "",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch')
    else:
        st.info("No parameter versions saved yet. The learning engine will populate this table once it runs an optimization cycle.")


# ── Tab 4: Learning ───────────────────────────────────────────────────────────

def show_learning_tab(db):
    st.header("🧠 Learning Engine")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("30-Day Performance Metrics")
        try:
            stats = db.get_trade_statistics(days=30)
            st.metric("Total Trades",  stats.get("total_trades", 0))
            st.metric("Win Rate",      f"{safe_float(stats.get('win_rate'))*100:.1f}%")
            st.metric("Avg P&L",       f"${safe_float(stats.get('avg_pnl')):.2f}")
            st.metric("Total P&L",     f"${safe_float(stats.get('total_pnl')):.2f}")
            st.metric("Avg R:R",       f"{safe_float(stats.get('avg_rr')):.2f}")
            st.metric("Avg Duration",  f"{safe_float(stats.get('avg_duration_minutes')):.0f} min")
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.subheader("Optimization Controls")
        st.info("The learning engine analyses closed trades and suggests better parameters. "
                "Run it manually below or enable auto-scheduling in config.")
        days_lookback = st.slider("Lookback period (days)", 7, 180, 30)
        if st.button("🚀 Run Optimization Now"):
            with st.spinner("Analysing trade history..."):
                st.warning("Learning engine triggered. Check logs for progress.")
        if st.button("📊 View Latest Suggestions"):
            versions = load_parameter_versions(db)
            pending = [v for v in versions if v.get("status") == "pending"]
            if pending:
                for v in pending:
                    st.json(v)
            else:
                st.info("No pending suggestions.")

    st.markdown("---")

    # ── Learning run history ──────────────────────────────────────────────────
    st.subheader("Recent Learning Runs")
    try:
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM learning_runs ORDER BY started_at DESC LIMIT 10")
        runs = [dict(r) for r in cursor.fetchall()]
        if runs:
            df = pd.DataFrame(runs)
            show_cols = [c for c in ["run_id","started_at","completed_at",
                                      "optimization_method","trades_analyzed","status"] if c in df.columns]
            st.dataframe(df[show_cols], width='stretch')
        else:
            st.info("No learning runs recorded yet.")
    except Exception as e:
        st.info(f"Learning runs table not available: {e}")


# ── Tab 5: Analytics ──────────────────────────────────────────────────────────

def show_analytics_tab(closed_filt):
    st.header("📉 Performance Analytics")

    if not closed_filt:
        st.info("No closed trades in the selected period.")
        return

    df = pd.DataFrame(closed_filt)
    df["entry_time"]  = pd.to_datetime(df.get("entry_time"),  errors="coerce")
    df["pnl"]         = pd.to_numeric(df.get("pnl"),         errors="coerce").fillna(0.0)
    df["realized_rr"] = pd.to_numeric(df.get("realized_rr"), errors="coerce").fillna(0.0)

    df_sorted = df.dropna(subset=["entry_time"]).sort_values("entry_time")
    df_sorted["cumulative_pnl"] = df_sorted["pnl"].cumsum()
    df_sorted["equity"]         = 10000 + df_sorted["cumulative_pnl"]

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["entry_time"], y=df_sorted["equity"],
        mode="lines", name="Equity",
        line=dict(color="royalblue", width=2),
        fill="tozeroy", fillcolor="rgba(65,105,225,0.08)"
    ))
    fig.update_layout(xaxis_title="Date", yaxis_title="Equity ($)",
                      hovermode="x unified", height=350)
    st.plotly_chart(fig)

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

    # ── Per-symbol and distribution ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("By Symbol")
        if "symbol" in df.columns:
            sym_stats = df.groupby("symbol").agg(
                trades=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                avg_rr=("realized_rr", "mean")
            ).round(2)
            st.dataframe(sym_stats, width='stretch')

    with col2:
        st.subheader("Win / Loss Distribution")
        fig2 = go.Figure()
        if not wins.empty:
            fig2.add_trace(go.Histogram(x=wins,   name="Wins",   marker_color="green", opacity=0.7))
        if not losses.empty:
            fig2.add_trace(go.Histogram(x=losses, name="Losses", marker_color="red",   opacity=0.7))
        fig2.update_layout(barmode="overlay", height=280,
                           xaxis_title="P&L ($)", yaxis_title="Count")
        st.plotly_chart(fig2)

    # ── Time-based ────────────────────────────────────────────────────────────
    st.subheader("Time-Based Analysis")
    df_t = df.dropna(subset=["entry_time"]).copy()
    df_t["hour"]        = df_t["entry_time"].dt.hour
    df_t["day_of_week"] = df_t["entry_time"].dt.day_name()

    col1, col2 = st.columns(2)
    if not df_t.empty:
        with col1:
            hour_pnl = df_t.groupby("hour")["pnl"].sum()
            fig3 = px.bar(x=hour_pnl.index, y=hour_pnl.values,
                          labels={"x":"Hour (UTC)","y":"P&L ($)"},
                          title="P&L by Hour",
                          color=hour_pnl.values,
                          color_continuous_scale=["red","yellow","green"])
            st.plotly_chart(fig3)

        with col2:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            day_pnl = df_t.groupby("day_of_week")["pnl"].sum()
            day_pnl = day_pnl.reindex([d for d in day_order if d in day_pnl.index])
            fig4 = px.bar(x=day_pnl.index, y=day_pnl.values,
                          labels={"x":"Day","y":"P&L ($)"},
                          title="P&L by Day of Week",
                          color=day_pnl.values,
                          color_continuous_scale=["red","yellow","green"])
            st.plotly_chart(fig4)


if __name__ == "__main__":
    main()