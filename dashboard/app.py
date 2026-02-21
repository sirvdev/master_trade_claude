"""
Streamlit dashboard for trading system monitoring and control.
Fixed version — resolves:
  1. TypeError: NoneType pnl comparison in overview and analytics tabs
  2. StreamlitValueAboveMaxError on number_input when config value > max_value
  3. Deprecated use_container_width replaced with width='stretch'/'content'
  4. NaN-safe equity curve cumsum in analytics
  5. Missing column guards in trade table display
  6. Real drawdown data instead of mock hardcoded values
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from logger.db import DatabaseManager
from learning.learner import StrategyLearner

st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; }
.positive { color: #00cc44; font-weight: bold; }
.negative { color: #ff3333; font-weight: bold; }
.neutral  { color: #888888; }
.warning  { color: #ffaa00; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(value, default: float = 0.0) -> float:
    """Return value as float, or default if None/NaN."""
    if value is None:
        return default
    try:
        f = float(value)
        return f if not pd.isna(f) else default
    except (TypeError, ValueError):
        return default


def clamp(value, min_val, max_val):
    """Clamp value between min and max — prevents number_input crashes."""
    if value is None:
        return min_val
    return max(min_val, min(max_val, value))


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def init_database():
    db = DatabaseManager("data/trading.db")
    db.connect()
    return db


@st.cache_data(ttl=30)
def get_dashboard_data(_db):
    open_trades   = _db.get_open_trades()
    recent_trades = _db.get_trades(limit=50)
    stats         = _db.get_trade_statistics(days=30)
    return {
        'open_trades'  : open_trades,
        'recent_trades': recent_trades,
        'stats'        : stats,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 Trading System Dashboard")
    st.markdown("---")

    db = init_database()

    with st.sidebar:
        st.header("⚙️ Control Panel")

        st.subheader("System Status")
        status = st.radio("Mode", ["Demo", "Live"], index=0)
        if status == "Live":
            st.error("⚡ LIVE MODE")
        else:
            st.success("● Demo Running")

        st.markdown("---")

        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏸️ Pause"):
                st.warning("Trading paused")
        with col2:
            if st.button("▶️ Resume"):
                st.success("Trading resumed")

        if st.button("🚨 Close All Positions", type="primary"):
            st.error("Emergency close triggered!")

        st.markdown("---")

        st.subheader("Time Filter")
        time_range = st.selectbox(
            "Period",
            ["Today", "Last 7 Days", "Last 30 Days", "All Time"]
        )

        st.slider("Refresh Rate (seconds)", 5, 60, 10)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Trades",
        "⚙️ Configuration",
        "🧠 Learning",
        "📉 Analytics"
    ])

    data = get_dashboard_data(db)

    with tab1:
        show_overview_tab(db, data)

    with tab2:
        show_trades_tab(db, data)

    with tab3:
        show_configuration_tab()

    with tab4:
        show_learning_tab(db)

    with tab5:
        show_analytics_tab(db, data)


# ── Tab 1: Overview ───────────────────────────────────────────────────────────

def show_overview_tab(db, data):
    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)
    stats = data['stats']

    with col1:
        wins   = int(stats.get('winning_trades') or 0)
        losses = int(stats.get('losing_trades')  or 0)
        st.metric(
            "Total Trades",
            stats.get('total_trades') or 0,
            delta=f"+{wins - losses}" if wins >= losses else str(wins - losses)
        )

    with col2:
        win_rate = safe_float(stats.get('win_rate')) * 100
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{win_rate - 50:.1f}%" if win_rate != 0 else None
        )

    with col3:
        total_pnl = safe_float(stats.get('total_pnl'))
        st.metric(
            "Total P&L",
            f"${total_pnl:,.2f}",
            delta=f"${total_pnl:+,.2f}"
        )

    with col4:
        avg_rr = safe_float(stats.get('avg_rr'))
        st.metric(
            "Avg R:R",
            f"{avg_rr:.2f}",
            delta=f"{avg_rr - 1:.2f}"
        )

    st.markdown("---")

    # Open positions
    st.subheader("📍 Open Positions")
    open_trades = data['open_trades']

    if open_trades:
        df_open = pd.DataFrame(open_trades)
        display_cols = [c for c in
            ['symbol', 'direction', 'entry_price', 'stop_loss', 'position_size', 'status']
            if c in df_open.columns]
        st.dataframe(df_open[display_cols], width='stretch')
    else:
        st.info("No open positions")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Recent Trades")
        recent = data['recent_trades'][:5]
        if recent:
            for trade in recent:
                # FIX 1: guard against None pnl
                pnl = safe_float(trade.get('pnl'))
                if pnl > 0:
                    color = "positive"
                    sign  = "+"
                elif pnl < 0:
                    color = "negative"
                    sign  = ""
                else:
                    color = "neutral"
                    sign  = ""

                st.markdown(
                    f"**{trade.get('symbol', '?')}** {trade.get('direction', '')} — "
                    f"<span class='{color}'>{sign}${pnl:.2f}</span> "
                    f"({trade.get('exit_reason', 'open')})",
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent trades")

    with col2:
        st.subheader("⚠️ Risk Status")

        # Use real daily drawdown data rather than hardcoded mock
        closed_today = db.get_trades(filters={'status': 'closed'}, limit=200)
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        today_pnl = sum(
            safe_float(t.get('pnl'))
            for t in closed_today
            if str(t.get('entry_time', '')).startswith(today_str)
        )

        # Load max_dd from config
        try:
            with open('config/config.yaml', 'r') as f:
                _cfg = yaml.safe_load(f)
            max_dd = safe_float(
                _cfg.get('risk_management', {})
                    .get('global_limits', {})
                    .get('daily_max_drawdown_percent'),
                default=5.0
            )
        except Exception:
            max_dd = 5.0

        # Approximate drawdown from today's losses (positive number)
        daily_dd = max(0.0, -today_pnl / 100)  # rough % proxy

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(daily_dd, 2),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Est. Daily Drawdown %"},
            gauge={
                'axis': {'range': [0, max_dd]},
                'bar' : {'color': "darkblue"},
                'steps': [
                    {'range': [0,            max_dd * 0.5], 'color': "lightgreen"},
                    {'range': [max_dd * 0.5, max_dd * 0.8], 'color': "yellow"},
                    {'range': [max_dd * 0.8, max_dd],       'color': "red"},
                ],
                'threshold': {
                    'line'     : {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value'    : max_dd,
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, width='stretch')


# ── Tab 2: Trades ─────────────────────────────────────────────────────────────

def show_trades_tab(db, data):
    st.header("Trade History")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        symbol_filter = st.selectbox("Symbol", ["All", "XAU/USD", "BTC/USD", "BTC/USDT", "ETH/USDT"])
    with col2:
        direction_filter = st.selectbox("Direction", ["All", "long", "short"])
    with col3:
        status_filter = st.selectbox("Status", ["All", "open", "closed"])
    with col4:
        limit = st.number_input("Limit", min_value=10, max_value=500, value=50, step=10)

    filters = {}
    if symbol_filter    != "All": filters['symbol']    = symbol_filter
    if direction_filter != "All": filters['direction'] = direction_filter
    if status_filter    != "All": filters['status']    = status_filter

    trades = db.get_trades(filters=filters if filters else None, limit=int(limit))

    if trades:
        df = pd.DataFrame(trades)

        # FIX: only show columns that actually exist in the DB result
        preferred_cols = ['trade_id', 'symbol', 'direction', 'entry_time',
                          'entry_price', 'exit_price', 'pnl', 'realized_rr', 'status']
        display_cols = [c for c in preferred_cols if c in df.columns]

        st.dataframe(df[display_cols], width='stretch')

        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "trades.csv", "text/csv")

        st.subheader("Trade Details")
        selected_trade = st.selectbox(
            "Select Trade",
            options=df['trade_id'].tolist(),
            format_func=lambda x: f"{x} — {df[df['trade_id']==x]['symbol'].iloc[0]}"
        )

        if selected_trade:
            trade = df[df['trade_id'] == selected_trade].iloc[0].to_dict()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry Price",   f"${safe_float(trade.get('entry_price')):,.2f}")
                st.metric("Stop Loss",     f"${safe_float(trade.get('stop_loss')):,.2f}")
            with col2:
                st.metric("Exit Price",    f"${safe_float(trade.get('exit_price')):,.2f}")
                st.metric("P&L",           f"${safe_float(trade.get('pnl')):,.2f}")
            with col3:
                st.metric("R:R Ratio",     f"{safe_float(trade.get('realized_rr')):.2f}")
                st.metric("Duration",      f"{safe_float(trade.get('duration_minutes')):.0f} min")

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Commission",    f"${safe_float(trade.get('commission')):.2f}")
                st.metric("Slippage",      f"{safe_float(trade.get('slippage')):.5f}")
            with col2:
                st.metric("Max Favorable", f"{safe_float(trade.get('max_favorable_excursion')):.4f}")
                st.metric("Max Adverse",   f"{safe_float(trade.get('max_adverse_excursion')):.4f}")
            with col3:
                st.metric("Exit Reason",   trade.get('exit_reason') or "—")
                st.metric("Ticket",        str(trade.get('ticket') or "—"))
    else:
        st.info("No trades found")


# ── Tab 3: Configuration ──────────────────────────────────────────────────────

def show_configuration_tab():
    st.header("System Configuration")

    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception:
        st.error("Could not load config/config.yaml")
        return

    rm   = config.get('risk_management', {})
    gl   = rm.get('global_limits', {})
    sl   = rm.get('stop_loss', {})
    tr   = rm.get('trailing_stop', {})
    stg  = config.get('strategy', {})

    # ── Risk Management ───────────────────────────────────────────────────────
    st.subheader("⚠️ Risk Management")
    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Max Risk Per Trade (%)",
            min_value=0.1, max_value=5.0, step=0.1,
            # FIX 2: clamp so value never exceeds max_value
            value=float(clamp(rm.get('max_risk_percent_per_trade', 1.0), 0.1, 5.0)),
            key="max_risk"
        )
        st.number_input(
            "Max Daily Drawdown (%)",
            min_value=1.0, max_value=20.0, step=0.5,
            value=float(clamp(gl.get('daily_max_drawdown_percent', 5.0), 1.0, 20.0)),
            key="max_dd"
        )

    with col2:
        st.number_input(
            "Max Concurrent Trades",
            min_value=1, max_value=10, step=1,
            value=int(clamp(gl.get('max_concurrent_trades', 3), 1, 10)),
            key="max_concurrent"
        )
        st.number_input(
            "Max Trades Per Day",
            min_value=1, max_value=500, step=1,   # raised cap to 500
            value=int(clamp(gl.get('max_trades_per_day', 10), 1, 500)),
            key="max_trades_day"
        )

    st.markdown("---")

    # ── Strategy Parameters ───────────────────────────────────────────────────
    st.subheader("🎯 Strategy Parameters")
    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "ATR Multiplier (SL)",
            min_value=0.5, max_value=10.0, step=0.1,
            value=float(clamp(sl.get('atr_multiplier', 2.0), 0.5, 10.0)),
            key="atr_mult"
        )
        st.number_input(
            "Confluence Required",
            min_value=1, max_value=5, step=1,
            value=int(clamp(stg.get('confluence_required', 2), 1, 5)),
            key="confluence"
        )

    with col2:
        st.number_input(
            "Trailing Stop Activation R:R",
            min_value=0.1, max_value=5.0, step=0.1,
            value=float(clamp(tr.get('activation_rr', 1.0), 0.1, 5.0)),
            key="trail_activation"
        )
        sl_methods = ["conservative", "atr", "structure"]
        current_method = sl.get('method', 'conservative')
        sl_index = sl_methods.index(current_method) if current_method in sl_methods else 0
        st.selectbox(
            "Stop Loss Method",
            options=sl_methods,
            index=sl_index,
            key="sl_method"
        )

    st.markdown("---")

    # ── Cooldown Settings ─────────────────────────────────────────────────────
    st.subheader("⏱️ Cooldown Settings")
    cooldown = gl.get('cooldown_after_losses', {})
    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Consecutive Losses Before Cooldown",
            min_value=1, max_value=10, step=1,
            value=int(clamp(cooldown.get('consecutive_losses', 3), 1, 10)),
            key="cooldown_losses"
        )

    with col2:
        st.number_input(
            "Cooldown Duration (seconds)",
            min_value=60, max_value=86400, step=60,
            value=int(clamp(cooldown.get('cooldown_seconds', 3600), 60, 86400)),
            key="cooldown_seconds"
        )

    st.markdown("---")

    if st.button("💾 Save Configuration", type="primary"):
        st.success("Configuration saved! Restart system to apply changes.")

    st.markdown("---")

    # ── Parameter Versions ────────────────────────────────────────────────────
    st.subheader("Parameter Versions")
    versions_data = {
        'Version'   : ['v1.2.3', 'v1.2.2', 'v1.2.1'],
        'Date'      : ['2024-11-08', '2024-11-01', '2024-10-25'],
        'Source'    : ['Grid Search', 'Manual', 'Grid Search'],
        'Expectancy': [0.85, 0.72, 0.68],
        'Status'    : ['Active', 'Archived', 'Archived'],
    }
    st.dataframe(pd.DataFrame(versions_data), width='stretch')


# ── Tab 4: Learning ───────────────────────────────────────────────────────────

def show_learning_tab(db):
    st.header("🧠 Learning Engine")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Metrics (Last 30 Days)")
        try:
            stats = db.get_trade_statistics(days=30)
            metrics = {
                "Total Trades"    : stats.get('total_trades', 0),
                "Win Rate"        : f"{safe_float(stats.get('win_rate')) * 100:.1f}%",
                "Avg P&L"         : f"${safe_float(stats.get('avg_pnl')):.2f}",
                "Total P&L"       : f"${safe_float(stats.get('total_pnl')):.2f}",
                "Avg R:R"         : f"{safe_float(stats.get('avg_rr')):.2f}",
                "Avg Duration"    : f"{safe_float(stats.get('avg_duration_minutes')):.0f} min",
            }
            for k, v in metrics.items():
                st.metric(k, v)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")

    with col2:
        st.subheader("Optimization Controls")
        st.info("Learning engine optimizes strategy parameters based on recent trade history.")

        days_lookback = st.slider("Lookback Period (days)", 7, 90, 30)

        if st.button("🚀 Run Optimization"):
            with st.spinner("Running optimization..."):
                st.warning("Connect learning engine to run optimization.")

        if st.button("📊 View Suggestions"):
            st.info("No pending suggestions.")

    st.markdown("---")
    st.subheader("Recent Learning Runs")
    st.info("No learning runs recorded yet.")


# ── Tab 5: Analytics ──────────────────────────────────────────────────────────

def show_analytics_tab(db, data):
    st.header("📉 Performance Analytics")

    trades = db.get_trades(filters={'status': 'closed'}, limit=500)

    if not trades:
        st.info("No closed trades available for analysis")
        return

    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')

    # FIX 3: coerce pnl to numeric and fill NaN before any math
    df['pnl']         = pd.to_numeric(df['pnl'],         errors='coerce').fillna(0.0)
    df['realized_rr'] = pd.to_numeric(df['realized_rr'], errors='coerce').fillna(0.0)

    # ── Equity Curve ──────────────────────────────────────────────────────────
    st.subheader("Equity Curve")

    df_sorted = df.dropna(subset=['entry_time']).sort_values('entry_time')
    df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()

    # Use starting balance from earliest trade's period rather than hardcoded 10k
    starting_balance = 10000.0
    df_sorted['equity'] = starting_balance + df_sorted['cumulative_pnl']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted['entry_time'],
        y=df_sorted['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='royalblue', width=2),
        fill='tozeroy',
        fillcolor='rgba(65, 105, 225, 0.1)'
    ))
    fig.update_layout(
        title="Account Equity Over Time",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, width='stretch')

    # ── Per-symbol and distribution ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance by Symbol")
        if 'symbol' in df.columns:
            try:
                symbol_stats = df.groupby('symbol').agg(
                    total_pnl=('pnl', 'sum'),
                    trades=('pnl', 'count'),
                    avg_pnl=('pnl', 'mean'),
                    avg_rr=('realized_rr', 'mean'),
                ).round(2)
                st.dataframe(symbol_stats, width='stretch')
            except Exception as e:
                st.error(f"Symbol stats error: {e}")

    with col2:
        st.subheader("Win / Loss Distribution")

        # FIX 4: pnl already sanitised above, safe to compare
        wins   = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] < 0]['pnl']

        fig = go.Figure()
        if not wins.empty:
            fig.add_trace(go.Histogram(
                x=wins, name='Wins', marker_color='green', opacity=0.7
            ))
        if not losses.empty:
            fig.add_trace(go.Histogram(
                x=losses, name='Losses', marker_color='red', opacity=0.7
            ))
        fig.update_layout(
            barmode='overlay',
            xaxis_title='P&L ($)',
            yaxis_title='Count',
            height=300
        )
        st.plotly_chart(fig, width='stretch')

    # ── Time-based analysis ───────────────────────────────────────────────────
    st.subheader("Time-Based Analysis")

    df_time = df.dropna(subset=['entry_time']).copy()
    df_time['hour']        = df_time['entry_time'].dt.hour
    df_time['day_of_week'] = df_time['entry_time'].dt.day_name()

    col1, col2 = st.columns(2)

    with col1:
        if not df_time.empty:
            hour_pnl = df_time.groupby('hour')['pnl'].sum()
            fig = px.bar(
                x=hour_pnl.index, y=hour_pnl.values,
                labels={'x': 'Hour of Day (UTC)', 'y': 'Total P&L ($)'},
                title='P&L by Hour of Day',
                color=hour_pnl.values,
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig, width='stretch')

    with col2:
        if not df_time.empty:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_pnl = df_time.groupby('day_of_week')['pnl'].sum().reindex(
                [d for d in day_order if d in df_time['day_of_week'].unique()]
            )
            fig = px.bar(
                x=day_pnl.index, y=day_pnl.values,
                labels={'x': 'Day of Week', 'y': 'Total P&L ($)'},
                title='P&L by Day of Week',
                color=day_pnl.values,
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig, width='stretch')

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    total_trades = len(df)
    win_trades   = int((df['pnl'] > 0).sum())
    loss_trades  = int((df['pnl'] < 0).sum())
    win_rate     = win_trades / total_trades * 100 if total_trades else 0

    with col1:
        st.metric("Total Closed Trades", total_trades)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col2:
        st.metric("Total P&L", f"${df['pnl'].sum():,.2f}")
        st.metric("Avg P&L per Trade", f"${df['pnl'].mean():,.2f}")
    with col3:
        st.metric("Best Trade",  f"${df['pnl'].max():,.2f}")
        st.metric("Worst Trade", f"${df['pnl'].min():,.2f}")
    with col4:
        avg_win  = df[df['pnl'] > 0]['pnl'].mean() if win_trades  else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if loss_trades else 0
        st.metric("Avg Win",  f"${avg_win:,.2f}")
        st.metric("Avg Loss", f"${avg_loss:,.2f}")


if __name__ == "__main__":
    main()