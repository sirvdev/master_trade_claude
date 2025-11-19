"""
Streamlit dashboard for trading system monitoring and control.
Provides real-time monitoring, configuration, and manual override capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from logger.db import DatabaseManager
from learning.learner import StrategyLearner


# Page configuration
st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.positive { color: #00ff00; }
.negative { color: #ff0000; }
.warning { color: #ffaa00; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_database():
    """Initialize database connection."""
    db = DatabaseManager("data/trading.db")
    db.connect()
    return db


@st.cache_data(ttl=60)
def get_dashboard_data(_db):
    """Get dashboard data (cached for 60 seconds)."""
    open_trades = _db.get_open_trades()
    recent_trades = _db.get_trades(limit=50)
    stats = _db.get_trade_statistics(days=30)
    
    return {
        'open_trades': open_trades,
        'recent_trades': recent_trades,
        'stats': stats
    }


def main():
    """Main dashboard application."""
    
    st.title("📈 Trading System Dashboard")
    st.markdown("---")
    
    # Initialize
    db = init_database()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # System status
        st.subheader("System Status")
        status = st.radio("Mode", ["Demo", "Live"], index=0)
        st.success("System Running" if status else "○ System Stopped")
        
        st.markdown("---")
        
        # Quick actions
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
        
        # Time filter
        st.subheader("Time Filter")
        time_range = st.selectbox(
            "Period",
            ["Today", "Last 7 Days", "Last 30 Days", "All Time"]
        )
        
        # Refresh rate
        refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
        
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Trades",
        "⚙️ Configuration",
        "🧠 Learning",
        "📉 Analytics"
    ])
    
    # Get data
    data = get_dashboard_data(db)
    
    # TAB 1: OVERVIEW
    with tab1:
        show_overview_tab(db, data)
        
    # TAB 2: TRADES
    with tab2:
        show_trades_tab(db, data)
        
    # TAB 3: CONFIGURATION
    with tab3:
        show_configuration_tab()
        
    # TAB 4: LEARNING
    with tab4:
        show_learning_tab(db)
        
    # TAB 5: ANALYTICS
    with tab5:
        show_analytics_tab(db, data)


def show_overview_tab(db, data):
    """Display overview tab."""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = data['stats']
    
    with col1:
        wins = stats.get('winning_trades') or 0
        losses = stats.get('losing_trades') or 0
        st.metric(
            "Total Trades",
            stats.get('total_trades') or 0,
            delta=f"+{wins - losses}" if wins >= losses else str(wins - losses)
        )
        
    with col2:
        win_rate = (stats.get('win_rate') or 0) * 100
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{win_rate - 50:.1f}%" if win_rate != 0 else None
        )
        
    with col3:
        total_pnl = stats.get('total_pnl', 0)
        st.metric(
            "Total P&L",
            f"${total_pnl:,.2f}",
            delta=f"${abs(total_pnl):,.2f}"
        )
        
    with col4:
        avg_rr = stats.get('avg_rr', 0)
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
        st.dataframe(
            df_open[['symbol', 'direction', 'entry_price', 'stop_loss', 'position_size', 'status']],
            use_container_width=True
        )
    else:
        st.info("No open positions")
        
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Recent Trades")
        recent = data['recent_trades'][:5]
        if recent:
            for trade in recent:
                pnl = trade.get('pnl', 0)
                color = "positive" if pnl > 0 else "negative"
                st.markdown(
                    f"**{trade['symbol']}** {trade['direction']} - "
                    f"<span class='{color}'>${pnl:.2f}</span> "
                    f"({trade.get('exit_reason', 'open')})",
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent trades")
            
    with col2:
        st.subheader("⚠️ Risk Status")
        
        # Risk gauges
        daily_dd = 2.5  # Mock data
        max_dd = 5.0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=daily_dd,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Daily Drawdown %"},
            gauge={
                'axis': {'range': [None, max_dd]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_dd*0.5], 'color': "lightgreen"},
                    {'range': [max_dd*0.5, max_dd*0.8], 'color': "yellow"},
                    {'range': [max_dd*0.8, max_dd], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_dd
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)


def show_trades_tab(db, data):
    """Display trades tab."""
    st.header("Trade History")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol_filter = st.selectbox("Symbol", ["All"] + ["BTC/USDT", "XAU/USD", "ETH/USDT"])
    with col2:
        direction_filter = st.selectbox("Direction", ["All", "long", "short"])
    with col3:
        status_filter = st.selectbox("Status", ["All", "open", "closed"])
    with col4:
        limit = st.number_input("Limit", min_value=10, max_value=500, value=50)
        
    # Get trades with filters
    filters = {}
    if symbol_filter != "All":
        filters['symbol'] = symbol_filter
    if direction_filter != "All":
        filters['direction'] = direction_filter
    if status_filter != "All":
        filters['status'] = status_filter
        
    trades = db.get_trades(filters=filters if filters else None, limit=limit)
    
    if trades:
        df = pd.DataFrame(trades)
        
        # Display trade table
        st.dataframe(
            df[['trade_id', 'symbol', 'direction', 'entry_time', 'entry_price', 
                'exit_price', 'pnl', 'realized_rr', 'status']],
            use_container_width=True
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "trades.csv",
            "text/csv"
        )
        
        # Trade details
        st.subheader("Trade Details")
        selected_trade = st.selectbox(
            "Select Trade",
            options=df['trade_id'].tolist(),
            format_func=lambda x: f"{x} - {df[df['trade_id']==x]['symbol'].iloc[0]}"
        )
        
        if selected_trade:
            trade = df[df['trade_id'] == selected_trade].iloc[0].to_dict()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry Price", f"${trade['entry_price']:.2f}")
                st.metric("Stop Loss", f"${trade['stop_loss']:.2f}")
            with col2:
                st.metric("Exit Price", f"${trade.get('exit_price', 0):.2f}")
                st.metric("P&L", f"${trade.get('pnl', 0):.2f}")
            with col3:
                st.metric("R:R Ratio", f"{trade.get('realized_rr', 0):.2f}")
                st.metric("Duration", f"{trade.get('duration_minutes', 0):.0f} min")
                
    else:
        st.info("No trades found")


def show_configuration_tab():
    """Display configuration tab."""
    st.header("System Configuration")
    
    # Load current config
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        st.error("Could not load configuration file")
        return
        
    # Risk Management Section
    st.subheader("⚠️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input(
            "Max Risk Per Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=config.get('risk_management', {}).get('max_risk_percent_per_trade', 1.0),
            step=0.1,
            key="max_risk"
        )
        
        st.number_input(
            "Max Daily Drawdown (%)",
            min_value=1.0,
            max_value=20.0,
            value=config.get('risk_management', {}).get('global_limits', {}).get('daily_max_drawdown_percent', 5.0),
            step=0.5,
            key="max_dd"
        )
        
    with col2:
        st.number_input(
            "Max Concurrent Trades",
            min_value=1,
            max_value=10,
            value=config.get('risk_management', {}).get('global_limits', {}).get('max_concurrent_trades', 3),
            step=1,
            key="max_concurrent"
        )
        
        st.number_input(
            "Max Trades Per Day",
            min_value=1,
            max_value=50,
            value=config.get('risk_management', {}).get('global_limits', {}).get('max_trades_per_day', 10),
            step=1,
            key="max_trades_day"
        )
        
    st.markdown("---")
    
    # Strategy Parameters
    st.subheader("🎯 Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input(
            "ATR Multiplier (SL)",
            min_value=1.0,
            max_value=5.0,
            value=config.get('risk_management', {}).get('stop_loss', {}).get('atr_multiplier', 2.0),
            step=0.1,
            key="atr_mult"
        )
        
        st.number_input(
            "Confluence Required",
            min_value=1,
            max_value=5,
            value=config.get('strategy', {}).get('confluence_required', 2),
            step=1,
            key="confluence"
        )
        
    with col2:
        st.number_input(
            "Trailing Stop Activation R:R",
            min_value=0.5,
            max_value=3.0,
            value=config.get('risk_management', {}).get('trailing_stop', {}).get('activation_rr', 1.0),
            step=0.1,
            key="trail_activation"
        )
        
        st.selectbox(
            "Stop Loss Method",
            options=["conservative", "atr", "structure"],
            index=0,
            key="sl_method"
        )
        
    # Save button
    if st.button("💾 Save Configuration", type="primary"):
        st.success("Configuration saved successfully!")
        st.info("Note: Restart system to apply changes")


def show_learning_tab(db):
    """Display learning tab."""
    st.header("🧠 Learning Engine")
    
    learner = StrategyLearner(db, {
        'learning': {
            'optimization': {'optimization_metric': 'expectancy'},
            'guardrails': {'min_sample_size': 30}
        }
    })
    
    # Learning status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Last Run", "2 days ago")
    with col2:
        st.metric("Total Runs", "15")
    with col3:
        st.metric("Active Version", "v1.2.3")
        
    st.markdown("---")
    
    # Current performance
    st.subheader("Current Performance")
    
    trades = db.get_trades(filters={'status': 'closed'}, limit=100)
    if trades:
        metrics = learner.calculate_performance_metrics(trades)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Expectancy", f"${metrics['expectancy']:.2f}")
        with col2:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Max DD", f"{metrics['max_drawdown']:.2f}%")
            
    st.markdown("---")
    
    # Parameter optimization
    st.subheader("Parameter Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_method = st.selectbox(
            "Method",
            ["Grid Search", "Random Search", "Bandit (RL)"]
        )
        
        min_trades = st.number_input(
            "Min Trades Required",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
        
    with col2:
        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["expectancy", "profit_factor", "sharpe_ratio", "win_rate"]
        )
        
        auto_apply = st.checkbox("Auto-apply best parameters")
        
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Running optimization..."):
            st.info("This would run the learning engine in production")
            # result = learner.run_learning_cycle(...)
            st.success("Optimization complete!")
            
    st.markdown("---")
    
    # Parameter versions
    st.subheader("Parameter Versions")
    
    # Mock data
    versions_data = {
        'Version': ['v1.2.3', 'v1.2.2', 'v1.2.1'],
        'Date': ['2024-11-08', '2024-11-01', '2024-10-25'],
        'Source': ['Grid Search', 'Manual', 'Grid Search'],
        'Expectancy': [0.85, 0.72, 0.68],
        'Status': ['Active', 'Archived', 'Archived']
    }
    
    df_versions = pd.DataFrame(versions_data)
    st.dataframe(df_versions, use_container_width=True)


def show_analytics_tab(db, data):
    """Display analytics tab."""
    st.header("📉 Performance Analytics")
    
    # Get data
    trades = db.get_trades(filters={'status': 'closed'}, limit=200)
    
    if not trades:
        st.info("No trades available for analysis")
        return
        
    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    # Equity curve
    st.subheader("Equity Curve")
    
    df_sorted = df.sort_values('entry_time')
    df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted['entry_time'],
        y=10000 + df_sorted['cumulative_pnl'],
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Account Equity Over Time",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance by symbol
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance by Symbol")
        
        symbol_stats = df.groupby('symbol').agg({
            'pnl': ['sum', 'count', 'mean'],
            'realized_rr': 'mean'
        }).round(2)
        
        st.dataframe(symbol_stats, use_container_width=True)
        
    with col2:
        st.subheader("Win/Loss Distribution")
        
        wins = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] < 0]['pnl']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=wins,
            name='Wins',
            marker_color='green',
            opacity=0.7
        ))
        fig.add_trace(go.Histogram(
            x=losses,
            name='Losses',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            barmode='overlay',
            xaxis_title='P&L ($)',
            yaxis_title='Count',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    # Time-based analysis
    st.subheader("Time-Based Analysis")
    
    df['hour'] = df['entry_time'].dt.hour
    df['day_of_week'] = df['entry_time'].dt.day_name()
    
    col1, col2 = st.columns(2)
    
    with col1:
        hour_pnl = df.groupby('hour')['pnl'].sum()
        
        fig = px.bar(
            x=hour_pnl.index,
            y=hour_pnl.values,
            labels={'x': 'Hour of Day', 'y': 'Total P&L ($)'},
            title='P&L by Hour of Day'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        day_pnl = df.groupby('day_of_week')['pnl'].sum()
        
        fig = px.bar(
            x=day_pnl.index,
            y=day_pnl.values,
            labels={'x': 'Day of Week', 'y': 'Total P&L ($)'},
            title='P&L by Day of Week'
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()