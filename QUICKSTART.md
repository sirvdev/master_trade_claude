# Quick Start Guide

Get your trading system up and running in 5 minutes.

## 📋 Prerequisites

- Python 3.10 or higher
- pip package manager
- 500MB disk space
- (Optional) MT5 account for metals trading
- (Optional) Binance account for crypto trading

## 🚀 Installation (5 Steps)

### 1. Clone & Install

```bash
# Clone repository
git clone <your-repo-url>
cd trader_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your details
nano .env  # or use any text editor
```

**Minimum configuration for demo mode:**
```bash
ENVIRONMENT=demo
DATABASE_PATH=data/trading.db
```

### 3. Test Installation

```bash
# Run tests to verify installation
pytest tests/ -v

# Should see: All tests passing ✓
```

### 4. Initialize Database

```bash
# Create data directory
mkdir -p data logs

# The database will auto-initialize on first run
```

### 5. Launch Dashboard

```bash
# Start the dashboard
streamlit run dashboard/app.py

# Open browser to: http://localhost:8501
```

## 🎯 Your First Backtest

Run a quick backtest to verify everything works:

```bash
python backtest/backtester.py
```

Expected output:
```
=== Backtest Complete: BTC/USDT ===
Total Trades: 15
Win Rate: 60%
Final Balance: $10,450
Return: 4.50%
Max Drawdown: -2.3%
```

## 🔧 Configuration

### Enable Binance (Testnet)

1. Get testnet API keys: https://testnet.binance.vision
2. Update `.env`:
```bash
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
BINANCE_MODE=testnet
```

3. Enable crypto in `config/config.yaml`:
```yaml
symbols:
  BTC/USDT:
    enabled: true
    platform: binance
    mode: demo
```

### Enable MT5

1. Setup MT5 EA bridge (see docs/MT5_SETUP.md)
2. Update `.env`:
```bash
MT5_BRIDGE_HOST=localhost
MT5_BRIDGE_PORT=9090
MT5_ACCOUNT=your_account
```

3. Enable metals in `config/config.yaml`:
```yaml
symbols:
  XAU/USD:
    enabled: true
    platform: mt5
    mode: demo
```

## 📊 Running the System

### Demo Mode (Paper Trading)

```bash
# Ensure demo mode in .env
ENVIRONMENT=demo

# Start system
python main.py
```

The system will:
- ✓ Fetch market data
- ✓ Analyze multiple timeframes
- ✓ Generate entry signals
- ✓ Simulate trades
- ✓ Log all decisions
- ✓ Update dashboard

### Monitor via Dashboard

While main.py is running, launch dashboard in another terminal:

```bash
streamlit run dashboard/app.py
```

Dashboard shows:
- 📊 Open positions
- 📈 Performance metrics
- 📝 Trade history
- ⚙️ Configuration controls
- 🧠 Learning engine status

## 🎓 Understanding the Flow

### 1. Market Analysis
System analyzes configured symbols every minute:
- Fetches 1H, 15m, 5m data
- Calculates 15+ indicators
- Detects market structure (HH/HL/LH/LL)
- Checks for entry confluence

### 2. Entry Decision
When conditions align:
- ✓ Higher timeframe bias matches
- ✓ Multiple indicators confirm
- ✓ Price structure supports
- ✓ Risk limits not exceeded
→ Trade signal generated

### 3. Risk Calculation
Before placing order:
- Calculate position size (1% risk default)
- Set stop loss (ATR or structure-based)
- Set take profit levels (TP1, TP2)
- Configure trailing stop activation

### 4. Execution
Order placed via configured platform:
- MT5: Socket bridge to EA
- Binance: REST API call
- Demo: Simulated execution

### 5. Position Management
While trade is open:
- Monitor for stop loss hit
- Check take profit levels
- Move to breakeven at 1:1 R:R
- Activate trailing stop
- Partial profit-taking

### 6. Learning
After sufficient trades:
- Calculate performance metrics
- Run parameter optimization
- Test improvements via backtest
- Suggest parameter updates
- Require manual approval

## 📈 Key Metrics to Watch

Monitor these in the dashboard:

### Performance
- **Win Rate**: Target 50-60%
- **Avg R:R**: Target >1.5
- **Expectancy**: Target >$0.50 per trade
- **Sharpe Ratio**: Target >1.0

### Risk
- **Max Drawdown**: Keep <10%
- **Daily Drawdown**: Keep <5%
- **Concurrent Trades**: Default max 3
- **Daily Trades**: Default max 10

## 🔄 Daily Workflow

### Morning (Market Open)
1. Check dashboard for overnight activity
2. Review any stopped-out trades
3. Verify system is running
4. Check global risk limits

### During Trading
1. Monitor open positions
2. Watch for entry signals
3. Review trade quality in logs
4. Adjust settings if needed

### Evening (Market Close)
1. Review daily performance
2. Export logs if needed
3. Check learning suggestions
4. Plan adjustments

### Weekly
1. Run learning optimization
2. Review parameter versions
3. Backtest any changes
4. Update configuration

## 🎛️ Common Adjustments

### Too Many Trades
```yaml
strategy:
  confluence_required: 3  # Increase from 2
```

### Too Few Trades
```yaml
strategy:
  confluence_required: 2  # Decrease from 3
  
filters:
  respect_higher_tf_bias: false  # Allow counter-trend
```

### Losses Too Large
```yaml
risk_management:
  max_risk_percent_per_trade: 0.5  # Reduce from 1.0
  
  stop_loss:
    atr_multiplier: 1.5  # Tighter from 2.0
```

### Want Bigger Winners
```yaml
risk_management:
  trailing_stop:
    activation_rr: 1.5  # Later from 1.0
    atr_multiplier: 2.0  # Wider from 1.5
```

## 🐛 Troubleshooting

### System Won't Start
```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check logs
tail -f logs/trading_system.log
```

### No Trades Executing
1. Check `config.yaml` - symbols enabled?
2. Check `.env` - mode set to demo?
3. Check dashboard - any error messages?
4. Check logs - analysis running?

### Dashboard Won't Load
```bash
# Check if port is in use
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Use different port
streamlit run dashboard/app.py --server.port 8502
```

### Database Errors
```bash
# Backup current database
cp data/trading.db data/trading.db.backup

# Delete and reinitialize
rm data/trading.db
python main.py  # Will recreate
```

## 📚 Next Steps

### Learn More
- Read full [README.md](README.md)
- Study [Architecture](docs/ARCHITECTURE.md)
- Review [API Documentation](docs/API.md)

### Customize
- Modify indicators in `indicators/indicators.py`
- Add entry logic in `strategy/engine.py`
- Create custom risk rules in `risk_management/`

### Advanced
- Implement custom learning algorithms
- Add sentiment analysis
- Integrate order book data
- Build mobile notifications

## ⚠️ Before Going Live

**Critical Checklist:**

- [ ] Tested in demo mode for 30+ days
- [ ] Reviewed all closed trades
- [ ] Understood every configuration option
- [ ] Set appropriate risk limits
- [ ] Have stop-loss on stop-loss (manual oversight)
- [ ] Tested emergency shutdown
- [ ] API keys properly secured
- [ ] Monitoring system in place
- [ ] Understand platform fees
- [ ] Comfortable with maximum loss

**Start with smallest possible live position size.**

## 💡 Pro Tips

1. **Be Patient**: Let the system accumulate 50+ trades before judging
2. **Trust the Process**: Don't override every signal
3. **Keep Learning On**: Auto-optimization improves over time
4. **Review Logs**: Your best debugging tool
5. **Start Conservative**: 0.5% risk, increase slowly
6. **Monitor Constantly**: First few weeks especially
7. **Have a Kill Switch**: Know how to shut down fast
8. **Diversify**: Don't put all capital in one system

## 🆘 Support

- **Issues**: Open GitHub issue
- **Questions**: Check docs/ folder
- **Updates**: Watch repository releases

---

**Ready to trade? Start with `python main.py` and watch the magic happen! 🚀**