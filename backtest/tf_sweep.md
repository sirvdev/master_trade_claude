# Timeframe Sweep — Quick Start

## Step 1: Apply Fix 6 to backtester (4 lines changed)

Open `backtest/backtester.py` and make the 4 changes described in
`fix_06_backtester_standalone.py`. This makes the backtester walk on
primary_tf instead of entry_tf, matching live behavior.

## Step 2: Copy tf_sweep.py

Place `tf_sweep.py` in your `backtest/` directory:
```
backtest/tf_sweep.py
```

## Step 3: Run the sweep

Make sure MT5 is running with the EA loaded (backtester fetches data from MT5).

### Test one symbol at a time:
```bash
# Gold — tests 7 timeframe configs
python -m backtest.tf_sweep --symbol XAUUSD --start 2026-04-12 --end 2026-04-17 --engine original

# BTC — tests 6 timeframe configs
python -m backtest.tf_sweep --symbol BTCUSD --start 2026-04-12 --end 2026-04-17 --engine original

# EUR — tests 4 timeframe configs
python -m backtest.tf_sweep --symbol EURUSD --start 2026-04-12 --end 2026-04-17 --engine original

# Silver — tests 4 timeframe configs
python -m backtest.tf_sweep --symbol XAGUSD --start 2026-04-12 --end 2026-04-17 --engine original
```

### Test all enabled symbols at once:
```bash
python -m backtest.tf_sweep --all --start 2026-04-12 --end 2026-04-17 --engine original
```

### Test with a different engine:
```bash
python -m backtest.tf_sweep --symbol XAUUSD --start 2026-04-12 --end 2026-04-17 --engine ict
python -m backtest.tf_sweep --symbol BTCUSD --start 2026-04-12 --end 2026-04-17 --engine smc
```

## Step 4: Read the results

The script outputs a ranked comparison table for each symbol:

```
  RESULTS COMPARISON: XAUUSD
  Config                              Trades    WR        PnL   MaxDD     PF  Sharpe   Expect
  B: 4H/15m/5m (Week2 config)            17  29.0%  $-1632.16  -1.60%  0.84   -0.38  $-96.01
  D: 4H/1H/15m                           12  58.3%   $+420.50  -0.52%  1.45    0.22  $+35.04
  ...
  
  🏆 WINNER: D: 4H/1H/15m
```

And at the end, a recommended config.yaml snippet you can copy-paste.

## Timeframe Configs Being Tested

### Gold (XAUUSD) — 7 configs:
| Config | Structure | Primary | Entry | Threshold |
|--------|-----------|---------|-------|-----------|
| A | 15m | 5m | 1m | 7 (Week 1 original) |
| B | 4H | 15m | 5m | 6 (Week 2 config) |
| C | 1H | 15m | 5m | 6 |
| D | 4H | 1H | 15m | 6 |
| E | 4H | 15m | 5m | 7 (tighter threshold) |
| F | 1D | 4H | 15m | 5 (swing trade) |
| G | 4H | 30m | 5m | 6 |

### BTC (BTCUSD) — 6 configs:
| Config | Structure | Primary | Entry | Threshold |
|--------|-----------|---------|-------|-----------|
| A | 1H | 15m | 1m | 7 (Week 1 original) |
| B | 4H | 1H | 15m | 6 (Week 2 config) |
| C | 1H | 15m | 5m | 6 |
| D | 4H | 15m | 5m | 6 |
| E | 4H | 1H | 5m | 6 |
| F | 1D | 4H | 1H | 5 (swing trade) |

## Adding Your Own Configs

Open `tf_sweep.py` and add entries to the `TF_CONFIGS_*` lists:

```python
{
    'name': 'My custom config',
    'timeframes': ['4H', '30m', '5m'],
    'primary_timeframe': '30m',
    'entry_timeframe': '5m',
    'confluence_threshold': 6,
},
```

## Notes

- Each config test takes 1-3 minutes depending on the timeframe granularity
- Gold with 7 configs ≈ 10-20 minutes total
- All 4 symbols ≈ 45-60 minutes
- The sweep fetches data from MT5 for each config, so MT5 must stay running
- Results are printed to console — pipe to a file to save:
  `python -m backtest.tf_sweep --all --start 2026-04-12 --end 2026-04-17 --engine original > sweep_results.txt 2>&1`