#!/usr/bin/env python3
"""
backtest/tf_sweep.py — Timeframe Configuration Sweep
=====================================================
Tests multiple timeframe configurations per symbol against historical data
to find the most profitable setup before running live.

Usage:
    python -m backtest.tf_sweep --symbol XAUUSD --start 2026-04-12 --end 2026-04-17
    python -m backtest.tf_sweep --symbol BTCUSD --start 2026-04-12 --end 2026-04-17
    python -m backtest.tf_sweep --all --start 2026-04-12 --end 2026-04-17

Requirements:
    - MT5 must be running with the EA loaded (for historical data fetch)
    - Apply Fix 6 to backtest/backtester.py first (primary_tf walk-forward)

What it does:
    For each symbol, tests every timeframe combination in TF_CONFIGS.
    Each config changes: structure TF, primary TF, entry TF, and confluence threshold.
    Runs a full walk-forward backtest for each, then prints a ranked comparison.
"""

import asyncio
import copy
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TIMEFRAME CONFIGURATIONS TO TEST
# ═══════════════════════════════════════════════════════════════════════════
#
# Each config is: {
#   'name':         human-readable label,
#   'timeframes':   [structure_tf, primary_tf, entry_tf],
#   'primary_timeframe': which TF triggers analysis,
#   'entry_timeframe':   which TF is used for entry precision,
#   'confluence_threshold': score threshold for volatile symbols,
# }
#
# The backtester will use structure_tf for HTF bias, primary_tf for signals,
# and entry_tf for SL/limit price precision — exactly as live does.

TF_CONFIGS_GOLD = [
    {
        'name': 'A: 15m/5m/1m (Week1 original)',
        'timeframes': ['15m', '5m', '1m'],
        'primary_timeframe': '5m',
        'entry_timeframe': '1m',
        'confluence_threshold': 7,
    },
    {
        'name': 'B: 4H/15m/5m (Week2 config)',
        'timeframes': ['4H', '15m', '5m'],
        'primary_timeframe': '15m',
        'entry_timeframe': '5m',
        'confluence_threshold': 6,
    },
    {
        'name': 'C: 1H/15m/5m',
        'timeframes': ['1H', '15m', '5m'],
        'primary_timeframe': '15m',
        'entry_timeframe': '5m',
        'confluence_threshold': 6,
    },
    {
        'name': 'D: 4H/1H/15m',
        'timeframes': ['4H', '1H', '15m'],
        'primary_timeframe': '1H',
        'entry_timeframe': '15m',
        'confluence_threshold': 6,
    },
    {
        'name': 'E: 4H/15m/5m threshold=7',
        'timeframes': ['4H', '15m', '5m'],
        'primary_timeframe': '15m',
        'entry_timeframe': '5m',
        'confluence_threshold': 7,
    },
    {
        'name': 'F: 1D/4H/15m',
        'timeframes': ['1D', '4H', '15m'],
        'primary_timeframe': '4H',
        'entry_timeframe': '15m',
        'confluence_threshold': 5,
    },
    {
        'name': 'G: 4H/30m/5m',
        'timeframes': ['4H', '30m', '5m'],
        'primary_timeframe': '30m',
        'entry_timeframe': '5m',
        'confluence_threshold': 6,
    },
]

TF_CONFIGS_BTC = [
    {
        'name': 'A: 1H/15m/1m (Week1 original)',
        'timeframes': ['1H', '15m', '1m'],
        'primary_timeframe': '15m',
        'entry_timeframe': '1m',
        'confluence_threshold': 7,
    },
    {
        'name': 'B: 4H/1H/15m (Week2 config)',
        'timeframes': ['4H', '1H', '15m'],
        'primary_timeframe': '1H',
        'entry_timeframe': '15m',
        'confluence_threshold': 6,
    },
    {
        'name': 'C: 1H/15m/5m',
        'timeframes': ['1H', '15m', '5m'],
        'primary_timeframe': '15m',
        'entry_timeframe': '5m',
        'confluence_threshold': 6,
    },
    {
        'name': 'D: 4H/15m/5m',
        'timeframes': ['4H', '15m', '5m'],
        'primary_timeframe': '15m',
        'entry_timeframe': '5m',
        'confluence_threshold': 6,
    },
    {
        'name': 'E: 4H/1H/5m',
        'timeframes': ['4H', '1H', '5m'],
        'primary_timeframe': '1H',
        'entry_timeframe': '5m',
        'confluence_threshold': 6,
    },
    {
        'name': 'F: 1D/4H/1H',
        'timeframes': ['1D', '4H', '1H'],
        'primary_timeframe': '4H',
        'entry_timeframe': '1H',
        'confluence_threshold': 5,
    },
]

TF_CONFIGS_EURUSD = [
    {
        'name': 'A: 4H/1H/15m (current)',
        'timeframes': ['4H', '1H', '15m'],
        'primary_timeframe': '1H',
        'entry_timeframe': '15m',
        'confluence_threshold': 5,
    },
    {
        'name': 'B: 1D/4H/15m',
        'timeframes': ['1D', '4H', '15m'],
        'primary_timeframe': '4H',
        'entry_timeframe': '15m',
        'confluence_threshold': 5,
    },
    {
        'name': 'C: 1D/4H/1H',
        'timeframes': ['1D', '4H', '1H'],
        'primary_timeframe': '4H',
        'entry_timeframe': '1H',
        'confluence_threshold': 5,
    },
    {
        'name': 'D: 4H/1H/15m threshold=6',
        'timeframes': ['4H', '1H', '15m'],
        'primary_timeframe': '1H',
        'entry_timeframe': '15m',
        'confluence_threshold': 6,
    },
]

TF_CONFIGS_XAGUSD = [
    {
        'name': 'A: 1D/4H/15m (current)',
        'timeframes': ['1D', '4H', '15m'],
        'primary_timeframe': '4H',
        'entry_timeframe': '15m',
        'confluence_threshold': 5,
    },
    {
        'name': 'B: 4H/1H/15m',
        'timeframes': ['4H', '1H', '15m'],
        'primary_timeframe': '1H',
        'entry_timeframe': '15m',
        'confluence_threshold': 5,
    },
    {
        'name': 'C: 4H/15m/5m',
        'timeframes': ['4H', '15m', '5m'],
        'primary_timeframe': '15m',
        'entry_timeframe': '5m',
        'confluence_threshold': 6,
    },
    {
        'name': 'D: 1D/4H/1H',
        'timeframes': ['1D', '4H', '1H'],
        'primary_timeframe': '4H',
        'entry_timeframe': '1H',
        'confluence_threshold': 5,
    },
]

SYMBOL_CONFIGS = {
    'XAUUSD': TF_CONFIGS_GOLD,
    'XAU/USD': TF_CONFIGS_GOLD,
    'BTCUSD': TF_CONFIGS_BTC,
    'BTC/USD': TF_CONFIGS_BTC,
    'EURUSD': TF_CONFIGS_EURUSD,
    'EUR/USD': TF_CONFIGS_EURUSD,
    'XAGUSD': TF_CONFIGS_XAGUSD,
    'XAG/USD': TF_CONFIGS_XAGUSD,
}


# ═══════════════════════════════════════════════════════════════════════════
# CORE SWEEP LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def build_config_for_test(base_config: Dict, symbol: str, tf_cfg: Dict) -> Dict:
    """
    Create a modified config dict with the given TF settings for one symbol.
    Deep-copies the base config so each test is independent.
    """
    config = copy.deepcopy(base_config)

    # Find the symbol key in config (handles XAU/USD vs XAUUSD)
    sym_key = None
    for k in config.get('symbols', {}):
        if k.replace('/', '') == symbol.replace('/', ''):
            sym_key = k
            break

    if not sym_key:
        # Symbol not in config — add it
        sym_key = symbol
        config.setdefault('symbols', {})[sym_key] = {
            'enabled': True,
            'platform': 'mt5',
            'mode': 'live',
        }

    # Apply TF settings
    config['symbols'][sym_key]['timeframes'] = tf_cfg['timeframes']
    config['symbols'][sym_key]['primary_timeframe'] = tf_cfg['primary_timeframe']
    config['symbols'][sym_key]['entry_timeframe'] = tf_cfg['entry_timeframe']
    config['symbols'][sym_key]['confluence_threshold'] = tf_cfg['confluence_threshold']
    config['symbols'][sym_key]['enabled'] = True

    # Update strategy threshold
    config.setdefault('strategy', {})['confluence_required'] = tf_cfg['confluence_threshold']

    # Set backtest timeframes to match
    config.setdefault('backtest', {}).setdefault('timeframes', {})
    config['backtest']['timeframes']['entry_timeframe'] = tf_cfg['entry_timeframe']
    config['backtest']['timeframes']['minimum_timeframe'] = '1m'

    # If Fix 6 applied, set primary_timeframe for walk-forward
    config['backtest']['timeframes']['primary_timeframe'] = tf_cfg['primary_timeframe']

    return config


async def run_single_test(
    config: Dict,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float,
    engine_class,
) -> Dict:
    """Run one backtest with the given config. Returns results dict."""
    from backtest.backtester import Backtester
    from risk_management.money_manager import MoneyManager
    from risk_management.stop_manager import StopManager

    backtester = Backtester(config)
    strategy_engine = engine_class(config)
    money_manager = MoneyManager(config)
    stop_manager = StopManager(config)

    try:
        results = await backtester.run_from_mt5(
            strategy_engine=strategy_engine,
            money_manager=money_manager,
            stop_manager=stop_manager,
            symbol=symbol.replace('/', ''),
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
        )
        return results
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {
            'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
            'max_drawdown': 0, 'profit_factor': 0, 'return_percent': 0,
            'sharpe_ratio': 0, 'expectancy': 0,
            'error': str(e),
        }


async def sweep_symbol(
    base_config: Dict,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float,
    engine_name: str,
) -> List[Tuple[str, Dict]]:
    """Run all TF configs for one symbol, return list of (config_name, results)."""

    # Import the right engine
    if engine_name == 'original':
        from strategy.engine import StrategyEngine as EngineClass
    elif engine_name == 'ict':
        from strategy.ict_engine import ICTStrategyEngine as EngineClass
    elif engine_name == 'smc':
        from strategy.smc_engine import SMCStrategyEngine as EngineClass
    else:
        from strategy.engine import StrategyEngine as EngineClass

    norm_symbol = symbol.replace('/', '')
    tf_configs = SYMBOL_CONFIGS.get(symbol) or SYMBOL_CONFIGS.get(norm_symbol)
    if not tf_configs:
        print(f"\n  No TF configs defined for {symbol}. Skipping.")
        return []

    print(f"\n{'#' * 70}")
    print(f"  SWEEP: {symbol} ({engine_name} engine)")
    print(f"  Period: {start_date.date()} → {end_date.date()}")
    print(f"  Testing {len(tf_configs)} configurations...")
    print(f"{'#' * 70}")

    all_results = []

    for i, tf_cfg in enumerate(tf_configs, 1):
        name = tf_cfg['name']
        print(f"\n  ── Config {i}/{len(tf_configs)}: {name} ──")

        config = build_config_for_test(base_config, symbol, tf_cfg)
        results = await run_single_test(
            config, symbol, start_date, end_date, initial_balance, EngineClass
        )
        results['config_name'] = name
        results['tf_config'] = tf_cfg
        all_results.append((name, results))

    return all_results


def print_comparison(symbol: str, all_results: List[Tuple[str, Dict]]):
    """Print a ranked comparison table of all configs for a symbol."""

    if not all_results:
        return

    print(f"\n{'=' * 90}")
    print(f"  RESULTS COMPARISON: {symbol}")
    print(f"{'=' * 90}")

    # Header
    print(f"  {'Config':<35s} {'Trades':>6s} {'WR':>6s} {'PnL':>10s} "
          f"{'MaxDD':>7s} {'PF':>6s} {'Sharpe':>7s} {'Expect':>8s}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*8}")

    # Sort by PnL descending
    sorted_results = sorted(all_results, key=lambda x: x[1].get('total_pnl', 0), reverse=True)

    for name, r in sorted_results:
        trades = r.get('total_trades', 0)
        wr = r.get('win_rate', 0) * 100
        pnl = r.get('total_pnl', 0)
        dd = r.get('max_drawdown', 0)
        pf = r.get('profit_factor', 0)
        sharpe = r.get('sharpe_ratio', 0)
        exp = r.get('expectancy', 0)
        err = r.get('error', '')

        if err:
            print(f"  {name:<35s} {'ERROR':>6s}  {err[:50]}")
        else:
            pnl_color = '+' if pnl >= 0 else ''
            print(f"  {name:<35s} {trades:>6d} {wr:>5.1f}% "
                  f"${pnl_color}{pnl:>8.2f} {dd:>6.2f}% "
                  f"{pf:>5.2f} {sharpe:>6.2f} ${exp:>7.2f}")

    # Winner
    winner_name, winner = sorted_results[0]
    print(f"\n  🏆 WINNER: {winner_name}")
    print(f"     Timeframes: {winner.get('tf_config', {}).get('timeframes', '?')}")
    print(f"     Primary TF: {winner.get('tf_config', {}).get('primary_timeframe', '?')}")
    print(f"     Entry TF:   {winner.get('tf_config', {}).get('entry_timeframe', '?')}")
    print(f"     Threshold:  {winner.get('tf_config', {}).get('confluence_threshold', '?')}")
    print(f"{'=' * 90}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description='Sweep timeframe configurations to find the best setup per symbol'
    )
    parser.add_argument('--symbol', default=None,
                        help='Symbol to test (e.g. XAUUSD, BTCUSD). Use --all for all symbols.')
    parser.add_argument('--all', action='store_true',
                        help='Test all configured symbols')
    parser.add_argument('--start', required=True,
                        help='Start date YYYY-MM-DD')
    parser.add_argument('--end', required=True,
                        help='End date YYYY-MM-DD')
    parser.add_argument('--balance', type=float, default=100000,
                        help='Initial balance (default: 100000)')
    parser.add_argument('--engine', default='original',
                        choices=['original', 'ict', 'smc'],
                        help='Which strategy engine to test (default: original)')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'config' / 'config.yaml'),
                        help='Path to base config file')
    args = parser.parse_args()

    # Load base config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    # Determine which symbols to test
    if args.all:
        symbols = [
            sym for sym, cfg in base_config.get('symbols', {}).items()
            if cfg.get('enabled') and cfg.get('platform') == 'mt5'
        ]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        print("ERROR: Specify --symbol XAUUSD or --all")
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"  TIMEFRAME SWEEP")
    print(f"  Engine:  {args.engine}")
    print(f"  Period:  {start_date.date()} → {end_date.date()}")
    print(f"  Balance: ${args.balance:,.0f}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"{'#' * 70}")

    # Run sweeps
    all_winners = {}

    for symbol in symbols:
        results = await sweep_symbol(
            base_config=base_config,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=args.balance,
            engine_name=args.engine,
        )
        print_comparison(symbol, results)

        if results:
            # Store winner
            best_name, best_result = max(results, key=lambda x: x[1].get('total_pnl', 0))
            all_winners[symbol] = {
                'name': best_name,
                'config': best_result.get('tf_config', {}),
                'pnl': best_result.get('total_pnl', 0),
                'win_rate': best_result.get('win_rate', 0),
                'max_drawdown': best_result.get('max_drawdown', 0),
            }

    # Final summary if multiple symbols tested
    if len(all_winners) > 1:
        print(f"\n\n{'#' * 70}")
        print(f"  FINAL SUMMARY — Optimal Config Per Symbol")
        print(f"{'#' * 70}")
        for sym, w in all_winners.items():
            print(f"\n  {sym}:")
            print(f"    Winner: {w['name']}")
            print(f"    Timeframes: {w['config'].get('timeframes', '?')}")
            print(f"    Primary TF: {w['config'].get('primary_timeframe', '?')}")
            print(f"    Entry TF:   {w['config'].get('entry_timeframe', '?')}")
            print(f"    Threshold:  {w['config'].get('confluence_threshold', '?')}")
            print(f"    PnL: ${w['pnl']:+,.2f} | WR: {w['win_rate']:.0%} | DD: {w['max_drawdown']:.2f}%")

        # Generate config snippet
        print(f"\n\n  ── Recommended config.yaml snippet ──\n")
        print("  symbols:")
        for sym, w in all_winners.items():
            cfg = w['config']
            print(f"    {sym}:")
            print(f"      enabled: true")
            print(f"      platform: mt5")
            print(f"      mode: live")
            print(f"      timeframes:")
            for tf in cfg.get('timeframes', []):
                print(f"      - {tf}")
            print(f"      primary_timeframe: {cfg.get('primary_timeframe')}")
            print(f"      entry_timeframe: {cfg.get('entry_timeframe')}")
            print(f"      confluence_threshold: {cfg.get('confluence_threshold')}")

    print("\n  Done.\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())