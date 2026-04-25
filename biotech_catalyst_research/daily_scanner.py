#!/usr/bin/env python3
"""
Daily scanner: finds today's crash-reversion trade candidates,
computes position sizes, and sends alerts to Telegram.

Run daily after market close (e.g., 4:30 PM ET via cron).

Usage:
    # First run — set env vars:
    export TELEGRAM_BOT_TOKEN="your-bot-token"
    export TELEGRAM_CHAT_ID="your-chat-id"

    # Then:
    python daily_scanner.py              # scan + send to telegram
    python daily_scanner.py --dry-run    # scan only, print to console
    python daily_scanner.py --backfill 5 # scan last 5 days
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================================
# STRATEGY PARAMETERS (from backtest optimisation)
# =====================================================================
PARAMS = {
    "crash_threshold": 0.08,       # minimum single-day drop
    "volume_multiple": 2.0,        # vs 20-day avg volume
    "min_price": 0.50,             # stock price floor (avoid sub-penny)
    "max_price": 10.0,             # stock price ceiling
    "prior_drawdown": 0.20,        # must be down >20% over 20 days
    "stop_loss": 0.15,             # exit if position drops 15%
    "hold_days": 20,               # trading days to hold
    "base_position_pct": 0.10,     # 10% of capital per trade
    "max_concurrent": 20,          # max open positions
    "regime_crash_threshold": 30,  # weekly crash count for regime scaling
    "regime_scale_factor": 0.50,   # halve size in stressed regime
    "cost_bps": 50,                # assumed round-trip cost
}

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "portfolio_state.json"
TICKER_FILE = DATA_DIR / "scanner_tickers.csv"

# =====================================================================
# TICKER UNIVERSE
# =====================================================================

def load_ticker_universe() -> list[str]:
    """Load the ticker universe. Uses saved file or default list."""
    if TICKER_FILE.exists():
        df = pd.read_csv(TICKER_FILE)
        return df["ticker"].dropna().unique().tolist()

    # Default: the 508 tickers from our backtest + any additions
    default = _get_default_tickers()
    pd.DataFrame({"ticker": default}).to_csv(TICKER_FILE, index=False)
    logger.info("Created ticker file with %d tickers at %s", len(default), TICKER_FILE)
    return default


def _get_default_tickers() -> list[str]:
    """Hard-coded starter universe. Edit scanner_tickers.csv to customise."""
    tickers = []

    # Pull from existing data if available
    events_file = DATA_DIR / "full_universe_events.csv"
    if events_file.exists():
        df = pd.read_csv(events_file)
        tickers = df["ticker"].dropna().unique().tolist()

    if not tickers:
        # Fallback minimal set
        tickers = [
            "ACAD","ALKS","ALDX","AMPH","ANVS","AQST","ARDX","ARMP","ARQT",
            "ARWR","AUPH","AXSM","BCRX","BMRN","BTAI","CAPR","CGTX","CLSK",
            "CRNX","DBVT","GALT","GERN","GKOS","HALO","HRTX","INCY","INO",
            "IONS","JAZZ","KNSA","LXRX","MARA","MREO","NKTR","NVAX","OMER",
            "PHAT","PLX","PTCT","RIGL","RIOT","RYTM","SCYX","SGMO","SMMT",
            "SRNE","SUPN","TGTX","TNXP","TVTX","VNDA","VYNE","LCID","RIVN",
            "QS","CHPT","BTG","HL","CDE","KTOS","ASTS","UPST","IONQ","SOFI",
        ]
    return tickers


# =====================================================================
# PORTFOLIO STATE
# =====================================================================

def load_state() -> dict:
    """Load portfolio state (open positions, capital, history)."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": 100000,
        "positions": [],
        "trade_history": [],
        "last_scan_date": None,
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# =====================================================================
# MARKET DATA
# =====================================================================

def get_daily_data(ticker: str, days_back: int = 30) -> pd.DataFrame:
    """Fetch recent daily data for a single ticker."""
    try:
        end = datetime.now()
        start = end - timedelta(days=int(days_back * 1.8))
        data = yf.download(
            ticker, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        logger.debug("Failed %s: %s", ticker, e)
        return pd.DataFrame()


# =====================================================================
# SCANNER: Find today's candidates
# =====================================================================

def scan_for_candidates(
    tickers: list[str],
    scan_date: str = None,
) -> tuple[list[dict], int]:
    """
    Scan all tickers for crash events on the given date.

    Returns:
        candidates: list of trade candidate dicts
        weekly_crash_count: number of stocks crashing >8% this week (regime)
    """
    if scan_date:
        target = pd.Timestamp(scan_date)
    else:
        target = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))

    candidates = []
    weekly_crashes = 0
    week_start = target - timedelta(days=target.weekday())

    logger.info("Scanning %d tickers for date %s ...", len(tickers), target.date())

    for ticker in tickers:
        data = get_daily_data(ticker, days_back=30)
        if data.empty or len(data) < 22:
            continue

        data["return"] = data["Close"].pct_change()
        data["vol_20d"] = data["Volume"].rolling(20).mean()
        data["vol_ratio"] = data["Volume"] / data["vol_20d"]

        # Count weekly crashes for regime signal
        week_data = data[(data.index >= week_start) & (data.index <= target)]
        if not week_data.empty:
            if (week_data["return"].min() < -PARAMS["crash_threshold"]):
                weekly_crashes += 1

        # Check if target date has a qualifying crash
        if target not in data.index:
            # Find nearest trading day
            close_dates = data.index[data.index <= target]
            if close_dates.empty:
                continue
            nearest = close_dates[-1]
            if (target - nearest).days > 3:
                continue
            target_idx = nearest
        else:
            target_idx = target

        idx_pos = data.index.get_loc(target_idx)
        row = data.iloc[idx_pos]

        daily_return = row["return"]
        vol_ratio = row["vol_ratio"]
        close = row["Close"]

        # Filter 1: crash threshold
        if daily_return >= -PARAMS["crash_threshold"]:
            continue

        # Filter 2: volume spike
        if pd.isna(vol_ratio) or vol_ratio < PARAMS["volume_multiple"]:
            continue

        # Filter 3: price range
        if close > PARAMS["max_price"] or close < PARAMS["min_price"]:
            continue

        # Filter 4: already in downtrend (price 20 trading days ago)
        if idx_pos < 20:
            continue
        price_20d_ago = data.iloc[idx_pos - 20]["Close"]
        drawdown_20d = close / price_20d_ago - 1
        if drawdown_20d > -PARAMS["prior_drawdown"]:
            continue

        # This is a candidate
        candidates.append({
            "ticker": ticker,
            "scan_date": str(target.date()),
            "close": round(float(close), 2),
            "daily_return_pct": round(float(daily_return) * 100, 1),
            "volume_ratio": round(float(vol_ratio), 1),
            "drawdown_20d_pct": round(float(drawdown_20d) * 100, 1),
            "price_20d_ago": round(float(price_20d_ago), 2),
            "stop_price": round(float(close) * (1 - PARAMS["stop_loss"]), 2),
        })

    logger.info("Found %d candidates, weekly crash count: %d",
                len(candidates), weekly_crashes)
    return candidates, weekly_crashes


# =====================================================================
# POSITION SIZING
# =====================================================================

def compute_position_sizes(
    candidates: list[dict],
    state: dict,
    weekly_crashes: int,
) -> list[dict]:
    """
    Compute position size for each candidate based on:
    - Current capital
    - Number of open positions
    - Regime stress level
    """
    capital = state["capital"]
    open_count = len(state["positions"])
    available_slots = PARAMS["max_concurrent"] - open_count

    if available_slots <= 0:
        logger.info("Max concurrent positions reached (%d). No new trades.",
                    PARAMS["max_concurrent"])
        return []

    # Regime scaling
    if weekly_crashes > PARAMS["regime_crash_threshold"]:
        scale = PARAMS["regime_scale_factor"]
        regime_status = f"STRESSED ({weekly_crashes} crashes this week, sizing at {scale*100:.0f}%)"
    else:
        scale = 1.0
        regime_status = f"NORMAL ({weekly_crashes} crashes this week)"

    base_size = capital * PARAMS["base_position_pct"] * scale

    # Check: already holding any of these tickers?
    held_tickers = {p["ticker"] for p in state["positions"]}

    orders = []
    for c in candidates:
        if c["ticker"] in held_tickers:
            logger.info("Skip %s — already holding", c["ticker"])
            continue

        if len(orders) >= available_slots:
            break

        shares = int(base_size / c["close"])
        if shares < 1:
            continue

        position_value = shares * c["close"]
        cost_est = position_value * PARAMS["cost_bps"] / 10000

        entry_date = pd.Timestamp(c["scan_date"]) + timedelta(days=1)
        exit_date = entry_date + timedelta(days=int(PARAMS["hold_days"] * 1.4))

        orders.append({
            **c,
            "action": "BUY",
            "shares": shares,
            "position_value": round(position_value, 2),
            "pct_of_capital": round(position_value / capital * 100, 1),
            "stop_price": round(c["close"] * (1 - PARAMS["stop_loss"]), 2),
            "entry_date": str(entry_date.date()),
            "target_exit_date": str(exit_date.date()),
            "regime_status": regime_status,
            "estimated_cost": round(cost_est, 2),
        })

    return orders


# =====================================================================
# CHECK EXISTING POSITIONS
# =====================================================================

def check_open_positions(state: dict) -> list[dict]:
    """Check stop-losses and expiry on open positions."""
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    alerts = []

    remaining = []
    for pos in state["positions"]:
        ticker = pos["ticker"]
        data = get_daily_data(ticker, days_back=5)
        if data.empty:
            remaining.append(pos)
            continue

        current_price = float(data["Close"].iloc[-1])
        entry_price = pos["entry_price"]
        pnl_pct = (current_price / entry_price - 1) * 100
        stop_price = pos["stop_price"]
        exit_date = pd.Timestamp(pos["target_exit_date"])

        if current_price <= stop_price:
            alerts.append({
                "action": "SELL (STOP HIT)",
                "ticker": ticker,
                "entry_price": entry_price,
                "current_price": round(current_price, 2),
                "stop_price": stop_price,
                "shares": pos["shares"],
                "pnl_pct": round(pnl_pct, 1),
                "held_days": (today - pd.Timestamp(pos["entry_date"])).days,
            })
            state["trade_history"].append({
                **pos,
                "exit_price": round(current_price, 2),
                "exit_date": str(today.date()),
                "exit_reason": "stop_loss",
                "pnl_pct": round(pnl_pct, 1),
            })
            state["capital"] += pos["shares"] * current_price
            continue

        if today >= exit_date:
            alerts.append({
                "action": "SELL (HOLD EXPIRED)",
                "ticker": ticker,
                "entry_price": entry_price,
                "current_price": round(current_price, 2),
                "shares": pos["shares"],
                "pnl_pct": round(pnl_pct, 1),
                "held_days": (today - pd.Timestamp(pos["entry_date"])).days,
            })
            state["trade_history"].append({
                **pos,
                "exit_price": round(current_price, 2),
                "exit_date": str(today.date()),
                "exit_reason": "hold_expired",
                "pnl_pct": round(pnl_pct, 1),
            })
            state["capital"] += pos["shares"] * current_price
            continue

        # Still open — add status
        pos["current_price"] = round(current_price, 2)
        pos["pnl_pct"] = round(pnl_pct, 1)
        pos["days_held"] = (today - pd.Timestamp(pos["entry_date"])).days
        pos["days_remaining"] = (exit_date - today).days
        remaining.append(pos)

    state["positions"] = remaining
    return alerts


# =====================================================================
# TELEGRAM MESSAGING
# =====================================================================

async def send_telegram(message: str, token: str, chat_id: str):
    """Send message via Telegram bot."""
    from telegram import Bot
    bot = Bot(token=token)
    # Split long messages (Telegram limit: 4096 chars)
    for i in range(0, len(message), 4000):
        await bot.send_message(
            chat_id=chat_id,
            text=message[i:i+4000],
            parse_mode="Markdown",
        )


def format_message(
    orders: list[dict],
    exit_alerts: list[dict],
    state: dict,
    weekly_crashes: int,
) -> str:
    """Format the daily alert message."""
    today = datetime.now().strftime("%Y-%m-%d")
    capital = state["capital"]
    n_positions = len(state["positions"])

    lines = [
        f"*CRASH REVERSION SCANNER — {today}*",
        f"{'='*40}",
        f"Capital: ${capital:,.0f}",
        f"Open positions: {n_positions}/{PARAMS['max_concurrent']}",
        f"Weekly crash count: {weekly_crashes}"
              + (" ⚠️ STRESSED" if weekly_crashes > PARAMS["regime_crash_threshold"] else " ✅ NORMAL"),
        "",
    ]

    # Exit alerts (stops and expirations)
    if exit_alerts:
        lines.append(f"*EXITS ({len(exit_alerts)}):*")
        for a in exit_alerts:
            emoji = "🔴" if a["pnl_pct"] < 0 else "🟢"
            lines.append(
                f"{emoji} {a['action']}: *{a['ticker']}* "
                f"@ ${a['current_price']} ({a['pnl_pct']:+.1f}%) "
                f"| {a.get('held_days', '?')}d held"
            )
        lines.append("")

    # New orders
    if orders:
        lines.append(f"*NEW TRADES ({len(orders)}):*")
        for o in orders:
            lines.append(
                f"🎯 *BUY {o['ticker']}*"
            )
            lines.append(
                f"   Shares: {o['shares']} @ ${o['close']}"
                f" (${o['position_value']:,.0f} = {o['pct_of_capital']:.1f}% of capital)"
            )
            lines.append(
                f"   Crash: {o['daily_return_pct']:+.1f}% | Vol: {o['volume_ratio']:.1f}x"
                f" | 20d drawdown: {o['drawdown_20d_pct']:+.1f}%"
            )
            lines.append(
                f"   Stop: ${o['stop_price']} (-{PARAMS['stop_loss']*100:.0f}%)"
                f" | Exit by: {o['target_exit_date']}"
            )
            lines.append("")
    else:
        lines.append("*No new trades today.*")
        lines.append("")

    # Open positions summary
    if state["positions"]:
        lines.append(f"*OPEN POSITIONS ({n_positions}):*")
        for p in sorted(state["positions"], key=lambda x: x.get("pnl_pct", 0)):
            pnl = p.get("pnl_pct", 0)
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < -5 else "⚪"
            lines.append(
                f"{emoji} {p['ticker']:>5} | {p['shares']} sh @ ${p['entry_price']}"
                f" → ${p.get('current_price', '?')} ({pnl:+.1f}%)"
                f" | {p.get('days_remaining', '?')}d left"
                f" | stop ${p['stop_price']}"
            )

    # Parameters
    lines.extend([
        "",
        f"*PARAMETERS:*",
        f"  Crash thr: >{PARAMS['crash_threshold']*100:.0f}% | Vol: >{PARAMS['volume_multiple']:.0f}x"
            f" | Price: <${PARAMS['max_price']:.0f}",
        f"  20d drawdown: >{PARAMS['prior_drawdown']*100:.0f}%"
            f" | Stop: {PARAMS['stop_loss']*100:.0f}% | Hold: {PARAMS['hold_days']}d",
        f"  Position: {PARAMS['base_position_pct']*100:.0f}% of capital"
            f" | Max concurrent: {PARAMS['max_concurrent']}",
        f"  Regime threshold: {PARAMS['regime_crash_threshold']} crashes/week"
            f" → {PARAMS['regime_scale_factor']*100:.0f}% sizing",
    ])

    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Daily crash-reversion scanner")
    parser.add_argument("--dry-run", action="store_true", help="Print only, don't send to Telegram")
    parser.add_argument("--backfill", type=int, default=0, help="Scan last N days")
    parser.add_argument("--date", type=str, default=None, help="Scan specific date (YYYY-MM-DD)")
    parser.add_argument("--reset", action="store_true", help="Reset portfolio state")
    parser.add_argument("--capital", type=float, default=None, help="Set initial capital")
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not args.dry_run and (not token or not chat_id):
        logger.error("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars, or use --dry-run")
        sys.exit(1)

    # Load state
    if args.reset:
        state = {"capital": args.capital or 100000, "positions": [], "trade_history": [], "last_scan_date": None}
        save_state(state)
        logger.info("Portfolio state reset. Capital: $%s", state["capital"])
        return

    state = load_state()
    if args.capital:
        state["capital"] = args.capital

    tickers = load_ticker_universe()

    # Determine scan date(s)
    if args.date:
        scan_dates = [args.date]
    elif args.backfill > 0:
        scan_dates = [
            (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(args.backfill, 0, -1)
        ]
    else:
        scan_dates = [datetime.now().strftime("%Y-%m-%d")]

    for scan_date in scan_dates:
        logger.info("=== Scanning %s ===", scan_date)

        # Check existing positions for stops/expiry
        exit_alerts = check_open_positions(state)

        # Scan for new candidates
        candidates, weekly_crashes = scan_for_candidates(tickers, scan_date)

        # Compute position sizes
        orders = compute_position_sizes(candidates, state, weekly_crashes)

        # Add new positions to state
        for o in orders:
            state["positions"].append({
                "ticker": o["ticker"],
                "shares": o["shares"],
                "entry_price": o["close"],
                "stop_price": o["stop_price"],
                "entry_date": o["entry_date"],
                "target_exit_date": o["target_exit_date"],
                "position_value": o["position_value"],
            })
            state["capital"] -= o["position_value"]

        state["last_scan_date"] = scan_date

        # Format message
        msg = format_message(orders, exit_alerts, state, weekly_crashes)

        if args.dry_run:
            # Replace markdown for console
            print(msg.replace("*", ""))
        else:
            import asyncio
            asyncio.run(send_telegram(msg, token, chat_id))
            logger.info("Telegram message sent.")

    save_state(state)
    logger.info("State saved. Capital: $%.0f, Positions: %d",
                state["capital"], len(state["positions"]))


if __name__ == "__main__":
    main()
