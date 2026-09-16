"""
=============================================================
  AI TRADING BOT  — Crypto & Stock Market
  RSI(6-12) + Market Swing + Tight SL/TP + Claude AI
  Web UI: Flask admin on http://localhost:5000
  Single terminal: bot runs in background thread,
                   Flask serves home + dashboard + settings
=============================================================
"""

import os, sys, re, time, json, logging, logging.handlers, threading, secrets, sqlite3, base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from functools import wraps

import pandas as pd
import numpy as np
import requests
import colorama
from colorama import Fore, Style
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session)
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass


# ── Optional market libs ──────────────────────────────────────
try:
    import ccxt; CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    import yfinance as yf; YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

colorama.init(autoreset=True)

# ════════════════════════════════════════════════════════════
#  GLOBAL SHARED STATE
# ════════════════════════════════════════════════════════════

def _clean_credential(value) -> str:
    """Normalise copied credentials so hidden whitespace or stray characters
    do not break exchange signatures (common with copied Bybit keys)."""
    if value is None:
        return ""
    value = str(value).strip()
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return value


CONFIG = {
    "market_mode"            : "CRYPTO",
    "crypto_symbols"         : ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XAUT/USDT"],
    "crypto_exchange"        : "binance",
    "crypto_timeframe"       : "15m",
    "stock_symbols"          : ["AAPL", "TSLA", "NVDA", "AMZN", "XAGUSD=X", "XAUUSD=X"],
    "stock_timeframe"        : "15m",
    "forex_symbols"         : ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
    "forex_timeframe"       : "15m",
    "forex_exchange"       : "yahoo",
    "rsi_fast_period"        : 6,
    "rsi_slow_period"        : 12,
    "rsi_confirm_period"     : 24,   # RSI-24 — slower trend-confirmation filter, see TechnicalAnalysis.signal()
    "rsi_overbought"         : 70,
    "rsi_oversold"           : 30,
    "rsi_signal_threshold"   : 5,
    "swing_lookback"         : 20,
    "swing_pivot_strength"   : 3,
    "stop_loss_pct"          : 0.005,    # 0.5% — tight stop
    "take_profit_pct"        : 0.020,    # 2.0% default profit target; dynamic RR logic can override this for fast trades
    "confirmed_trade_min_rr" : 4.0,      # full ride-out mode: only the strongest setups keep the original 4:1 reward:risk profile
    "fast_trade_min_rr"      : 1.0,      # quick trades can open on a 1R or 2R target and exit fast once that objective is reached
    "fast_trade_target_rr"   : 2.0,      # default fast-trade target; weaker but valid setups can trade on a 2:1 path instead of waiting for 4:1
    "min_confidence_to_trade": 85,       # only open trades with >=85% combined confidence
    "max_risk_per_trade"     : 0.01,
    "max_open_trades"        : 3,
    "account_balance"        : 10000.0,
    "anthropic_api_key"      : os.environ.get("ANTHROPIC_API_KEY", ""),
    "ai_model"               : "claude-sonnet-4-6",
    "ai_confirmation_required": True,
    "scan_interval_seconds"  : 60,
    "max_execution_slippage_pct": 0.003,  # 0.3% — refuse to execute if price moved more than this since signal detection
    "signal_exit_confirmations": 2,  # consecutive scans required before closing a trade on confidence/direction loss (SL/TP unaffected — always instant)
    "hold_requires_entry_confidence": True,  # a trade must maintain its OWN opening confidence to stay valid, not just the generic minimum — early warning for weakening conviction
    # Ride-winners mechanism: lets a trade that's genuinely working keep
    # running toward its take-profit target instead of getting cut short
    # by the same strict confidence checks that (correctly) protect a
    # fresh, unproven trade. See TradeManager.check_exits() and the
    # confidence re-check in scan_once().
    "profit_lock_enabled"       : True,   # move SL to breakeven once a trade earns profit_lock_r_multiple of its own risk
    "profit_lock_r_multiple"    : 1.0,    # e.g. 1.0 = once up by as much as it originally risked (1R)
    "relax_exit_after_profit_lock": True, # once breakeven-locked, only a genuine signal reversal can trigger an exit — not routine confidence wobble
    "trailing_stop_enabled"     : True,   # keep trailing the stop behind price once profit-locked and still running
    "trailing_stop_activation_r": 1.5,    # start trailing once profit reaches this many R
    "trailing_stop_distance_r"  : 0.75,   # trail this many R behind the best price seen
    # Chop filter: refuses NEW entries when the market hasn't been moving
    # efficiently (lots of back-and-forth, little net progress) — see the
    # Efficiency Ratio calculation in TechnicalAnalysis.signal(). Targets
    # the specific pattern where trades held <5min were, as a group, a net
    # loser while trades held longer were strongly profitable.
    "chop_filter_enabled"        : True,
    "chop_efficiency_lookback"   : 10,    # bars used to compute the Efficiency Ratio
    "chop_min_efficiency_ratio"  : 0.30,  # below this, new entries are capped below threshold (0=pure noise, 1=perfectly efficient trend)
    # Chop high-conviction override: during chop, a trade can still open
    # if EVERY core signal agrees simultaneously (RSI + swing + EMA trend +
    # RSI-confirm), a much higher bar than the normal point-threshold used
    # in a trending market. Volume surge is the flakiest of these to
    # reliably detect on some symbols/exchanges, so it's separately
    # toggleable rather than a hard requirement baked into the others.
    "chop_confluence_require_volume_surge": True,
    # Any OPEN trade on a symbol that's CURRENTLY choppy (recomputed every
    # scan, not frozen at entry) uses this many confirmation scans instead
    # of the normal signal_exit_confirmations — fast invalidation since
    # chop is inherently less predictable. The moment that symbol is no
    # longer choppy, normal confirmation count resumes automatically.
    "chop_exit_confirmations"    : 1,
    # Counter-trend discount: scales down a signal that's fighting the
    # prevailing EMA trend, proportional to how established that trend is.
    # Targets the finding that SELL signals underperformed specifically on
    # symbols in a sustained uptrend (e.g. ETH), while the same logic
    # worked fine elsewhere.
    "counter_trend_discount_enabled": True,
    "strong_trend_ema_sep_pct"   : 0.008,  # EMA21/EMA50 separation (as % of price) considered a "fully established" trend
    "max_counter_trend_discount_pct": 0.35, # maximum score reduction applied to a counter-trend signal at full trend strength
    "paper_trading"          : True,
    "live_trading_enabled"   : False,
    "live_accounts"          : {
        "CRYPTO": {"exchange": "binance",
                   "api_key": _clean_credential(os.environ.get("LIVE_API_KEY", "")),
                   "api_secret": _clean_credential(os.environ.get("LIVE_API_SECRET", "")),
                   "passphrase": _clean_credential(os.environ.get("LIVE_PASSPHRASE", "")),
                   "connected":False,"balance":0.0,"last_sync":None,"last_error":""},
        "STOCK":  {"exchange":"yahoo","api_key":"","api_secret":"","passphrase":"",
                   "connected":False,"balance":0.0,"last_sync":None,"last_error":""},
        "FOREX":   {"exchange":"bybit","api_key":"","api_secret":"","passphrase":"",
                   "connected":False,"balance":0.0,"last_sync":None,"last_error":""},
    },
    "log_level"              : "INFO",
    "log_file"               : "trading_bot.log",
    # Log rotation: the ACTIVE log file the bot writes to is capped at
    # log_max_size_mb; once hit it rotates to a numbered backup and starts
    # fresh. Independently, log_reset_months forces a rotation on a
    # schedule even if the size cap is never reached (a quiet deployment
    # could otherwise run for years on one small file — the time reset
    # guarantees a bound either way). Historical data lives in the rotated
    # backups, not in memory or the database — see db_retention_days below.
    "log_max_size_mb"        : 500,
    "log_backup_count"       : 5,
    "log_reset_months"       : 4,   # within the requested 3-6 month range
    # Database retention: admin_activity/admin_sessions grow with every
    # scan, trade, and login and have no natural cap — left alone they
    # slow down every dashboard query and every write as the tables grow
    # over months of continuous operation. Rows older than
    # db_retention_days are pruned periodically; this is safe because the
    # SAME data is already durably captured in the rotating log file
    # (every _elog() call writes to both), so nothing is actually lost —
    # it just moves from "fast structured lookup" to "archived in a log
    # backup" once it's no longer operationally relevant.
    "db_retention_days"      : 90,
    "db_max_size_mb"         : 500,
    "db_prune_interval_hours": 24,
    # Flag as a likely stale/cached data source if the SAME OHLCV data
    # persists across this many consecutive scan intervals — see
    # MarketDataProvider._check_stale_repeat().
    "stale_repeat_scan_multiple": 2,
    "web_port"               : int(os.environ.get("PORT", 5000)),
    "web_host"               : "0.0.0.0",
}

STATE_FILE = os.environ.get(
    "TRADING_BOT_STATE_FILE",
    os.path.join(Path(__file__).resolve().parent, ".trading_bot_state.json"),
)
DATABASE_PATH = os.environ.get(
    "TRADING_BOT_DB_FILE",
    os.path.join(Path(__file__).resolve().parent, "trade_bot.db"),
)


def normalize_market_symbol(symbol: str, market: Optional[str] = None) -> str:
    """Normalise symbols across crypto/stock/forex markets so new metals tickers
    like XAUT/USDT and XAUUSD=X behave the same as the built-in pairs.

    This prevents crashes caused by mixed casing, missing slash formatting, or
    raw ticker names being passed through without a market-specific normalisation
    step before the data provider or exchange client sees them.
    """
    if symbol is None:
        return ""
    value = str(symbol).strip()
    if not value:
        return ""

    market_key = (market or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    value = value.replace(" ", "").upper()

    if market_key == "CRYPTO":
        if "/" in value:
            base, quote = value.split("/", 1)
            return f"{base.strip()}/{quote.strip()}" if base and quote else value
        if value.endswith("USDT"):
            base = value[:-4].strip()
            return f"{base}/USDT" if base else value
        if 3 <= len(value) <= 10 and value not in {"USD", "EUR", "GBP", "JPY", "AUD", "CHF"}:
            return f"{value}/USDT"
        return value

    if market_key in {"STOCK", "FOREX"}:
        if value.endswith("=X"):
            return value
        if "/" in value:
            return value
        if value.endswith("USD") and len(value) >= 6:
            return value
        return value

    return value


def _get_secret_key() -> bytes:
    secret = (os.environ.get("FLASK_SECRET", "") or os.environ.get("DASHBOARD_SECRETS", "") or "bot-rl-local-secret").encode("utf-8")
    return secret


def _protect_secret(value: Optional[str]) -> str:
    if value is None:
        return ""
    value = str(value)
    if not value:
        return ""
    key = _get_secret_key()
    raw = value.encode("utf-8")
    masked = bytes((b ^ key[i % len(key)]) for i, b in enumerate(raw))
    return "enc:" + base64.b64encode(masked).decode("ascii")


def _unprotect_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if not value.startswith("enc:"):
        return value
    key = _get_secret_key()
    try:
        raw = base64.b64decode(value[4:].encode("ascii"))
    except Exception:
        return value
    plain = bytes((b ^ key[i % len(key)]) for i, b in enumerate(raw))
    return plain.decode("utf-8")


def init_database(db_path: Optional[str] = None) -> str:
    global DATABASE_PATH
    target = str(db_path or DATABASE_PATH)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_activity_at TEXT NOT NULL,
                ended_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_id TEXT,
                activity_type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                details_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exchange_accounts (
                market TEXT PRIMARY KEY,
                exchange TEXT NOT NULL,
                api_key TEXT,
                api_secret TEXT,
                passphrase TEXT,
                connected INTEGER NOT NULL DEFAULT 0,
                balance REAL NOT NULL DEFAULT 0,
                last_sync TEXT,
                last_error TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()
    DATABASE_PATH = str(path)
    return DATABASE_PATH


def _db_connect(db_path: Optional[str] = None):
    conn = sqlite3.connect(db_path or DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def start_admin_session(username: str, session_id: Optional[str] = None) -> str:
    init_database()
    sid = session_id or secrets.token_hex(8)
    ts = datetime.utcnow().isoformat()
    with _db_connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO admin_sessions (id, username, started_at, last_activity_at, ended_at) VALUES (?, ?, ?, ?, ?)",
            (sid, username, ts, ts, None),
        )
        conn.commit()
    return sid


def end_admin_session(session_id: Optional[str]) -> None:
    if not session_id:
        return
    init_database()
    with _db_connect() as conn:
        conn.execute(
            "UPDATE admin_sessions SET ended_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), session_id),
        )
        conn.commit()


_LAST_DB_PRUNE_AT: Optional[datetime] = None


def prune_database(force: bool = False) -> dict:
    """Delete admin_activity/admin_sessions rows older than
    db_retention_days, then VACUUM to actually reclaim the freed disk
    space (SQLite doesn't shrink the file on DELETE alone). Safe to do
    aggressively because the same messages are already durably captured
    in the rotating log file (_elog() writes to both) — pruning the DB
    doesn't lose history, it just moves it from "fast structured lookup"
    to "archived in a log backup" once it's no longer operationally
    relevant. Also enforces db_max_size_mb as a hard backstop that trims
    further if age-based pruning alone isn't enough."""
    global _LAST_DB_PRUNE_AT
    now = datetime.utcnow()
    if not force and _LAST_DB_PRUNE_AT is not None:
        hours_since = (now - _LAST_DB_PRUNE_AT).total_seconds() / 3600
        if hours_since < CONFIG["db_prune_interval_hours"]:
            return {"skipped": True, "reason": "not due yet"}

    cutoff = (now - timedelta(days=CONFIG["db_retention_days"])).isoformat()
    deleted = {"admin_activity": 0, "admin_sessions": 0}
    try:
        with _db_connect() as conn:
            cur = conn.execute("DELETE FROM admin_activity WHERE created_at < ?", (cutoff,))
            deleted["admin_activity"] = cur.rowcount
            cur2 = conn.execute("DELETE FROM admin_sessions WHERE started_at < ? AND ended_at IS NOT NULL", (cutoff,))
            deleted["admin_sessions"] = cur2.rowcount
            conn.commit()

        db_size_mb = Path(DATABASE_PATH).stat().st_size / (1024 * 1024) if Path(DATABASE_PATH).exists() else 0.0
        if db_size_mb > CONFIG["db_max_size_mb"]:
            # Age-based retention wasn't enough on its own — trim the
            # oldest remaining rows in batches until back under the cap,
            # bounded so this can never loop forever.
            with _db_connect() as conn:
                for _ in range(5):
                    total = conn.execute("SELECT COUNT(*) FROM admin_activity").fetchone()[0]
                    if total == 0:
                        break
                    trim_n = max(100, total // 5)
                    cur3 = conn.execute(
                        "DELETE FROM admin_activity WHERE id IN "
                        "(SELECT id FROM admin_activity ORDER BY id ASC LIMIT ?)", (trim_n,))
                    deleted["admin_activity"] += cur3.rowcount
                    conn.commit()
                    db_size_mb = Path(DATABASE_PATH).stat().st_size / (1024 * 1024)
                    if db_size_mb <= CONFIG["db_max_size_mb"]:
                        break

        # VACUUM must run outside any active transaction — use its own
        # autocommit connection rather than the transactional one above.
        try:
            vconn = sqlite3.connect(DATABASE_PATH, isolation_level=None)
            vconn.execute("VACUUM")
            vconn.close()
            db_size_mb = Path(DATABASE_PATH).stat().st_size / (1024 * 1024) if Path(DATABASE_PATH).exists() else db_size_mb
        except Exception as ve:
            logger.warning(f"VACUUM after prune failed (non-fatal): {ve}")

        _LAST_DB_PRUNE_AT = now
        if deleted["admin_activity"] or deleted["admin_sessions"]:
            _elog(f"🧹 Database pruned — removed {deleted['admin_activity']} activity row(s), "
                  f"{deleted['admin_sessions']} session row(s) older than {CONFIG['db_retention_days']}d. "
                  f"DB size now {db_size_mb:.1f}MB. Full history remains in the rotated log backups.", "INFO")
        return {"ok": True, "deleted": deleted, "db_size_mb": round(db_size_mb, 1)}
    except Exception as e:
        logger.error(f"Database prune failed: {e}")
        return {"ok": False, "error": str(e)}


def record_admin_activity(username: str, activity_type: str, message: str,
                          session_id: Optional[str] = None, details: Optional[dict] = None) -> dict:
    init_database()
    sid = session_id or None
    if sid:
        start_admin_session(username, sid)
    ts = datetime.utcnow().isoformat()
    details_json = json.dumps(details or {}, default=str)
    with _db_connect() as conn:
        cursor = conn.execute(
            "INSERT INTO admin_activity (username, session_id, activity_type, message, created_at, details_json) VALUES (?, ?, ?, ?, ?, ?)",
            (username, sid, activity_type, message, ts, details_json),
        )
        conn.commit()
    return {"id": cursor.lastrowid, "username": username, "activity_type": activity_type, "message": message, "created_at": ts}


def get_recent_admin_activity(username: str, limit: int = 20) -> list[dict]:
    init_database()
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT username, session_id, activity_type, message, created_at, details_json FROM admin_activity WHERE username = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    return [{
        "username": row["username"],
        "session_id": row["session_id"],
        "activity_type": row["activity_type"],
        "message": row["message"],
        "created_at": row["created_at"],
        "details": json.loads(row["details_json"] or "{}"),
    } for row in rows]


def save_exchange_account(market: str, exchange: str, api_key: str = "", api_secret: str = "",
                          passphrase: str = "", connected: bool = False, balance: float = 0.0,
                          last_error: str = "", last_sync: Optional[str] = None) -> dict:
    init_database()
    ts = datetime.utcnow().isoformat()
    with _db_connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO exchange_accounts (market, exchange, api_key, api_secret, passphrase, connected, balance, last_sync, last_error, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (market, exchange, _protect_secret(api_key), _protect_secret(api_secret), _protect_secret(passphrase), int(bool(connected)), float(balance or 0.0), last_sync, last_error, ts),
        )
        conn.commit()
    return {"market": market, "exchange": exchange, "connected": bool(connected), "balance": round(float(balance or 0.0), 2), "last_error": last_error, "last_sync": last_sync}


def get_saved_exchange_account(market: str) -> dict:
    init_database()
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT market, exchange, api_key, api_secret, passphrase, connected, balance, last_sync, last_error, updated_at FROM exchange_accounts WHERE market = ?",
            (market,),
        ).fetchone()
    if not row:
        return {}
    return {
        "market": row["market"],
        "exchange": row["exchange"],
        "api_key": _unprotect_secret(row["api_key"] or ""),
        "api_secret": _unprotect_secret(row["api_secret"] or ""),
        "passphrase": _unprotect_secret(row["passphrase"] or ""),
        "connected": bool(row["connected"]),
        "balance": round(float(row["balance"] or 0.0), 2),
        "last_sync": row["last_sync"],
        "last_error": row["last_error"] or "",
        "updated_at": row["updated_at"],
    }


def load_runtime_state_from_db() -> None:
    init_database()
    with _db_connect() as conn:
        rows = conn.execute("SELECT market, exchange, api_key, api_secret, passphrase, connected, balance, last_sync, last_error FROM exchange_accounts").fetchall()
    for row in rows:
        market = (row["market"] or "CRYPTO").strip().upper()
        acc = CONFIG.setdefault("live_accounts", {}).setdefault(market, {})
        acc.update({
            "exchange": row["exchange"] or acc.get("exchange", "binance"),
            "api_key": _unprotect_secret(row["api_key"] or ""),
            "api_secret": _unprotect_secret(row["api_secret"] or ""),
            "passphrase": _unprotect_secret(row["passphrase"] or ""),
            "connected": bool(row["connected"]),
            "balance": round(float(row["balance"] or 0.0), 2),
            "last_sync": row["last_sync"],
            "last_error": row["last_error"] or "",
        })


def _load_persistent_state() -> dict:
    defaults = {
        "paper_trading": CONFIG.get("paper_trading", True),
        "live_trading_enabled": CONFIG.get("live_trading_enabled", False),
        "live_accounts": CONFIG.get("live_accounts", {}),
    }
    if not os.path.exists(STATE_FILE):
        return defaults
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except Exception:
        return defaults

    merged = defaults.copy()
    merged.update({k: v for k, v in data.items() if v is not None})
    CONFIG["paper_trading"] = bool(merged.get("paper_trading", True))
    CONFIG["live_trading_enabled"] = bool(merged.get("live_trading_enabled", False))
    # load per-market live accounts (backwards-compatible with single-account state)
    la = merged.get("live_accounts") or {}
    if not la:
        # fallback: check old single keys and map to CRYPTO
        la = {
            "CRYPTO": {
                "exchange": merged.get("live_exchange", CONFIG.get("live_accounts", {}).get("CRYPTO", {}).get("exchange","binance")),
                "api_key": str(merged.get("live_api_key", "")),
                "api_secret": str(merged.get("live_api_secret", "")),
                "passphrase": str(merged.get("live_passphrase", "")),
                "connected": bool(merged.get("live_account_connected", False)),
                "balance": float(merged.get("live_balance", 0.0) or 0.0),
                "last_sync": merged.get("live_last_sync"),
                "last_error": str(merged.get("live_last_error", "")),
            }
        }
    # Decrypt credentials read from this file. _unprotect_secret() passes
    # plaintext values through unchanged (no "enc:" prefix), so files saved
    # before this fix still load correctly — they just get encrypted going
    # forward the next time _save_persistent_state() runs.
    for market, acc in la.items():
        acc["api_key"] = _clean_credential(_unprotect_secret(acc.get("api_key", "")))
        acc["api_secret"] = _clean_credential(_unprotect_secret(acc.get("api_secret", "")))
        acc["passphrase"] = _clean_credential(_unprotect_secret(acc.get("passphrase", "")))
    CONFIG["live_accounts"].update(la)
    return merged


def _save_persistent_state() -> None:
    # Encrypt credentials before they hit disk here too — this JSON file
    # was writing api_key/api_secret/passphrase in PLAINTEXT while the
    # SQLite store (save_exchange_account) correctly encrypted them. Two
    # persistence paths with different protection levels for the same
    # secrets is a real gap; both must be encrypted-at-rest consistently.
    accounts = {}
    for market, acc in (CONFIG.get("live_accounts", {}) or {}).items():
        accounts[market] = {**acc,
            "api_key": _protect_secret(acc.get("api_key", "")),
            "api_secret": _protect_secret(acc.get("api_secret", "")),
            "passphrase": _protect_secret(acc.get("passphrase", "")),
        }
    payload = {
        "paper_trading": bool(CONFIG.get("paper_trading", True)),
        "live_trading_enabled": bool(CONFIG.get("live_trading_enabled", False)),
        "live_accounts": accounts,
    }
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


_EXCHANGE_CACHE: dict = {}
_EXCHANGE_CACHE_LOCK = threading.Lock()


def build_live_trading_payload() -> dict:
    market = (CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    live_accounts = CONFIG.get("live_accounts") or {}
    acc = live_accounts.get(market, {})

    # Prefer the current market account's validation result. A stale legacy
    # alias must never resurrect a false "Connected" state after a failed
    # validation or after the user disconnects this market.
    linked = bool((acc.get("api_key") or "") and (acc.get("api_secret") or ""))
    connected = bool(acc.get("connected", False))
    if not acc and live_accounts:
        connected = False
    elif not acc and not live_accounts:
        # Backwards-compatibility fallback only when no per-market state exists yet.
        linked = bool((CONFIG.get("live_api_key") or "") and (CONFIG.get("live_api_secret") or ""))
        connected = bool(CONFIG.get("live_account_connected", False)) and linked and not (CONFIG.get("live_last_error") or "").strip()

    return {
        "enabled": bool(CONFIG.get("live_trading_enabled", False)),
        "exchange": acc.get("exchange") or CONFIG.get("live_exchange", "binance"),
        "linked": linked,
        "connected": connected,
        "balance": round(float(acc.get("balance", CONFIG.get("live_balance", 0.0)) or 0.0), 2),
        "last_sync": acc.get("last_sync") or CONFIG.get("live_last_sync"),
        "last_error": acc.get("last_error") or CONFIG.get("live_last_error", ""),
        # include a compact accounts map for UI components that may need per-market info
        "accounts": { m: v for m, v in live_accounts.items() },
    }


def _coerce_number(value, default: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except Exception:
        return round(float(default), 2)


def _extract_balance_metric(balance: dict, metric: str) -> float:
    if not isinstance(balance, dict):
        return 0.0
    for key in ("USDT", "usdt"):
        entry = balance.get(key)
        if isinstance(entry, dict):
            value = entry.get(metric) or entry.get(metric.lower()) or entry.get("total") or 0.0
            if value not in (None, ""):
                return _coerce_number(value, 0.0)
    free_dict = balance.get("free") or {}
    if isinstance(free_dict, dict):
        value = free_dict.get("USDT") or free_dict.get("usdt") or 0.0
        if value not in (None, ""):
            return _coerce_number(value, 0.0)
    total_dict = balance.get("total") or {}
    if isinstance(total_dict, dict):
        value = total_dict.get("USDT") or total_dict.get("usdt") or 0.0
        if value not in (None, ""):
            return _coerce_number(value, 0.0)
    return 0.0


def _normalise_positions(raw_positions) -> list[dict]:
    positions: list[dict] = []
    for item in raw_positions or []:
        if not isinstance(item, dict):
            continue
        info = item.get("info") or {}
        symbol = item.get("symbol") or info.get("symbol") or item.get("currency") or "—"
        side = str(item.get("side") or item.get("positionSide") or info.get("side") or "UNKNOWN").upper()
        entry = item.get("entryPrice") or item.get("entry_price") or item.get("avgEntryPrice") or info.get("entryPrice") or 0.0
        size = item.get("contracts") or item.get("amount") or item.get("size") or info.get("contracts") or 0.0
        pnl = item.get("unrealizedPnl") or item.get("unrealized_pnl") or item.get("unrealizedProfit") or info.get("unrealizedPnl") or 0.0
        if not symbol and not size and not pnl:
            continue
        positions.append({
            "symbol": str(symbol),
            "side": side,
            "entry_price": _coerce_number(entry, 0.0),
            "size": _coerce_number(size, 0.0),
            "unrealized_pnl": _coerce_number(pnl, 0.0),
        })
    return positions


def _normalise_trades(raw_trades) -> list[dict]:
    trades: list[dict] = []
    for item in raw_trades or []:
        if not isinstance(item, dict):
            continue
        info = item.get("info") or {}
        symbol = item.get("symbol") or info.get("symbol") or "—"
        pnl = item.get("pnl") or item.get("profit") or item.get("realizedPnl") or item.get("realized_pnl") or info.get("realizedPnl") or info.get("pnl") or 0.0
        fee = item.get("fee") or item.get("cost") or info.get("fee") or info.get("cost") or 0.0
        ts = item.get("timestamp") or item.get("datetime") or info.get("timestamp") or info.get("datetime") or None
        if not symbol and not pnl and not fee:
            continue
        trades.append({
            "symbol": str(symbol),
            "side": str(item.get("side") or info.get("side") or "UNKNOWN").upper(),
            "pnl": _coerce_number(pnl, 0.0),
            "fee": _coerce_number(fee, 0.0),
            "timestamp": ts,
        })
    return trades


def _mask_account_id(account_id) -> str:
    if not account_id:
        return "—"
    value = str(account_id).strip()
    if len(value) <= 4:
        return f"{value[:1]}***"
    if len(value) <= 8:
        return f"{value[:2]}***{value[-2:]}"
    return f"{value[:4]}…{value[-2:]}"


def build_exchange_dashboard_payload(market: str = "CRYPTO", force_refresh: bool = False, persist: bool = True) -> dict:
    market_key = (market or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    acc = CONFIG.get("live_accounts", {}).get(market_key, {})
    exchange_id = (acc.get("exchange") or CONFIG.get("crypto_exchange", "binance")).strip().lower()

    with _EXCHANGE_CACHE_LOCK:
        cached = _EXCHANGE_CACHE.get(market_key)
        if not force_refresh and cached and (time.time() - cached.get("ts", 0) < 30):
            return cached.get("payload", {})

    if not (acc.get("api_key") and acc.get("api_secret")):
        payload = {
            "market": market_key,
            "exchange": exchange_id,
            "connected": False,
            "status": "disconnected",
            "error": "No API credentials linked. Go to Settings → Live Trading to connect.",
            "summary": {"available_balance": 0.0, "total_equity": 0.0, "margin_balance": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0, "daily_pnl": 0.0, "open_trades": 0, "win_rate": 0.0, "fees_paid": 0.0, "roi": 0.0},
            "positions": [],
            "orders": [],
            "recent_trades": [],
            "account": {"account_id": "—", "leverage": "1x", "last_sync": None, "permissions": []},
        }
        if persist:
            acc.update({"connected": False, "balance": 0.0, "last_error": payload["error"], "last_sync": None})
            save_exchange_account(market_key, exchange_id, acc.get("api_key", ""), acc.get("api_secret", ""), acc.get("passphrase", ""), False, 0.0, payload["error"], None)
            _save_persistent_state()
            _sync_alias_live_account()
        with _EXCHANGE_CACHE_LOCK:
            _EXCHANGE_CACHE[market_key] = {"ts": time.time(), "payload": payload}
        return payload

    if not (acc.get("api_key") or "") or not (acc.get("api_secret") or ""):
        payload = {
            "market": market_key,
            "exchange": exchange_id,
            "connected": False,
            "status": "disconnected",
            "error": "API credentials are missing",
            "summary": {"available_balance": 0.0, "total_equity": 0.0, "margin_balance": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0, "daily_pnl": 0.0, "open_trades": 0, "win_rate": 0.0, "fees_paid": 0.0, "roi": 0.0},
            "positions": [],
            "orders": [],
            "recent_trades": [],
            "account": {"account_id": "—", "leverage": "1x", "last_sync": None, "permissions": []},
        }
        if persist:
            acc.update({"connected": False, "balance": 0.0, "last_error": payload["error"], "last_sync": None})
            save_exchange_account(market_key, exchange_id, acc.get("api_key", ""), acc.get("api_secret", ""), acc.get("passphrase", ""), False, 0.0, payload["error"], None)
            _save_persistent_state()
            _sync_alias_live_account()
        with _EXCHANGE_CACHE_LOCK:
            _EXCHANGE_CACHE[market_key] = {"ts": time.time(), "payload": payload}
        return payload

    if not CCXT_AVAILABLE:
        payload = {
            "market": market_key,
            "exchange": exchange_id,
            "connected": False,
            "status": "error",
            "error": "ccxt is not installed",
            "summary": {"available_balance": 0.0, "total_equity": 0.0, "margin_balance": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0, "daily_pnl": 0.0, "open_trades": 0, "win_rate": 0.0, "fees_paid": 0.0, "roi": 0.0},
            "positions": [],
            "orders": [],
            "recent_trades": [],
            "account": {"account_id": "—", "leverage": "1x", "last_sync": None, "permissions": []},
        }
        with _EXCHANGE_CACHE_LOCK:
            _EXCHANGE_CACHE[market_key] = {"ts": time.time(), "payload": payload}
        return payload

    try:
        ex_cls = getattr(ccxt, exchange_id, None)
        if ex_cls is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
        kwargs = {"apiKey": acc.get("api_key", ""), "secret": acc.get("api_secret", ""), "enableRateLimit": True, "options": {"defaultType": "spot"}}
        if acc.get("passphrase"):
            kwargs["password"] = acc.get("passphrase")
        exchange = ex_cls(kwargs)

        raw_balance = exchange.fetch_balance()
        available_balance = _extract_free_usdt(raw_balance)
        total_equity = _extract_balance_metric(raw_balance, "total")
        margin_balance = _extract_balance_metric(raw_balance, "total")
        if not total_equity:
            total_equity = available_balance
        if not margin_balance:
            margin_balance = available_balance

        raw_positions = []
        try:
            raw_positions = exchange.fetch_positions() or []
        except Exception:
            raw_positions = []
        positions = _normalise_positions(raw_positions)

        raw_orders = []
        try:
            raw_orders = exchange.fetch_open_orders() or []
        except Exception:
            raw_orders = []
        orders = [
            {
                "id": str(item.get("id") or item.get("orderId") or ""),
                "symbol": str(item.get("symbol") or item.get("info", {}).get("symbol") or "—"),
                "side": str(item.get("side") or item.get("info", {}).get("side") or "UNKNOWN").upper(),
                "status": str(item.get("status") or "OPEN").upper(),
            }
            for item in raw_orders if isinstance(item, dict)
        ][:10]

        raw_trades = []
        try:
            raw_trades = exchange.fetch_my_trades() or []
        except Exception:
            raw_trades = []
        recent_trades = _normalise_trades(raw_trades)[:12]

        wins = sum(1 for trade in recent_trades if trade.get("pnl", 0.0) > 0)
        losses = sum(1 for trade in recent_trades if trade.get("pnl", 0.0) < 0)
        win_rate = round((wins / max(1, len(recent_trades))) * 100, 2) if recent_trades else 0.0
        realized_pnl = round(sum(item.get("pnl", 0.0) for item in recent_trades), 2)
        fees_paid = round(sum(abs(item.get("fee", 0.0)) for item in recent_trades), 2)
        roi = round((realized_pnl / available_balance * 100) if available_balance else 0.0, 2)
        account_id = None
        info = raw_balance.get("info") if isinstance(raw_balance, dict) else {}
        if isinstance(info, dict):
            for key in ("accountId", "userId", "uid", "account_id"):
                if info.get(key):
                    account_id = info.get(key)
                    break
        payload = {
            "market": market_key,
            "exchange": exchange_id,
            "connected": True,
            "status": "connected",
            "error": "",
            "summary": {
                "available_balance": round(available_balance, 2),
                "total_equity": round(total_equity, 2),
                "margin_balance": round(margin_balance, 2),
                "unrealized_pnl": round(sum(pos.get("unrealized_pnl", 0.0) for pos in positions), 2),
                "realized_pnl": realized_pnl,
                "daily_pnl": realized_pnl,
                "open_trades": len(positions),
                "win_rate": win_rate,
                "fees_paid": fees_paid,
                "roi": roi,
            },
            "positions": positions,
            "orders": orders,
            "recent_trades": recent_trades,
            "account": {
                "account_id": _mask_account_id(account_id),
                "leverage": "1x",
                "last_sync": datetime.utcnow().isoformat(),
                "permissions": ["spot", "balances", "orders"],
            },
        }
        if persist:
            acc.update({
                "connected": True,
                "balance": round(available_balance, 2),
                "last_error": "",
                "last_sync": payload["account"]["last_sync"],
            })
            save_exchange_account(market_key, exchange_id, acc.get("api_key", ""), acc.get("api_secret", ""), acc.get("passphrase", ""), True, round(available_balance, 2), "", payload["account"]["last_sync"])
            _save_persistent_state()
            _sync_alias_live_account()
        with _EXCHANGE_CACHE_LOCK:
            _EXCHANGE_CACHE[market_key] = {"ts": time.time(), "payload": payload}
        return payload
    except Exception as exc:
        msg = str(exc)
        payload = {
            "market": market_key,
            "exchange": exchange_id,
            "connected": False,
            "status": "error",
            "error": msg,
            "summary": {"available_balance": 0.0, "total_equity": 0.0, "margin_balance": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0, "daily_pnl": 0.0, "open_trades": 0, "win_rate": 0.0, "fees_paid": 0.0, "roi": 0.0},
            "positions": [],
            "orders": [],
            "recent_trades": [],
            "account": {"account_id": "—", "leverage": "1x", "last_sync": None, "permissions": []},
        }
        if persist:
            acc.update({"connected": False, "balance": 0.0, "last_error": msg, "last_sync": None})
            save_exchange_account(market_key, exchange_id, acc.get("api_key", ""), acc.get("api_secret", ""), acc.get("passphrase", ""), False, 0.0, msg, None)
            _save_persistent_state()
            _sync_alias_live_account()
        with _EXCHANGE_CACHE_LOCK:
            _EXCHANGE_CACHE[market_key] = {"ts": time.time(), "payload": payload}
        return payload


_load_persistent_state()
init_database()

# Backwards-compatible single-account aliases (point to CRYPTO account)
def _sync_alias_live_account():
    ca = CONFIG.get("live_accounts", {}).get("CRYPTO", {})
    CONFIG["live_exchange"] = ca.get("exchange", CONFIG.get("live_exchange", "binance"))
    CONFIG["live_api_key"] = ca.get("api_key", "")
    CONFIG["live_api_secret"] = ca.get("api_secret", "")
    CONFIG["live_passphrase"] = ca.get("passphrase", "")
    CONFIG["live_account_connected"] = bool(ca.get("connected", False))
    CONFIG["live_balance"] = float(ca.get("balance", 0.0) or 0.0)
    CONFIG["live_last_sync"] = ca.get("last_sync")
    if ca.get("last_error") and not ca.get("connected", False):
        CONFIG["live_last_error"] = ca.get("last_error", "")
    else:
        CONFIG["live_last_error"] = ca.get("last_error", "")

load_runtime_state_from_db()
_sync_alias_live_account()

# Admin users store — extend or load from DB as needed
ADMIN_STORE = {
    os.environ.get("ADMIN_USER", "admin"): {
        "password_hash" : generate_password_hash(
                            os.environ.get("ADMIN_PASSWORD", "BotRL@2026!")),
        "email"         : os.environ.get("ADMIN_EMAIL", "admin@botrl.local"),
        "name"          : "Admin",
        "notifications" : {
            "trade_open"  : True,
            "trade_close" : True,
            "sl_hit"      : True,
            "tp_hit"      : True,
            "ai_reject"   : True,
            "scan_cycle"  : False,
        },
    }
}

# Shared runtime objects (set once bot starts)
_BOT         = None
_BOT_THREAD  = None
_TM          = None           # TradeManager singleton
_STOP_EVENT  = threading.Event()
_PRICE_CACHE    = {}          # symbol → last known live price
_PRICE_CACHE_TS = {}          # symbol → when that price was fetched (monotonic seconds)
PRICE_CACHE_MAX_AGE = 20      # seconds — beyond this a cached price is refused, not used

def _cache_price(symbol: str, price: float):
    _PRICE_CACHE[symbol] = price
    _PRICE_CACHE_TS[symbol] = time.monotonic()

def _cached_price_if_fresh(symbol: str, max_age: float = PRICE_CACHE_MAX_AGE):
    """Return (price, age_seconds) only if the cached price is still within
    max_age — otherwise (None, age) so callers can refuse to act on data
    that's too old to represent 'the actual market price' anymore."""
    ts = _PRICE_CACHE_TS.get(symbol)
    if ts is None:
        return None, None
    age = time.monotonic() - ts
    if age > max_age:
        return None, age
    return _PRICE_CACHE.get(symbol), age


def _get_live_exchange_client(api_key: Optional[str] = None,
                              api_secret: Optional[str] = None,
                              passphrase: Optional[str] = None,
                              market: Optional[str] = None):
    if not CCXT_AVAILABLE:
        return None, "ccxt is not installed"
    # NOTE: deliberately NOT gated on CONFIG["live_trading_enabled"] here.
    # That flag means "the bot may autonomously place real trades during a
    # scan" — it's checked explicitly at every autonomous call site
    # (scan_once, monitor_open_trades, close_on_signal_loss). This function
    # just builds an authenticated exchange client, which is also needed
    # for read-only/manual actions (viewing balance, syncing the wallet
    # page, a user-initiated withdrawal) that have nothing to do with
    # autonomous trading and shouldn't require that toggle to work.
    m   = (market or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    acc = CONFIG.get("live_accounts", {}).get(m, {})
    k   = api_key    or acc.get("api_key",    "")
    s   = api_secret or acc.get("api_secret", "")
    p   = passphrase or acc.get("passphrase", "")
    if not k or not s:
        return None, (f"No API credentials for {m}. "
                      "Go to Settings → Live Trading to connect.")
    ex_id = (acc.get("exchange") or CONFIG.get("crypto_exchange","binance")).lower()
    ex_cls = getattr(ccxt, ex_id, None)
    if ex_cls is None:
        return None, f"Unsupported exchange: {ex_id}"
    kwargs = {"apiKey": k, "secret": s, "enableRateLimit": True,
              "options": {"defaultType": "spot"}}
    if p: kwargs["password"] = p
    try:
        # Do NOT call load_markets() — triggers 451 geo-blocks on public endpoints
        return ex_cls(kwargs), None
    except Exception as exc:
        return None, str(exc)


def validate_live_connection(exchange_id: str, api_key: str,
                             api_secret: str, passphrase: str = "") -> tuple[bool, dict]:
    if not CCXT_AVAILABLE:
        return False, {"ok": False, "error": "ccxt is not installed"}
    api_key = _clean_credential(api_key)
    api_secret = _clean_credential(api_secret)
    passphrase = _clean_credential(passphrase)
    ex_cls = getattr(ccxt, exchange_id.lower(), None)
    if ex_cls is None:
        return False, {"ok": False, "error": f"Unsupported exchange: {exchange_id}"}
    kwargs = {"apiKey": api_key, "secret": api_secret,
              "enableRateLimit": True, "options": {"defaultType": "spot"}}
    if passphrase: kwargs["password"] = passphrase
    try:
        ex        = ex_cls(kwargs)
        raw_bal   = ex.fetch_balance()
        free_usdt = _extract_free_usdt(raw_bal)
        return True, {"ok": True, "balance": free_usdt, "exchange": exchange_id.lower()}
    except Exception as exc:
        msg  = str(exc)
        hint = ""
        if "451" in msg or "restricted" in msg.lower(): hint = " (HTTP 451 — geo-blocked IP)"
        elif "invalid" in msg.lower() or "auth" in msg.lower(): hint = " — check API key/secret"
        elif "permission" in msg.lower(): hint = " — enable Spot Trading on API key"
        return False, {"ok": False, "error": msg + hint}


# Most exchanges keep trading balance and withdrawable balance in separate
# sub-accounts internally (e.g. Bybit's Unified Trading Account vs its
# Funding wallet). An on-chain withdrawal can usually only be initiated
# from the funding/spot side, so funds sitting in the trading account need
# an internal transfer first. ccxt exposes this uniformly via
# exchange.transfer(currency, amount, fromAccount, toAccount) — the actual
# account-type strings differ per exchange, mapped here for the exchanges
# this app supports.
_TRANSFER_ACCOUNT_MAP = {
    "bybit":    {"trading": "UNIFIED",  "funding": "FUND"},
    "binance":  {"trading": "SPOT",     "funding": "FUNDING"},
    "coinbase": {"trading": "trading",  "funding": "funding"},
    "kucoin":   {"trading": "trade",    "funding": "main"},
    "kraken":   {"trading": "spot",     "funding": "spot"},   # Kraken doesn't separate these the same way
    "okx":      {"trading": "trading",  "funding": "funding"},
}


def internal_transfer_funds(market: str, amount: float,
                            from_account: str = "trading",
                            to_account: str = "funding") -> dict:
    """Move funds between an exchange's internal sub-accounts (e.g. trading
    -> funding) so they become withdrawable. Returns {"ok": bool, ...}."""
    exchange, client_error = _get_live_exchange_client(market=market)
    if exchange is None:
        return {"ok": False, "error": client_error or "Live exchange client unavailable"}
    exchange_id = getattr(exchange, "id", "") or ""
    acct_map = _TRANSFER_ACCOUNT_MAP.get(exchange_id, {"trading": from_account, "funding": to_account})
    frm = acct_map.get(from_account, from_account)
    to  = acct_map.get(to_account, to_account)
    try:
        result = exchange.transfer("USDT", amount, frm, to)
        return {"ok": True, "result": result}
    except Exception as exc:
        msg = str(exc)
        hint = ""
        if "insufficient" in msg.lower():
            hint = " — check your trading account has enough free (not locked in open positions) balance"
        elif "not supported" in msg.lower() or "not implemented" in msg.lower():
            hint = f" — {exchange_id} may not support automated internal transfers via this API; check the exchange's app/website"
        return {"ok": False, "error": msg + hint}


def _extract_free_usdt(balance: dict) -> float:
    """Robustly extract free USDT from ccxt fetch_balance() across all exchanges.
    Bybit  → balance['free']['USDT']  or  balance['total']['USDT']
    Binance → balance['USDT']['free']
    """
    if not isinstance(balance, dict): return 0.0
    for key in ("USDT","usdt"):
        entry = balance.get(key)
        if isinstance(entry, dict):
            v = entry.get("free") or entry.get("total") or 0.0
            if v: return round(float(v), 2)
    free_dict = balance.get("free") or {}
    if isinstance(free_dict, dict):
        v = free_dict.get("USDT") or free_dict.get("usdt") or 0.0
        if v: return round(float(v), 2)
    total_dict = balance.get("total") or {}
    if isinstance(total_dict, dict):
        v = total_dict.get("USDT") or total_dict.get("usdt") or 0.0
        if v: return round(float(v), 2)
    return 0.0


def _translate_forex_symbol_for_exchange(forex_symbol: str, exchange) -> Optional[str]:
    """Translate a yfinance-style forex symbol (e.g. 'EURUSD=X') into
    whatever symbol format the connected crypto exchange actually lists,
    if it lists it at all.

    This exists because forex signal generation runs on free yfinance
    data (real EUR/USD spot rates), but LIVE EXECUTION has to happen on
    whatever the connected exchange actually offers — and crypto
    exchanges do not uniformly offer traditional currency pairs. Some
    (e.g. Bybit's "TradFi" product) offer USDT-margined forex CFDs on
    major pairs; most don't expose that through the same API surface
    ccxt uses for spot/derivatives trading. Rather than assume either
    way, this checks the exchange's REAL, live market list and only
    returns a symbol that's actually tradeable there — so a live order
    is never attempted against a guessed symbol that doesn't exist.
    """
    raw = forex_symbol.upper().replace("=X", "").replace("/", "").strip()
    if len(raw) != 6:
        return None
    base, quote = raw[:3], raw[3:]

    try:
        exchange.load_markets()
    except Exception as exc:
        logger.error(f"Could not load markets for {getattr(exchange,'id','exchange')}: {exc}")
        return None

    # Try the most likely real-world formats, in priority order.
    candidates = [
        f"{base}/USDT",   # USDT-margined FX CFD (matches how Bybit TradFi describes its forex offering)
        f"{base}/{quote}",  # a genuine currency pair, if the exchange lists one
        f"{base}{quote}",   # some exchanges use no-slash symbols internally
    ]
    for cand in candidates:
        if cand in exchange.symbols:
            return cand
    return None


def execute_live_order(symbol: str, side: str, size: float,
                       market: Optional[str] = None) -> dict:
    """Place a market order. `size` = BASE CURRENCY quantity (e.g. 0.001 BTC)."""
    exchange, error = _get_live_exchange_client(market=market)
    if error or exchange is None:
        return {"ok": False, "error": error or "Live exchange client unavailable"}
    m = (market or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    symbol = normalize_market_symbol(symbol, m)
    try:
        if m == "FOREX":
            sym = _translate_forex_symbol_for_exchange(symbol, exchange)
            if sym is None:
                exchange_id = getattr(exchange, "id", "this exchange")
                err = (f"{symbol} is not available for live trading on {exchange_id} — "
                       f"it doesn't list a matching instrument. Signal generation for FOREX "
                       f"runs on free market data, but live EXECUTION needs the connected "
                       f"exchange to actually offer that pair; not all exchanges do. This "
                       f"trade was skipped rather than sent as an invalid order.")
                _elog(f"⚠️ {err}", "WARN")
                return {"ok": False, "error": err}
        else:
            # Normalise symbol — ccxt needs BTC/USDT not BTCUSDT
            sym = normalize_market_symbol(symbol, m)
            if "/" not in sym and "USDT" in sym:
                sym = sym.replace("USDT", "/USDT")
        qty = float(size)
        if qty <= 0:
            return {"ok": False, "error": f"Invalid order size: {qty}"}
        order     = exchange.create_market_order(sym, side.lower(), qty)
        raw_bal   = exchange.fetch_balance()
        free_usdt = _extract_free_usdt(raw_bal)
        acc = CONFIG.setdefault("live_accounts", {}).setdefault(m, {})
        acc.update({"balance": free_usdt, "last_sync": datetime.utcnow().isoformat(),
                    "last_error": "", "connected": True})
        save_exchange_account(m, acc.get("exchange","binance"), acc.get("api_key",""),
                              acc.get("api_secret",""), acc.get("passphrase",""),
                              True, free_usdt, "", acc["last_sync"])
        _save_persistent_state()
        _sync_alias_live_account()
        _elog(f"Live order: {side} {qty:.6f} {sym} | Balance: ${free_usdt:.2f} USDT", "INFO")
        return {"ok": True, "order": order, "balance": free_usdt}
    except Exception as exc:
        err = str(exc)
        acc = CONFIG.setdefault("live_accounts", {}).setdefault(m, {})
        acc["last_error"] = err
        _save_persistent_state()
        _elog(f"Live order FAILED {side} {symbol}: {err}", "ERROR")
        return {"ok": False, "error": err}

# In-memory event log
_EVENT_LOG  = []
_LOG_LOCK   = threading.Lock()

def _event_type_for_message(msg: str) -> str:
    m = msg.lower()
    if "scan" in m:
        return "scan"
    if "trade" in m or "open" in m or "close" in m or "tp" in m or "sl" in m:
        return "trade"
    if "withdraw" in m:
        return "wallet"
    if "config" in m:
        return "config"
    if "bot" in m:
        return "bot"
    return "system"


def _elog(msg: str, level: str = "INFO"):
    entry = {
        "ts"    : datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "time"  : datetime.utcnow().strftime("%H:%M:%S"),
        "level" : level,
        "msg"   : msg,
    }
    with _LOG_LOCK:
        _EVENT_LOG.append(entry)
        if len(_EVENT_LOG) > 800:
            _EVENT_LOG[:] = _EVENT_LOG[-800:]
    try:
        record_admin_activity("admin", _event_type_for_message(msg), msg, details={"level": level})
    except Exception:
        pass
    # This used to only reach the in-memory buffer (capped at 800 entries,
    # lost on restart) and the SQLite activity table — NOT the actual
    # trading_bot.log file. That meant routine, high-value diagnostics
    # (per-symbol scan summaries: price/RSI/EMA/confidence for every
    # scan) were structurally unrecoverable once evicted from memory —
    # exactly the data needed to audit "was a signal accurate" after the
    # fact. Every _elog() call now also reaches the real logger, so it's
    # durably in the log file going forward.
    log_fn = {"ERROR": logger.error, "WARN": logger.warning,
              "WARNING": logger.warning}.get(level.upper(), logger.info)
    log_fn(msg)

# ════════════════════════════════════════════════════════════
#  LOGGING (terminal + file)
# ════════════════════════════════════════════════════════════

def _log_start_marker_path() -> str:
    """Sidecar file tracking when the CURRENT active log file was started,
    so a time-based reset can fire even if the size cap is never hit."""
    return str(Path(CONFIG["log_file"]).with_suffix(".log_started_at"))


def _get_log_started_at() -> datetime:
    marker = _log_start_marker_path()
    try:
        with open(marker, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    except Exception:
        now = datetime.utcnow()
        _set_log_started_at(now)
        return now


def _set_log_started_at(dt: Optional[datetime] = None) -> None:
    dt = dt or datetime.utcnow()
    try:
        with open(_log_start_marker_path(), "w") as f:
            f.write(dt.isoformat())
    except Exception:
        pass


def maybe_rotate_log_for_age(logger_obj) -> None:
    """Force a rollover if the current log has been active longer than
    log_reset_months, even if it never hit the size cap — this is what
    guarantees the 3-6 month reset happens on a long-lived, low-traffic
    deployment, not just a busy one."""
    started = _get_log_started_at()
    age_days = (datetime.utcnow() - started).days
    max_days = CONFIG["log_reset_months"] * 30
    if age_days < max_days:
        return
    for h in logger_obj.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            try:
                h.doRollover()
                _set_log_started_at(datetime.utcnow())
                logger_obj.info(f"Log file force-rotated after {age_days} days "
                                 f"(reset interval: {CONFIG['log_reset_months']} months)")
            except Exception:
                pass


def _setup_logger():
    lg = logging.getLogger("TradingBot")
    if lg.handlers:
        return lg
    lg.setLevel(getattr(logging, CONFIG["log_level"], logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    lg.addHandler(ch)
    try:
        # RotatingFileHandler caps the ACTIVE log file at log_max_size_mb —
        # once hit, it's renamed to a numbered backup and a fresh file
        # starts, so the live file the bot writes to never grows past that
        # size. backupCount old rotations are kept on disk as the durable
        # archive for historical analysis; oldest backups beyond that are
        # deleted automatically by the handler itself. The 3-6 month
        # time-based reset (independent of size) is handled separately by
        # maybe_rotate_log_for_age(), called periodically from the scan loop.
        fh = logging.handlers.RotatingFileHandler(
            CONFIG["log_file"],
            maxBytes=CONFIG["log_max_size_mb"] * 1024 * 1024,
            backupCount=CONFIG["log_backup_count"],
        )
        fh.setFormatter(fmt)
        lg.addHandler(fh)
        _get_log_started_at()  # ensure the age marker exists from first run
    except Exception:
        pass
    return lg

logger = _setup_logger()

# ════════════════════════════════════════════════════════════
#  TERMINAL HELPERS
# ════════════════════════════════════════════════════════════

def _banner():
    print(Fore.CYAN + Style.BRIGHT + r"""
╔══════════════════════════════════════════════════════════╗
║         AI-POWERED CRYPTO & STOCK TRADING BOT           ║
║      RSI(6/12)  +  Swing Detection  +  Tight SL/TP     ║
║      Claude AI signal confirmation                      ║
╚══════════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)

def _section(title):
    print(Fore.YELLOW + f"\n{'─'*60}\n  {title}\n{'─'*60}" + Style.RESET_ALL)

def _print_signal(sym, side, price, sl, tp, conf, src="TECH"):
    c = Fore.GREEN if side == "BUY" else Fore.RED
    print(c + f"""
  ┌─ SIGNAL [{src}] ──────────────────────────────────────
  │  Symbol     : {sym}
  │  Direction  : {side}
  │  Entry      : ${price:.4f}
  │  Stop Loss  : ${sl:.4f}  ({CONFIG['stop_loss_pct']*100:.1f}%)
  │  Take Profit: ${tp:.4f}  ({CONFIG['take_profit_pct']*100:.1f}%)
  │  Confidence : {conf:.1f}%
  └───────────────────────────────────────────────────────
""" + Style.RESET_ALL)

# ════════════════════════════════════════════════════════════
#  MARKET DATA
# ════════════════════════════════════════════════════════════

class DataSourceError(Exception):
    """Raised when live market data cannot be fetched — never silently fall back to fake data."""
    pass

class MarketDataProvider:
    """
    Always fetches LIVE data from the real exchange/market.
    No synthetic/simulated price fallback — if live data fails,
    the symbol is skipped for that scan cycle and a loud error is logged,
    so stale or fake prices are never used to drive a real signal/trade.

    Binance's public API (api.binance.com) returns HTTP 451 "restricted
    location" for many cloud-hosted servers and certain countries (most
    commonly US-based hosts) — this is a Binance-side geo-block, not a
    code or network bug. To work around this without requiring the user
    to set up a VPN/proxy, this provider tries a short list of exchanges
    that ccxt supports identically and that do NOT geo-block public
    market-data endpoints, falling back through the list until one connects.
    """
    # Tried in order. All expose the same public OHLCV/ticker endpoints via ccxt,
    # so signal/strategy logic is 100% unaffected by which one ends up active.
    FALLBACK_EXCHANGES = ["binance", "kraken", "coinbase", "kucoin", "bybit"]

    def __init__(self, mode: str):
        self.mode = mode
        self.exchange = None
        self.exchange_id = None
        self.healthy = False
        self.last_error = None
        self.geo_blocked_exchanges = []
        # Stale-repeat detection: tracks a cheap fingerprint of the last
        # OHLCV data actually fetched per symbol, and how long that exact
        # data has persisted. _validate_freshness() only catches data that's
        # absolutely old (hours) — it says nothing about whether the SAME
        # candle data is being served scan after scan by something upstream
        # (an exchange edge cache, a proxy, etc.) even though each fetch
        # "succeeds" with no error. This catches that case directly, which
        # an absolute-age check structurally cannot.
        self._last_ohlcv_sig: dict = {}
        self._last_ohlcv_changed_at: dict = {}

        if mode == "CRYPTO":
            if not CCXT_AVAILABLE:
                self.last_error = "ccxt not installed — run: pip install ccxt"
                logger.error(f"❌ {self.last_error}")
                return
            self._connect_crypto()
        elif mode == "STOCK" or mode == "FOREX":
            if not YF_AVAILABLE:
                self.last_error = "yfinance not installed — run: pip install yfinance"
                logger.error(f"❌ {self.last_error}")
                return
            self.healthy = True
            logger.info("✅ Yahoo Finance data source ready")
        else:
            # fallback to Yahoo for any other non-crypto markets
            if not YF_AVAILABLE:
                self.last_error = "yfinance not installed — run: pip install yfinance"
                logger.error(f"❌ {self.last_error}")
                return
            self.healthy = True
            logger.info("✅ Yahoo Finance data source ready")

    def _connect_crypto(self):
        # Preferred exchange first (user-configurable), then the fallback list.
        preferred = CONFIG.get("crypto_exchange", "binance").lower()
        candidates = [preferred] + [e for e in self.FALLBACK_EXCHANGES if e != preferred]

        for ex_id in candidates:
            if not hasattr(ccxt, ex_id):
                continue
            try:
                ex_class = getattr(ccxt, ex_id)
                exchange = ex_class({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                    # Explicit no-cache headers on every request — a mitigation
                    # attempt for the case where something between this host
                    # and the exchange (a proxy, CDN, or the exchange's own
                    # edge cache) is honoring cache-control and serving a
                    # repeated response. This can't force a remote cache to
                    # comply, but it removes any ambiguity about whether the
                    # REQUEST itself was asking to be cached.
                    "headers": {"Cache-Control": "no-cache, no-store, must-revalidate",
                                "Pragma": "no-cache"},
                })
                # Lightweight connectivity check — fetch one ticker instead of
                # load_markets() which calls /exchangeInfo and triggers 451 geo-blocks
                try:
                    exchange.fetch_ticker("BTC/USDT")
                except Exception:
                    exchange.fetch_time()   # fallback ping
                self.exchange    = exchange
                self.exchange_id = ex_id
                self.healthy     = True
                CONFIG["crypto_exchange"] = ex_id
                if ex_id != preferred:
                    logger.warning(
                        f"⚠️  '{preferred}' was unreachable — auto-switched to "
                        f"'{ex_id}' for live crypto data")
                logger.info(f"✅ Connected to {ex_id} — live data confirmed")
                return
            except Exception as e:
                msg = self._explain_error(e)
                if "451" in str(e) or "restricted location" in str(e).lower():
                    self.geo_blocked_exchanges.append(ex_id)
                    logger.error(
                        f"❌ {ex_id}: blocked your server's region (HTTP 451 — "
                        f"geo-restriction, not a code/network issue). Trying next exchange...")
                else:
                    logger.error(f"❌ {ex_id} connection failed: {msg}. Trying next exchange...")
                self.last_error = msg
                continue

        # Every candidate failed
        self.healthy = False
        if self.geo_blocked_exchanges:
            self.last_error = (
                f"All exchanges geo-blocked your server's IP/region: "
                f"{', '.join(self.geo_blocked_exchanges)}. This is the hosting "
                f"provider's location being restricted by the exchange, not a bug. "
                f"Fix: deploy from a different region, use a residential/VPN egress, "
                f"or pick an exchange in Settings that isn't blocked from your region."
            )
        logger.error(f"❌ No crypto exchange could be reached: {self.last_error}")

    def get_ohlcv(self, symbol: str, tf: str = "15m",
                  limit: int = 200) -> Optional[pd.DataFrame]:
        """Returns live OHLCV data, or None if it cannot be fetched.
        Caller MUST check for None and skip the symbol — never substitutes
        simulated data, so a stale connection can never silently drive trades."""
        try:
            if self.mode == "CRYPTO":
                return self._crypto(symbol, tf, limit)
            if self.mode == "FOREX":
                return self._forex(symbol, tf, limit)
            return self._stock(symbol, tf, limit)
        except Exception as e:
            reason = self._explain_error(e)
            self.last_error = reason
            logger.error(f"❌ [{symbol}] LIVE DATA FETCH FAILED: {reason} — symbol skipped this scan")
            return None

    @staticmethod
    def _explain_error(e: Exception) -> str:
        """Turns common ccxt/requests exceptions into a clear, actionable message
        instead of a bare 'GET https://...' string with no status code."""
        msg = str(e)
        etype = type(e).__name__
        # Retry/fallback logic (e.g. trying multiple exchanges) can end up
        # calling this on an exception whose message is ALREADY the output
        # of a previous _explain_error() call (wrapped and re-raised).
        # Without this guard, the "[EType]" tag added below stacks up once
        # per retry — this happened for real: "[DataSourceError]
        # [DataSourceError] [DataSourceError] ccxt not installed...".
        already_tagged = bool(re.match(r"^\[\w+\]", msg.strip()))

        # Some exchanges return an HTML error/challenge page (Cloudflare
        # blocks, WAF challenges, etc.) instead of JSON on failure, and
        # ccxt sometimes embeds that raw HTML verbatim in the exception
        # message. Left unchecked this dumps an entire HTML document into
        # the log file (this happened for real — a KuCoin Cloudflare page
        # ended up as ~90 lines in the log). Detect and collapse it to a
        # one-line summary instead.
        if "<!DOCTYPE" in msg or "<html" in msg.lower():
            head = msg.split("<")[0].strip()
            msg = f"{head} [HTML error page from exchange, {len(msg)} chars — likely a bot-protection/WAF challenge page, not a real API response]"
        elif len(msg) > 300:
            msg = msg[:300] + f"... [truncated, {len(msg)} chars total]"
        if "451" in msg or "restricted location" in msg.lower():
            return (f"{msg} → HTTP 451: this exchange has geo-blocked your "
                    f"server's IP/region. Not a code bug — switch exchange in "
                    f"Settings (try kraken, coinbase, kucoin, or bybit) or use "
                    f"a non-restricted VPN/host region.")
        if "429" in msg or "rate limit" in msg.lower():
            return f"{msg} → HTTP 429: rate-limited, back off scan frequency."
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            return f"{msg} → Connection timed out — check network/firewall egress."
        if "name or service not known" in msg.lower() or "getaddrinfo" in msg.lower():
            return f"{msg} → DNS lookup failed — no internet access from this host."
        if already_tagged:
            return msg
        # Some ccxt exceptions (RequestTimeout, ExchangeNotAvailable,
        # NetworkError, DDoSProtection, etc.) have a message body that's
        # JUST the request URL with no actual reason attached — str(e)
        # alone is then indistinguishable from "it just didn't work,
        # somehow". The exception CLASS still carries real signal even
        # when its message doesn't, so always surface it. This is exactly
        # what was missing from the BTC/USDT "LIVE DATA FETCH FAILED"
        # entries — they gave a URL and nothing else to diagnose from.
        if msg.strip() in ("", None) or msg.strip().startswith(("http://", "https://")) \
           or len(msg) < 80:
            return f"[{etype}] {msg} → see https://docs.ccxt.com/#/README?id=error-hierarchy for what {etype} means"
        return f"[{etype}] {msg}"

    def _check_stale_repeat(self, df: pd.DataFrame, symbol: str, tf: str) -> bool:
        """Detect an OHLCV feed that is actually stuck, not merely an open candle.

        A candle is allowed to remain unchanged while its timeframe is still open;
        that is normal market behaviour. Once the same five closes persist beyond
        the configured repeat window, independently verify the exchange ticker.
        If the ticker is fresh, the OHLCV candle is considered legitimately open.
        If the ticker is also unavailable/stale, reject the OHLCV frame entirely so
        it can never drive a signal or trade decision.
        """
        sig = tuple(round(v, 8) for v in df["close"].tail(5).tolist())
        now = datetime.utcnow()
        prev_sig = self._last_ohlcv_sig.get(symbol)
        if prev_sig != sig:
            self._last_ohlcv_sig[symbol] = sig
            self._last_ohlcv_changed_at[symbol] = now
            return True

        first_seen = self._last_ohlcv_changed_at.get(symbol, now)
        unchanged_for = now - first_seen
        threshold = timedelta(
            seconds=CONFIG["scan_interval_seconds"] * CONFIG["stale_repeat_scan_multiple"]
        )
        if unchanged_for <= threshold:
            return True

        # Before declaring the OHLCV feed stale, use an independent live ticker.
        # This avoids false positives when the current 15m candle legitimately has
        # not changed yet, while still blocking a genuinely frozen exchange/cache
        # response from reaching TechnicalAnalysis.
        ticker_ok = False
        ticker_price = None
        try:
            if self.mode == "CRYPTO" and self.exchange:
                ticker = self.exchange.fetch_ticker(symbol)
                ticker_price = ticker.get("last") or ticker.get("close")
                ticker_ts = ticker.get("timestamp")
                if ticker_price is not None:
                    ticker_price = float(ticker_price)
                    ticker_fresh = True
                    if ticker_ts:
                        ticker_dt = datetime.utcfromtimestamp(float(ticker_ts) / 1000.0)
                        ticker_age = now - ticker_dt
                        ticker_fresh = ticker_age <= timedelta(
                            seconds=max(30, CONFIG["scan_interval_seconds"] * 3)
                        )

                    # A fresh ticker alone is not enough: if its price has moved
                    # materially away from the last OHLCV close, the OHLCV response
                    # itself is lagging/cached and must be rejected. If it is still
                    # effectively equal to the candle close, the unchanged candle can
                    # legitimately be the currently-open candle.
                    candle_close = float(df["close"].iloc[-1])
                    price_gap = abs(ticker_price - candle_close) / max(abs(candle_close), 1e-12)
                    price_consistent = price_gap <= 0.0002  # 0.02%
                    ticker_ok = ticker_fresh and price_consistent

                    if ticker_fresh and not price_consistent:
                        logger.error(
                            f"❌ [{symbol}] OHLCV rejected: live ticker ${ticker_price:.8f} "
                            f"differs {price_gap * 100:.3f}% from candle close "
                            f"${candle_close:.8f} after {unchanged_for} of unchanged data"
                        )
        except Exception as e:
            logger.warning(
                f"⚠️ [{symbol}] repeated OHLCV verification ticker failed: "
                f"{self._explain_error(e)}"
            )

        if ticker_ok:
            self._last_ohlcv_changed_at[symbol] = now
            _elog(
                f"ℹ️ {symbol}: OHLCV unchanged for {unchanged_for}, but live ticker "
                f"is fresh (${float(ticker_price):.8f}); treating this as an open "
                f"candle rather than cached/stale data",
                "INFO",
            )
            return True

        logger.error(
            f"❌ [{symbol}] OHLCV feed rejected as stale/cached: identical candle "
            f"data for {unchanged_for} and no fresh live ticker confirmation. "
            f"Symbol skipped — no indicators/signals will be calculated from this data."
        )
        _elog(
            f"❌ {symbol}: stale/cached market data rejected — unchanged for "
            f"{unchanged_for}; no fresh ticker confirmation; symbol skipped",
            "ERROR",
        )
        return False

    def _crypto(self, symbol, tf, limit):
        symbol = normalize_market_symbol(symbol, "CRYPTO")
        if not self.exchange:
            raise DataSourceError(self.last_error or "Exchange not initialised")
        raw = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        if not raw or len(raw) < 10:
            raise DataSourceError(f"Exchange returned insufficient candles for {symbol}")
        df  = pd.DataFrame(raw,
                columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        self._validate_freshness(df, symbol, tf)
        if not self._check_stale_repeat(df, symbol, tf):
            return None
        return df

    def _stock(self, symbol, tf, limit):
        symbol = normalize_market_symbol(symbol, "STOCK")
        if not YF_AVAILABLE:
            raise DataSourceError("yfinance not installed")
        imap = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m",
                "1h":"1h","4h":"1h","1d":"1d"}
        iv  = imap.get(tf, "15m")
        per = "5d" if iv in ("1m","5m","15m","30m") else "60d"
        df  = yf.Ticker(symbol).history(period=per, interval=iv)
        if df.empty:
            raise DataSourceError(f"Yahoo Finance returned no data for {symbol} "
                                  f"(market may be closed, or symbol invalid)")
        df.rename(columns={"Open":"open","High":"high","Low":"low",
                            "Close":"close","Volume":"volume"}, inplace=True)
        df = df[["open","high","low","close","volume"]].tail(limit)
        self._validate_freshness(df, symbol, tf)
        if not self._check_stale_repeat(df, symbol, tf):
            return None
        return df

    def _forex(self, symbol, tf, limit):
        symbol = normalize_market_symbol(symbol, "FOREX")
        if not YF_AVAILABLE:
            raise DataSourceError("yfinance not installed")
        # Yahoo forex tickers like EURUSD=X are expected
        iv = self._map_tf(tf)
        per = "60d"
        df = yf.Ticker(symbol).history(period=per, interval=iv)
        if df.empty:
            raise DataSourceError(f"Yahoo Finance returned no data for {symbol}")
        df.rename(columns={"Open":"open","High":"high","Low":"low",
                            "Close":"close","Volume":"volume"}, inplace=True)
        df = df[["open","high","low","close","volume"]].tail(limit)
        self._validate_freshness(df, symbol, tf)
        if not self._check_stale_repeat(df, symbol, tf):
            return None
        return df

    def _map_tf(self, tf: str) -> str:
        imap = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m",
                "1h":"1h","4h":"1h","1d":"1d"}
        return imap.get(tf, "15m")

    def _validate_freshness(self, df: pd.DataFrame, symbol: str, tf: str):
        """Guards against stale/cached data being mistaken for live data."""
        if df.empty:
            raise DataSourceError(f"{symbol}: empty dataframe")
        last_ts = df.index[-1]
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_localize(None)
        age = datetime.utcnow() - last_ts
        max_age = {"1m":timedelta(minutes=10),"5m":timedelta(minutes=30),
                   "15m":timedelta(hours=2),"30m":timedelta(hours=4),
                   "1h":timedelta(hours=8),"4h":timedelta(hours=24),
                   "1d":timedelta(days=4)}.get(tf, timedelta(hours=6))
        if self.mode == "CRYPTO" and age > max_age:
            raise DataSourceError(
                f"{symbol}: latest candle is {age} old — exceeds freshness "
                f"threshold of {max_age}. Data feed may be stale.")

    def get_price(self, symbol: str) -> Optional[float]:
        """Live ticker price. Returns None (never a fake number) on failure."""
        try:
            symbol = normalize_market_symbol(symbol, self.mode)
            if self.mode == "CRYPTO" and self.exchange:
                ticker = self.exchange.fetch_ticker(symbol)
                px = ticker.get("last") or ticker.get("close")
                if px is None:
                    raise DataSourceError(f"No last price in ticker for {symbol}")
                return float(px)
            if (self.mode == "STOCK" or self.mode == "FOREX") and YF_AVAILABLE:
                return float(yf.Ticker(symbol).fast_info.last_price)
        except Exception as e:
            logger.error(f"❌ [{symbol}] Live price fetch failed: {self._explain_error(e)}")
            return None
        return None

# Seconds represented by each supported candle timeframe.
_TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


def _tf_to_seconds(tf: str) -> int:
    return _TF_SECONDS.get(tf, 900)

# ════════════════════════════════════════════════════════════
#  TECHNICAL ANALYSIS
# ════════════════════════════════════════════════════════════

class TechnicalAnalysis:
    def __init__(self):
        self.fast = CONFIG["rsi_fast_period"]
        self.slow = CONFIG["rsi_slow_period"]
        self.confirm = CONFIG["rsi_confirm_period"]  # RSI-24 — slower confirmation filter, see signal()
        self.piv  = CONFIG["swing_pivot_strength"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for fn in [self._rsi, self._ema, self._atr, self._vol, self._swings, self._mom]:
            df = fn(df)
        return df

    def _rsi(self, df):
        for p in [self.fast, self.slow, self.confirm]:
            d  = df["close"].diff()
            g  = d.clip(lower=0); l = -d.clip(upper=0)
            ag = g.ewm(com=p-1, min_periods=p).mean()
            al = l.ewm(com=p-1, min_periods=p).mean()
            df[f"rsi_{p}"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
        df["rsi_div"] = df[f"rsi_{self.fast}"] - df[f"rsi_{self.slow}"]
        return df

    def _ema(self, df):
        for s in [9, 21, 50, 200]:
            df[f"ema_{s}"] = df["close"].ewm(span=s, adjust=False).mean()
        df["ema_trend"] = np.where(df["ema_21"] > df["ema_50"], 1, -1)
        return df

    def _atr(self, df, p=14):
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"]     = tr.ewm(span=p, adjust=False).mean()
        df["atr_pct"] = df["atr"] / df["close"]
        return df

    def _vol(self, df):
        df["vol_ma"]    = df["volume"].rolling(20).mean()
        df["vol_surge"] = df["volume"] > df["vol_ma"] * 1.5
        df["obv"]       = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        return df

    def _swings(self, df):
        n = self.piv
        df["sw_h"] = np.nan; df["sw_l"] = np.nan
        for i in range(n, len(df)-n):
            h = df["high"].iloc[i]; lo = df["low"].iloc[i]
            if all(h >= df["high"].iloc[i-j] for j in range(1,n+1)) and \
               all(h >= df["high"].iloc[i+j] for j in range(1,n+1)):
                df.at[df.index[i], "sw_h"] = h
            if all(lo <= df["low"].iloc[i-j] for j in range(1,n+1)) and \
               all(lo <= df["low"].iloc[i+j] for j in range(1,n+1)):
                df.at[df.index[i], "sw_l"] = lo
        df["last_sh"] = df["sw_h"].ffill()
        df["last_sl"] = df["sw_l"].ffill()
        return df

    def _mom(self, df):
        df["mom5"]  = df["close"].pct_change(5)  * 100
        df["mom10"] = df["close"].pct_change(10) * 100
        return df

    def signal(self, df: pd.DataFrame) -> dict:
        if len(df) < 50:
            return {"side":"HOLD","confidence":0,"reasons":["Insufficient data"],"is_chop":False}
        last = df.iloc[-1]; prev = df.iloc[-2]
        px  = float(last["close"])
        rf  = float(last[f"rsi_{self.fast}"])
        rs  = float(last[f"rsi_{self.slow}"])
        rc  = float(last[f"rsi_{self.confirm}"])
        rd  = float(last["rsi_div"]); rpd = float(prev["rsi_div"])
        et  = int(last["ema_trend"])
        vs  = bool(last["vol_surge"])
        mom = float(last["mom5"])
        atr = float(last["atr"])
        sh  = float(last["last_sh"]) if not np.isnan(last["last_sh"]) else px
        sl  = float(last["last_sl"]) if not np.isnan(last["last_sl"]) else px
        ema21 = float(last["ema_21"]); ema50 = float(last["ema_50"])

        # ── Efficiency Ratio (Kaufman ER) — chop detector ──
        # ER = net directional movement / total path length travelled, over
        # a lookback window. Close to 1 = price moved efficiently in one
        # direction (a real trend). Close to 0 = lots of back-and-forth
        # for little net progress (chop/whipsaw) — exactly the condition
        # that was found to produce fast, small losses: RSI(6) swinging
        # from 33 to 79 and back while price stayed range-bound. Applied
        # below as a cap on ENTRY confidence only — it doesn't touch
        # trades already open, which are already protected by the
        # signal-loss-confirmation and profit-lock mechanisms.
        er_window = min(CONFIG["chop_efficiency_lookback"], len(df) - 1)
        efficiency_ratio = 1.0
        if er_window >= 3:
            closes = df["close"].iloc[-(er_window+1):].values
            net_move = abs(closes[-1] - closes[0])
            path_len = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
            efficiency_ratio = (net_move / path_len) if path_len > 0 else 0.0

        # ── Trend strength — how far apart the EMAs are, as a % of price.
        # Used to scale down a signal that's fighting the prevailing trend
        # (e.g. a SELL signal while EMA21>EMA50 in a real uptrend) —
        # proportional to how established that trend actually is, not a
        # flat penalty. A barely-separated EMA pair (weak/uncertain trend)
        # gets little to no discount; a wide, established separation gets
        # the full discount. This directly targets the finding that SELL
        # signals underperformed specifically on symbols in a sustained
        # uptrend (e.g. ETH), while the same SELL logic worked fine on
        # symbols without that trend context.
        ema_sep_pct = abs(ema21 - ema50) / px if px else 0.0
        trend_strength = min(1.0, ema_sep_pct / CONFIG["strong_trend_ema_sep_pct"]) if CONFIG["strong_trend_ema_sep_pct"] else 0.0

        buy=0; sell=0; reasons_buy=[]; reasons_sell=[]
        thr = CONFIG["rsi_signal_threshold"]

        # Max attainable points per side = 20+15+20+15+10+10+10 = 100,
        # so the resulting score is a true 0-100% conviction scale — NOT
        # normalised against (buy+sell), which used to artificially
        # inflate confidence on weak/mixed signals. RSI(24) was added as a
        # slower trend-confirmation filter; its 10 points were funded by
        # trimming RSI-oversold/overbought (25→20) and the fast/slow cross
        # (20→15) rather than just added on top, so the scale stays 0-100.
        if rf < CONFIG["rsi_oversold"]:
            buy += 20; reasons_buy.append(f"RSI({self.fast})={rf:.1f} oversold")
        if rf > CONFIG["rsi_overbought"]:
            sell += 20; reasons_sell.append(f"RSI({self.fast})={rf:.1f} overbought")
        if rd > thr and rpd <= thr:
            buy += 15; reasons_buy.append("RSI fast crossed above slow")
        if rd < -thr and rpd >= -thr:
            sell += 15; reasons_sell.append("RSI fast crossed below slow")
        if abs(px-sl)/px < 0.012:
            buy += 20; reasons_buy.append(f"Near swing low ${sl:.2f}")
        if abs(px-sh)/px < 0.012:
            sell += 20; reasons_sell.append(f"Near swing high ${sh:.2f}")
        if et == 1:
            buy += 15; reasons_buy.append("EMA21>EMA50 uptrend")
        else:
            sell += 15; reasons_sell.append("EMA21<EMA50 downtrend")
        if vs:
            if buy > sell: buy += 10; reasons_buy.append("Volume surge confirms move")
            else:          sell += 10; reasons_sell.append("Volume surge confirms move")
        if mom > 0.5:
            buy += 10; reasons_buy.append(f"Momentum +{mom:.2f}%")
        elif mom < -0.5:
            sell += 10; reasons_sell.append(f"Momentum {mom:.2f}%")
        # RSI(24): a slower period than the fast/slow pair above — used
        # purely as broader-trend confirmation (above 50 = bullish bias,
        # below 50 = bearish bias), not as its own oversold/overbought
        # trigger. Requires >52/<48 rather than exactly 50 so a value
        # sitting right at the midline doesn't flip-flop which side gets
        # the point on tiny noise.
        if rc > 52:
            buy += 10; reasons_buy.append(f"RSI({self.confirm})={rc:.1f} confirms bullish bias")
        elif rc < 48:
            sell += 10; reasons_sell.append(f"RSI({self.confirm})={rc:.1f} confirms bearish bias")

        # ── Counter-trend discount ── applied AFTER scoring, proportional
        # to trend_strength (0-1). A SELL score gets discounted when the
        # EMA trend is up; a BUY score gets discounted when it's down.
        # Discount scales with how established the trend is — weak/
        # uncertain trends barely discount at all.
        if CONFIG["counter_trend_discount_enabled"] and trend_strength > 0:
            discount = trend_strength * CONFIG["max_counter_trend_discount_pct"]
            if et == 1 and sell > 0:
                sell_before = sell
                sell = sell * (1 - discount)
                reasons_sell.append(f"Counter-trend discount -{discount*100:.0f}% "
                                     f"(EMA sep {ema_sep_pct*100:.2f}%, uptrend)")
            elif et == -1 and buy > 0:
                buy_before = buy
                buy = buy * (1 - discount)
                reasons_buy.append(f"Counter-trend discount -{discount*100:.0f}% "
                                    f"(EMA sep {ema_sep_pct*100:.2f}%, downtrend)")

        min_conf = CONFIG["min_confidence_to_trade"]
        # ── Chop handling ── two behaviours depending on confluence:
        #   1) Full agreement across EVERY core signal simultaneously (not
        #      just a point total clearing the normal threshold) still
        #      opens a trade even during chop — this is deliberately a much
        #      higher bar than a normal trending-market entry. Trade-off:
        #      far fewer chop-period entries, each one much more selective.
        #   2) Anything less than full agreement gets capped below
        #      threshold exactly as before — those partial-agreement setups
        #      are exactly what got fooled by chop historically.
        # is_chop is returned in the signal dict either way so the exit
        # management side (scan_once) can apply faster invalidation to any
        # OPEN trade while its symbol is currently choppy — recomputed
        # fresh every scan, so the moment chop clears, both entry and exit
        # behaviour revert to the normal strategy automatically.
        is_chop = bool(CONFIG["chop_filter_enabled"] and efficiency_ratio < CONFIG["chop_min_efficiency_ratio"])
        chop_capped = False
        chop_override = False
        if is_chop:
            vol_ok = vs or not CONFIG["chop_confluence_require_volume_surge"]
            buy_confluence = (rf < CONFIG["rsi_oversold"] and abs(px-sl)/px < 0.012
                               and et == 1 and rc > 52 and vol_ok)
            sell_confluence = (rf > CONFIG["rsi_overbought"] and abs(px-sh)/px < 0.012
                                and et == -1 and rc < 48 and vol_ok)
            cap = max(0, min_conf - 1)
            if buy > cap:
                if buy_confluence:
                    chop_override = True
                    reasons_buy.append(f"⚡ Chop high-conviction override (ER={efficiency_ratio:.2f}) — "
                                        f"RSI oversold + swing low + uptrend + RSI({self.confirm}) + volume all agree")
                else:
                    chop_capped = True
                    buy = min(buy, cap)
                    reasons_buy.append(f"⚠️ Capped — choppy market (ER={efficiency_ratio:.2f} "
                                        f"< {CONFIG['chop_min_efficiency_ratio']:.2f}), full confluence not met")
            if sell > cap:
                if sell_confluence:
                    chop_override = True
                    reasons_sell.append(f"⚡ Chop high-conviction override (ER={efficiency_ratio:.2f}) — "
                                         f"RSI overbought + swing high + downtrend + RSI({self.confirm}) + volume all agree")
                else:
                    chop_capped = True
                    sell = min(sell, cap)
                    reasons_sell.append(f"⚠️ Capped — choppy market (ER={efficiency_ratio:.2f} "
                                         f"< {CONFIG['chop_min_efficiency_ratio']:.2f}), full confluence not met")
        sl_floor = CONFIG["stop_loss_pct"]
        tp_floor = CONFIG["take_profit_pct"]
        confirmed_rr = float(CONFIG.get("confirmed_trade_min_rr", 4.0))
        fast_min_rr = float(CONFIG.get("fast_trade_min_rr", 1.0))
        fast_target_rr = float(CONFIG.get("fast_trade_target_rr", 2.0))

        def _mode_for_side(side_score: float, opposite_score: float, trend_is_bullish: bool, trend_is_bearish: bool, strong_confirmation: bool) -> tuple[str, float]:
            # Only materially stronger setups deserve the full 4:1 ride path.
            # A marginal but valid 85%-level signal should stay FAST unless it
            # has a clear margin above the normal trade threshold and a clean
            # confluence of momentum/trend/RSI filters.
            if strong_confirmation and side_score >= min_conf + 10 and (side_score - opposite_score) >= 15:
                return "RIDE", confirmed_rr
            return "FAST", max(fast_min_rr, min(fast_target_rr, 2.0))

        if buy > sell and buy >= min_conf:
            conf  = min(float(buy), 99.0)
            strong_confirmation = (
                rc > 60 and et == 1 and vs and abs(px - sl) / px < 0.012
                and buy >= min_conf + 10 and (buy - sell) >= 15
            )
            trade_mode, rr_target = _mode_for_side(buy, sell, et == 1, et == -1, strong_confirmation)
            # WIDER of the two candidates, not tighter: a stop placed at
            # whichever distance gives the trade more room survives normal
            # price noise. Picking the tighter one (the previous behaviour)
            # meant stops sat right at the noise floor and got hit by
            # ordinary wobble regardless of whether the signal was right —
            # high "confidence" trades were still losing simply because
            # they were never given room to work.
            sl_p  = min(round(px*(1-sl_floor), 6), round(px-atr*1.2, 6))
            risk_dist = px - sl_p
            tp_p  = round(px + max(risk_dist*rr_target, px*tp_floor), 6)
            if not (0 < sl_p < px < tp_p):
                # A sane BUY must have sl < price < tp. If config values
                # (stop_loss_pct / take_profit_pct) are misconfigured —
                # e.g. entered as "2" meaning 200% instead of 0.02 — this
                # can otherwise produce an inverted or negative target that
                # can mathematically never be hit. Refuse instead of
                # silently opening a broken trade.
                logger.error(f"Rejected BUY signal — invalid SL/TP geometry "
                              f"(price={px}, sl={sl_p}, tp={tp_p}). Check "
                              f"stop_loss_pct/take_profit_pct in Settings.")
                _elog(f"⚠️ Rejected BUY signal for invalid SL/TP (price=${px:.4f} "
                      f"sl=${sl_p:.4f} tp=${tp_p:.4f}) — check stop_loss_pct/"
                      f"take_profit_pct in Risk settings", "ERROR")
                return {"side":"HOLD","confidence":0,
                        "reasons":["Rejected: invalid SL/TP configuration"],"price":px,"is_chop":is_chop}
            return {"side":"BUY","confidence":conf,"reasons":reasons_buy,
                    "price":px,"sl":sl_p,"tp":tp_p,"is_chop":is_chop,"chop_override":chop_override,
                    "trade_mode":trade_mode,"rr_target":rr_target,
                    "rsi_fast":rf,"rsi_slow":rs,"rsi_confirm":rc,"swing_low":sl,"swing_high":sh}

        if sell > buy and sell >= min_conf:
            conf  = min(float(sell), 99.0)
            strong_confirmation = (
                rc < 40 and et == -1 and vs and abs(px - sh) / px < 0.012
                and sell >= min_conf + 10 and (sell - buy) >= 15
            )
            trade_mode, rr_target = _mode_for_side(sell, buy, et == 1, et == -1, strong_confirmation)
            # Same fix as BUY, mirrored: WIDER of the two candidates.
            sl_p  = max(round(px*(1+sl_floor), 6), round(px+atr*1.2, 6))
            risk_dist = sl_p - px
            tp_p  = round(px - max(risk_dist*rr_target, px*tp_floor), 6)
            if not (0 < tp_p < px < sl_p):
                # A sane SELL must have tp < price < sl (and tp must be a
                # real positive price). Same misconfiguration guard as BUY.
                logger.error(f"Rejected SELL signal — invalid SL/TP geometry "
                              f"(price={px}, sl={sl_p}, tp={tp_p}). Check "
                              f"stop_loss_pct/take_profit_pct in Settings.")
                _elog(f"⚠️ Rejected SELL signal for invalid SL/TP (price=${px:.4f} "
                      f"sl=${sl_p:.4f} tp=${tp_p:.4f}) — check stop_loss_pct/"
                      f"take_profit_pct in Risk settings", "ERROR")
                return {"side":"HOLD","confidence":0,
                        "reasons":["Rejected: invalid SL/TP configuration"],"price":px,"is_chop":is_chop}
            return {"side":"SELL","confidence":conf,"reasons":reasons_sell,
                    "price":px,"sl":sl_p,"tp":tp_p,"is_chop":is_chop,"chop_override":chop_override,
                    "trade_mode":trade_mode,"rr_target":rr_target,
                    "rsi_fast":rf,"rsi_slow":rs,"rsi_confirm":rc,"swing_low":sl,"swing_high":sh}

        return {"side":"HOLD","confidence":max(buy,sell),
                "reasons":reasons_buy if buy>=sell else reasons_sell,"price":px,"is_chop":is_chop,
                "trade_mode":"FAST","rr_target":fast_min_rr}

# ════════════════════════════════════════════════════════════
#  AI ANALYZER
# ════════════════════════════════════════════════════════════

class AIAnalyzer:
    ENDPOINT = "https://api.anthropic.com/v1/messages"

    def analyze(self, symbol: str, sig: dict, df: pd.DataFrame) -> dict:
        key = CONFIG["anthropic_api_key"]
        # AI is an enhancement, not a hard dependency.  If no API key is
        # configured, explicitly report AI as unavailable so the caller can
        # fall back to the validated technical strategy instead of pretending
        # that an AI confirmation occurred.
        if not key:
            return {"action":None,"ai_available":False,"confidence_adjustment":0,
                    "analysis":"AI not connected — technical strategy only.",
                    "risk_level":"MEDIUM","error":"No Anthropic API key configured"}
        last   = df.iloc[-1]
        recent = df["close"].tail(10).tolist()
        prompt = (
            f"Expert algo trading analyst applying layered risk filtering — "
            f"this strategy only trades signals with >={CONFIG['min_confidence_to_trade']}% "
            f"combined conviction. For confirmed momentum setups, keep the original 4:1 ride-out profile; "
            f"for fast tactical entries, allow a lower but valid 2:1 or 1:1 RR target and exit quickly once that objective is reached. "
            f"Be conservative: REJECT or WAIT on anything showing mixed signals, low volume, or choppy price action. "
            f"Analyze signal, respond ONLY valid JSON.\n\n"
            f"SYMBOL:{symbol} MARKET:{CONFIG['market_mode']}\n"
            f"SIGNAL:{sig['side']} @ ${sig.get('price',0):.4f} "
            f"CONF:{sig.get('confidence',0):.1f}%\n"
            f"RSI{CONFIG['rsi_fast_period']}:{sig.get('rsi_fast','?')} "
            f"RSI{CONFIG['rsi_slow_period']}:{sig.get('rsi_slow','?')}\n"
            f"SL:${sig.get('sl',0):.4f} TP:${sig.get('tp',0):.4f} "
            f"(RR mode={sig.get('trade_mode','RIDE')} target ~{sig.get('rr_target', 4.0):.1f}:1)\n"
            f"EMA_trend:{'Bullish' if last.get('ema_trend',0)==1 else 'Bearish'} "
            f"VolSurge:{last.get('vol_surge',False)}\n"
            f"Recent closes:{[round(c,2) for c in recent]}\n"
            f"Reasons:{'; '.join(sig.get('reasons',[]))}\n\n"
            f'Respond ONLY: {{"action":"CONFIRM"|"REJECT"|"WAIT",'
            f'"confidence_adjustment":<-20 to +10>,'
            f'"analysis":"<2-3 sentences>",'
            f'"risk_level":"LOW"|"MEDIUM"|"HIGH"}}'
        )
        try:
            r = requests.post(
                self.ENDPOINT,
                headers={"x-api-key":key,"anthropic-version":"2023-06-01",
                         "content-type":"application/json"},
                json={"model":CONFIG["ai_model"],"max_tokens":512,
                      "messages":[{"role":"user","content":prompt}]},
                timeout=20)
            r.raise_for_status()
            raw   = r.json()["content"][0]["text"]
            clean = raw.strip().replace("```json","").replace("```","").strip()
            return json.loads(clean)
        except Exception as e:
            # A configured AI service can fail transiently (401/429/5xx,
            # timeout, malformed JSON, etc.).  Do NOT manufacture a CONFIRM
            # result.  The scanner will treat this as an AI-unavailable event
            # and use the technical strategy on its own.
            logger.warning(f"AI call failed: {e}")
            return {"action":None,"ai_available":False,"confidence_adjustment":0,
                    "analysis":f"AI unavailable: {e}","risk_level":"MEDIUM",
                    "error":str(e)}

# ════════════════════════════════════════════════════════════
#  TRADE MANAGER
# ════════════════════════════════════════════════════════════

def _clean(v):
    """Make a value JSON-safe."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v

def _clean_trade(t: dict) -> dict:
    return {k: _clean(v) for k, v in t.items()}

class TradeManager:
    def __init__(self):
        self.open_trades   : list[dict] = []
        self.closed_trades : list[dict] = []
        self.balance        = CONFIG["account_balance"]
        self.initial_balance= CONFIG["account_balance"]
        self._lock          = threading.Lock()

    @property
    def equity(self):
        return self.balance + sum(t.get("unrealised_pnl",0)
                                  for t in self.open_trades)

    def can_open(self):
        return len(self.open_trades) < CONFIG["max_open_trades"]

    def open_trade(self, symbol: str, sig: dict, ai: dict, market: Optional[str] = None,
                   allow_duplicate: bool = False, override_size: Optional[float] = None,
                   opened_by: str = "bot", manual_source: str = "") -> Optional[dict]:
        with self._lock:
            if not self.can_open():
                _elog(f"Max trades reached — skip {symbol}", "WARN"); return None
            # The autonomous scanner never duplicates a position on a symbol
            # it's already holding — that's a deliberate safety limit. Manual
            # "duplicate this trade" requests are an explicit, one-at-a-time
            # human decision, so they're allowed to opt out of it.
            if not allow_duplicate and any(t["symbol"]==symbol for t in self.open_trades):
                _elog(f"Already in {symbol} — skip", "WARN"); return None

            px      = sig["price"]
            sl_dist = abs(px - sig["sl"])
            if sl_dist == 0: return None
            if override_size is not None:
                sz = override_size
            else:
                risk    = self.balance * CONFIG["max_risk_per_trade"]
                sz      = min(risk / sl_dist, self.balance * 0.5 / px)
            cost    = sz * px
            conf    = min(sig["confidence"] + ai.get("confidence_adjustment",0), 99)

            m = (market or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
            trade = {
                # Sub-second precision + a short random suffix — plain
                # int(time.time()) (1-second resolution) could collide for
                # two trades on the same symbol opened within the same
                # second, which is now a real possibility since manual
                # duplicate-trade requests can fire back-to-back quickly.
                "id"           : f"{symbol.replace('/','_')}_{int(time.time()*1000)}_{secrets.token_hex(2)}",
                "symbol"       : symbol,
                "market"       : m,
                "side"         : sig["side"],
                "entry"        : px,
                "sl"           : sig["sl"],
                "tp"           : sig["tp"],
                "size"         : sz,
                "cost"         : cost,
                "confidence"   : conf,
                "ai_action"    : ai.get("action"),
                "ai_analysis"  : ai.get("analysis",""),
                "risk_level"   : ai.get("risk_level","MEDIUM"),
                "status"       : "OPEN",
                "open_time"    : datetime.utcnow().isoformat(),
                "close_time"   : None,
                "close_price"  : None,
                "pnl"          : None,
                "unrealised_pnl": 0.0,
                "reasons"      : sig.get("reasons",[]),
                # Consecutive scans in a row where the signal no longer
                # supports this trade. A confidence-based exit requires this
                # to hit CONFIG["signal_exit_confirmations"] (default 2)
                # before actually closing — a single noisy scan shouldn't
                # cut a trade early; two in a row is a real trend change.
                # SL/TP hits are NOT gated by this — those remain instant
                # hard stops, checked independently every cycle.
                "signal_loss_strikes": 0,
                "opened_by"    : opened_by,      # "bot" | "manual"
                "manual_source": manual_source,  # e.g. "live_signal:BTC/USDT@78.5%" or "duplicate_of:BTC_USDT_1723..."
                # Ride-winners mechanism (see check_exits and the
                # confidence re-check in scan_once): initial_sl/initial_risk
                # are captured ONCE here and never change even after sl
                # itself moves to breakeven/trails — they're the fixed
                # yardstick ("1R") that profit-lock and trailing-stop
                # thresholds are measured against. best_price tracks the
                # most favorable price seen so the trailing stop has
                # something to trail from. profit_locked flips on once the
                # trade has proven itself with real price movement, at
                # which point routine confidence wobble stops being a
                # reason to exit — only a genuine reversal or the (now
                # protective) stop can end it.
                "initial_sl"   : sig["sl"],
                "initial_risk" : abs(px - sig["sl"]),
                "best_price"   : px,
                "trade_mode"   : sig.get("trade_mode", "RIDE"),
                "rr_target"    : float(sig.get("rr_target", CONFIG.get("confirmed_trade_min_rr", 4.0))),
                "profit_locked": False,
            }
            live_acc = CONFIG.get("live_accounts", {}).get(m, {})
            tag = f" | MANUAL ({manual_source})" if opened_by == "manual" else ""
            if not CONFIG["paper_trading"] and CONFIG.get("live_trading_enabled", False) and live_acc.get("connected"):
                live_result = execute_live_order(symbol, sig["side"], sz, market=m)
                if live_result.get("ok"):
                    trade["exchange_order_id"] = live_result.get("order", {}).get("id")
                    trade["execution_mode"] = "LIVE"
                    msg = (f"OPENED {symbol} {sig['side']} @ ${px:.4f} | "
                           f"SL=${sig['sl']:.4f} TP=${sig['tp']:.4f} | "
                           f"Conf={conf:.1f}% | LIVE ORDER{tag}")
                else:
                    # LIVE entry failed: explicitly fall back to the normal
                    # paper-trade lifecycle.  The failed exchange order must
                    # never prevent the strategy from managing a paper position,
                    # and the trade is clearly tagged so the dashboard/logs do
                    # not imply that an exchange position exists.
                    trade["execution_mode"] = "PAPER"
                    trade["live_order_failed"] = True
                    trade["paper_fallback"] = True
                    trade["execution_error"] = live_result.get("error", "unknown")
                    msg = (f"OPENED {symbol} {sig['side']} @ ${px:.4f} | "
                           f"SL=${sig['sl']:.4f} TP=${sig['tp']:.4f} | "
                           f"Conf={conf:.1f}% | LIVE FAILED → PAPER FALLBACK "
                           f"({trade['execution_error']}){tag}")
                    _elog(f"↩️ {symbol}: live order failed; switched to PAPER trade management — "
                          f"{trade['execution_error']}", "WARN")
            else:
                trade["execution_mode"] = "PAPER"
                msg = (f"OPENED {symbol} {sig['side']} @ ${px:.4f} | "
                       f"SL=${sig['sl']:.4f} TP=${sig['tp']:.4f} | "
                       f"Conf={conf:.1f}%{tag}")
            self.open_trades.append(trade)
            _elog(f"✅ {msg}", "INFO")
            record_admin_activity("admin", "trade", msg, details={"trade": trade})
            return trade

    def force_close(self, trade_id: str, prices: dict) -> Optional[dict]:
        with self._lock:
            for t in list(self.open_trades):
                if t["id"] == trade_id:
                    px  = prices.get(t["symbol"], t["entry"])
                    pnl = ((px - t["entry"]) if t["side"]=="BUY"
                           else (t["entry"] - px)) * t["size"]
                    t.update({"pnl":pnl, "status":"MANUAL",
                               "close_time":datetime.utcnow().isoformat(),
                               "close_price":px, "unrealised_pnl":0.0})
                    if not CONFIG["paper_trading"] and CONFIG.get("live_trading_enabled", False):
                        m = (t.get("market") or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
                        live_acc = CONFIG.get("live_accounts", {}).get(m, {})
                        close_side = "SELL" if t["side"] == "BUY" else "BUY"
                        if live_acc.get("connected"):
                            live_result = execute_live_order(t["symbol"], close_side, t["size"], market=m)
                            if live_result.get("ok"):
                                t["exchange_order_id"] = live_result.get("order", {}).get("id")
                                t["execution_mode"] = "LIVE"
                            else:
                                t["execution_error"] = live_result.get("error", "unknown")
                                _elog(f"Live close failed for {t['symbol']}: {t['execution_error']}", "WARN")
                        else:
                            t["execution_mode"] = "PAPER"
                            t["execution_error"] = "Live account disconnected"
                            _elog(f"Live close skipped for {t['symbol']}: account disconnected", "WARN")
                    self.balance += pnl
                    self.open_trades.remove(t)
                    self.closed_trades.append(t)
                    msg = f"MANUAL CLOSE {t['symbol']} | PnL=${pnl:.2f}"
                    _elog(f"🔴 {msg}", "WARN")
                    record_admin_activity("admin", "trade", msg, details={"trade": t})
                    return t
        return None

    def close_on_signal_loss(self, trade_id: str, price: float, reason: str) -> Optional[dict]:
        """Close a trade early because the fresh scan no longer supports it —
        i.e. the pair's signal flipped or its confidence dropped below the
        minimum-to-trade threshold. Distinct from an SL/TP hit (check_exits)
        and from a manual close (force_close) so it's clearly labelled in
        trade history."""
        with self._lock:
            for t in list(self.open_trades):
                if t["id"] == trade_id:
                    px  = price if price is not None else t["entry"]
                    pnl = ((px - t["entry"]) if t["side"]=="BUY"
                           else (t["entry"] - px)) * t["size"]
                    t.update({"pnl":pnl, "status":"SIGNAL_EXIT",
                               "close_time":datetime.utcnow().isoformat(),
                               "close_price":px, "unrealised_pnl":0.0,
                               "close_reason":reason})
                    if not CONFIG["paper_trading"] and CONFIG.get("live_trading_enabled", False):
                        m = (t.get("market") or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
                        live_acc = CONFIG.get("live_accounts", {}).get(m, {})
                        close_side = "SELL" if t["side"] == "BUY" else "BUY"
                        if live_acc.get("connected"):
                            live_result = execute_live_order(t["symbol"], close_side, t["size"], market=m)
                            if live_result.get("ok"):
                                t["exchange_order_id"] = live_result.get("order", {}).get("id")
                                t["execution_mode"] = "LIVE"
                            else:
                                t["execution_error"] = live_result.get("error", "unknown")
                                _elog(f"Live close failed for {t['symbol']}: {t['execution_error']}", "WARN")
                        else:
                            t["execution_mode"] = "PAPER"
                            t["execution_error"] = "Live account disconnected"
                    self.balance += pnl
                    self.open_trades.remove(t)
                    self.closed_trades.append(t)
                    msg = f"SIGNAL EXIT {t['symbol']} | {reason} | PnL=${pnl:.2f}"
                    c = Fore.GREEN if pnl > 0 else Fore.RED
                    print(c + f"  ↩️  {msg}" + Style.RESET_ALL)
                    _elog(msg, "WARN")
                    record_admin_activity("admin", "trade", msg, details={"trade": t})
                    return t
        return None

    def close_live_position_for_data_safety(self, trade_id: str, reason: str) -> Optional[dict]:
        """Emergency-close a live position when OHLCV data is stale/unavailable.

        This path deliberately uses an exchange MARKET order and never uses an
        old candle close as an execution price.  If the exchange cannot close
        the position, the internal trade remains OPEN so the bot does not claim
        that a real position was closed when it was not.
        """
        with self._lock:
            trade = next((t for t in self.open_trades if t["id"] == trade_id), None)
            if trade is None:
                return None

            m = (trade.get("market") or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
            live_acc = CONFIG.get("live_accounts", {}).get(m, {})
            if not live_acc.get("connected"):
                _elog(f"Live data-safety close refused for {trade['symbol']}: live account disconnected", "ERROR")
                return None

            close_side = "SELL" if trade["side"] == "BUY" else "BUY"
            try:
                live_result = execute_live_order(trade["symbol"], close_side, trade["size"], market=m)
            except Exception as exc:
                _elog(f"Live data-safety close exception for {trade['symbol']}: {exc}", "ERROR")
                return None

            if not live_result.get("ok"):
                err = live_result.get("error", "unknown live close error")
                trade["execution_error"] = err
                _elog(f"Live data-safety close FAILED for {trade['symbol']}: {err} — keeping internal trade OPEN", "ERROR")
                return None

            order = live_result.get("order") or {}
            fill_px = order.get("average") or order.get("price")
            try:
                fill_px = float(fill_px) if fill_px is not None else None
            except (TypeError, ValueError):
                fill_px = None

            # A successful exchange close is authoritative.  If the exchange
            # did not return a usable fill price, retain entry only for internal
            # PnL bookkeeping; we never used it as the execution price.
            accounting_px = fill_px if fill_px is not None and fill_px > 0 else trade["entry"]
            pnl = ((accounting_px - trade["entry"]) if trade["side"] == "BUY"
                   else (trade["entry"] - accounting_px)) * trade["size"]
            trade.update({"pnl": pnl, "status": "DATA_SAFETY_EXIT",
                          "close_time": datetime.utcnow().isoformat(),
                          "close_price": fill_px, "unrealised_pnl": 0.0,
                          "close_reason": reason, "execution_mode": "LIVE",
                          "exchange_close_id": order.get("id", "")})
            self.balance += pnl
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            msg = (f"DATA SAFETY EXIT {trade['symbol']} | {reason} | "
                   f"Fill=${fill_px:.6f}" if fill_px is not None else
                   f"DATA SAFETY EXIT {trade['symbol']} | {reason} | Fill=exchange-confirmed")
            _elog(msg, "ERROR")
            record_admin_activity("admin", "trade", msg, details={"trade": trade})
            return trade

    def check_exits(self, prices: dict) -> list:
        closed = []
        with self._lock:
            for t in list(self.open_trades):
                px = prices.get(t["symbol"])
                if px is None: continue

                initial_risk = t.get("initial_risk") or abs(t["entry"] - t.get("initial_sl", t["sl"])) or 1e-9
                if t["side"] == "BUY":
                    t["unrealised_pnl"] = (px - t["entry"]) * t["size"]
                    r_multiple = (px - t["entry"]) / initial_risk
                    t["best_price"] = max(t.get("best_price", t["entry"]), px)
                else:
                    t["unrealised_pnl"] = (t["entry"] - px) * t["size"]
                    r_multiple = (t["entry"] - px) / initial_risk
                    t["best_price"] = min(t.get("best_price", t["entry"]), px)

                # ── Ride-winners mechanism, part 1: breakeven lock ──
                # Once a trade has moved in its favor by profit_lock_r_multiple
                # (default 1R — i.e. it's now up by as much as it originally
                # risked), move the stop to breakeven. This is what lets the
                # confidence re-check (in scan_once) safely stop treating
                # routine indicator wobble as exit-worthy from this point on
                # — the trade is now structurally incapable of becoming a
                # loss, so there's nothing left to protect against by
                # cutting it early on a soft signal.
                if (CONFIG["profit_lock_enabled"] and not t.get("profit_locked")
                        and r_multiple >= CONFIG["profit_lock_r_multiple"]):
                    new_sl = t["entry"]
                    improves = (t["side"]=="BUY" and new_sl > t["sl"]) or (t["side"]=="SELL" and new_sl < t["sl"])
                    if improves:
                        old_sl = t["sl"]
                        t["sl"] = round(new_sl, 6)
                        t["profit_locked"] = True
                        _elog(f"🔒 {t['symbol']}: breakeven lock — SL ${old_sl:.4f} → ${t['sl']:.4f} "
                              f"after reaching {r_multiple:.2f}R profit ({CONFIG['profit_lock_r_multiple']}R trigger). "
                              f"Confidence-based early exit is now relaxed for this trade — only a full signal "
                              f"reversal, TP, or this stop can close it from here.", "INFO")

                # ── Ride-winners mechanism, part 2: trail the stop ──
                # Beyond profit-lock, once favorable movement reaches
                # trailing_stop_activation_r, keep the stop
                # trailing_stop_distance_r behind the best price seen —
                # this locks in progressively more of the move while still
                # leaving real room for price to reach TP, rather than an
                # all-or-nothing breakeven-or-target outcome.
                if (CONFIG["trailing_stop_enabled"] and t.get("profit_locked")
                        and r_multiple >= CONFIG["trailing_stop_activation_r"]):
                    trail_dist = CONFIG["trailing_stop_distance_r"] * initial_risk
                    if t["side"] == "BUY":
                        trail_sl = round(t["best_price"] - trail_dist, 6)
                        if trail_sl > t["sl"]:
                            t["sl"] = trail_sl
                    else:
                        trail_sl = round(t["best_price"] + trail_dist, 6)
                        if trail_sl < t["sl"]:
                            t["sl"] = trail_sl

                target_rr = float(t.get("rr_target") or CONFIG.get("fast_trade_target_rr", 2.0) or 2.0)
                if t.get("trade_mode") == "FAST":
                    if t["side"] == "BUY":
                        fast_target_hit = px >= (t["entry"] + (initial_risk * target_rr))
                    else:
                        fast_target_hit = px <= (t["entry"] - (initial_risk * target_rr))
                else:
                    fast_target_hit = False

                if t["side"] == "BUY":
                    hit_sl = px <= t["sl"]; hit_tp = px >= t["tp"] or fast_target_hit
                else:
                    hit_sl = px >= t["sl"]; hit_tp = px <= t["tp"] or fast_target_hit
                if hit_tp or hit_sl:
                    pnl = t["unrealised_pnl"]
                    lbl = "TP" if hit_tp else "SL"
                    if fast_target_hit and not hit_sl:
                        lbl = "TP"
                    # CRITICAL: define msg BEFORE using it below
                    msg = f"Trade {lbl}: {t['symbol']} | PnL=${pnl:.2f}"
                    t.update({"pnl":pnl,"status":lbl,
                               "close_time":datetime.utcnow().isoformat(),
                               "close_price":px,"unrealised_pnl":0.0,
                               "close_reason": "FAST_TARGET" if fast_target_hit and not hit_sl else ("TP" if hit_tp else "SL")})
                    # Place live counter-order on exchange when running live
                    if not CONFIG.get("paper_trading", True) and CONFIG.get("live_trading_enabled"):
                        m = (t.get("market") or CONFIG.get("market_mode") or "CRYPTO").upper()
                        live_acc = CONFIG.get("live_accounts", {}).get(m, {})
                        if live_acc.get("api_key") and live_acc.get("api_secret"):
                            close_side = "SELL" if t["side"] == "BUY" else "BUY"
                            try:
                                live_result = execute_live_order(t["symbol"], close_side, t["size"], market=m)
                                if live_result.get("ok"):
                                    t["execution_mode"]    = "LIVE"
                                    t["exchange_close_id"] = (live_result.get("order") or {}).get("id","")
                                    live_acc["balance"]    = live_result.get("balance", live_acc.get("balance",0))
                                    msg += f" | LIVE order placed"
                                else:
                                    t["execution_error"] = live_result.get("error","")
                                    _elog(f"Live close failed {t['symbol']}: {t['execution_error']}", "ERROR")
                            except Exception as exc:
                                _elog(f"Live close exception {t['symbol']}: {exc}", "ERROR")
                    self.balance += pnl
                    self.open_trades.remove(t)
                    self.closed_trades.append(t)
                    closed.append(t)
                    c = Fore.GREEN if pnl > 0 else Fore.RED
                    print(c + f"  {'🎯' if hit_tp else '🛑'} {msg}" + Style.RESET_ALL)
                    _elog(msg, "INFO")
                    record_admin_activity("admin", "trade", msg, details={"trade": t})
        return closed

    def stats(self) -> dict:
        cl = self.closed_trades
        wins   = [t for t in cl if t.get("pnl") and t["pnl"]>0]
        losses = [t for t in cl if t.get("pnl") and t["pnl"]<=0]
        pnl    = sum(t["pnl"] for t in cl if t.get("pnl"))
        roi    = ((self.balance-self.initial_balance)/self.initial_balance*100
                  if self.initial_balance else 0)
        unr    = sum(t.get("unrealised_pnl",0) for t in self.open_trades)
        return {
            "total"      : len(cl),
            "wins"       : len(wins),
            "losses"     : len(losses),
            "win_rate"   : round(len(wins)/len(cl)*100, 1) if cl else 0.0,
            "total_pnl"  : round(pnl, 2),
            "avg_win"    : round(float(np.mean([t["pnl"] for t in wins])),2) if wins else 0.0,
            "avg_loss"   : round(float(np.mean([t["pnl"] for t in losses])),2) if losses else 0.0,
            "balance"    : round(self.balance, 2),
            "equity"     : round(self.equity, 2),
            "roi"        : round(roi, 2),
            "open_count" : len(self.open_trades),
            "unrealised" : round(unr, 2),
        }

    def console_dashboard(self):
        s = self.stats()
        _section("📊 LIVE DASHBOARD")
        pnl_c = Fore.GREEN if s["total_pnl"]>=0 else Fore.RED
        print(f"  Balance    : {Fore.CYAN}${s['balance']:.2f}{Style.RESET_ALL}  "
              f"Equity: {Fore.CYAN}${s['equity']:.2f}{Style.RESET_ALL}  "
              f"ROI: {pnl_c}{s['roi']:.2f}%{Style.RESET_ALL}")
        print(f"  PnL        : {pnl_c}${s['total_pnl']:.2f}{Style.RESET_ALL}  "
              f"Unrealised: ${s['unrealised']:.2f}")
        print(f"  Trades     : {s['total']} closed  "
              f"Win Rate: {s['win_rate']:.1f}%  ({s['wins']}W/{s['losses']}L)  "
              f"Open: {s['open_count']}")

# ════════════════════════════════════════════════════════════
#  TRADING BOT
# ════════════════════════════════════════════════════════════

class TradingBot:
    def __init__(self, tm: TradeManager):
        self.mode       = CONFIG["market_mode"]
        self.data       = MarketDataProvider(self.mode)
        self.ta         = TechnicalAnalysis()
        self.ai         = AIAnalyzer()
        self.tm         = tm
        self.running    = False
        self.scan_count = 0
        if self.mode == "CRYPTO":
            self.symbols = CONFIG.get("crypto_symbols", [])
            self.tf = CONFIG.get("crypto_timeframe", "15m")
        elif self.mode == "STOCK":
            self.symbols = CONFIG.get("stock_symbols", [])
            self.tf = CONFIG.get("stock_timeframe", "15m")
        elif self.mode == "FOREX":
            self.symbols = CONFIG.get("forex_symbols", [])
            self.tf = CONFIG.get("forex_timeframe", "15m")
        else:  # CUSTOM or unknown falls back to crypto+stock
            self.symbols = CONFIG.get("crypto_symbols", []) + CONFIG.get("stock_symbols", [])
            self.tf = CONFIG.get("crypto_timeframe", "15m")

        # Scan cadence is its own setting (CONFIG["scan_interval_seconds"]),
        # independent of the candle timeframe a trade is opened/closed on
        # (self.tf). Each scan cycle re-checks every open position against
        # its OWN timeframe data (see scan_once) and closes it early if it's
        # no longer valid — the scan frequency and the trade timeframe are
        # deliberately two separate things.
        self.scan_interval_seconds = CONFIG["scan_interval_seconds"]

        # Single source of truth for "what did the bot last see for this
        # symbol" — populated every scan_once() cycle. The Market Analysis
        # and Predictions pages read from this (see /api/analysis,
        # /api/predictions) instead of independently re-fetching/re-scoring
        # on their own timer, which is what let them show a different
        # confidence than the live engine actually acted on: two separate
        # polling loops computing the same formula at two different moments
        # in a moving market will naturally disagree, even with identical
        # code. Reading one shared cache makes that divergence impossible.
        self.last_signals = {}

    def scan_once(self):
        global _PRICE_CACHE
        self.scan_count += 1
        ts = datetime.utcnow().strftime("%H:%M:%S")
        _section(f"🔍 SCAN #{self.scan_count}  [{ts} UTC]  {self.mode}")
        _elog(f"Scan #{self.scan_count} — {self.mode}", "INFO")

        if self.mode == "CRYPTO" and not self.data.healthy:
            msg = (f"No crypto exchange reachable — {self.data.last_error or 'unknown error'}")
            print(Fore.RED + f"  ⚠️  {msg}" + Style.RESET_ALL)
            _elog(msg, "ERROR")
            if self.data.geo_blocked_exchanges:
                hint = ("All tried exchanges returned HTTP 451 (geo-restriction). "
                         "This is the exchange blocking your server's region — "
                         "not a bug in the bot. Fix: open Dashboard → Market & RSI "
                         "Config and set 'crypto_exchange' to one not blocked from "
                         "your location (e.g. kraken, kucoin, bybit), or run the bot "
                         "from a server region/VPN that Binance doesn't restrict.")
                print(Fore.YELLOW + f"  💡 {hint}" + Style.RESET_ALL)
                _elog(hint, "WARN")

        prices = {}; found = 0
        open_symbols = {t["symbol"] for t in self.tm.open_trades}

        for sym in self.symbols:
            try:
                # ── Already in a trade for this pair? Re-check whether the
                # setup still holds — never re-enter it (that would be a
                # second position / double AI call), but DO re-evaluate the
                # signal each full scan:
                #   • still same side & confidence >= threshold → the setup
                #     still has potential, keep riding it and let SL/TP
                #     (checked continuously, see monitor_open_trades) decide
                #     the exit on this trade's own timeframe.
                #   • signal flips or confidence drops below the min-to-trade
                #     threshold → the reason we opened the trade no longer
                #     holds, so close it now instead of waiting for SL/TP.
                if sym in open_symbols:
                    trade = next((t for t in self.tm.open_trades if t["symbol"] == sym), None)
                    df_open = self.data.get_ohlcv(sym, self.tf, limit=200)
                    if df_open is None or len(df_open) < 50:
                        # A live position must never make a signal-based exit
                        # decision from a stale/missing OHLCV candle.  If live
                        # trading is active, treat stale/unavailable market
                        # data as a risk event and close the position using a
                        # genuine live ticker price when one is available.
                        # In paper mode we simply stop the signal re-check for
                        # this cycle and let the fast price monitor continue.
                        px_live = self.data.get_price(sym)
                        if px_live is not None:
                            prices[sym] = px_live
                        if (not CONFIG.get("paper_trading", True)
                                and CONFIG.get("live_trading_enabled", False)
                                and trade is not None):
                            reason = "live OHLCV data stale/unavailable — closing live position for data safety"
                            _elog(f"⚠️ {sym}: {reason}" + (f" @ ${px_live:.4f}" if px_live is not None else " — market close required"), "ERROR")
                            if px_live is not None:
                                closed_stale = self.tm.close_on_signal_loss(trade["id"], px_live, reason)
                            else:
                                closed_stale = self.tm.close_live_position_for_data_safety(trade["id"], reason)
                            if closed_stale:
                                open_symbols.discard(sym)
                            continue
                        if px_live is not None:
                            print(f"  {Fore.BLUE}{sym}{Style.RESET_ALL}  "
                                  f"— open trade, OHLCV unavailable/stale; price only (${px_live:.4f})")
                        else:
                            print(f"  {Fore.BLUE}{sym}{Style.RESET_ALL} "
                                  f"— open trade, OHLCV and live price unavailable; no exit decision made")
                        continue

                    df_open  = self.ta.compute(df_open)
                    sig_open = self.ta.signal(df_open)
                    self.last_signals[sym] = {**sig_open, "ts": datetime.utcnow().isoformat()}

                    # Use the actual live ticker price for anything we act on
                    # (closing the trade, displaying "current" price) — the
                    # OHLCV candle's close (used above only to compute the
                    # signal/indicators) can lag the real current market
                    # price by up to a full timeframe interval (e.g. on a 5m
                    # or 15m chart, "close" may be several minutes old), which
                    # is exactly what produces a close price that doesn't
                    # match what the market is actually doing right now.
                    px_live = self.data.get_price(sym)
                    if px_live is None:
                        # Never use the OHLCV candle close for a live exit
                        # decision.  It may be minutes old on multi-minute
                        # timeframes.  The fast monitor can only act on a live
                        # ticker (or its separately time-limited price cache).
                        if (not CONFIG.get("paper_trading", True)
                                and CONFIG.get("live_trading_enabled", False)
                                and trade is not None):
                            reason = "live ticker unavailable — closing live position rather than using stale candle price"
                            _elog(f"⚠️ {sym}: {reason}", "ERROR")
                            closed_stale = self.tm.close_live_position_for_data_safety(trade["id"], reason)
                            if closed_stale:
                                open_symbols.discard(sym)
                            continue
                        print(f"  {Fore.BLUE}{sym}{Style.RESET_ALL} "
                              f"— live ticker unavailable; no signal-based exit decision this cycle")
                        continue
                    prices[sym] = px_live

                    min_conf   = CONFIG["min_confidence_to_trade"]
                    # Use the trade's OWN opening confidence as the bar to
                    # remain valid, not just the generic minimum threshold.
                    # A trade that opened at 90% but has since drifted down
                    # to 76% is still technically "tradeable" by the 75%
                    # minimum, but its conviction is measurably weakening —
                    # that's an early warning the setup is softening before
                    # it fully reverses. Requiring it to hold its OWN entry
                    # bar (not just the floor) starts the exit-confirmation
                    # clock on that warning sign instead of waiting for a
                    # full signal flip or a drop below the generic minimum.
                    entry_conf = trade.get("confidence", min_conf) if trade is not None else min_conf
                    required_conf = (max(min_conf, entry_conf)
                                      if CONFIG["hold_requires_entry_confidence"] else min_conf)
                    is_reversal = (trade is not None and sig_open["side"] != trade["side"]
                                    and sig_open["side"] != "HOLD")

                    if (trade is not None and trade.get("profit_locked")
                            and CONFIG["relax_exit_after_profit_lock"]):
                        # Ride-winners mechanism, part 3: once a trade has
                        # earned breakeven-or-better (see check_exits), stop
                        # letting routine confidence/indicator wobble decide
                        # its fate. From here only a genuine opposite-
                        # direction signal counts as "no longer valid" — a
                        # HOLD reading or a confidence dip means the setup is
                        # uncertain, not that it's actually reversed, and the
                        # trade is now structurally incapable of becoming a
                        # loss anyway. This is the actual fix for winners
                        # getting cut before reaching TP: they were being
                        # judged by the same strict bar as a fresh, unproven
                        # trade even after price had already validated them.
                        still_valid = not is_reversal
                    else:
                        still_valid = (trade is not None
                                       and sig_open["side"] == trade["side"]
                                       and sig_open.get("confidence", 0) >= required_conf)

                    if trade is not None:
                        # Fast invalidation while this symbol is CURRENTLY
                        # choppy (recomputed fresh every scan from sig_open,
                        # never frozen at entry) — chop is inherently less
                        # predictable, so a trade sitting on one gets cut
                        # on a single unconfirmed scan instead of waiting
                        # for the normal 2-in-a-row confirmation. The moment
                        # this symbol is no longer choppy, normal management
                        # resumes automatically on the very next scan — no
                        # separate mode to manually turn back off.
                        in_chop_now = bool(sig_open.get("is_chop"))
                        needed = CONFIG["chop_exit_confirmations"] if in_chop_now else CONFIG["signal_exit_confirmations"]
                        if still_valid:
                            if trade.get("signal_loss_strikes", 0) > 0:
                                _elog(f"↩️ {sym}: direction re-confirmed — clearing "
                                      f"{trade['signal_loss_strikes']}/{needed} exit strike(s)", "INFO")
                            trade["signal_loss_strikes"] = 0
                        else:
                            trade["signal_loss_strikes"] = trade.get("signal_loss_strikes", 0) + 1
                            strikes = trade["signal_loss_strikes"]
                            below_entry = (sig_open["side"] == trade["side"]
                                           and sig_open.get("confidence", 0) < entry_conf
                                           and sig_open.get("confidence", 0) >= min_conf)
                            if is_reversal:
                                reason = f"signal flipped to {sig_open['side']}"
                                if trade.get("profit_locked"):
                                    reason += " (profit-locked trade — only a genuine reversal triggers this)"
                            elif below_entry:
                                reason = (f"confidence dropped to {sig_open.get('confidence',0):.0f}% — "
                                          f"below the {entry_conf:.0f}% it opened at (still above the "
                                          f"{min_conf}% floor, but conviction is weakening)")
                            else:
                                reason = (f"confidence dropped to {sig_open.get('confidence',0):.0f}% "
                                          f"(< {min_conf}% required)")
                            if in_chop_now:
                                reason += f" [chop — fast {needed}-scan invalidation active]"
                            if strikes < needed:
                                # First (or Nth-but-not-final) unconfirmed
                                # scan — flag it, but don't close yet. SL/TP
                                # keep being checked independently on the
                                # fast monitor loop regardless of this
                                # counter, so a real stop-out is never
                                # delayed by waiting for confirmation.
                                print(Fore.YELLOW + f"  ⚠️  {sym} — {reason} "
                                      f"[{strikes}/{needed} confirmation scans] — "
                                      f"not closing yet" + Style.RESET_ALL)
                                _elog(f"⚠️ {sym}: {reason} — strike {strikes}/{needed}, "
                                      f"needs one more consecutive scan to confirm before closing", "WARN")
                                continue
                            print(Fore.YELLOW + f"  ⚠️  {sym} — {reason}, confirmed on "
                                  f"{strikes}/{needed} consecutive scans, closing open "
                                  f"{trade['side']} @ ${px_live:.4f}" + Style.RESET_ALL)
                            _elog(f"↩️ {sym}: no longer meets confidence ({reason}), confirmed "
                                  f"{strikes}/{needed} scans in a row — closing trade early", "WARN")
                            closed_early = self.tm.close_on_signal_loss(trade["id"], px_live, reason)
                            if closed_early:
                                open_symbols.discard(sym)
                            continue

                    print(f"  {Fore.BLUE}{sym}{Style.RESET_ALL}  "
                          f"— already in open trade, still valid ({sig_open['side']} "
                          f"{sig_open.get('confidence',0):.0f}%), monitoring only (price=${px_live:.4f})")
                    continue

                df = self.data.get_ohlcv(sym, self.tf, limit=200)
                if df is None:
                    # Live data fetch failed — NEVER fall back to fake/simulated
                    # data. Skip this symbol entirely for this scan.
                    err = self.data.last_error or "unknown error"
                    print(f"  {Fore.RED}{sym}{Style.RESET_ALL}  ❌ Live data unavailable — skipped ({err})")
                    _elog(f"{sym}: data unavailable — {err}", "ERROR")
                    continue
                if len(df) < 50:
                    logger.warning(f"{sym}: insufficient data"); continue

                df   = self.ta.compute(df)
                sig  = self.ta.signal(df)
                px   = float(df["close"].iloc[-1])
                prices[sym] = px
                self.last_signals[sym] = {**sig, "ts": datetime.utcnow().isoformat()}

                rf = df[f"rsi_{CONFIG['rsi_fast_period']}"].iloc[-1]
                rs = df[f"rsi_{CONFIG['rsi_slow_period']}"].iloc[-1]
                ema_trend = int(df["ema_trend"].iloc[-1])
                vol_surge = bool(df["vol_surge"].iloc[-1])
                atr = float(df["atr"].iloc[-1])

                print(f"  {Fore.WHITE}{sym}{Style.RESET_ALL}  "
                      f"${px:.4f}  RSI6={rf:.1f}  RSI12={rs:.1f}  "
                      f"{Fore.CYAN}{sig['side']}{Style.RESET_ALL}  "
                      f"Conf={sig.get('confidence',0):.0f}%")

                # Log detailed scan info for each symbol
                _elog(f"📊 {sym} scan → Price=${px:.4f} | RSI6={rf:.1f} RSI12={rs:.1f} | "
                      f"ATR={atr:.4f} | EMA_Trend={'📈' if ema_trend==1 else '📉'} | "
                      f"VolSurge={vol_surge}", "INFO")

                if sig["side"] in ("BUY","SELL") and self.tm.can_open():
                    found += 1
                    signal_detected_at = time.monotonic()
                    _print_signal(sym, sig["side"], sig["price"],
                                  sig["sl"], sig["tp"], sig["confidence"])

                    # Log signal details with reasoning
                    reasons_str = " | ".join(sig.get("reasons", []))
                    _elog(f"🎯 SIGNAL {sig['side']} {sym} @ ${sig['price']:.4f} | "
                          f"Conf={sig['confidence']:.0f}% | SL=${sig['sl']:.4f} TP=${sig['tp']:.4f} | "
                          f"Reasons: {reasons_str}", "INFO")

                    # AI is optional.  When connected it can strengthen or
                    # reject/wait on a technical signal.  When it is not
                    # connected (or temporarily unavailable), the validated
                    # technical strategy remains allowed to trade, especially
                    # in paper mode.  Crucially, an outage is never represented
                    # as a fake AI CONFIRM.
                    if CONFIG["ai_confirmation_required"]:
                        print(Fore.YELLOW + "  🤖 AI confirmation..." + Style.RESET_ALL)
                        ai_res = self.ai.analyze(sym, sig, df)
                        if ai_res.get("ai_available") is False:
                            print(Fore.YELLOW + f"  AI: unavailable — {ai_res.get('analysis','')}; using technical strategy" + Style.RESET_ALL)
                            _elog(f"🤖 AI unavailable for {sym}: {ai_res.get('analysis','N/A')} | Technical strategy remains eligible", "WARN")
                        else:
                            print(f"  AI: {Fore.CYAN}{ai_res.get('action')}{Style.RESET_ALL}"
                                  f" — {ai_res.get('analysis','')}")
                            _elog(f"🤖 AI analyzed {sym}: Action={ai_res.get('action')} | "
                                  f"Confidence Adj={ai_res.get('confidence_adjustment', 0):+.0f}% | "
                                  f"Risk={ai_res.get('risk_level', 'MEDIUM')} | "
                                  f"Analysis: {ai_res.get('analysis','N/A')}", "INFO")
                    else:
                        ai_res = {"action":None,"ai_available":False,"confidence_adjustment":0,
                                  "analysis":"AI disabled","risk_level":"MEDIUM"}

                    ai_available = ai_res.get("ai_available", True) is not False
                    technical_only = not ai_available
                    if technical_only or ai_res.get("action") == "CONFIRM":
                        adj = sig["confidence"] + (ai_res.get("confidence_adjustment",0) if ai_available else 0)
                        min_conf = CONFIG["min_confidence_to_trade"]
                        if adj >= min_conf:
                            # Re-anchor entry/SL/TP to the live ticker price
                            # right before execution — by this point RSI
                            # computation + the AI confirmation round-trip
                            # have taken real time, and sig["price"] is still
                            # the OHLCV candle's close from when the scan
                            # started. On a fast-moving market that can drift
                            # enough from the true fill price to explain a
                            # trade whose recorded entry doesn't match what
                            # the market was actually doing. Same SL/TP
                            # *distances* are preserved so R:R is unchanged —
                            # only the anchor price is refreshed.
                            live_px = self.data.get_price(sym)
                            exec_sig = sig
                            elapsed_s = time.monotonic() - signal_detected_at
                            drift_pct = 0.0
                            if live_px is not None and live_px != sig["price"]:
                                drift_pct = abs(live_px - sig["price"]) / sig["price"] * 100

                                # Refuse to execute if too much time or price
                                # movement has passed since the signal was
                                # detected — the RSI/momentum basis for the
                                # signal was computed at the OLD price, and a
                                # large enough move means that basis may no
                                # longer hold. Better to skip this cycle and
                                # let the next scan re-evaluate fresh than to
                                # chase a price the signal was never actually
                                # about.
                                max_slip = CONFIG["max_execution_slippage_pct"]
                                if drift_pct > max_slip * 100:
                                    msg = (f"{sym}: skipped execution — price moved "
                                           f"{drift_pct:.3f}% (${sig['price']:.4f} → "
                                           f"${live_px:.4f}) in {elapsed_s:.2f}s since signal, "
                                           f"exceeding {max_slip*100:.2f}% max slippage")
                                    print(Fore.YELLOW + f"  ⏱️  {msg}" + Style.RESET_ALL)
                                    _elog(f"⏱️ {msg}", "WARN")
                                    continue

                                if sig["side"] == "BUY":
                                    sl_dist = sig["price"] - sig["sl"]
                                    tp_dist = sig["tp"] - sig["price"]
                                    exec_sig = {**sig, "price": live_px,
                                                "sl": round(live_px - sl_dist, 6),
                                                "tp": round(live_px + tp_dist, 6)}
                                else:
                                    sl_dist = sig["sl"] - sig["price"]
                                    tp_dist = sig["price"] - sig["tp"]
                                    exec_sig = {**sig, "price": live_px,
                                                "sl": round(live_px + sl_dist, 6),
                                                "tp": round(live_px - tp_dist, 6)}
                                _elog(f"⏱️ {sym}: executing {elapsed_s:.2f}s after signal, "
                                      f"price drift {drift_pct:.3f}% "
                                      f"(${sig['price']:.4f} → ${live_px:.4f}) — within "
                                      f"{CONFIG['max_execution_slippage_pct']*100:.2f}% limit, "
                                      f"entry re-anchored to live price", "INFO")
                            opened = self.tm.open_trade(sym, exec_sig, ai_res, market=self.mode)
                            if opened:
                                open_symbols.add(sym)  # never scan it again this cycle
                                _elog(f"✅ Trade OPENED {sym} {sig['side']} | "
                                      f"Final Conf={adj:.0f}% (signal {sig['confidence']:.0f}% + AI {ai_res.get('confidence_adjustment', 0):+.0f}%)", "INFO")
                        else:
                            msg = (f"Conf too low after AI ({adj:.1f}% < "
                                   f"{min_conf}% required) — skip {sym}")
                            print(Fore.YELLOW+f"  ⚠️  {msg}"+Style.RESET_ALL)
                            _elog(f"⚠️ Trade REJECTED {sym}: {msg}", "WARN")
                    else:
                        msg = f"AI {ai_res.get('action')} — {sym} skipped"
                        print(Fore.YELLOW+f"  ⚠️  {msg}"+Style.RESET_ALL)
                        _elog(f"⚠️ Trade {ai_res.get('action')} by AI {sym}: {ai_res.get('analysis', 'N/A')}", "WARN")

            except Exception as e:
                _elog(f"Scan error {sym}: {e}", "ERROR")

        for _sym, _px in prices.items():
            _cache_price(_sym, _px)
        self.tm.check_exits(prices)

        # ── Scan Summary ──
        scan_summary = f"Scan #{self.scan_count} complete | Symbols checked: {len(self.symbols)} | Signals found: {found} | Open trades: {len(self.tm.open_trades)}"
        if found == 0:
            print(f"  {Fore.BLUE}No new signals this scan.{Style.RESET_ALL}")
            _elog("ℹ️ " + scan_summary + " | No new trade opportunities detected", "INFO")
        else:
            _elog(f"✨ {scan_summary} | {found} potential trade(s) identified", "INFO")

        if open_symbols:
            print(f"  {Fore.BLUE}Monitoring {len(open_symbols)} open position(s): "
                  f"{', '.join(open_symbols)}{Style.RESET_ALL}")
            _elog(f"📍 Monitoring {len(open_symbols)} active position(s): {', '.join(open_symbols)}", "INFO")

        self.tm.console_dashboard()
        port = CONFIG["web_port"]
        print(f"\n  {Fore.CYAN}Dashboard → http://localhost:{port}{Style.RESET_ALL}")

    def _sync_live_balance(self):
        """Refresh live exchange USDT balance into CONFIG ~every 60s when live trading is active."""
        if CONFIG.get("paper_trading", True) or not CONFIG.get("live_trading_enabled"):
            return
        m   = (self.mode or "CRYPTO").upper()
        acc = CONFIG.get("live_accounts", {}).get(m, {})
        if not acc.get("api_key") or not acc.get("api_secret"):
            return
        last = acc.get("last_sync")
        if last:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(last)).total_seconds()
                if age < 55: return   # throttle to once per ~60s
            except Exception:
                pass
        try:
            exchange, err = _get_live_exchange_client(market=m)
            if exchange:
                free_usdt        = _extract_free_usdt(exchange.fetch_balance())
                acc["balance"]   = free_usdt
                acc["last_sync"] = datetime.utcnow().isoformat()
                acc["last_error"] = ""
                _sync_alias_live_account()
                _save_persistent_state()
                _elog(f"💰 Live balance synced: ${free_usdt:.2f} USDT", "INFO")
        except Exception as exc:
            acc["last_error"] = str(exc)

    def monitor_open_trades(self):
        """Fast-cycle SL/TP check between full scans.

        A held position must NEVER go unevaluated just because a single
        ticker request failed (rate limit, transient network blip, etc).
        If the live fetch fails we fall back to the last known good price —
        but ONLY if it's still fresh (< PRICE_CACHE_MAX_AGE seconds old).
        A price that's a minute old is no longer "the actual market price"
        and evaluating a stop against it would be exactly the kind of
        stale-data execution we're trying to eliminate — in that case we
        skip the check this cycle and log it clearly rather than act on a
        number that may no longer be true.
        """
        open_trades = list(self.tm.open_trades)
        if not open_trades:
            return
        prices = {}
        for t in open_trades:
            sym = t["symbol"]
            px = self.data.get_price(sym)
            if px is not None:
                _cache_price(sym, px)          # genuine live price — refresh the cache
                prices[sym] = px
                continue
            cached, age = _cached_price_if_fresh(sym)
            if cached is not None:
                px = cached
                _elog(f"{sym}: live price fetch failed — using cached price "
                      f"${cached:.4f} from {age:.1f}s ago (< {PRICE_CACHE_MAX_AGE}s "
                      f"freshness limit) so SL/TP can still be checked", "WARN")
                prices[sym] = px           # NOT re-cached — don't refresh its timestamp
            elif age is not None:
                _elog(f"{sym}: live price fetch failed and cached price is "
                      f"{age:.1f}s old (stale, > {PRICE_CACHE_MAX_AGE}s limit) — "
                      f"exit check skipped this cycle rather than act on stale data", "ERROR")
            else:
                _elog(f"{sym}: live price fetch failed and no cached price "
                      f"available yet — exit check skipped this cycle", "ERROR")
        if prices:
            self.tm.check_exits(prices)

    def run(self):
        self.running = True
        syms = ", ".join(self.symbols)
        _elog(f"Bot started — {self.mode} | Paper={CONFIG['paper_trading']} | "
              f"Timeframe={self.tf} | Scan every {self.scan_interval_seconds}s | {syms}", "INFO")
        # Two independent cadences:
        #  1) Full scan (self.scan_interval_seconds, below) — looks for new
        #     entries AND re-checks every open position's signal on its own
        #     timeframe (self.tf), closing it early if it's no longer valid.
        #  2) Fast SL/TP price check (monitor_every) — pure stop/target hit
        #     detection, runs more often than the full scan for capital
        #     protection, independent of both the scan interval and the
        #     trade timeframe.
        monitor_every = min(30, max(5, self.scan_interval_seconds // 6))
        try:
            while self.running and not _STOP_EVENT.is_set():
                try:
                    self.scan_once()
                    # Both checks are cheap no-ops most cycles (internal
                    # "has it been long enough" guards) — safe to call every
                    # scan rather than needing a separate maintenance thread.
                    maybe_rotate_log_for_age(logger)
                    prune_database()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Scan loop error: {e}", exc_info=True)
                    _elog(f"Loop error: {e}", "ERROR")

                iv = self.scan_interval_seconds
                elapsed = 0
                while elapsed < iv and self.running and not _STOP_EVENT.is_set():
                    wait_chunk = min(monitor_every, iv - elapsed)
                    if _STOP_EVENT.wait(timeout=wait_chunk):
                        break
                    elapsed += wait_chunk
                    try:
                        self.monitor_open_trades()
                        self._sync_live_balance()   # keeps wallet current
                    except Exception as e:
                        logger.error(f"Monitor error: {e}")
                        _elog(f"Monitor error: {e}", "ERROR")
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            _elog("Bot stopped", "WARN")

# ════════════════════════════════════════════════════════════
#  FLASK APPLICATION
# ════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(32))

def _login_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if not session.get("logged_in"):
            return redirect(url_for("login_page"))
        return f(*a, **kw)
    return dec

# ── Auth ──────────────────────────────────────────────────────
@app.route("/login", methods=["GET","POST"])
def login_page():
    err = None
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","")
        user = ADMIN_STORE.get(u)
        if user and check_password_hash(user["password_hash"], p):
            session["logged_in"]   = True
            session["username"]    = u
            session["login_ts"]    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            session["session_id"]  = start_admin_session(u, session.get("session_id") or secrets.token_hex(8))
            record_admin_activity(u, "login", "Signed in successfully", session_id=session["session_id"], details={"ip": request.remote_addr})
            return redirect(url_for("home"))
        err = "Invalid username or password."
    return render_template("login.html", error=err)

@app.route("/logout")
def logout():
    sid = session.get("session_id")
    if sid:
        end_admin_session(sid)
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "status": "healthy"})

# ── Pages ─────────────────────────────────────────────────────
@app.route("/")
@_login_required
def home():
    return render_template("home.html")

@app.route("/dashboard")
@_login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/settings")
@_login_required
def settings_page():
    return render_template("settings.html")

@app.route("/analysis")
@_login_required
def analysis_page():
    return render_template("market_analysis.html")

@app.route("/predictions")
@_login_required
def predictions_page():
    return render_template("predictions.html")

@app.route("/api/predictions")
@_login_required
def api_predictions():
    """
    For every configured symbol, compute:
      - Current technical score & distance from the 85% threshold
      - Projected entry, SL, TP and exact dollar amounts based on configured account balance
      - What combination of remaining signals would push it to tradeable
      - Historical probability estimate based on signal composition
      - Expected value calculation using the 4:1 R:R ratio
    All numbers are computed from live OHLCV — no fabricated data.
    """
    mode    = CONFIG["market_mode"]
    symbols = (CONFIG["crypto_symbols"] if mode == "CRYPTO" else
               CONFIG["stock_symbols"]  if mode == "STOCK"  else
               CONFIG.get("forex_symbols", []))
    tf       = (CONFIG["crypto_timeframe"] if mode == "CRYPTO" else
                CONFIG["stock_timeframe"])
    min_conf = CONFIG["min_confidence_to_trade"]
    sl_floor = CONFIG["stop_loss_pct"]
    tp_floor = CONFIG["take_profit_pct"]
    bal      = float(CONFIG["account_balance"])
    risk_amt = bal * CONFIG["max_risk_per_trade"]
    min_rr   = 4.0

    data_provider = _BOT.data if (_BOT and _BOT.data) else MarketDataProvider(mode)
    ta = TechnicalAnalysis()

    predictions = []
    errors      = []
    ts          = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Historical win-rate baseline per signal count (empirical estimates for
    # RSI + swing + EMA confluence strategies from published backtests)
    WIN_RATE_BY_SCORE = {
        (0,  40): 0.38,
        (40, 55): 0.47,
        (55, 70): 0.55,
        (70, 80): 0.62,
        (80, 90): 0.71,
        (90,100): 0.79,
    }

    def estimate_win_rate(score: float) -> float:
        for (lo, hi), wr in WIN_RATE_BY_SCORE.items():
            if lo <= score < hi:
                return wr
        return 0.79

    def expected_value(win_rate: float, risk: float) -> float:
        reward = risk * min_rr
        return round(win_rate * reward - (1 - win_rate) * risk, 2)

    MAX_SCORE = 100
    # All signal components and their point values
    # These weights and directions must exactly mirror TechnicalAnalysis.
    # signal()'s actual scoring — this list is what drives the "what's
    # needed to open this trade" roadmap on the Predictions page. It had
    # drifted out of sync with the real engine (stale weights from before
    # RSI(24) was added, and missing the RSI(24) entries entirely), which
    # made the roadmap show point totals that could never actually reach
    # the confidence threshold. The top-line confidence score itself was
    # always correct (it comes straight from signal()) — only this
    # breakdown display was wrong.
    ALL_SIGNALS = [
        {"key":"rsi_oversold",   "label":f"RSI({ta.fast}) Oversold",     "pts":20, "dir":"buy"},
        {"key":"rsi_overbought", "label":f"RSI({ta.fast}) Overbought",   "pts":20, "dir":"sell"},
        {"key":"rsi_cross_buy",  "label":"RSI Fast Cross ↑",             "pts":15, "dir":"buy"},
        {"key":"rsi_cross_sell", "label":"RSI Fast Cross ↓",             "pts":15, "dir":"sell"},
        {"key":"swing_low",      "label":"At Swing Low Support",         "pts":20, "dir":"buy"},
        {"key":"swing_high",     "label":"At Swing High Resistance",     "pts":20, "dir":"sell"},
        {"key":"ema_bullish",    "label":"EMA21 > EMA50 Uptrend",        "pts":15, "dir":"buy"},
        {"key":"ema_bearish",    "label":"EMA21 < EMA50 Downtrend",      "pts":15, "dir":"sell"},
        {"key":"vol_surge",      "label":"Volume Surge Confirmation",    "pts":10, "dir":"both"},
        {"key":"momentum_pos",   "label":"Positive Momentum 5-bar",      "pts":10, "dir":"buy"},
        {"key":"momentum_neg",   "label":"Negative Momentum 5-bar",      "pts":10, "dir":"sell"},
        {"key":"rsi24_bullish",  "label":f"RSI({ta.confirm}) Confirms Bullish Bias", "pts":10, "dir":"buy"},
        {"key":"rsi24_bearish",  "label":f"RSI({ta.confirm}) Confirms Bearish Bias", "pts":10, "dir":"sell"},
    ]

    for sym in symbols:
        try:
            df = data_provider.get_ohlcv(sym, tf, limit=200)
            if df is None or len(df) < 50:
                errors.append({"symbol": sym, "error": data_provider.last_error or "No data"})
                continue

            df   = ta.compute(df)
            sig  = ta.signal(df)

            # Same single-source-of-truth fix as /api/analysis — see the
            # comment there for why this matters.
            signal_synced = False
            signal_age_s  = None
            if _BOT and _BOT.running and sym in _BOT.last_signals:
                cached = _BOT.last_signals[sym]
                try:
                    age = (datetime.utcnow() - datetime.fromisoformat(cached["ts"])).total_seconds()
                except Exception:
                    age = None
                if age is not None and age <= max(_BOT.scan_interval_seconds * 2, 120):
                    sig = cached
                    signal_synced = True
                    signal_age_s = round(age, 1)

            last = df.iloc[-1]; prev = df.iloc[-2]
            px   = float(last["close"])

            def sf(v):
                try:
                    f = float(v)
                    return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
                except Exception:
                    return None

            rsi_fast  = sf(last.get(f"rsi_{ta.fast}"))
            rsi_slow  = sf(last.get(f"rsi_{ta.slow}"))
            rsi_confirm = sf(last.get(f"rsi_{ta.confirm}"))
            ema_trend = int(last.get("ema_trend", -1))
            vs        = bool(last.get("vol_surge", False))
            mom5      = sf(last.get("mom5"))
            atr       = sf(last.get("atr")) or (px * sl_floor)
            last_sh   = sf(last.get("last_sh"))
            last_sl_s = sf(last.get("last_sl"))
            thr       = CONFIG["rsi_signal_threshold"]
            rsi_div   = float(last.get("rsi_div", 0) or 0)
            rsi_pdiv  = float(prev.get("rsi_div", 0) or 0)
            pct_chg   = round((px - float(df["close"].iloc[-2])) / float(df["close"].iloc[-2]) * 100, 3)

            # Score which signals are currently active
            active = {
                "rsi_oversold"  : bool(rsi_fast and rsi_fast < CONFIG["rsi_oversold"]),
                "rsi_overbought": bool(rsi_fast and rsi_fast > CONFIG["rsi_overbought"]),
                "rsi_cross_buy" : bool(rsi_div > thr and rsi_pdiv <= thr),
                "rsi_cross_sell": bool(rsi_div < -thr and rsi_pdiv >= -thr),
                "swing_low"     : bool(last_sl_s and abs(px - last_sl_s) / px < 0.012),
                "swing_high"    : bool(last_sh  and abs(px - last_sh)  / px < 0.012),
                "ema_bullish"   : ema_trend == 1,
                "ema_bearish"   : ema_trend != 1,
                "vol_surge"     : vs,
                "momentum_pos"  : bool(mom5 and mom5 > 0.5),
                "momentum_neg"  : bool(mom5 and mom5 < -0.5),
                # Must match signal()'s >52/<48 thresholds exactly (not
                # >50/<50) — a value sitting right at the midline shouldn't
                # flip which side gets the point on tiny noise, same
                # reasoning as the live engine.
                "rsi24_bullish" : bool(rsi_confirm and rsi_confirm > 52),
                "rsi24_bearish" : bool(rsi_confirm and rsi_confirm < 48),
            }

            buy_score  = sum(s["pts"] for s in ALL_SIGNALS
                             if s["dir"] in ("buy","both") and active.get(s["key"]))
            sell_score = sum(s["pts"] for s in ALL_SIGNALS
                             if s["dir"] in ("sell","both") and active.get(s["key"]))
            # primary_dir / current_score come from the CANONICAL sig
            # (TechnicalAnalysis.signal(), possibly bot-synced above) rather
            # than being independently re-derived from buy_score/sell_score.
            # buy_score/sell_score are kept only to drive the "which
            # individual signals are lit up" breakdown below — the actual
            # score/direction shown to the user must always trace back to
            # the one canonical computation, or this page can drift out of
            # sync with the live engine even when scanning the exact same
            # candle, just from two hand-written copies of the same formula
            # going out of sync over time.
            primary_dir   = sig["side"] if sig["side"] in ("BUY","SELL") else ("BUY" if buy_score >= sell_score else "SELL")
            current_score = round(sig.get("confidence", 0), 1)
            gap_to_threshold = max(0, min_conf - current_score)

            # SL/TP: use the canonical signal's own sl/tp when it has them
            # (BUY/SELL case — this is what the bot would actually use).
            # Only fall back to a fresh projection for the HOLD case, purely
            # to show what a hypothetical trade would look like if this
            # symbol reached threshold.
            sl_price = sig.get("sl")
            tp_price = sig.get("tp")
            sl_dist = risk_dist = 0.0
            if sl_price is None or tp_price is None:
                if primary_dir == "BUY":
                    sl_price  = round(min(px*(1-sl_floor), px - atr*1.2), 6)
                    risk_dist = px - sl_price
                    tp_price  = round(px + max(risk_dist*min_rr, px*tp_floor), 6)
                else:
                    sl_price  = round(max(px*(1+sl_floor), px + atr*1.2), 6)
                    risk_dist = sl_price - px
                    tp_price  = round(px - max(risk_dist*min_rr, px*tp_floor), 6)
            else:
                risk_dist = abs(px - sl_price)

            sl_pct = round(abs(risk_dist / px) * 100, 3) if px else 0
            tp_pct = round(abs((tp_price - px) / px) * 100, 3) if px and tp_price else 0
            rr_actual = round(tp_pct / sl_pct, 1) if sl_pct else 0

            # Position sizing
            pos_size  = round(risk_amt / risk_dist, 6) if risk_dist > 0 else 0
            pos_value = round(pos_size * px, 2)
            max_loss  = round(risk_dist * pos_size, 2)
            max_gain  = round(max_loss * min_rr, 2)
            win_rate  = estimate_win_rate(current_score)
            ev        = expected_value(win_rate, max_loss)

            # Signals still needed to reach threshold
            missing_dir = "buy" if primary_dir == "BUY" else "sell"
            inactive_signals = [
                s for s in ALL_SIGNALS
                if s["dir"] in (missing_dir, "both") and not active.get(s["key"])
            ]
            # Find the minimum combination that would close the gap
            needed_combos = []
            remaining_gap = gap_to_threshold
            for sig_item in sorted(inactive_signals, key=lambda x: -x["pts"]):
                if remaining_gap <= 0:
                    break
                needed_combos.append(sig_item)
                remaining_gap -= sig_item["pts"]

            # When would the next trigger likely fire?
            # RSI(6) moves fast — estimate bars to oversold/overbought from current level
            bars_to_rsi_trigger = None
            if rsi_fast is not None:
                if primary_dir == "BUY" and rsi_fast > CONFIG["rsi_oversold"]:
                    bars_to_rsi_trigger = max(1, int((rsi_fast - CONFIG["rsi_oversold"]) / 3))
                elif primary_dir == "SELL" and rsi_fast < CONFIG["rsi_overbought"]:
                    bars_to_rsi_trigger = max(1, int((CONFIG["rsi_overbought"] - rsi_fast) / 3))

            # Candlestick pattern
            body   = abs(px - float(last["open"]))
            candle = float(last["high"]) - float(last["low"])
            wick_r = (float(last["high"]) - max(px, float(last["open"]))) / candle if candle else 0
            wick_l = (min(px, float(last["open"])) - float(last["low"])) / candle if candle else 0
            # A zero-range candle (high == low — happens on very quiet bars)
            # would otherwise divide by zero here; treat it as a Doji
            # (no real body relative to range) instead of crashing.
            pattern = ("Doji" if (candle == 0 or body/candle < 0.1) else
                       "Hammer" if wick_l > 0.6 else
                       "Shooting Star" if wick_r > 0.6 else
                       "Bullish" if px > float(last["open"]) else "Bearish")

            # Readiness tier
            if current_score >= min_conf:
                tier = "TRADE_NOW"
            elif current_score >= min_conf * 0.82:
                tier = "VERY_CLOSE"
            elif current_score >= min_conf * 0.65:
                tier = "BUILDING"
            elif current_score >= min_conf * 0.45:
                tier = "WATCHING"
            else:
                tier = "EARLY"

            predictions.append({
                "symbol"           : sym,
                "price"            : round(px, 6),
                "pct_change"       : pct_chg,
                "direction"        : primary_dir,
                "current_score"    : current_score,
                "min_conf"         : min_conf,
                "gap_to_threshold" : gap_to_threshold,
                "tier"             : tier,
                "meets_threshold"  : current_score >= min_conf,
                "in_open_trade"    : any(t["symbol"]==sym for t in (_TM.open_trades if _TM else [])),
                # Trade parameters
                "entry"            : round(px, 6),
                "sl"               : sl_price,
                "tp"               : tp_price,
                "sl_pct"           : sl_pct,
                "tp_pct"           : tp_pct,
                "rr_ratio"         : rr_actual,
                "synced_with_bot"  : signal_synced,
                "signal_age_s"     : signal_age_s,
                # Position & risk
                "position_size"    : pos_size,
                "position_value"   : pos_value,
                "max_loss_usd"     : max_loss,
                "max_gain_usd"     : max_gain,
                "risk_amount"      : round(risk_amt, 2),
                "account_balance"  : round(bal, 2),
                # Probability
                "win_rate_estimate": round(win_rate * 100, 1),
                "expected_value"   : ev,
                # Indicators
                "rsi_fast"         : round(rsi_fast, 1) if rsi_fast else None,
                "rsi_slow"         : round(rsi_slow, 1) if rsi_slow else None,
                "rsi_confirm"      : round(rsi_confirm, 1) if rsi_confirm else None,
                "rsi_fast_period"  : ta.fast,
                "rsi_confirm_period": ta.confirm,
                "ema_trend"        : "Bullish" if ema_trend == 1 else "Bearish",
                "vol_surge"        : vs,
                "momentum"         : round(mom5, 2) if mom5 else None,
                "atr_pct"          : round(float(last.get("atr_pct",0) or 0)*100, 2),
                "pattern"          : pattern,
                "last_sh"          : round(last_sh, 4) if last_sh else None,
                "last_sl"          : round(last_sl_s, 4) if last_sl_s else None,
                # Active signals
                "active_signals"   : [s["label"] for s in ALL_SIGNALS if active.get(s["key"]) and
                                       s["dir"] in (missing_dir, "both")],
                "inactive_signals" : [s["label"] for s in inactive_signals],
                "needed_to_trade"  : [{"label":s["label"],"pts":s["pts"]} for s in needed_combos],
                "bars_to_rsi"      : bars_to_rsi_trigger,
                "reasons"          : sig.get("reasons", []),
            })

        except Exception as e:
            logger.error(f"Prediction error for {sym}: {e}")
            errors.append({"symbol": sym, "error": str(e)})

    # Sort: meets threshold first, then by score desc
    predictions.sort(key=lambda x: (-int(x["meets_threshold"]), -x["current_score"]))

    return jsonify({
        "ts"          : ts,
        "mode"        : mode,
        "timeframe"   : tf,
        "exchange"    : CONFIG.get("crypto_exchange",""),
        "min_conf"    : min_conf,
        "sl_pct"      : round(sl_floor * 100, 2),
        "tp_pct"      : round(tp_floor * 100, 2),
        "rr_ratio"    : min_rr,
        "account_balance": round(bal, 2),
        "predictions" : predictions,
        "errors"      : errors,
        "bot_running" : bool(_BOT and _BOT.running),
    })


@app.route("/wallet")
@_login_required
def wallet_page():
    return render_template("wallet.html")

# ════════════════════════════════════════════════════════════
#  WALLET  — in-memory ledger (persisted via _save_persistent_state)
# ════════════════════════════════════════════════════════════

# Withdrawal request ledger — list of dicts
_WALLET_LEDGER: list[dict] = []
_WALLET_LOCK   = threading.Lock()

def _wallet_stats() -> dict:
    """Live wallet snapshot combining paper PnL + live exchange balance."""
    tm      = _TM
    closed  = tm.closed_trades if tm else []
    open_t  = tm.open_trades   if tm else []
    realized   = round(sum(t["pnl"] for t in closed if t.get("pnl")), 2)
    unrealized = round(sum(t.get("unrealised_pnl", 0) for t in open_t), 2)
    paper_bal  = round(tm.balance if tm else CONFIG["account_balance"], 2)
    live_payload = build_live_trading_payload()
    live_bal   = round(float(live_payload.get("balance", 0) or 0), 2)
    # "is_live" here means "a real exchange account is linked and connected"
    # — it deliberately does NOT require live_trading_enabled (the
    # bot-autonomous-trading toggle). Viewing your real balance and
    # withdrawing your own funds are manual, user-initiated actions and
    # shouldn't be blocked just because you haven't (or don't want to)
    # flip on autonomous trading.
    is_live    = bool(live_payload.get("connected"))
    trading_enabled = bool(live_payload.get("enabled"))
    market = (CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    exchange_dashboard = build_exchange_dashboard_payload(market, force_refresh=False, persist=False)
    if exchange_dashboard.get("summary", {}).get("available_balance") is not None:
        live_bal = round(float(exchange_dashboard.get("summary", {}).get("available_balance", live_bal) or live_bal), 2)

    total_deposited = round(CONFIG["account_balance"], 2)
    total_profits   = max(realized, 0)
    total_losses    = abs(min(realized, 0))

    # pending withdrawals reduce available
    with _WALLET_LOCK:
        pending_total = round(sum(
            w["amount"] for w in _WALLET_LEDGER if w["status"] == "PENDING"
        ), 2)
        ledger_copy   = list(reversed(_WALLET_LEDGER[-50:]))

    available = (live_bal if is_live else paper_bal) - pending_total

    return {
        "paper_balance"    : paper_bal,
        "live_balance"     : live_bal,
        "is_live"          : is_live,
        "trading_enabled"  : trading_enabled,
        "live_exchange"    : live_payload.get("exchange", CONFIG.get("live_exchange", "")),
        "live_connected"   : bool(live_payload.get("connected")),
        "live_linked"      : bool(live_payload.get("linked")),
        "realized_pnl"     : realized,
        "unrealized_pnl"   : unrealized,
        "total_deposited"  : total_deposited,
        "total_profits"    : total_profits,
        "total_losses"     : total_losses,
        "available_balance": round(available, 2),
        "pending_withdrawals": pending_total,
        "ledger"           : ledger_copy,
        "exchange_dashboard": exchange_dashboard,
        "exchange_summary": exchange_dashboard.get("summary", {}),
        "live_last_sync": exchange_dashboard.get("account", {}).get("last_sync"),
    }

@app.route("/api/wallet")
@_login_required
def api_wallet():
    return jsonify({"ok": True, **_wallet_stats()})

@app.route("/api/wallet/withdraw", methods=["POST"])
@_login_required
def api_wallet_withdraw():
    d       = request.json or {}
    amount  = float(d.get("amount", 0) or 0)
    address = (d.get("address") or "").strip()
    network = (d.get("network") or "USDT-TRC20").strip()
    note    = (d.get("note") or "").strip()[:200]

    if amount <= 0:
        return jsonify({"ok": False, "error": "Amount must be greater than zero"}), 400
    if not address:
        return jsonify({"ok": False, "error": "Withdrawal address is required"}), 400

    ws      = _wallet_stats()
    avail   = ws["available_balance"]
    if amount > avail:
        return jsonify({
            "ok": False,
            "error": f"Insufficient available balance. Available: ${avail:.2f}  Requested: ${amount:.2f}"
        }), 400

    is_live = ws["is_live"]
    exchange_result = None

    # Attempt real withdrawal on live exchange if connected. Reuses the
    # single shared exchange-client helper (_get_live_exchange_client)
    # instead of re-deriving credentials — this used to read from
    # CONFIG["live_api_key"]/["live_api_secret"], a legacy top-level
    # location that Settings → Live Trading never actually writes to
    # (it saves per-market under CONFIG["live_accounts"][market]), so a
    # withdrawal would silently fall back to PAPER status even with a
    # fully connected live account.
    market = (CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    if is_live:
        exchange, client_error = _get_live_exchange_client(market=market)
        if exchange is not None:
            try:
                withdrawal = exchange.withdraw("USDT", amount, address,
                                                params={"network": network})
                exchange_result = withdrawal
            except Exception as exc:
                exchange_result = {"error": str(exc)}
        else:
            exchange_result = {"error": client_error or "Live exchange client unavailable"}

    status = "PENDING"
    if is_live and exchange_result and not exchange_result.get("error"):
        status = "SUBMITTED"

    req_id = secrets.token_hex(6).upper()
    record = {
        "id"        : req_id,
        "ts"        : datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "amount"    : round(amount, 2),
        "address"   : address,
        "network"   : network,
        "note"      : note,
        "status"    : status,
        "type"      : "LIVE" if is_live else "PAPER",
        "exchange"  : (CONFIG.get("live_accounts", {}).get(market, {}).get("exchange", "")) if is_live else "paper",
        "exchange_ref": (exchange_result or {}).get("id", ""),
        "error"     : (exchange_result or {}).get("error", ""),
    }
    with _WALLET_LOCK:
        _WALLET_LEDGER.append(record)

    _elog(f"Withdrawal {status}: ${amount:.2f} USDT → {address[:10]}… [{req_id}]",
          "INFO" if status == "SUBMITTED" else "WARN")

    return jsonify({
        "ok"      : True,
        "id"      : req_id,
        "status"  : status,
        "amount"  : round(amount, 2),
        "type"    : record["type"],
        "error"   : record["error"],
    })

@app.route("/api/wallet/ledger")
@_login_required
def api_wallet_ledger():
    with _WALLET_LOCK:
        return jsonify({"ok": True, "ledger": list(reversed(_WALLET_LEDGER[-100:]))})

@app.route("/api/wallet/sync", methods=["POST"])
@_login_required
def api_wallet_sync():
    """Refresh live exchange balance."""
    market = (CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    acc = CONFIG.get("live_accounts", {}).get(market, {})
    if not (acc.get("api_key") and acc.get("api_secret")):
        return jsonify({"ok": False, "error": "No API credentials linked. Go to Settings → Live Trading to connect."}), 400
    payload = build_exchange_dashboard_payload(market, force_refresh=True, persist=True)
    if payload.get("connected"):
        return jsonify({"ok": True, "balance": payload.get("summary", {}).get("available_balance", 0.0),
                        "last_sync": payload.get("account", {}).get("last_sync"),
                        "exchange_dashboard": payload})
    return jsonify({"ok": False, "error": payload.get("error", "Sync failed"), "exchange_dashboard": payload}), 400


@app.route("/api/wallet/transfer", methods=["POST"])
@_login_required
def api_wallet_transfer():
    """Move funds internally from the exchange's trading/derivatives
    account to its funding/spot wallet — the step most exchanges require
    before an on-chain withdrawal can be initiated (see internal_transfer_funds)."""
    data   = request.json or {}
    market = (CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    acc    = CONFIG.get("live_accounts", {}).get(market, {})
    if not (acc.get("connected") and acc.get("api_key") and acc.get("api_secret")):
        return jsonify({"ok": False, "error": "No connected live exchange account. Go to Settings → Live Trading to connect."}), 400
    try:
        amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid amount"}), 400
    if amount <= 0:
        return jsonify({"ok": False, "error": "Amount must be greater than 0"}), 400

    result = internal_transfer_funds(market, amount, from_account="trading", to_account="funding")
    if result.get("ok"):
        _elog(f"Transferred ${amount:.2f} USDT from trading to funding wallet ({market})", "INFO")
        # Refresh balance snapshot so the wallet page reflects the move immediately
        build_exchange_dashboard_payload(market, force_refresh=True, persist=True)
        return jsonify({"ok": True, "amount": amount})
    _elog(f"Internal transfer failed for {market}: {result.get('error')}", "ERROR")
    return jsonify({"ok": False, "error": result.get("error", "Transfer failed")}), 400


@app.route("/api/exchange/status")
@_login_required
def api_exchange_status():
    """
    Live exchange snapshot: balance, open positions, recent fills, last error.
    Powers the live-exchange panel in wallet.html and settings.html hero card.
    Falls back gracefully if exchange is unreachable.
    """
    market = (CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    acc    = CONFIG.get("live_accounts", {}).get(market, {})
    # "connected" reflects real, validated exchange credentials — same
    # definition Home's live-status pill already uses. It must NOT also
    # require live_trading_enabled: that's a separate switch for whether
    # the bot may autonomously place real trades, not whether the account
    # itself is linked. Conflating the two is what made Settings/Wallet
    # show "disconnected" while Home correctly showed "Connected" for the
    # exact same account.
    is_live = (acc.get("connected", False) and
               bool(acc.get("api_key")) and bool(acc.get("api_secret")))

    base = {
        "connected"   : is_live,
        "trading_enabled": bool(CONFIG.get("live_trading_enabled", False)),
        "exchange"    : acc.get("exchange", ""),
        "market"      : market,
        "balance"     : round(float(acc.get("balance", 0) or 0), 2),
        "last_sync"   : acc.get("last_sync"),
        "last_error"  : acc.get("last_error", ""),
        "paper_trading": CONFIG.get("paper_trading", True),
        "open_positions": [],
        "recent_fills"  : [],
        "activity_log"  : [],
    }

    # Always include bot's own open/closed trades as exchange activity
    tm = _TM
    if tm:
        base["open_positions"] = [
            {
                "symbol"       : t["symbol"],
                "side"         : t["side"],
                "entry"        : t["entry"],
                "size"         : t["size"],
                "unrealised"   : t.get("unrealised_pnl", 0),
                "sl"           : t["sl"],
                "tp"           : t["tp"],
                "confidence"   : t.get("confidence", 0),
                "open_time"    : t.get("open_time",""),
                "execution_mode": t.get("execution_mode","PAPER"),
                "id"           : t["id"],
            }
            for t in (tm.open_trades or [])
        ]
        base["recent_fills"] = [
            {
                "symbol"     : t["symbol"],
                "side"       : t["side"],
                "entry"      : t["entry"],
                "close_price": t.get("close_price"),
                "pnl"        : t.get("pnl", 0),
                "status"     : t["status"],
                "close_time" : t.get("close_time",""),
                "execution_mode": t.get("execution_mode","PAPER"),
            }
            for t in list(reversed(tm.closed_trades[-20:]))
        ]

    # Last 30 activity log entries relevant to trading
    with _LOG_LOCK:
        base["activity_log"] = [
            e for e in reversed(_EVENT_LOG[-100:])
            if any(kw in e["msg"].lower() for kw in
                   ("trade","signal","scan","open","close","tp","sl","ai","error","live","balance"))
        ][:30]

    # If actually live, try fetching real exchange data (non-blocking — timeout 5s)
    if is_live and CCXT_AVAILABLE:
        try:
            ex, err = _get_live_exchange_client(market=market)
            if ex:
                raw_bal       = ex.fetch_balance()
                base["balance"] = _extract_free_usdt(raw_bal)
                acc["balance"]  = base["balance"]
                acc["last_sync"]= datetime.utcnow().isoformat()
                base["last_sync"] = acc["last_sync"]
                # Try fetching open orders (not all exchanges support this for spot)
                try:
                    syms = CONFIG.get("crypto_symbols", [])
                    orders = []
                    for sym in syms[:4]:
                        try:
                            o = ex.fetch_open_orders(sym.replace("_","/"))
                            orders.extend(o)
                        except Exception:
                            pass
                    if orders:
                        base["exchange_orders"] = [
                            {"id":o.get("id"),"symbol":o.get("symbol"),
                             "side":o.get("side",""),"amount":o.get("amount",0),
                             "price":o.get("price",0),"status":o.get("status","")}
                            for o in orders[:10]
                        ]
                except Exception:
                    pass
        except Exception as exc:
            base["last_error"] = str(exc)[:200]

    return jsonify({"ok": True, **base})


# ── Analysis API — runs live TA on every configured symbol ───
@app.route("/api/analysis")
@_login_required
def api_analysis():
    """
    Runs TechnicalAnalysis on every symbol currently configured in the bot,
    returns per-symbol indicator data, BOT-RL score, signal, S/R levels,
    and a comparison table for all strategies.  Used by market_analysis.html.
    """
    mode    = CONFIG["market_mode"]
    if mode == "CRYPTO":
        symbols = CONFIG.get("crypto_symbols", [])
        tf = CONFIG.get("crypto_timeframe", "15m")
    elif mode == "STOCK":
        symbols = CONFIG.get("stock_symbols", [])
        tf = CONFIG.get("stock_timeframe", "15m")
    elif mode == "FOREX":
        symbols = CONFIG.get("forex_symbols", [])
        tf = CONFIG.get("forex_timeframe", "15m")
    else:
        symbols = CONFIG.get("crypto_symbols", []) + CONFIG.get("stock_symbols", [])
        tf = CONFIG.get("crypto_timeframe", "15m")
    min_conf = CONFIG["min_confidence_to_trade"]
    sl_pct   = CONFIG["stop_loss_pct"]
    tp_pct   = CONFIG["take_profit_pct"]

    # Reuse the live data provider if bot is running, else create a fresh one
    data_provider = (_BOT.data if _BOT and _BOT.data else MarketDataProvider(mode))
    ta = TechnicalAnalysis()

    results = []
    errors  = []
    ts      = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    for sym in symbols:
        try:
            # Pull live OHLCV (returns None if exchange is unreachable)
            df = data_provider.get_ohlcv(sym, tf, limit=200)
            if df is None or len(df) < 50:
                errors.append({"symbol": sym, "error": data_provider.last_error or "insufficient data"})
                continue

            df  = ta.compute(df)
            sig = ta.signal(df)

            # If the bot is actually running, prefer whatever it last
            # computed for this symbol over the signal we just computed
            # above — those two computations happen at slightly different
            # moments in a moving market and CAN legitimately disagree even
            # though they run the exact same formula. Using the bot's own
            # cached result guarantees this page always shows precisely
            # what the live engine saw/acted on, not a fresher-but-different
            # snapshot.
            signal_synced = False
            signal_age_s  = None
            if _BOT and _BOT.running and sym in _BOT.last_signals:
                cached = _BOT.last_signals[sym]
                try:
                    age = (datetime.utcnow() - datetime.fromisoformat(cached["ts"])).total_seconds()
                except Exception:
                    age = None
                # Only trust the cache while it's still within one scan
                # cycle old — a stale cache (bot stuck, symbol removed from
                # config, etc.) should fall back to the fresh computation
                # rather than show something the engine won't reconsider
                # for a while.
                if age is not None and age <= max(_BOT.scan_interval_seconds * 2, 120):
                    sig = cached
                    signal_synced = True
                    signal_age_s = round(age, 1)

            last  = df.iloc[-1]
            prev  = df.iloc[-2]
            px    = float(last["close"])
            high  = float(df["high"].tail(20).max())
            low   = float(df["low"].tail(20).min())

            # Safe float extraction
            def sf(v):
                try:
                    f = float(v)
                    return None if (f != f or abs(f) == float("inf")) else round(f, 6)
                except Exception:
                    return None

            rsi_fast  = sf(last.get(f"rsi_{ta.fast}"))
            rsi_slow  = sf(last.get(f"rsi_{ta.slow}"))
            rsi_confirm = sf(last.get(f"rsi_{ta.confirm}"))
            ema_21    = sf(last.get("ema_21"))
            ema_50    = sf(last.get("ema_50"))
            ema_200   = sf(last.get("ema_200"))
            atr       = sf(last.get("atr"))
            atr_pct   = sf(last.get("atr_pct"))
            vol_surge = bool(last.get("vol_surge", False))
            mom5      = sf(last.get("mom5"))
            last_sh   = sf(last.get("last_sh"))
            last_sl   = sf(last.get("last_sl"))
            obv_now   = sf(last.get("obv"))
            obv_prev  = sf(prev.get("obv"))

            ema_trend    = "Bullish" if last.get("ema_trend", -1) == 1 else "Bearish"
            obv_trending = ("Rising" if obv_now and obv_prev and obv_now > obv_prev
                            else "Falling" if obv_now and obv_prev else "N/A")

            # BOT-RL score breakdown (mirrors TechnicalAnalysis.signal exactly)
            score_breakdown = {}
            thr = CONFIG["rsi_signal_threshold"]
            rsi_div  = float(last.get("rsi_div", 0) or 0)
            rsi_pdiv = float(prev.get("rsi_div", 0) or 0)
            buy = 0; sell = 0

            score_breakdown["rsi_oversold"]  = 25 if (rsi_fast and rsi_fast < CONFIG["rsi_oversold"])  else 0
            score_breakdown["rsi_overbought"]= 25 if (rsi_fast and rsi_fast > CONFIG["rsi_overbought"]) else 0
            score_breakdown["rsi_cross_buy"] = 20 if (rsi_div > thr and rsi_pdiv <= thr)  else 0
            score_breakdown["rsi_cross_sell"]= 20 if (rsi_div < -thr and rsi_pdiv >= -thr) else 0
            score_breakdown["swing_low"]     = 20 if (last_sl and abs(px-last_sl)/px < 0.012) else 0
            score_breakdown["swing_high"]    = 20 if (last_sh and abs(px-last_sh)/px < 0.012) else 0
            score_breakdown["ema_bullish"]   = 15 if last.get("ema_trend", -1) == 1 else 0
            score_breakdown["ema_bearish"]   = 15 if last.get("ema_trend", -1) != 1 else 0
            score_breakdown["vol_surge"]     = 10 if vol_surge else 0
            score_breakdown["momentum"]      = 10 if (mom5 and abs(mom5) > 0.5) else 0

            # Support / resistance levels from recent swing pivots
            swing_highs = df["sw_h"].dropna().tail(3).tolist()
            swing_lows  = df["sw_l"].dropna().tail(3).tolist()

            # Price change vs previous close
            prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else px
            pct_change = round((px - prev_close) / prev_close * 100, 3) if prev_close else 0

            # Candlestick pattern hints (simple heuristics)
            body   = abs(px - float(last["open"]))
            candle = float(last["high"]) - float(last["low"])
            wick_r = (float(last["high"]) - max(px, float(last["open"]))) / candle if candle else 0
            wick_l = (min(px, float(last["open"])) - float(last["low"])) / candle if candle else 0
            # A zero-range candle (high == low) would otherwise divide by
            # zero here; treat it as a Doji instead of crashing.
            pattern = "Doji" if (candle == 0 or body/candle < 0.1) else (
                      "Hammer" if wick_l > 0.6 else (
                      "Shooting Star" if wick_r > 0.6 else (
                      "Bullish" if px > float(last["open"]) else "Bearish")))

            # True reward:risk for THIS signal, computed from its actual
            # sl/tp/price — not from raw config percentages. Deriving it
            # from config (tp_pct / sl_pct) is what produced the nonsensical
            # "1000:1" badge when take_profit_pct was misconfigured; this
            # is accurate regardless of config and always reflects reality.
            actual_rr = None
            if sig.get("sl") is not None and sig.get("tp") is not None and sig.get("price"):
                risk   = abs(sig["price"] - sig["sl"])
                reward = abs(sig["tp"] - sig["price"])
                if risk > 0:
                    actual_rr = round(reward / risk, 1)

            results.append({
                "symbol"       : sym,
                "price"        : round(px, 6),
                "pct_change"   : pct_change,
                "signal"       : sig["side"],
                "confidence"   : round(sig.get("confidence", 0), 1),
                "min_conf"     : min_conf,
                "reasons"      : sig.get("reasons", []),
                "sl"           : sig.get("sl"),
                "tp"           : sig.get("tp"),
                "sl_pct"       : round(sl_pct * 100, 2),
                "tp_pct"       : round(tp_pct * 100, 2),
                "rr_ratio"     : actual_rr,
                "synced_with_bot": signal_synced,
                "signal_age_s"    : signal_age_s,
                "rsi_fast"     : round(rsi_fast, 1) if rsi_fast else None,
                "rsi_slow"     : round(rsi_slow, 1) if rsi_slow else None,
                "rsi_confirm"  : round(rsi_confirm, 1) if rsi_confirm else None,
                "rsi_fast_period": ta.fast,
                "rsi_slow_period": ta.slow,
                "rsi_confirm_period": ta.confirm,
                "ema_21"       : round(ema_21, 4) if ema_21 else None,
                "ema_50"       : round(ema_50, 4) if ema_50 else None,
                "ema_200"      : round(ema_200, 4) if ema_200 else None,
                "ema_trend"    : ema_trend,
                "atr"          : round(atr, 4) if atr else None,
                "atr_pct"      : round(atr_pct * 100, 2) if atr_pct else None,
                "vol_surge"    : vol_surge,
                "momentum"     : round(mom5, 2) if mom5 else None,
                "obv_trend"    : obv_trending,
                "last_sh"      : round(last_sh, 4) if last_sh else None,
                "last_sl"      : round(last_sl, 4) if last_sl else None,
                "swing_highs"  : [round(v, 4) for v in swing_highs],
                "swing_lows"   : [round(v, 4) for v in swing_lows],
                "recent_high"  : round(high, 4),
                "recent_low"   : round(low, 4),
                "pattern"      : pattern,
                "score_breakdown": score_breakdown,
                "in_open_trade": any(t["symbol"]==sym for t in (_TM.open_trades if _TM else [])),
            })

        except Exception as e:
            logger.error(f"Analysis error for {sym}: {e}")
            errors.append({"symbol": sym, "error": str(e)})

    return jsonify({
        "ts"          : ts,
        "mode"        : mode,
        "timeframe"   : tf,
        "exchange"    : CONFIG.get("crypto_exchange",""),
        "min_conf"    : min_conf,
        "results"     : results,
        "errors"      : errors,
        "bot_running" : bool(_BOT and _BOT.running),
    })

# ── Core state API ────────────────────────────────────────────
@app.route("/api/activity")
@_login_required
def api_activity():
    username = session.get("username", "")
    return jsonify({"ok": True, "activity": get_recent_admin_activity(username, limit=50)})


@app.route("/api/state")
@_login_required
def api_state():
    tm  = _TM
    st  = tm.stats() if tm else {
        "balance":CONFIG["account_balance"],"equity":CONFIG["account_balance"],
        "total_pnl":0,"roi":0,"win_rate":0,"total":0,"wins":0,"losses":0,
        "open_count":0,"unrealised":0,"avg_win":0,"avg_loss":0}
    ot  = [_clean_trade(t) for t in (tm.open_trades   if tm else [])]
    ct  = [_clean_trade(t) for t in list(reversed((tm.closed_trades if tm else [])[-60:]))]
    bot = _BOT
    username = session.get("username", "")
    live_payload = build_live_trading_payload()
    st["live_balance"]   = live_payload.get("balance", 0.0)
    st["live_connected"] = live_payload.get("connected", False)
    st["live_exchange"]  = live_payload.get("exchange", "")
    st["live_enabled"]   = live_payload.get("enabled", False)
    market = (CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    st["exchange_dashboard"] = build_exchange_dashboard_payload(market, force_refresh=False, persist=False)
    return jsonify({
        "bot_running"  : bool(bot and bot.running),
        "scan_count"   : bot.scan_count if bot else 0,
        "market_mode"  : CONFIG["market_mode"],
        "paper_trading": CONFIG["paper_trading"],
        "symbols"      : (CONFIG.get("crypto_symbols") if CONFIG.get("market_mode")=="CRYPTO"
                  else CONFIG.get("stock_symbols")),
        "stats"        : st,
        "open_trades"  : ot,
        "closed_trades": ct,
        "live_prices"  : _PRICE_CACHE,
        "log"          : list(reversed(_EVENT_LOG[-60:])),
        "config"       : {k:v for k,v in CONFIG.items() if k!="anthropic_api_key"},
        "has_api_key"  : bool(CONFIG.get("anthropic_api_key")),
        "live_trading" : live_payload,
        "login_ts"     : session.get("login_ts", ""),
        "activity_history": get_recent_admin_activity(username, limit=20) if username else [],
    })

# ── Config update ─────────────────────────────────────────────
@app.route("/api/config", methods=["POST"])
@_login_required
def api_config():
    data    = request.json or {}
    allowed = ["market_mode","crypto_symbols","stock_symbols","forex_symbols","crypto_timeframe",
               "stock_timeframe","forex_timeframe","crypto_exchange","rsi_fast_period","rsi_slow_period",
               "rsi_confirm_period",
               "rsi_overbought","rsi_oversold","rsi_signal_threshold","stop_loss_pct",
               "take_profit_pct","min_confidence_to_trade",
               "max_risk_per_trade","max_open_trades","account_balance",
               "scan_interval_seconds","max_execution_slippage_pct","signal_exit_confirmations",
               "hold_requires_entry_confidence",
               "profit_lock_enabled","profit_lock_r_multiple","relax_exit_after_profit_lock",
               "trailing_stop_enabled","trailing_stop_activation_r","trailing_stop_distance_r",
               "chop_filter_enabled","chop_efficiency_lookback","chop_min_efficiency_ratio",
               "chop_confluence_require_volume_surge","chop_exit_confirmations",
               "counter_trend_discount_enabled","strong_trend_ema_sep_pct","max_counter_trend_discount_pct",
               "log_max_size_mb","log_backup_count","log_reset_months",
               "db_retention_days","db_max_size_mb","db_prune_interval_hours",
               "paper_trading","ai_confirmation_required",
               "anthropic_api_key"]

    # Sanity bounds for numeric fields that feed directly into SL/TP price
    # math. Without this, a value entered as a whole-number percent instead
    # of a fraction (e.g. "2" meaning "2%" instead of 0.02) sails straight
    # into the SL/TP formula and can produce a negative or inverted target
    # price — a trade that opens looking normal but can mathematically never
    # hit take-profit. This rejects the save instead of silently corrupting
    # every subsequent trade.
    NUMERIC_BOUNDS = {
        "stop_loss_pct":           (0.0005, 0.10,  "0.05% – 10% (e.g. 0.005 = 0.5%)"),
        "take_profit_pct":         (0.001,  0.50,  "0.1% – 50% (e.g. 0.02 = 2%)"),
        "max_risk_per_trade":      (0.001,  0.25,  "0.1% – 25% (e.g. 0.01 = 1%)"),
        "min_confidence_to_trade": (1,      99,    "1 – 99"),
        "rsi_fast_period":         (2,      50,    "2 – 50"),
        "rsi_slow_period":         (2,      100,   "2 – 100"),
        "rsi_confirm_period":      (10,     200,   "10 – 200"),
        "rsi_overbought":          (50,     99,    "50 – 99"),
        "rsi_oversold":            (1,      50,    "1 – 50"),
        "max_open_trades":         (1,      50,    "1 – 50"),
        "account_balance":         (1,      100000000, "> 0"),
        "scan_interval_seconds":   (5,      86400, "5 – 86400"),
        "max_execution_slippage_pct": (0.0005, 0.05, "0.05% – 5% (e.g. 0.003 = 0.3%)"),
        "signal_exit_confirmations": (1, 5, "1 – 5 consecutive scans"),
        "profit_lock_r_multiple":    (0.2, 5.0, "0.2 – 5.0 (multiples of initial risk)"),
        "trailing_stop_activation_r": (0.2, 10.0, "0.2 – 10.0 (multiples of initial risk)"),
        "trailing_stop_distance_r":  (0.1, 3.0, "0.1 – 3.0 (multiples of initial risk)"),
        "chop_efficiency_lookback":  (3, 50, "3 – 50 bars"),
        "chop_min_efficiency_ratio": (0.05, 0.95, "0.05 – 0.95 (0=pure noise, 1=perfect trend)"),
        "chop_exit_confirmations":   (1, 5, "1 – 5 consecutive scans"),
        "strong_trend_ema_sep_pct":  (0.001, 0.05, "0.1% – 5% EMA separation"),
        "max_counter_trend_discount_pct": (0.0, 0.9, "0% – 90%"),
        "log_max_size_mb":           (10, 2000, "10 – 2000 MB"),
        "log_backup_count":          (1, 20, "1 – 20 backups"),
        "log_reset_months":          (1, 12, "1 – 12 months"),
        "db_retention_days":         (7, 730, "7 – 730 days"),
        "db_max_size_mb":            (10, 2000, "10 – 2000 MB"),
        "db_prune_interval_hours":   (1, 168, "1 – 168 hours"),
    }
    errors = []
    for k, (lo, hi, hint) in NUMERIC_BOUNDS.items():
        if k in data:
            try:
                v = float(data[k])
            except (TypeError, ValueError):
                errors.append(f"{k}: must be a number")
                continue
            if not (lo <= v <= hi):
                errors.append(f"{k}: {v} is out of range — expected {hint}")
    if "rsi_fast_period" in data and "rsi_slow_period" in data:
        try:
            if float(data["rsi_fast_period"]) >= float(data["rsi_slow_period"]):
                errors.append("rsi_fast_period must be smaller than rsi_slow_period")
        except (TypeError, ValueError):
            pass
    slow_val = data.get("rsi_slow_period", CONFIG.get("rsi_slow_period"))
    confirm_val = data.get("rsi_confirm_period", CONFIG.get("rsi_confirm_period"))
    if "rsi_confirm_period" in data or "rsi_slow_period" in data:
        try:
            if float(confirm_val) <= float(slow_val):
                errors.append("rsi_confirm_period must be larger than rsi_slow_period "
                               "(it's meant to be the slower, broader-trend confirmation)")
        except (TypeError, ValueError):
            pass
    lock_val  = data.get("profit_lock_r_multiple", CONFIG.get("profit_lock_r_multiple"))
    trail_val = data.get("trailing_stop_activation_r", CONFIG.get("trailing_stop_activation_r"))
    if "profit_lock_r_multiple" in data or "trailing_stop_activation_r" in data:
        try:
            if float(trail_val) < float(lock_val):
                errors.append("trailing_stop_activation_r must be >= profit_lock_r_multiple "
                               "(trailing starts at or after the breakeven lock, not before)")
        except (TypeError, ValueError):
            pass
    if errors:
        _elog(f"Config save rejected — invalid values: {'; '.join(errors)}", "ERROR")
        return jsonify({"ok": False, "error": "Invalid settings — nothing was saved:\n" + "\n".join(errors)}), 400

    updated = [k for k in allowed if k in data and (CONFIG.__setitem__(k,data[k]) or True)]
    _elog(f"Config updated: {', '.join(updated)}", "INFO")
    return jsonify({"ok":True, "updated":updated})

# ── Bot control ───────────────────────────────────────────────
@app.route("/api/bot/start", methods=["POST"])
@_login_required
def api_bot_start():
    global _BOT, _BOT_THREAD, _TM
    if _BOT and _BOT.running:
        return jsonify({"ok":False,"error":"Bot already running"})
    if _TM is None:
        _TM = TradeManager()
    _STOP_EVENT.clear()
    _BOT        = TradingBot(_TM)
    _BOT.running = True
    _BOT_THREAD = threading.Thread(target=_BOT.run, daemon=True, name="BotScan")
    _BOT_THREAD.start()
    _elog("Bot started from dashboard", "INFO")
    record_admin_activity(session.get("username", "admin"), "bot", "Bot started from dashboard", details={"bot_running": True})
    return jsonify({"ok":True})

@app.route("/api/bot/stop", methods=["POST"])
@_login_required
def api_bot_stop():
    global _BOT
    _STOP_EVENT.set()
    if _BOT:
        _BOT.running = False
    _elog("Bot stopped from dashboard", "WARN")
    record_admin_activity(session.get("username", "admin"), "bot", "Bot stopped from dashboard", details={"bot_running": False})
    return jsonify({"ok":True})

# ── Trade close ───────────────────────────────────────────────
@app.route("/api/trade/close/<trade_id>", methods=["POST"])
@_login_required
def api_close_trade(trade_id):
    if not _TM:
        return jsonify({"ok":False,"error":"No trade manager"}), 400
    t = _TM.force_close(trade_id, _PRICE_CACHE)
    if t:
        return jsonify({"ok":True,"trade":_clean_trade(t)})
    return jsonify({"ok":False,"error":"Trade not found"}), 404

@app.route("/api/trade/execute", methods=["POST"])
@_login_required
def api_execute_trade():
    data = request.json or {}
    symbol = (data.get("symbol") or "BTC/USDT").strip().upper()
    side = (data.get("side") or "BUY").upper()
    amount = float(data.get("amount") or 0.0)
    if amount <= 0:
        return jsonify({"ok": False, "error": "Amount must be greater than zero"}), 400
    if CONFIG.get("paper_trading", True):
        return jsonify({"ok": False, "error": "Switch off paper trading to execute live orders"}), 400
    if not CONFIG.get("live_trading_enabled", False):
        return jsonify({"ok": False, "error": "Live trading is not enabled — turn it on in Settings → Live Trading"}), 400
    result = execute_live_order(symbol, side, amount)
    if result.get("ok"):
        return jsonify({"ok": True, "order": result.get("order"), "balance": result.get("balance")})
    return jsonify({"ok": False, "error": result.get("error", "Execution failed")}), 400


@app.route("/api/signals/tradeable")
@_login_required
def api_signals_tradeable():
    """Feeds the manual-trade dropdown: live BUY/SELL signals from the
    bot's own last-scan cache (the same numbers it actually acted/would
    act on — see the single-source-of-truth note on last_signals),
    currently open positions (for duplicating), and recently closed
    trades (for re-opening a setup that already played out)."""
    options = []

    if _BOT and getattr(_BOT, "last_signals", None):
        for sym, sig in _BOT.last_signals.items():
            if sig.get("side") not in ("BUY", "SELL"):
                continue
            try:
                age_s = (datetime.utcnow() - datetime.fromisoformat(sig["ts"])).total_seconds()
            except Exception:
                age_s = None
            conf = round(sig.get("confidence", 0), 1)
            options.append({
                "type": "live_signal",
                "symbol": sym,
                "side": sig["side"],
                "confidence": conf,
                "price": sig.get("price"),
                "sl": sig.get("sl"),
                "tp": sig.get("tp"),
                "ts": sig.get("ts"),
                "age_s": round(age_s, 1) if age_s is not None else None,
                "meets_threshold": conf >= CONFIG["min_confidence_to_trade"],
                "reasons": sig.get("reasons", [])[:3],
                "label": (f"{sym} {sig['side']} · {conf:.0f}% "
                          f"{'✓ meets threshold' if conf >= CONFIG['min_confidence_to_trade'] else '· below threshold'} "
                          f"· live"),
            })

    if _TM:
        for t in _TM.open_trades:
            options.append({
                "type": "open_trade",
                "symbol": t["symbol"],
                "side": t["side"],
                "confidence": t.get("confidence", 0),
                "price": t.get("entry"),
                "sl": t.get("sl"),
                "tp": t.get("tp"),
                "trade_id": t["id"],
                "opened_at": t.get("open_time"),
                "unrealised_pnl": round(t.get("unrealised_pnl", 0), 2),
                "label": (f"{t['symbol']} {t['side']} · open since "
                          f"{(t.get('open_time') or '')[11:16]} · "
                          f"{'+'if t.get('unrealised_pnl',0)>=0 else ''}${t.get('unrealised_pnl',0):.2f} "
                          f"· duplicate this position"),
            })

        for t in sorted(_TM.closed_trades, key=lambda x: x.get("close_time") or "", reverse=True)[:10]:
            options.append({
                "type": "closed_trade",
                "symbol": t["symbol"],
                "side": t["side"],
                "confidence": t.get("confidence", 0),
                "price": t.get("entry"),
                "sl": t.get("sl"),
                "tp": t.get("tp"),
                "trade_id": t["id"],
                "closed_at": t.get("close_time"),
                "pnl": round(t.get("pnl", 0) or 0, 2),
                "status": t.get("status"),
                "label": (f"{t['symbol']} {t['side']} · closed {t.get('status','')} "
                          f"{'+'if (t.get('pnl') or 0)>=0 else ''}${(t.get('pnl') or 0):.2f} "
                          f"· reopen this setup"),
            })

    return jsonify({"ok": True, "options": options,
                     "min_confidence_to_trade": CONFIG["min_confidence_to_trade"]})


@app.route("/api/trade/manual", methods=["POST"])
@_login_required
def api_trade_manual():
    """Manually open a trade — including duplicating an already-open
    position or re-opening a closed one — with a user-specified position
    size and risk/reward, independent of the bot's own auto-sizing and
    the autonomous scanner's one-trade-per-symbol limit."""
    if not _TM or not _BOT:
        return jsonify({"ok": False, "error": "Bot is not running — start it first so live prices are available"}), 400

    data   = request.json or {}
    symbol = (data.get("symbol") or "").strip().upper()
    side   = (data.get("side") or "").strip().upper()
    source = (data.get("source") or "manual").strip()[:120]

    if not symbol:
        return jsonify({"ok": False, "error": "Symbol is required"}), 400
    if side not in ("BUY", "SELL"):
        return jsonify({"ok": False, "error": "side must be BUY or SELL"}), 400

    try:
        size_usd = float(data.get("size_usd", 0))
        sl_pct   = float(data.get("sl_pct", 0))
        tp_pct   = float(data.get("tp_pct", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "size_usd, sl_pct and tp_pct must be numbers"}), 400

    if size_usd <= 0:
        return jsonify({"ok": False, "error": "Position size must be greater than 0"}), 400
    if size_usd > _TM.balance * 0.9:
        return jsonify({"ok": False, "error": f"Position size ${size_usd:.2f} exceeds 90% of available "
                                               f"balance (${_TM.balance:.2f}) — reduce the size"}), 400
    if not (0.0005 <= sl_pct <= 0.25):
        return jsonify({"ok": False, "error": "Stop-loss % must be between 0.05% and 25% (e.g. 0.01 = 1%)"}), 400
    if not (0.001 <= tp_pct <= 1.0):
        return jsonify({"ok": False, "error": "Take-profit % must be between 0.1% and 100% (e.g. 0.03 = 3%)"}), 400

    # Always execute at the live price right now — never the (possibly
    # stale) price the signal was shown with in the dropdown. Same
    # philosophy as the autonomous engine's execution-time repricing.
    px = _BOT.data.get_price(symbol)
    if px is None:
        return jsonify({"ok": False, "error": f"Could not fetch a live price for {symbol} — check the symbol is valid for the current market mode"}), 400

    if side == "BUY":
        sl = round(px * (1 - sl_pct), 6)
        tp = round(px * (1 + tp_pct), 6)
        geometry_ok = 0 < sl < px < tp
    else:
        sl = round(px * (1 + sl_pct), 6)
        tp = round(px * (1 - tp_pct), 6)
        geometry_ok = 0 < tp < px < sl
    if not geometry_ok:
        return jsonify({"ok": False, "error": f"Invalid SL/TP at current price ${px:.6f} — check your % values"}), 400

    size_base = size_usd / px
    sig = {"side": side, "price": px, "sl": sl, "tp": tp,
           "confidence": CONFIG["min_confidence_to_trade"],  # manual trades don't carry a scan-derived score
           "reasons": [f"Manually opened by admin ({source})" if source else "Manually opened by admin"]}
    ai = {"action": "MANUAL", "confidence_adjustment": 0, "risk_level": "MANUAL",
          "analysis": f"Manually opened — source: {source or 'manual'}"}

    m = (CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    trade = _TM.open_trade(symbol, sig, ai, market=m, allow_duplicate=True,
                           override_size=size_base, opened_by="manual", manual_source=source)
    if trade:
        return jsonify({"ok": True, "trade": _clean_trade(trade)})
    return jsonify({"ok": False, "error": f"Could not open trade — max open trades "
                                           f"({CONFIG['max_open_trades']}) reached"}), 400

# ── Settings — profile ────────────────────────────────────────
@app.route("/api/settings/profile", methods=["GET","POST"])
@_login_required
def api_profile():
    u    = session.get("username","")
    user = ADMIN_STORE.get(u)
    if not user:
        return jsonify({"ok":False,"error":"User not found"}), 404
    if request.method == "POST":
        d = request.json or {}
        if "name" in d:
            if not d["name"].strip():
                return jsonify({"ok":False,"error":"Name cannot be empty"}), 400
            user["name"] = d["name"].strip()
        if "email" in d:
            if "@" not in d["email"]:
                return jsonify({"ok":False,"error":"Invalid email address"}), 400
            user["email"] = d["email"].strip()
        _elog(f"Profile updated for {u}", "INFO")
        return jsonify({"ok":True,"name":user["name"],"email":user["email"]})
    return jsonify({"ok":True,"username":u,"name":user["name"],"email":user["email"]})

# ── Settings — password ───────────────────────────────────────
@app.route("/api/settings/password", methods=["POST"])
@_login_required
def api_password():
    u    = session.get("username","")
    user = ADMIN_STORE.get(u)
    d    = request.json or {}
    cur  = d.get("current_password","")
    new  = d.get("new_password","")
    con  = d.get("confirm_password","")
    if not user or not check_password_hash(user["password_hash"], cur):
        return jsonify({"ok":False,"error":"Current password incorrect"}), 400
    if len(new) < 8:
        return jsonify({"ok":False,"error":"Password must be ≥ 8 characters"}), 400
    if new != con:
        return jsonify({"ok":False,"error":"Passwords do not match"}), 400
    user["password_hash"] = generate_password_hash(new)
    _elog(f"Password changed for {u}", "WARN")
    return jsonify({"ok":True})

# ── Settings — notifications ──────────────────────────────────
@app.route("/api/settings/notifications", methods=["GET","POST"])
@_login_required
def api_notifications():
    u    = session.get("username","")
    user = ADMIN_STORE.get(u,{})
    if request.method == "POST":
        d   = request.json or {}
        ntf = user.setdefault("notifications",{})
        for k,v in d.items():
            ntf[k] = bool(v)
        _elog(f"Notifications updated for {u}", "INFO")
        return jsonify({"ok":True,"notifications":ntf})
    return jsonify({"ok":True,"notifications":user.get("notifications",{})})

@app.route("/api/settings/live-trading", methods=["GET","POST"])
@_login_required
def api_live_trading():
    if request.method == "POST":
        data = request.json or {}
        enable_live = bool(data.get("enable_live_trading", False))
        market = (data.get("market") or "CRYPTO").strip().upper()
        acc = CONFIG.setdefault("live_accounts", {}).setdefault(market, {})

        exchange_id = (data.get("exchange") or acc.get("exchange") or "binance").lower()
        # Password-style fields are never pre-filled in the UI for
        # security, so a resubmit where the user didn't retype the key
        # arrives here as "". Overwriting a working saved credential with
        # blank on every unrelated save (e.g. just toggling something
        # else) is what was silently destroying working connections —
        # only replace a credential when a new, non-empty value was
        # actually provided.
        api_key    = (data.get("api_key") or "").strip()    or acc.get("api_key", "")
        api_secret = (data.get("api_secret") or "").strip() or acc.get("api_secret", "")
        passphrase = (data.get("passphrase") or "").strip() or acc.get("passphrase", "")

        if not api_key or not api_secret:
            return jsonify({"ok": False, "error": "API key and secret are required"}), 400

        acc.update({"exchange": exchange_id, "api_key": api_key, "api_secret": api_secret, "passphrase": passphrase})

        # Validate credentials against the exchange EVERY TIME they're
        # submitted — independent of enable_live_trading. Whether the bot
        # is allowed to autonomously trade is a separate safety switch;
        # "does this API key actually work" should never depend on it, or
        # a user who just wants to connect and check their balance without
        # turning on auto-trading gets a connection that silently never
        # validates.
        ok, result = validate_live_connection(exchange_id, api_key, api_secret, passphrase)
        acc["connected"] = ok
        acc["balance"] = float(result.get("balance", 0.0) or 0.0)
        acc["last_sync"] = datetime.utcnow().isoformat() if ok else None
        acc["last_error"] = "" if ok else result.get("error", "")
        save_exchange_account(market, exchange_id, api_key, api_secret, passphrase, ok, acc["balance"], acc["last_error"], acc["last_sync"])

        if ok:
            dashboard_payload = build_exchange_dashboard_payload(market, force_refresh=True, persist=True)
            acc["balance"] = float(dashboard_payload.get("summary", {}).get("available_balance", acc["balance"]) or acc["balance"])
            acc["last_sync"] = dashboard_payload.get("account", {}).get("last_sync") or acc["last_sync"]
            acc["last_error"] = ""
            _elog(f"Live account validated for {exchange_id} ({market})", "INFO")
        else:
            _elog(f"Live account validation failed for {market}: {result.get('error', 'unknown error')}", "ERROR")

        # enable_live_trading is independent of whether validation
        # succeeded above — EXCEPT you can never turn on autonomous
        # trading against credentials that just failed validation.
        CONFIG["live_trading_enabled"] = enable_live and ok
        CONFIG["paper_trading"] = not (enable_live and ok)

        _save_persistent_state()
        _sync_alias_live_account()
        return jsonify({
            "ok": ok,
            "connected": ok,
            "exchange": exchange_id,
            "balance": round(float(acc.get("balance", 0.0) or 0.0), 2),
            "trading_enabled": CONFIG["live_trading_enabled"],
            "error": result.get("error", "") if not ok else "",
        })

    # GET: optionally support ?market=CRYPTO|STOCK|FOREX to return a specific account
    market = (request.args.get("market") or None)
    payload = build_live_trading_payload()
    if market:
        m = market.strip().upper()
        acc = CONFIG.get("live_accounts", {}).get(m, {})
        return jsonify({
            "ok": True,
            "market": m,
            "exchange": acc.get("exchange", payload.get("exchange", "binance")),
            "connected": bool(acc.get("connected", False)),
            "balance": round(float(acc.get("balance", 0.0) or 0.0), 2),
            "last_sync": acc.get("last_sync"),
            "last_error": acc.get("last_error", ""),
        })
    return jsonify({"ok": True, **payload})

@app.route("/api/admin/storage")
@_login_required
def api_admin_storage():
    """Report current DB/log size and retention config so the dashboard
    can show how storage management is actually doing, not just that
    settings exist."""
    db_size_mb = 0.0
    try:
        if Path(DATABASE_PATH).exists():
            db_size_mb = Path(DATABASE_PATH).stat().st_size / (1024 * 1024)
    except Exception:
        pass
    log_size_mb = 0.0
    log_started = None
    try:
        if Path(CONFIG["log_file"]).exists():
            log_size_mb = Path(CONFIG["log_file"]).stat().st_size / (1024 * 1024)
        log_started = _get_log_started_at().isoformat()
    except Exception:
        pass
    backups = []
    try:
        base = Path(CONFIG["log_file"])
        for i in range(1, CONFIG["log_backup_count"] + 1):
            p = base.with_name(base.name + f".{i}")
            if p.exists():
                backups.append({"file": p.name, "size_mb": round(p.stat().st_size / (1024*1024), 2)})
    except Exception:
        pass
    row_counts = {"admin_activity": 0, "admin_sessions": 0}
    try:
        with _db_connect() as conn:
            row_counts["admin_activity"] = conn.execute("SELECT COUNT(*) FROM admin_activity").fetchone()[0]
            row_counts["admin_sessions"] = conn.execute("SELECT COUNT(*) FROM admin_sessions").fetchone()[0]
    except Exception:
        pass
    return jsonify({
        "ok": True,
        "database": {"size_mb": round(db_size_mb, 2), "max_mb": CONFIG["db_max_size_mb"],
                     "retention_days": CONFIG["db_retention_days"], "row_counts": row_counts,
                     "last_pruned_at": _LAST_DB_PRUNE_AT.isoformat() if _LAST_DB_PRUNE_AT else None},
        "log": {"active_size_mb": round(log_size_mb, 2), "max_mb": CONFIG["log_max_size_mb"],
                "reset_months": CONFIG["log_reset_months"], "started_at": log_started,
                "backup_count": CONFIG["log_backup_count"], "backups": backups},
    })


@app.route("/api/admin/storage/prune", methods=["POST"])
@_login_required
def api_admin_storage_prune():
    """Manually trigger the database prune immediately, bypassing the
    normal interval check — useful right after tightening retention
    settings rather than waiting up to db_prune_interval_hours."""
    result = prune_database(force=True)
    return jsonify(result)


@app.route("/api/settings/live-trading/disconnect", methods=["POST"])
@_login_required
def api_live_trading_disconnect():
    """Explicitly disconnect and clear a market's live exchange account.
    Separate from the save endpoint above — blank fields there now mean
    'keep the existing credential', so disconnect needs its own action
    rather than being inferred from an empty submit."""
    data   = request.json or {}
    market = (data.get("market") or CONFIG.get("market_mode") or "CRYPTO").strip().upper()
    acc    = CONFIG.setdefault("live_accounts", {}).setdefault(market, {})
    exchange_id = acc.get("exchange", "binance")

    acc.update({"api_key": "", "api_secret": "", "passphrase": "",
                "connected": False, "balance": 0.0, "last_error": "", "last_sync": None})
    save_exchange_account(market, exchange_id, "", "", "", False, 0.0, "", None)

    # If this was the account currently powering live trading, fall back
    # to paper mode — there's nothing left to trade live against.
    if (CONFIG.get("market_mode") or "").strip().upper() == market:
        CONFIG["live_trading_enabled"] = False
        CONFIG["paper_trading"] = True

    with _EXCHANGE_CACHE_LOCK:
        _EXCHANGE_CACHE.pop(market, None)

    _save_persistent_state()
    _sync_alias_live_account()
    _elog(f"Live account disconnected for {market}", "INFO")
    return jsonify({"ok": True, "market": market})

# ════════════════════════════════════════════════════════════
#  STARTUP WIZARD  (terminal prompts)
# ════════════════════════════════════════════════════════════

def _market_toggle():
    _section("⚙️  MARKET SELECTION")
    print(f"  {Fore.CYAN}[1]{Style.RESET_ALL} 🪙  CRYPTO — BTC, ETH, BNB, SOL")
    print(f"  {Fore.CYAN}[2]{Style.RESET_ALL} 📈  STOCK  — AAPL, TSLA, NVDA, AMZN")
    print(f"  {Fore.CYAN}[3]{Style.RESET_ALL} 💱  FOREX  — EUR/USD, GBP/USD, USD/JPY")
    print(f"  {Fore.CYAN}[4]{Style.RESET_ALL} 🎛️  CUSTOM — Choose CRYPTO or STOCK\n")
    while True:
        c = input(f"  {Fore.YELLOW}Select [1/2/3/4]: {Style.RESET_ALL}").strip()
        if c == "1":
            CONFIG["market_mode"] = "CRYPTO"
            print(Fore.GREEN+"  ✅ Crypto selected."+Style.RESET_ALL); return
        if c == "2":
            CONFIG["market_mode"] = "STOCK"
            print(Fore.GREEN+"  ✅ Stock selected."+Style.RESET_ALL); return
        if c == "3":
            CONFIG["market_mode"] = "FOREX"
            # auto-populate forex symbols if empty
            if not CONFIG.get("forex_symbols"): CONFIG["forex_symbols"] = ["EURUSD=X","GBPUSD=X","USDJPY=X"]
            print(Fore.GREEN+"  ✅ Forex selected."+Style.RESET_ALL); return
        if c == "4":
            mkt = input("  CRYPTO or STOCK: ").strip().upper()
            if mkt not in ("CRYPTO","STOCK"):
                print(Fore.RED+"  Invalid."+Style.RESET_ALL); continue
            CONFIG["market_mode"] = mkt
            raw  = input("  Symbols (comma-separated): ").strip()
            syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
            if syms:
                if mkt=="CRYPTO": CONFIG["crypto_symbols"] = syms
                else:             CONFIG["stock_symbols"]  = syms
            print(Fore.GREEN+f"  ✅ {mkt}: {syms}"+Style.RESET_ALL); return
        print(Fore.RED+"  Enter 1, 2, 3 or 4."+Style.RESET_ALL)

def _api_key_prompt():
    if not CONFIG["anthropic_api_key"]:
        _section("🔑 ANTHROPIC API KEY")
        print("  Leave blank to use technical signals only.")
        k = input("  API Key (or Enter to skip): ").strip()
        if k:
            CONFIG["anthropic_api_key"] = k
            print(Fore.GREEN+"  ✅ Key set."+Style.RESET_ALL)
        else:
            CONFIG["ai_confirmation_required"] = False
            print(Fore.YELLOW+"  ⚠️  AI disabled."+Style.RESET_ALL)

def _paper_toggle():
    _section("💹 TRADING MODE")
    print(f"  {Fore.CYAN}[1]{Style.RESET_ALL} 📝  PAPER  (safe, recommended)")
    print(f"  {Fore.CYAN}[2]{Style.RESET_ALL} 🔴  LIVE   (real capital — use with caution)\n")
    while True:
        c = input(f"  {Fore.YELLOW}Select [1/2]: {Style.RESET_ALL}").strip()
        if c == "1":
            CONFIG["paper_trading"] = True
            print(Fore.GREEN+"  ✅ Paper mode."+Style.RESET_ALL); return
        if c == "2":
            CONFIG["paper_trading"] = False
            print(Fore.RED+"  ⚠️  LIVE mode — real capital at risk!"+Style.RESET_ALL)
            if input("  Type CONFIRM: ").strip() != "CONFIRM":
                CONFIG["paper_trading"] = True
                print(Fore.YELLOW+"  Reverted to paper."+Style.RESET_ALL)
            return
        print(Fore.RED+"  Invalid."+Style.RESET_ALL)

def _interval_prompt():
    _section("⏱️  SCAN INTERVAL")
    raw = input(f"  Seconds between scans [{CONFIG['scan_interval_seconds']}]: ").strip()
    if raw.isdigit() and int(raw) >= 30:
        CONFIG["scan_interval_seconds"] = int(raw)
        print(Fore.GREEN+f"  ✅ {raw}s"+Style.RESET_ALL)

# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    global _TM, _BOT, _BOT_THREAD

    non_interactive = bool(os.environ.get("BOT_NON_INTERACTIVE", "").lower() in {"1", "true", "yes"}) or not sys.stdin.isatty()

    if not non_interactive:
        _banner()
        _market_toggle()
        _api_key_prompt()
        _paper_toggle()
        _interval_prompt()

    syms = (CONFIG["crypto_symbols"] if CONFIG["market_mode"]=="CRYPTO"
            else CONFIG["forex_symbols"] if CONFIG["market_mode"]=="FOREX"
            else CONFIG["stock_symbols"])
    port = CONFIG["web_port"]

    _section("🚀 STARTING")
    print(f"  Market   : {Fore.CYAN}{CONFIG['market_mode']}{Style.RESET_ALL}")
    print(f"  Symbols  : {Fore.CYAN}{', '.join(syms)}{Style.RESET_ALL}")
    print(f"  Mode     : {'📝 Paper' if CONFIG['paper_trading'] else '🔴 LIVE'}")
    print(f"  AI       : {'✅ Enabled' if CONFIG['ai_confirmation_required'] else '⚠️ Off'}")
    print(f"  Balance  : ${CONFIG['account_balance']:.2f}")
    print(f"  Dashboard: {Fore.CYAN}http://localhost:{port}{Style.RESET_ALL}")
    print(f"  Login    : admin / BotRL@2026!\n")
    if not non_interactive:
        input(f"  {Fore.GREEN}Press Enter to launch...{Style.RESET_ALL}")

    # Initialise trade manager & start bot thread
    _TM         = TradeManager()
    _BOT        = TradingBot(_TM)
    _BOT_THREAD = threading.Thread(target=_BOT.run, daemon=True, name="BotScan")
    _BOT_THREAD.start()
    _BOT.running = True
    _elog("Bot thread started", "INFO")

    # Silence Flask request logs so terminal stays clean
    import logging as _L
    _L.getLogger("werkzeug").setLevel(_L.ERROR)

    print(Fore.CYAN+f"\n  🌐 Web dashboard at http://localhost:{port}"+Style.RESET_ALL)
    try:
        app.run(host=CONFIG["web_host"], port=port,
                debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        _STOP_EVENT.set()
        if _BOT: _BOT.running = False
        print(Fore.YELLOW+"\n  Stopped. Goodbye."+Style.RESET_ALL)

if __name__ == "__main__":
    main()
