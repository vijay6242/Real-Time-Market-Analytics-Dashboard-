# database.py — SQLite persistence layer for the Market Dashboard
# ═══════════════════════════════════════════════════════════════════════════
# Uses Python's built-in sqlite3 module — no server, no extra install, no
# credentials. The DB file is created automatically at database/market.db.
#
# Thread safety: Streamlit's main thread and the WebSocket background thread
# both write concurrently, so all writes go through a single shared
# connection guarded by a threading.Lock, with WAL mode enabled for smoother
# concurrent reads/writes.
#
# Public API (unchanged regardless of backend, so app.py / websocket.py never
# need to change):
#   init_db()
#   save_historical_df(symbol, df) / get_historical_df(symbol)
#   save_live_tick(instrument_key, tick) / get_recent_ticks(...) / purge_old_ticks(...)
#   save_technical_indicators(symbol, df)
#   save_option_chain(symbol, chain_df, expiry)
#   save_prediction(symbol, pred, ...) / get_predictions(symbol)
#   add_alert(...) / get_alerts(...) / mark_alert_triggered(id) / delete_alert(id)
#   add_to_watchlist(...) / remove_from_watchlist(...) / get_watchlist()
#   set_setting(...) / get_setting(...) / get_all_settings()
#   get_access_token() / set_access_token(token)
# ═══════════════════════════════════════════════════════════════════════════

import sqlite3
import threading
from pathlib import Path
from datetime import datetime, date
from contextlib import contextmanager

import pandas as pd

from logger import get_logger
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────
# Location & connection
# ─────────────────────────────────────────────────────────
DB_DIR  = Path(__file__).parent / "database"
DB_NAME = DB_DIR / "market.db"
DB_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    """Returns the shared connection, creating it on first use."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_NAME), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")   # smoother concurrent read/write
        _conn.execute("PRAGMA foreign_keys=ON;")
    return _conn


@contextmanager
def _cursor(commit: bool = False):
    """Yields a cursor under the write lock; commits/rolls back as needed."""
    conn = get_connection()
    with _lock:
        cur = conn.cursor()
        try:
            yield cur
            if commit:
                conn.commit()
        except Exception:
            if commit:
                conn.rollback()
            raise


# ─────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS historical_data (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol  TEXT NOT NULL,
    date    DATE NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    UNIQUE(symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_hist_symbol_date ON historical_data(symbol, date);

CREATE TABLE IF NOT EXISTS live_ticks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_key  TEXT NOT NULL,
    ltp             REAL,
    change          REAL,
    change_percent  REAL,
    tick_time       DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ticks_key_time ON live_ticks(instrument_key, tick_time);

CREATE TABLE IF NOT EXISTS technical_indicators (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    date        DATE NOT NULL,
    rsi         REAL,
    macd        REAL,
    signal      REAL,
    ema20       REAL,
    ema50       REAL,
    ema200      REAL,
    atr         REAL,
    adx         REAL,
    supertrend  REAL,
    vwap        REAL,
    UNIQUE(symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_ti_symbol_date ON technical_indicators(symbol, date);

CREATE TABLE IF NOT EXISTS option_chain (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    strike       REAL NOT NULL,
    option_type  TEXT NOT NULL,      -- 'CE' / 'PE'
    oi           INTEGER,
    change_oi    INTEGER,
    volume       INTEGER,
    iv           REAL,
    ltp          REAL,
    expiry       DATE,
    captured_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_oc_symbol_expiry ON option_chain(symbol, expiry, captured_at);

CREATE TABLE IF NOT EXISTS predictions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol               TEXT NOT NULL,
    prediction_date      DATE NOT NULL,
    predicted_direction  TEXT,
    predicted_close      REAL,
    confidence           REAL,
    model_name           TEXT NOT NULL,
    accuracy             REAL,
    created_at           DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pred_symbol_date ON predictions(symbol, prediction_date);

CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    alert_type   TEXT NOT NULL,      -- 'Price >', 'RSI <', etc.
    alert_value  REAL,
    note         TEXT,
    status       TEXT DEFAULT 'active',   -- active | triggered | deleted
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    triggered_at DATETIME
);

CREATE TABLE IF NOT EXISTS watchlist (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    instrument_key  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS settings (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT NOT NULL UNIQUE,
    value  TEXT
);
"""


def init_db():
    """Create all tables if they don't already exist. Call once at app startup."""
    with _cursor(commit=True) as cur:
        cur.executescript(_SCHEMA)
    logger.info(f"[DB] Ready at {DB_NAME}")


# ═══════════════════════════════════════════════════════════
# 1. Historical Market Data
# ═══════════════════════════════════════════════════════════
def save_historical_df(symbol: str, df: pd.DataFrame):
    """Bulk upsert a historical OHLCV DataFrame (expects Date/Open/High/Low/Close/Volume)."""
    if df is None or df.empty:
        return
    rows = [
        (symbol, r["Date"].strftime("%Y-%m-%d") if hasattr(r["Date"], "strftime") else str(r["Date"]),
         r.get("Open"), r.get("High"), r.get("Low"), r.get("Close"), r.get("Volume"))
        for _, r in df.iterrows()
    ]
    with _cursor(commit=True) as cur:
        cur.executemany(
            """INSERT INTO historical_data (symbol,date,open,high,low,close,volume)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(symbol,date) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, volume=excluded.volume""",
            rows,
        )
    logger.info(f"[DB] Upserted {len(rows)} historical rows for {symbol}")


def get_historical_df(symbol: str) -> pd.DataFrame:
    with _cursor() as cur:
        cur.execute("SELECT date,open,high,low,close,volume FROM historical_data "
                    "WHERE symbol=? ORDER BY date", (symbol,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])


# ═══════════════════════════════════════════════════════════
# 2. Live Market Data
# ═══════════════════════════════════════════════════════════
def save_live_tick(instrument_key: str, tick: dict):
    """tick: dict as produced by websocket.TickStore (ltp, change, change_pct)."""
    with _cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO live_ticks (instrument_key, ltp, change, change_percent, tick_time)
               VALUES (?,?,?,?,?)""",
            (instrument_key, tick.get("ltp"), tick.get("change"),
             tick.get("change_pct"), datetime.now()),
        )


def get_recent_ticks(instrument_key: str, limit: int = 300) -> pd.DataFrame:
    with _cursor() as cur:
        cur.execute(
            """SELECT ltp, change, change_percent, tick_time FROM live_ticks
               WHERE instrument_key=? ORDER BY tick_time DESC LIMIT ?""",
            (instrument_key, limit),
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["ltp", "change", "change_percent", "tick_time"]).iloc[::-1]


def purge_old_ticks(keep_hours: int = 24):
    with _cursor(commit=True) as cur:
        cur.execute("DELETE FROM live_ticks WHERE tick_time < datetime('now', ?)",
                    (f"-{keep_hours} hours",))


# ═══════════════════════════════════════════════════════════
# 3. Technical Indicators
# ═══════════════════════════════════════════════════════════
def save_technical_indicators(symbol: str, df: pd.DataFrame):
    """df: output of data.compute_technicals() — columns EMA_20/EMA_50/EMA_200 etc."""
    if df is None or df.empty:
        return
    src_cols = ["RSI", "MACD", "MACD_Signal", "EMA_20", "EMA_50", "EMA_200",
                "ATR", "ADX", "SuperTrend", "VWAP"]
    rows = []
    for _, r in df.iterrows():
        d = r["Date"].strftime("%Y-%m-%d") if hasattr(r["Date"], "strftime") else str(r["Date"])
        vals = [r.get(c) if c in df.columns else None for c in src_cols]
        rows.append((symbol, d, *vals))
    with _cursor(commit=True) as cur:
        cur.executemany(
            """INSERT INTO technical_indicators
                 (symbol,date,rsi,macd,signal,ema20,ema50,ema200,atr,adx,supertrend,vwap)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol,date) DO UPDATE SET
                 rsi=excluded.rsi, macd=excluded.macd, signal=excluded.signal,
                 ema20=excluded.ema20, ema50=excluded.ema50, ema200=excluded.ema200,
                 atr=excluded.atr, adx=excluded.adx, supertrend=excluded.supertrend,
                 vwap=excluded.vwap""",
            rows,
        )


# ═══════════════════════════════════════════════════════════
# 4. Option Chain
# ═══════════════════════════════════════════════════════════
def save_option_chain(symbol: str, chain_df: pd.DataFrame, expiry: str = None):
    """chain_df: output of data.generate_option_chain() (Call_/Put_ columns per strike)."""
    if chain_df is None or chain_df.empty:
        return
    rows = []
    for _, r in chain_df.iterrows():
        rows.append((symbol, r["Strike"], "CE", r.get("Call_OI"), r.get("Call_ChgOI"),
                     None, r.get("Call_IV"), r.get("Call_Price"), expiry))
        rows.append((symbol, r["Strike"], "PE", r.get("Put_OI"), r.get("Put_ChgOI"),
                     None, r.get("Put_IV"), r.get("Put_Price"), expiry))
    with _cursor(commit=True) as cur:
        cur.executemany(
            """INSERT INTO option_chain
                 (symbol,strike,option_type,oi,change_oi,volume,iv,ltp,expiry)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )


# ═══════════════════════════════════════════════════════════
# 5. ML Predictions
# ═══════════════════════════════════════════════════════════
def save_prediction(symbol: str, pred: dict, model_name: str = "ensemble",
                     accuracy: float = None, prediction_date: str = None):
    """pred: dict as returned by ml_models.predict_tomorrow()/predict_today()."""
    if not pred or "error" in pred:
        return
    with _cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO predictions
                 (symbol,prediction_date,predicted_direction,predicted_close,
                  confidence,model_name,accuracy)
               VALUES (?,?,?,?,?,?,?)""",
            (symbol, prediction_date or date.today().isoformat(),
             pred.get("direction"), pred.get("price_forecast"),
             pred.get("confidence"), model_name, accuracy),
        )


def get_predictions(symbol: str, limit: int = 50) -> pd.DataFrame:
    with _cursor() as cur:
        cur.execute(
            """SELECT prediction_date,predicted_direction,predicted_close,confidence,
                      model_name,accuracy,created_at
               FROM predictions WHERE symbol=? ORDER BY created_at DESC LIMIT ?""",
            (symbol, limit),
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["prediction_date", "predicted_direction",
                                        "predicted_close", "confidence", "model_name",
                                        "accuracy", "created_at"])


# ═══════════════════════════════════════════════════════════
# 6. Alerts
# ═══════════════════════════════════════════════════════════
def add_alert(symbol: str, alert_type: str, alert_value: float, note: str = "") -> int:
    with _cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO alerts (symbol,alert_type,alert_value,note,status)
               VALUES (?,?,?,?,'active')""",
            (symbol, alert_type, alert_value, note),
        )
        return cur.lastrowid


def get_alerts(status: str = None) -> list:
    with _cursor() as cur:
        if status:
            cur.execute("SELECT * FROM alerts WHERE status=? ORDER BY created_at DESC", (status,))
        else:
            cur.execute("SELECT * FROM alerts ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def mark_alert_triggered(alert_id: int):
    with _cursor(commit=True) as cur:
        cur.execute("UPDATE alerts SET status='triggered', triggered_at=? WHERE id=?",
                    (datetime.now(), alert_id))


def delete_alert(alert_id: int):
    with _cursor(commit=True) as cur:
        cur.execute("UPDATE alerts SET status='deleted' WHERE id=?", (alert_id,))


# ═══════════════════════════════════════════════════════════
# 7. Watchlist
# ═══════════════════════════════════════════════════════════
def add_to_watchlist(symbol: str, instrument_key: str):
    with _cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO watchlist (symbol,instrument_key) VALUES (?,?)
               ON CONFLICT(instrument_key) DO UPDATE SET symbol=excluded.symbol""",
            (symbol, instrument_key),
        )


def remove_from_watchlist(instrument_key: str):
    with _cursor(commit=True) as cur:
        cur.execute("DELETE FROM watchlist WHERE instrument_key=?", (instrument_key,))


def get_watchlist() -> dict:
    """Returns {instrument_key: symbol} — same shape as st.session_state.watchlist."""
    with _cursor() as cur:
        cur.execute("SELECT instrument_key, symbol FROM watchlist")
        return {r["instrument_key"]: r["symbol"] for r in cur.fetchall()}


# ═══════════════════════════════════════════════════════════
# 8. Settings  (generic key-value store — also used for the access token)
# ═══════════════════════════════════════════════════════════
def set_setting(name: str, value):
    with _cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO settings (name,value) VALUES (?,?)
               ON CONFLICT(name) DO UPDATE SET value=excluded.value""",
            (name, str(value)),
        )


def get_setting(name: str, default=None):
    with _cursor() as cur:
        cur.execute("SELECT value FROM settings WHERE name=?", (name,))
        row = cur.fetchone()
    return row["value"] if row else default


def get_all_settings() -> dict:
    with _cursor() as cur:
        cur.execute("SELECT name, value FROM settings")
        return {r["name"]: r["value"] for r in cur.fetchall()}


# ── Access token convenience wrappers ──────────────────────
# Stored in `settings` under a reserved key so it survives Streamlit session
# resets (session_state alone is wiped on every browser refresh).
_TOKEN_KEY = "UPSTOX_ACCESS_TOKEN"


def get_access_token(default: str = "") -> str:
    return get_setting(_TOKEN_KEY, default)


def set_access_token(token: str):
    set_setting(_TOKEN_KEY, token)