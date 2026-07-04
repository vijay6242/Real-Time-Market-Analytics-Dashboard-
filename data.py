# data.py  ─  Upstox Instrument Cache · Historical Loader · All Technical Indicators
# ═══════════════════════════════════════════════════════════════════════════════════
# Indicators: RSI, MACD, EMA, Bollinger Bands, ATR, ADX, SuperTrend, VWAP
# Risk:       Max Drawdown, Sharpe, Sortino, Calmar, Win Rate
# New:        Option-chain analytics, Market Breadth (simulated), Gap Analysis
# ═══════════════════════════════════════════════════════════════════════════════════

import ssl, urllib.request, csv, gzip, io, os, glob, threading
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import norm

from logger import get_logger
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────
# SSL
# ─────────────────────────────────────────────────────────
def _ssl():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        c = ssl.create_default_context()
        c.check_hostname = False
        c.verify_mode = ssl.CERT_NONE
        return c

_SSL = _ssl()

# ─────────────────────────────────────────────────────────
# Instrument cache  (Upstox complete CSV, once per day)
# ─────────────────────────────────────────────────────────
_CSV_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
_cache: dict = {}

def _today_csv() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"instruments_{date.today().isoformat()}.csv")

def delete_old_csv():
    today = date.today().isoformat()
    for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruments_*.csv")):
        if os.path.basename(p)[len("instruments_"):-len(".csv")] != today:
            try: os.remove(p)
            except: pass

def _download_csv() -> list:
    req = urllib.request.Request(_CSV_URL)
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
        raw = r.read()
    buf = io.BytesIO(raw)
    with gzip.open(buf, 'rt', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    buf.seek(0)
    with gzip.open(buf, 'rt', encoding='utf-8') as f_in, \
         open(_today_csv(), 'w', encoding='utf-8', newline='') as f_out:
        f_out.write(f_in.read())
    logger.info(f"[CSV] Downloaded {len(rows)} instruments.")
    return rows

def load_instrument_cache() -> list:
    today = date.today().isoformat()
    if _cache.get('date') == today and _cache.get('data'):
        return _cache['data']
    p = _today_csv()
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        _cache.update({'date': today, 'data': rows})
        return rows
    rows = _download_csv()
    _cache.update({'date': today, 'data': rows})
    return rows

# ─────────────────────────────────────────────────────────
# UpstoxData  —  symbol search + instrument key resolution
# ─────────────────────────────────────────────────────────
class UpstoxData:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self._instruments = None

    def _load(self):
        if self._instruments is None:
            self._instruments = load_instrument_cache()
        return self._instruments

    def search_instruments(self, query: str, exchange: str = None) -> list:
        q = query.strip().upper()
        out = []
        for inst in self._load():
            sym  = inst.get('tradingsymbol', '').upper()
            name = inst.get('name', '').upper()
            exch = inst.get('exchange', '')
            if exchange and exch != exchange: continue
            if q in sym or q in name:
                out.append({'symbol': inst.get('tradingsymbol',''), 'name': inst.get('name',''),
                            'exchange': exch, 'instrument_key': inst.get('instrument_key',''),
                            'instrument_type': inst.get('instrument_type','')})
            if len(out) >= 30: break
        return out

    def get_instrument_key(self, symbol: str, preferred_exchange: str = None) -> str:
        sym = symbol.strip().upper()
        prio = ['NSE_INDEX','NSE_EQ','BSE_EQ','NSE_FO']
        if preferred_exchange:
            prio = [preferred_exchange] + [p for p in prio if p != preferred_exchange]
        found = {}
        for inst in self._load():
            ts = inst.get('tradingsymbol','').upper()
            ex = inst.get('exchange','')
            if ts == sym and ex in prio and ex not in found:
                found[ex] = inst.get('instrument_key','')
        for ex in prio:
            if ex in found:
                return found[ex]
        raise ValueError(f"Instrument key not found: '{symbol}'")

    @staticmethod
    def get_popular_indices() -> dict:
        return {
            "NIFTY 50":    "NSE_INDEX|Nifty 50",
            "NIFTY BANK":  "NSE_INDEX|Nifty Bank",
            "NIFTY IT":    "NSE_INDEX|Nifty IT",
            "NIFTY MIDCAP":"NSE_INDEX|NIFTY MIDCAP 100",
            "SENSEX":      "BSE_INDEX|SENSEX",
            "INDIA VIX":   "NSE_INDEX|India VIX",
            "FIN NIFTY":   "NSE_INDEX|Nifty Fin Service",
        }

# ─────────────────────────────────────────────────────────
# Historical CSV loader
# ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

HISTORICAL_FILES = {
    "NIFTY 50":   "NIFTY 50-24-06-2025-to-24-06-2026.csv",
    "NIFTY BANK": "NIFTY BANK-24-06-2025-to-24-06-2026.csv",
    "INDIA VIX":  "hist_india_vix_-24-06-2025-to-24-06-2026.csv",
}

def load_historical(name: str) -> pd.DataFrame:
    fn = HISTORICAL_FILES.get(name)
    if not fn: return pd.DataFrame()
    p = DATA_DIR / fn
    if not p.exists():
        logger.warning(f"[Data] File not found: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%b-%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Open","High","Low","Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Shares Traded" in df.columns:
        df["Volume"] = pd.to_numeric(df["Shares Traded"], errors="coerce")
    elif "Volume" not in df.columns:
        df["Volume"] = np.nan
    if "% Change" in df.columns:
        df["Pct_Change"] = pd.to_numeric(df["% Change"].astype(str).str.strip(), errors="coerce")
    else:
        df["Pct_Change"] = df["Close"].pct_change() * 100
    return df

# ═══════════════════════════════════════════════════════════
# TECHNICAL INDICATORS  (all formulas fully implemented)
# ═══════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

# ── RSI ───────────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI.
      delta  = Close - Prev Close
      avg_gain = EMA(max(delta,0), period)
      avg_loss = EMA(max(-delta,0), period)
      RS  = avg_gain / avg_loss
      RSI = 100 - 100/(1+RS)
    """
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

# ── MACD ──────────────────────────────────────────────────
def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    """
    MACD Line   = EMA(fast) − EMA(slow)
    Signal Line = EMA(MACD, signal)
    Histogram   = MACD − Signal
    """
    macd = _ema(close, fast) - _ema(close, slow)
    sig  = _ema(macd, signal)
    return macd, sig, macd - sig

# ── Bollinger Bands ───────────────────────────────────────
def compute_bollinger(close: pd.Series, period=20, std_mult=2.0):
    """
    Mid   = SMA(close, period)
    Upper = Mid + std_mult × StdDev(close, period)
    Lower = Mid − std_mult × StdDev(close, period)
    """
    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    return mid + std_mult*sigma, mid, mid - std_mult*sigma

# ── ATR ───────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period=14) -> pd.Series:
    """
    True Range = max(H-L, |H-PrevC|, |L-PrevC|)
    ATR = Wilder EMA(TR, period)
    """
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# ── ADX (+DI / -DI / ADX) ────────────────────────────────
def compute_adx(df: pd.DataFrame, period=14):
    """
    +DM = High - PrevHigh  (if positive and > -DM, else 0)
    -DM = PrevLow - Low    (if positive and > +DM, else 0)
    +DI = 100 × EMA(+DM) / ATR
    -DI = 100 × EMA(-DM) / ATR
    DX  = 100 × |+DI - -DI| / (+DI + -DI)
    ADX = EMA(DX, period)

    ADX < 20  → Ranging market
    20-25     → Trend forming
    25-40     → Strong trend
    40-60     → Very strong trend
    >60       → Extreme trend (rare)
    """
    high, low = df["High"], df["Low"]
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    atr_val  = compute_atr(df, period)
    plus_di  = 100 * plus_dm.ewm(alpha=1/period,  adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val

    dx  = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan) * 100)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return plus_di, minus_di, adx

def adx_signal(adx_val: float) -> str:
    if adx_val < 20:   return "Ranging"
    if adx_val < 25:   return "Trend Forming"
    if adx_val < 40:   return "Strong Trend"
    if adx_val < 60:   return "Very Strong Trend"
    return "Extreme Trend"

# ── SuperTrend ────────────────────────────────────────────
def compute_supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    """
    HL2  = (High + Low) / 2
    Upper Band = HL2 + multiplier × ATR(period)   ← basic
    Lower Band = HL2 − multiplier × ATR(period)   ← basic

    Bands are then adjusted so they never widen when price is already past them.
    Direction: 1 = Bullish (Buy), -1 = Bearish (Sell)
    Signal:    Price > SuperTrend line → Buy
               Price < SuperTrend line → Sell
    """
    atr  = compute_atr(df, period)
    hl2  = (df["High"] + df["Low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    st  = pd.Series(np.nan, index=df.index)
    dir_ = pd.Series(1,    index=df.index, dtype=int)

    for i in range(1, len(df)):
        # Adjust bands so they don't move against the trend
        if lower.iloc[i] < lower.iloc[i-1] or df["Close"].iloc[i-1] < lower.iloc[i-1]:
            lower.iloc[i] = lower.iloc[i]
        else:
            lower.iloc[i] = lower.iloc[i-1]

        if upper.iloc[i] > upper.iloc[i-1] or df["Close"].iloc[i-1] > upper.iloc[i-1]:
            upper.iloc[i] = upper.iloc[i]
        else:
            upper.iloc[i] = upper.iloc[i-1]

        # Direction
        if dir_.iloc[i-1] == 1:
            dir_.iloc[i] = -1 if df["Close"].iloc[i] < lower.iloc[i] else 1
        else:
            dir_.iloc[i] =  1 if df["Close"].iloc[i] > upper.iloc[i] else -1

        st.iloc[i] = lower.iloc[i] if dir_.iloc[i] == 1 else upper.iloc[i]

    return st, dir_

# ── VWAP ──────────────────────────────────────────────────
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Typical Price = (High + Low + Close) / 3
    VWAP = Σ(TP × Volume) / Σ(Volume)

    For daily bars: cumulative since first available row.
    For intraday: reset per session (handled by app when using tick data).

    Interpretation:
      Price > VWAP → Bullish (buying pressure above avg cost)
      Price < VWAP → Bearish (selling below avg institutional cost)
    """
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0)
    return (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)

# ── Gap Analysis ──────────────────────────────────────────
def compute_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gap %    = (Open - PrevClose) / PrevClose × 100
    Gap Up   : Open > PrevClose + 0.1%
    Gap Down : Open < PrevClose - 0.1%
    Gap Fill : Whether price returned to fill the gap intra-day
    """
    df = df.copy()
    df["Prev_Close"] = df["Close"].shift(1)
    df["Gap_Pct"]    = (df["Open"] - df["Prev_Close"]) / df["Prev_Close"] * 100
    df["Gap_Type"]   = df["Gap_Pct"].apply(
        lambda x: "Gap Up" if x > 0.1 else ("Gap Down" if x < -0.1 else "Flat"))
    df["Gap_Fill"]   = df.apply(
        lambda r: (r["Low"]  <= r["Prev_Close"]) if r["Gap_Type"] == "Gap Up"
             else (r["High"] >= r["Prev_Close"]) if r["Gap_Type"] == "Gap Down"
             else np.nan, axis=1)
    return df

# ── Rolling Volatility ────────────────────────────────────
def rolling_vol(df: pd.DataFrame, window=20) -> pd.Series:
    """Annualised rolling volatility."""
    return df["Close"].pct_change().rolling(window).std() * np.sqrt(252) * 100

# ═══════════════════════════════════════════════════════════
# MASTER compute_technicals  (attaches everything to df)
# ═══════════════════════════════════════════════════════════
def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    c = df["Close"]

    # ── Returns ──
    df["Daily_Ret"] = c.pct_change()

    # ── EMA ──
    df["EMA_20"]  = _ema(c, 20)
    df["EMA_50"]  = _ema(c, 50)
    df["EMA_200"] = _ema(c, 200)

    # ── RSI ──
    df["RSI"] = compute_rsi(c)

    # ── MACD ──
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(c)

    # ── Bollinger ──
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(c)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"] * 100  # %B width

    # ── ATR ──
    df["ATR"] = compute_atr(df)
    df["ATR_Pct"] = df["ATR"] / c * 100  # ATR as % of price

    # ── ADX ──
    df["DI_Plus"], df["DI_Minus"], df["ADX"] = compute_adx(df)
    df["ADX_Signal"] = df["ADX"].apply(adx_signal)

    # ── SuperTrend ──
    if len(df) >= 14:
        df["SuperTrend"], df["ST_Direction"] = compute_supertrend(df)
        df["ST_Signal"] = df["ST_Direction"].map({1: "Buy", -1: "Sell"})
    else:
        df["SuperTrend"] = np.nan
        df["ST_Direction"] = 1
        df["ST_Signal"] = "—"

    # ── VWAP ──
    if "Volume" in df.columns and df["Volume"].notna().any():
        df["VWAP"] = compute_vwap(df)
    else:
        df["VWAP"] = np.nan

    # ── Gap ──
    df = compute_gap(df)

    # ── Rolling Vol ──
    df["RollingVol_20"] = rolling_vol(df)

    # ── Trend/Range regime ──
    df["Market_Regime"] = df["ADX"].apply(
        lambda x: "Ranging" if x < 20 else ("Trending" if x < 40 else "Strong Trend"))

    return df

# ═══════════════════════════════════════════════════════════
# RISK METRICS
# ═══════════════════════════════════════════════════════════
def compute_risk_metrics(df: pd.DataFrame) -> dict:
    """
    Returns a comprehensive risk KPI dict:
      Max Drawdown, Sharpe, Sortino, Calmar, Win Rate,
      VaR (95%), CVaR (95%), Best/Worst day, Avg daily gain/loss
    """
    ret = df["Daily_Ret"].dropna()
    if ret.empty: return {}

    cum   = (1 + ret).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    ann_ret = float((cum.iloc[-1] ** (252 / len(cum)) - 1) * 100)
    ann_vol = float(ret.std() * np.sqrt(252) * 100)
    mdd     = float(dd.min() * 100)
    rf      = 6.5

    sharpe  = (ann_ret - rf) / ann_vol if ann_vol else 0
    down_vol = ret[ret < 0].std() * np.sqrt(252) * 100
    sortino  = (ann_ret - rf) / down_vol if down_vol else 0
    calmar   = ann_ret / abs(mdd) if mdd else 0
    win_rate = float((ret > 0).mean() * 100)
    ytd_ret  = float((cum.iloc[-1] - 1) * 100)

    var95  = float(np.percentile(ret, 5) * 100)
    cvar95 = float(ret[ret <= np.percentile(ret, 5)].mean() * 100)

    return {
        "YTD Return":      f"{ytd_ret:+.2f}%",
        "Ann. Return":     f"{ann_ret:+.2f}%",
        "Ann. Volatility": f"{ann_vol:.2f}%",
        "Max Drawdown":    f"{mdd:.2f}%",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Sortino Ratio":   f"{sortino:.3f}",
        "Calmar Ratio":    f"{calmar:.3f}",
        "Win Rate":        f"{win_rate:.1f}%",
        "VaR (95%)":       f"{var95:.2f}%",
        "CVaR (95%)":      f"{cvar95:.2f}%",
        "Best Day":        f"{ret.max()*100:+.2f}%",
        "Worst Day":       f"{ret.min()*100:+.2f}%",
        "Latest Close":    f"₹{df['Close'].iloc[-1]:,.2f}",
        "52W High":        f"₹{df['High'].max():,.2f}",
        "52W Low":         f"₹{df['Low'].min():,.2f}",
    }

# ═══════════════════════════════════════════════════════════
# OPTION CHAIN ANALYTICS  (simulated — replace with live API)
# ═══════════════════════════════════════════════════════════

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes pricing formula.
    S     = Spot price
    K     = Strike
    T     = Time to expiry (years)
    r     = Risk-free rate (decimal)
    sigma = Implied volatility (decimal)

    Returns: (price, delta, gamma, theta, vega, rho)
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic, 0, 0, 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho_v = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho_v = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type=="call"
                                          else norm.cdf(-d2))) / 365
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    return price, delta, gamma, theta, vega, rho_v

def generate_option_chain(spot: float, days_to_expiry: int = 7,
                           iv_base: float = 0.18, rf: float = 0.065) -> pd.DataFrame:
    """
    Generates a synthetic option chain around the spot price.
    In production: replace with Upstox /v2/option/chain API call.

    Columns:
      Strike, Call_OI, Call_ChgOI, Call_IV, Call_Price, Call_Delta,
      Call_Gamma, Call_Theta, Call_Vega,
      Put_OI,  Put_ChgOI,  Put_IV,  Put_Price,  Put_Delta,
      Put_Gamma, Put_Theta, Put_Vega, PCR, ATM
    """
    T   = max(days_to_expiry / 365, 0.001)
    atm = round(spot / 50) * 50
    strikes = [atm + i * 50 for i in range(-10, 11)]
    rows = []
    rng  = np.random.default_rng(42)

    for K in strikes:
        moneyness = abs(K - spot) / spot
        iv_call   = iv_base * (1 + moneyness * 2 + rng.normal(0, 0.01))
        iv_put    = iv_base * (1 + moneyness * 2.2 + rng.normal(0, 0.01))
        iv_call   = max(0.05, iv_call)
        iv_put    = max(0.05, iv_put)

        cp, cd, cg, ct, cv, cr = black_scholes(spot, K, T, rf, iv_call, "call")
        pp, pd_, pg, pt, pv, pr = black_scholes(spot, K, T, rf, iv_put,  "put")

        # Simulated OI (higher near ATM)
        oi_scale  = max(1, int(1e6 * np.exp(-moneyness * 15)))
        call_oi   = int(oi_scale * rng.uniform(0.5, 1.5))
        put_oi    = int(oi_scale * rng.uniform(0.6, 1.8))
        call_chg  = int(call_oi * rng.uniform(-0.15, 0.25))
        put_chg   = int(put_oi  * rng.uniform(-0.10, 0.30))

        rows.append({
            "Strike":      K,
            "ATM":         K == atm,
            "Call_OI":     call_oi,
            "Call_ChgOI":  call_chg,
            "Call_IV":     round(iv_call * 100, 2),
            "Call_Price":  round(cp, 2),
            "Call_Delta":  round(cd, 4),
            "Call_Gamma":  round(cg, 6),
            "Call_Theta":  round(ct, 2),
            "Call_Vega":   round(cv, 2),
            "Put_OI":      put_oi,
            "Put_ChgOI":   put_chg,
            "Put_IV":      round(iv_put * 100, 2),
            "Put_Price":   round(pp, 2),
            "Put_Delta":   round(pd_, 4),
            "Put_Gamma":   round(pg, 6),
            "Put_Theta":   round(pt, 2),
            "Put_Vega":    round(pv, 2),
        })

    df = pd.DataFrame(rows)
    total_call_oi = df["Call_OI"].sum()
    total_put_oi  = df["Put_OI"].sum()
    df["PCR"] = total_put_oi / total_call_oi if total_call_oi else 1.0

    return df

def compute_max_pain(chain_df: pd.DataFrame) -> float:
    """
    Max Pain = Strike where total option seller payout is minimised.
    At expiry: Call writers pay max(Spot-K,0) × OI
               Put  writers pay max(K-Spot,0) × OI
    """
    strikes = chain_df["Strike"].tolist()
    min_pain, max_pain_strike = float("inf"), strikes[0]
    for spot_test in strikes:
        call_loss = sum(max(spot_test - K, 0) * oi
                        for K, oi in zip(chain_df["Strike"], chain_df["Call_OI"]))
        put_loss  = sum(max(K - spot_test, 0) * oi
                        for K, oi in zip(chain_df["Strike"], chain_df["Put_OI"]))
        total = call_loss + put_loss
        if total < min_pain:
            min_pain = total
            max_pain_strike = spot_test
    return max_pain_strike

def pcr_signal(pcr: float) -> tuple:
    """
    PCR < 0.7  → Bullish (more calls than puts → market optimism)
    PCR 0.7-1.2 → Neutral
    PCR > 1.2  → Bearish (heavy put buying → hedging / fear)
    Returns (label, color)
    """
    if pcr < 0.7:  return "Bullish", "green"
    if pcr < 1.2:  return "Neutral", "orange"
    return "Bearish", "red"

# ═══════════════════════════════════════════════════════════
# MARKET BREADTH  (simulated — replace with live NSE data)
# ═══════════════════════════════════════════════════════════

def get_market_breadth(df_n50: pd.DataFrame) -> dict:
    """
    Simulates market breadth metrics for NIFTY 50 universe.
    In production: fetch from NSE /api/live-analysis-advance-decline

    Advance Decline Ratio = Advancing / Declining
    AD Line = cumulative sum of (Adv - Dec)
    New High New Low = stocks at 52W high vs low
    """
    rng   = np.random.default_rng(int(df_n50["Close"].iloc[-1]) % 1000)
    total = 50
    mkt_ret = df_n50["Daily_Ret"].iloc[-1] if "Daily_Ret" in df_n50.columns else 0

    # Skew toward market direction
    base_adv = 0.55 if mkt_ret > 0 else 0.38
    advancing = int(rng.binomial(total, base_adv))
    declining  = total - advancing - rng.integers(0, 3)
    unchanged  = total - advancing - max(declining, 0)

    ad_ratio = advancing / max(declining, 1)

    # Rolling AD Line from historical
    dates  = df_n50["Date"].tolist()
    rets   = df_n50["Daily_Ret"].fillna(0).tolist()
    ad_line = []
    val = 0
    for r in rets:
        adv_sim = int(rng.binomial(50, 0.55 if r > 0 else 0.38))
        dec_sim = 50 - adv_sim
        val += (adv_sim - dec_sim)
        ad_line.append(val)

    # Sector performance (simulated)
    sectors = {
        "BANK":     round(float(rng.normal(0.3, 0.8)), 2),
        "IT":       round(float(rng.normal(-0.1, 0.9)), 2),
        "FMCG":     round(float(rng.normal(0.1, 0.5)), 2),
        "AUTO":     round(float(rng.normal(0.2, 0.7)), 2),
        "PHARMA":   round(float(rng.normal(-0.2, 0.6)), 2),
        "REALTY":   round(float(rng.normal(0.4, 1.2)), 2),
        "ENERGY":   round(float(rng.normal(-0.3, 0.8)), 2),
        "METAL":    round(float(rng.normal(0.5, 1.0)), 2),
    }

    new_highs = int(rng.integers(2, 12))
    new_lows  = int(rng.integers(0, 8))

    return {
        "advancing":  advancing,
        "declining":  max(declining, 0),
        "unchanged":  max(unchanged, 0),
        "total":      total,
        "ad_ratio":   round(ad_ratio, 3),
        "ad_line":    ad_line,
        "ad_dates":   dates,
        "new_highs":  new_highs,
        "new_lows":   new_lows,
        "nh_nl_ratio": round(new_highs / max(new_lows, 1), 2),
        "sectors":    sectors,
        "breadth_signal": "Broad Rally" if advancing > 35
                          else "Broad Decline" if declining > 35
                          else "Mixed",
    }

# ═══════════════════════════════════════════════════════════
# COMBINED RISK SCORE  (0–100)
# ═══════════════════════════════════════════════════════════

def compute_risk_score(df: pd.DataFrame, vix_val: float,
                       breadth: dict, pcr_val: float) -> dict:
    """
    Composite intraday risk score from 5 pillars:
      1. VIX (25%)
      2. RSI extremes (20%)
      3. ADX / trend clarity (20%)
      4. Market breadth (20%)
      5. PCR (15%)

    Score 0-40  → Low risk   → Trade freely
    Score 40-65 → Medium     → Trade with caution
    Score 65-100→ High risk  → Avoid / hedge
    """
    scores = {}

    # 1. VIX score (higher VIX = higher risk)
    scores["vix"] = min(vix_val / 40 * 100, 100)

    # 2. RSI (oversold/overbought = risk)
    if "RSI" in df.columns:
        rsi_v = df["RSI"].iloc[-1]
        rsi_risk = abs(rsi_v - 50) / 50 * 100   # 0 at RSI=50, 100 at RSI=0 or 100
        scores["rsi"] = rsi_risk
    else:
        scores["rsi"] = 50

    # 3. ADX (low ADX = uncertain / ranging = higher risk for trend traders)
    if "ADX" in df.columns:
        adx_v = df["ADX"].iloc[-1]
        scores["adx"] = max(0, 100 - adx_v * 2)  # ADX 50+ = low risk; <20 = high risk
    else:
        scores["adx"] = 60

    # 4. Breadth (fewer advancing = higher risk)
    adv_pct = breadth.get("advancing", 25) / breadth.get("total", 50) * 100
    scores["breadth"] = max(0, 100 - adv_pct * 2)

    # 5. PCR (>1.5 = extreme fear = high risk)
    pcr_risk = min(pcr_val / 1.5 * 80, 100) if pcr_val < 1.5 else 90
    scores["pcr"] = pcr_risk

    # Weighted composite
    weights = {"vix": 0.25, "rsi": 0.20, "adx": 0.20, "breadth": 0.20, "pcr": 0.15}
    total   = sum(scores[k] * weights[k] for k in weights)

    level   = "Low Risk 🟢" if total < 40 else ("Medium Risk 🟡" if total < 65 else "High Risk 🔴")
    advice  = ("Good to trade — trend is clear." if total < 40
               else "Trade with caution — use stop losses." if total < 65
               else "Avoid fresh positions — high uncertainty.")

    return {
        "score":        round(total, 1),
        "level":        level,
        "advice":       advice,
        "pillars":      {k: round(v, 1) for k, v in scores.items()},
        "weights":      weights,
    }
