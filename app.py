# app.py  ─  Real-Time Market Analytics Dashboard
# Run:  streamlit run app.py
# ════════════════════════════════════════════════════════════════════
# Tabs:
#   1. Market Overview    9.  Option Chain Analytics
#   2. Live Market        10. Market Breadth
#   3. OHLC Chart         11. Performance Dashboard
#   4. Technical Indicators   12. ML Predictions
#   5. Risk Analytics     13. Alerts
#   6. Gap Analysis       14. Backtesting
#   7. Volatility         15. Settings
#   8. India VIX
# ════════════════════════════════════════════════════════════════════

import os, time, threading
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv

from data import (
    UpstoxData, load_historical, compute_technicals, compute_risk_metrics,
    HISTORICAL_FILES, generate_option_chain, compute_max_pain, pcr_signal,
    get_market_breadth, compute_risk_score, adx_signal
)
from websocket import ws_hub, tick_store
from logger import get_logger
import database as db

load_dotenv()
logger = get_logger(__name__)
db.init_db()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Market Pulse · NSE Live",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Dark theme CSS
# ─────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root{
  --bg:#0a0b0d; --bg2:#111318; --panel:#14171d; --panel-2:#1a1e25;
  --border:#242a33; --border-soft:#1c212a;
  --text:#eef1f6; --muted:#8a93a3; --muted-2:#5c6576;
  --amber:#f5b64c; --amber-soft:rgba(245,182,76,.14); --amber-glow:rgba(245,182,76,.35);
  --green:#26d07c; --green-soft:rgba(38,208,124,.12);
  --red:#ff5c72; --red-soft:rgba(255,92,114,.12);
  --blue:#5b9dff;
  --radius:12px;
}

html,body,[class*="css"]{font-family:'Inter',sans-serif; color:var(--text)}
h1,h2,h3,h4,h5,h6{font-family:'Space Grotesk',sans-serif!important; letter-spacing:-.2px}
code,pre,.mono,[data-testid="stMetricValue"],.tp,.ti .tp{font-family:'JetBrains Mono',monospace!important}

/* ── App shell ─────────────────────────────────────── */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 500px at 15% -10%, rgba(245,182,76,.06), transparent 60%),
    radial-gradient(900px 400px at 100% 0%, rgba(91,157,255,.04), transparent 55%),
    var(--bg);
}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.1rem; max-width:1500px}

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--panel-2) 0%, var(--bg2) 100%);
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] *{color:var(--text)!important}
[data-testid="stSidebar"] hr{border-color:var(--border-soft); margin:14px 0}
.brand{display:flex; align-items:center; gap:9px; padding:2px 0 0 0}
.brand-mark{width:9px; height:9px; border-radius:2px; background:var(--amber);
  box-shadow:0 0 10px var(--amber-glow); flex-shrink:0}
.brand-title{font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:17px;
  letter-spacing:.2px; color:var(--text)}
.brand-sub{font-size:10.5px; color:var(--muted-2); letter-spacing:.6px;
  text-transform:uppercase; margin:2px 0 0 18px}
[data-testid="stSidebar"] h3{
  font-size:11px!important; font-weight:700!important; letter-spacing:.9px;
  text-transform:uppercase; color:var(--muted)!important; margin-top:2px!important;
}

/* status badge */
.badge{display:inline-flex; align-items:center; gap:6px; font-size:11.5px;
  font-weight:600; padding:4px 10px; border-radius:999px; border:1px solid var(--border)}
.dot{width:7px; height:7px; border-radius:50%; flex-shrink:0}
.dot-live{background:var(--green); box-shadow:0 0 8px var(--green); animation:pulse 1.8s infinite}
.dot-off{background:var(--muted-2)}
.dot-err{background:var(--red); box-shadow:0 0 8px var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ── Section headers (signature: amber terminal tick) ─ */
.sec{
  display:flex; align-items:center; gap:8px;
  font-family:'Space Grotesk',sans-serif; font-size:12.5px; font-weight:700;
  color:var(--text); letter-spacing:.3px; text-transform:uppercase;
  padding:2px 0 10px 0; margin-bottom:14px;
  border-bottom:1px solid var(--border-soft); position:relative;
}
.sec::before{content:"▍"; color:var(--amber); font-size:14px; line-height:0}
.sec::after{content:""; position:absolute; left:0; bottom:-1px; width:46px; height:1px;
  background:linear-gradient(90deg,var(--amber),transparent)}

/* ── Metric cards ──────────────────────────────────── */
[data-testid="stMetric"]{
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  padding:14px 16px 12px 16px; position:relative; overflow:hidden;
  transition:border-color .15s ease, transform .15s ease;
}
[data-testid="stMetric"]::before{content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--amber),transparent 70%)}
[data-testid="stMetric"]:hover{border-color:#3a4150; transform:translateY(-1px)}
[data-testid="stMetricLabel"]{color:var(--muted)!important; font-size:10.5px!important;
  text-transform:uppercase; letter-spacing:.6px; font-weight:600!important}
[data-testid="stMetricValue"]{color:var(--text)!important; font-size:21px!important; font-weight:600!important}
[data-testid="stMetricDelta"] svg{display:none}

/* ── Ticker tape ───────────────────────────────────── */
.pulse-bar{height:2px; width:100%; background-size:200% 100%;
  animation:shimmer 3.5s linear infinite}
.pulse-up{background-image:linear-gradient(90deg,transparent,var(--green),transparent)}
.pulse-down{background-image:linear-gradient(90deg,transparent,var(--red),transparent)}
@keyframes shimmer{0%{background-position:0% 0}100%{background-position:200% 0}}
.ticker{background:var(--panel); border:1px solid var(--border); border-top:none;
  border-radius:0 0 var(--radius) var(--radius);
  padding:10px 18px; display:flex; gap:30px; overflow-x:auto; white-space:nowrap;
  margin-bottom:18px;}
.ticker::-webkit-scrollbar{display:none}
.ti{display:inline-flex; flex-direction:column; min-width:112px; gap:2px}
.tn{font-size:9.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.6px; font-weight:600}
.tp{font-size:16px; font-weight:600; color:var(--text); letter-spacing:-.2px}
.tu{font-size:10.5px; color:var(--green); font-weight:600}
.td{font-size:10.5px; color:var(--red); font-weight:600}

/* ── Buttons ───────────────────────────────────────── */
.stButton>button, button[kind="secondary"]{
  background:var(--panel-2)!important; border:1px solid var(--border)!important;
  color:var(--text)!important; border-radius:8px!important; font-weight:600!important;
  transition:all .15s ease!important;
}
.stButton>button:hover, button[kind="secondary"]:hover{
  border-color:var(--amber)!important; color:var(--amber)!important;
}
button[kind="primary"]{
  background:var(--amber)!important; border:1px solid var(--amber)!important;
  color:#1a1305!important; border-radius:8px!important; font-weight:700!important;
  box-shadow:0 0 0 rgba(245,182,76,0); transition:box-shadow .15s ease!important;
}
button[kind="primary"]:hover{box-shadow:0 0 16px var(--amber-glow)!important}

/* ── Inputs ────────────────────────────────────────── */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-baseweb="select"] > div, [data-baseweb="input"]{
  background:var(--panel)!important; border:1px solid var(--border)!important;
  color:var(--text)!important; border-radius:8px!important;
}
[data-testid="stTextInput"] input:focus, [data-baseweb="select"] > div:focus-within{
  border-color:var(--amber)!important; box-shadow:0 0 0 1px var(--amber)!important;
}
[data-baseweb="radio"] label, [data-baseweb="checkbox"] label{color:var(--text)!important}
[data-testid="stSlider"] div[role="slider"]{background:var(--amber)!important; border-color:var(--amber)!important}
[data-testid="stSlider"] .st-emotion-cache-1dj0hjr, [data-testid="stTickBar"]{background:var(--border)!important}
div[data-baseweb="slider"] > div > div{background:var(--amber)!important}
[data-testid="stToggle"] [aria-checked="true"]{background:var(--amber)!important}

/* ── Tabs ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{gap:2px; border-bottom:1px solid var(--border-soft)}
.stTabs [data-baseweb="tab"]{
  font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:13px;
  color:var(--muted); padding:8px 14px; border-radius:8px 8px 0 0;
}
.stTabs [aria-selected="true"]{color:var(--amber)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--amber)!important; height:2px}

/* ── Alerts / callouts ─────────────────────────────── */
[data-testid="stAlertContainer"]{border-radius:10px!important; border:1px solid var(--border)!important}

/* ── DataFrames & tables ───────────────────────────── */
[data-testid="stDataFrame"]{border:1px solid var(--border); border-radius:10px; overflow:hidden}

/* ── Misc panels ───────────────────────────────────── */
.kpi{background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:12px 15px}
.risk-low{color:var(--green); font-weight:700}
.risk-med{color:var(--amber); font-weight:700}
.risk-hi{color:var(--red); font-weight:700}
[data-testid="stExpander"]{background:var(--panel); border:1px solid var(--border)!important; border-radius:10px!important}

/* ── Scrollbars ────────────────────────────────────── */
::-webkit-scrollbar{width:6px; height:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border); border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--amber)}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Color palette & Plotly defaults
# ─────────────────────────────────────────────
C = dict(bg="#0a0b0d", panel="#14171d", border="#242a33", text="#eef1f6",
         muted="#8a93a3", green="#26d07c", red="#ff5c72", blue="#5b9dff",
         orange="#f5b64c", purple="#bc8cff", cyan="#39d353", yellow="#f5b64c")

CHART = dict(
    paper_bgcolor=C["panel"], plot_bgcolor=C["bg"],
    font=dict(color=C["text"], family="monospace", size=11),
    xaxis=dict(gridcolor=C["border"], showgrid=True, zeroline=False,
               tickfont=dict(color=C["muted"])),
    yaxis=dict(gridcolor=C["border"], showgrid=True, zeroline=False,
               tickfont=dict(color=C["muted"])),
    legend=dict(bgcolor=C["panel"], bordercolor=C["border"],
                borderwidth=1, font=dict(color=C["text"])),
    margin=dict(l=8, r=8, t=30, b=8),
    hovermode="x unified",
)

# ─────────────────────────────────────────────
# Session state & Global Helpers
# ─────────────────────────────────────────────
def _load_token() -> str:
    """DB is the source of truth (survives session resets); .env is the
    one-time seed the first time the app runs against a fresh database."""
    tok = db.get_access_token("")
    if not tok:
        tok = os.getenv("UPSTOX_ACCESS_TOKEN", "")
        if tok:
            db.set_access_token(tok)
    return tok

def _load_watchlist() -> dict:
    default = {"NSE_INDEX|Nifty 50":"NIFTY 50",
               "NSE_INDEX|Nifty Bank":"NIFTY BANK",
               "NSE_INDEX|India VIX":"INDIA VIX"}
    saved = db.get_watchlist()
    if saved:
        return saved
    for key, name in default.items():
        db.add_to_watchlist(name, key)
    return default

def _load_settings() -> dict:
    base = dict(rf_rate=6.5, adx_period=14, rsi_period=14,
                st_multiplier=3.0, st_period=10, theme="Dark")
    saved = db.get_all_settings()
    for k in base:
        if k in saved:
            try: base[k] = type(base[k])(saved[k])
            except (TypeError, ValueError): pass
    return base

def _load_alert_rules() -> list:
    rules = []
    for row in db.get_alerts(status="active"):
        rules.append({
            "id": row["id"], "symbol": row["symbol"], "type": row["alert_type"],
            "value": row["alert_value"], "note": row.get("note",""),
            "active": True, "created": row["created_at"],
        })
    return rules

def _init():
    defaults = dict(
        access_token=_load_token(),
        ws_started=False,
        watchlist=_load_watchlist(),
        selected_chart="NIFTY 50",
        chart_period="1Y",
        auto_refresh=True,
        refresh_interval=3,
        upstox_data=None,
        search_results=[],
        hist_cache={},
        ml_cache={},
        alert_rules=_load_alert_rules(),
        alert_log=[],
        settings=_load_settings(),
    )
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

def hex_to_rgba(hex_color, alpha=0.1):
    """Converts a standard hex color (e.g., '#58a6ff') to an rgba string for Plotly."""
    if str(hex_color).startswith('#') and len(hex_color) == 7:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color

# ─────────────────────────────────────────────
# General Helpers
# ─────────────────────────────────────────────
def get_upstox() -> UpstoxData:
    if st.session_state.upstox_data is None and st.session_state.access_token:
        st.session_state.upstox_data = UpstoxData(st.session_state.access_token)
    return st.session_state.upstox_data

def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5: return False
    t = now.hour * 60 + now.minute
    return 555 <= t <= 930

def get_hist(name: str) -> pd.DataFrame:
    if name not in st.session_state.hist_cache:
        df = load_historical(name)
        if not df.empty:
            df = compute_technicals(df)
            try:
                db.save_historical_df(name, df)
                db.save_technical_indicators(name, df)
            except Exception as e:
                logger.warning(f"[DB] Could not persist {name}: {e}")
        st.session_state.hist_cache[name] = df
    return st.session_state.hist_cache[name]

def live_price(key: str) -> dict:
    return tick_store.get(key) or {}

def merge_live(df: pd.DataFrame, key: str) -> pd.DataFrame:
    tick = live_price(key)
    if not tick or df.empty: return df
    ltp = tick.get("ltp", 0)
    if ltp <= 0: return df
    today = pd.Timestamp(date.today())
    if df["Date"].iloc[-1] >= today: return df
    last = df.iloc[-1]
    new_row = dict(Date=today, Open=last["Close"], High=max(last["Close"],ltp),
                   Low=min(last["Close"],ltp), Close=ltp, Volume=np.nan,
                   Daily_Ret=(ltp-last["Close"])/last["Close"])
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

KEY_MAP = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    "INDIA VIX": "NSE_INDEX|India VIX",
}

def get_key(name): return KEY_MAP.get(name) or \
    {v:k for k,v in st.session_state.watchlist.items()}.get(name,"")

# ─────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────
def _period_df(df, period):
    n = {"1M":21,"3M":63,"6M":126,"1Y":252,"ALL":len(df)}.get(period,252)
    return df.tail(n).copy()

def candlestick_chart(df, name, period="1Y", height=680) -> go.Figure:
    df = _period_df(df, period)
    if df.empty: return go.Figure()

    fig = make_subplots(rows=5, cols=1,
        row_heights=[0.42,0.13,0.15,0.15,0.15],
        shared_xaxes=True, vertical_spacing=0.018,
        subplot_titles=("","Volume","RSI (14)","MACD","ADX"))

    # ── Candles ──
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=C["green"], decreasing_line_color=C["red"],
        increasing_fillcolor=C["green"], decreasing_fillcolor=C["red"],
        name="OHLC", showlegend=False), row=1, col=1)

    # ── EMAs ──
    for span,col,lbl in [(20,C["cyan"],"EMA20"),(50,C["orange"],"EMA50"),(200,C["purple"],"EMA200")]:
        if f"EMA_{span}" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"],y=df[f"EMA_{span}"],
                line=dict(color=col,width=1),name=lbl), row=1, col=1)

    # ── Bollinger ──
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["Date"],df["Date"][::-1]]),
            y=pd.concat([df["BB_Upper"],df["BB_Lower"][::-1]]),
            fill="toself",fillcolor="rgba(88,166,255,0.05)",
            line=dict(color="rgba(0,0,0,0)"),name="BB",showlegend=True), row=1, col=1)
        for col_n,clr in [("BB_Upper",C["blue"]),("BB_Mid",C["blue"]),("BB_Lower",C["blue"])]:
            fig.add_trace(go.Scatter(x=df["Date"],y=df[col_n],
                line=dict(color=clr,width=0.6,dash="dot"),
                name=col_n,showlegend=False), row=1, col=1)

    # ── SuperTrend ──
    if "SuperTrend" in df.columns:
        bull = df[df["ST_Direction"]==1]
        bear = df[df["ST_Direction"]==-1]
        fig.add_trace(go.Scatter(x=bull["Date"],y=bull["SuperTrend"],
            mode="markers",marker=dict(color=C["green"],size=3),name="ST Buy",
            showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=bear["Date"],y=bear["SuperTrend"],
            mode="markers",marker=dict(color=C["red"],size=3),name="ST Sell",
            showlegend=True), row=1, col=1)

    # ── VWAP ──
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["VWAP"],
            line=dict(color=C["yellow"],width=1,dash="dot"),name="VWAP"), row=1, col=1)

    # ── Volume ──
    vol_c = [C["green"] if c>=o else C["red"] for c,o in zip(df["Close"],df["Open"])]
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df["Date"],y=df["Volume"],
            marker_color=vol_c,name="Volume",showlegend=False), row=2, col=1)

    # ── RSI ──
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["RSI"],
            line=dict(color=C["cyan"],width=1.2),name="RSI"), row=3, col=1)
        for lvl,clr in [(70,C["red"]),(30,C["green"]),(50,C["muted"])]:
            fig.add_hline(y=lvl,line_dash="dash",line_color=clr,
                          line_width=0.7,row=3,col=1)
        fig.add_hrect(y0=70,y1=100,fillcolor=hex_to_rgba(C["red"], 0.05),row=3,col=1)
        fig.add_hrect(y0=0,y1=30,fillcolor=hex_to_rgba(C["green"], 0.05),row=3,col=1)

    # ── MACD ──
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["MACD"],
            line=dict(color=C["blue"],width=1.2),name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["Date"],y=df["MACD_Signal"],
            line=dict(color=C["orange"],width=1),name="Signal"), row=4, col=1)
        hc = [C["green"] if v>=0 else C["red"] for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df["Date"],y=df["MACD_Hist"],
            marker_color=hc,name="Hist",showlegend=False), row=4, col=1)

    # ── ADX ──
    if "ADX" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["ADX"],
            line=dict(color=C["yellow"],width=1.2),name="ADX"), row=5, col=1)
        fig.add_trace(go.Scatter(x=df["Date"],y=df["DI_Plus"],
            line=dict(color=C["green"],width=0.9),name="+DI"), row=5, col=1)
        fig.add_trace(go.Scatter(x=df["Date"],y=df["DI_Minus"],
            line=dict(color=C["red"],width=0.9),name="-DI"), row=5, col=1)
        fig.add_hline(y=25,line_dash="dash",line_color=C["cyan"],
                      line_width=0.8,row=5,col=1)

    fig.update_layout(**CHART,
        title=dict(text=f"<b>{name}</b>",font=dict(size=13,color=C["text"])),
        height=height, xaxis_rangeslider_visible=False)
    fig.update_yaxes(row=3,range=[0,100])
    return fig

def vix_gauge_fig(vix_val: float) -> go.Figure:
    steps = [dict(range=[0,15],color="#1a3a2a"),
             dict(range=[15,20],color="#3a3010"),
             dict(range=[20,25],color="#3a2010"),
             dict(range=[25,40],color="#3a1010")]
    label = ("LOW — Calm" if vix_val<15 else "MEDIUM" if vix_val<20
             else "HIGH — Caution" if vix_val<25 else "EXTREME — Fear")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=vix_val,
        title=dict(text=f"India VIX  {label}",
                   font=dict(color=C["muted"],size=11)),
        number=dict(font=dict(color=C["text"],size=34)),
        gauge=dict(axis=dict(range=[0,40],tickcolor=C["muted"],
                             tickfont=dict(color=C["muted"])),
                   bar=dict(color=C["orange"],thickness=0.25),
                   bgcolor=C["panel"],borderwidth=1,bordercolor=C["border"],
                   steps=steps,
                   threshold=dict(line=dict(color=C["red"],width=3),
                                  thickness=0.75,value=25))))
    fig.update_layout(paper_bgcolor=C["panel"],font=dict(color=C["text"]),
                      margin=dict(l=20,r=20,t=40,b=10),height=250)
    return fig

def returns_heatmap(df: pd.DataFrame, name: str) -> go.Figure:
    if df.empty or "Daily_Ret" not in df.columns: return go.Figure()
    df = df.copy()
    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("Month").apply(
        lambda g: (1+g["Daily_Ret"].fillna(0)).prod()-1).reset_index()
    monthly.columns = ["Month","Return"]
    monthly["Year"]  = monthly["Month"].dt.year
    monthly["MonthN"]= monthly["Month"].dt.month
    monthly["Pct"]   = (monthly["Return"]*100).round(2)
    m_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["Label"] = monthly["MonthN"].map(m_labels)
    fig = go.Figure(go.Heatmap(
        x=monthly["Label"], y=monthly["Year"], z=monthly["Pct"],
        text=monthly["Pct"].apply(lambda v: f"{v:+.1f}%"),
        texttemplate="%{text}",
        colorscale=[[0,C["red"]],[0.5,"#1a1a2e"],[1,C["green"]]],
        zmid=0, colorbar=dict(tickfont=dict(color=C["muted"]))))
    fig.update_layout(**CHART,
        title=dict(text=f"<b>{name} — Monthly Returns</b>",
                   font=dict(color=C["text"])),height=200)
    return fig

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark"></span>'
            '<span class="brand-title">MARKET PULSE</span></div>'
            '<div class="brand-sub">NSE Live Terminal</div>',
            unsafe_allow_html=True)
        st.caption("NIFTY 50 · NIFTY BANK · VIX + Live")
        st.markdown("---")

        # Connection
        st.markdown("### 🔌 Upstox Connection")
        tok = st.text_input("Access Token", value=st.session_state.access_token,
                             type="password", placeholder="Paste your token...")
        if tok != st.session_state.access_token:
            db.set_access_token(tok)
        st.session_state.access_token = tok
        c1,c2 = st.columns(2)
        if c1.button("▶ Connect", use_container_width=True, disabled=not tok,
                      type="primary"):
            _start_ws()
        if c2.button("⏹ Stop", use_container_width=True):
            ws_hub.stop(); st.session_state.ws_started = False

        status = ws_hub.status
        dot = ("dot-live" if "Connected" in status
               else "dot-err" if "Error" in status else "dot-off")
        st.markdown(f'<span class="badge"><span class="dot {dot}"></span>{status}</span>',
                    unsafe_allow_html=True)
        if ws_hub.is_running:
            d = ws_hub.diagnostics
            st.caption(f"🔢 {d['subscribed_keys']} subs | 📡 {d['tick_count']} ticks")

        st.markdown("---")

        # Add symbol
        st.markdown("### ➕ Add Symbol")
        q = st.text_input("Search", placeholder="RELIANCE, HDFC, INFY ...",
                           key="search_input")
        if q and len(q) >= 2:
            upstox = get_upstox()
            if upstox:
                with st.spinner("Searching..."):
                    results = upstox.search_instruments(q)
                    st.session_state.search_results = results
            else:
                st.session_state.search_results = []

        for r in st.session_state.search_results[:6]:
            lbl = f"{r['symbol']} ({r['exchange']})"
            if st.button(f"+ {lbl}", key=f"add_{r['instrument_key']}",
                          use_container_width=True):
                st.session_state.watchlist[r['instrument_key']] = r['symbol']
                db.add_to_watchlist(r['symbol'], r['instrument_key'])
                ws_hub.add_subscription(r['instrument_key'])
                st.session_state.search_results = []
                st.rerun()

        # Quick add
        st.markdown("### ⚡ Quick Add")
        popular = {"NIFTY 50":"NSE_INDEX|Nifty 50","NIFTY BANK":"NSE_INDEX|Nifty Bank",
                   "FIN NIFTY":"NSE_INDEX|Nifty Fin Service","NIFTY IT":"NSE_INDEX|Nifty IT",
                   "MIDCAP":"NSE_INDEX|NIFTY MIDCAP 100","SENSEX":"BSE_INDEX|SENSEX"}
        cols = st.columns(2)
        for i,(name,key) in enumerate(popular.items()):
            if cols[i%2].button(name, key=f"q_{key}", use_container_width=True):
                st.session_state.watchlist[key] = name
                db.add_to_watchlist(name, key)
                ws_hub.add_subscription(key); st.rerun()

        st.markdown("---")

        # Chart settings
        st.markdown("### ⚙️ Chart Settings")
        hist_opts = list(HISTORICAL_FILES.keys())
        all_opts  = hist_opts + [v for v in st.session_state.watchlist.values()
                                  if v not in hist_opts]
        idx = all_opts.index(st.session_state.selected_chart) \
              if st.session_state.selected_chart in all_opts else 0
        st.session_state.selected_chart = st.selectbox("Chart Symbol",
                                                         all_opts, index=idx)
        st.session_state.chart_period = st.radio("Period",
            ["1M","3M","6M","1Y","ALL"], index=3, horizontal=True)
        st.session_state.auto_refresh = st.toggle("Auto Refresh", value=True)
        st.session_state.refresh_interval = st.slider(
            "Refresh (sec)", 1, 30, st.session_state.refresh_interval)

def _start_ws():
    for key in st.session_state.watchlist:
        ws_hub.add_subscription(key)
    ws_hub.start(st.session_state.access_token)
    st.session_state.ws_started = True

# ─────────────────────────────────────────────
# Ticker tape
# ─────────────────────────────────────────────
def ticker_tape():
    ticks  = tick_store.get_all()
    items  = ""
    keys   = {**{"NSE_INDEX|Nifty 50":"NIFTY 50",
                  "NSE_INDEX|Nifty Bank":"NIFTY BANK",
                  "NSE_INDEX|India VIX":"INDIA VIX"},
               **st.session_state.watchlist}
    nifty_pct = 0
    for key,name in keys.items():
        tick = ticks.get(key,{})
        ltp  = tick.get("ltp",0)
        pct  = tick.get("change_pct",0)
        ts   = tick.get("timestamp","")
        if ltp == 0 and name in HISTORICAL_FILES:
            df = get_hist(name)
            if not df.empty:
                ltp = df["Close"].iloc[-1]
                pct = df["Pct_Change"].iloc[-1] if "Pct_Change" in df.columns else 0
        if name == "NIFTY 50":
            nifty_pct = pct
        if ltp == 0: continue
        sign = "▲" if pct>=0 else "▼"
        cls  = "tu" if pct>=0 else "td"
        items += (f'<div class="ti"><span class="tn">{name}</span>'
                  f'<span class="tp">₹{ltp:,.2f}</span>'
                  f'<span class="{cls}">{sign} {abs(pct):.2f}% {ts}</span></div>')
    mkt_open = is_market_open()
    mkt_dot  = "dot-live" if mkt_open else "dot-off"
    mkt_lbl  = "MARKET OPEN" if mkt_open else "MARKET CLOSED"
    now_s = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    pulse_cls = "pulse-up" if nifty_pct >= 0 else "pulse-down"
    st.markdown(f'<div class="pulse-bar {pulse_cls}"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ticker">{items}'
                f'<div class="ti" style="margin-left:auto; align-items:flex-end">'
                f'<span class="tn">{now_s}</span>'
                f'<span class="badge" style="margin-top:3px">'
                f'<span class="dot {mkt_dot}"></span>{mkt_lbl}</span></div></div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────
def kpi_row():
    cols = st.columns(3)
    for col,(key,hist_name) in zip(cols,[
        ("NSE_INDEX|Nifty 50","NIFTY 50"),
        ("NSE_INDEX|Nifty Bank","NIFTY BANK"),
        ("NSE_INDEX|India VIX","INDIA VIX")]):
        tick = live_price(key)
        df   = get_hist(hist_name)
        ltp = tick.get("ltp",0) if tick and tick.get("ltp",0)>0 else \
              (df["Close"].iloc[-1] if not df.empty else 0)
        pct = tick.get("change_pct",0) if tick and tick.get("ltp",0)>0 else \
              (df["Pct_Change"].iloc[-1] if not df.empty and "Pct_Change" in df.columns else 0)
        col.metric(hist_name, f"₹{ltp:,.2f}" if ltp else "—", f"{pct:+.2f}%")

# ══════════════════════════════════════════════════════════════
# TAB RENDERERS
# ══════════════════════════════════════════════════════════════

# ── Tab 1: Market Overview ────────────────────────────────
def tab_market_overview():
    st.markdown('<div class="sec">📊 Market Overview</div>', unsafe_allow_html=True)

    df50  = get_hist("NIFTY 50")
    dfbnk = get_hist("NIFTY BANK")
    dfvix = get_hist("INDIA VIX")

    if df50.empty:
        st.info("Load NIFTY 50 CSV into data/ folder.")
        return

    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                         vertical_spacing=0.04,
                         row_heights=[0.38,0.38,0.24])
                         
    for row,df,color,name in [
        (1,df50, C["blue"],"NIFTY 50"),
        (2,dfbnk,C["purple"],"NIFTY BANK")]:
        
        # Convert the hex color to RGBA for the fill
        fill_rgba = hex_to_rgba(color, 0.1)
        
        fig.add_trace(go.Scatter(x=df["Date"],y=df["Close"],
            line=dict(color=color,width=1.4),name=name,
            fill="tozeroy",fillcolor=fill_rgba), row=row,col=1)
            
        ytd = (df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100
        fig.add_annotation(xref="paper",yref=f"y{row}",x=0.99,
            y=df["Close"].iloc[-1],
            text=f"{ytd:+.2f}% YTD",
            font=dict(color=C["green"] if ytd>=0 else C["red"],size=10),
            showarrow=False, row=row, col=1)

    if not dfvix.empty:
        # Convert VIX color
        vix_fill = hex_to_rgba(C["orange"], 0.1)
        
        fig.add_trace(go.Scatter(x=dfvix["Date"],y=dfvix["Close"],
            line=dict(color=C["orange"],width=1.4),name="India VIX",
            fill="tozeroy",fillcolor=vix_fill), row=3,col=1)
            
        for lvl,clr in [(15,C["green"]),(20,C["yellow"]),(25,C["red"])]:
            fig.add_hline(y=lvl,line_dash="dash",line_color=clr,
                          line_width=0.8,row=3,col=1)

    fig.update_layout(**CHART,height=560,
        title="<b>NIFTY 50 · NIFTY BANK · India VIX  —  1 Year</b>")
    
    st.plotly_chart(fig, width="stretch")

    # Correlation
    st.markdown('<div class="sec">Correlation Matrix</div>', unsafe_allow_html=True)
    frames = {}
    for nm in ["NIFTY 50","NIFTY BANK","INDIA VIX"]:
        d = get_hist(nm)
        if not d.empty:
            frames[nm] = d.set_index("Date")["Close"]
            
    if len(frames) >= 2:
        corr = pd.DataFrame(frames).pct_change().corr().round(3)
        fig2 = go.Figure(go.Heatmap(
            z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),
            text=corr.values.round(2),texttemplate="%{text}",
            colorscale=[[0,C["red"]],[0.5,"#1a1a2e"],[1,C["green"]]],
            zmid=0,zmin=-1,zmax=1))
        fig2.update_layout(**CHART,height=220,title="<b>Return Correlation</b>")
        st.plotly_chart(fig2, width="stretch")

# ── Tab 2: Live Market ────────────────────────────────────
def tab_live_market():
    st.markdown('<div class="sec">⚡ Live Market Feed</div>', unsafe_allow_html=True)

    all_ticks = tick_store.get_all()
    if not all_ticks:
        st.info("Connect to Upstox WebSocket to see live data. "
                "While offline, showing last historical close.")
        all_ticks = {}

    rows = []
    for key,name in st.session_state.watchlist.items():
        tick = all_ticks.get(key,{})
        ltp  = tick.get("ltp",0)
        pct  = tick.get("change_pct",0)
        chg  = tick.get("change",0)
        ts   = tick.get("timestamp","—")
        # Fallback
        if ltp==0 and name in HISTORICAL_FILES:
            df = get_hist(name)
            if not df.empty:
                ltp = df["Close"].iloc[-1]
                pct = df["Pct_Change"].iloc[-1] if "Pct_Change" in df.columns else 0
                chg = df["Close"].iloc[-1] - df["Close"].iloc[-2] if len(df)>1 else 0
                ts  = df["Date"].iloc[-1].strftime("Last: %d %b")
        rows.append(dict(Symbol=name, LTP=f"₹{ltp:,.2f}", Change=f"{chg:+.2f}",
                         Chg_Pct=f"{pct:+.2f}%", Time=ts, Key=key))

    df_tbl = pd.DataFrame(rows)
    if not df_tbl.empty:
        st.dataframe(df_tbl.drop(columns=["Key"]), use_container_width=True, hide_index=True)

    if ws_hub.is_running:
        st.json(ws_hub.diagnostics)

    # Watchlist manage
    st.markdown('<div class="sec">Manage Watchlist</div>', unsafe_allow_html=True)
    remove_keys = []
    for key,name in list(st.session_state.watchlist.items()):
        c1,c2 = st.columns([8,1])
        c1.write(f"**{name}** `{key}`")
        if key not in KEY_MAP.values():
            if c2.button("✕", key=f"rm_{key}"):
                remove_keys.append(key)
    for k in remove_keys:
        del st.session_state.watchlist[k]
        db.remove_from_watchlist(k)
        ws_hub.remove_subscription(k)
    if remove_keys: st.rerun()

# ── Tab 3: OHLC Chart ─────────────────────────────────────
def tab_ohlc_chart():
    selected = st.session_state.selected_chart
    period   = st.session_state.chart_period
    key      = get_key(selected)

    df = get_hist(selected) if selected in HISTORICAL_FILES else pd.DataFrame()
    if df.empty:
        st.info(f"No data for {selected}.")
        return
    df = merge_live(df, key)

    # Indicator toggle
    c1,c2,c3,c4 = st.columns(4)
    show_bb = c1.checkbox("Bollinger", value=True)
    show_ema = c2.checkbox("EMAs", value=True)
    show_st = c3.checkbox("SuperTrend", value=True)
    show_vwap = c4.checkbox("VWAP", value=True)

    st.plotly_chart(candlestick_chart(df, selected, period), width="stretch")
    st.plotly_chart(returns_heatmap(df, selected), width="stretch")

# ── Tab 4: Technical Indicators ──────────────────────────
def tab_technicals():
    selected = st.session_state.selected_chart
    df = get_hist(selected) if selected in HISTORICAL_FILES else pd.DataFrame()
    if df.empty: st.info("No data."); return
    df = _period_df(df, st.session_state.chart_period)

    latest = df.iloc[-1]

    # ── Signal cards ──
    st.markdown('<div class="sec">Latest Signals</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    
    # RSI
    rsi_v = latest.get("RSI", 50)
    rsi_s = "Overbought 🔴" if rsi_v>70 else ("Oversold 🟢" if rsi_v<30 else "Neutral 🟡")
    cols[0].metric("RSI (14)", f"{rsi_v:.1f}", rsi_s)
    
    # ADX
    adx_v = latest.get("ADX", 0)
    cols[1].metric("ADX (14)", f"{adx_v:.1f}", adx_signal(adx_v))
    
    # SuperTrend
    st_s  = latest.get("ST_Signal","—")
    cols[2].metric("SuperTrend", st_s,
                   f"+DI={latest.get('DI_Plus',0):.1f}  -DI={latest.get('DI_Minus',0):.1f}")
                   
    # VWAP
    vwap_v = latest.get("VWAP", 0)
    vwap_s = "Above VWAP 🟢" if latest["Close"]>vwap_v else "Below VWAP 🔴"
    cols[3].metric("VWAP", f"₹{vwap_v:,.2f}", vwap_s)

    st.markdown("")
    cols2 = st.columns(4)
    cols2[0].metric("MACD", f"{latest.get('MACD',0):.2f}",
                    "Bullish" if latest.get("MACD",0)>latest.get("MACD_Signal",0) else "Bearish")
    cols2[1].metric("ATR (14)", f"{latest.get('ATR',0):.2f}",
                    f"{latest.get('ATR_Pct',0):.2f}% of price")
    cols2[2].metric("BB Width", f"{latest.get('BB_Width',0):.2f}%",
                    "Squeeze" if latest.get("BB_Width",0)<2 else "Normal")
    cols2[3].metric("Market Regime", latest.get("Market_Regime","—"), "")

    st.markdown("---")

    # ── ADX Deep Dive ──
    st.markdown('<div class="sec">ADX — Trend Strength</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05,
                         row_heights=[0.55,0.45])
    fig.add_trace(go.Scatter(x=df["Date"],y=df["Close"],
        line=dict(color=C["text"],width=1),name="Close"), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Date"],y=df["ADX"],
        line=dict(color=C["yellow"],width=1.3),name="ADX"), row=2,col=1)
    fig.add_trace(go.Scatter(x=df["Date"],y=df["DI_Plus"],
        line=dict(color=C["green"],width=0.9),name="+DI"), row=2,col=1)
    fig.add_trace(go.Scatter(x=df["Date"],y=df["DI_Minus"],
        line=dict(color=C["red"],width=0.9),name="-DI"), row=2,col=1)
        
    for lvl,clr,lbl in [(20,C["muted"],"Ranging"),(25,C["cyan"],"Trend"),
                         (40,C["orange"],"Strong")]:
        fig.add_hline(y=lvl,line_dash="dash",line_color=clr,line_width=0.7,
                      annotation_text=lbl,row=2,col=1)
                      
    fig.update_layout(**CHART,height=380,title="<b>ADX / +DI / -DI</b>")
    st.plotly_chart(fig, width="stretch")

    # ── SuperTrend ──
    st.markdown('<div class="sec">SuperTrend (ATR 10, Mult 3)</div>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Date"],y=df["Close"],
        line=dict(color=C["text"],width=1.2),name="Close"))
        
    if "SuperTrend" in df.columns:
        bull = df[df["ST_Direction"]==1]
        bear = df[df["ST_Direction"]==-1]
        fig2.add_trace(go.Scatter(x=bull["Date"],y=bull["SuperTrend"],
            mode="lines",line=dict(color=C["green"],width=1.5,dash="solid"),
            name="SuperTrend Buy"))
        fig2.add_trace(go.Scatter(x=bear["Date"],y=bear["SuperTrend"],
            mode="lines",line=dict(color=C["red"],width=1.5,dash="solid"),
            name="SuperTrend Sell"))
            
    fig2.update_layout(**CHART,height=320,title="<b>SuperTrend Signal</b>")
    st.plotly_chart(fig2, width="stretch")

    # ── VWAP ──
    st.markdown('<div class="sec">VWAP</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["Date"],y=df["Close"],
        line=dict(color=C["blue"],width=1.2),name="Close"))
        
    if "VWAP" in df.columns:
        vwap_band_color = hex_to_rgba(C["yellow"], 0.1)
        fig3.add_trace(go.Scatter(x=df["Date"],y=df["VWAP"],
            line=dict(color=C["yellow"],width=1.5,dash="dot"),name="VWAP"))
            
        fig3.add_trace(go.Scatter(
            x=pd.concat([df["Date"],df["Date"][::-1]]),
            y=pd.concat([df["VWAP"]*1.01, df["VWAP"][::-1]*0.99]),
            fill="toself", fillcolor=vwap_band_color,
            line=dict(color="rgba(0,0,0,0)"),name="VWAP Band"))
            
    fig3.update_layout(**CHART,height=280,title="<b>VWAP — Volume Weighted Avg Price</b>")
    st.plotly_chart(fig3, width="stretch")

# ── Tab 5: Risk Analytics ─────────────────────────────────
def tab_risk():
    st.markdown('<div class="sec">⚠️ Risk Analytics</div>', unsafe_allow_html=True)
    tabs = st.tabs(["NIFTY 50","NIFTY BANK","Comparison","Risk Score"])

    for tab,name in zip(tabs[:2],["NIFTY 50","NIFTY BANK"]):
        with tab:
            df = get_hist(name)
            if df.empty: st.info("Load CSV."); continue
            metrics = compute_risk_metrics(df)
            pairs = list(metrics.items())
            cols  = st.columns(5)
            for i,(label,val) in enumerate(pairs[:10]):
                cols[i%5].metric(label, val)
            # Drawdown
            ret = df["Daily_Ret"].dropna()
            cum = (1+ret).cumprod()
            dd  = ((cum-cum.cummax())/cum.cummax())*100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"].iloc[1:],y=dd.values,
                fill="tozeroy",fillcolor=hex_to_rgba(C["red"], 0.3),
                line=dict(color=C["red"],width=1),name="Drawdown"))
            fig.add_hline(y=dd.min(),line_dash="dash",line_color=C["yellow"],
                          annotation_text=f"Max DD: {dd.min():.2f}%",
                          annotation_font_color=C["yellow"])
            fig.update_layout(**CHART,height=220,title="<b>Drawdown (%)</b>")
            st.plotly_chart(fig, width="stretch")
            
            # Cumulative return
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["Date"].iloc[1:],y=(cum-1)*100,
                line=dict(color=C["blue"],width=1.4),
                fill="tozeroy",fillcolor=hex_to_rgba(C["blue"], 0.15),name="Cum Return"))
            fig2.add_hline(y=0,line_color=C["muted"],line_width=0.8)
            fig2.update_layout(**CHART,height=200,title="<b>Cumulative Return (%)</b>")
            st.plotly_chart(fig2, width="stretch")

    with tabs[2]:
        rows2 = []
        for nm in ["NIFTY 50","NIFTY BANK"]:
            df = get_hist(nm)
            m  = compute_risk_metrics(df) if not df.empty else {}
            row = {"Index":nm}; row.update(m); rows2.append(row)
        st.dataframe(pd.DataFrame(rows2).set_index("Index").T, use_container_width=True)

    with tabs[3]:
        st.markdown("#### Combined Intraday Risk Score")
        df50  = get_hist("NIFTY 50")
        dfvix = get_hist("INDIA VIX")
        vix_v = dfvix["Close"].iloc[-1] if not dfvix.empty else 17.0
        breadth = get_market_breadth(df50)
        chain   = generate_option_chain(df50["Close"].iloc[-1] if not df50.empty else 24000)
        pcr_v   = float(chain["PCR"].iloc[0]) if "PCR" in chain.columns else 1.0
        rs      = compute_risk_score(df50, vix_v, breadth, pcr_v)

        score = rs["score"]
        level = rs["level"]
        color = C["green"] if score<40 else C["yellow"] if score<65 else C["red"]

        c1,c2 = st.columns([1,2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title=dict(text=level,font=dict(color=color,size=12)),
                number=dict(font=dict(color=color,size=40)),
                gauge=dict(axis=dict(range=[0,100]),
                           bar=dict(color=color,thickness=0.3),
                           bgcolor=C["panel"],bordercolor=C["border"],
                           steps=[dict(range=[0,40],color="#1a3a2a"),
                                  dict(range=[40,65],color="#3a3010"),
                                  dict(range=[65,100],color="#3a1a1a")])))
            fig.update_layout(paper_bgcolor=C["panel"],height=230,
                              margin=dict(l=15,r=15,t=30,b=5))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown(f"**Advice:** {rs['advice']}")
            st.markdown("**Pillars (0–100, lower = safer):**")
            pillars = rs["pillars"]
            for k,v in pillars.items():
                bar_color = "green" if v<40 else ("orange" if v<65 else "red")
                st.progress(int(v), text=f"{k.upper()}  {v:.0f}/100")

# ── Tab 6: Gap Analysis ───────────────────────────────────
def tab_gap():
    st.markdown('<div class="sec">🕳️ Gap Analysis</div>', unsafe_allow_html=True)
    selected = st.session_state.selected_chart
    df = get_hist(selected) if selected in HISTORICAL_FILES else pd.DataFrame()
    if df.empty: st.info("No data."); return

    gap_df = df.dropna(subset=["Gap_Pct"]) if "Gap_Pct" in df.columns else pd.DataFrame()
    if gap_df.empty: st.info("Gap data not computed."); return

    # Stats
    gaps_only = gap_df[gap_df["Gap_Type"]!="Flat"]
    up_gaps   = gap_df[gap_df["Gap_Type"]=="Gap Up"]
    dn_gaps   = gap_df[gap_df["Gap_Type"]=="Gap Down"]
    fill_rate = gaps_only["Gap_Fill"].mean()*100 if not gaps_only.empty else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Gap Up Days",    len(up_gaps))
    c2.metric("Gap Down Days",  len(dn_gaps))
    c3.metric("Gap Fill Rate",  f"{fill_rate:.1f}%")
    c4.metric("Avg Gap %",
              f"{gaps_only['Gap_Pct'].mean():.2f}%" if not gaps_only.empty else "—")

    # Scatter
    fig = go.Figure()
    colors_map = {"Gap Up":C["green"],"Gap Down":C["red"],"Flat":C["muted"]}
    for gt,col in colors_map.items():
        subset = gap_df[gap_df["Gap_Type"]==gt]
        fig.add_trace(go.Scatter(
            x=subset["Date"],y=subset["Gap_Pct"],mode="markers",
            marker=dict(color=col,size=5,opacity=0.8),name=gt))
    fig.add_hline(y=0,line_color=C["muted"],line_width=0.8)
    fig.update_layout(**CHART,height=280,title=f"<b>{selected} — Gap % per Session</b>")
    st.plotly_chart(fig, width="stretch")

    # Distribution
    fig2 = px.histogram(gap_df, x="Gap_Pct", nbins=40,
        color_discrete_sequence=[C["blue"]], title="Gap % Distribution")
    fig2.update_layout(**CHART,height=240)
    st.plotly_chart(fig2, width="stretch")

    # Table (recent 20 gaps)
    st.markdown('<div class="sec">Recent Gaps (Top 20)</div>', unsafe_allow_html=True)
    show_cols = ["Date","Open","Close","Gap_Pct","Gap_Type","Gap_Fill"]
    show_cols = [c for c in show_cols if c in gap_df.columns]
    st.dataframe(gaps_only[show_cols].sort_values("Date",ascending=False).head(20),
                 use_container_width=True, hide_index=True)

# ── Tab 7: Volatility ─────────────────────────────────────
def tab_volatility():
    st.markdown('<div class="sec">📉 Volatility Analysis</div>',
                unsafe_allow_html=True)
    df50  = get_hist("NIFTY 50")
    dfbnk = get_hist("NIFTY BANK")
    dfvix = get_hist("INDIA VIX")

    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.04,
                         row_heights=[0.35,0.35,0.30])
    for row,df,color,name in [(1,df50,C["blue"],"NIFTY 50"),
                               (2,dfbnk,C["purple"],"NIFTY BANK")]:
        if df.empty: continue
        rv = df["RollingVol_20"] if "RollingVol_20" in df.columns else pd.Series()
        if not rv.empty:
            fig.add_trace(go.Scatter(x=df["Date"],y=rv,
                line=dict(color=color,width=1.4),name=f"{name} Vol 20d",
                fill="tozeroy",fillcolor=hex_to_rgba(color, 0.15)), row=row,col=1)
            fig.add_hline(y=rv.mean(),line_dash="dash",line_color=C["yellow"],
                          line_width=0.8,annotation_text=f"Mean {rv.mean():.1f}%",
                          row=row,col=1)

    if not dfvix.empty:
        fig.add_trace(go.Scatter(x=dfvix["Date"],y=dfvix["Close"],
            line=dict(color=C["orange"],width=1.4),name="India VIX",
            fill="tozeroy",fillcolor=hex_to_rgba(C["orange"], 0.15)), row=3,col=1)
        for lvl,clr in [(15,C["green"]),(20,C["yellow"]),(25,C["red"])]:
            fig.add_hline(y=lvl,line_dash="dash",line_color=clr,
                          line_width=0.7,row=3,col=1)

    fig.update_layout(**CHART,height=540,
        title="<b>Rolling 20d Annualised Volatility + India VIX</b>")
    st.plotly_chart(fig, width="stretch")

    # ATR
    st.markdown('<div class="sec">ATR — Average True Range</div>',
                unsafe_allow_html=True)
    if not df50.empty and "ATR" in df50.columns:
        fig2 = make_subplots(rows=2,cols=1,shared_xaxes=True,
                              vertical_spacing=0.04,row_heights=[0.55,0.45])
        fig2.add_trace(go.Scatter(x=df50["Date"],y=df50["Close"],
            line=dict(color=C["blue"],width=1.2),name="Close"), row=1,col=1)
        fig2.add_trace(go.Scatter(x=df50["Date"],y=df50["ATR"],
            line=dict(color=C["orange"],width=1.3),fill="tozeroy",
            fillcolor=hex_to_rgba(C["orange"], 0.15),name="ATR 14"), row=2,col=1)
        fig2.add_hline(y=df50["ATR"].mean(),line_dash="dash",line_color=C["cyan"],
                       annotation_text=f"Mean ATR {df50['ATR'].mean():.0f}",
                       row=2,col=1)
        fig2.update_layout(**CHART,height=360,
            title="<b>NIFTY 50 — ATR (Average True Range)</b>")
        st.plotly_chart(fig2, width="stretch")

# ── Tab 8: India VIX ─────────────────────────────────────
def tab_vix():
    st.markdown('<div class="sec">🌡️ India VIX Dashboard</div>',
                unsafe_allow_html=True)
    dfvix = get_hist("INDIA VIX")
    df50  = get_hist("NIFTY 50")
    tick  = live_price("NSE_INDEX|India VIX")
    vix_now = (tick.get("ltp",0) if tick and tick.get("ltp",0)>0
               else (dfvix["Close"].iloc[-1] if not dfvix.empty else 17.0))

    c1,c2 = st.columns([1,2])
    with c1:
        st.plotly_chart(vix_gauge_fig(vix_now), width="stretch")
        if not dfvix.empty:
            v = dfvix["Close"]
            low  = (v<15).sum(); med = ((v>=15)&(v<20)).sum()
            high = ((v>=20)&(v<25)).sum(); ext = (v>=25).sum()
            tot  = len(v)
            for lbl,cnt,clr in [("🟢 Low (<15)",low,C["green"]),
                                 ("🟡 Med (15-20)",med,C["yellow"]),
                                 ("🔴 High (20-25)",high,C["orange"]),
                                 ("💥 Extreme (>25)",ext,C["red"])]:
                st.markdown(f"<span style='color:{clr}'>{lbl}: "
                            f"**{cnt}** ({cnt/tot*100:.0f}%)</span>",
                            unsafe_allow_html=True)
    with c2:
        if not dfvix.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dfvix["Date"],y=dfvix["Close"],
                line=dict(color=C["orange"],width=1.5),
                fill="tozeroy",fillcolor=hex_to_rgba(C["orange"], 0.15),name="VIX"))
            fig.add_trace(go.Scatter(x=dfvix["Date"],
                y=dfvix["Close"].rolling(20).mean(),
                line=dict(color=C["cyan"],width=1.2,dash="dash"),name="MA 20"))
            for lvl,clr in [(15,C["green"]),(20,C["yellow"]),(25,C["red"])]:
                fig.add_hline(y=lvl,line_dash="dash",line_color=clr,line_width=0.7)
            fig.update_layout(**CHART,height=320,title="<b>India VIX — 1 Year</b>")
            st.plotly_chart(fig, width="stretch")

    if not dfvix.empty and not df50.empty:
        st.markdown('<div class="sec">VIX vs NIFTY 50 Returns</div>',
                    unsafe_allow_html=True)
        merged = pd.merge(df50[["Date","Daily_Ret"]],
                          dfvix[["Date","Close"]].rename(columns={"Close":"VIX"}),
                          on="Date",how="inner").dropna()
        fig2 = px.scatter(merged, x="VIX", y="Daily_Ret",
            color="Daily_Ret",
            color_continuous_scale=[[0,C["red"]],[0.5,"white"],[1,C["green"]]],
            labels={"VIX":"India VIX","Daily_Ret":"NIFTY 50 Daily Return"},
            title="<b>VIX vs Return Scatter</b>")
        fig2.update_layout(**CHART,height=300,coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")

# ── Tab 9: Option Chain ───────────────────────────────────
def tab_option_chain():
    st.markdown('<div class="sec">🔗 Option Chain Analytics</div>',
                unsafe_allow_html=True)
    df50 = get_hist("NIFTY 50")
    spot = df50["Close"].iloc[-1] if not df50.empty else 24000.0

    col1,col2,col3 = st.columns(3)
    spot    = col1.number_input("Spot Price (₹)", value=float(spot), step=50.0)
    dte     = col2.number_input("Days to Expiry", value=7, min_value=1, max_value=90)
    iv_base = col3.slider("IV Base (%)", 10, 40, 18) / 100

    chain   = generate_option_chain(spot, dte, iv_base)
    max_pain_strike = compute_max_pain(chain)
    total_call_oi   = chain["Call_OI"].sum()
    total_put_oi    = chain["Put_OI"].sum()
    pcr_v           = total_put_oi / total_call_oi if total_call_oi else 1.0
    pcr_lbl, pcr_clr = pcr_signal(pcr_v)
    atm_strike = round(spot/50)*50

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Spot",         f"₹{spot:,.0f}")
    c2.metric("ATM Strike",   f"₹{atm_strike:,.0f}")
    c3.metric("Max Pain",     f"₹{max_pain_strike:,.0f}")
    c4.metric("PCR",          f"{pcr_v:.3f}", pcr_lbl)
    c5.metric("IV Base",      f"{iv_base*100:.1f}%")

    st.markdown(f"**PCR Signal:** <span style='color:{C[pcr_clr]};font-weight:700'>"
                f"{pcr_lbl}</span>  —  "
                f"{'Heavy put buying → Bearish/Hedge' if pcr_v>1.2 else 'Heavy call writing → Bullish' if pcr_v<0.7 else 'Balanced market'}",
                unsafe_allow_html=True)

    # OI bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chain["Strike"],y=chain["Call_OI"],
        name="Call OI",marker_color=C["red"],opacity=0.8))
    fig.add_trace(go.Bar(x=chain["Strike"],y=-chain["Put_OI"],
        name="Put OI",marker_color=C["green"],opacity=0.8))
    fig.add_vline(x=spot,line_dash="dash",line_color=C["cyan"],
                  annotation_text="Spot",annotation_font_color=C["cyan"])
    fig.add_vline(x=max_pain_strike,line_dash="dot",line_color=C["yellow"],
                  annotation_text="Max Pain",annotation_font_color=C["yellow"])
    fig.update_layout(**CHART,height=320,barmode="overlay",
        title="<b>Open Interest — Calls (↑) vs Puts (↓)</b>")
    st.plotly_chart(fig, width="stretch")

    # IV Smile
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=chain["Strike"],y=chain["Call_IV"],
        line=dict(color=C["red"],width=1.5),name="Call IV"))
    fig2.add_trace(go.Scatter(x=chain["Strike"],y=chain["Put_IV"],
        line=dict(color=C["green"],width=1.5),name="Put IV"))
    fig2.add_vline(x=spot,line_dash="dash",line_color=C["cyan"])
    fig2.update_layout(**CHART,height=260,title="<b>IV Smile / Skew</b>",
        yaxis_title="Implied Volatility (%)")
    st.plotly_chart(fig2, width="stretch")

    # Greeks table
    st.markdown('<div class="sec">Option Chain Table (with Greeks)</div>',
                unsafe_allow_html=True)
    display_cols = ["Strike","Call_Price","Call_OI","Call_IV","Call_Delta",
                    "Call_Gamma","Call_Theta","Call_Vega",
                    "Put_Price","Put_OI","Put_IV","Put_Delta","Put_Theta"]
    display_cols = [c for c in display_cols if c in chain.columns]
    st.dataframe(chain[display_cols].set_index("Strike"), use_container_width=True)

# ── Tab 10: Market Breadth ────────────────────────────────
def tab_breadth():
    st.markdown('<div class="sec">📊 Market Breadth</div>', unsafe_allow_html=True)
    df50 = get_hist("NIFTY 50")
    if df50.empty: st.info("Load NIFTY 50 data."); return

    breadth = get_market_breadth(df50)
    adv = breadth["advancing"]; dec = breadth["declining"]
    unch= breadth["unchanged"]; tot = breadth["total"]

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Advancing",     f"{adv}/{tot}",
              f"{adv/tot*100:.0f}%")
    c2.metric("Declining",     f"{dec}/{tot}",
              f"-{dec/tot*100:.0f}%")
    c3.metric("Unchanged",     unch)
    c4.metric("A/D Ratio",     f"{breadth['ad_ratio']:.2f}",
              "Bullish" if breadth["ad_ratio"]>1 else "Bearish")
    c5.metric("New 52W H/L",
              f"{breadth['new_highs']}/{breadth['new_lows']}",
              f"NH-NL: {breadth['new_highs']-breadth['new_lows']:+d}")

    st.markdown(f"**Breadth Signal:** `{breadth['breadth_signal']}`")
    st.markdown("")

    col1,col2 = st.columns([1,2])
    with col1:
        # Donut chart
        fig = go.Figure(go.Pie(
            labels=["Advancing","Declining","Unchanged"],
            values=[adv,dec,unch],
            hole=0.55,
            marker_colors=[C["green"],C["red"],C["muted"]],
            textfont=dict(color=C["text"])))
        fig.update_layout(paper_bgcolor=C["panel"],height=260,
            legend=dict(font=dict(color=C["text"])),
            margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, width="stretch")

    with col2:
        # AD Line
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df50["Date"].tolist(), y=breadth["ad_line"],
            line=dict(color=C["blue"],width=1.3),
            fill="tozeroy",fillcolor=hex_to_rgba(C["blue"], 0.15),name="AD Line"))
        fig2.add_hline(y=0,line_color=C["muted"],line_width=0.8)
        fig2.update_layout(**CHART,height=260,title="<b>Advance–Decline Line</b>")
        st.plotly_chart(fig2, width="stretch")

    # Sector heatmap
    st.markdown('<div class="sec">Sector Performance</div>', unsafe_allow_html=True)
    sectors = breadth["sectors"]
    sec_df  = pd.DataFrame(list(sectors.items()),columns=["Sector","Return %"])
    fig3 = go.Figure(go.Bar(
        x=sec_df["Sector"], y=sec_df["Return %"],
        marker_color=[C["green"] if v>=0 else C["red"] for v in sec_df["Return %"]],
        text=sec_df["Return %"].apply(lambda v:f"{v:+.2f}%"),
        textposition="outside"))
    fig3.update_layout(**CHART,height=300,title="<b>Sector Performance (Today, Simulated)</b>")
    st.plotly_chart(fig3, width="stretch")

# ── Tab 11: Performance Dashboard ────────────────────────
# ── Tab 11: Performance Dashboard ────────────────────────
def tab_performance():
    st.markdown('<div class="sec">🏆 Performance Dashboard</div>',
                unsafe_allow_html=True)
    for name in ["NIFTY 50","NIFTY BANK"]:
        df = get_hist(name)
        if df.empty: continue
        st.markdown(f"#### {name}")
        m = compute_risk_metrics(df)
        cols = st.columns(5)
        for i,(k,v) in enumerate(list(m.items())[:10]):
            cols[i%5].metric(k,v)
            
        # FIX: Added a unique key to prevent duplicate ID collisions
        st.plotly_chart(returns_heatmap(df,name), width="stretch", key=f"perf_heatmap_{name}")
        
        st.markdown("---")

# ── Tab 12: ML Predictions ────────────────────────────────
def tab_ml():
    st.markdown('<div class="sec">🤖 Machine Learning Predictions</div>',
                unsafe_allow_html=True)
    from ml_models import full_training_report, predict_tomorrow, predict_today

    df50  = get_hist("NIFTY 50")
    dfvix = get_hist("INDIA VIX")
    if df50.empty: st.info("Load NIFTY 50 data."); return

    vix_series = dfvix["Close"].reset_index(drop=True) if not dfvix.empty else None

    col1,col2 = st.columns(2)
    run_now  = col1.button("▶ Run Full ML Report (slow ~30s)", type="primary")
    run_pred = col2.button("⚡ Quick Predict Only (~5s)")

    if run_pred:
        with st.spinner("Predicting ..."):
            pred = predict_tomorrow(df50, vix_series)
        st.session_state.ml_cache["pred"] = pred
        try:
            db.save_prediction("NIFTY 50", pred, model_name="ensemble_vote")
        except Exception as e:
            logger.warning(f"[DB] Could not save prediction: {e}")

    if run_now:
        with st.spinner("Training all models — please wait (~30s) ..."):
            report = full_training_report(df50, vix_series)
        st.session_state.ml_cache["report"] = report

    # ── Tomorrow's prediction ──
    pred = st.session_state.ml_cache.get("pred") or \
           st.session_state.ml_cache.get("report",{}).get("prediction",{})

    if pred and "error" not in pred:
        st.markdown("### 🔮 Tomorrow's Prediction")
        direction = pred["direction"]
        confidence= pred["confidence"]
        probs     = pred["probabilities"]
        col1,col2,col3 = st.columns(3)

        icon = "📈" if direction=="Up" else ("📉" if direction=="Down" else "↔️")
        dir_color = C["green"] if direction=="Up" else (C["red"] if direction=="Down" else C["yellow"])

        col1.markdown(f"""
        <div style='background:{C["panel"]};border:1px solid {dir_color};
          border-radius:10px;padding:16px;text-align:center'>
          <div style='font-size:36px'>{icon}</div>
          <div style='font-size:22px;font-weight:700;color:{dir_color}'>{direction}</div>
          <div style='color:{C["muted"]};font-size:11px'>Consensus {confidence:.0f}%</div>
        </div>""", unsafe_allow_html=True)

        col2.markdown("**Model Votes:**")
        for model,vote in pred.get("model_votes",{}).items():
            vc = C["green"] if vote=="Up" else (C["red"] if vote=="Down" else C["yellow"])
            muted = C["muted"]
            col2.markdown(f"<span style='color:{muted}'>{model}</span> → "
                          f"<span style='color:{vc};font-weight:700'>{vote}</span>",
                          unsafe_allow_html=True)

        col3.markdown("**Probability:**")
        for cls,prob in probs.items():
            pc = C["green"] if cls=="Up" else (C["red"] if cls=="Down" else C["yellow"])
            col3.progress(int(prob), text=f"{cls}  {prob:.1f}%")

        if pred.get("price_forecast"):
            st.info(pred["signal"])

        # Feature importance
        if pred.get("feature_importance"):
            st.markdown("**Top Feature Importance:**")
            fi = pred["feature_importance"]
            fig = go.Figure(go.Bar(
                x=list(fi.values()), y=list(fi.keys()),
                orientation="h",
                marker_color=C["blue"]))
            fig.update_layout(**CHART,height=280,
                title="<b>Feature Importance (avg across models)</b>",
                xaxis_title="Importance (%)")
            st.plotly_chart(fig, width="stretch")

    elif pred and "error" in pred:
        st.error(pred["error"])
    else:
        st.info("Click **Run Full ML Report** or **Quick Predict Only** to generate predictions.")

    # ── Model evaluation results ──
    report = st.session_state.ml_cache.get("report",{})
    if report and "error" not in report:
        st.markdown("### 📊 Model Evaluation (Walk-Forward CV)")
        tabs = st.tabs(["Classification","Regression"])

        with tabs[0]:
            cls_res = report.get("classification",{})
            if cls_res:
                df_cls = pd.DataFrame(cls_res).T
                st.dataframe(df_cls.style.background_gradient(cmap="RdYlGn",
                    subset=["Accuracy","F1 Score"]), use_container_width=True)

        with tabs[1]:
            reg_res = report.get("regression",{})
            if reg_res:
                df_reg = pd.DataFrame(reg_res).T
                st.dataframe(df_reg, use_container_width=True)

        st.caption(f"Samples used: {report.get('n_samples',0)} | "
                   f"Features: {len(report.get('feature_cols',[]))}")

# ── Tab 13: Alerts ────────────────────────────────────────
def tab_alerts():
    st.markdown('<div class="sec">🔔 Price & Indicator Alerts</div>',
                unsafe_allow_html=True)

    with st.expander("➕ Add New Alert", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        a_symbol = c1.selectbox("Symbol",
            list(st.session_state.watchlist.values()) + list(HISTORICAL_FILES.keys()),
            key="a_sym")
        a_type   = c2.selectbox("Alert Type",
            ["Price >","Price <","RSI >","RSI <","ADX >","MACD Cross"],
            key="a_type")
        a_val    = c3.number_input("Value", value=24000.0, key="a_val")
        a_note   = c4.text_input("Note", placeholder="e.g. Breakout", key="a_note")
        if st.button("Add Alert"):
            new_id = db.add_alert(a_symbol, a_type, a_val, a_note)
            st.session_state.alert_rules.append({
                "id": new_id,
                "symbol": a_symbol, "type": a_type,
                "value": a_val,     "note": a_note,
                "active": True,     "created": datetime.now().strftime("%H:%M:%S"),
            })
            st.success(f"Alert added: {a_symbol} {a_type} {a_val}")

    # Existing alerts
    if st.session_state.alert_rules:
        st.markdown("**Active Alerts:**")
        for i,rule in enumerate(st.session_state.alert_rules):
            c1,c2,c3 = st.columns([5,2,1])
            c1.write(f"**{rule['symbol']}** — {rule['type']} **{rule['value']}** "
                     f"_{rule.get('note','')}_")
            c2.write(f"Added {rule['created']}")
            if c3.button("🗑", key=f"del_alert_{i}"):
                if rule.get("id") is not None:
                    db.delete_alert(rule["id"])
                st.session_state.alert_rules.pop(i); st.rerun()

        # Check alerts
        st.markdown("**Alert Log:**")
        ticks  = tick_store.get_all()
        for rule in st.session_state.alert_rules:
            name = rule["symbol"]
            key  = get_key(name)
            tick = ticks.get(key,{})
            ltp  = tick.get("ltp",0)
            if ltp == 0 and name in HISTORICAL_FILES:
                df = get_hist(name)
                ltp = df["Close"].iloc[-1] if not df.empty else 0
            triggered = False
            if   rule["type"]=="Price >" and ltp > rule["value"]: triggered=True
            elif rule["type"]=="Price <" and ltp < rule["value"]: triggered=True
            if triggered:
                msg = (f"🔔 **{name}** — {rule['type']} {rule['value']}  "
                       f"(LTP: {ltp:.2f})  [{datetime.now().strftime('%H:%M:%S')}]")
                st.warning(msg)
                st.session_state.alert_log.append(msg)
                if rule.get("id") is not None:
                    db.mark_alert_triggered(rule["id"])

        if st.session_state.alert_log:
            for log in st.session_state.alert_log[-10:]:
                st.info(log)

# ── Tab 14: Backtesting ───────────────────────────────────
def tab_backtest():
    st.markdown('<div class="sec">📈 Simple Strategy Backtester</div>',
                unsafe_allow_html=True)
    selected = st.session_state.selected_chart
    df = get_hist(selected) if selected in HISTORICAL_FILES else pd.DataFrame()
    if df.empty: st.info("No data."); return

    st.markdown("#### Strategy: EMA Crossover + RSI Filter")
    c1,c2,c3,c4 = st.columns(4)
    fast_ema  = c1.number_input("Fast EMA", value=20, min_value=5)
    slow_ema  = c2.number_input("Slow EMA", value=50, min_value=10)
    rsi_low   = c3.number_input("RSI Buy <", value=40.0)
    capital   = c4.number_input("Capital (₹)", value=100000.0, step=10000.0)

    if st.button("▶ Run Backtest"):
        df_bt = df.copy()
        df_bt["F_EMA"] = df_bt["Close"].ewm(span=int(fast_ema),adjust=False).mean()
        df_bt["S_EMA"] = df_bt["Close"].ewm(span=int(slow_ema),adjust=False).mean()
        df_bt["Signal"] = 0
        df_bt.loc[(df_bt["F_EMA"]>df_bt["S_EMA"]) &
                  (df_bt["RSI"]<rsi_low if "RSI" in df_bt.columns
                   else True),"Signal"] = 1
        df_bt["Signal"] = df_bt["Signal"].shift(1).fillna(0)
        df_bt["Strat_Ret"] = df_bt["Signal"] * df_bt["Daily_Ret"].fillna(0)
        df_bt["BH_Ret"]    = df_bt["Daily_Ret"].fillna(0)

        cum_strat = (1 + df_bt["Strat_Ret"]).cumprod()
        cum_bh    = (1 + df_bt["BH_Ret"]).cumprod()

        strat_final = capital * cum_strat.iloc[-1]
        bh_final    = capital * cum_bh.iloc[-1]
        trades      = df_bt["Signal"].diff().abs().sum() / 2

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Strategy Return", f"{(cum_strat.iloc[-1]-1)*100:+.2f}%")
        c2.metric("Buy & Hold",      f"{(cum_bh.iloc[-1]-1)*100:+.2f}%")
        c3.metric("Final Capital",   f"₹{strat_final:,.0f}")
        c4.metric("No. of Trades",   int(trades))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_bt["Date"],y=(cum_strat-1)*100,
            line=dict(color=C["green"],width=1.5),name="EMA Cross Strategy"))
        fig.add_trace(go.Scatter(x=df_bt["Date"],y=(cum_bh-1)*100,
            line=dict(color=C["blue"],width=1.2,dash="dot"),name="Buy & Hold"))
        fig.add_hline(y=0,line_color=C["muted"],line_width=0.8)
        fig.update_layout(**CHART,height=320,
            title=f"<b>{selected} — Strategy vs Buy & Hold</b>",
            yaxis_title="Return (%)")
        st.plotly_chart(fig, width="stretch")

# ── Tab 15: Settings ──────────────────────────────────────
def tab_settings():
    st.markdown('<div class="sec">⚙️ Settings</div>', unsafe_allow_html=True)
    s = st.session_state.settings

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Risk-Free Rate (India 10Y)**")
        s["rf_rate"] = st.number_input("Rf (%)", value=s["rf_rate"],
                                        min_value=0.0, max_value=20.0, step=0.1)
        st.markdown("**ADX Period**")
        s["adx_period"] = st.number_input("ADX Period", value=s["adx_period"],
                                           min_value=7, max_value=30)
        st.markdown("**RSI Period**")
        s["rsi_period"] = st.number_input("RSI Period", value=s["rsi_period"],
                                           min_value=5, max_value=30)
    with c2:
        st.markdown("**SuperTrend ATR Period**")
        s["st_period"] = st.number_input("ST Period", value=s["st_period"],
                                          min_value=5, max_value=30)
        st.markdown("**SuperTrend Multiplier**")
        s["st_multiplier"] = st.number_input("ST Multiplier", value=s["st_multiplier"],
                                              min_value=1.0, max_value=6.0, step=0.5)

    if st.button("💾 Save Settings"):
        for k, v in s.items():
            db.set_setting(k, v)
        st.success("Settings saved — will be restored on next launch.")

    st.markdown("---")
    st.markdown("**Clear Caches**")
    if st.button("🗑 Clear Historical Cache"):
        st.session_state.hist_cache = {}
        st.success("Historical cache cleared — will reload on next access.")
    if st.button("🗑 Clear ML Cache"):
        st.session_state.ml_cache = {}
        st.success("ML cache cleared.")

    st.markdown("---")
    st.markdown("**File Status**")
    for name,fn in HISTORICAL_FILES.items():
        p = Path(__file__).parent / "data" / fn
        exists = p.exists()
        st.write(f"{'✅' if exists else '❌'} `{fn}` — {'Found' if exists else 'NOT FOUND — copy to data/ folder'}")

    st.markdown("---")
    st.markdown("**Dashboard Version:** 2.0  |  Indicators: RSI · MACD · EMA · BB · ATR · ADX · SuperTrend · VWAP  |  ML: RF · XGB · LGB")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    sidebar()

    # Auto-start WS if token saved
    if (st.session_state.access_token and
            not st.session_state.ws_started and not ws_hub.is_running):
        _start_ws()

    st.markdown(
        "<h1 style='color:var(--text);margin-bottom:2px;font-size:23px;"
        "font-family:Space Grotesk,sans-serif;font-weight:700'>"
        "Real-Time Market Analytics</h1>"
        "<p style='color:var(--muted);margin:0 0 14px 0;font-size:11.5px;"
        "letter-spacing:.2px'>"
        "NIFTY 50 · NIFTY BANK · India VIX — Historical + Live Upstox &nbsp;·&nbsp; "
        "ADX · SuperTrend · VWAP · Option Chain · ML Predictions</p>",
        unsafe_allow_html=True)

    ticker_tape()
    kpi_row()
    st.markdown("")

    tab_labels = [
        "📊 Overview","⚡ Live","🕯 Chart","🔧 Technicals",
        "⚠️ Risk","🕳 Gap","📉 Volatility","🌡 VIX",
        "🔗 Options","📐 Breadth","🏆 Performance",
        "🤖 ML","🔔 Alerts","📈 Backtest","⚙️ Settings"
    ]
    tabs = st.tabs(tab_labels)

    renderers = [
        tab_market_overview, tab_live_market, tab_ohlc_chart, tab_technicals,
        tab_risk, tab_gap, tab_volatility, tab_vix,
        tab_option_chain, tab_breadth, tab_performance,
        tab_ml, tab_alerts, tab_backtest, tab_settings,
    ]
    for tab, renderer in zip(tabs, renderers):
        with tab:
            renderer()

    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
