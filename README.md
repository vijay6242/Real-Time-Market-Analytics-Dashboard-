# 📈 Real-Time Market Analytics Dashboard

A professional real-time stock market analytics dashboard built using **Python**, **Streamlit**, **Plotly**, and the **Upstox API**.

The dashboard combines **historical market analysis**, **live market data**, **technical indicators**, **option chain analytics**, **risk metrics**, and **machine learning predictions** into a single application.



## 🚀 Features

### 📊 Market Analytics
- Live Market Dashboard
- Market Overview
- OHLC Candlestick Charts
- Technical Indicators
- Gap Analysis
- Volatility Dashboard
- India VIX Dashboard
- Market Breadth
- Performance Dashboard
- Risk Analytics

### 📈 Technical Indicators

Built-in indicators include:

- RSI
- MACD
- EMA (20,50,200)
- Bollinger Bands
- ATR
- ADX
- SuperTrend
- VWAP

---

### 📉 Risk Analytics

- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Win Rate
- Volatility Analysis

---

### 📌 Option Analytics

- Option Chain
- Put Call Ratio (PCR)
- Max Pain
- Market Sentiment

---

### 🤖 Machine Learning

Market prediction models:

- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Logistic Regression

Predictions include:

- Next Day Direction
- Next Close Price
- Classification Accuracy
- Regression Metrics

---

### ⚡ Live Market Data

Real-time market streaming using Upstox WebSocket.

Features:

- Live LTP
- Live Change %
- Watchlist
- Dynamic Symbol Search
- Auto Refresh
- Background WebSocket Thread
- Thread-safe Tick Store

---

## 🖥 Dashboard Modules

The application contains **15 major dashboards**:

1. Market Overview
2. Live Market
3. OHLC Chart
4. Technical Indicators
5. Risk Analytics
6. Gap Analysis
7. Volatility
8. India VIX
9. Option Chain Analytics
10. Market Breadth
11. Performance Dashboard
12. ML Predictions
13. Alerts
14. Backtesting
15. Settings

---

## 📂 Project Structure

```
Market-Dashboard/
│
├── app.py
├── data.py
├── websocket.py
├── logger.py
├── ml_models.py
├── MarketDataFeedV3_pb2.py
├── requirements.txt
│
├── data/
│   ├── NIFTY 50.csv
│   ├── NIFTY BANK.csv
│   └── INDIA VIX.csv
│


## 🛠 Technologies Used

- Python
- Streamlit
- Plotly
- Pandas
- NumPy
- SciPy
- Scikit-Learn
- XGBoost
- LightGBM
- Upstox API
- WebSockets
- Protocol Buffers

---

## 🔑 Setup

Create a **.env** file

```text
UPSTOX_ACCESS_TOKEN=YOUR_ACCESS_TOKEN
```

---

## ▶ Run

```bash
streamlit run app.py
```

---

## 📊 Supported Market Data

- NIFTY 50
- NIFTY BANK
- INDIA VIX
- NSE Stocks
- BSE Stocks
- Custom Watchlist

---

## 📈 Charts

- Candlestick Chart
- Volume Chart
- RSI
- MACD
- ADX
- Bollinger Bands
- EMA
- SuperTrend
- VWAP
- Heatmaps
- Gauges
- Performance Charts

---

## 🤖 Machine Learning Features

The project includes:

- Feature Engineering
- Time Series Cross Validation
- Classification Models
- Regression Models
- Accuracy Evaluation
- Confusion Matrix
- Prediction Dashboard

---

## ⚡ WebSocket Architecture

```
Upstox WebSocket
        │
        ▼
Background Thread
        │
        ▼
Thread-safe Tick Store
        │
        ▼
Streamlit Dashboard
        │
        ▼
Real-time Charts & Metrics
```

---

## 📌 Future Improvements

- user friendly interface
- 

## 📄 Requirements

Major libraries:

- streamlit
- plotly
- pandas
- numpy
- scipy
- scikit-learn
- xgboost
- lightgbm
- aiohttp
- websockets
- protobuf
- python-dotenv

---
