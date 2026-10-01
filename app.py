import numpy as np, pandas as pd, yfinance as yf, streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")
FEATURES = ["Close", "SMA20", "SMA50", "RSI14"]

@st.cache_data(ttl=3600)
def load(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    d = df["Close"].diff()
    g = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["RSI14"] = 100 - 100 / (1 + g / l)
    return df.dropna()

@st.cache_resource(show_spinner="Training LSTM...")
def train(ticker, start, window, epochs):
    df = add_indicators(load(ticker, start))
    data = df[FEATURES].values
    n = len(data); cut = int(n * 0.85)
    sc = MinMaxScaler().fit(data[:int(n * 0.70)])
    s = sc.transform(data)
    X = np.array([s[i-window:i] for i in range(window, n)])
    y = np.array([s[i, 0] for i in range(window, n)])
    idx = np.arange(window, n)
    tr, te = idx < cut, idx >= cut
    m = keras.Sequential([
        layers.Input((window, len(FEATURES))),
        layers.LSTM(64, return_sequences=True), layers.Dropout(0.2),
        layers.LSTM(32), layers.Dropout(0.2),
        layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile("adam", "mse")
    m.fit(X[tr], y[tr], validation_split=0.15, epochs=epochs, batch_size=32, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    inv = lambda v: (np.asarray(v) - sc.min_[0]) / sc.scale_[0]
    pred = inv(m.predict(X[te], verbose=0).ravel())
    nxt = float(inv(m.predict(s[-window:][None], verbose=0).ravel())[0])
    return df, idx[te], pred, nxt

st.title("LSTM Stock Price Trend Predictor")
with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    start = st.date_input("Start date", pd.Timestamp("2015-01-01")).strftime("%Y-%m-%d")
    window = st.slider("Look-back window (days)", 30, 120, 60, 10)
    epochs = st.slider("Max epochs", 5, 100, 30, 5)
    run = st.button("Train & predict", type="primary")

df = add_indicators(load(ticker, start))
c1, c2 = st.columns([3, 1])
with c1:
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(df.index, df["Close"], label="Close"); ax[0].plot(df.index, df["SMA20"], label="SMA20")
    ax[0].plot(df.index, df["SMA50"], label="SMA50"); ax[0].legend()
    ax[1].plot(df.index, df["RSI14"], color="purple")
    ax[1].axhline(70, color="r", ls="--"); ax[1].axhline(30, color="g", ls="--"); ax[1].set_ylabel("RSI")
    st.pyplot(fig)
with c2:
    rsi = df["RSI14"].iloc[-1]
    st.metric("Last close", f"{df['Close'].iloc[-1]:.2f}")
    st.metric("RSI(14)", f"{rsi:.1f}", "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral", delta_color="off")

if run:
    d, tidx, pred, nxt = train(ticker, start, window, epochs)
    actual = d["Close"].values[tidx]; prev = d["Close"].values[tidx - 1]
    rmse = np.sqrt(np.mean((actual - pred) ** 2)); mape = np.mean(np.abs((actual - pred) / actual)) * 100
    dacc = np.mean(np.sign(actual - prev) == np.sign(pred - prev)) * 100
    nrmse = np.sqrt(np.mean((actual - prev) ** 2))
    a, b, c, e = st.columns(4)
    a.metric("Next-day forecast", f"{nxt:.2f}", f"{nxt - d['Close'].iloc[-1]:+.2f}")
    b.metric("Test RMSE", f"{rmse:.2f}", f"naive {nrmse:.2f}", delta_color="off")
    c.metric("Test MAPE", f"{mape:.2f}%")
    e.metric("Directional accuracy", f"{dacc:.1f}%")
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(d.index[tidx], actual, label="Actual"); ax2.plot(d.index[tidx], pred, label="LSTM")
    ax2.set_title("Test set: predicted vs actual"); ax2.legend(); st.pyplot(fig2)
    st.caption("Educational demo only, not financial advice.")
else:
    st.info("Set parameters in the sidebar and click **Train & predict**.")