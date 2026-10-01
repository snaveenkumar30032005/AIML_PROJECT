# %% [markdown]
# # Stock Price Trend Prediction with LSTM
# **Pipeline:** fetch data (yfinance) → indicators (SMA, RSI) → scale & window → LSTM (Keras) → train/validate → predictions vs actual → save weights.
# 
# > Educational project. Model output is not financial advice.

# %%
# !pip install yfinance tensorflow pandas numpy matplotlib scikit-learn joblib
import os, random, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

TICKER = "AAPL"
START, END = "2015-01-01", None      # None = today
WINDOW = 60                          # days of history per sample
EPOCHS, BATCH = 60, 32
FEATURES = ["Close", "SMA20", "SMA50", "RSI14"]
os.makedirs("models", exist_ok=True); os.makedirs("graphs", exist_ok=True)

# %% [markdown]
# ## 1. Fetch data

# %%
raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw = raw[["Open","High","Low","Close","Volume"]].dropna()
print(raw.shape, raw.index.min().date(), "->", raw.index.max().date())
raw.tail()

# %% [markdown]
# ## 2. Technical indicators (moving averages & RSI)

# %%
def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()   # Wilder smoothing
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / loss)
    return df.dropna()

df = add_indicators(raw)

fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios":[3,1]})
ax[0].plot(df.index, df["Close"], label="Close", lw=1.2)
ax[0].plot(df.index, df["SMA20"], label="SMA20"); ax[0].plot(df.index, df["SMA50"], label="SMA50")
ax[0].set_title(f"{TICKER} price with moving averages"); ax[0].legend()
ax[1].plot(df.index, df["RSI14"], color="purple", lw=1)
ax[1].axhline(70, color="r", ls="--"); ax[1].axhline(30, color="g", ls="--"); ax[1].set_ylabel("RSI(14)")
plt.tight_layout(); plt.savefig("graphs/01_price_indicators.png", dpi=150); plt.show()

# %% [markdown]
# ## 3. Normalize & build sequences
# Chronological split (70/15/15). The scaler is fit on the **training period only** to avoid look-ahead leakage.

# %%
data = df[FEATURES].values
n = len(data)
train_end = int(n * 0.70); val_end = int(n * 0.85)

scaler = MinMaxScaler().fit(data[:train_end])
scaled = scaler.transform(data)

X, y, tgt_idx = [], [], []
for i in range(WINDOW, n):
    X.append(scaled[i-WINDOW:i]); y.append(scaled[i, 0]); tgt_idx.append(i)
X, y, tgt_idx = np.array(X), np.array(y), np.array(tgt_idx)

tr = tgt_idx < train_end
va = (tgt_idx >= train_end) & (tgt_idx < val_end)
te = tgt_idx >= val_end
X_tr, y_tr, X_va, y_va, X_te, y_te = X[tr], y[tr], X[va], y[va], X[te], y[te]
print("train/val/test:", X_tr.shape, X_va.shape, X_te.shape)

def inv_close(v):      # scaled Close -> price
    return (np.asarray(v) - scaler.min_[0]) / scaler.scale_[0]

# %% [markdown]
# ## 4. Build the LSTM

# %%
model = keras.Sequential([
    layers.Input((WINDOW, len(FEATURES))),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.summary()

# %% [markdown]
# ## 5. Train & validate

# %%
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    keras.callbacks.ModelCheckpoint("models/lstm_best.weights.h5", monitor="val_loss",
                                    save_best_only=True, save_weights_only=True),
]
hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                 epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks, verbose=2)

plt.figure(figsize=(9,4))
plt.plot(hist.history["loss"], label="train"); plt.plot(hist.history["val_loss"], label="validation")
plt.title("Training curve (MSE, scaled)"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout(); plt.savefig("graphs/02_training_curve.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Evaluate on the test set: predictions vs actual
# The naive baseline ("tomorrow = today") is hard to beat on prices, so we compare against it.

# %%
pred = inv_close(model.predict(X_te, verbose=0).ravel())
actual = inv_close(y_te)
naive = df["Close"].values[tgt_idx[te] - 1]
dates = df.index[tgt_idx[te]]

def metrics(a, p, prev):
    rmse = np.sqrt(np.mean((a-p)**2)); mae = np.mean(np.abs(a-p))
    mape = np.mean(np.abs((a-p)/a))*100
    dir_acc = np.mean(np.sign(a-prev) == np.sign(p-prev))*100
    return dict(RMSE=rmse, MAE=mae, MAPE_pct=mape, Directional_acc_pct=dir_acc)

res = pd.DataFrame({"LSTM": metrics(actual, pred, naive),
                    "Naive (yesterday)": metrics(actual, naive, naive)}).round(3)
print(res)

fig, ax = plt.subplots(figsize=(13,5))
ax.plot(dates, actual, label="Actual", lw=1.4)
ax.plot(dates, pred, label="LSTM prediction", lw=1.2)
ax.set_title(f"{TICKER}: predicted vs actual close (test set)"); ax.legend()
plt.tight_layout(); plt.savefig("graphs/03_pred_vs_actual.png", dpi=150); plt.show()

# %% [markdown]
# ## 7. Next-day forecast

# %%
last = scaled[-WINDOW:][None, ...]
next_close = float(inv_close(model.predict(last, verbose=0).ravel())[0])
print(f"Last close: {df['Close'].iloc[-1]:.2f}  |  Predicted next close: {next_close:.2f}")

# %% [markdown]
# ## 8. Save artifacts (weights, full model, scaler, config)

# %%
model.save_weights("models/lstm_final.weights.h5")
model.save("models/lstm_model.keras")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(dict(ticker=TICKER, window=WINDOW, features=FEATURES), "models/config.pkl")
res.to_csv("graphs/test_metrics.csv")
print(os.listdir("models"))