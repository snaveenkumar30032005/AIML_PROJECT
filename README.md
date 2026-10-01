# LSTM Stock Price Trend Prediction

## Run
```bash
pip install -r requirements.txt
jupyter notebook stock_lstm.ipynb     # run all cells -> creates models/ and graphs/
streamlit run app.py                  # local dashboard
```

## Deploy the dashboard (free)
1. Push this folder to a GitHub repo (include `app.py` and `requirements.txt`).
2. Go to https://share.streamlit.io, click "New app", pick the repo and `app.py`.
3. Deploy; the resulting URL is your Streamlit link.
   (Tip: use `tensorflow-cpu` in requirements.txt for faster builds.)

## Outputs
- `models/lstm_best.weights.h5`, `lstm_final.weights.h5`, `lstm_model.keras`, `scaler.pkl`
- `graphs/*.png`, `graphs/test_metrics.csv`