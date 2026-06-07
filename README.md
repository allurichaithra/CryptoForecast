## Hybrid ARIMA–LSTM–XGBoost Cryptocurrency Forecasting

**🆕 Now with Live Data & Automatic Daily Retraining!**

End-to-end production-ready ML project that combines:

1. **Backend** – trains ARIMA, LSTM, and XGBoost models with live data from CoinGecko API
2. **Auto-Retraining** – models retrain automatically every 24 hours with latest data
3. **Model Caching** – trained models are cached for fast responses (< 1 second)
4. **Live Prices** – fetches current cryptocurrency prices from API
5. **Frontend** – modern dashboard consuming real-time predictions

---

### 1. Project Structure

- `backend/`
  - `app.py` – FastAPI application (`/api/forecasts`)
  - `forecast_service.py` – data preparation + model orchestration
  - `data.py`, `models_*.py`, `ensemble.py` – reusable building blocks
- `data/sample_prices.csv` – demo OHLCV dataset (replace with your own)
- `train.py` – CLI to retrain the ensemble and export metrics/plots
- `index.html`, `styles.css`, `script.js` – frontend dashboard
- `requirements.txt` – Python dependencies for backend + training

---

### 2. Installation

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

### 3. Run the Backend API

**Easy Way:**
```bash
run.bat
```

**Manual:**
```bash
uvicorn backend.app:app --reload
```

**API Endpoints:**
- `GET /api/forecasts` - ML predictions for BTC, ETH, BNB, SOL
- `GET /api/live-prices` - Current live prices from CoinGecko
- `GET /api/metrics` - Model performance metrics
- `GET /api/status` - Training status and schedule
- `POST /api/retrain` - Force immediate retraining

**Key Features:**
- **Auto-retraining**: Models retrain every 24 hours automatically
- **Fast responses**: Cached models respond in < 1 second
- **Live data**: Fetches latest prices from CoinGecko API
- **Real predictions**: Based on current market data

---

### 4. Retrain / Evaluate Models

```bash
python train.py --csv_path data/sample_prices.csv --close_col close --window 30
```

Outputs:

- `outputs/metrics.csv` – RMSE/MAE/MAPE for each base model and the hybrid ensemble
- `outputs/forecast_plot.png` – Actual vs ensemble predictions

Use these artifacts to update the “Results” section of the dashboard.

---

### 5. Frontend Dashboard

1. Start the backend (step 3).
2. Open `index.html` in your browser.
3. The “Cryptocurrency Price Predictions and Forecasts” table automatically fetches live data from `/api/forecasts`. If the backend is offline, the table shows an inline error message.

Feel free to customize copy, colors, or embed additional plots/images.

---

### 6. Customization Ideas

- **Data**: Replace `data/sample_prices.csv` with longer histories or multiple coins.
- **Models**: Tune hyperparameters inside `backend/models_*.py` or add new indicators/features.
- **API**: Add endpoints for metrics, chart data, or streaming updates.
- **UI**: Integrate logos, add more sections, or hook into real-time price feeds.

This repository is now a presentation-ready artifact demonstrating the Hybrid ARIMA–LSTM–XGBoost ensemble concept from backend to frontend.


