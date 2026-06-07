"""
Standalone script to train models for all cryptocurrencies
"""
from backend.forecast_service import ForecastService
from backend.live_data import LiveDataFetcher
from pathlib import Path
import json
from datetime import datetime

DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CRYPTO_CONFIGS = [
    {"symbol": "BTC", "name": "Bitcoin", "csv": "BTC.csv"},
    {"symbol": "ETH", "name": "Ethereum", "csv": "ETH.csv"},
    {"symbol": "BNB", "name": "BNB", "csv": "BNB.csv"},
    {"symbol": "SOL", "name": "Solana", "csv": "SOL.csv"},
]

print("[*] Starting model training for all cryptocurrencies...")

live_fetcher = LiveDataFetcher()
all_forecasts = []

for config in CRYPTO_CONFIGS:
    symbol = config["symbol"]
    csv_path = DATA_DIR / config["csv"]
    
    if not csv_path.exists():
        print(f"  [!] {csv_path.name} not found, skipping {symbol}")
        continue
    
    print(f"  [*] Training {symbol}...")
    
    # Update data with latest from API
    try:
        live_fetcher.update_csv_with_live_data(symbol, str(csv_path))
    except Exception as e:
        print(f"    [!] Could not update {symbol} data from API: {e}")
    
    # Train model for this cryptocurrency
    service = ForecastService(
        csv_path=str(csv_path),
        close_col="close",
        window=30
    )
    
    # Generate forecast using its own model
    forecast = service.generate_single_forecast(symbol, config["name"])
    all_forecasts.append(forecast.model_dump())
    
    print(f"    [*] {symbol} complete - Price: ${forecast.price:,.2f}, 1y forecast: ${forecast.forecasts['1y']:,.2f}")

# Save to cache
forecast_cache = CACHE_DIR / "forecast_cache.json"
metadata_cache = CACHE_DIR / "cache_metadata.json"

with open(forecast_cache, 'w') as f:
    json.dump(all_forecasts, f, indent=2)

metadata = {
    'last_trained': datetime.now().isoformat(),
    'data_dir': str(DATA_DIR),
    'cryptocurrencies': [c['symbol'] for c in CRYPTO_CONFIGS]
}

with open(metadata_cache, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[*] Training complete! Cache saved to {CACHE_DIR}")
print(f"\nForecasts generated:")
for forecast in all_forecasts:
    print(f"  {forecast['symbol']}: ${forecast['price']:,.2f} -> ${forecast['forecasts']['1y']:,.2f} (+{((forecast['forecasts']['1y'] / forecast['price']) - 1) * 100:.1f}%)")


