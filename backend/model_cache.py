"""
Model caching and scheduled retraining system for multiple cryptocurrencies
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

from .forecast_service import ForecastService
from .live_data import LiveDataFetcher


class ModelCache:
    """
    Manages model training, caching, and scheduled updates for multiple cryptocurrencies.
    Each cryptocurrency gets its own trained model for accurate predictions.
    """
    
    CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
    
    # Define cryptocurrencies to train
    CRYPTO_CONFIGS = [
        {"symbol": "BTC", "name": "Bitcoin", "csv": "BTC.csv"},
        {"symbol": "ETH", "name": "Ethereum", "csv": "ETH.csv"},
        {"symbol": "BNB", "name": "BNB", "csv": "BNB.csv"},
        {"symbol": "SOL", "name": "Solana", "csv": "SOL.csv"},
    ]
    
    def __init__(self, data_dir: str, close_col: str = "close", window: int = 30):
        self.data_dir = Path(data_dir)
        self.close_col = close_col
        self.window = window
        
        self._services: Dict[str, ForecastService] = {}  # One service per crypto
        self._last_trained: Optional[datetime] = None
        self._training_lock = threading.Lock()
        self._is_training = False
        self._cached_overview = None
        
        self.live_fetcher = LiveDataFetcher()
        
        # Ensure cache directory exists
        self.CACHE_DIR.mkdir(exist_ok=True)
        
        # Load cached data if available
        self._load_cache()
        
        # Start background training thread
        self._start_background_trainer()
    
    @property
    def service(self) -> ForecastService:
        """Get cached BTC service (for backwards compatibility)"""
        if "BTC" not in self._services:
            self._train_models()
        return self._services.get("BTC")
    
    def _needs_retraining(self) -> bool:
        """Check if models need retraining (once per day)"""
        if self._last_trained is None:
            return True
        
        # Retrain if more than 24 hours old
        return (datetime.now() - self._last_trained) > timedelta(hours=24)
    
    def _load_cache(self):
        """Load cached forecasts and metadata from disk"""
        try:
            forecast_cache = self.CACHE_DIR / "forecast_cache.json"
            metadata_cache = self.CACHE_DIR / "cache_metadata.json"
            
            if forecast_cache.exists() and metadata_cache.exists():
                with open(forecast_cache, 'r') as f:
                    self._cached_overview = json.load(f)
                
                with open(metadata_cache, 'r') as f:
                    metadata = json.load(f)
                    self._last_trained = datetime.fromisoformat(metadata['last_trained'])
                
                print(f"[*] Loaded cached data for {len(self._cached_overview)} cryptocurrencies")
                print(f"  Last trained: {self._last_trained.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"No cache found, will train on startup: {e}")
    
    def _save_cache(self):
        """Save forecasts and metadata to disk"""
        try:
            forecast_cache = self.CACHE_DIR / "forecast_cache.json"
            metadata_cache = self.CACHE_DIR / "cache_metadata.json"
            
            with open(forecast_cache, 'w') as f:
                json.dump(self._cached_overview, f)
            
            metadata = {
                'last_trained': self._last_trained.isoformat() if self._last_trained else None,
                'data_dir': str(self.data_dir),
                'cryptocurrencies': [c['symbol'] for c in self.CRYPTO_CONFIGS]
            }
            
            with open(metadata_cache, 'w') as f:
                json.dump(metadata, f)
            
            print("[*] Saved to cache")
        except Exception as e:
            print(f"Could not save cache: {e}")
    
    def get_cached_overview(self):
        """Get cached forecasts if available"""
        return self._cached_overview
    
    def _train_models(self):
        """Train models for all cryptocurrencies with thread safety"""
        if self._is_training:
            print("Training already in progress, waiting...")
            return
        
        with self._training_lock:
            self._is_training = True
            try:
                print(f"[*] Starting model training for {len(self.CRYPTO_CONFIGS)} cryptocurrencies...")
                
                all_forecasts = []
                
                # Train a separate model for each cryptocurrency
                for config in self.CRYPTO_CONFIGS:
                    symbol = config["symbol"]
                    csv_path = self.data_dir / config["csv"]
                    
                    if not csv_path.exists():
                        print(f"  [!] {csv_path.name} not found, skipping {symbol}")
                        continue
                    
                    print(f"  [*] Training {symbol}...")
                    
                    # Update data with latest from API
                    try:
                        self.live_fetcher.update_csv_with_live_data(symbol, str(csv_path))
                    except Exception as e:
                        print(f"    [!] Could not update {symbol} data from API: {e}")
                    
                    # Train model for this cryptocurrency
                    service = ForecastService(
                        csv_path=str(csv_path),
                        close_col=self.close_col,
                        window=self.window
                    )
                    self._services[symbol] = service
                    
                    # Generate forecast using its own model
                    forecast = service.generate_single_forecast(symbol, config["name"])
                    all_forecasts.append(forecast.model_dump())
                    
                    print(f"    [*] {symbol} complete")
                
                self._cached_overview = all_forecasts
                self._last_trained = datetime.now()
                print(f"[*] Training complete for all cryptocurrencies")
                
                # Save to cache
                self._save_cache()
                
            except Exception as e:
                print(f"Error during training: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._is_training = False
    
    def _start_background_trainer(self):
        """Start background thread for scheduled retraining"""
        def training_loop():
            # Don't train on startup - wait for manual trigger
            # This avoids server startup delays
            print("[*] Background trainer ready (trigger manually with /api/retrain)")
            
            # Retrain every 24 hours if needed
            while True:
                time.sleep(3600)  # Check every hour
                
                if self._needs_retraining():
                    print(f"[*] Scheduled retraining triggered")
                    self._train_models()
        
        thread = threading.Thread(target=training_loop, daemon=True)
        thread.start()
    
    def force_retrain(self):
        """Force immediate retraining (for manual triggers)"""
        print("[*] Manual retrain triggered")
        self._last_trained = None
        self._train_models()
    
    def get_training_status(self) -> dict:
        """Get status information about model training"""
        return {
            "is_training": self._is_training,
            "last_trained": self._last_trained.isoformat() if self._last_trained else None,
            "needs_retraining": self._needs_retraining(),
            "next_training": (self._last_trained + timedelta(hours=24)).isoformat() if self._last_trained else "pending",
            "cryptocurrencies_trained": list(self._services.keys())
        }
