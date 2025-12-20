"""
Live cryptocurrency data fetching from CoinGecko API
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import requests


class LiveDataFetcher:
    """Fetch live cryptocurrency prices and historical data"""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    CRYPTO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "SOL": "solana",
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_live_prices(self) -> Dict[str, float]:
        """Get current live prices for all cryptocurrencies"""
        ids = ",".join(self.CRYPTO_IDS.values())
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": ids,
            "vs_currencies": "usd",
            "include_24hr_change": "true"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            prices = {}
            for symbol, coin_id in self.CRYPTO_IDS.items():
                if coin_id in data:
                    prices[symbol] = {
                        "price": data[coin_id]["usd"],
                        "change_24h": data[coin_id].get("usd_24h_change", 0) / 100  # Convert to decimal
                    }
            
            return prices
        except Exception as e:
            print(f"Error fetching live prices: {e}")
            return {}
    
    def get_historical_data(self, symbol: str = "BTC", days: int = 365) -> pd.DataFrame:
        """
        Fetch historical price data for training
        
        Args:
            symbol: Cryptocurrency symbol (BTC, ETH, etc.)
            days: Number of days of historical data
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        coin_id = self.CRYPTO_IDS.get(symbol, "bitcoin")
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            df = pd.DataFrame({
                "timestamp": [p[0] for p in prices],
                "close": [p[1] for p in prices],
                "volume": [v[1] for v in volumes] if volumes else [0] * len(prices)
            })
            
            # Convert timestamp to date
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop("timestamp", axis=1)
            
            # Add OHLC (for daily data, we approximate open/high/low from close)
            df["open"] = df["close"].shift(1).fillna(df["close"])
            df["high"] = df["close"] * 1.01  # Approximate
            df["low"] = df["close"] * 0.99   # Approximate
            
            # Reorder columns
            df = df[["date", "open", "high", "low", "close", "volume"]]
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def update_csv_with_live_data(self, symbol: str = "BTC", csv_path: str = "data/BTC.csv"):
        """
        Update existing CSV file with latest data from API
        
        This merges historical CSV data with recent API data to keep it current
        """
        try:
            # Load existing data
            existing_df = pd.read_csv(csv_path)
            existing_df["date"] = pd.to_datetime(existing_df["date"])
            last_date = existing_df["date"].max()
            
            # Calculate days since last update
            days_since = (datetime.now() - last_date).days
            
            if days_since > 1:
                # Fetch new data
                new_df = self.get_historical_data(symbol, days=days_since + 7)
                
                if not new_df.empty:
                    # Filter to only new dates
                    new_df = new_df[new_df["date"] > last_date]
                    
                    # Append and save
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                    updated_df = updated_df.sort_values("date").drop_duplicates(subset=["date"])
                    updated_df.to_csv(csv_path, index=False)
                    
                    print(f"Updated {csv_path} with {len(new_df)} new rows")
                    return True
            else:
                print(f"{csv_path} is already up to date")
                return False
                
        except Exception as e:
            print(f"Error updating CSV: {e}")
            return False
