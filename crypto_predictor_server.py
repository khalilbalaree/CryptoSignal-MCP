#!/usr/bin/env python3
"""
Crypto Price Prediction MCP Server

This server provides tools for predicting cryptocurrency price movements
using historical data from Binance API.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
import pytz
from typing import Any, Dict, List, Optional
import time

import httpx
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFECV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import talib
import re
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Crypto Predictor")

class CryptoPredictionService:
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.models = {}
        self.scalers = {}
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.rate_limit = 1200  # Binance limit per minute
        self.request_timestamps = []
        self.news_cache = {}
        self.news_cache_ttl = 900  # 15 minutes for news
        
    def _get_prediction_time_window(self, interval: str, current_hour_timestamp: int) -> Dict[str, str]:
        """Calculate exact prediction time window in ET timezone"""
        try:
            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(current_hour_timestamp / 1000, tz=timezone.utc)
            
            # Convert to ET timezone
            et_tz = pytz.timezone('US/Eastern')
            dt_et = dt.astimezone(et_tz)
            
            # Calculate interval duration
            interval_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, 
                '1h': 60, '2h': 120, '4h': 240
            }.get(interval, 60)
            
            # Calculate end time
            end_dt = dt_et + timedelta(minutes=interval_minutes)
            
            # Format times
            start_time = dt_et.strftime("%B %d, %I:%M%p ET").replace(" 0", " ")
            end_time = end_dt.strftime("%I:%M%p ET").replace(" 0", " ")
            current_time = datetime.now(et_tz).strftime("%B %d, %I:%M:%S%p ET").replace(" 0", " ")
            
            # Create prediction window description
            if interval_minutes == 60:
                prediction_window = f"{dt_et.strftime('%B %d, %I-%p').replace(' 0', ' ')}{end_dt.strftime('%I%p ET').replace(' 0', ' ')}"
            else:
                prediction_window = f"{start_time} - {end_time}"
            
            return {
                "prediction_window": prediction_window,
                "start_time": start_time,
                "end_time": end_time,
                "current_time": current_time,
                "interval_minutes": interval_minutes
            }
        except Exception as e:
            # Fallback to basic format
            return {
                "prediction_window": f"Current {interval} period",
                "start_time": "Unknown",
                "end_time": "Unknown", 
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "interval_minutes": 60
            }
    
    def _validate_symbol(self, symbol: str) -> str:
        """Validate and format trading pair symbol"""
        symbol = symbol.upper().strip()
        if not symbol.endswith('USDT') and not symbol.endswith('BTC') and not symbol.endswith('ETH'):
            if 'USDT' not in symbol and 'BTC' not in symbol:
                symbol += 'USDT'
        return symbol
    
    def _validate_interval(self, interval: str) -> str:
        """Validate time interval"""
        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {', '.join(valid_intervals)}")
        return interval
    
    def _check_rate_limit(self):
        """Check if we're within rate limits"""
        now = time.time()
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        if len(self.request_timestamps) >= self.rate_limit:
            raise Exception("Rate limit exceeded. Please wait before making more requests.")
        
        self.request_timestamps.append(now)
    
    def _get_cache_key(self, symbol: str, interval: str, limit: int) -> str:
        """Generate cache key"""
        return f"{symbol}_{interval}_{limit}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        return time.time() - self.cache[cache_key]['timestamp'] < self.cache_ttl
        
    async def get_historical_data(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """Fetch historical kline data from Binance with caching and rate limiting"""
        try:
            # Validate inputs
            symbol = self._validate_symbol(symbol)
            interval = self._validate_interval(interval)
                
            # Check cache first
            cache_key = self._get_cache_key(symbol, interval, limit)
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Check rate limit
            self._check_rate_limit()
            
            url = f"{self.base_url}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                raise ValueError(f"No data returned for {symbol}")
            
            # Convert to structured format
            klines = []
            for item in data:
                try:
                    klines.append({
                        "open_time": int(item[0]),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "close_time": int(item[6]),
                        "quote_volume": float(item[7]),
                        "count": int(item[8]),
                        "taker_buy_base": float(item[9]),
                        "taker_buy_quote": float(item[10])
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid data point: {e}")
                    continue
                    
            # Cache the results
            self.cache[cache_key] = {
                'data': klines,
                'timestamp': time.time()
            }
            
            return klines
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise ValueError(f"Invalid symbol or parameters: {symbol}")
            elif e.response.status_code == 429:
                raise Exception("Rate limit exceeded by Binance. Please wait.")
            else:
                raise Exception(f"API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def _is_period_incomplete(self, period_data: Dict) -> bool:
        """Check if a trading period is incomplete based on its close_time vs current time"""
        close_time = period_data['close_time']
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # A period is incomplete if its close_time is in the future
        return close_time > current_time
        
    def _filter_complete_periods(self, data: List[Dict]) -> List[Dict]:
        """Filter out incomplete periods from the data"""
        if not data:
            return data
            
        # Check if the last period is incomplete
        if self._is_period_incomplete(data[-1]):
            return data[:-1]  # Remove incomplete period
        else:
            return data  # All periods are complete
    
    def _ensure_json_serializable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all values in dataframe are JSON serializable by converting numpy types to Python types"""
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(bool)  # Convert numpy bool to python bool
            elif df[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                df[col] = df[col].astype(int)   # Convert numpy int to python int
            elif df[col].dtype in ['float16', 'float32', 'float64']:
                df[col] = df[col].astype(float) # Convert numpy float to python float
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators and features for prediction"""
        try:
            # Convert to numpy arrays for TA-Lib
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=10).std()
            df['price_acceleration'] = df['price_change'].diff()
            
            # Enhanced Moving Averages with more periods
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean() if len(df) >= 50 else df['close'].rolling(window=min(len(df), 20)).mean()
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            # Moving average convergence/divergence signals (add small epsilon to prevent division by zero)
            df['sma_5_20_ratio'] = df['sma_5'] / (df['sma_20'] + 1e-8)
            df['ema_12_26_ratio'] = df['ema_12'] / (df['ema_26'] + 1e-8)
            df['price_vs_sma20'] = df['close'] / (df['sma_20'] + 1e-8)
            df['price_vs_ema50'] = df['close'] / (df['ema_50'] + 1e-8)
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI with multiple periods
            try:
                df['rsi_14'] = pd.Series(talib.RSI(close, timeperiod=14), index=df.index)
                df['rsi_7'] = pd.Series(talib.RSI(close, timeperiod=7), index=df.index)
            except:
                # Fallback calculation for RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-8)  # Prevent division by zero
                df['rsi_14'] = 100 - (100 / (1 + rs))
                
                # Calculate RSI-7 separately
                gain_7 = (delta.where(delta > 0, 0)).rolling(window=7).mean()
                loss_7 = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
                rs_7 = gain_7 / (loss_7 + 1e-8)
                df['rsi_7'] = 100 - (100 / (1 + rs_7))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            
            # Stochastic Oscillator
            try:
                stoch_k, stoch_d = talib.STOCH(high, low, close)
                df['stoch_k'] = pd.Series(stoch_k, index=df.index)
                df['stoch_d'] = pd.Series(stoch_d, index=df.index)
            except:
                # Fallback calculation
                lowest_low = df['low'].rolling(window=14).min()
                highest_high = df['high'].rolling(window=14).max()
                df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-8)
                df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            try:
                df['williams_r'] = pd.Series(talib.WILLR(high, low, close), index=df.index)
            except:
                highest_high = df['high'].rolling(window=14).max()
                lowest_low = df['low'].rolling(window=14).min()
                df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-8)
            
            # Average True Range (ATR)
            try:
                df['atr'] = pd.Series(talib.ATR(high, low, close), index=df.index)
            except:
                df['tr1'] = df['high'] - df['low']
                df['tr2'] = abs(df['high'] - df['close'].shift(1))
                df['tr3'] = abs(df['low'] - df['close'].shift(1))
                df['atr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1).rolling(window=14).mean()
                df.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
            df['volume_rate_of_change'] = df['volume'].pct_change()
            
            # On-Balance Volume (OBV)
            df['obv'] = (df['volume'] * df['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
            df['obv_sma'] = df['obv'].rolling(window=10).mean()
            
            # Price patterns
            df['high_low_ratio'] = df['high'] / (df['low'] + 1e-8)
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['body_size'] = abs(df['close'] - df['open']) / (df['close'] + 1e-8)
            
            # Momentum indicators
            df['momentum_10'] = df['close'] / (df['close'].shift(10) + 1e-8)
            df['rate_of_change'] = df['close'].pct_change(periods=10)
            
            # Support and Resistance levels
            df['resistance_level'] = df['high'].rolling(window=20).max()
            df['support_level'] = df['low'].rolling(window=20).min()
            df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / (df['close'] + 1e-8)
            df['distance_to_support'] = (df['close'] - df['support_level']) / (df['close'] + 1e-8)
            
            # Enhanced Market structure and trend analysis
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            df['trend_strength'] = df['higher_high'].rolling(window=5).sum() - df['lower_low'].rolling(window=5).sum()
            
            # Fractal patterns removed due to look-ahead bias
            
            # Price momentum across multiple timeframes
            df['momentum_3'] = df['close'] / (df['close'].shift(3) + 1e-8) - 1
            df['momentum_5'] = df['close'] / (df['close'].shift(5) + 1e-8) - 1
            if len(df) >= 20:
                df['momentum_20'] = df['close'] / (df['close'].shift(20) + 1e-8) - 1
            else:
                df['momentum_20'] = 0
            
            # Velocity and acceleration
            df['price_velocity'] = df['close'].diff().rolling(window=3).mean()
            df['price_acceleration'] = df['price_velocity'].diff()
            
            # Volatility-adjusted metrics
            df['sharpe_ratio'] = df['price_change'].rolling(window=20).mean() / (df['volatility'] + 1e-8)
            df['volatility_ratio'] = df['volatility'] / (df['volatility'].rolling(window=20).mean() + 1e-8)
            
            # Market regime indicators
            df['volatility_regime'] = (df['volatility'] > df['volatility'].rolling(window=50).quantile(0.7)).astype(int)
            df['trend_regime'] = (abs(df['trend_strength']) > 2).astype(int)
            
            # Time-based features using cyclical encoding
            import numpy as np
            timestamps = pd.to_datetime(df.index)
            df['hour'] = timestamps.hour
            df['day_of_week'] = timestamps.dayofweek
            df['month'] = timestamps.month
            
            # Cyclical encoding for time features (preserves time relationships)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Market session indicators (crypto markets are 24/7 but patterns still exist)
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int) 
            df['is_ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Drop raw time columns as we're using cyclical encoding
            df = df.drop(['hour', 'day_of_week', 'month'], axis=1)
            
            # Target: 1 if next period close is higher than current period close, 0 if lower
            # This predicts price direction change from current period to next period
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Normalize all features for consistent scaling
            df = self._normalize_features(df)
            
            # Ensure all values are JSON serializable
            df = self._ensure_json_serializable(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            # Fallback to basic features
            df['price_change'] = df['close'].pct_change()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df = self._normalize_features(df)
            df = self._ensure_json_serializable(df)
            return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all technical indicators and features for consistent scaling"""
        try:
            # Define feature categories for different normalization strategies
            
            # Already normalized features (0-100 scale or ratios)
            already_normalized = [
                'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d', 'williams_r',
                'bb_position', 'close_position', 'higher_high', 'lower_low',
                'volatility_regime', 'trend_regime',
                'target'  # Don't normalize target
            ]
            
            # Ratio features (should be standardized around 1.0)
            ratio_features = [
                'sma_5_20_ratio', 'ema_12_26_ratio', 'price_vs_sma20', 'price_vs_ema50',
                'high_low_ratio', 'volume_ratio', 'momentum_10', 'volatility_ratio'
            ]
            
            # Percentage change features (already in percentage form)
            pct_features = [
                'price_change', 'price_acceleration', 'volume_rate_of_change',
                'momentum_3', 'momentum_5', 'momentum_20', 'rate_of_change',
                'distance_to_resistance', 'distance_to_support'
            ]
            
            # Price-based features (need robust scaling due to outliers)
            price_features = [
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_12', 'ema_26', 'ema_50',
                'bb_middle', 'bb_upper', 'bb_lower', 'resistance_level', 'support_level',
                'upper_shadow', 'lower_shadow', 'price_velocity'
            ]
            
            # Volume and OBV features (log transformation + standardization)
            volume_features = [
                'volume_sma', 'obv', 'obv_sma'
            ]
            
            # Technical indicators (standardization)
            technical_features = [
                'macd', 'macd_signal', 'macd_histogram', 'volatility', 'bb_width',
                'atr', 'body_size', 'trend_strength', 'sharpe_ratio'
            ]
            
            # Create a copy to avoid modifying original
            df_normalized = df.copy()
            
            # 1. Handle infinite values and extreme outliers
            for col in df_normalized.columns:
                if col not in already_normalized and df_normalized[col].dtype in ['float64', 'int64']:
                    # Replace infinite values with NaN
                    df_normalized[col] = df_normalized[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Cap extreme outliers at 99.5th percentile
                    if not df_normalized[col].isna().all():
                        upper_bound = df_normalized[col].quantile(0.995)
                        lower_bound = df_normalized[col].quantile(0.005)
                        df_normalized[col] = np.clip(df_normalized[col], lower_bound, upper_bound)
            
            # 2. Normalize ratio features (center around 1.0, scale by std)
            for feature in ratio_features:
                if feature in df_normalized.columns and not df_normalized[feature].isna().all():
                    # Log transform ratios to handle multiplicative relationships
                    df_normalized[f'{feature}_log'] = np.log(df_normalized[feature] + 1e-8)
                    # Remove original ratio feature to avoid redundancy
                    df_normalized.drop(feature, axis=1, inplace=True)
            
            # 3. Standardize percentage features (z-score normalization)
            for feature in pct_features:
                if feature in df_normalized.columns and not df_normalized[feature].isna().all():
                    mean_val = df_normalized[feature].mean()
                    std_val = df_normalized[feature].std()
                    if std_val > 0:
                        df_normalized[feature] = (df_normalized[feature] - mean_val) / std_val
            
            # 4. Robust scaling for price features (less sensitive to outliers)
            for feature in price_features:
                if feature in df_normalized.columns and not df_normalized[feature].isna().all():
                    median_val = df_normalized[feature].median()
                    mad = np.median(np.abs(df_normalized[feature] - median_val))
                    if mad > 0:
                        df_normalized[feature] = (df_normalized[feature] - median_val) / (1.4826 * mad)
            
            # 5. Log transform + standardize volume features
            for feature in volume_features:
                if feature in df_normalized.columns and not df_normalized[feature].isna().all():
                    # Log transform (add small constant to handle zeros)
                    min_positive = df_normalized[feature][df_normalized[feature] > 0].min()
                    log_feature = np.log(df_normalized[feature] + min_positive / 10)
                    # Standardize the log values
                    mean_val = log_feature.mean()
                    std_val = log_feature.std()
                    if std_val > 0:
                        df_normalized[feature] = (log_feature - mean_val) / std_val
            
            # 6. Standard scaling for technical indicators
            for feature in technical_features:
                if feature in df_normalized.columns and not df_normalized[feature].isna().all():
                    mean_val = df_normalized[feature].mean()
                    std_val = df_normalized[feature].std()
                    if std_val > 0:
                        df_normalized[feature] = (df_normalized[feature] - mean_val) / std_val
            
            # 7. Min-max scaling for RSI and oscillators (keep in 0-1 range)
            oscillator_features = ['rsi_14', 'rsi_7', 'stoch_k', 'stoch_d']
            for feature in oscillator_features:
                if feature in df_normalized.columns and not df_normalized[feature].isna().all():
                    df_normalized[feature] = df_normalized[feature] / 100.0  # Convert to 0-1 scale
            
            # 8. Handle Williams %R (convert from -100,0 to 0,1 scale)
            if 'williams_r' in df_normalized.columns:
                df_normalized['williams_r'] = (df_normalized['williams_r'] + 100) / 100.0
            
            # 9. Fill remaining NaN values with rolling median (conservative approach)
            for col in df_normalized.columns:
                if col not in already_normalized and df_normalized[col].dtype in ['float64', 'int64']:
                    # Use rolling median with expanding window for early values
                    rolling_median = df_normalized[col].expanding().median()
                    df_normalized[col] = df_normalized[col].fillna(rolling_median)
                    
                    # If still NaN (all NaN column), fill with 0
                    df_normalized[col] = df_normalized[col].fillna(0)
            
            # 10. Ensure no features exceed reasonable bounds after normalization
            # Exclude price data columns from clipping
            price_columns = ['open', 'high', 'low', 'close', 'volume', 'open_time', 'close_time', 'quote_volume', 'count', 'taker_buy_base', 'taker_buy_quote']
            for col in df_normalized.columns:
                if col not in already_normalized and col not in price_columns and df_normalized[col].dtype in ['float64', 'int64']:
                    # Cap normalized values at ±5 standard deviations
                    df_normalized[col] = np.clip(df_normalized[col], -5, 5)
            
            logger.info(f"Normalized {len(df_normalized.columns)} features successfully")
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            # Return original dataframe if normalization fails
            return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for model training with comprehensive feature set"""
        # Filter to only include columns that exist in the dataframe
        available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        
        if len(available_features) < 5:
            raise ValueError(f"Not enough features available. Found: {available_features}")
        
        # Remove rows with NaN values (last row will have NaN target automatically)
        df_clean = df[available_features + ['target']].dropna()
        
        if len(df_clean) < 100:
            logger.warning(f"Limited training data: {len(df_clean)} samples")
            if len(df_clean) < 50:
                raise ValueError(f"Insufficient clean data for training: {len(df_clean)} samples")
        
        X = df_clean[available_features].values
        y = df_clean['target'].values
        
        # Feature importance logging
        feature_names = available_features
        logger.info(f"Training with {len(feature_names)} features: {', '.join(feature_names[:10])}...")
        
        return X, y, feature_names
    
    async def train_model(self, symbol: str, interval: str = "1h", limit: int = 1000, model_type: str = "ensemble"):
        """Train enhanced prediction model with cross-validation"""
        try:
            symbol = self._validate_symbol(symbol)
            interval = self._validate_interval(interval)
            
            # Get historical data (request extra to account for incomplete current period)
            data = await self.get_historical_data(symbol, interval, limit + 1)
            if len(data) < 100:
                raise ValueError(f"Insufficient data: {len(data)} samples")
                
            # Filter out incomplete periods from training data
            training_data = self._filter_complete_periods(data)
            if len(training_data) < 100:
                raise ValueError(f"Insufficient training data after removing current period: {len(training_data)} samples")
                
            df = pd.DataFrame(training_data)
            
            # Create features
            df = self.create_features(df)
            
            # Prepare training data
            X, y, feature_names = self.prepare_training_data(df)
            
            if len(np.unique(y)) < 2:
                raise ValueError("Insufficient class diversity in target variable")
            
            # Advanced feature selection with multiple methods
            # First pass: Remove highly correlated features (more aggressive threshold)
            correlation_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.7)]
            
            # Keep features that aren't highly correlated
            remaining_features = [f for f in feature_names if f not in high_corr_features]
            remaining_indices = [i for i, f in enumerate(feature_names) if f in remaining_features]
            X_reduced = X[:, remaining_indices]
            
            # Second pass: Use multiple selection methods
            if len(remaining_features) > 12:
                # Use RFECV for optimal feature count
                rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                selector = RFECV(rf_temp, step=1, cv=3, scoring='accuracy', min_features_to_select=8)
                X_selected = selector.fit_transform(X_reduced, y)
                selected_mask = selector.support_
                selected_feature_names = [remaining_features[i] for i, selected in enumerate(selected_mask) if selected]
            else:
                X_selected = X_reduced
                selected_feature_names = remaining_features
                selector = None
            
            # Robust scaling for better handling of outliers
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            
            # Simplified ensemble with diverse algorithms
            models = {}
            
            # Gradient Boosting with better parameters
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            
            # SVM with RBF kernel for non-linear patterns
            svm_model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                random_state=42,
                probability=True  # Enable probability estimates for ensemble
            )
            
            
            # Train models with time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            gb_model.fit(X_scaled, y)
            svm_model.fit(X_scaled, y)
            
            models['gradient_boosting'] = gb_model
            models['svm'] = svm_model
            
            # Time series cross-validation scores
            gb_cv_scores = cross_val_score(gb_model, X_scaled, y, cv=tscv, scoring='accuracy')
            svm_cv_scores = cross_val_score(svm_model, X_scaled, y, cv=tscv, scoring='accuracy')
            
            # Create meta-ensemble (voting classifier)
            voting_clf = VotingClassifier(
                estimators=[
                    ('gb', gb_model),
                    ('svm', svm_model)
                ],
                voting='soft',
                weights=[0.6, 0.4]  # Slightly favor GB over SVM
            )
            voting_clf.fit(X_scaled, y)
            models['ensemble'] = voting_clf
            
            # Feature importance from gradient boosting model
            feature_importance = dict(zip(selected_feature_names, gb_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Store enhanced model info
            self.models[symbol] = {
                'models': models,
                'feature_names': selected_feature_names,
                'original_feature_names': feature_names,
                'remaining_features': remaining_features,
                'feature_selector': selector,
                'feature_importance': feature_importance,
                'model_type': model_type,
                'interval': interval,
                'cv_scores': {
                    'gradient_boosting': gb_cv_scores,
                    'svm': svm_cv_scores
                },
                'removed_features': high_corr_features,
                'correlation_threshold': 0.7
            }
            self.scalers[symbol] = scaler
            
            return {
                "symbol": symbol,
                "gradient_boosting_accuracy": float(np.mean(gb_cv_scores)),
                "svm_accuracy": float(np.mean(svm_cv_scores)),
                "ensemble_performance": {
                    "gb_std": float(np.std(gb_cv_scores)),
                    "svm_std": float(np.std(svm_cv_scores))
                },
                "samples": len(X),
                "features": len(selected_feature_names),
                "removed_correlated_features": len(high_corr_features),
                "top_features": [f"{name}: {importance:.4f}" for name, importance in top_features],
                "class_distribution": {f"class_{i}": int(np.sum(y == i)) for i in np.unique(y)},
                "model_trained": True,
                "training_period": f"{len(data)} {interval} candles",
                "selected_features": selected_feature_names[:10]
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            raise
    
    async def predict_current_hour(self, symbol: str, interval: str = "1h", training_periods: int = 100) -> Dict[str, Any]:
        """Predict if current hour will close higher or lower than it opened
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval for prediction
            training_periods: Number of historical periods to use for training (default: 100)
        """
        try:
            symbol = self._validate_symbol(symbol)
            interval = self._validate_interval(interval)
            
            # Check if model exists and was trained for the same interval
            if symbol not in self.models or self.models[symbol].get('interval') != interval:
                await self.train_model(symbol, interval, training_periods)
            
            # Get recent data - use same amount as training for consistency
            data = await self.get_historical_data(symbol, interval, training_periods)
            if len(data) < 50:
                raise ValueError(f"Insufficient recent data: {len(data)} samples")
                
            # Get current period information (use latest period whether complete or incomplete)
            current_period_data = data[-1]  # Latest period
            is_current_incomplete = self._is_period_incomplete(current_period_data)
            
            df = pd.DataFrame(self._filter_complete_periods(data))
            
            current_period_open = float(current_period_data['open'])
            current_price = float(current_period_data['close'])  # Current price
            
            # Calculate time window information
            time_info = self._get_prediction_time_window(interval, current_period_data['open_time'])
            
            # Create features
            df = self.create_features(df)
            
            # Get model info
            model_info = self.models[symbol]
            feature_names = model_info['feature_names']
            remaining_features = model_info['remaining_features']
            removed_features = model_info['removed_features']
            feature_selector = model_info.get('feature_selector')
            models = model_info['models']
            
            # Get latest complete features (use -1 since we now align data properly)
            try:
                # Use the original feature names from training to maintain consistency
                original_feature_names = model_info['original_feature_names']
                available_features = [col for col in original_feature_names if col in df.columns]
                
                # Use the last complete row with all features
                latest_all_features = df[available_features].iloc[-1].values.reshape(1, -1)
                
                # Apply the same correlation filtering as used in training
                remaining_feature_indices = [i for i, f in enumerate(available_features) if f in remaining_features]
                latest_reduced_features = latest_all_features[:, remaining_feature_indices]
                
                # Apply the same feature selection as used in training
                if feature_selector is not None:
                    latest_features_selected = feature_selector.transform(latest_reduced_features)
                else:
                    latest_features_selected = latest_reduced_features
                
            except KeyError as e:
                raise ValueError(f"Missing feature in prediction data: {e}")
            
            if np.isnan(latest_features_selected).any():
                raise ValueError("NaN values in latest features")
            
            # Scale features using the same scaler as training
            scaler = self.scalers[symbol]
            latest_features_scaled = scaler.transform(latest_features_selected)
            
            # Enhanced ensemble predictions with dynamic weighting
            predictions = {}
            probabilities = {}
            stored_cv_scores = model_info.get('cv_scores', {})
            
            # Calculate mean CV scores for all models
            cv_scores = {}
            for model_name in models.keys():
                if model_name in stored_cv_scores:
                    cv_scores[model_name] = np.mean(stored_cv_scores[model_name])
                else:
                    cv_scores[model_name] = 0.5  # Default for missing scores
            
            # Get predictions from all models
            for model_name, model in models.items():
                if model_name == 'ensemble':  # Skip the meta-ensemble for individual predictions
                    continue
                pred = model.predict(latest_features_scaled)[0]
                prob = model.predict_proba(latest_features_scaled)[0]
                predictions[model_name] = pred
                probabilities[model_name] = prob
            
            # Use the meta-ensemble if available, otherwise weighted ensemble
            if 'ensemble' in models:
                ensemble_pred = models['ensemble'].predict(latest_features_scaled)[0]
                ensemble_prob = models['ensemble'].predict_proba(latest_features_scaled)[0]
                ensemble_prob_up = float(ensemble_prob[1] if len(ensemble_prob) > 1 else 0.5)
                ensemble_prob_down = 1 - ensemble_prob_up
                ensemble_prediction = ensemble_pred
            else:
                # Fallback to weighted ensemble
                total_score = sum(cv_scores.values())
                weights = {name: score/total_score for name, score in cv_scores.items() if name != 'ensemble'}
                
                ensemble_prob_up = sum(weights.get(name, 0) * (prob[1] if len(prob) > 1 else 0.5) 
                                     for name, prob in probabilities.items())
                ensemble_prob_down = 1 - ensemble_prob_up
                ensemble_prediction = 1 if ensemble_prob_up > 0.5 else 0
            
            # Enhanced confidence scoring based on multiple factors
            prob_diff = abs(ensemble_prob_up - ensemble_prob_down)
            
            # Factor 1: Probability difference
            conf_prob = min(prob_diff * 2, 1.0)  # Normalize to 0-1
            
            # Factor 2: Model agreement (how many models agree with ensemble)
            agreement_count = sum(1 for pred in predictions.values() if pred == ensemble_prediction)
            model_agreement = agreement_count / max(len(predictions), 1)
            
            # Factor 3: CV score quality (higher average CV score = more confidence)
            avg_cv_score = np.mean(list(cv_scores.values()))
            cv_confidence = max(0, (avg_cv_score - 0.5) * 2)  # Normalize 0.5-1 to 0-1
            
            # Combined confidence score
            combined_confidence = (conf_prob * 0.4 + model_agreement * 0.4 + cv_confidence * 0.2)
            
            # Confidence levels with stricter thresholds
            if combined_confidence > 0.7 and prob_diff > 0.2:
                confidence_level = "HIGH"
            elif combined_confidence > 0.5 and prob_diff > 0.1:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
            
            # current_price already set from current_hour_data above
            price_change_24h = float(((current_price - df['close'].iloc[-25]) / df['close'].iloc[-25]) * 100) if len(df) > 25 else 0
            
            # Market conditions - get from latest complete features row 
            latest_complete_row = df.iloc[-1] if len(df) > 0 else {}
            market_conditions = {
                "rsi_oversold": bool(float(latest_complete_row.get('rsi_14', 50)) < 30),
                "rsi_overbought": bool(float(latest_complete_row.get('rsi_14', 50)) > 70),
                "high_volatility": bool(float(latest_complete_row.get('volatility', 0)) > float(df['volatility'].quantile(0.8)) if len(df) > 0 else False),
                "strong_volume": bool(float(latest_complete_row.get('volume_ratio', 1)) > 1.5),
                "near_resistance": bool(float(latest_complete_row.get('distance_to_resistance', 0)) < 0.02),
                "near_support": bool(float(latest_complete_row.get('distance_to_support', 0)) < 0.02)
            }
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "period_open": current_period_open,
                "period_change": float((current_price - current_period_open) / current_period_open * 100),
                "price_change_24h": price_change_24h,
                "prediction": "UP" if ensemble_prediction == 1 else "DOWN",
                "prediction_window": time_info["prediction_window"],
                "prediction_for_time": time_info["end_time"],
                "prediction_target": f"Next period will close higher than current period close (${current_price:.2f}) by {time_info['end_time']}",
                "current_time_et": time_info["current_time"],
                "time_remaining": f"Prediction valid until {time_info['end_time']}",
                "confidence": float(max(ensemble_prob_up, ensemble_prob_down)),
                "confidence_level": confidence_level,
                "probability_up": float(ensemble_prob_up),
                "probability_down": float(ensemble_prob_down),
                "model_predictions": {
                    name: "UP" if pred == 1 else "DOWN" 
                    for name, pred in predictions.items()
                },
                "model_probabilities": {
                    name: {"up": float(prob[1] if len(prob) > 1 else 0.5), "down": float(prob[0])} 
                    for name, prob in probabilities.items()
                },
                "market_conditions": market_conditions,
                "timestamp": datetime.now().isoformat(),
                "interval": interval,
                "data_quality": {
                    "samples_used": len(data),
                    "features_count": len(feature_names),
                    "has_sufficient_data": len(data) >= 100
                },
                "model_weights": weights if 'weights' in locals() else {},
                "cv_scores": cv_scores,
                "confidence_breakdown": {
                    "probability_confidence": float(conf_prob),
                    "model_agreement": float(model_agreement),
                    "cv_confidence": float(cv_confidence),
                    "combined_confidence": float(combined_confidence),
                    "agreeing_models": agreement_count,
                    "total_models": len(predictions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            raise
    
    def get_crypto_news_search_query(self, symbol: str = "bitcoin", limit: int = 5) -> Dict[str, Any]:
        """Generate optimized search query for crypto news using Claude Code's WebSearch"""
        try:
            # Normalize symbol for news queries
            news_symbol = symbol.replace("USDT", "").replace("BTC", "").lower()
            if news_symbol == "btc":
                news_symbol = "bitcoin"
            elif news_symbol == "eth":
                news_symbol = "ethereum"
            
            # Create optimized search query
            search_query = f"{news_symbol} cryptocurrency news today price analysis market sentiment"
            
            return {
                "symbol": news_symbol,
                "search_query": search_query,
                "suggested_domains": ["coindesk.com", "cointelegraph.com", "decrypt.co", "bitcoinmagazine.com", "cryptonews.com"],
                "search_instructions": f"Search for recent news about {news_symbol} and analyze market sentiment",
                "analysis_prompt": f"Analyze the sentiment and market impact of recent {news_symbol} news. Focus on: 1) Overall sentiment (bullish/bearish/neutral), 2) Key events affecting price, 3) Market impact assessment, 4) Trading implications",
                "timestamp": datetime.now().isoformat(),
                "status": "ready_for_websearch",
                "note": "Use this query with Claude Code's WebSearch tool for real-time news analysis"
            }
            
        except Exception as e:
            logger.error(f"Error preparing crypto news search: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }

    async def fetch_trader_activities(self, trader_address: str, limit: int = 200) -> Dict[str, Any]:
        """Fetch Polymarket trader activities for analysis"""
        try:
            url = "https://data-api.polymarket.com/activity"
            params = {
                'user': trader_address,
                'limit': limit,
                'sortBy': 'TIMESTAMP',
                'sortDirection': 'DESC'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            activities = data if isinstance(data, list) else data.get('data', [])
            
            # Group activities by slug with filtered content
            grouped_activities = {}
            for activity in activities:
                slug = activity.get("slug")
                if slug not in grouped_activities:
                    grouped_activities[slug] = []
                
                filtered_activity = {
                    "side": activity.get("side"),  # BUY or SELL
                    "size": activity.get("size"),  # Number of shares
                    "usdcSize": activity.get("usdcSize"),  # USD amount
                    "price": activity.get("price"),  # Price per share
                    "outcome": activity.get("outcome"),  # Up/Down/Yes/No
                }
                grouped_activities[slug].append(filtered_activity)
            
            return grouped_activities
            
        except Exception as e:
            return {
                "error": str(e),
                "trader_address": trader_address,
                "timestamp": datetime.now().isoformat()
            }

    

# Global feature columns definition - optimized set with reduced correlation
FEATURE_COLUMNS = [
    # Core price momentum (keep most informative)
    'price_change', 'volatility', 'momentum_5',
    
    # Moving averages (reduced set - keep key levels only)
    'sma_20', 'ema_12', 'price_vs_sma20',
    
    # MACD (keep histogram as most informative)
    'macd_histogram',
    
    # Oscillators (keep RSI-14 and one stochastic)
    'rsi_14', 'stoch_k', 'williams_r',
    
    # Bollinger Bands (keep position, remove width)
    'bb_position',
    
    # Volatility measure
    'atr',
    
    # Volume (keep ratio and OBV)
    'volume_ratio', 'obv',
    
    # Price patterns (keep most informative)
    'close_position', 'body_size',
    
    # Support/Resistance
    'distance_to_support', 'distance_to_resistance',
    
    # Market structure
    'trend_strength', 'higher_high', 'lower_low',
    
    # Advanced metrics
    'sharpe_ratio', 'volatility_regime'
]

# Initialize service
crypto_service = CryptoPredictionService()


@mcp.tool()
async def predict_crypto_direction(symbol: str, interval: str = "1h", training_periods: int = 1000) -> str:
    """
    Advanced ML prediction: Trains ensemble models to predict if crypto will close UP or DOWN.
    Uses Random Forest + Gradient Boosting with 30+ technical indicators. Auto-trains if needed.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
        interval: Time interval for prediction ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M') - Default: '1h'
        training_periods: Number of historical periods to use for training - Default: 1000
    
    Returns: JSON object with ML prediction including:
        Prediction: direction (UP/DOWN), confidence_score, probability_up, probability_down
        Model Performance: accuracy, precision, recall, f1_score
        Market Context: current_price, rsi, volume_ratio, trend_strength
        Feature Importance: top contributing technical indicators
        Training Info: periods_used, model_type, training_timestamp
        Risk Assessment: volatility, support_distance, resistance_distance
    """
    try:
        prediction = await crypto_service.predict_current_hour(symbol, interval, training_periods)
        return json.dumps(prediction, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
async def analyze_crypto_indicators(symbol: str, interval: str = "1h", limit: int = 100) -> str:
    """
    Technical analysis: Calculates essential RSI, moving averages, volume, and trend signals.
    Fast analysis without ML training - provides immediate market insights.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
        interval: Time interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M') - Default: '1h'
        limit: Number of historical data points to analyze - Default: 100
    
    Returns: JSON object with streamlined technical analysis including:
        Core Metrics: symbol, current_price, price_change_pct, rsi, volume_ratio
        Key Moving Averages: ma_20, ema_12, price_vs_ma20
        Essential Signals: rsi_oversold/overbought, above_ma_20, high_volume, momentum
        Support/Resistance: basic levels and distances
        Data Quality: periods_analyzed, sufficient_history
    """
    try:
        data = await crypto_service.get_historical_data(symbol, interval, limit)
        
        # Get current price from raw data (may be incomplete period)
        current_price = data[-1]['close'] if data else 0
        current_period_data = data[-1] if data else {}
        
        # Filter complete periods for technical indicators
        complete_data = crypto_service._filter_complete_periods(data)
        if not complete_data:
            return json.dumps({"error": "No complete periods available for analysis"}, indent=2)
            
        # Create features from complete data only
        df_complete = pd.DataFrame(complete_data)
        df_complete = crypto_service.create_features(df_complete)
        latest_complete = df_complete.iloc[-1]
        
        # Create current period features if we have incomplete period
        if len(data) > len(complete_data):
            # There's an incomplete current period
            df_current = pd.DataFrame(data)
            df_current = crypto_service.create_features(df_current)
            current_period_features = df_current.iloc[-1]
        else:
            # No incomplete period, current = latest complete
            current_period_features = latest_complete
        
        # Simple trend calculation
        def get_trend(series, periods=5):
            if len(series) < periods:
                return "neutral"
            return "rising" if series.iloc[-1] > series.iloc[-periods] else "falling"
        
        # Basic support/resistance (use complete data)
        support_level = df_complete['low'].rolling(window=20).min().iloc[-1] if len(df_complete) >= 20 else df_complete['low'].min()
        resistance_level = df_complete['high'].rolling(window=20).max().iloc[-1] if len(df_complete) >= 20 else df_complete['high'].max()
        
        # Time info
        current_hour_timestamp = current_period_data.get('open_time', int(time.time() * 1000))
        time_info = crypto_service._get_prediction_time_window(interval, current_hour_timestamp)
        
        # Streamlined analysis
        analysis = {
            "symbol": symbol,
            "current_price": float(current_price),
            "analysis_time": time_info["current_time"],
            
            # Core indicators (use current period for live data)
            "price_change_pct": float(current_period_features['price_change'] * 100) if pd.notna(current_period_features['price_change']) else 0,
            "rsi": float(latest_complete['rsi_14']) if pd.notna(latest_complete['rsi_14']) else 50,
            "volume_ratio": float(current_period_features['volume_ratio']) if 'volume_ratio' in current_period_features and pd.notna(current_period_features['volume_ratio']) else 1,
            
            # Key moving averages (use complete data for stability)
            "ma_20": float(latest_complete['sma_20']) if pd.notna(latest_complete['sma_20']) else 0,
            "ema_12": float(latest_complete['ema_12']) if pd.notna(latest_complete['ema_12']) else 0,
            
            # Momentum (use current vs complete data)
            "momentum_5": float((current_price / df_complete['close'].iloc[-6] - 1) * 100) if len(df_complete) > 5 else 0,
            
            # Support/Resistance (basic)
            "support_level": float(support_level),
            "resistance_level": float(resistance_level),
            "distance_to_support": float((current_price - support_level) / current_price * 100),
            "distance_to_resistance": float((resistance_level - current_price) / current_price * 100),
            
            # Essential signals only
            "signals": {
                "rsi_oversold": bool(float(latest_complete['rsi_14']) < 30) if pd.notna(latest_complete['rsi_14']) else False,
                "rsi_overbought": bool(float(latest_complete['rsi_14']) > 70) if pd.notna(latest_complete['rsi_14']) else False,
                "above_ma_20": bool(float(current_price) > float(latest_complete['sma_20'])) if pd.notna(latest_complete['sma_20']) else False,
                "high_volume": bool(float(current_period_features['volume_ratio']) > 1.5) if 'volume_ratio' in current_period_features and pd.notna(current_period_features['volume_ratio']) else False,
                "bullish_momentum": bool(get_trend(df_complete['close']) == "rising" and ((current_price / df_complete['close'].iloc[-6] - 1) * 100) > 0) if len(df_complete) > 5 else False,
                "bearish_momentum": bool(get_trend(df_complete['close']) == "falling" and ((current_price / df_complete['close'].iloc[-6] - 1) * 100) < 0) if len(df_complete) > 5 else False
            },
            
            # Basic data quality
            "data_quality": {
                "periods_analyzed": len(df_complete),
                "sufficient_history": bool(len(df_complete) >= 50)
            }
        }
        
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
async def get_crypto_news_search(symbol: str = "bitcoin") -> str:
    """
    Generate optimized search query for cryptocurrency news analysis.
    Returns structured data to be used with Claude Code's WebSearch tool.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTCUSDT', 'bitcoin', 'ethereum') - Default: 'bitcoin'
    
    Returns: JSON object with:
        Search Query: optimized query for WebSearch tool
        Suggested Domains: reliable crypto news sources  
        Analysis Prompt: structured prompt for sentiment analysis
        Search Instructions: how to use the query effectively
        
    Usage: Copy the search_query and use it with Claude Code's WebSearch tool, 
    then apply the analysis_prompt to the results for market sentiment analysis.
    """
    try:
        search_data = crypto_service.get_crypto_news_search_query(symbol)
        return json.dumps(search_data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
async def monitor_polymarket_trader(trader_address: str, limit: int = 100) -> str:
    """
    Fetch Polymarket trader activities for behavioral analysis.
    Returns raw activity data to understand trader patterns and strategies.
    
    Args:
        trader_address: Ethereum wallet address of the trader to monitor
        limit: Number of recent activities to fetch (default: 200, max: 1000)
    
    Returns: JSON object with:
        Trader Info: address, total activities count
        Activities: Complete list of trading activities with timestamps, markets, outcomes, sizes, prices
        Raw Data: Full activity details for comprehensive analysis
        
    Usage: Analyze the returned activities to understand:
    - Trading patterns and preferences (UP/DOWN, YES/NO outcomes)
    - Market focus areas and diversification
    - Position sizing and risk management
    - Trading frequency and timing
    - P&L performance across different markets
    """
    try:
        trader_data = await crypto_service.fetch_trader_activities(trader_address, limit)
        return json.dumps(trader_data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)



if __name__ == "__main__":
    mcp.run()