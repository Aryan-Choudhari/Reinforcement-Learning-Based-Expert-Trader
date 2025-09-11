"""
Data preparation and feature engineering utilities
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    def __init__(self, config):
        self.config = config

    def load_featured_data(self, file_path):
        """Load the pre-featured trading data and add regime detection"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Featured data file not found: {file_path}")

        data = pd.read_csv(file_path)

        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp').reset_index(drop=True)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # **ENHANCED FIX: Convert all non-timestamp columns to numeric**
        for col in data.columns:
            if col != 'timestamp':
                if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                    data[col] = pd.to_numeric(data[col], errors='coerce')

        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.fillna(0, inplace=True)

        # Add regime detection features
        data = self.add_regime_detection(data)
        data = self.add_extended_trend_regimes(data)

        print(f"Loaded data. Shape: {data.shape}")
        print(f"Date range: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
        print(f"Added regime detection features")

        return data
    
    def split_data(self, data):
        """Split data to support exactly 4 walk-forward steps"""
        data_size = len(data)
        
        # For 4 walk-forward steps, design the split carefully
        # Initial train size: 50% of data
        # Each step adds: 8% of data
        # Total for training: 50% + 3*8% = 74% (leaving 26% for validation and test)
        initial_train_ratio = 0.50
        step_ratio = 0.08
        validation_ratio = 0.15
        # Test gets the remainder: ~10%
        
        initial_train_size = int(data_size * initial_train_ratio)
        step_size = int(data_size * step_ratio)
        validation_size = int(data_size * validation_ratio)
        
        # Training data includes initial + 3 steps worth of data for walk-forward
        # This ensures we have enough data for 4 walk-forward steps
        total_train_size = initial_train_size + (3 * step_size)
        
        train_data = data.iloc[:total_train_size].reset_index(drop=True)
        val_data = data.iloc[total_train_size:total_train_size + validation_size].reset_index(drop=True)
        test_data = data.iloc[total_train_size + validation_size:].reset_index(drop=True)
        
        print(f"Data split for 4 walk-forward steps:")
        print(f"  Initial train size: {initial_train_size} ({initial_train_ratio*100:.1f}%)")
        print(f"  Step size: {step_size} ({step_ratio*100:.1f}%)")
        print(f"  Total train data: {len(train_data)} ({len(train_data)/data_size*100:.1f}%)")
        print(f"  Validation data: {len(val_data)} ({len(val_data)/data_size*100:.1f}%)")
        print(f"  Test data: {len(test_data)} ({len(test_data)/data_size*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def select_features(self, train_data):
        """Select top features using Random Forest, always keeping OHLCV and VWAP"""
        # Always include core price and volume features
        core_features = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        available_core = [col for col in core_features if col in train_data.columns]

        # Get all other feature columns (excluding timestamp and core features)
        other_feature_columns = [col for col in train_data.columns
                            if col not in ['timestamp'] + available_core]

        valid_data = train_data.dropna()
        if len(valid_data) < self.config.MIN_VALID_ROWS:
            raise ValueError(f"Insufficient valid data: {len(valid_data)} samples")

        # Use past data only - no future leakage
        y_train = (valid_data['close'] - valid_data['close'].shift(1)) / valid_data['close'].shift(1)
        y_train.fillna(0, inplace=True)

        # Feature selection on non-core features only
        X_train_fs = valid_data[other_feature_columns].copy()
        
        # **FIX: Convert all columns to numeric, handling any remaining categorical/string columns**
        for col in X_train_fs.columns:
            if X_train_fs[col].dtype == 'object' or X_train_fs[col].dtype.name == 'category':
                # Convert categorical/string columns to numeric
                X_train_fs[col] = pd.to_numeric(X_train_fs[col], errors='coerce')
        
        # Fill any NaN values created by conversion
        X_train_fs.fillna(0, inplace=True)

        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_depth=12)
        rf.fit(X_train_fs, y_train)

        importances = pd.Series(rf.feature_importances_, index=other_feature_columns)

        # Select top features from non-core features
        remaining_slots = max(10, self.config.SELECTED_FEATURES - len(available_core))
        selected_other_features = importances.nlargest(remaining_slots).index.tolist()

        # Combine core + selected features
        feature_columns = available_core + selected_other_features

        print(f"Selected {len(feature_columns)} features:")
        print(f" Core OHLCV features: {len(available_core)}")
        print(f" Selected technical features: {len(selected_other_features)}")

        return feature_columns

    
    def prepare_scaler(self, train_data, feature_columns):
        """Prepare and fit the scaler"""
        scaler = StandardScaler()
        scaler.fit(train_data[feature_columns])
        return scaler

    def add_regime_detection(self, data):
        """Add regime detection features using only past data (no forward bias)"""
        df = data.copy()
        
        # Import technical analysis library
        try:
            import talib as ta
        except ImportError:
            print("Warning: talib not available, using simple RSI calculation")
            ta = None
        
        # Market volatility regime using rolling volatility
        df['returns'] = df['close'].pct_change()
        df['rolling_vol'] = df['returns'].rolling(window=20, min_periods=10).std() * np.sqrt(252)
        
        # Volatility regime classification (using past data only)
        # Calculate regime thresholds using expanding window (no forward bias)
        # SHIFT by 1 to ensure we only use data up to previous day
        vol_expanding_mean = df['rolling_vol'].shift(1).expanding(min_periods=40).mean()
        vol_expanding_std = df['rolling_vol'].shift(1).expanding(min_periods=40).std()
        
        # Define regimes based on expanding statistics (using previous day's thresholds)
        low_vol_threshold = vol_expanding_mean - 0.5 * vol_expanding_std
        high_vol_threshold = vol_expanding_mean + 0.5 * vol_expanding_std
        
        # Compare PREVIOUS day's volatility to thresholds
        prev_vol = df['rolling_vol'].shift(1)
        df['vol_regime'] = 'medium'  # default
        df.loc[prev_vol < low_vol_threshold, 'vol_regime'] = 'low'
        df.loc[prev_vol > high_vol_threshold, 'vol_regime'] = 'high'
        
        # Convert to numeric for RL
        regime_map = {'low': 0, 'medium': 1, 'high': 2}
        df['vol_regime_numeric'] = df['vol_regime'].map(regime_map)
        
        # Trend regime using multiple timeframes (past data only)
        # SHIFT all moving averages to use only data up to previous day
        df['sma_5'] = df['close'].shift(1).rolling(window=5).mean()
        df['sma_20'] = df['close'].shift(1).rolling(window=20).mean()
        df['sma_50'] = df['close'].shift(1).rolling(window=50).mean()
        
        # Trend alignment score using previous day's close vs previous SMAs
        prev_close = df['close'].shift(1)
        conditions = [
            prev_close > df['sma_5'],
            df['sma_5'] > df['sma_20'], 
            df['sma_20'] > df['sma_50']
        ]
        df['trend_alignment'] = sum(conditions) - 1.5  # Range: -1.5 to 1.5
        
        # Momentum strength using RSI and price momentum
        if 'rsi_14' not in df.columns:
            # Calculate RSI using only past data
            if ta:
                df['rsi_14'] = ta.RSI(df['close'].shift(1), timeperiod=14)
            else:
                # Simple RSI calculation if talib not available
                df['rsi_14'] = self._calculate_rsi(df['close'].shift(1), 14)
        else:
            # Ensure RSI is also shifted if pre-calculated
            df['rsi_14'] = df['rsi_14'].shift(1)
        
        # Price momentum using data from 6 days ago to 1 day ago (5-day lookback)
        df['price_momentum_5'] = (df['close'].shift(1) - df['close'].shift(6)) / df['close'].shift(6)
        df['momentum_strength'] = (df['rsi_14'] - 50) / 50  # Normalize RSI around 0
        
        # Market structure - Use previous day's data for HH/LL detection
        # Look for HH/LL in the past 20 days ending yesterday
        df['hh_20'] = df['high'].shift(1).rolling(20).max() == df['high'].shift(1)
        df['ll_20'] = df['low'].shift(1).rolling(20).min() == df['low'].shift(1)
        df['market_structure'] = (df['hh_20'].astype(int) - df['ll_20'].astype(int))
        
        # Support/Resistance levels using past data only
        # Calculate S/R levels using data up to 2 days ago, apply to yesterday's decision
        df['resistance_level'] = df['high'].shift(2).rolling(window=20, center=False).max()
        df['support_level'] = df['low'].shift(2).rolling(window=20, center=False).min()
        
        # Distance to S/R levels using previous day's close
        prev_close = df['close'].shift(1)
        df['dist_to_resistance'] = (df['resistance_level'] - prev_close) / prev_close
        df['dist_to_support'] = (prev_close - df['support_level']) / prev_close
        
        return df
    
    def add_extended_trend_regimes(self, df):
        """Add extended trend regime classification"""
        # Expand trend alignment to capture more granular regimes
        df['strong_bull_trend'] = (df['trend_alignment'] > 0.6).astype(int)
        df['bullish_trend'] = ((df['trend_alignment'] > 0.3) & (df['trend_alignment'] <= 0.6)).astype(int)
        df['weak_bull_trend'] = ((df['trend_alignment'] > 0.1) & (df['trend_alignment'] <= 0.3)).astype(int)
        
        df['strong_bear_trend'] = (df['trend_alignment'] < -0.6).astype(int)
        df['bearish_trend'] = ((df['trend_alignment'] < -0.3) & (df['trend_alignment'] >= -0.6)).astype(int)
        df['weak_bear_trend'] = ((df['trend_alignment'] < -0.1) & (df['trend_alignment'] >= -0.3)).astype(int)
        
        df['neutral_trend'] = (abs(df['trend_alignment']) <= 0.1).astype(int)
        
        # Add short-term correction signals
        df['bull_correction'] = ((df['trend_alignment'] > 0.3) & (df['price_momentum_5'] < -0.02)).astype(int)
        df['bear_correction'] = ((df['trend_alignment'] < -0.3) & (df['price_momentum_5'] > 0.02)).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Simple RSI calculation if talib not available"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
