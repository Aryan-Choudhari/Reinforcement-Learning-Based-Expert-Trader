"""
Enhanced Universal Data Handler with Regime Detection
Complete corrected version integrating all improvements
"""
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
import pandas_ta as ta

warnings.filterwarnings('ignore')


class UniversalDataHandler:
    """Enhanced DataHandler for universal multi-stock training with regime detection"""
    
    def __init__(self, config, lookback_context=40):
        self.config = config
        self.lookback_context = lookback_context
    
    def get_asset_and_timeframe(self, filepath):
        """Extracts asset name and timeframe from a filename using regex."""
        filename = os.path.basename(filepath)
        match = re.match(r'([A-Z\-]+)_(\d+[a-zA-Z]+)\.csv', filename, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def calculate_vwap_properly(self, df):
        """Calculate VWAP without forward bias using only past data."""
        if 'vwap' not in df.columns:
            df = df.copy()
            df['date'] = df.index.date
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['volume_price'] = df['typical_price'] * df['volume']
            df['cum_volume_price'] = df.groupby('date')['volume_price'].cumsum()
            df['cum_volume'] = df.groupby('date')['volume'].cumsum()
            df['vwap'] = df['cum_volume_price'] / df['cum_volume']
            df = df.drop(['date', 'typical_price', 'volume_price', 'cum_volume_price', 'cum_volume'], axis=1)
        return df
    
    def calculate_universal_features(self, df):
        """
        Calculate PRICE-SCALE INVARIANT features for universal training
        All features are relative/normalized to work across different price scales
        """
        df = df.copy()
        
        # ============================================
        # 1. NORMALIZED PRICES (relative to rolling average)
        # ============================================
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            # Calculate rolling average for normalization
            rolling_avg = df['close'].rolling(window=20, min_periods=1).mean()
            
            # Normalize all prices to this average
            df['close_normalized'] = df['close'] / rolling_avg
            df['open_normalized'] = df['open'] / rolling_avg
            df['high_normalized'] = df['high'] / rolling_avg
            df['low_normalized'] = df['low'] / rolling_avg
            
            # ============================================
            # 2. INTRABAR RATIOS (already scale-invariant)
            # ============================================
            df['hl_range'] = df['high'] - df['low']
            df['oc_range'] = abs(df['close'] - df['open'])
            
            # Price position within bar (0-1 scale)
            df['price_position'] = np.where(df['hl_range'] != 0,
                                          (df['close'] - df['low']) / df['hl_range'],
                                          0.5)
            
            # Body size relative to range (0-1 scale)
            df['body_to_range'] = np.where(df['hl_range'] != 0,
                                         df['oc_range'] / df['hl_range'],
                                         0)
            
            # Bar classification
            df['is_bull_bar'] = (df['close'] > df['open']).astype(int)
            df['is_bear_bar'] = (df['close'] < df['open']).astype(int)
            df['is_doji'] = (abs(df['close'] - df['open']) / (df['hl_range'] + 1e-8) < 0.1).astype(int)
            
            # ============================================
            # 3. PERCENTAGE-BASED CHANGES (scale-invariant)
            # ============================================
            df['price_change_pct_1'] = df['close'].pct_change(1)
            df['price_change_pct_3'] = df['close'].pct_change(3)
            df['price_change_pct_5'] = df['close'].pct_change(5)
            
            # Returns from lags
            for lag in [1, 2, 3]:
                lagged_price = df['close'].shift(lag)
                df[f'return_from_lag_{lag}'] = (df['close'] - lagged_price) / (lagged_price + 1e-8)
            
            # ============================================
            # 4. NORMALIZED VOLUME (relative to own average)
            # ============================================
            avg_volume = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_normalized'] = df['volume'] / (avg_volume + 1e-8)
            
            for lag in [1, 2, 3]:
                lagged_vol = df['volume'].shift(lag)
                df[f'volume_normalized_lag_{lag}'] = lagged_vol / (avg_volume + 1e-8)
            
            # Volume ratios
            df['pos_volume'] = df['volume'].where(df['close'] > df['open'], 0)
            df['neg_volume'] = df['volume'].where(df['close'] <= df['open'], 0)
            df['volume_ratio'] = np.where(df['volume'] > 0, 
                                         df['pos_volume'] / df['volume'], 0.5)
            
            # Volume change percentages
            df['volume_change_pct_1'] = df['volume'].pct_change(1)
            
            # ============================================
            # 5. VOLATILITY AS PERCENTAGE
            # ============================================
            df['volatility_20'] = df['price_change_pct_1'].rolling(20).std()
            df['hl_range_pct'] = df['hl_range'] / (df['close'] + 1e-8)
            
        return df
    
    def calculate_market_regime_features(self, df):
        """
        Calculate regime-aware features WITHOUT separate training
        These features help the model adapt to different market conditions
        """
        df = df.copy()
        
        print("  Calculating market regime features...")
        
        # 1. VOLATILITY REGIME (3 states: low, medium, high)
        volatility = df['close'].pct_change().rolling(20).std()
        vol_percentiles = volatility.rolling(100, min_periods=20).quantile(0.5)
        
        df['vol_regime_low'] = (volatility < vol_percentiles * 0.7).astype(int)
        df['vol_regime_high'] = (volatility > vol_percentiles * 1.3).astype(int)
        df['vol_regime_medium'] = (~df['vol_regime_low'] & ~df['vol_regime_high']).astype(int)
        
        # Volatility trend (increasing/decreasing)
        df['vol_trend'] = volatility.diff().rolling(5).mean()
        
        # 2. TREND REGIME (3 states: uptrend, downtrend, sideways)
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        df['trend_strength'] = (sma_20 - sma_50) / (sma_50 + 1e-8)
        df['in_uptrend'] = (df['trend_strength'] > 0.02).astype(int)
        df['in_downtrend'] = (df['trend_strength'] < -0.02).astype(int)
        df['in_sideways'] = (~df['in_uptrend'] & ~df['in_downtrend']).astype(int)
        
        # Trend persistence (days in current trend)
        trend_direction = (df['trend_strength'] > 0).astype(int)
        trend_change = trend_direction.ne(trend_direction.shift())
        df['trend_days'] = df.groupby(trend_change.cumsum()).cumcount()
        
        # 3. MOMENTUM REGIME (accelerating/decelerating)
        momentum = df['close'].pct_change(10)
        df['momentum_regime'] = np.sign(momentum.diff())
        df['momentum_acceleration'] = momentum.diff().rolling(5).mean()
        
        # 4. VOLUME REGIME (high/low relative activity)
        avg_volume = df['volume'].rolling(20).mean()
        df['volume_regime_high'] = (df['volume'] > avg_volume * 1.5).astype(int)
        df['volume_surge'] = (df['volume'] > avg_volume * 2.0).astype(int)
        
        # 5. RANGE REGIME (expanding/contracting)
        price_range = df['high'] - df['low']
        avg_range = price_range.rolling(20).mean()
        df['range_expansion'] = (price_range > avg_range * 1.3).astype(int)
        df['range_contraction'] = (price_range < avg_range * 0.7).astype(int)
        
        # 6. COMPOSITE REGIME SCORE
        # Combine multiple signals into single score
        df['favorable_long_regime'] = (
            df['in_uptrend'] * 0.4 +
            df['vol_regime_low'] * 0.3 +
            df['volume_regime_high'] * 0.2 +
            (df['momentum_regime'] > 0).astype(int) * 0.1
        )
        
        df['favorable_short_regime'] = (
            df['in_downtrend'] * 0.4 +
            df['vol_regime_low'] * 0.3 +
            df['volume_regime_high'] * 0.2 +
            (df['momentum_regime'] < 0).astype(int) * 0.1
        )
        
        # 7. REGIME TRANSITION DETECTION
        # Detect when regime is changing (high uncertainty)
        vol_std = volatility.std()
        df['regime_uncertainty'] = (
            ((df['vol_trend'].abs() > vol_std) | (df['trend_days'] < 5))
        ).astype(int)
        
        print("  ✓ Regime features calculated")
        
        return df
    
    def calculate_adaptive_indicators(self, df):
        """
        Indicators that adapt to current regime
        More responsive in volatile regimes, more stable in calm regimes
        """
        df = df.copy()
        
        print("  Calculating adaptive indicators...")
        
        # Calculate volatility percentile for adaptive periods
        volatility = df['close'].pct_change().rolling(20).std()
        vol_percentile = volatility.rolling(100, min_periods=20).rank(pct=True)
        
        # Adaptive period lengths (stored for reference, not used in calculations yet)
        # These could be used to calculate adaptive RSI/MA in future versions
        df['adaptive_short_period'] = (10 * (1 - vol_percentile * 0.3)).clip(7, 15)
        df['adaptive_long_period'] = (20 * (1 - vol_percentile * 0.3)).clip(15, 30)
        
        # Calculate adaptive volatility measure
        # Faster response in high volatility, slower in low volatility
        df['adaptive_volatility'] = volatility * (1 + vol_percentile * 0.5)
        
        print("  ✓ Adaptive indicators calculated")
        
        return df
    
    def perform_feature_engineering(self, df, min_valid_rows=50, for_universal=True):
        """
        Complete feature engineering pipeline with regime detection
        If for_universal=True, generates price-scale invariant features
        """
        print("Starting feature engineering...")
        original_rows = len(df)
        
        # Prepare data
        df.columns = [col.lower() for col in df.columns]
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove duplicate indices
        if not df.index.is_unique:
            print("Warning: Duplicate index labels found. Removing duplicates.")
            df = df[~df.index.duplicated(keep='first')]
        
        # Calculate VWAP
        df = self.calculate_vwap_properly(df)
        
        print(f"Data preparation complete: {len(df)} rows ready for feature generation")
        
        # Define technical indicators strategy
        custom_strategy = ta.Strategy(
            name="Universal_Strategy",
            ta=[
                # Oscillators (0-100 bounded - scale invariant)
                {"kind": "rsi", "length": 14},
                {"kind": "stoch", "k": 14, "d": 3},
                {"kind": "cci", "length": 14},
                {"kind": "roc", "length": 10},
                {"kind": "willr", "length": 14},
                {"kind": "stochrsi", "length": 14},
                {"kind": "ao"},
                {"kind": "mom", "length": 10},
                
                # Trend indicators
                {"kind": "sma", "length": 10},
                {"kind": "sma", "length": 20},
                {"kind": "ema", "length": 10},
                {"kind": "ema", "length": 20},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "adx", "length": 14},
                {"kind": "aroon", "length": 14},
                
                # Volatility indicators
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "atr", "length": 14},
                
                # Volume indicators
                {"kind": "obv"},
                {"kind": "mfi", "length": 14},
                {"kind": "cmf", "length": 20},
                {"kind": "efi", "length": 13},
            ]
        )
        
        # Apply indicators
        df.ta.strategy(custom_strategy)
        
        # Add universal features
        df = self.calculate_universal_features(df)
        
        # Add regime detection features
        df = self.calculate_market_regime_features(df)
        
        # Add adaptive indicators
        df = self.calculate_adaptive_indicators(df)
        
        if for_universal:
            # Create relative MA features (scale-invariant)
            if 'SMA_10' in df.columns:
                df['close_vs_sma10'] = df['close'] / (df['SMA_10'] + 1e-8) - 1
            if 'EMA_10' in df.columns:
                df['close_vs_ema10'] = df['close'] / (df['EMA_10'] + 1e-8) - 1
            if 'SMA_20' in df.columns:
                df['close_vs_sma20'] = df['close'] / (df['SMA_20'] + 1e-8) - 1
            if 'vwap' in df.columns:
                df['close_vs_vwap'] = df['close'] / (df['vwap'] + 1e-8) - 1
            
            # ATR as percentage
            if 'ATRr_14' in df.columns:
                df['atr_pct'] = df['ATRr_14']
            elif 'atr_14' in df.columns:
                df['atr_pct'] = df['atr_14'] / (df['close'] + 1e-8)
            
            # Bollinger Band position (0-1 scale)
            if all(col in df.columns for col in ['BBL_20_2.0', 'BBU_20_2.0']):
                bb_range = df['BBU_20_2.0'] - df['BBL_20_2.0']
                df['bbp_20_2.0'] = np.where(bb_range != 0,
                                           (df['close'] - df['BBL_20_2.0']) / bb_range,
                                           0.5)
            
            # BB width as percentage
            if all(col in df.columns for col in ['BBL_20_2.0', 'BBU_20_2.0', 'BBM_20_2.0']):
                df['bbw_20_2.0'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / (df['BBM_20_2.0'] + 1e-8)
        
        # Handle NaN values
        df = self.handle_nan_values(df, min_valid_rows)
        
        # Reset index
        df = df.reset_index()
        
        print(f"Feature engineering complete. Original rows: {original_rows}, Final rows: {len(df)}")
        print(f"Features generated: {len(df.columns)}")
        
        # Print regime feature summary
        if 'vol_regime_low' in df.columns:
            print("\nRegime Distribution:")
            print(f"  Low Volatility: {df['vol_regime_low'].sum()} days ({df['vol_regime_low'].mean()*100:.1f}%)")
            print(f"  High Volatility: {df['vol_regime_high'].sum()} days ({df['vol_regime_high'].mean()*100:.1f}%)")
            print(f"  Uptrend: {df['in_uptrend'].sum()} days ({df['in_uptrend'].mean()*100:.1f}%)")
            print(f"  Downtrend: {df['in_downtrend'].sum()} days ({df['in_downtrend'].mean()*100:.1f}%)")
            print(f"  Sideways: {df['in_sideways'].sum()} days ({df['in_sideways'].mean()*100:.1f}%)")
        
        return df
    
    def handle_nan_values(self, df, min_valid_rows):
        """Handle NaN values in the feature-engineered data."""
        core_columns = ['open', 'high', 'low', 'close', 'volume']
        available_core = [col for col in core_columns if col in df.columns]
        
        df = df.dropna(subset=available_core)
        rows_after_basic_cleaning = len(df)
        print(f"Rows after ensuring core OHLCV data is complete: {rows_after_basic_cleaning}")
        
        if rows_after_basic_cleaning < min_valid_rows:
            raise ValueError(f"Insufficient data: {rows_after_basic_cleaning} rows")
        
        # Fill NaN values strategically
        volume_indicators = ['OBV', 'MFI_14', 'CMF_20', 'EFI_13']
        for col in volume_indicators:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        momentum_indicators = ['RSI_14', 'CCI_14_0.015', 'WILLR_14']
        for col in momentum_indicators:
            if col in df.columns:
                if 'RSI' in col:
                    df[col] = df[col].fillna(50)
                elif 'CCI' in col:
                    df[col] = df[col].fillna(0)
                elif 'WILLR' in col:
                    df[col] = df[col].fillna(-50)
        
        # Fill regime features with neutral values
        regime_cols = ['vol_regime_low', 'vol_regime_high', 'vol_regime_medium',
                      'in_uptrend', 'in_downtrend', 'in_sideways',
                      'volume_regime_high', 'range_expansion', 'range_contraction',
                      'regime_uncertainty']
        for col in regime_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Forward fill remaining
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        final_nan_count = df.isnull().sum().sum()
        if final_nan_count > 0:
            print(f"Warning: {final_nan_count} NaN values remain")
            # Show which columns have NaNs
            nan_cols = df.columns[df.isnull().any()].tolist()
            print(f"  Columns with NaNs: {nan_cols[:10]}")
        else:
            print("✓ No NaN values in final dataset")
        
        return df
    
    def get_universal_feature_columns(self, df):
        """
        Get feature columns suitable for universal training
        Excludes absolute price features, includes regime features
        """
        # Features to EXCLUDE (price-dependent)
        exclude_features = [
            'timestamp', 'close', 'open', 'high', 'low', 'volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20',
            'DEMA_10', 'DEMA_20', 'vwap', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'ATR_14', 'hl_range', 'oc_range',
            'DCL_20', 'DCM_20', 'DCU_20',
            'adaptive_short_period', 'adaptive_long_period'  # These are metadata
        ]
        
        # Get all columns
        all_cols = df.columns.tolist()
        
        # Filter out excluded features
        universal_features = [col for col in all_cols 
                            if col not in exclude_features 
                            and not col.startswith('_')
                            and not col.startswith('KC')  # Keltner Channels
                            and col not in ['date', 'Date']]
        
        # Count regime features
        regime_features = [f for f in universal_features if any(x in f.lower() for x in 
                          ['regime', 'trend_strength', 'trend_days', 'favorable', 'uncertainty'])]
        
        print(f"Universal features identified: {len(universal_features)}")
        print(f"  Including {len(regime_features)} regime-aware features")
        
        return universal_features
    
    def prepare_stock_for_universal_training(self, stock_data, stock_name):
        """
        Prepare a single stock for universal training.
        Adds metadata and ensures readiness, keeps all columns intact.
        """
        print(f"\nPreparing {stock_name} for universal training...")
        
        # Get universal features to report the count
        universal_features = self.get_universal_feature_columns(stock_data)
        
        # Keep original dataframe intact
        prepared_data = stock_data.copy()
        
        # Add stock identifier (for tracking, not as feature)
        prepared_data['_stock_name'] = stock_name
        
        # Calculate difficulty score for this stock
        difficulty = self.calculate_stock_difficulty(prepared_data)
        prepared_data['_difficulty_score'] = difficulty
        
        print(f"  Available universal features: {len(universal_features)}")
        print(f"  Total columns available: {len(prepared_data.columns)}")
        print(f"  Difficulty score: {difficulty:.3f}")
        
        return prepared_data
    
    def calculate_stock_difficulty(self, stock_data):
        """
        Calculate difficulty score for a stock
        Used for curriculum learning
        """
        try:
            returns = stock_data['close'].pct_change()
            
            # Volatility component
            volatility = returns.std() * np.sqrt(252)
            
            # Trend consistency
            ma_20 = stock_data['close'].rolling(20).mean()
            above_ma = (stock_data['close'] > ma_20).sum() / len(stock_data)
            trend_consistency = 1 - abs(above_ma - 0.5) * 2
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Composite difficulty (0-1, higher = harder)
            difficulty = (
                volatility * 0.4 +
                trend_consistency * 0.3 +
                max_drawdown * 0.3
            )
            
            return min(1.0, max(0.0, difficulty))
        except:
            return 0.5  # Default to medium difficulty
    
    def prepare_all_stocks_for_universal_training(self, all_stock_data):
        """
        Prepare all stocks for universal training
        """
        print("\n" + "="*80)
        print("PREPARING ALL STOCKS FOR UNIVERSAL TRAINING")
        print("="*80)
        
        normalized_stocks = {}
        difficulty_scores = {}
        
        for stock_name, stock_data in all_stock_data.items():
            try:
                normalized_data = self.prepare_stock_for_universal_training(
                    stock_data.copy(), stock_name
                )
                normalized_stocks[stock_name] = normalized_data
                difficulty_scores[stock_name] = normalized_data['_difficulty_score'].iloc[0]
                
            except Exception as e:
                print(f"  ERROR preparing {stock_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ Successfully prepared {len(normalized_stocks)} stocks")
        
        # Print difficulty ranking
        if difficulty_scores:
            print("\nStock Difficulty Ranking (for curriculum learning):")
            sorted_stocks = sorted(difficulty_scores.items(), key=lambda x: x[1])
            for stock, score in sorted_stocks:
                difficulty_label = "EASY" if score < 0.3 else "MEDIUM" if score < 0.6 else "HARD"
                print(f"  {stock:15s}: {score:.3f} [{difficulty_label}]")
        
        # Verify feature consistency
        self._verify_feature_consistency(normalized_stocks)
        
        return normalized_stocks
    
    def _verify_feature_consistency(self, normalized_stocks):
        """Verify all stocks have the same features"""
        print("\nVerifying feature consistency...")
        
        if not normalized_stocks:
            print("  WARNING: No stocks to verify")
            return
        
        first_stock = list(normalized_stocks.values())[0]
        reference_features = sorted([c for c in first_stock.columns if not c.startswith('_')])
        
        print(f"  Reference features: {len(reference_features)}")
        
        all_consistent = True
        for stock_name, stock_data in normalized_stocks.items():
            stock_features = sorted([c for c in stock_data.columns if not c.startswith('_')])
            
            if stock_features != reference_features:
                all_consistent = False
                missing = set(reference_features) - set(stock_features)
                extra = set(stock_features) - set(reference_features)
                
                print(f"\n  WARNING: {stock_name} inconsistent")
                if missing:
                    print(f"    Missing: {list(missing)[:5]}")
                if extra:
                    print(f"    Extra: {list(extra)[:5]}")
        
        if all_consistent:
            print("  ✓ All stocks have consistent features!")
    
    def get_common_features(self, normalized_stocks):
        """Get features common to ALL stocks"""
        if not normalized_stocks:
            return []
        
        # Get universal features from first stock
        first_stock_df = list(normalized_stocks.values())[0]
        potential_universal_features = self.get_universal_feature_columns(first_stock_df)
        
        common_features = set(potential_universal_features)
        
        # Intersect with all other stocks
        for stock_data in normalized_stocks.values():
            stock_features = set(stock_data.columns)
            common_features.intersection_update(stock_features)
        
        common_features = sorted(list(common_features))
        
        # Separate regime features for reporting
        regime_features = [f for f in common_features if any(x in f.lower() for x in 
                          ['regime', 'trend_strength', 'trend_days', 'favorable', 'uncertainty'])]
        
        print(f"\nFound {len(common_features)} common universal features across all stocks.")
        print(f"  Including {len(regime_features)} regime-aware features:")
        for rf in regime_features:
            print(f"    - {rf}")
        
        return common_features
    
    def load_featured_data(self, file_path, perform_feature_engineering=True):
        """Load trading data with optional feature engineering"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"Loading data from: {file_path}")
        data = pd.read_csv(file_path)
        
        if perform_feature_engineering:
            data = self.perform_feature_engineering(data, for_universal=True)
        else:
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp').reset_index(drop=True)
            
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            data.fillna(0, inplace=True)
        
        print(f"Loaded data. Shape: {data.shape}")
        return data
    
    def get_regime_features_for_state(self, data_row):
        """
        Extract regime features from a data row for environment state
        This is called by trading_environment.py
        """
        regime_features = [
            data_row.get('vol_regime_low', 0),
            data_row.get('vol_regime_high', 0),
            data_row.get('in_uptrend', 0),
            data_row.get('in_downtrend', 0),
            data_row.get('favorable_long_regime', 0),
            data_row.get('favorable_short_regime', 0),
            data_row.get('regime_uncertainty', 0),
        ]
        
        return regime_features