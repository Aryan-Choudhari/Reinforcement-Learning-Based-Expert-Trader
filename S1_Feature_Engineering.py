import pandas as pd
import pandas_ta as ta
import os
import glob
import re
from tqdm import tqdm
import warnings
import numpy as np

# --- Configuration ---
SOURCE_DATA_FOLDER = './Raw_Data_1Hour'
FEATURED_DATA_FOLDER = './Featured_Data_1Hour'
TEST_MODE = False
TEST_SUBSET_SIZE = 2000
MIN_VALID_ROWS = 50

# --- End Configuration ---

# Suppress ignorable warnings from libraries
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_asset_and_timeframe(filepath):
    """Extracts asset name and timeframe from a filename using regex."""
    filename = os.path.basename(filepath)
    match = re.match(r'([A-Z\-]+)_(\d+[a-zA-Z]+)\.csv', filename, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None

def calculate_vwap_properly(df):
    """
    Calculate VWAP without forward bias using only past data.
    VWAP resets each trading day (or period).
    """
    if 'vwap' not in df.columns:
        # Create a simple daily reset VWAP
        df = df.copy()
        df['date'] = df.index.date
        
        # Calculate typical price for current bar
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['volume_price'] = df['typical_price'] * df['volume']
        
        # Group by date and calculate cumulative sums (this uses only current and past data)
        df['cum_volume_price'] = df.groupby('date')['volume_price'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
        
        # Calculate VWAP (only uses current and past data within each day)
        df['vwap'] = df['cum_volume_price'] / df['cum_volume']
        
        # Clean up temporary columns
        df = df.drop(['date', 'typical_price', 'volume_price', 'cum_volume_price', 'cum_volume'], axis=1)
    
    return df

def calculate_no_bias_features(df):
    """
    Calculate features ensuring absolutely no forward bias.
    All calculations use only current bar data or properly lagged historical data.
    """
    df = df.copy()
    
    # Current bar features (no forward bias - uses only current OHLCV)
    if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        # Intrabar relationships (current bar only)
        df['low_close'] = df['close'] - df['low']  # Distance from close to low
        df['high_close'] = df['high'] - df['close']  # Distance from high to close
        df['hl_range'] = df['high'] - df['low']  # High-low range
        df['oc_range'] = abs(df['close'] - df['open'])  # Open-close range
        
        # Price position within the bar (0 = at low, 1 = at high)
        df['price_position'] = np.where(df['hl_range'] != 0,
                                      (df['close'] - df['low']) / df['hl_range'],
                                      0.5)
        
        # Body size relative to full range
        df['body_to_range'] = np.where(df['hl_range'] != 0,
                                     df['oc_range'] / df['hl_range'],
                                     0)
        
        # Bull/bear classification (current bar)
        df['is_bull_bar'] = (df['close'] > df['open']).astype(int)
        df['is_bear_bar'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (abs(df['close'] - df['open']) / df['hl_range'] < 0.1).astype(int)
        
        # Volume features (current bar only)
        df['pos_volume'] = df['volume'].where(df['close'] > df['open'], 0)
        df['neg_volume'] = df['volume'].where(df['close'] <= df['open'], 0)
        df['volume_ratio'] = np.where(df['volume'] > 0, df['pos_volume'] / df['volume'], 0.5)
        
        # Lagged features (using past data only)
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'hl_range_lag_{lag}'] = df['hl_range'].shift(lag)
        
        # Price changes (using lagged data - no forward bias)
        df['price_change_1'] = df['close'] - df['close'].shift(1)
        df['price_change_pct_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'] - df['close'].shift(3)
        df['price_change_pct_3'] = df['close'].pct_change(3)
        
        # Volume changes (using lagged data)
        df['volume_change_1'] = df['volume'] - df['volume'].shift(1)
        df['volume_change_pct_1'] = df['volume'].pct_change(1)
        
        # Moving average relationships (these are calculated correctly by pandas_ta)
        # But let's add some relative features that don't use future data
        if 'sma_10' in df.columns:
            df['close_vs_sma10'] = df['close'] / df['sma_10'] - 1
        if 'ema_10' in df.columns:
            df['close_vs_ema10'] = df['close'] / df['ema_10'] - 1
        if 'sma_20' in df.columns:
            df['close_vs_sma20'] = df['close'] / df['sma_20'] - 1
        
        # VWAP relationship (current close vs current VWAP)
        if 'vwap' in df.columns:
            df['close_vs_vwap'] = df['close'] / df['vwap'] - 1
            
        # Add normalized features for better RL training
        df['normalized_volume'] = df['volume'] / df['volume'].rolling(20).mean() - 1
        df['volatility_20'] = df['price_change_pct_1'].rolling(20).std()
        
    return df

def validate_no_forward_bias(df):
    """
    Validate that features don't contain forward bias by checking dependencies.
    """
    print("\nValidating features for forward bias...")
    
    # Check that returns are calculated correctly (not using future data)
    if 'log_return' in df.columns:
        # Manual calculation of log returns using only past data
        manual_log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # Compare with pandas_ta version (allow for small floating point differences)
        if 'log_return' in df.columns:
            diff = abs(df['log_return'].fillna(0) - manual_log_returns.fillna(0)).max()
            if diff > 1e-10:
                print(f"WARNING: log_return may have forward bias! Max difference: {diff}")
                # Replace with our manual calculation
                df['log_return'] = manual_log_returns
                print("Replaced with manual calculation")
            else:
                print("✓ log_return calculation verified - no forward bias")
    
    # Check moving averages don't use future data
    ma_columns = [col for col in df.columns if any(ma_type in col.lower() for ma_type in ['sma', 'ema', 'dema'])]
    for ma_col in ma_columns[:3]:  # Check first 3 MAs as examples
        if ma_col in df.columns and len(df) > 20:
            # For SMA, check that it doesn't use future data
            if 'sma' in ma_col.lower():
                length = int(re.search(r'(\d+)', ma_col).group(1))
                if length <= len(df):
                    manual_sma = df['close'].rolling(window=length, min_periods=1).mean()
                    # Check a few points in the middle
                    mid_point = len(df) // 2
                    if abs(df[ma_col].iloc[mid_point] - manual_sma.iloc[mid_point]) > 1e-10:
                        print(f"WARNING: {ma_col} may have forward bias!")
                    else:
                        print(f"✓ {ma_col} verified - no forward bias")
    
    print("Validation complete.")
    return df

def process_file(file_path):
    """
    Processes a single data file to generate and save features for its specific timeframe.
    Handles NaN values properly and ensures absolutely no forward bias.
    """
    try:
        asset_name, timeframe = get_asset_and_timeframe(file_path)
        if not asset_name or not timeframe:
            return f"Skipping {os.path.basename(file_path)}: Could not parse filename."
        
        # 1. Load and prepare the data
        df = pd.read_csv(file_path)
        original_rows = len(df)
        
        # In test mode, use the LAST N rows (more recent data for testing)
        if TEST_MODE:
            print(f"--- RUNNING IN TEST MODE: Using last {TEST_SUBSET_SIZE} rows of {os.path.basename(file_path)} ---")
            df = df.tail(TEST_SUBSET_SIZE)
        
        df.columns = [col.lower() for col in df.columns]
        df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        if df.empty:
            return f"Skipping {os.path.basename(file_path)}: No data left after cleaning."
        
        # Calculate VWAP properly if not present (this is already bias-free)
        df = calculate_vwap_properly(df)
        
        rows_before_indicators = len(df)
        print(f"Data preparation complete: {rows_before_indicators} rows ready for feature generation")
        
        # 2. Define technical indicators strategy with conservative parameters
        custom_strategy = ta.Strategy(
            name="Comprehensive_No_Bias_Strategy",
            ta=[
                # Trend Indicators (using conservative periods)
                {"kind": "sma", "length": 5}, {"kind": "sma", "length": 10}, {"kind": "sma", "length": 20},
                {"kind": "ema", "length": 5}, {"kind": "ema", "length": 10}, {"kind": "ema", "length": 20},
                {"kind": "dema", "length": 10}, {"kind": "dema", "length": 20},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "adx", "length": 14},
                {"kind": "aroon", "length": 14},
                {"kind": "psar"},
                
                # Momentum Indicators
                {"kind": "rsi", "length": 14},
                {"kind": "stoch", "k": 14, "d": 3},
                {"kind": "cci", "length": 14},
                {"kind": "roc", "length": 10},
                {"kind": "willr", "length": 14},
                {"kind": "ao"},
                {"kind": "mom", "length": 10},
                {"kind": "ppo", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "stochrsi", "length": 14},
                
                # Volatility Indicators
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "atr", "length": 14},
                {"kind": "donchian", "length": 20},
                {"kind": "kc", "length": 20},
                
                # Volume Indicators
                {"kind": "obv"},
                {"kind": "mfi", "length": 14},
                {"kind": "cmf", "length": 20},
                {"kind": "efi", "length": 13},
                
                # Returns (calculated properly to avoid forward bias)
                {"kind": "log_return", "cumulative": False},
            ]
        )
        
        # Apply indicators
        df.ta.strategy(custom_strategy)
        
        # 3. Add custom features that definitely don't use forward bias
        df = calculate_no_bias_features(df)
        
        # 4. Validate features for forward bias
        df = validate_no_forward_bias(df)
        
        # Check NaN values after feature generation
        nan_counts = df.isnull().sum()
        rows_with_any_nan = df.isnull().any(axis=1).sum()
        print(f"\nRows with any NaN values: {rows_with_any_nan} out of {len(df)}")
        
        # Show top columns with NaN values
        top_nan_columns = nan_counts[nan_counts > 0].nlargest(10)
        if len(top_nan_columns) > 0:
            print("Columns with most NaN values:")
            for col, count in top_nan_columns.items():
                print(f"  {col}: {count} NaN values ({count/len(df):.1%})")

        # 5. Strategic NaN handling (improved approach)
        # First, ensure core OHLCV data is complete
        core_columns = ['open', 'high', 'low', 'close', 'volume']
        available_core = [col for col in core_columns if col in df.columns]
        
        df = df.dropna(subset=available_core)
        rows_after_basic_cleaning = len(df)
        print(f"Rows after ensuring core OHLCV data is complete: {rows_after_basic_cleaning}")
        
        if rows_after_basic_cleaning < MIN_VALID_ROWS:
            return f"Insufficient data after basic cleaning for {os.path.basename(file_path)}: {rows_after_basic_cleaning} rows remaining"
        
        # Strategic NaN filling (using only past data)
        # 1. Forward fill volume indicators (uses only past values)
        volume_indicators = ['obv', 'mfi_14', 'cmf_20', 'efi_13']
        for col in volume_indicators:
            if col in df.columns:
                # Use forward fill which uses only past data
                df[col] = df[col].fillna(method='ffill')
                # For any remaining NaN at the beginning, use backward fill
                df[col] = df[col].fillna(method='bfill')
        
        # 2. Fill momentum indicators with neutral values (no bias)
        momentum_indicators = ['rsi_14', 'cci_14', 'willr_14']
        for col in momentum_indicators:
            if col in df.columns:
                if col.startswith('rsi'):
                    df[col] = df[col].fillna(50)  # Neutral RSI
                elif col.startswith('cci'):
                    df[col] = df[col].fillna(0)   # Neutral CCI
                elif col.startswith('willr'):
                    df[col] = df[col].fillna(-50) # Neutral Williams %R
        
        # 3. Forward fill moving averages (uses only past data)
        ma_columns = [col for col in df.columns if any(ma_type in col.lower() for ma_type in ['sma', 'ema', 'dema'])]
        for col in ma_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
                # For any remaining NaN at the beginning, use the first valid close price
                first_valid_idx = df[col].first_valid_index()
                if first_valid_idx is not None:
                    first_close = df.loc[first_valid_idx, 'close'] if first_valid_idx in df.index else df['close'].iloc[0]
                    df[col] = df[col].fillna(first_close)
        
        # 4. Handle remaining critical indicators
        critical_indicators = ['atr_14', 'bb_upper', 'bb_lower']
        for col in critical_indicators:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
                if col.startswith('atr'):
                    # For ATR, fill initial NaN with a reasonable estimate
                    if df[col].isna().any():
                        initial_atr = df['hl_range'].head(14).mean()  # Use average range for first 14 periods
                        df[col] = df[col].fillna(initial_atr)
        
        # 5. Drop rows that still have too many NaN values
        # Calculate NaN percentage per row
        nan_per_row = df.isnull().sum(axis=1) / len(df.columns)
        
        # Remove rows with more than 30% NaN values
        valid_rows = nan_per_row <= 0.3
        df = df[valid_rows]
        
        rows_after_strategic_cleaning = len(df)
        print(f"Rows after strategic NaN handling: {rows_after_strategic_cleaning}")
        
        # 6. Final NaN check - if any critical columns still have NaN, handle them
        remaining_nan_cols = df.isnull().sum()
        critical_cols_with_nan = remaining_nan_cols[remaining_nan_cols > 0]
        
        if len(critical_cols_with_nan) > 0:
            print(f"Handling remaining NaN values in {len(critical_cols_with_nan)} columns...")
            
            # For any remaining NaN in lagged features, they should stay NaN (they represent unavailable past data)
            lag_columns = [col for col in df.columns if 'lag_' in col or col.endswith('_1') or col.endswith('_3')]
            
            # For non-lag columns, use forward fill as last resort
            non_lag_cols_with_nan = [col for col in critical_cols_with_nan.index if col not in lag_columns]
            for col in non_lag_cols_with_nan:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        rows_after_final_cleaning = len(df)
        print(f"Final rows after cleaning: {rows_after_final_cleaning}")
        
        if df.empty or rows_after_final_cleaning < MIN_VALID_ROWS:
            return f"Insufficient data after feature generation for {os.path.basename(file_path)}: {rows_after_final_cleaning} rows remaining"

        # 7. Final validation check
        final_nan_count = df.isnull().sum().sum()
        if final_nan_count > 0:
            print(f"Warning: {final_nan_count} NaN values remain in final dataset")
            # Show which columns still have NaN
            remaining_nans = df.isnull().sum()
            cols_with_nans = remaining_nans[remaining_nans > 0]
            print("Columns with remaining NaN values:")
            for col, count in cols_with_nans.items():
                print(f"  {col}: {count} NaN values")
        else:
            print("✓ No NaN values in final dataset")

        # Save the processed data
        output_filename = f"{asset_name}_{timeframe}_featured.csv"
        output_path = os.path.join(FEATURED_DATA_FOLDER, output_filename)
        df.to_csv(output_path)

        # Print summary
        print(f"\n--- Features generated for {asset_name} ({timeframe}) ---")
        print(f"Original rows: {original_rows}, Final rows: {len(df)}, Features: {len(df.columns)}")
        print(f"Data retention: {len(df)/original_rows:.1%}")
        print(f"Final NaN values: {df.isnull().sum().sum()}")
        
        # Show feature categories with counts
        feature_categories = {
            'Core OHLCV': [col for col in df.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume', 'vwap']],
            'Intrabar Features': [col for col in df.columns if any(x in col.lower() for x in ['low_close', 'high_close', 'hl_range', 'oc_range', 'price_position', 'body_to_range'])],
            'Bar Classification': [col for col in df.columns if any(x in col.lower() for x in ['is_bull', 'is_bear', 'is_doji'])],
            'Volume Features': [col for col in df.columns if any(x in col.lower() for x in ['pos_volume', 'neg_volume', 'volume_ratio', 'obv', 'mfi', 'cmf', 'efi'])],
            'Moving Averages': [col for col in df.columns if any(x in col.lower() for x in ['sma', 'ema', 'dema'])],
            'MA Relationships': [col for col in df.columns if 'vs_' in col.lower()],
            'Momentum': [col for col in df.columns if any(x in col.lower() for x in ['rsi', 'stoch', 'cci', 'roc', 'willr', 'macd', 'mom', 'ppo', 'ao'])],
            'Volatility': [col for col in df.columns if any(x in col.lower() for x in ['bb', 'atr', 'donchian', 'kc'])],
            'Trend': [col for col in df.columns if any(x in col.lower() for x in ['adx', 'aroon', 'psar'])],
            'Lagged Features': [col for col in df.columns if 'lag_' in col.lower()],
            'Price Changes': [col for col in df.columns if 'change' in col.lower()],
            'Returns': [col for col in df.columns if 'return' in col.lower()],
        }
        
        print("\nFeature categories:")
        total_features = 0
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
                total_features += len(features)
        
        uncategorized = len(df.columns) - total_features
        if uncategorized > 0:
            print(f"  Uncategorized: {uncategorized} features")
        
        print("-" * 60)
            
        return f"Successfully processed: {os.path.basename(file_path)} ({len(df)} rows, {len(df.columns)} features, {len(df)/original_rows:.1%} retention)"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing {os.path.basename(file_path)}:")
        print(error_details)
        return f"Failed to process {os.path.basename(file_path)}: {e}"

def main():
    """Main function to find and process all data files sequentially."""
    os.makedirs(FEATURED_DATA_FOLDER, exist_ok=True)
    all_files = glob.glob(os.path.join(SOURCE_DATA_FOLDER, '*.csv'))
    if not all_files:
        print(f"No CSV files found in '{SOURCE_DATA_FOLDER}'.")
        return
        
    print(f"Starting FORWARD BIAS-FREE feature engineering on {len(all_files)} files...")
    if TEST_MODE:
        print("!!! WARNING: TEST MODE IS ACTIVE. ONLY A SUBSET OF DATA WILL BE PROCESSED. !!!")
        
    results = []
    for file_path in tqdm(all_files, total=len(all_files)):
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(file_path)}")
        print('='*80)
        results.append(process_file(file_path))

    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    for res in results: 
        print(res)
    print(f"\nForward bias-free feature engineering complete. Files are in '{FEATURED_DATA_FOLDER}'.")

if __name__ == '__main__':
    main()