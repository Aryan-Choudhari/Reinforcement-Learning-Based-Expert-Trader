"""
Enhanced Configuration settings for the Multi-Model Trading System
Includes regime-aware training and improved hyperparameters
"""

import random
import uuid
import numpy as np


class TradingConfig:
    # ===== ENHANCED TRAINING PARAMETERS =====
    
    # Base training (significantly increased for universal training)
    EPISODES = 10000  # Increased from 1000
    PATIENCE = 1000   # Increased from 100
    VALIDATION_INTERVAL = 1  # Validate every episode instead of 2
    
    # Learning rate schedule (more gradual decay)
    LR = 5e-5  # Slightly lower starting LR for stability (was 8e-5)
    LR_DECAY_STEP_SIZE = 6000  # Decay less frequently (was 2000)
    LR_GAMMA = 0.98  # More gradual decay (was 0.95)
    
    # Learning rate warmup
    USE_LR_WARMUP = True
    WARMUP_EPISODES = 5
    WARMUP_START_LR = 1e-6
    
    # Memory and replay (enhanced)
    MEMORY_SIZE = 150000  # Increased from 80000
    BATCH_SIZE = 128  # Increased from 64
    
    # Exploration schedule (longer exploration)
    EPS_DECAY_STEPS = 400000  # Increased from 200000
    EPS_START = 1.0
    EPS_END = 0.05  # Slightly lower minimum (was 0.08)
    
    # Advanced training techniques
    GAMMA = 0.99
    TAU = 5e-3  # Increased from 3e-3 for faster adaptation
    GRAD_CLIP_NORM = 1.0  # Gradient clipping
    USE_DOUBLE_DQN = True
    
    # Prioritized Experience Replay
    PRIORITY_ALPHA = 0.7  # Increased from 0.6
    PRIORITY_BETA_START = 0.4
    PRIORITY_BETA_END = 1.0
    PRIORITY_BETA_EPISODES = 500
    
    # Trading Configuration
    INITIAL_CASH = 100000
    TRANSACTION_COST = 0.0006
    MAX_POSITIONS = 6
    MAX_PORTFOLIO_RISK = 0.05
    VOLATILITY_LOOKBACK = 25
    
    # Position Management
    LARGE_POSITION_THRESHOLD = 0.25
    LARGE_POSITION_STOP_LOSS = 0.05
    MIN_PROFIT_TAKE_PCT = 0.15
    TARGET_CAPITAL_FREED_PCT = 0.20
    FIRST_TRADE_CAPITAL_USE = 0.75
    SUBSEQUENT_TRADE_CAPITAL_USE = 0.95
    MIN_CASH_BUFFER_PCT = 0.00

    # Walk-forward validation
    USE_CURRICULUM_LEARNING = False  # Set to True to enable
    WALK_FORWARD_STEPS = 3
    INITIAL_TRAIN_RATIO = 0.60
    VALIDATION_SIZE = 0.10
    STEP_SIZE = 0.10

    MIN_VALID_ROWS = 100
    SELECTED_FEATURES = 40
    
    # Multi-Model Configuration
    MODELS_TO_TRAIN = [
        'simple_dqn', 'simple_dropout', 'simple_residual',
        'dueling', 'lstm', 'attention',
        'deep_dueling', 'transformer', 'hybrid_cnn_lstm'
    ]
    
    # Feature Group Configuration
    USE_FEATURE_GROUPS = True
    
    # ===== IMPROVED MODEL-SPECIFIC CONFIGS FOR UNIVERSAL TRAINING =====
    UNIVERSAL_MODEL_CONFIG = {
        # Simple models - increased episodes
        'simple_dqn': {
            'episodes': 50,  # Increased from 10
            'batch_size': 128,
            'lr': 5e-5,
            'validation_interval': 2
        },
        'simple_dropout': {
            'episodes': 50,
            'batch_size': 128,
            'lr': 5e-5,
            'validation_interval': 2
        },
        'simple_residual': {
            'episodes': 50,
            'batch_size': 128,
            'lr': 5e-5,
            'validation_interval': 2
        },
        
        # Original models - balanced training
        'dueling': {
            'episodes': 80,  # Increased from 12
            'batch_size': 128,
            'lr': 4e-5,
            'validation_interval': 2
        },
        'lstm': {
            'episodes': 100,  # Increased from 15, LSTM needs more
            'batch_size': 64,  # Keep smaller for LSTM
            'lr': 3e-5,
            'validation_interval': 3
        },
        'attention': {
            'episodes': 80,
            'batch_size': 128,
            'lr': 4e-5,
            'validation_interval': 2
        },
        
        # Complex models - extended training
        'deep_dueling': {
            'episodes': 100,
            'batch_size': 128,
            'lr': 3e-5,
            'validation_interval': 3
        },
        'transformer': {
            'episodes': 120,  # Transformer needs most episodes
            'batch_size': 64,
            'lr': 2e-5,
            'validation_interval': 3
        },
        'hybrid_cnn_lstm': {
            'episodes': 100,
            'batch_size': 64,
            'lr': 3e-5,
            'validation_interval': 3
        }
    }
    
    # ===== CURRICULUM LEARNING CONFIGURATION =====
    CURRICULUM_PHASES = [
        {
            'name': 'easy_stocks',
            'description': 'Start with trending, low-volatility stocks',
            'episodes_fraction': 0.3,  # 30% of total episodes
            'stock_selection': 'low_volatility',
            'epsilon_start': 0.9
        },
        {
            'name': 'medium_stocks',
            'description': 'Add moderate volatility stocks',
            'episodes_fraction': 0.4,  # 40% of total episodes
            'stock_selection': 'medium_volatility',
            'epsilon_start': 0.7
        },
        {
            'name': 'all_stocks',
            'description': 'Train on all stocks including volatile ones',
            'episodes_fraction': 0.3,  # 30% of total episodes
            'stock_selection': 'all',
            'epsilon_start': 0.5
        }
    ]
    
    # ===== EARLY STOPPING =====
    EARLY_STOPPING_ENABLED = True
    EARLY_STOPPING_METRIC = 'universal_score'
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_PATIENCE = 150
    
    # Checkpointing
    SAVE_CHECKPOINT_INTERVAL = 10
    KEEP_BEST_N_CHECKPOINTS = 3
    
    # ===== DATA AUGMENTATION =====
    USE_DATA_AUGMENTATION = False  # Set to True to enable
    AUGMENTATION_TECHNIQUES = {
        'noise_injection': 0.001,
        'time_shift': True,
        'feature_dropout': 0.05,
    }
    
    @classmethod
    def get_model_config(cls, model_name):
        """Get model-specific configuration for universal training"""
        if model_name in cls.UNIVERSAL_MODEL_CONFIG:
            return cls.UNIVERSAL_MODEL_CONFIG[model_name]
        return {
            'episodes': cls.EPISODES,
            'batch_size': cls.BATCH_SIZE,
            'lr': cls.LR,
            'validation_interval': cls.VALIDATION_INTERVAL
        }
    
    @classmethod
    def apply_model_config(cls, model_name):
        """Apply model-specific configuration"""
        model_config = cls.get_model_config(model_name)
        
        original_values = {
            'EPISODES': cls.EPISODES,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'LR': cls.LR,
            'VALIDATION_INTERVAL': cls.VALIDATION_INTERVAL
        }
        
        cls.EPISODES = model_config.get('episodes', cls.EPISODES)
        cls.BATCH_SIZE = model_config.get('batch_size', cls.BATCH_SIZE)
        cls.LR = model_config.get('lr', cls.LR)
        cls.VALIDATION_INTERVAL = model_config.get('validation_interval', cls.VALIDATION_INTERVAL)
        
        return original_values
    
    @classmethod
    def restore_config(cls, original_values):
        """Restore original configuration values"""
        cls.EPISODES = original_values['EPISODES']
        cls.BATCH_SIZE = original_values['BATCH_SIZE']
        cls.LR = original_values['LR']
        cls.VALIDATION_INTERVAL = original_values['VALIDATION_INTERVAL']


class FeatureGroupConfig:
    """
    Enhanced feature groups including regime-aware features
    """
    
    # Group 1: Price Action & Momentum with Regime Context
    PRICE_MOMENTUM = {
        'name': 'price_momentum',
        'description': 'Core price action, momentum, and regime context',
        'features': [
            # Price-based (Normalized)
            'price_change_pct_1', 'price_change_pct_3', 'price_change_pct_5',
            'price_position', 'body_to_range',
            'is_bull_bar', 'is_bear_bar', 'is_doji',
            'close_normalized', 'open_normalized', 'high_normalized', 'low_normalized',
            
            # Momentum indicators
            'RSI_14', 'ROC_10', 'MOM_10',
            'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3', 'WILLR_14',
            
            # Moving averages (as ratios)
            'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
            
            # Regime features
            'momentum_regime', 'momentum_acceleration',
            'favorable_long_regime', 'favorable_short_regime',
        ]
    }
    
    # Group 2: Trend & Volatility with Regime Detection
    TREND_VOLATILITY = {
        'name': 'trend_volatility',
        'description': 'Trend identification, volatility, and regime states',
        'features': [
            # Trend indicators
            'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
            'ADX_14', 'DMP_14', 'DMN_14',
            'AROONU_14', 'AROOND_14', 'AROONOSC_14',
            'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
            
            # Volatility indicators
            'ATRr_14', 'volatility_20', 'adaptive_volatility',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'BBB_20_2.0', 'BBP_20_2.0',
            
            # Price relationships
            'close_vs_vwap',
            
            # Regime features
            'vol_regime_low', 'vol_regime_high', 'vol_regime_medium',
            'in_uptrend', 'in_downtrend', 'in_sideways',
            'trend_strength', 'trend_days',
            'regime_uncertainty',
        ]
    }
    
    # Group 3: Volume & Market Microstructure
    VOLUME_MICROSTRUCTURE = {
        'name': 'volume_microstructure',
        'description': 'Volume analysis and market microstructure with regime',
        'features': [
            # Volume features
            'volume_change_pct_1',
            'pos_volume', 'neg_volume', 'volume_ratio',
            'volume_normalized', 'volume_normalized_lag_1', 
            'volume_normalized_lag_2', 'volume_normalized_lag_3',
            
            # Volume indicators
            'OBV', 'MFI_14', 'CMF_20', 'EFI_13',
            
            # Price & VWAP
            'close_vs_vwap',
            'price_position', 'body_to_range',
            'is_bull_bar', 'is_bear_bar',
            
            # Moving averages
            'SMA_10', 'EMA_10',
            
            # Regime features
            'volume_regime_high', 'volume_surge',
            'range_expansion', 'range_contraction',
        ]
    }
    
    # Group 4: Mean Reversion & Oscillators
    MEAN_REVERSION = {
        'name': 'mean_reversion',
        'description': 'Oscillators, mean reversion, and regime awareness',
        'features': [
            # Oscillators
            'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'CCI_14_0.015', 'WILLR_14', 'AO_5_34',
            'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3',
            
            # Bollinger Bands
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'BBB_20_2.0', 'BBP_20_2.0',
            
            # Price relationships
            'close_vs_vwap',
            
            # Price action
            'price_change_pct_1', 'price_change_pct_3',
            'volatility_20',
            
            # Regime features
            'vol_regime_low', 'vol_regime_high',
            'in_sideways', 'regime_uncertainty',
        ]
    }
    
    # Group 5: Multi-Timeframe Composite
    MULTI_TIMEFRAME = {
        'name': 'multi_timeframe',
        'description': 'Multiple timeframe analysis with regime context',
        'features': [
            # Short-term
            'RSI_14', 'ROC_10',
            'return_from_lag_1', 'return_from_lag_2', 'return_from_lag_3',
            
            # Medium-term
            'SMA_10', 'EMA_10',
            'ATRr_14', 'ADX_14',
            'BBM_20_2.0', 'BBB_20_2.0',
            
            # Long-term
            'SMA_20', 'EMA_20',
            'volatility_20', 'CMF_20',
            
            # Price relationships
            'close_vs_vwap',
            
            # Core data
            'close_normalized',
            'price_change_pct_1', 'volume_normalized',
            
            # Regime features (multi-timeframe perspective)
            'trend_strength', 'trend_days',
            'vol_trend', 'adaptive_volatility',
            'favorable_long_regime', 'favorable_short_regime',
        ]
    }
    
    # Group 6: Comprehensive Balanced with Full Regime Suite
    COMPREHENSIVE = {
        'name': 'comprehensive',
        'description': 'Balanced mix of all indicators plus complete regime analysis',
        'features': [
            # Core price action
            'close_normalized', 'hl_range_pct', 'price_position',
            'is_bull_bar', 'is_bear_bar', 'price_change_pct_1',
            
            # Trend
            'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
            'MACD_12_26_9', 'ADX_14', 'AROONOSC_14',
            
            # Momentum
            'RSI_14', 'ROC_10', 'MOM_10',
            
            # Volatility
            'ATRr_14', 'volatility_20', 'BBB_20_2.0',
            
            # Volume
            'volume_normalized', 'OBV', 'MFI_14', 'CMF_20',
            
            # Price relationships
            'close_vs_vwap',
            
            # Oscillators
            'CCI_14_0.015', 'WILLR_14', 'AO_5_34',
            
            # Complete regime suite
            'vol_regime_low', 'vol_regime_high',
            'in_uptrend', 'in_downtrend', 'in_sideways',
            'trend_strength', 'trend_days',
            'momentum_regime', 'volume_regime_high',
            'favorable_long_regime', 'favorable_short_regime',
            'regime_uncertainty',
        ]
    }
    
    @classmethod
    def get_all_groups(cls):
        """Return all feature groups"""
        return {
            'price_momentum': cls.PRICE_MOMENTUM,
            'trend_volatility': cls.TREND_VOLATILITY,
            'volume_microstructure': cls.VOLUME_MICROSTRUCTURE,
            'mean_reversion': cls.MEAN_REVERSION,
            'multi_timeframe': cls.MULTI_TIMEFRAME,
            'comprehensive': cls.COMPREHENSIVE
        }
    
    @classmethod
    def get_group(cls, group_name):
        """Get specific feature group by name"""
        groups = cls.get_all_groups()
        if group_name not in groups:
            raise ValueError(f"Unknown group: {group_name}. Available: {list(groups.keys())}")
        return groups[group_name]
    
    @classmethod
    def filter_features(cls, data_columns, feature_list):
        """
        Filter feature list to only include columns that exist in data
        """
        available_features = []
        
        for feature in feature_list:
            if feature.endswith('_'):
                # Partial match
                matching = [col for col in data_columns if col.startswith(feature)]
                available_features.extend(matching)
            elif feature in data_columns:
                # Exact match
                available_features.append(feature)
        
        # Ensure 'close' is included
        if 'close' in data_columns and 'close' not in available_features:
            available_features.append('close')
        
        return list(set(available_features))
    
    @classmethod
    def get_model_feature_assignments(cls):
        """Assign feature groups to models for optimal diversity"""
        return {
            # Simple models
            'simple_dqn': 'price_momentum',
            'simple_dropout': 'trend_volatility',
            'simple_residual': 'volume_microstructure',
            
            # Original models
            'dueling': 'mean_reversion',
            'lstm': 'multi_timeframe',
            'attention': 'comprehensive',
            
            # Complex models
            'deep_dueling': 'comprehensive',
            'transformer': 'multi_timeframe',
            'hybrid_cnn_lstm': 'trend_volatility'
        }
    
    @classmethod
    def print_group_summary(cls):
        """Print summary of all feature groups"""
        groups = cls.get_all_groups()
        
        print("\n" + "="*80)
        print("FEATURE GROUPS SUMMARY (WITH REGIME FEATURES)")
        print("="*80)
        
        for name, group in groups.items():
            regime_features = [f for f in group['features'] if any(x in f.lower() 
                             for x in ['regime', 'trend_strength', 'trend_days', 
                                      'favorable', 'uncertainty', 'adaptive'])]
            
            print(f"\n{group['name'].upper()}")
            print(f"Description: {group['description']}")
            print(f"Total Features: {len(group['features'])}")
            print(f"Regime Features: {len(regime_features)}")
            if regime_features:
                print(f"  Regime features: {', '.join(regime_features[:5])}")
        
        print("\n" + "="*80)
        print("MODEL-FEATURE GROUP ASSIGNMENTS")
        print("="*80)
        
        assignments = cls.get_model_feature_assignments()
        for model, group in assignments.items():
            print(f"  {model:20s} -> {group}")
        
        print("="*80 + "\n")


class StockDifficultyRanker:
    """Rank stocks by difficulty for curriculum learning"""
    
    @staticmethod
    def calculate_difficulty_score(stock_data):
        """Calculate difficulty score (0-1, higher = harder)"""
        returns = stock_data['close'].pct_change()
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Trend consistency
        ma_20 = stock_data['close'].rolling(20).mean()
        trend_consistency = ((stock_data['close'] > ma_20).sum() / len(stock_data))
        trend_consistency = 1 - abs(trend_consistency - 0.5) * 2
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        difficulty = (
            volatility * 0.4 +
            trend_consistency * 0.3 +
            max_drawdown * 0.3
        )
        
        return min(1.0, max(0.0, difficulty))
    
    @staticmethod
    def rank_stocks_by_difficulty(stock_data_dict):
        """Rank all stocks by difficulty"""
        scores = {}
        for stock_name, data in stock_data_dict.items():
            try:
                score = StockDifficultyRanker.calculate_difficulty_score(data)
                scores[stock_name] = score
            except:
                scores[stock_name] = 0.5
        
        sorted_stocks = dict(sorted(scores.items(), key=lambda x: x[1]))
        
        print("\nStock Difficulty Ranking (for curriculum learning):")
        print("="*60)
        for stock, score in sorted_stocks.items():
            difficulty_label = "EASY" if score < 0.3 else "MEDIUM" if score < 0.6 else "HARD"
            print(f"  {stock:15s}: {score:.3f} [{difficulty_label}]")
        
        return sorted_stocks
    
    @staticmethod
    def select_stocks_by_difficulty(stock_data_dict, difficulty_level='all'):
        """Select stocks based on difficulty level"""
        difficulty_scores = StockDifficultyRanker.rank_stocks_by_difficulty(stock_data_dict)
        
        if difficulty_level == 'low_volatility':
            # Take easiest 30%
            n_stocks = max(2, len(stock_data_dict) // 3)
            selected = dict(list(difficulty_scores.items())[:n_stocks])
        
        elif difficulty_level == 'medium_volatility':
            # Take middle 50%
            start_idx = len(stock_data_dict) // 4
            end_idx = start_idx + (len(stock_data_dict) // 2)
            selected = dict(list(difficulty_scores.items())[start_idx:end_idx])
        
        elif difficulty_level == 'high_volatility':
            # Take hardest 30%
            n_stocks = max(2, len(stock_data_dict) // 3)
            selected = dict(list(difficulty_scores.items())[-n_stocks:])
        
        else:  # 'all'
            selected = difficulty_scores
        
        selected_stocks = {name: stock_data_dict[name] for name in selected.keys()}
        
        print(f"\nSelected {len(selected_stocks)} stocks for difficulty level: {difficulty_level}")
        return selected_stocks


if __name__ == "__main__":
    print("Enhanced Trading Configuration")
    print("="*80)
    print(f"Episodes: {TradingConfig.EPISODES}")
    print(f"Batch Size: {TradingConfig.BATCH_SIZE}")
    print(f"Learning Rate: {TradingConfig.LR}")
    print(f"Memory Size: {TradingConfig.MEMORY_SIZE}")
    print(f"Patience: {TradingConfig.PATIENCE}")
    print("\nFeature Groups:")
    FeatureGroupConfig.print_group_summary()