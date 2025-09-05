"""
Enhanced Configuration settings for Expert Trader PPO System
"""

class TradingConfig:
    # PPO Configuration - Enhanced for expert trader behavior
    LR = 3e-4  # Learning rate
    GAMMA = 0.99  # Higher discount factor for longer-term thinking
    CLIP_EPSILON = 0.2  # Standard PPO clipping
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.03  # Higher exploration initially
    MAX_GRAD_NORM = 0.8
    
    # Trading Configuration - Expert Trader Focus
    INITIAL_CASH = 100000
    TRANSACTION_COST = 0.0006
    MAX_POSITIONS = 6  # Concentrated positions like expert traders
    MAX_PORTFOLIO_RISK = 0.035  # Conservative risk per position
    VOLATILITY_LOOKBACK = 25
    
    # Training Configuration - Expert Learning Phases
    EPISODES = 720  # More episodes for complex expert behaviors
    WALK_FORWARD_STEPS = 4
    PATIENCE = 65  # More patience for expert behavior development
    UPDATE_FREQUENCY = 768
    
    # Data Configuration
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 0.875
    MIN_VALID_ROWS = 80
    SELECTED_FEATURES = 45
    
    # Expert Trader Parameters
    MIN_EXPERT_HOLD_DAYS = 5  # Minimum holding period for trend positions
    BREAKEVEN_EXIT_PERCENTAGE = 0.25  # Exit 25% at breakeven
    STOP_ADJUSTMENT_THRESHOLD = 0.6  # Trend strength needed for adjustment
    POSITION_MANAGEMENT_BONUS = 3.0  # Reward for intelligent management
    EXPERT_TREND_THRESHOLD = 0.5  # Threshold for expert trend recognition
    
    # Risk Management - Expert Style
    GENEROUS_STOP_MULTIPLIER = 4.0  # Wider stops in strong trends
    BREAKEVEN_BUFFER = 0.001  # Small buffer when moving to breakeven
    MIN_ADJUSTMENT_HOLD = 3  # Days to hold after stop adjustment
    
    # Market Bias Parameters
    STRONG_TREND_THRESHOLD = 0.6  # Strong trend identification
    CORRECTION_RSI_OVERSOLD = 40  # RSI level for correction buying
    CORRECTION_RSI_OVERBOUGHT = 60  # RSI level for correction selling
    RANGING_MARKET_THRESHOLD = 0.3  # Trend alignment for ranging markets