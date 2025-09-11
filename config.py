"""
Enhanced PPO Configuration with Specialized Agents
"""
class TradingConfig:
    # Environment
    INITIAL_CASH = 100000
    TRANSACTION_COST = 0.001
    MAX_POSITIONS = 8
    MAX_PORTFOLIO_RISK = 0.03
    VOLATILITY_LOOKBACK = 20
    
    # Enhanced PPO Parameters
    LR = 1e-4
    GAMMA = 0.99
    LAMBDA_GAE = 0.95
    CLIP_EPSILON = 0.25
    ENTROPY_COEF = 0.12
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    VALUE_CLIP = True
    
    # Training
    UPDATE_FREQUENCY = 512
    WALK_FORWARD_STEPS = 5
    PATIENCE = 40
    
    # Specialized Agent Parameters
    LONG_AGENT_WEIGHT = 0.4
    SHORT_AGENT_WEIGHT = 0.4
    COMBINED_WEIGHT = 0.2
    
    # **FIXED: Enhanced Trading Behavior with Minimum Holding**
    MIN_HOLDING_PERIOD = 3  # Minimum days to hold any position
    BREAKEVEN_BUFFER = 0.025
    STOP_ADJUSTMENT_THRESHOLD = 0.025
    
    # Data Processing
    MIN_VALID_ROWS = 100
    SELECTED_FEATURES = 25
    
    # **NEW: Verbosity Control**
    VERBOSE_TRAINING = False  # Reduce console output
    PHASE_SUMMARY_ONLY = True  # Only show phase summaries
    METRIC_REPORT_INTERVAL = 20  # Less frequent reporting
