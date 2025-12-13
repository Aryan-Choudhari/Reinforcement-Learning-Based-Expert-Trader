"""
Advanced risk management for trading positions
"""
import numpy as np
import pandas as pd

class AdvancedRiskManager:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.INITIAL_CASH
        self.max_portfolio_risk = config.MAX_PORTFOLIO_RISK
        self.volatility_lookback = config.VOLATILITY_LOOKBACK
        self.max_positions = config.MAX_POSITIONS

    def calculate_position_size(self, env, entry_price, stop_loss_price, 
                              position_type='long', confidence_score=1.2):
        """Calculate optimal position size for long or short positions"""
        portfolio_value = env.get_portfolio_value()
        
        # Calculate risk per share based on position type
        if position_type.lower() == 'long':
            risk_per_share = abs(entry_price - stop_loss_price)
        else:  # short
            risk_per_share = abs(stop_loss_price - entry_price)
        
        if risk_per_share == 0:
            return 0

        # Enhanced capital deployment
        total_positions = len(env.long_positions) + len(env.short_positions)
        remaining_positions = max(1, self.max_positions - total_positions)
        
        # Use 92% of capital aggressively
        available_capital = portfolio_value * 0.92
        target_position_value = available_capital / remaining_positions

        # Risk-based sizing
        max_risk_per_position = portfolio_value * self.max_portfolio_risk
        risk_based_size = max_risk_per_position / risk_per_share

        # Value-based sizing
        value_based_size = target_position_value / entry_price

        # Use the more aggressive, but respect risk limits
        position_size = min(value_based_size, risk_based_size * 1.8)

        # Ensure minimum meaningful position size
        min_position_value = portfolio_value * 0.15
        min_position_size = min_position_value / entry_price
        position_size = max(position_size, min_position_size)

        # Ensure we have enough cash
        required_cash = position_size * entry_price * 1.001
        if required_cash > env.cash:
            position_size = env.cash / (entry_price * 1.001)

        return max(0, int(position_size * confidence_score))

    def calculate_dynamic_stop_loss(self, env, entry_price, position_type='long'):
        """Calculate dynamic stop loss for long and short positions"""
        atr = self._calculate_atr(env)
        volatility = self._calculate_recent_volatility(env)

        # Adaptive stop loss based on volatility
        if volatility > 0.4:
            atr_multiplier = 2.8
        elif volatility > 0.25:
            atr_multiplier = 2.2
        else:
            atr_multiplier = 1.9

        if position_type.lower() == 'long':
            stop_loss = entry_price - (atr * atr_multiplier)
            return max(0, stop_loss)
        else:  # short position
            stop_loss = entry_price + (atr * atr_multiplier)
            return stop_loss

    def _calculate_recent_volatility(self, env):
        """Calculate recent price volatility (annualized)"""
        current_step = env.current_step
        lookback = min(self.volatility_lookback, current_step)
        
        if lookback < 2:
            return 0.2

        prices = env.data.iloc[current_step-lookback:current_step+1]['close']
        returns = prices.pct_change().dropna()
        
        if len(returns) < 2:
            return 0.2

        return returns.std() * np.sqrt(252)

    def _calculate_atr(self, env, period=14):
        """Calculate Average True Range"""
        current_step = env.current_step
        lookback = min(period, current_step)
        
        if lookback < 2:
            return env.data.iloc[current_step]['close'] * 0.015

        data_slice = env.data.iloc[current_step-lookback:current_step+1]
        
        high_low = data_slice['high'] - data_slice['low']
        high_close = abs(data_slice['high'] - data_slice['close'].shift(1))
        low_close = abs(data_slice['low'] - data_slice['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.mean()
        
        return atr if not np.isnan(atr) else data_slice['close'].iloc[-1] * 0.015