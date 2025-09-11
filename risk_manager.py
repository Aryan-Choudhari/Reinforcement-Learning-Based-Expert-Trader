"""
Enhanced risk management with expert human trader-like position management
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
        """Enhanced position sizing with trend awareness"""
        portfolio_value = env.get_portfolio_value()
        
        # Get trend information
        current_step = env.current_step
        trend_multiplier = 1.0
        
        if current_step < len(env.data):
            trend_alignment = env.data.iloc[current_step].get('trend_alignment', 0)
            momentum_strength = env.data.iloc[current_step].get('momentum_strength', 0)
            vol_regime = env.data.iloc[current_step].get('vol_regime_numeric', 1)
            
            # Increase position sizes in strong, low-volatility trends
            if position_type.lower() == 'long':
                if trend_alignment > 0.6 and momentum_strength > 0.4 and vol_regime <= 1:
                    trend_multiplier = 1.8  # Significantly larger positions
                elif trend_alignment > 0.4 and momentum_strength > 0.2:
                    trend_multiplier = 1.4
                elif trend_alignment < -0.3:  # Against trend
                    trend_multiplier = 0.6
            
            elif position_type.lower() == 'short':
                if trend_alignment < -0.6 and momentum_strength < -0.4 and vol_regime <= 1:
                    trend_multiplier = 1.8
                elif trend_alignment < -0.4 and momentum_strength < -0.2:
                    trend_multiplier = 1.4
                elif trend_alignment > 0.3:  # Against trend
                    trend_multiplier = 0.6

        # Calculate risk per share
        if position_type.lower() == 'long':
            risk_per_share = abs(entry_price - stop_loss_price)
        else:
            risk_per_share = abs(stop_loss_price - entry_price)

        if risk_per_share == 0:
            return 0

        # Enhanced capital deployment
        long_positions = len(env.long_positions) if hasattr(env, 'long_positions') else 0
        short_positions = len(env.short_positions) if hasattr(env, 'short_positions') else 0
        total_positions = long_positions + short_positions

        remaining_positions = max(1, self.max_positions - total_positions)
        available_capital = portfolio_value * 0.92  # Slightly more aggressive

        target_position_value = available_capital / remaining_positions

        # More conservative position sizing
        max_risk_per_position = portfolio_value * self.max_portfolio_risk * trend_multiplier * 0.8
        risk_based_size = max_risk_per_position / risk_per_share
        
        # Ensure minimum position size is meaningful but not too small
        min_position_value = portfolio_value * 0.08  # Reduced from 15% to 8%
        min_position_size = min_position_value / entry_price

        # Value-based sizing
        value_based_size = target_position_value / entry_price

        # Enhanced position sizing
        position_size = min(value_based_size, risk_based_size * 2.2)

        # Minimum meaningful position
        min_position_value = portfolio_value * 0.15
        min_position_size = min_position_value / entry_price
        position_size = max(position_size, min_position_size)

        # Cash constraint check
        if position_type.lower() == 'long':
            required_cash = position_size * entry_price * 1.001
            if required_cash > env.cash:
                position_size = env.cash / (entry_price * 1.001)

        final_size = max(0, int(position_size * confidence_score * trend_multiplier))
        return final_size

    def calculate_dynamic_stop_loss(self, env, entry_price, position_type='long'):
        """Trend-aware stop loss calculation"""
        atr = self._calculate_atr(env)
        volatility = self._calculate_recent_volatility(env)
        
        # Get trend strength for adaptive stops
        current_step = env.current_step
        trend_strength = 0
        if current_step < len(env.data):
            trend_strength = abs(env.data.iloc[current_step].get('trend_alignment', 0))

        # Adaptive stop loss based on volatility and trend
        if volatility > 0.4:
            base_multiplier = 3.2
        elif volatility > 0.25:
            base_multiplier = 2.6
        else:
            base_multiplier = 2.2

        # Wider stops in strong trends
        if trend_strength > 0.6:
            atr_multiplier = base_multiplier * 1.4  # 40% wider stops
        elif trend_strength > 0.4:
            atr_multiplier = base_multiplier * 1.2  # 20% wider stops
        else:
            atr_multiplier = base_multiplier

        if position_type.lower() == 'long':
            stop_loss = entry_price - (atr * atr_multiplier)
            return max(0, stop_loss)
        else:
            stop_loss = entry_price + (atr * atr_multiplier)
            return stop_loss

    def should_adjust_stop_loss(self, env, current_price, position):
        """Determine if stop loss should be adjusted - RESPECTS minimum holding period"""
        current_step = env.current_step
        holding_period = current_step - position.get('entry_step', current_step)
        
        # **FIXED: Only adjust after minimum holding period**
        if holding_period < self.config.MIN_HOLDING_PERIOD:
            return False, None
        
        if current_step < 20 or current_step >= len(env.data):
            return False, None

        current_bar = env.data.iloc[current_step]
        trend_alignment = current_bar.get('trend_alignment', 0)
        momentum_strength = current_bar.get('momentum_strength', 0)
        vol_regime = current_bar.get('vol_regime_numeric', 1)
        rsi = current_bar.get('rsi_14', 50)

        entry_price = position['entry_price']

        if position['position_type'] == 'LONG':
            current_loss = (entry_price - current_price) / entry_price
            
            if (current_loss > 0.02 and
                trend_alignment > 0.6 and
                momentum_strength > 0.4 and
                vol_regime <= 1 and
                rsi < 40):
                atr = self._calculate_atr(env)
                new_stop = entry_price - (atr * 4.0)
                return True, max(0, new_stop)

        elif position['position_type'] == 'SHORT':
            current_loss = (current_price - entry_price) / entry_price
            
            if (current_loss > 0.02 and
                trend_alignment < -0.6 and
                momentum_strength < -0.4 and
                vol_regime <= 1 and
                rsi > 60):
                atr = self._calculate_atr(env)
                new_stop = entry_price + (atr * 4.0)
                return True, new_stop

        return False, None

    def should_exit_partial_position(self, env, current_price, position):
        """Check partial exit - ONLY after minimum holding period"""
        entry_price = position['entry_price']
        position_type = position['position_type']
        holding_days = env.current_step - position.get('entry_step', env.current_step)

        # **FIXED: Respect minimum holding period**
        if holding_days < self.config.MIN_HOLDING_PERIOD:
            return False

        if position_type == 'LONG':
            price_diff = (current_price - entry_price) / entry_price
            if -0.005 <= price_diff <= 0.005:
                return True
        elif position_type == 'SHORT':
            price_diff = (entry_price - current_price) / entry_price
            if -0.005 <= price_diff <= 0.005:
                return True

        return False

    def should_move_to_breakeven(self, env, position):
        """Move stop to breakeven - ONLY after minimum holding + adjustment"""
        holding_days = env.current_step - position.get('entry_step', env.current_step)

        # **FIXED: Require minimum holding + stop adjustment**
        if (holding_days >= self.config.MIN_HOLDING_PERIOD + 2 and
            position.get('stop_adjusted', False) and
            not position.get('moved_to_breakeven', False)):
            return True

        return False

    def calculate_market_bias(self, env):
        """NEW: Calculate overall market bias for position direction preference"""
        current_step = env.current_step
        if current_step < 50 or current_step >= len(env.data):
            return 'neutral'
            
        current_bar = env.data.iloc[current_step]
        trend_alignment = current_bar.get('trend_alignment', 0)
        momentum_strength = current_bar.get('momentum_strength', 0)
        vol_regime = current_bar.get('vol_regime_numeric', 1)
        rsi = current_bar.get('rsi_14', 50)
        
        # Look at recent price action
        recent_data = env.data.iloc[current_step-20:current_step+1]
        price_momentum = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # Strong bullish bias
        if (trend_alignment > 0.5 and 
            momentum_strength > 0.3 and 
            price_momentum > 0.02 and
            rsi < 70):
            return 'bullish'
            
        # Strong bearish bias  
        elif (trend_alignment < -0.5 and 
              momentum_strength < -0.3 and 
              price_momentum < -0.02 and
              rsi > 30):
            return 'bearish'
            
        # Correction in uptrend (buy the dip)
        elif (trend_alignment > 0.3 and 
              price_momentum < -0.01 and 
              rsi < 45):
            return 'correction_buy'
            
        # Rally in downtrend (sell the rally)
        elif (trend_alignment < -0.3 and 
              price_momentum > 0.01 and 
              rsi > 55):
            return 'correction_sell'
            
        # Sideways/ranging market
        elif (abs(trend_alignment) < 0.3 and 
              abs(momentum_strength) < 0.2 and
              vol_regime >= 1):
            return 'ranging'
            
        return 'neutral'

    def get_position_direction_preference(self, env):
        """NEW: Get preferred position direction based on market conditions"""
        bias = self.calculate_market_bias(env)
        
        preferences = {
            'bullish': {'long': 0.8, 'short': 0.2},
            'bearish': {'long': 0.2, 'short': 0.8},
            'correction_buy': {'long': 0.9, 'short': 0.1},
            'correction_sell': {'long': 0.1, 'short': 0.9},
            'ranging': {'long': 0.4, 'short': 0.4},  # Favor mean reversion
            'neutral': {'long': 0.5, 'short': 0.5}
        }
        
        return preferences.get(bias, {'long': 0.5, 'short': 0.5}), bias

    def _calculate_recent_volatility(self, env):
        """Calculate recent price volatility"""
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