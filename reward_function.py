"""
Enhanced reward function focused on expert human trader behavior
"""

import numpy as np

class ImprovedRewardFunction:
    def __init__(self, lookback_window=20, risk_free_rate=0.02):
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate

    def calculate_enhanced_reward(self, env, action, trade_occurred, current_step):
        """Enhanced reward function mimicking expert human trader behavior"""
        reward = 0.0
        current_price = env.data.iloc[current_step]['close']
        portfolio_value = env.get_portfolio_value()

        # 1. BENCHMARK COMPARISON REWARD (HIGHEST PRIORITY)
        if len(env.portfolio_value_history) > 1:
            agent_return = (portfolio_value - env.portfolio_value_history[-2]) / env.portfolio_value_history[-2]
            if current_step > 0:
                bnh_return = (current_price - env.data.iloc[current_step-1]['close']) / env.data.iloc[current_step-1]['close']
                excess_return = agent_return - bnh_return
                reward += excess_return * 200.0

        # 2. EXPERT POSITION MANAGEMENT REWARDS
        position_management_reward = self._calculate_position_management_reward(env, action, current_step)
        reward += position_management_reward

        # 3. MARKET BIAS ALIGNMENT REWARD
        market_bias_reward = self._calculate_market_bias_reward(env, action, current_step)
        reward += market_bias_reward

        # 4. TREND FOLLOWING WITH CONVICTION REWARD
        trend_reward = self._calculate_expert_trend_reward(env, action, current_step)
        reward += trend_reward

        # 5. PARTIAL EXIT AND RISK MANAGEMENT REWARD
        if trade_occurred and env.trades:
            last_trade = env.trades[-1]
            if last_trade['type'] == 'PARTIAL_SELL' or last_trade['type'] == 'PARTIAL_COVER':
                reward += 2.0  # Reward risk management through partial exits
                
            # Enhanced holding period bonus for full exits
            elif last_trade['type'] in ['SELL', 'COVER', 'STOP-LOSS', 'LIQUIDATE']:
                holding_reward = self._calculate_enhanced_holding_reward(env, last_trade, current_step)
                reward += holding_reward

        # 6. POSITION SIZING INTELLIGENCE
        sizing_reward = self._calculate_position_sizing_reward(env, action)
        reward += sizing_reward

        # 7. STOP LOSS ADJUSTMENT REWARD
        adjustment_reward = self._calculate_stop_adjustment_reward(env)
        reward += adjustment_reward

        return np.clip(reward, -15.0, 15.0)

    def _calculate_position_management_reward(self, env, action, current_step):
        """Reward intelligent position management like an expert trader"""
        if current_step < 20:
            return 0.0

        current_bar = env.data.iloc[current_step]
        trend_alignment = current_bar.get('trend_alignment', 0)
        momentum_strength = current_bar.get('momentum_strength', 0)
        
        reward = 0.0

        # Reward holding positions with adjusted stops in strong trends
        for position in env.long_positions:
            if position.get('stop_adjusted', False) and trend_alignment > 0.5:
                holding_days = current_step - position.get('entry_step', current_step)
                if holding_days >= 3:  # Reward patience after adjustment
                    reward += min(2.0, holding_days / 10.0)

        for position in env.short_positions:
            if position.get('stop_adjusted', False) and trend_alignment < -0.5:
                holding_days = current_step - position.get('entry_step', current_step)
                if holding_days >= 3:
                    reward += min(2.0, holding_days / 10.0)

        # Reward breakeven management
        for position in env.long_positions + env.short_positions:
            if position.get('moved_to_breakeven', False) and position.get('partial_exit_done', False):
                reward += 1.5  # Good risk management

        return reward

    def _calculate_market_bias_reward(self, env, action, current_step):
        """Reward alignment with market bias like an expert trader"""
        if current_step < 30:
            return 0.0

        direction_prefs, market_bias = env.risk_manager.get_position_direction_preference(env)
        reward = 0.0

        # Reward actions aligned with market bias
        if market_bias == 'bullish' and action == 1:  # Buy in bullish market
            reward += 2.0
        elif market_bias == 'bearish' and action == 2:  # Short in bearish market
            reward += 2.0
        elif market_bias == 'correction_buy' and action == 1:  # Buy the dip
            reward += 2.5
        elif market_bias == 'correction_sell' and action == 2:  # Sell the rally
            reward += 2.5
        elif market_bias == 'ranging' and action == 0:  # Patience in ranging markets
            reward += 1.0

        # Penalize actions against strong market bias
        elif market_bias == 'bullish' and action == 2:  # Short in strong bull market
            reward -= 2.0
        elif market_bias == 'bearish' and action == 1:  # Buy in strong bear market
            reward -= 2.0

        return reward

    def _calculate_expert_trend_reward(self, env, action, current_step):
        """Enhanced trend following reward with expert trader conviction"""
        if current_step < 20:
            return 0.0

        current_bar = env.data.iloc[current_step]
        trend_alignment = current_bar.get('trend_alignment', 0)
        momentum_strength = current_bar.get('momentum_strength', 0)
        vol_regime = current_bar.get('vol_regime_numeric', 1)
        
        reward = 0.0

        # Strong trend with low volatility - expert trader's favorite setup
        if abs(trend_alignment) > 0.6 and abs(momentum_strength) > 0.4 and vol_regime <= 1:
            if (trend_alignment > 0 and action == 1) or (trend_alignment < 0 and action == 2):
                reward += 4.0  # High conviction setup
                
                # Extra reward for adding to winning positions in strong trends
                existing_positions = len(env.long_positions) if action == 1 else len(env.short_positions)
                if existing_positions > 0:
                    reward += 1.5  # Pyramiding in strong trends

        # Medium strength trends - moderate conviction
        elif abs(trend_alignment) > 0.4 and abs(momentum_strength) > 0.2:
            if (trend_alignment > 0 and action == 1) or (trend_alignment < 0 and action == 2):
                reward += 2.0

        # Reward patience during trend development
        elif action == 0 and abs(trend_alignment) > 0.3:
            reward += 0.5

        return reward

    def _calculate_enhanced_holding_reward(self, env, last_trade, current_step):
        """Enhanced reward for intelligent holding periods"""
        if 'entry_step' not in last_trade or last_trade.get('profit', 0) <= 0:
            return 0.0

        entry_step = last_trade['entry_step']
        if entry_step >= len(env.data):
            return 0.0

        holding_period = current_step - entry_step
        reward = 0.0

        # Base holding reward
        if holding_period >= 5:  # Minimum expert holding period
            reward += min(4.0, holding_period / 8.0)

        # Bonus for holding through adjustments and management
        if holding_period >= 10 and last_trade['type'] not in ['STOP-LOSS']:
            reward += 2.0  # Patience pays off

        # Extra bonus for very profitable long-term holds
        if holding_period >= 15 and last_trade.get('profit', 0) > 0:
            profit_ratio = last_trade['profit'] / (last_trade['entry_price'] * last_trade['shares'])
            if profit_ratio > 0.05:  # 5% profit
                reward += min(3.0, profit_ratio * 20)

        return reward

    def _calculate_position_sizing_reward(self, env, action):
        """Reward intelligent position sizing"""
        if action == 0:  # Hold
            return 0.0

        portfolio_value = env.get_portfolio_value()
        if portfolio_value <= 0:
            return 0.0

        current_price = env.data.iloc[env.current_step]['close']
        total_long_exposure = sum(pos['shares'] * current_price for pos in env.long_positions)
        total_short_exposure = sum(pos['shares'] * current_price for pos in env.short_positions)
        total_exposure = (total_long_exposure + total_short_exposure) / portfolio_value

        reward = 0.0

        # Reward optimal capital utilization (like expert traders)
        if 0.7 <= total_exposure <= 1.2:
            reward += 1.5
        elif 0.5 <= total_exposure <= 1.5:
            reward += 0.5
        elif total_exposure < 0.3:  # Under-utilized
            reward -= 0.5

        # Reward position diversification
        total_positions = len(env.long_positions) + len(env.short_positions)
        if 2 <= total_positions <= env.max_positions - 1:
            reward += 1.0

        return reward

    def _calculate_stop_adjustment_reward(self, env):
        """Reward intelligent stop loss adjustments"""
        reward = 0.0

        # Check recent stop adjustments
        for position in env.long_positions + env.short_positions:
            if position.get('stop_adjusted', False):
                # Reward if the adjustment helped avoid a premature stop
                entry_price = position['entry_price']
                current_price = env.data.iloc[env.current_step]['close']
                original_stop = position.get('original_stop', position['stop_loss_price'])
                
                if position['position_type'] == 'LONG':
                    # If current price is above original stop but we're still in
                    if current_price > original_stop and current_price > entry_price:
                        reward += 2.0  # Good adjustment saved the position
                elif position['position_type'] == 'SHORT':
                    # If current price is below original stop but we're still in
                    if current_price < original_stop and current_price < entry_price:
                        reward += 2.0  # Good adjustment saved the position

        return reward