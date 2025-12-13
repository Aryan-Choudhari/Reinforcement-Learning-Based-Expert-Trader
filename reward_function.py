"""
Enhanced reward function with regime awareness
Replace your existing reward_function.py with this complete file
"""
import numpy as np


class ImprovedRewardFunction:
    """Enhanced reward function with regime awareness and better incentive structure"""
    
    def __init__(self, lookback_window=20, risk_free_rate=0.02):
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        
        # Track performance across regimes
        self.regime_performance = {
            'uptrend': [],
            'downtrend': [],
            'sideways': [],
            'high_vol': [],
            'low_vol': []
        }
    
    def calculate_enhanced_reward(self, env, action, trade_occurred, current_step):
        """
        Enhanced reward with regime adaptation
        Rewards models for performing well in their current market regime
        """
        reward = 0.0
        current_price = env.data.iloc[current_step]['close']
        portfolio_value = env.get_portfolio_value()
        
        if portfolio_value <= 0:
            return -100.0
        
        # Get regime information
        regime_info = self._get_regime_info(env, current_step)
        
        # 1. BENCHMARK COMPARISON (60% weight) - WITH REGIME ADJUSTMENT
        if len(env.portfolio_value_history) > 1:
            agent_return = (portfolio_value - env.portfolio_value_history[-2]) / env.portfolio_value_history[-2]
            
            # Catastrophic loss check
            if agent_return <= -0.5:
                return -50.0
            
            if current_step > 0:
                bnh_return = (current_price - env.data.iloc[current_step-1]['close']) / env.data.iloc[current_step-1]['close']
                excess_return = agent_return - bnh_return
                
                # Adjust reward based on regime difficulty
                regime_multiplier = self._get_regime_difficulty_multiplier(regime_info)
                reward += excess_return * 150.0 * regime_multiplier
        
        # 2. REGIME-APPROPRIATE POSITION REWARD
        reward += self._calculate_regime_position_reward(env, regime_info, action)
        
        # 3. REWARD FOR TAKING POSITIONS (encourage action when appropriate)
        position_count = len(env.long_positions) + len(env.short_positions)
        if position_count > 0:
            reward += 0.5
        elif portfolio_value > env.portfolio_peak * 0.95:
            # Penalize staying in cash when portfolio is healthy
            reward -= 1.0
        
        # 4. REALIZED PROFIT REWARD (heavily weighted) - ENHANCED WITH REGIME CONTEXT
        if trade_occurred and env.trades:
            last_trade = env.trades[-1]
            reward += self._calculate_trade_reward(last_trade, regime_info, current_step)
        
        # 5. DRAWDOWN PENALTY (from portfolio peak) - STRICTER IN HIGH VOL
        drawdown_penalty = self._calculate_drawdown_penalty(env, regime_info)
        reward += drawdown_penalty
        
        # 6. SHARPE RATIO COMPONENT (encourage consistent performance)
        if len(env.portfolio_value_history) >= self.lookback_window:
            sharpe_reward = self._calculate_sharpe_component(env)
            # Bonus for good Sharpe in difficult regimes
            sharpe_reward *= self._get_regime_difficulty_multiplier(regime_info)
            reward += sharpe_reward * 0.3
        
        # 7. MOMENTUM ALIGNMENT BONUS (small)
        if current_step >= 5:
            momentum = (current_price - env.data.iloc[current_step-5]['close']) / env.data.iloc[current_step-5]['close']
            
            # Reward being long in uptrend or short in downtrend
            if len(env.long_positions) > 0 and momentum > 0.02:
                reward += 0.3
            elif len(env.short_positions) > 0 and momentum < -0.02:
                reward += 0.3
        
        # 8. REGIME TRANSITION PENALTY
        # Penalize being heavily invested during regime transitions (high uncertainty)
        if regime_info.get('regime_uncertainty', 0) > 0.5:
            position_ratio = len(env.positions) / env.max_positions
            if position_ratio > 0.7:
                reward -= 1.0  # Discourage high exposure during uncertainty
        
        return np.clip(reward, -30.0, 30.0)
    
    def _get_regime_info(self, env, current_step):
        """Extract regime information from current state"""
        if current_step >= len(env.data):
            return {}
        
        row = env.data.iloc[current_step]
        return {
            'in_uptrend': row.get('in_uptrend', 0),
            'in_downtrend': row.get('in_downtrend', 0),
            'in_sideways': row.get('in_sideways', 0),
            'vol_regime_high': row.get('vol_regime_high', 0),
            'vol_regime_low': row.get('vol_regime_low', 0),
            'favorable_long_regime': row.get('favorable_long_regime', 0),
            'favorable_short_regime': row.get('favorable_short_regime', 0),
            'regime_uncertainty': row.get('regime_uncertainty', 0)
        }
    
    def _get_regime_difficulty_multiplier(self, regime_info):
        """
        Reward more for performing well in difficult regimes
        1.0 = normal, >1.0 = harder regime (more reward), <1.0 = easier (less reward)
        """
        multiplier = 1.0
        
        # High volatility is harder
        if regime_info.get('vol_regime_high', 0) > 0:
            multiplier *= 1.3
        
        # Sideways markets are harder to profit from
        if regime_info.get('in_sideways', 0) > 0:
            multiplier *= 1.2
        
        # Regime uncertainty is harder
        if regime_info.get('regime_uncertainty', 0) > 0.5:
            multiplier *= 1.25
        
        return multiplier
    
    def _calculate_regime_position_reward(self, env, regime_info, action):
        """Reward taking positions appropriate for current regime"""
        reward = 0.0
        has_long = len(env.long_positions) > 0
        has_short = len(env.short_positions) > 0
        
        # Reward being long in favorable long regime
        if has_long and regime_info.get('favorable_long_regime', 0) > 0.5:
            reward += 0.5
        
        # Reward being short in favorable short regime
        if has_short and regime_info.get('favorable_short_regime', 0) > 0.5:
            reward += 0.5
        
        # Penalize wrong direction
        if has_long and regime_info.get('favorable_short_regime', 0) > 0.7:
            reward -= 0.3
        
        if has_short and regime_info.get('favorable_long_regime', 0) > 0.7:
            reward -= 0.3
        
        # Reward staying out during high uncertainty
        no_positions = len(env.positions) == 0
        if no_positions and regime_info.get('regime_uncertainty', 0) > 0.7:
            reward += 0.3
        
        return reward
    
    def _calculate_trade_reward(self, trade, regime_info, current_step):
        """Calculate reward for completed trade, adjusted for regime"""
        reward = 0.0
        
        exit_types = ['SELL_LONG_PROFIT_TARGET', 'SELL_LONG_PEAK_DECLINE', 
                     'SELL_LONG_WEAK_PROFIT', 'COVER_SHORT_PROFIT_TARGET',
                     'COVER_SHORT_TROUGH_RISE', 'COVER_SHORT_WEAK_PROFIT',
                     'SELL_LONG', 'COVER_SHORT', 'SELL_LONG_LOSS_LIMIT',
                     'COVER_SHORT_LOSS_LIMIT', 'SELL_LONG_STOP_LOSS',
                     'COVER_SHORT_STOP_LOSS', 'SELL_LONG_TIME_EXIT',
                     'COVER_SHORT_TIME_EXIT', 'EMERGENCY_SELL_LONG',
                     'EMERGENCY_COVER_SHORT']
        
        if trade['type'] not in exit_types:
            return 0.0
        
        if trade['shares'] == 0 or trade['entry_price'] == 0:
            return 0.0
        
        profit_ratio = trade['profit'] / (trade['entry_price'] * trade['shares'])
        holding_period = current_step - trade.get('entry_step', current_step)
        
        # Base profit reward
        if profit_ratio > 0:
            # Reward quick profits
            time_factor = max(0.8, 2.5 - holding_period / 15.0)
            reward += profit_ratio * 100.0 * time_factor
            
            # Bonus for profit in difficult regime
            regime_multiplier = self._get_regime_difficulty_multiplier(regime_info)
            reward *= regime_multiplier
        else:
            # Penalize losses, but less if cut quickly
            time_penalty = min(2.0, 1.0 + holding_period / 20.0)
            reward += profit_ratio * 80.0 * time_penalty
        
        # Specific exit type rewards
        if 'PROFIT_TARGET' in trade['type']:
            reward += 2.0
        elif 'PEAK_DECLINE' in trade['type'] or 'TROUGH_RISE' in trade['type']:
            reward += 1.5
        elif 'WEAK_PROFIT' in trade['type']:
            reward += 1.0
        elif 'LOSS_LIMIT' in trade['type'] and holding_period < 15:
            reward += 0.5  # Good to cut losses quickly
        
        return reward
    
    def _calculate_drawdown_penalty(self, env, regime_info):
        """Calculate drawdown penalty, stricter in high vol regimes"""
        if not hasattr(env, 'portfolio_peak'):
            return 0.0
        
        portfolio_value = env.get_portfolio_value()
        drawdown_from_peak = (env.portfolio_peak - portfolio_value) / env.portfolio_peak
        
        if drawdown_from_peak <= 0.03:
            return 0.0
        
        # Base penalty
        penalty = drawdown_from_peak * 35.0
        
        # Increase penalty in high volatility regimes (should be more careful)
        if regime_info.get('vol_regime_high', 0) > 0:
            penalty *= 1.5
        
        return -penalty
    
    def _calculate_sharpe_component(self, env):
        """Calculate Sharpe-based reward component"""
        history = env.portfolio_value_history
        if len(history) < self.lookback_window:
            return 0.0
        
        recent_values = np.array(history[-self.lookback_window:])
        returns = (recent_values[1:] - recent_values[:-1]) / recent_values[:-1]
        
        if returns.std() == 0:
            return 0.0
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        if sharpe > 1.5:
            return sharpe * 12.0
        elif sharpe > 0.5:
            return sharpe * 8.0
        else:
            return (sharpe - 0.5) * 15.0