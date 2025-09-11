"""
Fixed Specialized PPO Trading Agents for Long and Short Strategies
"""
import torch
import numpy as np
from trading_agent import PPOTradingAgent

class LongOnlyPPOAgent(PPOTradingAgent):
    """Specialized agent for long-only trading strategies"""
    
    def __init__(self, state_size, action_size, config):
        super().__init__(state_size, action_size, config)
        # Enhanced entropy for long agent exploration
        self.entropy_coef = config.ENTROPY_COEF * 1.2

    def _get_ensemble_probs(self, state_tensor):
        """Get ensemble probabilities from all models"""
        with torch.no_grad():
            # Get action probabilities from each model
            action_probs = {}
            
            # Standard Actor-Critic
            action_probs['standard'] = self.actors['standard'](state_tensor)
            
            # LSTM Actor-Critic
            lstm_probs, _, self.hidden_states['lstm'] = self.actors['lstm'](
                state_tensor, self.hidden_states['lstm'])
            action_probs['lstm'] = lstm_probs
            
            # Attention Actor-Critic
            attn_probs, _ = self.actors['attention'](state_tensor)
            action_probs['attention'] = attn_probs
            
            # Weighted ensemble
            weights = torch.tensor([self.model_weights[name] for name in action_probs.keys()],
                                 device=self.device)
            ensemble_probs = sum(action_probs[name] * weights[i] 
                               for i, name in enumerate(action_probs.keys()))
            
            return ensemble_probs

    def act(self, state, training=True):
        """Override to only allow long or hold actions"""
        action, log_prob, value = super().act(state, training)
        
        # Mask short action (2) => convert to hold (0) and adjust probabilities properly
        if action == 2:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs = self._get_ensemble_probs(state_tensor)
                
                # Create masked distribution: redistribute short probability to hold
                masked_probs = action_probs.clone()
                masked_probs[0, 0] += masked_probs[0, 2]  # Add short prob to hold
                masked_probs[0, 2] = 1e-8  # Keep small value for numerical stability
                
                # Renormalize
                masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)
                
                # Calculate corrected log probability
                dist = torch.distributions.Categorical(masked_probs)
                action = 0  # Force to hold
                log_prob = dist.log_prob(torch.tensor([0]).to(self.device)).item()
        
        return action, log_prob, value

class ShortOnlyPPOAgent(PPOTradingAgent):
    """Specialized agent for short-only trading strategies"""
    
    def __init__(self, state_size, action_size, config):
        super().__init__(state_size, action_size, config)
        # Enhanced entropy for short agent exploration
        self.entropy_coef = config.ENTROPY_COEF * 1.2

    def _get_ensemble_probs(self, state_tensor):
        """Get ensemble probabilities from all models"""
        with torch.no_grad():
            # Get action probabilities from each model
            action_probs = {}
            
            # Standard Actor-Critic
            action_probs['standard'] = self.actors['standard'](state_tensor)
            
            # LSTM Actor-Critic
            lstm_probs, _, self.hidden_states['lstm'] = self.actors['lstm'](
                state_tensor, self.hidden_states['lstm'])
            action_probs['lstm'] = lstm_probs
            
            # Attention Actor-Critic
            attn_probs, _ = self.actors['attention'](state_tensor)
            action_probs['attention'] = attn_probs
            
            # Weighted ensemble
            weights = torch.tensor([self.model_weights[name] for name in action_probs.keys()],
                                 device=self.device)
            ensemble_probs = sum(action_probs[name] * weights[i] 
                               for i, name in enumerate(action_probs.keys()))
            
            return ensemble_probs

    def act(self, state, training=True):
        """Override to only allow short or hold actions"""
        action, log_prob, value = super().act(state, training)
        
        # Mask long action (1) => convert to hold (0) and adjust probabilities properly
        if action == 1:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs = self._get_ensemble_probs(state_tensor)
                
                # Create masked distribution: redistribute long probability to hold
                masked_probs = action_probs.clone()
                masked_probs[0, 0] += masked_probs[0, 1]  # Add long prob to hold
                masked_probs[0, 1] = 1e-8  # Keep small value for numerical stability
                
                # Renormalize
                masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)
                
                # Calculate corrected log probability
                dist = torch.distributions.Categorical(masked_probs)
                action = 0  # Force to hold
                log_prob = dist.log_prob(torch.tensor([0]).to(self.device)).item()
        
        return action, log_prob, value

class CombinedSpecializedAgent:
    """Master agent combining long and short specialized agents"""
    
    def __init__(self, state_size, action_size, config):
        self.long_agent = LongOnlyPPOAgent(state_size, action_size, config)
        self.short_agent = ShortOnlyPPOAgent(state_size, action_size, config)
        self.config = config
        
        # Enhanced weights for better signal combination
        self.long_weight = getattr(config, 'LONG_AGENT_WEIGHT', 0.5)
        self.short_weight = getattr(config, 'SHORT_AGENT_WEIGHT', 0.5)

    def act(self, state, training=True):
        """Enhanced action selection with improved signal combination"""
        # Get actions from both specialized agents
        long_action, long_log_prob, long_value = self.long_agent.act(state, training)
        short_action, short_log_prob, short_value = self.short_agent.act(state, training)
        
        # Extract market regime information from state
        market_signals = self._extract_market_signals(state)
        
        # Improved signal combination logic
        final_action = self._enhanced_combine_signals(
            long_action, short_action, market_signals)
        
        # Weight combination of log probs and values
        combined_log_prob = (long_log_prob * self.long_weight + 
                           short_log_prob * self.short_weight)
        combined_value = (long_value * self.long_weight + 
                        short_value * self.short_weight)
        
        return final_action, combined_log_prob, combined_value

    def _extract_market_signals(self, state):
        """Extract market regime signals from state vector"""
        # Assuming state structure from your environment
        state_array = np.array(state)
        
        # Extract key market features (adjust indices based on your actual state structure)
        try:
            # These indices should match your actual state construction in trading_environment.py
            trend_alignment = state_array[-8] if len(state_array) > 8 else 0  # trend_alignment position
            momentum_strength = state_array[-7] if len(state_array) > 7 else 0  # momentum_strength position
            vol_regime = state_array[-9] if len(state_array) > 9 else 1  # vol_regime position
            rsi_norm = state_array[-6] if len(state_array) > 6 else 0  # rsi_normalized position
            
            return {
                'trend_alignment': trend_alignment,
                'momentum_strength': momentum_strength,
                'vol_regime': vol_regime,
                'rsi_norm': rsi_norm
            }
        except:
            # Fallback to neutral signals if state extraction fails
            return {
                'trend_alignment': 0,
                'momentum_strength': 0,
                'vol_regime': 1,
                'rsi_norm': 0
            }

    def _enhanced_combine_signals(self, long_action, short_action, market_signals):
        """Enhanced signal combination with less conservative logic"""
        trend = market_signals['trend_alignment']
        momentum = market_signals['momentum_strength']
        vol_regime = market_signals['vol_regime']
        rsi = market_signals['rsi_norm']
        
        # Strong bullish conditions - favor long signals
        if trend > 0.4 and momentum > 0.2 and vol_regime <= 1:
            if long_action == 1:  # Long agent wants to buy
                return 1
            elif short_action == 0:  # Short agent is neutral
                return 0
            else:
                return 1 if trend > 0.6 else 0  # Be more aggressive in strong trends
        
        # Strong bearish conditions - favor short signals
        elif trend < -0.4 and momentum < -0.2 and vol_regime <= 1:
            if short_action == 2:  # Short agent wants to sell short
                return 2
            elif long_action == 0:  # Long agent is neutral
                return 0
            else:
                return 2 if trend < -0.6 else 0  # Be more aggressive in strong downtrends
        
        # Oversold bounce opportunity (bullish correction)
        elif trend > 0.2 and rsi < -0.3:  # Strong trend but oversold
            return 1 if long_action == 1 else 0
        
        # Overbought short opportunity (bearish correction)  
        elif trend < -0.2 and rsi > 0.3:  # Strong downtrend but overbought
            return 2 if short_action == 2 else 0
        
        # Neutral/mixed conditions - be less conservative
        else:
            # Give preference to any clear signal from either agent
            if long_action == 1 and short_action == 0:
                return 1
            elif short_action == 2 and long_action == 0:
                return 2
            elif long_action == 1 and short_action == 2:
                # Conflicting signals - use trend as tiebreaker
                return 1 if trend > 0 else (2 if trend < -0.2 else 0)
            else:
                return 0  # Both agents want to hold

    def remember(self, state, action, log_prob, value, reward, next_state, done):
        """Store experience in both agents"""
        self.long_agent.remember(state, action, log_prob, value, reward, next_state, done)
        self.short_agent.remember(state, action, log_prob, value, reward, next_state, done)

    def update(self):
        """Update both specialized agents separately"""
        losses_long = self.long_agent.update()
        losses_short = self.short_agent.update()
        
        return {
            'long_agent': losses_long, 
            'short_agent': losses_short
        }

    def save_models(self, path_prefix):
        """Save both specialized agents"""
        self.long_agent.save_models(f'{path_prefix}_long')
        self.short_agent.save_models(f'{path_prefix}_short')

    def load_models(self, path_prefix):
        """Load both specialized agents"""
        self.long_agent.load_models(f'{path_prefix}_long')
        self.short_agent.load_models(f'{path_prefix}_short')
