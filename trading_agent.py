"""
Pure PPO Trading Agent - No LLM components
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from models import ActorNetwork, CriticNetwork, LSTMActorCritic, AttentionActorCritic

class PPOTradingAgent:
    def __init__(self, state_size, action_size, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # PPO hyperparameters
        self.clip_epsilon = config.CLIP_EPSILON
        self.value_clip = config.VALUE_CLIP
        self.entropy_coef = config.ENTROPY_COEF
        self.value_coef = config.VALUE_COEF
        self.max_grad_norm = config.MAX_GRAD_NORM

        # Initialize models
        self._initialize_models()
        self._initialize_optimizers()

        # Experience storage
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
            'next_states': []
        }

        # Model performance tracking
        self.model_weights = {'standard': 0.4, 'lstm': 0.3, 'attention': 0.3}
        self.performance_history = {name: deque(maxlen=100) for name in self.model_weights.keys()}

    def _initialize_models(self):
        """Initialize ensemble of Actor-Critic models"""
        self.actors = {
            'standard': ActorNetwork(self.state_size, self.action_size).to(self.device),
            'lstm': LSTMActorCritic(self.state_size, self.action_size).to(self.device),
            'attention': AttentionActorCritic(self.state_size, self.action_size).to(self.device)
        }

        self.critics = {
            'standard': CriticNetwork(self.state_size).to(self.device)
        }

        # LSTM and Attention have integrated critics
        self.hidden_states = {'lstm': None}

    def _initialize_optimizers(self):
        """Initialize optimizers for all models"""
        self.actor_optimizers = {}
        self.critic_optimizers = {}

        # Standard Actor-Critic
        self.actor_optimizers['standard'] = optim.Adam(
            self.actors['standard'].parameters(), lr=self.config.LR)
        self.critic_optimizers['standard'] = optim.Adam(
            self.critics['standard'].parameters(), lr=self.config.LR)

        # LSTM Actor-Critic (single optimizer)
        self.actor_optimizers['lstm'] = optim.Adam(
            self.actors['lstm'].parameters(), lr=self.config.LR)

        # Attention Actor-Critic (single optimizer)
        self.actor_optimizers['attention'] = optim.Adam(
            self.actors['attention'].parameters(), lr=self.config.LR)

    def act(self, state, training=True):
        """Pure RL action selection - No LLM guidance"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get action probabilities from each model
            action_probs = {}
            values = {}

            # Standard Actor-Critic
            action_probs['standard'] = self.actors['standard'](state_tensor)
            values['standard'] = self.critics['standard'](state_tensor)

            # LSTM Actor-Critic
            lstm_probs, lstm_value, self.hidden_states['lstm'] = self.actors['lstm'](
                state_tensor, self.hidden_states['lstm'])
            action_probs['lstm'] = lstm_probs
            values['lstm'] = lstm_value

            # Attention Actor-Critic
            attn_probs, attn_value = self.actors['attention'](state_tensor)
            action_probs['attention'] = attn_probs
            values['attention'] = attn_value

            # Weighted ensemble (Pure RL)
            weights = torch.tensor([self.model_weights[name] for name in action_probs.keys()],
                                 device=self.device)
            ensemble_probs = sum(action_probs[name] * weights[i]
                               for i, name in enumerate(action_probs.keys()))
            ensemble_value = sum(values[name] * weights[i]
                               for i, name in enumerate(values.keys()))

            if training:
                # Sample action from probability distribution
                dist = torch.distributions.Categorical(ensemble_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item(), ensemble_value.item()
            else:
                # Take most probable action
                return torch.argmax(ensemble_probs).item(), 0.0, ensemble_value.item()

    def remember(self, state, action, log_prob, value, reward, next_state, done):
        """Store experience"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state)
        self.memory['dones'].append(done)

    def update(self):
        """Update all models using PPO"""
        if len(self.memory['states']) == 0:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.LongTensor(self.memory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        rewards = torch.FloatTensor(self.memory['rewards']).to(self.device)
        dones = torch.BoolTensor(self.memory['dones']).to(self.device)

        # Calculate advantages and returns
        advantages, returns = self._calculate_gae(rewards, old_values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        losses = {}

        # Update each model
        for name in self.actors.keys():
            model_losses = self._update_model(name, states, actions, old_log_probs,
                                            old_values, advantages, returns)
            losses[name] = model_losses

            # Update model weights based on performance
            avg_loss = sum(model_losses.values()) / len(model_losses)
            self.performance_history[name].append(1.0 / (1.0 + avg_loss))

            if len(self.performance_history[name]) >= 10:
                self.model_weights[name] = np.mean(list(self.performance_history[name]))

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}

        # Clear memory
        self._clear_memory()

        return losses

    def _update_model(self, model_name, states, actions, old_log_probs,
                     old_values, advantages, returns):
        """Update individual model"""
        losses = {}

        for epoch in range(4):  # PPO epochs
            if model_name == 'standard':
                # Standard Actor-Critic update
                action_probs = self.actors[model_name](states)
                values = self.critics[model_name](states).squeeze()

                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Actor loss (PPO clipped)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Critic loss
                if self.value_clip:
                    value_clipped = old_values + torch.clamp(values - old_values,
                                                           -self.clip_epsilon, self.clip_epsilon)
                    value_loss1 = F.mse_loss(values, returns)
                    value_loss2 = F.mse_loss(value_clipped, returns)
                    critic_loss = torch.max(value_loss1, value_loss2)
                else:
                    critic_loss = F.mse_loss(values, returns)

                # Update
                self.actor_optimizers[model_name].zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actors[model_name].parameters(), self.max_grad_norm)
                self.actor_optimizers[model_name].step()

                self.critic_optimizers[model_name].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[model_name].parameters(), self.max_grad_norm)
                self.critic_optimizers[model_name].step()

                losses[f'{model_name}_actor'] = actor_loss.item()
                losses[f'{model_name}_critic'] = critic_loss.item()

            else:
                # Integrated Actor-Critic models (LSTM, Attention)
                if model_name == 'lstm':
                    action_probs, values, _ = self.actors[model_name](states)
                else:  # attention
                    action_probs, values = self.actors[model_name](states)

                values = values.squeeze()

                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Combined loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                if self.value_clip:
                    value_clipped = old_values + torch.clamp(values - old_values,
                                                           -self.clip_epsilon, self.clip_epsilon)
                    value_loss1 = F.mse_loss(values, returns)
                    value_loss2 = F.mse_loss(value_clipped, returns)
                    critic_loss = torch.max(value_loss1, value_loss2)
                else:
                    critic_loss = F.mse_loss(values, returns)

                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.actor_optimizers[model_name].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[model_name].parameters(), self.max_grad_norm)
                self.actor_optimizers[model_name].step()

                losses[f'{model_name}_total'] = total_loss.item()

        return losses

    def _calculate_gae(self, rewards, values, dones, gamma=0.99, lambda_gae=0.95):
        """Calculate Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_advantage = 0
        last_return = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (~dones[t]).float() - values[t]
            advantages[t] = delta + gamma * lambda_gae * (~dones[t]).float() * last_advantage
            returns[t] = rewards[t] + gamma * (~dones[t]).float() * last_return

            last_advantage = advantages[t]
            last_return = returns[t]

        return advantages, returns

    def _clear_memory(self):
        """Clear experience memory"""
        for key in self.memory:
            self.memory[key] = []

    def save_models(self, path_prefix):
        """Save all models"""
        for name, actor in self.actors.items():
            torch.save(actor.state_dict(), f"{path_prefix}_{name}_actor.pth")

        for name, critic in self.critics.items():
            torch.save(critic.state_dict(), f"{path_prefix}_{name}_critic.pth")

        # Save weights and performance
        model_data = {
            'model_weights': self.model_weights,
            'performance_history': {k: list(v) for k, v in self.performance_history.items()}
        }

        torch.save(model_data, f"{path_prefix}_model_data.pth")

    def load_models(self, path_prefix):
        """Load all models"""
        for name, actor in self.actors.items():
            actor.load_state_dict(torch.load(f"{path_prefix}_{name}_actor.pth"))
            actor.eval()

        for name, critic in self.critics.items():
            critic.load_state_dict(torch.load(f"{path_prefix}_{name}_critic.pth"))
            critic.eval()

        try:
            model_data = torch.load(f"{path_prefix}_model_data.pth")
            self.model_weights = model_data.get('model_weights', self.model_weights)
            performance_data = model_data.get('performance_history', {})
            
            for name, history in performance_data.items():
                self.performance_history[name] = deque(history, maxlen=100)
                
        except FileNotFoundError:
            print("Model data not found, using defaults")
