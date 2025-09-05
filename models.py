"""
Actor-Critic models for PPO training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorNetwork, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.feature_layer, self.policy_head]:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, nn.Linear):
                        nn.init.orthogonal_(sub_layer.weight, gain=0.01)
                        nn.init.constant_(sub_layer.bias, 0)
    
    def forward(self, x):
        features = self.feature_layer(x)
        logits = self.policy_head(features)
        return F.softmax(logits, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super(CriticNetwork, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.feature_layer, self.value_head]:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, nn.Linear):
                        nn.init.orthogonal_(sub_layer.weight)
                        nn.init.constant_(sub_layer.bias, 0)
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_head(features)
        return value

class LSTMActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, num_layers=2):
        super(LSTMActorCritic, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM backbone
        self.lstm = nn.LSTM(state_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_size)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        
        for layer in [self.actor_head, self.critic_head]:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear):
                    nn.init.orthogonal_(sub_layer.weight)
    
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        
        # Actor output (action probabilities)
        logits = self.actor_head(lstm_out)
        action_probs = F.softmax(logits, dim=-1)
        
        # Critic output (state value)
        value = self.critic_head(lstm_out)
        
        return action_probs, value, hidden

class AttentionActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, num_heads=4):
        super(AttentionActorCritic, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3)
        )
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, 
                                             dropout=0.3, batch_first=True)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_size)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
        
        self.ln = nn.LayerNorm(hidden_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.feature_extractor, self.actor_head, self.critic_head]:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, nn.Linear):
                        nn.init.orthogonal_(sub_layer.weight)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        attn_output, _ = self.attention(features, features, features)
        attn_output = self.ln(features + attn_output)
        
        if attn_output.dim() == 3:
            attn_output = attn_output.squeeze(1)
        
        # Actor output
        logits = self.actor_head(attn_output)
        action_probs = F.softmax(logits, dim=-1)
        
        # Critic output
        value = self.critic_head(attn_output)
        
        return action_probs, value
