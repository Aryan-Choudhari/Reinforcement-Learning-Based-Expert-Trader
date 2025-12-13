"""
Extended neural network models for individual model comparison
Contains 9 diverse architectures: 3 simple, 3 original, 3 complex
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F

# ============================================================================
# SIMPLE MODELS (3)
# ============================================================================

class SimpleDQN(nn.Module):
    """Simple 2-layer DQN baseline"""
    def __init__(self, state_size, action_size):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleDropoutDQN(nn.Module):
    """Simple DQN with dropout for regularization"""
    def __init__(self, state_size, action_size):
        super(SimpleDropoutDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, action_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class SimpleResidualDQN(nn.Module):
    """Simple DQN with residual connection"""
    def __init__(self, state_size, action_size):
        super(SimpleResidualDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.5)
    
    def forward(self, x):
        identity = self.fc1(x)
        out = F.relu(identity)
        out = self.fc2(out)
        out = F.relu(out + identity)  # Residual connection
        return self.fc3(out)


# ============================================================================
# ORIGINAL MODELS (3)
# ============================================================================

class DuelingDQN(nn.Module):
    """Original Dueling DQN architecture"""
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.feature_layer, self.value_stream, self.advantage_stream]:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, nn.Linear):
                        nn.init.xavier_uniform_(sub_layer.weight, gain=0.5)
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean()


class LSTMDQN(nn.Module):
    """Original LSTM-based DQN"""
    def __init__(self, state_size, action_size, hidden_size=128, num_layers=1):
        super(LSTMDQN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(state_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, action_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=0.5)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=0.5)
    
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]
        q_values = self.fc(lstm_out)
        
        return q_values, hidden


class AttentionDQN(nn.Module):
    """Original Attention-based DQN"""
    def __init__(self, state_size, action_size, hidden_size=128, num_heads=4):
        super(AttentionDQN, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3)
        )
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, 
                                             dropout=0.3, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.feature_extractor, self.fc1, self.fc2]:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, nn.Linear):
                        nn.init.xavier_uniform_(sub_layer.weight, gain=0.5)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        attn_output, _ = self.attention(features, features, features)
        attn_output = self.ln1(features + self.dropout(attn_output))
        
        if attn_output.dim() == 3:
            attn_output = attn_output.squeeze(1)
        
        fc_output = F.relu(self.fc1(attn_output))
        fc_output = self.dropout(fc_output)
        fc_output = self.fc2(fc_output)
        
        return fc_output


# ============================================================================
# COMPLEX MODELS (3)
# ============================================================================

class DeepDuelingDQN(nn.Module):
    """Deeper Dueling DQN with more layers"""
    def __init__(self, state_size, action_size):
        super(DeepDuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.feature_layer, self.value_stream, self.advantage_stream]:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, nn.Linear):
                        nn.init.xavier_uniform_(sub_layer.weight, gain=0.5)
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean()


class TransformerDQN(nn.Module):
    """Transformer-based DQN for sequential patterns"""
    def __init__(self, state_size, action_size, d_model=128, nhead=4, num_layers=2):
        super(TransformerDQN, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Linear(state_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.5)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)


class HybridCNNLSTMDQN(nn.Module):
    """Hybrid CNN-LSTM for feature extraction and temporal modeling"""
    def __init__(self, state_size, action_size):
        super(HybridCNNLSTMDQN, self).__init__()
        
        # CNN for feature extraction (treat state as 1D signal)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(16)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3)
        
        # Final decision layers
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.conv2.weight, gain=0.5)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
    
    def forward(self, x, hidden=None):
        # Reshape for CNN: (batch, 1, state_size)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM temporal modeling
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]
        
        # Final decision
        return self.fc(lstm_out), hidden


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_name, state_size, action_size):
    """Factory function to create models by name"""
    models = {
        # Simple models
        'simple_dqn': SimpleDQN,
        'simple_dropout': SimpleDropoutDQN,
        'simple_residual': SimpleResidualDQN,
        
        # Original models
        'dueling': DuelingDQN,
        'lstm': LSTMDQN,
        'attention': AttentionDQN,
        
        # Complex models
        'deep_dueling': DeepDuelingDQN,
        'transformer': TransformerDQN,
        'hybrid_cnn_lstm': HybridCNNLSTMDQN
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](state_size, action_size)