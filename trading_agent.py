"""
Individual model trading agent - no ensemble
Each model trained independently with its assigned feature group
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from models import create_model

class IndividualTradingAgent:
    def __init__(self, state_size, action_size, config, model_name='dueling'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for model: {model_name}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.model_name = model_name
        
        # Initialize memory and exploration
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.priorities = deque(maxlen=config.MEMORY_SIZE)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        
        # Epsilon schedule
        self.epsilon_start = 1.0
        self.epsilon_end = 0.08
        self.epsilon_decay_steps = config.EPS_DECAY_STEPS
        self.steps_done = 0
        self.epsilon = self.epsilon_start
        self.current_phase = 0
        
        # Initialize single model
        self._initialize_model()
        self._initialize_optimizer()
        
        # Hidden state for LSTM and hybrid models
        self.hidden_state = None
    
    def _initialize_model(self):
        """Initialize single model and its target"""
        self.model = create_model(self.model_name, self.state_size, self.action_size).to(self.device)
        self.target_model = create_model(self.model_name, self.state_size, self.action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
    def _initialize_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.config.LR, weight_decay=2e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=self.config.LR_DECAY_STEP_SIZE, 
                                                   gamma=self.config.LR_GAMMA)
        self.criterion = nn.SmoothL1Loss()
        self.tau = self.config.TAU

    def act(self, state, training=True):
        """Action selection with epsilon-greedy"""
        # Update epsilon
        decay_progress = self.steps_done / self.epsilon_decay_steps
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-3 * decay_progress)
        self.steps_done += 1
        
        # Epsilon-greedy exploration
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Get Q-values from model
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            
            # Handle different model outputs
            if self.model_name in ['lstm', 'hybrid_cnn_lstm']:
                q_values, self.hidden_state = self.model(state_tensor, self.hidden_state)
            else:
                q_values = self.model(state_tensor)
            
            self.model.train()
        
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def replay(self):
        """Experience replay with prioritized sampling"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return 0
        
        self.model.train()
        self.target_model.eval()
        
        # Prioritized experience replay
        priorities = np.array(self.priorities)
        sampling_probs = priorities ** self.alpha
        sampling_probs /= sampling_probs.sum()
        
        indices = np.random.choice(len(self.memory), self.config.BATCH_SIZE, 
                                 p=sampling_probs, replace=False)
        batch = [self.memory[idx] for idx in indices]
        
        weights = (len(self.memory) * sampling_probs[indices]) ** (-self.beta)
        weights = torch.FloatTensor(weights / weights.max()).to(self.device)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Train the model
        loss, td_errors = self._train_step(states, actions, rewards, next_states, dones, weights)
        
        # Update priorities
        for i, idx in enumerate(indices):
            self.priorities[idx] = td_errors[i] + 1e-5
        
        return loss

    def _train_step(self, states, actions, rewards, next_states, dones, weights):
        """Single training step"""
        # Handle different model outputs
        if self.model_name in ['lstm', 'hybrid_cnn_lstm']:
            current_q, _ = self.model(states)
            current_q = current_q.gather(1, actions)
            
            with torch.no_grad():
                next_q, _ = self.target_model(next_states)
                next_actions = self.model(next_states)[0].max(1)[1].unsqueeze(1)
                next_q = self.target_model(next_states)[0].gather(1, next_actions).squeeze()
        else:
            current_q = self.model(states).gather(1, actions)
            
            with torch.no_grad():
                next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
                next_q = self.target_model(next_states).gather(1, next_actions).squeeze()
        
        target_q = rewards + (self.config.GAMMA * next_q * ~dones)
        
        # Compute loss with importance sampling weights
        loss = self.criterion(current_q.squeeze() * weights, target_q * weights)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()
        
        # Track TD errors
        td_errors = (current_q.squeeze() - target_q).abs().detach().cpu().numpy()
        
        return loss.item(), td_errors

    def soft_update_target_network(self):
        """Soft update target network"""
        for target_param, param in zip(self.target_model.parameters(), 
                                       self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1.0 - self.tau) * target_param.data)

    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'current_phase': self.current_phase,
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.current_phase = checkpoint.get('current_phase', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.model.train()

    def reset_epsilon_for_phase(self):
        """Reset epsilon for new curriculum phase"""
        phase_start_epsilons = [1.0, 0.7, 0.5]
        self.epsilon_start = phase_start_epsilons[min(self.current_phase, 2)]
        self.epsilon = self.epsilon_start
        self.steps_done = 0
    
    def reset_hidden_state(self):
        """Reset hidden state for LSTM/hybrid models"""
        self.hidden_state = None