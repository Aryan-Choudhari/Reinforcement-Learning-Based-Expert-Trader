"""
Enhanced PPO training module with expert human trader behavior patterns
"""

import numpy as np
from trading_environment import AdvancedTradingEnvironment
from utils import PerformanceMetrics
import os
from datetime import datetime

def enhanced_walk_forward_training(agent, train_data, feature_columns, scaler, config, save_dir=None):
    """Enhanced walk-forward training with expert trader learning phases"""
    if save_dir is None:
        save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)

    MODEL_SAVE_PATH = os.path.join(save_dir, f'expert_trader_ppo_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    # Expert trader learning phases
    training_phases = [
        {
            'episodes': 180, 'entropy_coef': 0.05, 'lr_multiplier': 1.3, 
            'name': 'market_reading', 'focus': 'learn_market_patterns',
            'position_management_weight': 0.3
        },
        {
            'episodes': 220, 'entropy_coef': 0.035, 'lr_multiplier': 1.1, 
            'name': 'trend_mastery', 'focus': 'master_trend_following',
            'position_management_weight': 0.6
        },
        {
            'episodes': 200, 'entropy_coef': 0.025, 'lr_multiplier': 0.9, 
            'name': 'risk_management', 'focus': 'expert_risk_control',
            'position_management_weight': 0.8
        },
        {
            'episodes': 120, 'entropy_coef': 0.015, 'lr_multiplier': 0.7, 
            'name': 'position_mastery', 'focus': 'advanced_position_management',
            'position_management_weight': 1.0
        }
    ]

    initial_train_size = int(len(train_data) * 0.50)
    validation_size = int(len(train_data) * 0.15)
    step_size = int(len(train_data) * 0.10)

    best_overall_stability = -np.inf
    episode_returns = []

    walk_forward_steps = min(config.WALK_FORWARD_STEPS,
                             (len(train_data) - validation_size - initial_train_size) // step_size)

    for step in range(walk_forward_steps):
        print(f"\n🚀 Walk-Forward Step {step + 1}/{walk_forward_steps}")

        # Define expanding window
        train_start = 0
        train_end = initial_train_size + (step * step_size)
        val_start = train_end
        val_end = min(val_start + validation_size, len(train_data))

        current_train_data = train_data.iloc[train_start:train_end].reset_index(drop=True)
        current_val_data = train_data.iloc[val_start:val_end].reset_index(drop=True)

        print(f"📊 Data sizes - Train: {len(current_train_data)}, Val: {len(current_val_data)}")

        # Train with expert trader phases
        for phase_config in training_phases:
            phase_returns = train_expert_trader_phase(
                agent, current_train_data, current_val_data,
                feature_columns, scaler, config, phase_config)
            episode_returns.extend(phase_returns)

        # Evaluate on holdout with expert trader metrics
        test_start = val_end
        test_end = min(test_start + 50, len(train_data))
        if test_end > test_start:
            test_data_slice = train_data.iloc[test_start:test_end].reset_index(drop=True)
            test_env = evaluate_expert_trader(agent, test_data_slice, feature_columns, scaler, config)
            test_portfolio = np.array(test_env.portfolio_value_history)
            test_benchmark = test_data_slice['close'].values / test_data_slice['close'].iloc[0] * config.INITIAL_CASH
            test_metrics = PerformanceMetrics.calculate_metrics(
                test_portfolio, test_benchmark, config.INITIAL_CASH)

            stability_score = calculate_expert_stability_score(test_metrics, test_env)
            
            print(f"📊 STEP {step + 1} EXPERT EVALUATION:")
            print(f" Expert Score: {stability_score:7.2f}")
            print(f" Total Return: {test_metrics['total_return']*100:+6.2f}%")
            print(f" Excess Return: {test_metrics['excess_return']*100:+6.2f}%")
            print(f" Position Adjustments: {count_position_adjustments(test_env)}")

            if stability_score > best_overall_stability:
                best_overall_stability = stability_score
                agent.save_models(MODEL_SAVE_PATH)
                print(f" ⭐ NEW BEST EXPERT TRADER MODEL!")

    return episode_returns, MODEL_SAVE_PATH

def train_expert_trader_phase(agent, train_data, val_data, feature_columns, scaler, config, phase_config):
    """Train agent to behave like an expert human trader"""
    episode_returns = []
    best_val_return = -np.inf
    patience_count = 0
    update_frequency = config.UPDATE_FREQUENCY

    # Apply phase-specific parameters
    original_entropy = agent.entropy_coef
    original_lrs = {}
    
    agent.entropy_coef = phase_config['entropy_coef']
    
    # Adjust learning rates
    for name, optimizer in agent.actor_optimizers.items():
        original_lrs[name] = []
        for param_group in optimizer.param_groups:
            original_lrs[name].append(param_group['lr'])
            param_group['lr'] = config.LR * phase_config['lr_multiplier']

    print(f"🎯 Expert Training Phase: {phase_config['name']} ({phase_config['episodes']} episodes)")
    print(f"   Focus: {phase_config['focus']}")
    print(f"   Position Management Weight: {phase_config['position_management_weight']}")
    print(f"   Entropy: {phase_config['entropy_coef']}, LR: {phase_config['lr_multiplier']}x")

    for episode in range(phase_config['episodes']):
        # Training episode with expert trader environment
        train_env = AdvancedTradingEnvironment(train_data, feature_columns, scaler, config)
        state = train_env.reset()
        done = False

        # Reset LSTM hidden states
        if hasattr(agent, 'hidden_states'):
            agent.hidden_states = {'lstm': None}

        step_count = 0
        position_adjustments = 0
        expert_actions = 0

        while not done:
            action, log_prob, value = agent.act(state, training=True)
            next_state, reward, done, info = train_env.step(action)
            
            # Track expert trader behaviors
            if info.get('market_bias') in ['correction_buy', 'correction_sell', 'ranging']:
                expert_actions += 1
            
            # Enhanced reward based on phase focus
            enhanced_reward = enhance_reward_for_phase(reward, phase_config, train_env, info)
            
            agent.remember(state, action, log_prob, value, enhanced_reward, next_state, done)
            state = next_state
            step_count += 1

            # Count position management actions
            if train_env.trades and train_env.trades[-1]['type'] in ['PARTIAL_SELL', 'PARTIAL_COVER']:
                position_adjustments += 1

            # Update after collecting enough experiences
            if len(agent.memory['states']) >= update_frequency:
                losses = agent.update()
                if episode % 25 == 0:
                    print(f"Episode {episode+1}: Updated models, Expert Actions: {expert_actions}")

        train_return = (train_env.get_portfolio_value() - config.INITIAL_CASH) / config.INITIAL_CASH * 100
        episode_returns.append(train_return)

        # Expert trader validation
        validation_frequency = 12 if phase_config['name'] == 'market_reading' else 10
        
        if (episode + 1) % validation_frequency == 0:
            val_env = evaluate_expert_trader(agent, val_data, feature_columns, scaler, config)
            val_return = (val_env.get_portfolio_value() - config.INITIAL_CASH) / config.INITIAL_CASH * 100
            val_adjustments = count_position_adjustments(val_env)
            
            print(f"Episode {episode+1}: Train: {train_return:+6.2f}%, Val: {val_return:+6.2f}%, Adjustments: {val_adjustments}")
            
            # Expert trader evaluation criteria
            expert_score = calculate_expert_episode_score(val_return, val_adjustments, val_env)
            
            if expert_score > best_val_return:
                best_val_return = expert_score
                patience_count = 0
            else:
                patience_count += 1

            phase_patience = config.PATIENCE + 2 if phase_config['name'] == 'market_reading' else config.PATIENCE
            
            if patience_count >= phase_patience:
                print(f"Early stopping at episode {episode + 1} in {phase_config['name']} phase")
                break

    # Restore original parameters
    agent.entropy_coef = original_entropy
    
    for name, optimizer in agent.actor_optimizers.items():
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = original_lrs[name][i]

    print(f"✅ Completed expert phase: {phase_config['name']}")
    return episode_returns

def enhance_reward_for_phase(base_reward, phase_config, env, info):
    """Enhance reward based on training phase focus"""
    enhanced_reward = base_reward
    phase_weight = phase_config['position_management_weight']
    
    # Market reading phase - reward understanding market conditions
    if phase_config['name'] == 'market_reading':
        market_bias = info.get('market_bias', 'neutral')
        if market_bias in ['correction_buy', 'correction_sell', 'ranging']:
            enhanced_reward += 1.0  # Reward recognizing special conditions
            
    # Trend mastery phase - reward trend following
    elif phase_config['name'] == 'trend_mastery':
        if hasattr(env, 'long_positions') and hasattr(env, 'short_positions'):
            # Reward holding positions in trending markets
            current_step = env.current_step
            if current_step < len(env.data):
                trend_alignment = env.data.iloc[current_step].get('trend_alignment', 0)
                if abs(trend_alignment) > 0.5:
                    total_positions = len(env.long_positions) + len(env.short_positions)
                    if total_positions > 0:
                        enhanced_reward += 1.5 * phase_weight
                        
    # Risk management phase - reward defensive actions
    elif phase_config['name'] == 'risk_management':
        if env.trades and env.trades[-1]['type'] in ['PARTIAL_SELL', 'PARTIAL_COVER']:
            enhanced_reward += 3.0 * phase_weight
            
    # Position mastery phase - reward advanced position management
    elif phase_config['name'] == 'position_mastery':
        # Reward any position adjustments or management
        for position in getattr(env, 'long_positions', []) + getattr(env, 'short_positions', []):
            if position.get('stop_adjusted', False) or position.get('moved_to_breakeven', False):
                enhanced_reward += 2.5 * phase_weight
                break
    
    return enhanced_reward

def evaluate_expert_trader(agent, data, feature_columns, scaler, config):
    """Enhanced evaluation focusing on expert trader behaviors"""
    env = AdvancedTradingEnvironment(data, feature_columns, scaler, config)
    env.training = False
    state = env.reset()
    done = False

    # Reset LSTM hidden states
    if hasattr(agent, 'hidden_states'):
        agent.hidden_states = {'lstm': None}

    while not done:
        action, _, _ = agent.act(state, training=False)
        state, _, done, _ = env.step(action)

    # Enhanced liquidation with final position management
    if env.positions:
        last_price = env.data.iloc[-1]['close']
        total_liquidated_value = 0
        
        for position in env.positions:
            proceeds = position['shares'] * last_price * (1 - env.transaction_cost)
            profit = (last_price - position['entry_price']) * position['shares'] - \
                    (position['shares'] * last_price * env.transaction_cost)
            
            env.cash += proceeds
            total_liquidated_value += proceeds
            
            env.trades.append({
                'step': env.current_step,
                'type': 'LIQUIDATE',
                'shares': position['shares'],
                'price': last_price,
                'profit': profit,
                'entry_price': position['entry_price'],
                'entry_step': position.get('entry_step', env.current_step)
            })
        
        env.positions = []

    return env

def calculate_expert_stability_score(metrics, env):
    """Calculate stability score with expert trader behavior weighting"""
    def normalize_metric(value, is_positive=True):
        if is_positive:
            return max(0, min(1, (value + 1) / 2))
        else:
            return max(0, min(1, 1 - value))

    # Base performance score
    base_score = (
        normalize_metric(metrics['excess_return']) * 0.40 +
        normalize_metric(metrics['sharpe_ratio']) * 0.25 +
        normalize_metric(metrics['sortino_ratio']) * 0.20 +
        normalize_metric(metrics['max_drawdown'], is_positive=False) * 0.15
    )

    # Expert trader behavior bonuses
    position_adjustments = count_position_adjustments(env)
    adjustment_bonus = min(0.1, position_adjustments / 50.0)  # Up to 10% bonus
    
    # Risk management bonus
    risk_management_trades = count_risk_management_trades(env)
    risk_bonus = min(0.05, risk_management_trades / 20.0)  # Up to 5% bonus
    
    # Long-term holding bonus
    avg_holding_period = calculate_average_holding_period(env)
    holding_bonus = min(0.05, avg_holding_period / 15.0)  # Up to 5% bonus for 15+ day holds

    expert_score = base_score + adjustment_bonus + risk_bonus + holding_bonus
    return expert_score

def calculate_expert_episode_score(return_pct, adjustments, env):
    """Calculate episode score emphasizing expert behaviors"""
    base_score = return_pct
    
    # Bonus for position management
    management_bonus = adjustments * 2.0
    
    # Bonus for risk management trades
    risk_trades = count_risk_management_trades(env)
    risk_bonus = risk_trades * 3.0
    
    return base_score + management_bonus + risk_bonus

def count_position_adjustments(env):
    """Count position management actions"""
    if not hasattr(env, 'trades') or not env.trades:
        return 0
    
    adjustments = 0
    for trade in env.trades:
        if trade['type'] in ['PARTIAL_SELL', 'PARTIAL_COVER']:
            adjustments += 1
            
    # Also count positions with adjusted stops
    for position in getattr(env, 'long_positions', []) + getattr(env, 'short_positions', []):
        if position.get('stop_adjusted', False):
            adjustments += 1
        if position.get('moved_to_breakeven', False):
            adjustments += 1
            
    return adjustments

def count_risk_management_trades(env):
    """Count risk management focused trades"""
    if not hasattr(env, 'trades') or not env.trades:
        return 0
    
    risk_trades = 0
    for trade in env.trades:
        if trade['type'] in ['PARTIAL_SELL', 'PARTIAL_COVER', 'STOP-LOSS']:
            risk_trades += 1
    
    return risk_trades

def calculate_average_holding_period(env):
    """Calculate average holding period for completed trades"""
    if not hasattr(env, 'trades') or not env.trades:
        return 0
    
    holding_periods = []
    exit_trades = [t for t in env.trades if t['type'] in ['SELL', 'COVER', 'LIQUIDATE']]
    
    for trade in exit_trades:
        if 'entry_step' in trade:
            holding_period = trade['step'] - trade['entry_step']
            holding_periods.append(holding_period)
    
    return np.mean(holding_periods) if holding_periods else 0

def adaptive_position_sizing_schedule(episode, total_episodes, env):
    """Adaptive position sizing based on training progress"""
    progress = episode / total_episodes
    
    # Early training: smaller positions for exploration
    if progress < 0.3:
        return 0.7
    # Mid training: normal positions
    elif progress < 0.7:
        return 1.0
    # Late training: larger positions with more confidence
    else:
        return 1.2

def calculate_trend_consistency_bonus(env, recent_trades, current_trend):
    """Calculate bonus for consistent expert trend-following behavior"""
    if len(recent_trades) < 2:
        return 0
    
    trend_consistent_trades = 0
    expert_entries = 0
    
    for trade in recent_trades[-10:]:  # Look at last 10 trades
        # Reward buying in strong uptrends
        if current_trend > 0.5 and trade['type'] == 'BUY':
            trend_consistent_trades += 1
            expert_entries += 1
        # Reward shorting in strong downtrends
        elif current_trend < -0.5 and trade['type'] == 'SELL_SHORT':
            trend_consistent_trades += 1
            expert_entries += 1
        # Reward partial exits for risk management
        elif trade['type'] in ['PARTIAL_SELL', 'PARTIAL_COVER']:
            expert_entries += 1
    
    consistency_ratio = trend_consistent_trades / max(len(recent_trades[-10:]), 1)
    expert_ratio = expert_entries / max(len(recent_trades[-10:]), 1)
    
    return (consistency_ratio * 0.5) + (expert_ratio * 0.3)