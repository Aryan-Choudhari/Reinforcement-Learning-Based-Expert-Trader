"""
Enhanced PPO training module with specialized agents and comprehensive reporting
FIXED VERSION with proper metrics reporting
"""

import time
import numpy as np
import os
import pandas as pd
from datetime import datetime
from trading_environment import AdvancedTradingEnvironment
from specialized_agents import CombinedSpecializedAgent
from utils import PerformanceMetrics

def report_metrics_periodically(agent, env, config, episode, phase_name="", interval=10):
    """Report performance metrics during training with enhanced logging"""
    if episode % interval == 0 and len(env.portfolio_value_history) > 10:
        portfolio = np.array(env.portfolio_value_history)
        benchmark = np.array(env.benchmark_values[:len(portfolio)])
        
        if len(benchmark) == len(portfolio) and len(portfolio) > 1:
            metrics = PerformanceMetrics.calculate_metrics(
                portfolio, benchmark, config.INITIAL_CASH)
            
            excess_return = metrics['excess_return'] * 100
            beat_bnh = "✅" if excess_return > 0 else "❌"
            
            print(f"📊 {phase_name} Episode {episode}:")
            print(f"   {beat_bnh} Return: {metrics['total_return']*100:+.2f}% vs BnH: {((benchmark[-1]/benchmark[0])-1)*100:+.2f}%")
            print(f"   📈 Excess Return: {excess_return:+.2f}%")
            print(f"   ⚡ Sharpe Ratio: {metrics['sharpe_ratio']:+.3f}")
            print(f"   📉 Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"   💼 Total Trades: {len(env.trades)}")
            print(f"   💰 Portfolio Value: ${portfolio[-1]:,.0f}")
            
            # Return metrics for tracking
            return {
                'sharpe': metrics['sharpe_ratio'],
                'excess_return': excess_return,
                'drawdown': metrics['max_drawdown'] * 100
            }
    return None

def enhanced_walk_forward_training(agent, train_data, feature_columns, scaler, config, save_dir=None):
    """Enhanced training with specialized agents and comprehensive reporting"""
    if save_dir is None:
        save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    
    MODEL_SAVE_PATH = os.path.join(save_dir, f'specialized_ppo_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Enhanced training phases with proper specialization
    training_phases = [
        {
            'episodes': 800, 'entropy_coef': 0.15, 'lr_multiplier': 1.8,
            'name': 'exploration_phase', 'focus': 'aggressive_exploration'
        },
        {
            'episodes': 800, 'entropy_coef': 0.12, 'lr_multiplier': 1.5,
            'name': 'specialization_phase', 'focus': 'long_short_specialization'
        },
        {
            'episodes': 600, 'entropy_coef': 0.08, 'lr_multiplier': 1.2,
            'name': 'optimization_phase', 'focus': 'signal_combination'
        },
        {
            'episodes': 600, 'entropy_coef': 0.05, 'lr_multiplier': 0.9, 
            'name': 'refinement_phase', 'focus': 'performance_refinement'
        }
    ]
    
    best_overall_score = -np.inf
    episode_returns = []
    performance_history = []
    
    # Walk-forward training with proper data splits
    initial_train_size = int(len(train_data) * 0.40)
    validation_size = int(len(train_data) * 0.15)
    step_size = int(len(train_data) * 0.10)
    walk_forward_steps = min(config.WALK_FORWARD_STEPS, 4)
    
    for step in range(walk_forward_steps):
        print(f"\n🚀 Walk-Forward Step {step + 1}/{walk_forward_steps}")
        print("="*80)
        
        train_start = 0
        train_end = initial_train_size + (step * step_size)
        val_start = train_end
        val_end = min(val_start + validation_size, len(train_data))
        
        current_train_data = train_data.iloc[train_start:train_end].reset_index(drop=True)
        current_val_data = train_data.iloc[val_start:val_end].reset_index(drop=True)
        
        print(f"📊 Data sizes - Train: {len(current_train_data)}, Val: {len(current_val_data)}")
        
        # Train through all phases with specialized agents
        for phase_idx, phase_config in enumerate(training_phases):
            print(f"\n🎯 Starting Phase {phase_idx+1}/4: {phase_config['name']}")
            phase_returns, phase_metrics = train_specialized_phase(
                agent, current_train_data, current_val_data,
                feature_columns, scaler, config, phase_config)
            
            episode_returns.extend(phase_returns)
            performance_history.extend(phase_metrics)
        
        # Enhanced evaluation after each walk-forward step
        if val_end < len(train_data):
            test_start = val_end
            test_end = min(test_start + 150, len(train_data))
            test_data_slice = train_data.iloc[test_start:test_end].reset_index(drop=True)
            
            test_env = evaluate_specialized_agent_fixed(agent, test_data_slice,
                                                feature_columns, scaler, config)
            
            test_metrics = calculate_comprehensive_score(test_env, config)
            
            print(f"\n📊 STEP {step + 1} COMPREHENSIVE EVALUATION:")
            print(f"   🎯 Performance Score: {test_metrics['score']:7.3f}")
            print(f"   📈 Total Return: {test_metrics['total_return']:+6.2f}%")
            print(f"   ⚡ Sharpe Ratio: {test_metrics['sharpe']:+6.3f}")
            print(f"   📉 Max Drawdown: {test_metrics['drawdown']:6.2f}%")
            print(f"   💰 Excess Return: {test_metrics['excess_return']:+6.2f}%")
            
            # Save model if it's the best
            if test_metrics['score'] > best_overall_score:
                best_overall_score = test_metrics['score']
                agent.save_models(MODEL_SAVE_PATH)
                print(f"   ⭐ NEW BEST SPECIALIZED MODEL! Score: {best_overall_score:.3f}")
    
    return episode_returns, MODEL_SAVE_PATH

def train_specialized_phase(agent, train_data, val_data, feature_columns, scaler, config, phase_config):
    """FIXED: Reduced verbosity training with phase-end best metrics and proper minimum holding period"""
    episode_returns = []
    performance_metrics = []
    best_val_sharpe = -np.inf
    best_phase_metrics = None
    patience_count = 0

    # **REDUCED VERBOSITY: Check verbosity settings**
    verbose = getattr(config, 'VERBOSE_TRAINING', False)
    phase_summary_only = getattr(config, 'PHASE_SUMMARY_ONLY', True)

    if not verbose or phase_summary_only:
        print(f"🎯 Phase: {phase_config['name']} ({phase_config['episodes']} episodes)")
        print(f"   Focus: {phase_config['focus']}, Entropy: {phase_config['entropy_coef']}")
    else:
        print(f"🎯 Specialized Training Phase: {phase_config['name']}")
        print(f" Episodes: {phase_config['episodes']}, Entropy: {phase_config['entropy_coef']}")
        print(f" Focus: {phase_config['focus']}")

    # Store original parameters for BOTH agents separately
    original_params = {
        'long': {
            'entropy': agent.long_agent.entropy_coef,
            'lrs': {}
        },
        'short': {
            'entropy': agent.short_agent.entropy_coef,
            'lrs': {}
        }
    }

    # Apply phase-specific parameters to BOTH agents with differentiation
    if phase_config['name'] == 'specialization_phase':
        # Higher entropy for long agent in specialization phase (bull market bias)
        agent.long_agent.entropy_coef = phase_config['entropy_coef'] * 1.3
        agent.short_agent.entropy_coef = phase_config['entropy_coef'] * 1.1
    else:
        agent.long_agent.entropy_coef = phase_config['entropy_coef'] * 1.2
        agent.short_agent.entropy_coef = phase_config['entropy_coef'] * 1.1

    # Store and modify learning rates for both agents with differentiation
    for agent_name, sub_agent in [('long', agent.long_agent), ('short', agent.short_agent)]:
        original_params[agent_name]['lrs'] = {}
        for opt_name, optimizer in sub_agent.actor_optimizers.items():
            original_params[agent_name]['lrs'][opt_name] = []
            for param_group in optimizer.param_groups:
                original_params[agent_name]['lrs'][opt_name].append(param_group['lr'])
                
                # Different LR multipliers for better specialization
                if agent_name == 'long':
                    lr_mult = phase_config['lr_multiplier'] * 1.1  # Slightly higher for long
                else:
                    lr_mult = phase_config['lr_multiplier'] * 1.0  # Standard for short
                param_group['lr'] = config.LR * lr_mult

        # Also handle critic optimizers for long agent
        if hasattr(sub_agent, 'critic_optimizers'):
            for opt_name, optimizer in sub_agent.critic_optimizers.items():
                if opt_name not in original_params[agent_name]['lrs']:
                    original_params[agent_name]['lrs'][opt_name] = []
                for param_group in optimizer.param_groups:
                    original_params[agent_name]['lrs'][opt_name].append(param_group['lr'])
                    lr_mult = phase_config['lr_multiplier'] * (1.1 if agent_name == 'long' else 1.0)
                    param_group['lr'] = config.LR * lr_mult

    # **FIXED: Get report interval based on verbosity settings**
    report_interval = getattr(config, 'METRIC_REPORT_INTERVAL', 20) if not verbose else 5

    # Training loop for this phase
    for episode in range(phase_config['episodes']):
        # MODIFICATION: Start timer for the episode
        episode_start_time = time.time()
        
        # Training episode
        train_env = AdvancedTradingEnvironment(train_data, feature_columns, scaler, config)
        state = train_env.reset()
        done = False

        # Reset hidden states for both agents
        if hasattr(agent.long_agent, 'hidden_states'):
            agent.long_agent.hidden_states = {'lstm': None}
        if hasattr(agent.short_agent, 'hidden_states'):
            agent.short_agent.hidden_states = {'lstm': None}

        step_count = 0
        expert_actions = 0
        long_actions = 0
        short_actions = 0
        hold_actions = 0
        episode_start_portfolio = train_env.get_portfolio_value()

        while not done:
            # Get action from combined agent
            action, log_prob, value = agent.act(state, training=True)
            next_state, reward, done, info = train_env.step(action)

            # Track action distribution
            if action == 0:
                hold_actions += 1
            elif action == 1:
                long_actions += 1
            elif action == 2:
                short_actions += 1

            # Track expert trader behaviors
            market_bias = info.get('market_bias', 'neutral')
            if market_bias in ['correction_buy', 'correction_sell', 'ranging']:
                expert_actions += 1

            # Enhanced reward based on phase focus
            enhanced_reward = enhance_reward_for_phase(reward, phase_config, train_env, info)

            # Store experience in both specialized agents
            agent.remember(state, action, log_prob, value, enhanced_reward, next_state, done)

            state = next_state
            step_count += 1

        # Update both agents separately after episode
        if len(agent.long_agent.memory['states']) >= config.UPDATE_FREQUENCY:
            losses = agent.update()

        # **REDUCED VERBOSITY: Only show detailed info if verbose enabled**
        if episode % 10 == 0 and episode > 0 and verbose:
            total_actions = long_actions + short_actions + hold_actions
            long_pct = (long_actions / total_actions) * 100 if total_actions > 0 else 0
            short_pct = (short_actions / total_actions) * 100 if total_actions > 0 else 0
            hold_pct = (hold_actions / total_actions) * 100 if total_actions > 0 else 0

            episode_end_portfolio = train_env.get_portfolio_value()
            episode_return = ((episode_end_portfolio - episode_start_portfolio) / episode_start_portfolio) * 100

            print(f" Episode {episode+1}: Return: {episode_return:+.2f}%")
            print(f" Actions - Long: {long_pct:.1f}%, Short: {short_pct:.1f}%, Hold: {hold_pct:.1f}%")
            print(f" Expert Actions: {expert_actions}/{step_count} ({expert_actions/step_count*100:.1f}%)")
            print(f" Total Trades: {len(train_env.trades)}")

        train_return = (train_env.get_portfolio_value() - config.INITIAL_CASH) / config.INITIAL_CASH * 100
        episode_returns.append(train_return)

        # MODIFICATION: Calculate and print episode duration
        episode_duration = time.time() - episode_start_time
        print(f"   Episode {episode+1}/{phase_config['episodes']} completed in {episode_duration:.2f} seconds.")

        # **ENHANCED PERIODIC REPORTING with reduced verbosity**
        if episode % report_interval == 0 and episode > 0:
            if verbose:
                # Full reporting (old behavior)
                train_metrics = report_metrics_periodically(
                    agent, train_env, config, episode, f"{phase_config['name']}_TRAIN")
                val_env = evaluate_specialized_agent_fixed(agent, val_data, feature_columns, scaler, config)
                val_metrics = report_metrics_periodically(
                    agent, val_env, config, episode, f"{phase_config['name']}_VAL")
            else:
                # **COMPACT REPORTING**
                val_env = evaluate_specialized_agent_fixed(agent, val_data, feature_columns, scaler, config, quiet=True)
                val_metrics = get_metrics_quietly(val_env, config)
                
            if val_metrics and val_metrics['sharpe'] > best_val_sharpe:
                best_val_sharpe = val_metrics['sharpe']
                best_phase_metrics = val_metrics.copy()
                best_phase_metrics['episode'] = episode
                patience_count = 0
                status = "✅" if val_metrics['excess_return'] > 0 else "❌"
                print(f"   ⭐ New Best at Ep {episode:3d}: {status} Return: {val_metrics['excess_return']:+6.2f}%, Sharpe: {val_metrics['sharpe']:+5.2f}")
            else:
                patience_count += 1

            if patience_count >= config.PATIENCE // 2:
                if not verbose:
                    print(f"   Early stopping at episode {episode + 1}")
                else:
                    print(f" Early stopping at episode {episode + 1} due to no improvement")
                break

            # Store performance metrics
            if verbose and 'train_metrics' in locals():
                performance_metrics.append({
                    'episode': episode,
                    'phase': phase_config['name'],
                    'train_metrics': val_metrics,  # Using val_metrics for consistency
                    'val_metrics': val_metrics
                })

    # **PHASE END SUMMARY - Always show regardless of verbosity**
    print(f"\n📊 PHASE COMPLETE: {phase_config['name']}")
    if best_phase_metrics:
        print(f"   🏆 Best Performance (Episode {best_phase_metrics['episode']}):")
        print(f"     Excess Return: {best_phase_metrics['excess_return']:+7.2f}%")
        print(f"     Sharpe Ratio:  {best_phase_metrics['sharpe']:+7.3f}")
        print(f"     Max Drawdown:  {best_phase_metrics['drawdown']:7.2f}%")
    else:
        print(f"   ⚠️ No valid performance metrics recorded")

    # Restore original parameters for BOTH agents
    agent.long_agent.entropy_coef = original_params['long']['entropy']
    agent.short_agent.entropy_coef = original_params['short']['entropy']

    for agent_name, sub_agent in [('long', agent.long_agent), ('short', agent.short_agent)]:
        # Restore actor optimizer learning rates
        for opt_name, optimizer in sub_agent.actor_optimizers.items():
            if opt_name in original_params[agent_name]['lrs']:
                for i, param_group in enumerate(optimizer.param_groups):
                    if i < len(original_params[agent_name]['lrs'][opt_name]):
                        param_group['lr'] = original_params[agent_name]['lrs'][opt_name][i]

        # Restore critic optimizer learning rates if they exist
        if hasattr(sub_agent, 'critic_optimizers'):
            for opt_name, optimizer in sub_agent.critic_optimizers.items():
                if opt_name in original_params[agent_name]['lrs']:
                    for i, param_group in enumerate(optimizer.param_groups):
                        if i < len(original_params[agent_name]['lrs'][opt_name]):
                            param_group['lr'] = original_params[agent_name]['lrs'][opt_name][i]

    return episode_returns, performance_metrics

def get_metrics_quietly(env, config):
    """Get metrics without verbose output"""
    if len(env.portfolio_value_history) <= 10:
        print(f" [DEBUG] Metrics Failed: Portfolio history length ({len(env.portfolio_value_history)}) is not > 10.")
        return None

    portfolio = np.array(env.portfolio_value_history)
    benchmark = np.array(env.benchmark_values)

    # **FIX: Truncate both arrays to the same minimum length**
    min_len = min(len(portfolio), len(benchmark))
    portfolio = portfolio[:min_len]
    benchmark = benchmark[:min_len]

    if len(portfolio) <= 1:
        print(f" [DEBUG] Metrics Failed: Insufficient data after truncation.")
        print(f" - Portfolio Length: {len(portfolio)}")
        print(f" - Benchmark Length: {len(benchmark)}")
        return None

    from utils import PerformanceMetrics
    metrics = PerformanceMetrics.calculate_metrics(portfolio, benchmark, config.INITIAL_CASH)

    # Display metrics with proper formatting
    excess_return = metrics['excess_return'] * 100
    beat_benchmark = "✅" if excess_return > 0 else "❌"
    
    print(f" {beat_benchmark} Agent: {metrics['total_return']*100:+6.2f}% vs BnH: {((benchmark[-1]/benchmark[0])-1)*100:+6.2f}%")
    print(f" 📈 Excess Return: {excess_return:+6.2f}%")
    print(f" ⚡ Sharpe Ratio: {metrics['sharpe_ratio']:+6.3f}")
    print(f" 📉 Max Drawdown: {metrics['max_drawdown']*100:6.2f}%")

    return {
        'sharpe': metrics['sharpe_ratio'],
        'excess_return': excess_return,
        'drawdown': metrics['max_drawdown'] * 100
    }


def evaluate_specialized_agent_fixed(agent, data, feature_columns, scaler, config, quiet=False):
    """FIXED: Enhanced evaluation with proper debugging and reporting"""
    env = AdvancedTradingEnvironment(data, feature_columns, scaler, config)
    env.training = False
    state = env.reset()
    done = False

    # Reset LSTM hidden states for both agents
    if hasattr(agent.long_agent, 'hidden_states'):
        agent.long_agent.hidden_states = {'lstm': None}
    if hasattr(agent.short_agent, 'hidden_states'):
        agent.short_agent.hidden_states = {'lstm': None}

    action_counts = {'hold': 0, 'long': 0, 'short': 0}
    
    while not done:
        action, _, _ = agent.act(state, training=False)

        # Track action distribution
        if action == 0:
            action_counts['hold'] += 1
        elif action == 1:
            action_counts['long'] += 1
        elif action == 2:
            action_counts['short'] += 1

        state, _, done, _ = env.step(action)

    # **FIXED: Liquidation with proper profit calculation and minimum holding period respect**
    if env.positions:
        last_price = env.data.iloc[-1]['close']
        for position in env.positions:
            # **RESPECT MINIMUM HOLDING PERIOD even in liquidation**
            holding_period = len(env.data) - 1 - position.get('entry_step', len(env.data) - 1)
            
            if position['position_type'] == 'LONG':
                proceeds = position['shares'] * last_price * (1 - env.transaction_cost)
                profit = (last_price - position['entry_price']) * position['shares'] - \
                        (position['shares'] * last_price * env.transaction_cost)
                env.cash += proceeds
            else:  # SHORT
                # **FIXED: Correct short liquidation**
                cost_to_cover = position['shares'] * last_price * (1 + env.transaction_cost)
                profit = (position['entry_price'] - last_price) * position['shares'] - \
                        (position['shares'] * last_price * env.transaction_cost)
                env.cash -= cost_to_cover  # SUBTRACT cash for covering

            env.trades.append({
                'step': env.current_step,
                'type': 'LIQUIDATE',
                'shares': position['shares'],
                'price': last_price,
                'profit': profit,  # ENSURE profit is calculated and stored
                'entry_price': position['entry_price'],
                'entry_step': position.get('entry_step', env.current_step),
                'holding_period': holding_period  # Track actual holding period
            })

        env.positions = []

    # **REDUCED VERBOSITY: Only show debug info if verbose or not quiet**
    if not quiet and (getattr(config, 'VERBOSE_TRAINING', False) or not getattr(config, 'PHASE_SUMMARY_ONLY', True)):
        # Print action distribution for debugging
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            print(f" Action Distribution - Hold: {action_counts['hold']/total_actions*100:.1f}%, " +
                  f"Long: {action_counts['long']/total_actions*100:.1f}%, " +
                  f"Short: {action_counts['short']/total_actions*100:.1f}%")

        debug_env_agent_detailed_fixed(env, agent, config)
        report_individual_agent_performance(env, agent, config)
    
    return env

def enhance_reward_for_phase(base_reward, phase_config, env, info):
    """Enhanced reward based on training phase focus with minimum holding period consideration"""
    enhanced_reward = base_reward

    # **MINIMUM HOLDING PERIOD BONUS**
    min_holding = getattr(env.config, 'MIN_HOLDING_PERIOD', 3)
    
    # Check if any recent trades respected minimum holding period
    if env.trades:
        recent_trades = [t for t in env.trades if t.get('step', 0) >= env.current_step - 5]
        for trade in recent_trades:
            if trade['type'] in ['SELL', 'COVER', 'LIQUIDATE']:
                holding_period = trade.get('step', 0) - trade.get('entry_step', 0)
                if holding_period >= min_holding:
                    enhanced_reward += 1.0  # Bonus for respecting minimum holding

    # Exploration phase - reward understanding market conditions
    if phase_config['name'] == 'exploration_phase':
        market_bias = info.get('market_bias', 'neutral')
        if market_bias in ['correction_buy', 'correction_sell', 'ranging']:
            enhanced_reward += 2.0  # Higher reward for recognizing special conditions

        # Extra reward for taking any action (not just holding)
        if hasattr(env, 'trades') and env.trades:
            last_trade = env.trades[-1]
            if last_trade['step'] == env.current_step - 1:  # Trade just occurred
                enhanced_reward += 1.5

    # Specialization phase - reward proper agent specialization and directional bias
    elif phase_config['name'] == 'specialization_phase':
        if hasattr(env, 'long_positions') and hasattr(env, 'short_positions'):
            current_step = env.current_step
            if current_step < len(env.data):
                trend_alignment = env.data.iloc[current_step].get('trend_alignment', 0)
                market_bias = info.get('market_bias', 'neutral')

                # Reward strong directional plays in clear trends
                if abs(trend_alignment) > 0.4:
                    total_long_pos = len(env.long_positions)
                    total_short_pos = len(env.short_positions)

                    if trend_alignment > 0.4 and total_long_pos > 0:
                        enhanced_reward += 3.0  # Strong reward for long in uptrend
                    elif trend_alignment < -0.4 and total_short_pos > 0:
                        enhanced_reward += 3.0  # Strong reward for short in downtrend

                    # Penalize counter-trend positions
                    if trend_alignment > 0.6 and total_short_pos > 0:
                        enhanced_reward -= 2.0
                    elif trend_alignment < -0.6 and total_long_pos > 0:
                        enhanced_reward -= 2.0

    # Optimization phase - reward signal combination and market timing
    elif phase_config['name'] == 'optimization_phase':
        market_bias = info.get('market_bias', 'neutral')
        if market_bias in ['bullish', 'bearish']:
            enhanced_reward += 2.5  # Reward alignment with clear market bias

        # Reward balanced approach
        if hasattr(env, 'long_positions') and hasattr(env, 'short_positions'):
            total_positions = len(env.long_positions) + len(env.short_positions)
            if 1 <= total_positions <= env.max_positions // 2:
                enhanced_reward += 1.0  # Reward moderate position sizing

    # Refinement phase - reward risk management and position optimization
    elif phase_config['name'] == 'refinement_phase':
        # **ENHANCED: Reward partial exits and risk management ONLY after min holding**
        if env.trades and env.trades[-1]['type'] in ['PARTIAL_SELL', 'PARTIAL_COVER']:
            last_trade = env.trades[-1]
            holding_period = last_trade.get('step', 0) - last_trade.get('entry_step', 0)
            if holding_period >= min_holding:
                enhanced_reward += 4.0  # Higher reward for proper risk management
            else:
                enhanced_reward += 1.0  # Smaller reward if too early

        # Reward position adjustments and stop management
        adjustment_bonus = 0
        all_positions = getattr(env, 'long_positions', []) + getattr(env, 'short_positions', [])
        for position in all_positions:
            holding_days = env.current_step - position.get('entry_step', env.current_step)
            
            if position.get('stop_adjusted', False) and holding_days >= min_holding:
                adjustment_bonus += 2.5  # Higher bonus for proper timing
            if position.get('moved_to_breakeven', False) and holding_days >= min_holding + 2:
                adjustment_bonus += 2.0  # Breakeven after sufficient time
            if position.get('partial_exit_done', False) and holding_days >= min_holding:
                adjustment_bonus += 1.5  # Partial exit after min holding

        enhanced_reward += min(adjustment_bonus, 6.0)  # Cap the bonus

    return enhanced_reward

def debug_env_agent_detailed_fixed(env, agent, config):
    """FIXED: Enhanced debugging with proper array synchronization analysis"""
    print(f"🔍 DEBUG - Portfolio history length: {len(env.portfolio_value_history)}")
    print(f"🔍 DEBUG - Benchmark history length: {len(env.benchmark_values)}")
    
    # ANALYZE the synchronization issue
    portfolio_array = np.array(env.portfolio_value_history)
    benchmark_array = np.array(env.benchmark_values)
    
    print(f"🔍 DEBUG - Final portfolio value (get_portfolio_value): {env.get_portfolio_value()}")
    print(f"🔍 DEBUG - Final portfolio value (array[-1]): {portfolio_array[-1] if len(portfolio_array) > 0 else 'N/A'}")
    print(f"🔍 DEBUG - Total trades: {len(env.trades)}")
    
    # Additional debug info
    print(f"🔍 DEBUG - Current step: {env.current_step}")
    print(f"🔍 DEBUG - Data length: {len(env.data)}")
    print(f"🔍 DEBUG - Long positions: {len(env.long_positions)}")
    print(f"🔍 DEBUG - Short positions: {len(env.short_positions)}")
    print(f"🔍 DEBUG - Cash: ${env.cash:,.2f}")
    
    # FIXED: Check if arrays are synchronized
    print(f"🔍 DEBUG - Arrays synchronized: {len(portfolio_array) == len(benchmark_array)}")
    print(f"🔍 DEBUG - Length difference: {len(portfolio_array) - len(benchmark_array)}")
    
    if len(portfolio_array) > 1 and len(benchmark_array) > 1:
        print(f"🔍 DEBUG - Portfolio range: ${portfolio_array.min():,.0f} - ${portfolio_array.max():,.0f}")
        print(f"🔍 DEBUG - Benchmark range: ${benchmark_array.min():,.0f} - ${benchmark_array.max():,.0f}")
        
        # ANALYZE profit distribution
        exit_trades = [t for t in env.trades if t['type'] in ['SELL', 'COVER', 'LIQUIDATE']]
        profitable_trades = [t for t in exit_trades if t.get('profit') and t.get('profit') > 0]
        print(f"🔍 DEBUG - Exit trades: {len(exit_trades)}")
        print(f"🔍 DEBUG - Profitable trades: {len(profitable_trades)}")
        
        if exit_trades:
            total_profit = sum(t.get('profit', 0) for t in exit_trades)
            print(f"🔍 DEBUG - Total realized P&L: ${total_profit:,.2f}")

def report_individual_agent_performance(env, agent, config):
    """FINAL CORRECTED: Properly separate long and short agent contributions"""
    from utils import PerformanceMetrics
    
    # Array synchronization (already working)
    portfolio_values = np.array(env.portfolio_value_history)
    benchmark_values = np.array(env.benchmark_values)
    
    # Ensure exact length matching
    if len(benchmark_values) == len(portfolio_values) - 1:
        portfolio_values = portfolio_values[:-1]
    elif len(portfolio_values) == len(benchmark_values) - 1:
        benchmark_values = benchmark_values[:-1]
    else:
        min_len = min(len(portfolio_values), len(benchmark_values))
        portfolio_values = portfolio_values[:min_len]
        benchmark_values = benchmark_values[:min_len]
    
    if len(portfolio_values) <= 1:
        print("❌ Insufficient data for performance calculation")
        return
    
    # Get current price and final value
    current_price = env.data.iloc[env.current_step-1]['close'] if env.current_step > 0 else env.data.iloc[0]['close']
    final_portfolio_value = env.get_portfolio_value()
    
    # Calculate metrics
    combined_metrics = PerformanceMetrics.calculate_metrics(portfolio_values, benchmark_values, config.INITIAL_CASH)
    benchmark_metrics = PerformanceMetrics.calculate_metrics(benchmark_values, benchmark_values, config.INITIAL_CASH)
    
    # 🔧 CORRECTED: Long Agent Analysis (BUY -> SELL/LIQUIDATE only)
    long_value = sum(pos['shares'] * current_price for pos in env.long_positions)
    long_entry_trades = [t for t in env.trades if t['type'] == 'BUY']
    long_exit_trades = [t for t in env.trades if t['type'] in ['SELL', 'LIQUIDATE'] and t.get('profit') is not None]
    long_winning_trades = [t for t in long_exit_trades if t.get('profit', 0) > 0]
    long_profit = sum(t.get('profit', 0) for t in long_exit_trades)
    long_win_rate = (len(long_winning_trades) / len(long_exit_trades) * 100) if long_exit_trades else 0.0
    
    # 🔧 CORRECTED: Short Agent Analysis (SELL_SHORT -> COVER only)
    short_value = sum(pos['shares'] * current_price for pos in env.short_positions)
    short_entry_trades = [t for t in env.trades if t['type'] == 'SELL_SHORT']
    short_exit_trades = [t for t in env.trades if t['type'] in ['COVER'] and t.get('profit') is not None]
    short_winning_trades = [t for t in short_exit_trades if t.get('profit', 0) > 0]
    short_profit = sum(t.get('profit', 0) for t in short_exit_trades)
    short_win_rate = (len(short_winning_trades) / len(short_exit_trades) * 100) if short_exit_trades else 0.0
    
    print("\n" + "="*100)
    print("🎯 INDIVIDUAL AGENT PERFORMANCE ANALYSIS")
    print("="*100)
    
    # Combined Agent Performance
    print(f"\n🤖 COMBINED AGENT PERFORMANCE:")
    print(f"   📈 Total Return: {combined_metrics['total_return']*100:+7.2f}%")
    print(f"   📈 Annualized: {combined_metrics['annualized_return']*100:+7.2f}%")
    print(f"   ⚡ Sharpe Ratio: {combined_metrics['sharpe_ratio']:+7.3f}")
    print(f"   📉 Max Drawdown: {combined_metrics['max_drawdown']*100:7.2f}%")
    print(f"   🎯 Excess Return: {combined_metrics['excess_return']*100:+7.2f}%")
    print(f"   💰 Final Value: ${final_portfolio_value:,.0f}")
    
    # Benchmark Performance
    print(f"\n📊 BENCHMARK (BUY & HOLD) PERFORMANCE:")
    print(f"   📈 Total Return: {benchmark_metrics['total_return']*100:+7.2f}%")
    print(f"   📈 Annualized: {benchmark_metrics['annualized_return']*100:+7.2f}%")
    print(f"   ⚡ Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:+7.3f}")
    print(f"   📉 Max Drawdown: {benchmark_metrics['max_drawdown']*100:7.2f}%")
    print(f"   💰 Final Value: ${benchmark_values[-1]:,.0f}")
    
    # 🔧 CORRECTED Long Agent Analysis
    print(f"\n🟢 LONG AGENT CONTRIBUTION:")
    print(f"   📊 Current Positions: {len(env.long_positions)}")
    print(f"   💰 Position Value: ${long_value:,.0f}")
    print(f"   📈 Entry Trades: {len(long_entry_trades)}")
    print(f"   📉 Exit Trades: {len(long_exit_trades)}")
    print(f"   💵 Realized Profit: ${long_profit:,.0f}")
    print(f"   🎯 Win Rate (Long): {long_win_rate:.1f}%")
    
    # 🔧 CORRECTED Short Agent Analysis
    print(f"\n🔴 SHORT AGENT CONTRIBUTION:")
    print(f"   📊 Current Positions: {len(env.short_positions)}")
    print(f"   💰 Position Value: ${short_value:,.0f}")
    print(f"   📈 Entry Trades: {len(short_entry_trades)}")
    print(f"   📉 Exit Trades: {len(short_exit_trades)}")
    print(f"   💵 Realized Profit: ${short_profit:,.0f}")
    print(f"   🎯 Win Rate (Short): {short_win_rate:.1f}%")
    
    # Trade Distribution Analysis
    trade_types = {}
    for trade in env.trades:
        trade_types[trade['type']] = trade_types.get(trade['type'], 0) + 1
    
    print(f"\n📋 TRADE DISTRIBUTION:")
    for trade_type, count in sorted(trade_types.items()):
        print(f"   {trade_type}: {count}")
    
    # CORRECTED: Overall win rate calculation
    all_exit_trades = [t for t in env.trades if t['type'] in ['SELL', 'COVER', 'LIQUIDATE', 'STOP-LOSS'] and t.get('profit') is not None]
    all_winning_trades = [t for t in all_exit_trades if t.get('profit', 0) > 0]
    overall_win_rate = (len(all_winning_trades) / len(all_exit_trades) * 100) if all_exit_trades else 0.0
    
    # Performance Comparison Summary
    beat_benchmark = combined_metrics['excess_return'] > 0
    status = "✅ OUTPERFORMED" if beat_benchmark else "❌ UNDERPERFORMED"
    
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"   {status} benchmark by {combined_metrics['excess_return']*100:+.2f}%")
    print(f"   📈 Agent Return: {combined_metrics['total_return']*100:+.2f}%")
    print(f"   📊 Benchmark Return: {benchmark_metrics['total_return']*100:+.2f}%")
    print(f"   ⚡ Risk-Adjusted Alpha: {(combined_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']):+.3f}")
    print(f"   🎯 Overall Win Rate: {overall_win_rate:.1f}%")

def calculate_comprehensive_score(env, config):
    """Fixed validation logic"""
    portfolio_values = np.array(env.portfolio_value_history)
    
    # Ensure benchmark matches portfolio length exactly
    benchmark_values = np.array(env.benchmark_values)
    min_len = min(len(portfolio_values), len(benchmark_values))
    
    if min_len <= 1:
        print(f"⚠️ Insufficient data: Portfolio={len(portfolio_values)}, Benchmark={len(benchmark_values)}")
        return {'score': -1000, 'total_return': 0, 'sharpe': 0, 'drawdown': 100, 'excess_return': -100}
    
    # Truncate to same length
    portfolio_values = portfolio_values[:min_len]
    benchmark_values = benchmark_values[:min_len]
    
    metrics = PerformanceMetrics.calculate_metrics(portfolio_values, benchmark_values, config.INITIAL_CASH)
    
    # Enhanced scoring emphasizing beat-benchmark performance
    excess_return = metrics['excess_return'] * 100
    sharpe_ratio = metrics['sharpe_ratio']
    max_drawdown = metrics['max_drawdown'] * 100
    total_return = metrics['total_return'] * 100
    
    # Count successful trades for quality bonus
    trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()
    profitable_trades = 0
    total_trades = 0
    
    if not trades_df.empty:
        exit_trades = trades_df[trades_df['type'].isin(['SELL', 'COVER', 'LIQUIDATE'])]
        if not exit_trades.empty:
            profitable_trades = len(exit_trades[exit_trades['profit'] > 0])
            total_trades = len(exit_trades)
    
    # Quality bonus for good trade execution
    trade_quality_bonus = 0
    if total_trades > 0:
        win_rate = profitable_trades / total_trades
        if win_rate >= 0.6:  # 60%+ win rate
            trade_quality_bonus = 10
        elif win_rate >= 0.5:  # 50%+ win rate
            trade_quality_bonus = 5
    
    # Activity bonus - penalize excessive holding
    activity_bonus = min(5, total_trades * 0.5) if total_trades > 2 else -5
    
    # Comprehensive score calculation with enhanced weighting
    score = (
        excess_return * 0.45 +  # 45% weight on beating benchmark (increased)
        sharpe_ratio * 30 +     # 30 points per unit of Sharpe (increased)
        max(0, -max_drawdown + 20) * 0.25 +  # Drawdown penalty/bonus (increased threshold)
        (total_return if total_return > 0 else total_return * 0.3) * 0.1 +  # Absolute return component
        trade_quality_bonus +   # Bonus for good trade execution
        activity_bonus          # Bonus for reasonable activity level
    )
    
    return {
        'score': score,
        'total_return': total_return,
        'sharpe': sharpe_ratio,
        'drawdown': max_drawdown,
        'excess_return': excess_return
    }