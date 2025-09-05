"""
Utility functions for performance evaluation and results management
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(portfolio_values, benchmark_values, initial_capital, trading_days=252):
        """Calculate comprehensive performance metrics"""
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        # Total returns
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        benchmark_total_return = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]
        
        # Excess return
        excess_return = total_return - benchmark_total_return
        
        # Annualized returns
        periods = len(portfolio_values) / trading_days
        annualized_return = (portfolio_values[-1] / initial_capital) ** (1/periods) - 1 if periods > 0 else 0
        benchmark_annualized = (benchmark_values[-1] / benchmark_values[0]) ** (1/periods) - 1 if periods > 0 else 0
        excess_annualized_return = annualized_return - benchmark_annualized
        
        # Volatility
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(trading_days) if len(portfolio_returns) > 1 else 0
        
        # Sharpe Ratio
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Sortino Ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(trading_days) if len(downside_returns) > 1 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        return {
            'total_return': total_return,
            'excess_return': excess_return,
            'excess_annualized_return': excess_annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'annualized_return': annualized_return
        }
    
    @staticmethod
    def calculate_stability_score(train_metrics, val_metrics, train_weight=0.3, val_weight=0.7):
        """Calculate stability score with emphasis on beating benchmark"""
        def normalize_metric(value, is_positive=True):
            if is_positive:
                return max(0, min(1, (value + 1) / 2))
            else:
                return max(0, min(1, 1 - value))
        
        # Train score components - using excess_annualized_return to beat benchmark
        train_score = (
            normalize_metric(train_metrics['excess_annualized_return']) * 0.40 +
            normalize_metric(train_metrics['sharpe_ratio']) * 0.30 +
            normalize_metric(train_metrics['sortino_ratio']) * 0.20 +
            normalize_metric(train_metrics['max_drawdown'], is_positive=False) * 0.10
        )
        
        # Validation score components - using excess_annualized_return to beat benchmark
        val_score = (
            normalize_metric(val_metrics['excess_annualized_return']) * 0.35 +
            normalize_metric(val_metrics['sharpe_ratio']) * 0.30 +
            normalize_metric(val_metrics['sortino_ratio']) * 0.25 +
            normalize_metric(val_metrics['max_drawdown'], is_positive=False) * 0.10
        )
        
        # Combined stability score with penalty for large differences
        diff_penalty = abs(train_score - val_score) * 0.5
        raw_score = (train_score * train_weight + val_score * val_weight)
        stability_score = raw_score - diff_penalty
        
        return stability_score, train_score, val_score

class PerformanceEvaluator:
    def __init__(self, config):
        self.config = config
    
    def comprehensive_evaluation(self, agent, test_data, feature_columns, scaler):
        """Comprehensive evaluation on test set"""
        from trading_environment import AdvancedTradingEnvironment
        
        results = {}
        
        # B&H benchmark starts from INITIAL_CASH
        benchmark_values = test_data['close'].values / test_data['close'].iloc[0] * self.config.INITIAL_CASH
        
        # PPO evaluation starts with fresh INITIAL_CASH
        env = AdvancedTradingEnvironment(test_data, feature_columns, scaler, self.config)
        env.training = False
        state = env.reset()
        done = False
        
        # Reset LSTM hidden states for PPO agent
        if hasattr(agent, 'hidden_states'):
            agent.hidden_states = {'lstm': None}
        
        while not done:
            action, _, _ = agent.act(state, training=False)  # PPO returns 3 values
            state, _, done, _ = env.step(action)
        
        # Liquidate remaining positions
        self._liquidate_positions(env)
        
        portfolio_values = np.array(env.portfolio_value_history)
        metrics = PerformanceMetrics.calculate_metrics(portfolio_values, benchmark_values, self.config.INITIAL_CASH)
        
        results['ensemble'] = {
            'env': env,
            'portfolio_values': portfolio_values,
            'metrics': metrics
        }
        
        return results
    
    def _liquidate_positions(self, env):
        """Liquidate remaining positions"""
        if env.positions:
            last_price = env.data.iloc[-1]['close']
            for position in env.positions:
                proceeds = position['shares'] * last_price * (1 - env.transaction_cost)
                profit = (last_price - position['entry_price']) * position['shares'] - \
                        (position['shares'] * last_price * env.transaction_cost)
                env.cash += proceeds
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

class ResultsSaver:
    def save_comprehensive_results(self, test_results, asset, timeframe, results_folder,
                                 episode_returns, test_data, config):
        """Save all results comprehensively"""
        os.makedirs(results_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ensemble_env = test_results['ensemble']['env']
        ensemble_metrics = test_results['ensemble']['metrics']
        
        # Calculate performance metrics
        final_portfolio_value = ensemble_env.get_portfolio_value()
        test_annual_return = self._calculate_annualized_return(
            config.INITIAL_CASH, final_portfolio_value, len(test_data))
        test_bnh_return = self._calculate_annualized_return(
            test_data.iloc[0]['close'], test_data.iloc[-1]['close'], len(test_data))
        excess_annual = test_annual_return - test_bnh_return
        
        # Save results summary
        results_summary = {
            'asset': asset,
            'timeframe': timeframe,
            'timestamp': timestamp,
            'algorithm': 'PPO',  # Added algorithm type
            'final_portfolio_value': final_portfolio_value,
            'agent_annual_return': test_annual_return * 100,
            'bnh_annual_return': test_bnh_return * 100,
            'excess_annual_return': excess_annual * 100,
            'beat_benchmark': excess_annual > 0,
            'sharpe_ratio': ensemble_metrics['sharpe_ratio'],
            'max_drawdown': ensemble_metrics['max_drawdown'] * 100
        }
        
        with open(os.path.join(results_folder, f'{asset}_{timeframe}_results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save detailed trades
        trades_df = pd.DataFrame(ensemble_env.trades)
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(results_folder, f'{asset}_{timeframe}_detailed_trades.csv'), index=False)
        
        # Ensure arrays are same length
        portfolio_history = ensemble_env.portfolio_value_history
        close_prices = test_data['close'].values
        
        # Truncate to the minimum length to ensure alignment
        min_length = min(len(portfolio_history), len(close_prices))
        portfolio_history = portfolio_history[:min_length]
        close_prices = close_prices[:min_length]
        
        # Calculate benchmark values based on actual portfolio history length
        benchmark_values = close_prices / close_prices[0] * config.INITIAL_CASH
        
        # Save portfolio history with matching lengths
        portfolio_df = pd.DataFrame({
            'step': range(len(portfolio_history)),
            'portfolio_value': portfolio_history,
            'benchmark_value': benchmark_values
        })
        portfolio_df.to_csv(os.path.join(results_folder, f'{asset}_{timeframe}_portfolio_history.csv'), index=False)
        
        print(f"✅ Results saved for {asset}_{timeframe}")
    
    def _calculate_annualized_return(self, start_value, end_value, periods, trading_days=252):
        """Calculate annualized return"""
        if periods <= 0 or start_value <= 0:
            return 0.0
        
        years = periods / trading_days
        if years <= 0:
            return 0.0
        
        return (end_value / start_value) ** (1/years) - 1
