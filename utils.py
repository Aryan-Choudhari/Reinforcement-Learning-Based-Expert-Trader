"""
Utility functions for performance evaluation and results management
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(portfolio_values, benchmark_values, initial_capital, trading_days=252):
        """Calculate comprehensive performance metrics with proper alignment"""
        
        # CRITICAL FIX: Ensure arrays are same length
        min_length = min(len(portfolio_values), len(benchmark_values))
        portfolio_values = portfolio_values[:min_length]
        benchmark_values = benchmark_values[:min_length]
        
        if len(portfolio_values) < 2:
            # Return default metrics if insufficient data
            return {
                'total_return': 0.0, 'benchmark_total_return': 0.0, 'excess_return': 0.0,
                'annualized_return': 0.0, 'benchmark_annualized_return': 0.0, 'excess_annualized_return': 0.0,
                'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0,
                'benchmark_max_drawdown': 0.0, 'benchmark_sharpe_ratio': 0.0, 'benchmark_sortino_ratio': 0.0
            }
        
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        benchmark_total_return = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]
        
        excess_return = total_return - benchmark_total_return
        
        periods = len(portfolio_values) / trading_days
        annualized_return = (portfolio_values[-1] / initial_capital) ** (1/periods) - 1 if periods > 0 else 0
        benchmark_annualized_return = (benchmark_values[-1] / benchmark_values[0]) ** (1/periods) - 1 if periods > 0 else 0
        excess_annualized_return = annualized_return - benchmark_annualized_return
        
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(trading_days) if len(portfolio_returns) > 1 else 0
        benchmark_vol = np.std(benchmark_returns) * np.sqrt(trading_days) if len(benchmark_returns) > 1 else 0

        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        benchmark_sharpe_ratio = (benchmark_annualized_return - risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0

        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(trading_days) if len(downside_returns) > 1 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # ADDED: Benchmark Sortino Ratio Calculation
        benchmark_downside_returns = benchmark_returns[benchmark_returns < 0]
        benchmark_downside_vol = np.std(benchmark_downside_returns) * np.sqrt(trading_days) if len(benchmark_downside_returns) > 1 else 0
        benchmark_sortino_ratio = (benchmark_annualized_return - risk_free_rate) / benchmark_downside_vol if benchmark_downside_vol > 0 else 0

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        benchmark_peak = np.maximum.accumulate(benchmark_values)
        benchmark_drawdown = (benchmark_peak - benchmark_values) / benchmark_peak
        benchmark_max_drawdown = np.max(benchmark_drawdown) if len(benchmark_drawdown) > 0 else 0
        
        return {
            'total_return': total_return,
            'benchmark_total_return': benchmark_total_return,
            'excess_return': excess_return,
            'annualized_return': annualized_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'excess_annualized_return': excess_annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
            'benchmark_sortino_ratio': benchmark_sortino_ratio, # <-- ADDED
        }
    
    @staticmethod
    def calculate_stability_score(train_metrics, val_metrics, train_weight=0.35, val_weight=0.65):
        """
        Calculate stability score with emphasis on beating benchmark.
        Returns -1 if train OR validation total return is below -50%.
        """
        # CHECK FOR CATASTROPHIC FAILURE FIRST
        if train_metrics['excess_return'] <= -0.5 or val_metrics['excess_return'] <= -0.5:
            return -1.0, -1.0, -1.0
        
        def normalize_metric(value, is_positive=True):
            if is_positive:
                return max(0, min(1, (value + 1) / 2))
            else:
                return max(0, min(1, 1 - value))
        
        train_score = (
            normalize_metric(train_metrics['excess_annualized_return']) * 0.50 +
            normalize_metric(train_metrics['sharpe_ratio']) * 0.30 +
            normalize_metric(train_metrics['max_drawdown'], is_positive=False) * 0.20
        )
        
        val_score = (
            normalize_metric(val_metrics['excess_annualized_return']) * 0.50 +
            normalize_metric(val_metrics['sharpe_ratio']) * 0.30 +
            normalize_metric(val_metrics['max_drawdown'], is_positive=False) * 0.20
        )
        
        diff_penalty = abs(train_score - val_score) * 0.5
        raw_score = (train_score * train_weight + val_score * val_weight)
        stability_score = raw_score - diff_penalty
        
        return stability_score, train_score, val_score


class PerformanceEvaluator:
    def __init__(self, config):
        self.config = config
    
    def comprehensive_evaluation(self, agent, test_data, feature_columns, scaler):
        """Comprehensive evaluation on test set against baseline strategies."""
        from trading_environment import AdvancedTradingEnvironment
        
        results = {}
        
        # --- 1. Evaluate the RL Agent ---
        env = AdvancedTradingEnvironment(test_data, feature_columns, scaler, self.config)
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, training=False)
            state, _, done, _ = env.step(action)
        self._liquidate_positions(env)
        
        # Get agent portfolio values (already includes all steps from 0)
        agent_portfolio_values = np.array(env.portfolio_value_history)
        
        # --- 2. Create Buy & Hold benchmark aligned to agent's timeline ---
        # Buy & Hold buys at step 0 and holds
        initial_price = test_data['close'].iloc[0]
        bnh_shares = self.config.INITIAL_CASH / initial_price
        
        # Create benchmark for each step the agent traded
        bnh_portfolio_values = np.array([
            bnh_shares * test_data['close'].iloc[i] 
            for i in range(len(agent_portfolio_values))
        ])
        
        # Ensure same length (should already be true, but safety check)
        min_length = min(len(agent_portfolio_values), len(bnh_portfolio_values))
        agent_portfolio_values = agent_portfolio_values[:min_length]
        bnh_portfolio_values = bnh_portfolio_values[:min_length]
        
        print(f"Debug - Agent portfolio length: {len(agent_portfolio_values)}")
        print(f"Debug - Benchmark portfolio length: {len(bnh_portfolio_values)}")
        
        # --- Calculate metrics for all strategies ---
        results['ensemble'] = {
            'env': env,
            'portfolio_values': agent_portfolio_values,
            'metrics': PerformanceMetrics.calculate_metrics(
                agent_portfolio_values, bnh_portfolio_values, self.config.INITIAL_CASH
            )
        }
        results['buy_and_hold'] = {
            'portfolio_values': bnh_portfolio_values,
            'metrics': PerformanceMetrics.calculate_metrics(
                bnh_portfolio_values, bnh_portfolio_values, self.config.INITIAL_CASH
            )
        }
        
        return results

    def _liquidate_positions(self, env):
        """Liquidate remaining positions"""
        if env.positions:
            last_price = env.data.iloc[-1]['close']
            for position in env.positions:
                proceeds = position['shares'] * last_price * (1 - env.transaction_cost)
                env.cash += proceeds
            env.positions = []

class ResultsSaver:
    def save_comprehensive_results(self, test_results, asset, timeframe, results_folder,
                                episode_returns, test_data, config):
        """Save all results comprehensively and return a summary dictionary."""
        os.makedirs(results_folder, exist_ok=True)
        
        ensemble_metrics = test_results['ensemble']['metrics']
        
        trades_df = pd.DataFrame(test_results['ensemble']['env'].trades)
        exit_trades = trades_df[trades_df['type'].isin(['SELL', 'COVER', 'STOP-LOSS', 'LIQUIDATE', 
                                                         'LIQUIDATE_LONG', 'LIQUIDATE_SHORT'])] if not trades_df.empty else pd.DataFrame()
        win_rate = len(exit_trades[exit_trades['profit'] > 0]) / len(exit_trades) * 100 if len(exit_trades) > 0 else 0

        if not exit_trades.empty:
            profitable_sum = exit_trades[exit_trades['profit'] > 0]['profit'].sum()
            losing_sum = exit_trades[exit_trades['profit'] <= 0]['profit'].sum()
            profit_factor = abs(profitable_sum / losing_sum) if losing_sum != 0 else float('inf')
        else:
            profit_factor = 0.0

        results_summary = {
            'asset_timeframe': f"{asset}_{timeframe}",
            'final_portfolio_value': test_results['ensemble']['portfolio_values'][-1],
            'total_return_pct': ensemble_metrics['total_return'] * 100,
            'benchmark_total_return_pct': ensemble_metrics['benchmark_total_return'] * 100,
            'sharpe_ratio': ensemble_metrics['sharpe_ratio'],
            'sortino_ratio': ensemble_metrics['sortino_ratio'],
            'max_drawdown_pct': ensemble_metrics['max_drawdown'] * 100,
            'total_trades': len(exit_trades),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor
        }
        
        with open(os.path.join(results_folder, f'{asset}_{timeframe}_results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(results_folder, f'{asset}_{timeframe}_detailed_trades.csv'), index=False)
        
        # FIXED: Ensure arrays are same length before creating DataFrame
        agent_portfolio = test_results['ensemble']['portfolio_values']
        bnh_portfolio = test_results['buy_and_hold']['portfolio_values']
        
        min_length = min(len(agent_portfolio), len(bnh_portfolio))
        
        portfolio_df = pd.DataFrame({
            'step': range(min_length),
            'agent_portfolio': agent_portfolio[:min_length],
            'buy_and_hold_portfolio': bnh_portfolio[:min_length]
        })
        portfolio_df.to_csv(os.path.join(results_folder, f'{asset}_{timeframe}_portfolio_history.csv'), index=False)
        
        print(f"✅ Results saved for {asset}_{timeframe}")
        return results_summary

class TradingVisualizer:
    def __init__(self):
        pass
    
    def create_and_save_individual_charts(self, env, asset_name, save_dir="trading_charts"):
        """Create and save all trading charts as separate images"""
        os.makedirs(save_dir, exist_ok=True)
        
        prices = env.data['close'].values
        portfolio_values = np.array(env.portfolio_value_history)
        
        # Handle dates properly
        if 'timestamp' in env.data.columns:
            dates = pd.to_datetime(env.data['timestamp']).values
            # Align dates with portfolio history length
            portfolio_dates = dates[:len(portfolio_values)]
        else:
            dates = pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
            portfolio_dates = dates[:len(portfolio_values)]
        
        chart_paths = {}
        
        # Generate each chart separately
        chart_paths['price_action'] = self.plot_price_action(env, dates, asset_name, save_dir)
        chart_paths['portfolio_performance'] = self.plot_portfolio_performance(portfolio_dates, portfolio_values, prices, asset_name, save_dir)
        chart_paths['drawdown'] = self.plot_drawdown(portfolio_dates, portfolio_values, prices, asset_name, save_dir)
        chart_paths['rolling_returns'] = self.plot_rolling_returns(portfolio_dates, portfolio_values, prices, asset_name, save_dir)
        chart_paths['trade_pnl'] = self.plot_trade_pnl(env, asset_name, save_dir)
        chart_paths['position_allocation'] = self.plot_position_allocation(env, portfolio_dates, asset_name, save_dir)
        chart_paths['performance_summary'] = self.plot_performance_summary(env, asset_name, save_dir)
        
        return chart_paths
    
    def plot_portfolio_performance(self, portfolio_dates, portfolio_values, prices, asset_name, save_dir):
        """Plot portfolio performance vs benchmark with proper alignment"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create benchmark aligned to portfolio history length
        initial_price = prices[0]
        benchmark_shares = portfolio_values[0] / initial_price
        
        # Generate benchmark values for the same number of steps
        benchmark_values = np.array([
            benchmark_shares * prices[i] 
            for i in range(min(len(portfolio_values), len(prices)))
        ])
        
        # Ensure same length
        min_length = min(len(portfolio_values), len(benchmark_values), len(portfolio_dates))
        portfolio_values = portfolio_values[:min_length]
        benchmark_values = benchmark_values[:min_length]
        portfolio_dates = portfolio_dates[:min_length]
        
        ax.plot(portfolio_dates, portfolio_values, 
               label='Agent Portfolio', color='green', linewidth=2.5, alpha=0.9)
        ax.plot(portfolio_dates, benchmark_values,
               label='Buy & Hold', color='blue', alpha=0.7, linewidth=2)
        
        # Calculate performance metrics for title
        final_agent_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
        final_bnh_return = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0] * 100
        excess_return = final_agent_return - final_bnh_return
        
        ax.set_title(f'{asset_name} - Portfolio Performance\nAgent: {final_agent_return:+.1f}% vs B&H: {final_bnh_return:+.1f}% (Excess: {excess_return:+.1f}%)',
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f'{asset_name}_portfolio_performance.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return path
    
    def plot_price_action(self, env, dates, asset_name, save_dir):
            """Plot price action with trade signals - CORRECTED FOR DESCRIPTIVE LABELS"""
            fig, ax = plt.subplots(figsize=(15, 8))
            prices = env.data['close'].values

            ax.plot(dates, prices, label='Close Price', color='blue', linewidth=1.5, alpha=0.8)

            trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()

            if not trades_df.empty:
                # Define keywords for different trade types
                buy_long_kw = 'BUY_LONG'
                sell_short_kw = 'SELL_SHORT'
                sell_long_kw = 'SELL_LONG'
                cover_short_kw = 'COVER_SHORT'

                # Helper function to plot trades
                def plot_trades(keyword, marker, color, label, size=120):
                    trades = trades_df[trades_df['type'].str.contains(keyword, na=False)]
                    if not trades.empty:
                        visible_trades = []
                        for _, trade in trades.iterrows():
                            if trade['step'] < len(dates):
                                visible_trades.append((dates[trade['step']], trade['price']))
                        if visible_trades:
                            trade_dates, trade_prices = zip(*visible_trades)
                            ax.scatter(trade_dates, trade_prices, color=color, marker=marker,
                                    s=size, alpha=0.8, zorder=5, label=label)

                # Plot entry signals
                plot_trades(buy_long_kw, '^', 'green', 'Buy Long')
                plot_trades(sell_short_kw, 'v', 'red', 'Sell Short')

                # Plot exit signals (differentiating profit/loss)
                def plot_exits(keyword, marker, profit_color, loss_color, label_prefix, size=100):
                    # Ensure we don't double-plot entry signals
                    entry_keywords = ['BUY_LONG', 'SELL_SHORT']
                    exit_trades = trades_df[trades_df['type'].str.contains(keyword, na=False) & 
                                            ~trades_df['type'].isin(entry_keywords)]
                    
                    if not exit_trades.empty:
                        profit_exits, loss_exits = [], []
                        for _, trade in exit_trades.iterrows():
                            if trade['step'] < len(dates):
                                (profit_exits if trade.get('profit', 0) > 0 else loss_exits).append((dates[trade['step']], trade['price']))

                        if profit_exits:
                            dates_p, prices_p = zip(*profit_exits)
                            ax.scatter(dates_p, prices_p, color=profit_color, marker=marker, s=size, alpha=0.8, zorder=5, label=f'{label_prefix} (Profit)')
                        if loss_exits:
                            dates_l, prices_l = zip(*loss_exits)
                            ax.scatter(dates_l, prices_l, color=loss_color, marker=marker, s=size, alpha=0.8, zorder=5, label=f'{label_prefix} (Loss)')

                plot_exits(sell_long_kw, 'x', 'cyan', 'purple', 'Exit Long')
                plot_exits(cover_short_kw, 'x', 'lime', 'orange', 'Cover Short')

            ax.set_title(f'{asset_name} - Price Action with Trading Signals',
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=12)

            # Create a legend with unique labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            
            # Format dates on x-axis
            if isinstance(dates[0], (pd.Timestamp, datetime)):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // (30 * 6)))) # Aim for ~6 labels
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            
            plt.tight_layout()
            path = os.path.join(save_dir, f'{asset_name}_price_action.png')
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            return path

    def plot_drawdown(self, portfolio_dates, portfolio_values, prices, asset_name, save_dir):
        """Plot drawdown analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Agent drawdown
        agent_peak = np.maximum.accumulate(portfolio_values)
        agent_drawdown = (agent_peak - portfolio_values) / agent_peak * 100
        
        ax1.fill_between(portfolio_dates, -agent_drawdown, 0,
                        alpha=0.6, color='red', label=f'Agent DD (Max: {np.max(agent_drawdown):.1f}%)')
        ax1.axhline(-15, color='red', linestyle='--', alpha=0.5, label='15% Warning Level')
        ax1.set_title('Agent Drawdown Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Drawdown (%)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Buy & Hold drawdown
        benchmark_values = prices / prices[0] * portfolio_values[0]
        benchmark_portfolio = benchmark_values[:len(portfolio_values)]
        bnh_peak = np.maximum.accumulate(benchmark_portfolio)
        bnh_drawdown = (bnh_peak - benchmark_portfolio) / bnh_peak * 100
        
        ax2.fill_between(portfolio_dates, -bnh_drawdown, 0,
                        alpha=0.4, color='blue', label=f'B&H DD (Max: {np.max(bnh_drawdown):.1f}%)')
        ax2.axhline(-15, color='red', linestyle='--', alpha=0.5, label='15% Warning Level')
        ax2.set_title('Buy & Hold Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f'{asset_name}_drawdown.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return path
    
    def plot_rolling_returns(self, portfolio_dates, portfolio_values, prices, asset_name, save_dir):
        """Plot rolling returns comparison"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        window = min(30, max(5, len(portfolio_values)//10))
        
        if len(portfolio_values) > window:
            agent_rolling = pd.Series(portfolio_values).pct_change(window).fillna(0) * 100
            
            benchmark_values = prices / prices[0] * portfolio_values[0]
            benchmark_portfolio = benchmark_values[:len(portfolio_values)]
            bnh_rolling = pd.Series(benchmark_portfolio).pct_change(window).fillna(0) * 100
            
            ax.plot(portfolio_dates, agent_rolling, 
                   label=f'Agent {window}D Returns', color='green', alpha=0.8, linewidth=2)
            ax.plot(portfolio_dates, bnh_rolling,
                   label=f'B&H {window}D Returns', color='blue', alpha=0.8, linewidth=2)
            ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title(f'{asset_name} - {window}-Day Rolling Returns Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Returns (%)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f'{asset_name}_rolling_returns.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return path
    
    def plot_trade_pnl(self, env, asset_name, save_dir):
        """Plot individual trade P&L distribution"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()
        
        if not trades_df.empty:
            # Get only exit trades for P&L analysis
            exit_trades = trades_df[trades_df['type'].isin(['SELL', 'COVER', 'STOP-LOSS', 'LIQUIDATE'])]
            
            if not exit_trades.empty:
                profits = exit_trades['profit'].values
                colors = ['green' if p > 0 else 'red' for p in profits]
                
                bars = ax.bar(range(len(profits)), profits, color=colors, alpha=0.7, width=0.8)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # Add value labels on significant trades
                for i, (bar, profit) in enumerate(zip(bars, profits)):
                    if abs(profit) > max(abs(profits)) * 0.3:  # Label only significant trades
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'${profit:.0f}',
                               ha='center', va='bottom' if profit > 0 else 'top',
                               fontsize=8, fontweight='bold')
                
                ax.set_title(f'{asset_name} - Individual Trade P&L (Total Trades: {len(exit_trades)})', 
                           fontsize=16, fontweight='bold')
                ax.set_ylabel('Profit/Loss ($)', fontsize=12)
                ax.set_xlabel('Trade Number', fontsize=12)
        else:
            ax.set_title(f'{asset_name} - No Trades Executed', fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f'{asset_name}_trade_pnl.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return path
    
    def plot_position_allocation(self, env, portfolio_dates, asset_name, save_dir):
        """Plot position allocation over time - CORRECTED VERSION"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Check what position data is actually available in the environment
        if hasattr(env, 'positions') and env.positions is not None:
            # Get position history if it exists, otherwise show current position count
            if hasattr(env, 'position_history'):
                # If environment tracks position history over time
                position_counts = env.position_history
            else:
                # Current positions repeated over time (fallback)
                current_positions = len(env.positions)
                position_counts = [current_positions] * len(portfolio_dates)
            
            max_positions = getattr(env, 'max_positions', 4)
            
            # Plot actual position count
            ax.plot(portfolio_dates, position_counts, color='blue', linewidth=2, 
                label=f'Active Positions (Current: {len(env.positions)})', alpha=0.8)
            
            # Show maximum allowed positions as reference line
            ax.axhline(y=max_positions, color='red', linestyle='--', alpha=0.7, 
                    label=f'Max Positions ({max_positions})')
            
            ax.set_ylabel('Number of Positions', fontsize=12)
            
        else:
            # No position data available
            ax.plot(portfolio_dates, [0] * len(portfolio_dates), color='gray', 
                linewidth=2, label='No Position Data Available')
            ax.set_ylabel('Position Count', fontsize=12)
        
        ax.set_title(f'{asset_name} - Position Allocation Over Time', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        if len(portfolio_dates) > 0:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f'{asset_name}_position_allocation.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return path

    
    def plot_performance_summary(self, env, asset_name, save_dir):
        """Plot comprehensive performance summary"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        # Calculate comprehensive metrics
        portfolio_values = np.array(env.portfolio_value_history)
        trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()
        
        total_trades = len(trades_df) if not trades_df.empty else 0
        exit_trades = trades_df[trades_df['type'].isin(['SELL', 'COVER', 'STOP-LOSS', 'LIQUIDATE'])] if not trades_df.empty else pd.DataFrame()
        win_rate = len(exit_trades[exit_trades['profit'] > 0]) / len(exit_trades) * 100 if len(exit_trades) > 0 else 0
        
        # Trade performance metrics
        if len(exit_trades) > 0:
            avg_win = exit_trades[exit_trades['profit'] > 0]['profit'].mean() if len(exit_trades[exit_trades['profit'] > 0]) > 0 else 0
            avg_loss = exit_trades[exit_trades['profit'] <= 0]['profit'].mean() if len(exit_trades[exit_trades['profit'] <= 0]) > 0 else 0
            max_win = exit_trades['profit'].max()
            max_loss = exit_trades['profit'].min()
            
            profitable_sum = exit_trades[exit_trades['profit'] > 0]['profit'].sum()
            losing_sum = exit_trades[exit_trades['profit'] <= 0]['profit'].sum()
            profit_factor = abs(profitable_sum / losing_sum) if losing_sum != 0 else float('inf')
        else:
            avg_win = avg_loss = max_win = max_loss = 0
            profit_factor = 0
        
        # Portfolio metrics
        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Risk metrics
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Comprehensive performance summary
        summary_text = f"""
📊 COMPREHENSIVE PERFORMANCE SUMMARY - {asset_name}

💰 Portfolio Metrics:
   Initial Value: ${initial_value:,.0f}
   Final Value: ${final_value:,.0f}
   Total Return: {total_return:+.2f}%
   Max Drawdown: {max_drawdown:.2f}%

📋 Trading Statistics:
   Total Signals: {total_trades}
   Completed Trades: {len(exit_trades)}
   Win Rate: {win_rate:.1f}%
   Profit Factor: {profit_factor:.2f}

💵 Trade Performance:
   Average Win: ${avg_win:,.0f}
   Average Loss: ${avg_loss:,.0f}
   Best Trade: ${max_win:,.0f}
   Worst Trade: ${max_loss:,.0f}

💰 Final Status:
   Cash: ${getattr(env, 'cash', 0):,.0f}
   Positions: {len(getattr(env, 'positions', []))}
   Total Equity: ${final_value:,.0f}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        path = os.path.join(save_dir, f'{asset_name}_performance_summary.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return path
    
"""
Enhanced Logging System for Detailed Trade Analysis and Walk-Forward Results - CORRECTED
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

class DetailedTradeLogger:
    """Logs detailed trade information for each best model"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.logs_dir = os.path.join(save_dir, 'trade_logs')
        os.makedirs(self.logs_dir, exist_ok=True)

    def log_best_model_trades(self, env, model_id, metrics, phase_info=None):
            """
            Save comprehensive trade log when a new best model is found - CORRECTED
            """
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"trades_{model_id}.csv"
            log_path = os.path.join(self.logs_dir, log_filename)

            if not env.trades:
                print(f"   No trades to log for {model_id}")
                return None

            trades_df = pd.DataFrame(env.trades).reset_index(drop=True)

            # Add calculated columns
            trades_df['holding_period'] = trades_df['step'] - trades_df['entry_step']
            trades_df['profit_pct'] = trades_df.apply(
                lambda row: (row['profit'] / (row['entry_price'] * row['shares']) * 100)
                if row['shares'] > 0 and row['entry_price'] > 0 else 0,
                axis=1
            ).round(8)
            trades_df['trade_value'] = (trades_df['shares'] * trades_df['price']).round(8)
            trades_df['is_profitable'] = trades_df['profit'] > 0
            trades_df['transaction_cost'] = (trades_df['shares'] * trades_df['price'] * env.transaction_cost).round(8)

            # --- FIX: Make exit trade detection robust ---
            # Instead of a fixed list, check for keywords that signify an exit.
            exit_keywords = ['SELL_LONG', 'COVER_SHORT', 'LIQUIDATE', 'EMERGENCY']
            exit_trades_mask = trades_df['type'].apply(lambda x: any(keyword in x for keyword in exit_keywords))
            
            # Correctly calculate cumulative profit
            trades_df['cumulative_profit'] = trades_df['profit'].where(exit_trades_mask).cumsum().fillna(method='ffill').fillna(0.0).round(8)

            # Calculate portfolio history using the corrected history calculator
            try:
                # Pass the full environment data to the history calculator
                portfolio_history = self._calculate_portfolio_history(trades_df, env.initial_cash, env.transaction_cost, env.data)
                
                if len(portfolio_history) == len(trades_df):
                    trades_df['portfolio_value'] = portfolio_history
                else:
                    # Fallback for length mismatch
                    trades_df['portfolio_value'] = env.initial_cash + trades_df['cumulative_profit']
                    
            except Exception as e:
                print(f"   ERROR calculating portfolio history: {e}")
                trades_df['portfolio_value'] = env.initial_cash + trades_df['cumulative_profit']

            # Reorder columns for better readability
            cols = ['step', 'type', 'shares', 'price', 'profit', 'entry_price', 'entry_step',
                    'holding_period', 'profit_pct', 'trade_value', 'is_profitable',
                    'transaction_cost', 'cumulative_profit', 'portfolio_value']
            
            # Ensure all columns exist before reordering
            final_cols = [col for col in cols if col in trades_df.columns]
            trades_df = trades_df[final_cols]

            trades_df.to_csv(log_path, index=False)
            self._create_trade_summary_report(trades_df, model_id, metrics, phase_info, timestamp)
            print(f"   Trade log saved: {log_filename}")
            return log_path

    def _calculate_portfolio_history(self, trades_df, initial_cash, transaction_cost, market_data):
            """
            CORRECTED: Calculate portfolio value for each trade row using actual market prices.
            """
            portfolio_values = []
            running_cash = initial_cash
            
            open_long_positions = {}
            open_short_positions = {}
            
            for i, trade in trades_df.iterrows():
                trade_type = trade['type']
                shares = trade['shares']
                price = trade['price']
                entry_step = trade['entry_step']
                current_step = trade['step']
                
                # --- FIX: Use the correct market price for valuation ---
                # Get the closing market price at the step the trade occurred.
                if current_step < len(market_data):
                    market_price_at_step = market_data.iloc[current_step]['close']
                else: # Handle trades at the very end
                    market_price_at_step = market_data.iloc[-1]['close']

                # Process the trade to update running_cash and open positions
                if 'BUY_LONG' in trade_type:
                    cost = shares * price * (1 + transaction_cost)
                    running_cash -= cost
                    if entry_step not in open_long_positions:
                        open_long_positions[entry_step] = {'shares': 0, 'entry_price': price}
                    open_long_positions[entry_step]['shares'] += shares
                    
                elif 'SELL_SHORT' in trade_type:
                    proceeds = shares * price * (1 - transaction_cost)
                    running_cash += proceeds
                    if entry_step not in open_short_positions:
                        open_short_positions[entry_step] = {'shares': 0, 'entry_price': price}
                    open_short_positions[entry_step]['shares'] += shares
                    
                elif 'SELL_LONG' in trade_type or 'LIQUIDATE_LONG' in trade_type:
                    if entry_step in open_long_positions:
                        proceeds = shares * price * (1 - transaction_cost)
                        running_cash += proceeds
                        open_long_positions[entry_step]['shares'] -= shares
                        if open_long_positions[entry_step]['shares'] < 0.001:
                            del open_long_positions[entry_step]
                            
                elif 'COVER_SHORT' in trade_type or 'LIQUIDATE_SHORT' in trade_type:
                    if entry_step in open_short_positions:
                        cost = shares * price * (1 + transaction_cost)
                        running_cash -= cost
                        open_short_positions[entry_step]['shares'] -= shares
                        if open_short_positions[entry_step]['shares'] < 0.001:
                            del open_short_positions[entry_step]
                
                # --- Correctly calculate portfolio value AFTER the trade ---
                # Start with the updated cash balance
                current_portfolio_value = running_cash
                
                # Add value of all open long positions using the MARKET price
                for pos in open_long_positions.values():
                    current_portfolio_value += pos['shares'] * market_price_at_step
                
                # Subtract liability of all open short positions using the MARKET price
                for pos in open_short_positions.values():
                    current_portfolio_value -= pos['shares'] * market_price_at_step
                
                portfolio_values.append(round(current_portfolio_value, 2))
            
            return portfolio_values

    def _create_trade_summary_report(self, trades_df, model_id, metrics, phase_info, timestamp):
        # This method remains unchanged but will now receive a correctly calculated DataFrame.
        # ... (rest of the method is the same as the original)
        """Create detailed summary report for the trades"""
        report_filename = f"trade_summary_{model_id}.txt"
        report_path = os.path.join(self.logs_dir, report_filename)

        exit_trades = trades_df[trades_df['type'].isin(['SELL', 'COVER', 'STOP-LOSS', 'LIQUIDATE', 'LIQUIDATE_LONG', 'LIQUIDATE_SHORT', 'PARTIAL_SELL'])]

        total_transaction_costs = trades_df['transaction_cost'].sum()

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"DETAILED TRADE SUMMARY REPORT - {model_id}\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("="*80 + "\n\n")

            if phase_info:
                f.write("PHASE INFORMATION:\n")
                f.write("-" * 80 + "\n")
                for key, value in phase_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'return' in key.lower() or 'drawdown' in key.lower():
                        f.write(f"  {key}: {value*100:+.2f}%\n")
                    else:
                        f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("TRADE STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Signals: {len(trades_df)}\n")
            f.write(f"  Total Exit Trades: {len(exit_trades)}\n")

            if not exit_trades.empty:
                winning_trades = exit_trades[exit_trades['profit'] > 0]
                losing_trades = exit_trades[exit_trades['profit'] <= 0]

                f.write(f"  Winning Trades: {len(winning_trades)}\n")
                f.write(f"  Losing Trades: {len(losing_trades)}\n")
                f.write(f"  Win Rate: {len(winning_trades)/len(exit_trades)*100:.2f}%\n\n")

                f.write("PROFIT ANALYSIS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Total Net Profit: ${exit_trades['profit'].sum():,.2f}\n")
                f.write(f"  Total Transaction Costs: ${total_transaction_costs:,.2f}\n")
                f.write(f"  Average Profit per Trade: ${exit_trades['profit'].mean():,.2f}\n")
                f.write(f"  Median Profit per Trade: ${exit_trades['profit'].median():,.2f}\n")
                f.write(f"  Best Trade: ${exit_trades['profit'].max():,.2f}\n")
                f.write(f"  Worst Trade: ${exit_trades['profit'].min():,.2f}\n\n")

                if len(winning_trades) > 0:
                    f.write(f"  Average Win: ${winning_trades['profit'].mean():,.2f}\n")
                    f.write(f"  Largest Win: ${winning_trades['profit'].max():,.2f}\n")

                if len(losing_trades) > 0:
                    f.write(f"  Average Loss: ${losing_trades['profit'].mean():,.2f}\n")
                    f.write(f"  Largest Loss: ${losing_trades['profit'].min():,.2f}\n\n")

                if len(winning_trades) > 0 and len(losing_trades) > 0:
                    gross_profit = winning_trades['profit'].sum()
                    gross_loss = abs(losing_trades['profit'].sum())
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    f.write(f"  Profit Factor: {profit_factor:.2f}\n\n")

                f.write("HOLDING PERIOD ANALYSIS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Average Holding Period: {exit_trades['holding_period'].mean():.1f} steps\n")
                f.write(f"  Median Holding Period: {exit_trades['holding_period'].median():.1f} steps\n")
                f.write(f"  Shortest Hold: {exit_trades['holding_period'].min()} steps\n")
                f.write(f"  Longest Hold: {exit_trades['holding_period'].max()} steps\n\n")

                f.write("TRADE TYPE BREAKDOWN:\n")
                f.write("-" * 80 + "\n")
                type_summary = trades_df['type'].value_counts()
                for trade_type, count in type_summary.items():
                    f.write(f"  {trade_type}: {count}\n")
                f.write("\n")

                if not exit_trades.empty:
                    f.write("EXIT TYPE PROFIT ANALYSIS:\n")
                    f.write("-" * 80 + "\n")
                    for exit_type in exit_trades['type'].unique():
                        type_trades = exit_trades[exit_trades['type'] == exit_type]
                        avg_profit = type_trades['profit'].mean()
                        total_profit = type_trades['profit'].sum()
                        f.write(f"  {exit_type}:\n")
                        f.write(f"    Count: {len(type_trades)}\n")
                        f.write(f"    Total Profit: ${total_profit:,.2f}\n")
                        f.write(f"    Average Profit: ${avg_profit:,.2f}\n\n")

                f.write("TOP 10 MOST PROFITABLE TRADES:\n")
                f.write("-" * 80 + "\n")
                top_trades = exit_trades.nlargest(10, 'profit')[['step', 'type', 'shares', 'price', 'profit', 'profit_pct', 'holding_period']]
                f.write(top_trades.to_string(index=False))
                f.write("\n\n")

                f.write("TOP 10 WORST TRADES:\n")
                f.write("-" * 80 + "\n")
                worst_trades = exit_trades.nsmallest(10, 'profit')[['step', 'type', 'shares', 'price', 'profit', 'profit_pct', 'holding_period']]
                f.write(worst_trades.to_string(index=False))
                f.write("\n\n")

            f.write("CAPITAL UTILIZATION ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            buy_trades = trades_df[trades_df['type'].isin(['BUY', 'SELL_SHORT'])]
            if not buy_trades.empty:
                avg_position_size = (buy_trades['shares'] * buy_trades['price']).mean()
                max_position_size = (buy_trades['shares'] * buy_trades['price']).max()
                f.write(f"  Average Position Size: ${avg_position_size:,.2f}\n")
                f.write(f"  Maximum Position Size: ${max_position_size:,.2f}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"   Trade summary saved: {report_filename}")

class WalkForwardLogger:
    """Logs comprehensive walk-forward validation results"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.wf_logs_dir = os.path.join(save_dir, 'walkforward_logs')
        os.makedirs(self.wf_logs_dir, exist_ok=True)
        
        self.walk_forward_results = []
    
    def log_walk_forward_step(self, step_num, train_data, val_data, test_data, 
                              train_metrics, val_metrics, test_metrics, 
                              stability_score, model_path):
        """Log results from a single walk-forward step"""
        
        step_result = {
            'step': step_num,
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data) if test_data is not None else 0,
                'train_date_range': f"{train_data['timestamp'].iloc[0]} to {train_data['timestamp'].iloc[-1]}",
                'val_date_range': f"{val_data['timestamp'].iloc[0]} to {val_data['timestamp'].iloc[-1]}",
            },
            'train_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                            for k, v in train_metrics.items()},
            'val_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                          for k, v in val_metrics.items()},
            'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                           for k, v in test_metrics.items()} if test_metrics else None,
            'stability_score': float(stability_score),
            'model_path': model_path
        }
        
        self.walk_forward_results.append(step_result)
        
        self._save_step_report(step_result)
    
    def _save_step_report(self, step_result):
        """Save detailed report for a single walk-forward step"""
        step_num = step_result['step']
        filename = f"walkforward_step_{step_num}_report.txt"
        filepath = os.path.join(self.wf_logs_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"WALK-FORWARD STEP {step_num} DETAILED REPORT\n")
            f.write(f"Generated: {step_result['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            f.write("DATA INFORMATION:\n")
            f.write("-" * 80 + "\n")
            for key, value in step_result['data_info'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("TRAINING SET PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            self._write_metrics(f, step_result['train_metrics'])
            f.write("\n")
            
            f.write("VALIDATION SET PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            self._write_metrics(f, step_result['val_metrics'])
            f.write("\n")
            
            if step_result['test_metrics']:
                f.write("TEST SET PERFORMANCE (HOLDOUT):\n")
                f.write("-" * 80 + "\n")
                self._write_metrics(f, step_result['test_metrics'])
                f.write("\n")
            
            f.write("STABILITY ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Overall Stability Score: {step_result['stability_score']:.4f}\n")
            
            train_return = step_result['train_metrics'].get('total_return', 0)
            val_return = step_result['val_metrics'].get('total_return', 0)
            return_diff = abs(train_return - val_return)
            
            f.write(f"  Train-Val Return Difference: {return_diff*100:.2f}%\n")
            
            if return_diff > 0.20:
                f.write("  WARNING: Large train-validation discrepancy detected!\n")
            
            f.write(f"\n  Model saved to: {step_result['model_path']}\n")
            f.write("\n" + "="*80 + "\n")
    
    def _write_metrics(self, f, metrics):
        """Helper to write metrics in formatted way"""
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'return' in key.lower() or 'drawdown' in key.lower():
                    f.write(f"  {key}: {value*100:+.2f}%\n")
                else:
                    f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
    
    def save_walk_forward_summary(self, asset_name):
        """Save comprehensive summary of all walk-forward steps"""
        summary_file = os.path.join(self.wf_logs_dir, f"walkforward_summary_{asset_name}.json")
        
        with open(summary_file, 'w') as f:
            json.dump({
                'asset': asset_name,
                'total_steps': len(self.walk_forward_results),
                'walk_forward_results': self.walk_forward_results
            }, f, indent=2)
        
        self._create_walk_forward_comparison_csv(asset_name)
        
        print(f"Walk-forward summary saved: {summary_file}")
    
    def _create_walk_forward_comparison_csv(self, asset_name):
        """Create CSV comparing all walk-forward steps"""
        csv_file = os.path.join(self.wf_logs_dir, f"walkforward_comparison_{asset_name}.csv")
        
        comparison_data = []
        for result in self.walk_forward_results:
            row = {
                'step': result['step'],
                'train_samples': result['data_info']['train_samples'],
                'val_samples': result['data_info']['val_samples'],
                'train_return': result['train_metrics'].get('total_return', 0) * 100,
                'val_return': result['val_metrics'].get('total_return', 0) * 100,
                'train_sharpe': result['train_metrics'].get('sharpe_ratio', 0),
                'val_sharpe': result['val_metrics'].get('sharpe_ratio', 0),
                'train_drawdown': result['train_metrics'].get('max_drawdown', 0) * 100,
                'val_drawdown': result['val_metrics'].get('max_drawdown', 0) * 100,
                'stability_score': result['stability_score']
            }
            
            if result['test_metrics']:
                row['test_return'] = result['test_metrics'].get('total_return', 0) * 100
                row['test_sharpe'] = result['test_metrics'].get('sharpe_ratio', 0)
                row['test_drawdown'] = result['test_metrics'].get('max_drawdown', 0) * 100
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(csv_file, index=False)
        print(f"Walk-forward comparison CSV saved: {csv_file}")
        
class DiscrepancyDetector:
    """Detects and reports discrepancies in trading results"""
    
    @staticmethod
    def check_capital_conservation(env, initial_cash):
        """Enhanced capital conservation check with detailed breakdown"""
        issues = []
        
        final_portfolio_value = env.get_portfolio_value()
        
        # Categorize trades
        entry_trades = [t for t in env.trades if t['type'] in ['BUY_LONG', 'SELL_SHORT']]
        exit_trades = [t for t in env.trades if 'SELL_LONG' in t['type'] or 'COVER_SHORT' in t['type'] 
                    or t['type'] in ['LIQUIDATE_LONG', 'LIQUIDATE_SHORT', 'EMERGENCY_SELL_LONG', 'EMERGENCY_COVER_SHORT']]
        
        # Calculate cash flows
        cash_out = 0  # Money spent on long entries
        cash_in_from_shorts = 0  # Money received from short entries
        cash_in_from_long_exits = 0  # Money from selling longs
        cash_out_to_cover = 0  # Money spent covering shorts
        
        for trade in env.trades:
            if trade['type'] == 'BUY_LONG':
                cash_out += trade['shares'] * trade['price'] * (1 + env.transaction_cost)
            elif trade['type'] == 'SELL_SHORT':
                cash_in_from_shorts += trade['shares'] * trade['price'] * (1 - env.transaction_cost)
            elif 'SELL_LONG' in trade['type'] or trade['type'] in ['LIQUIDATE_LONG', 'EMERGENCY_SELL_LONG']:
                cash_in_from_long_exits += trade['shares'] * trade['price'] * (1 - env.transaction_cost)
            elif 'COVER_SHORT' in trade['type'] or trade['type'] in ['LIQUIDATE_SHORT', 'EMERGENCY_COVER_SHORT']:
                cash_out_to_cover += trade['shares'] * trade['price'] * (1 + env.transaction_cost)
        
        # Calculate expected final cash
        expected_cash = initial_cash - cash_out + cash_in_from_shorts + cash_in_from_long_exits - cash_out_to_cover
        
        # Get open position values
        current_price = env.data.iloc[env.current_step]['close'] if env.current_step < len(env.data) else env.data.iloc[-1]['close']
        open_long_value = sum(pos['shares'] * current_price for pos in env.long_positions)
        open_short_liability = sum(pos['shares'] * current_price for pos in env.short_positions)
        
        expected_final = expected_cash + open_long_value - open_short_liability
        
        tolerance = max(initial_cash * 0.001, 10.0)
        difference = abs(expected_final - final_portfolio_value)
        
        if difference > tolerance:
            issues.append(f"WARNING: Capital discrepancy detected!")
            issues.append(f"  Expected: ${expected_final:,.2f}")
            issues.append(f"  Actual: ${final_portfolio_value:,.2f}")
            issues.append(f"  Difference: ${difference:,.2f} ({difference/initial_cash*100:.2f}%)")
            issues.append(f"  ")
            issues.append(f"  Cash Flow Breakdown:")
            issues.append(f"    Initial Cash: ${initial_cash:,.2f}")
            issues.append(f"    - Spent on Long Entries: ${cash_out:,.2f}")
            issues.append(f"    + Received from Short Entries: ${cash_in_from_shorts:,.2f}")
            issues.append(f"    + Received from Long Exits: ${cash_in_from_long_exits:,.2f}")
            issues.append(f"    - Spent Covering Shorts: ${cash_out_to_cover:,.2f}")
            issues.append(f"    = Expected Cash: ${expected_cash:,.2f}")
            issues.append(f"    + Open Long Value: ${open_long_value:,.2f}")
            issues.append(f"    - Open Short Liability: ${open_short_liability:,.2f}")
            issues.append(f"    = Expected Portfolio: ${expected_final:,.2f}")
            issues.append(f"  ")
            issues.append(f"  Trade Counts:")
            issues.append(f"    Long Entries: {len([t for t in entry_trades if t['type'] == 'BUY_LONG'])}")
            issues.append(f"    Short Entries: {len([t for t in entry_trades if t['type'] == 'SELL_SHORT'])}")
            issues.append(f"    Long Exits: {len([t for t in exit_trades if 'SELL_LONG' in t['type']])}")
            issues.append(f"    Short Covers: {len([t for t in exit_trades if 'COVER_SHORT' in t['type']])}")
            issues.append(f"  Open Longs: {len(env.long_positions)}, Open Shorts: {len(env.short_positions)}")
        
        return issues

    @staticmethod
    def check_trade_consistency(trades_df):
        """Check for inconsistencies in trade data"""
        issues = []
        
        if trades_df.empty:
            return ["No trades executed"]
        
        if (trades_df['shares'] < 0).any():
            issues.append("WARNING: Negative share quantities detected")
        
        if (trades_df['price'] <= 0).any():
            issues.append("WARNING: Invalid prices detected (zero or negative)")
        
        exit_trades = trades_df[trades_df['type'].isin(['SELL', 'COVER', 'STOP-LOSS', 'LIQUIDATE', 'LIQUIDATE_LONG', 'LIQUIDATE_SHORT'])]
        if not exit_trades.empty:
            avg_profit = exit_trades['profit'].mean()
            std_profit = exit_trades['profit'].std()
            
            if std_profit > 0:
                outliers = exit_trades[abs(exit_trades['profit'] - avg_profit) > 5 * std_profit]
                if len(outliers) > 0:
                    issues.append(f"WARNING: {len(outliers)} potential outlier trades detected")
        
        return issues
    
"""
Comprehensive performance reporting for individual models
Generates detailed comparison reports and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class ModelPerformanceReporter:
    """Generate detailed performance reports for all models"""
    
    def __init__(self, save_dir='reports'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def generate_comprehensive_report(self, test_results, asset_name, config):
        """Generate comprehensive HTML report with all metrics"""
        
        report_path = os.path.join(self.save_dir, f'{asset_name}_comprehensive_report.html')
        
        # Create comprehensive metrics table
        metrics_df = self._create_metrics_dataframe(test_results)
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Performance Report - {asset_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .best-model {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .metric-positive {{
            color: #27ae60;
            font-weight: bold;
        }}
        .metric-negative {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        .summary-box {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Model Performance Report: {asset_name}</h1>
    <div class="info-box">
        <strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>Initial Capital:</strong> ${config.INITIAL_CASH:,.0f}<br>
        <strong>Models Evaluated:</strong> {len(test_results)}<br>
        <strong>Transaction Cost:</strong> {config.TRANSACTION_COST*100:.2f}%
    </div>
    
    <h2>Overall Performance Summary</h2>
    {self._generate_summary_section(test_results)}
    
    <h2>Detailed Metrics Comparison</h2>
    {metrics_df.to_html(index=False, classes='dataframe', escape=False)}
    
    <h2>Risk-Adjusted Performance</h2>
    {self._generate_risk_adjusted_section(test_results)}
    
    <h2>Trade Statistics</h2>
    {self._generate_trade_statistics_section(test_results)}
    
    <h2>Feature Group Analysis</h2>
    {self._generate_feature_group_section(test_results)}
    
    <h2>Model Category Comparison</h2>
    {self._generate_category_comparison(test_results)}
    
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive report saved to: {report_path}")
        return report_path
    
    def _create_metrics_dataframe(self, test_results):
        """Create formatted metrics dataframe"""
        data = []
        
        for result in test_results:
            metrics = result['metrics']
            row = {
                'Model': result['model_name'],
                'Feature Group': result['feature_group'],
                'Features': result['num_features'],
                'Final Value': f"${result['final_value']:,.0f}",
                'Return (%)': f"{metrics['total_return']*100:+.2f}",
                'B&H Return (%)': f"{metrics['benchmark_total_return']*100:+.2f}",
                'Excess (%)': self._format_metric(metrics['excess_return']*100, is_percentage=True),
                'Sharpe': f"{metrics['sharpe_ratio']:.3f}",
                'Sortino': f"{metrics['sortino_ratio']:.3f}",
                'Max DD (%)': f"{metrics['max_drawdown']*100:.2f}",
                'Trades': result['trades']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by Sharpe ratio (extract numeric value)
        df['_sharpe_numeric'] = df['Sharpe'].astype(float)
        df = df.sort_values('_sharpe_numeric', ascending=False)
        df = df.drop('_sharpe_numeric', axis=1)
        
        # Highlight best model
        df.iloc[0] = df.iloc[0].apply(lambda x: f'<span class="best-model">{x}</span>')
        
        return df
    
    def _format_metric(self, value, is_percentage=False):
        """Format metric with color coding"""
        if is_percentage:
            formatted = f"{value:+.2f}%"
        else:
            formatted = f"{value:+.2f}"
        
        if value > 0:
            return f'<span class="metric-positive">{formatted}</span>'
        elif value < 0:
            return f'<span class="metric-negative">{formatted}</span>'
        return formatted
    
    def _generate_summary_section(self, test_results):
        """Generate summary statistics section"""
        sorted_results = sorted(test_results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        best = sorted_results[0]
        worst = sorted_results[-1]
        
        avg_return = np.mean([r['metrics']['total_return'] for r in test_results]) * 100
        avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in test_results])
        
        return f"""
        <div class="summary-box">
            <h3>Best Performing Model</h3>
            <strong>{best['model_name']}</strong> ({best['feature_group']})<br>
            Return: {best['metrics']['total_return']*100:+.2f}% | Sharpe: {best['metrics']['sharpe_ratio']:.3f}<br>
            Final Value: ${best['final_value']:,.0f}
            
            <h3 style="margin-top: 15px;">Performance Statistics</h3>
            Average Return: {avg_return:+.2f}%<br>
            Average Sharpe Ratio: {avg_sharpe:.3f}<br>
            Best Return: {max(r['metrics']['total_return'] for r in test_results)*100:+.2f}%<br>
            Worst Return: {min(r['metrics']['total_return'] for r in test_results)*100:+.2f}%
        </div>
        """
    
    def _generate_risk_adjusted_section(self, test_results):
        """Generate risk-adjusted performance metrics"""
        data = []
        
        for result in test_results:
            metrics = result['metrics']
            
            # Calculate additional risk metrics
            return_to_dd_ratio = abs(metrics['total_return'] / metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
            
            data.append({
                'Model': result['model_name'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Sortino Ratio': metrics['sortino_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                'Return/DD Ratio': return_to_dd_ratio,
                'Excess Return (%)': metrics['excess_return'] * 100
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Sharpe Ratio', ascending=False)
        
        return df.to_html(index=False, float_format='%.3f', classes='dataframe')
    
    def _generate_trade_statistics_section(self, test_results):
            """Generate trade statistics section"""
            data = []

            for result in test_results:
                env = result['env']
                trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()

                if not trades_df.empty:
                    exit_trades = trades_df[trades_df['type'].isin(['SELL', 'COVER', 'STOP-LOSS', 'LIQUIDATE',
                                                                    'LIQUIDATE_LONG', 'LIQUIDATE_SHORT'])]

                    if not exit_trades.empty:
                        winning_trades = len(exit_trades[exit_trades['profit'] > 0])
                        total_trades = len(exit_trades)
                        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

                        avg_profit = exit_trades[exit_trades['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
                        avg_loss = exit_trades[exit_trades['profit'] <= 0]['profit'].mean() if total_trades > winning_trades else 0
                        
                        # FIX 1: Add check for sum of losses being zero before division
                        sum_of_losses = exit_trades[exit_trades['profit'] <= 0]['profit'].sum()
                        if sum_of_losses != 0:
                            profit_factor = abs(exit_trades[exit_trades['profit'] > 0]['profit'].sum() / sum_of_losses)
                        else:
                            profit_factor = float('inf')
                    else:
                        total_trades = 0
                        win_rate = 0
                        avg_profit = 0
                        avg_loss = 0
                        profit_factor = 0
                else:
                    total_trades = 0
                    win_rate = 0
                    avg_profit = 0
                    avg_loss = 0
                    profit_factor = 0

                data.append({
                    'Model': result['model_name'],
                    'Total Trades': total_trades,
                    'Win Rate (%)': win_rate,
                    'Avg Win ($)': avg_profit,
                    'Avg Loss ($)': avg_loss,
                    # FIX 2: Use np.nan instead of a string for infinite profit factor
                    'Profit Factor': profit_factor if profit_factor != float('inf') else np.nan
                })

            df = pd.DataFrame(data)
            df = df.sort_values('Win Rate (%)', ascending=False)

            # FIX 3: Add na_rep to handle np.nan values gracefully in the HTML output
            return df.to_html(index=False, float_format='%.2f', classes='dataframe', na_rep='N/A')

    def _generate_feature_group_section(self, test_results):
        """Analyze performance by feature group"""
        group_performance = {}
        
        for result in test_results:
            group = result['feature_group']
            if group not in group_performance:
                group_performance[group] = []
            
            group_performance[group].append({
                'model': result['model_name'],
                'sharpe': result['metrics']['sharpe_ratio'],
                'return': result['metrics']['total_return'] * 100
            })
        
        html = "<table><tr><th>Feature Group</th><th>Models Using</th><th>Avg Sharpe</th><th>Avg Return (%)</th><th>Best Model</th></tr>"
        
        for group, models in group_performance.items():
            avg_sharpe = np.mean([m['sharpe'] for m in models])
            avg_return = np.mean([m['return'] for m in models])
            best_model = max(models, key=lambda x: x['sharpe'])
            
            html += f"""<tr>
                <td>{group}</td>
                <td>{len(models)}</td>
                <td>{avg_sharpe:.3f}</td>
                <td>{avg_return:+.2f}%</td>
                <td>{best_model['model']} ({best_model['sharpe']:.3f})</td>
            </tr>"""
        
        html += "</table>"
        return html
    
    def _generate_category_comparison(self, test_results):
        """Compare performance across model categories"""
        categories = {
            'Simple': ['simple_dqn', 'simple_dropout', 'simple_residual'],
            'Original': ['dueling', 'lstm', 'attention'],
            'Complex': ['deep_dueling', 'transformer', 'hybrid_cnn_lstm']
        }
        
        html = "<table><tr><th>Category</th><th>Models</th><th>Avg Sharpe</th><th>Avg Return (%)</th><th>Best Model</th></tr>"
        
        for category, model_names in categories.items():
            category_results = [r for r in test_results if r['model_name'] in model_names]
            
            if category_results:
                avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in category_results])
                avg_return = np.mean([r['metrics']['total_return'] for r in category_results]) * 100
                best = max(category_results, key=lambda x: x['metrics']['sharpe_ratio'])
                
                html += f"""<tr>
                    <td><strong>{category}</strong></td>
                    <td>{len(category_results)}</td>
                    <td>{avg_sharpe:.3f}</td>
                    <td>{avg_return:+.2f}%</td>
                    <td>{best['model_name']} ({best['metrics']['sharpe_ratio']:.3f})</td>
                </tr>"""
        
        html += "</table>"
        return html
    
    def create_comparison_charts(self, test_results, asset_name):
        """Create visual comparison charts"""
        charts_dir = os.path.join(self.save_dir, 'comparison_charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Chart 1: Sharpe Ratio Comparison
        self._create_sharpe_comparison_chart(test_results, asset_name, charts_dir)
        
        # Chart 2: Return vs Risk Scatter
        self._create_return_risk_scatter(test_results, asset_name, charts_dir)
        
        # Chart 3: Feature Group Performance
        self._create_feature_group_chart(test_results, asset_name, charts_dir)
        
        # Chart 4: Model Category Performance
        self._create_category_performance_chart(test_results, asset_name, charts_dir)
        
        print(f"Comparison charts saved to: {charts_dir}")
    
    def _create_sharpe_comparison_chart(self, test_results, asset_name, charts_dir):
        """Create bar chart comparing Sharpe ratios"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        models = [r['model_name'] for r in test_results]
        sharpes = [r['metrics']['sharpe_ratio'] for r in test_results]
        
        # Sort by Sharpe ratio
        sorted_indices = np.argsort(sharpes)[::-1]
        models = [models[i] for i in sorted_indices]
        sharpes = [sharpes[i] for i in sorted_indices]
        
        colors = ['green' if s > 0 else 'red' for s in sharpes]
        
        bars = ax.barh(models, sharpes, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'{asset_name} - Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, sharpe) in enumerate(zip(bars, sharpes)):
            ax.text(sharpe, i, f' {sharpe:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f'{asset_name}_sharpe_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_return_risk_scatter(self, test_results, asset_name, charts_dir):
        """Create scatter plot of return vs risk"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        returns = [r['metrics']['total_return'] * 100 for r in test_results]
        drawdowns = [r['metrics']['max_drawdown'] * 100 for r in test_results]
        models = [r['model_name'] for r in test_results]
        sharpes = [r['metrics']['sharpe_ratio'] for r in test_results]
        
        scatter = ax.scatter(drawdowns, returns, s=200, c=sharpes, 
                           cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Add labels for each point
        for i, model in enumerate(models):
            ax.annotate(model, (drawdowns[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{asset_name} - Return vs Risk (colored by Sharpe Ratio)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f'{asset_name}_return_risk_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_group_chart(self, test_results, asset_name, charts_dir):
        """Create bar chart showing performance by feature group"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        group_performance = {}
        for result in test_results:
            group = result['feature_group']
            if group not in group_performance:
                group_performance[group] = []
            group_performance[group].append(result['metrics']['sharpe_ratio'])
        
        groups = list(group_performance.keys())
        avg_sharpes = [np.mean(group_performance[g]) for g in groups]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
        bars = ax.bar(groups, avg_sharpes, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Average Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'{asset_name} - Performance by Feature Group', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, sharpe in zip(bars, avg_sharpes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{sharpe:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f'{asset_name}_feature_group_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_category_performance_chart(self, test_results, asset_name, charts_dir):
        """Create grouped bar chart comparing model categories"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        categories = {
            'Simple': ['simple_dqn', 'simple_dropout', 'simple_residual'],
            'Original': ['dueling', 'lstm', 'attention'],
            'Complex': ['deep_dueling', 'transformer', 'hybrid_cnn_lstm']
        }
        
        category_data = {}
        for cat_name, model_names in categories.items():
            cat_results = [r for r in test_results if r['model_name'] in model_names]
            if cat_results:
                category_data[cat_name] = {
                    'avg_sharpe': np.mean([r['metrics']['sharpe_ratio'] for r in cat_results]),
                    'avg_return': np.mean([r['metrics']['total_return'] for r in cat_results]) * 100,
                    'avg_drawdown': np.mean([r['metrics']['max_drawdown'] for r in cat_results]) * 100
                }
        
        x = np.arange(len(category_data))
        width = 0.25
        
        sharpes = [category_data[c]['avg_sharpe'] for c in category_data.keys()]
        returns = [category_data[c]['avg_return'] for c in category_data.keys()]
        drawdowns = [category_data[c]['avg_drawdown'] for c in category_data.keys()]
        
        ax.bar(x - width, sharpes, width, label='Avg Sharpe', color='steelblue', alpha=0.8)
        ax.bar(x, returns, width, label='Avg Return (%)', color='green', alpha=0.8)
        ax.bar(x + width, drawdowns, width, label='Avg Drawdown (%)', color='red', alpha=0.8)
        
        ax.set_xlabel('Model Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{asset_name} - Performance by Model Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(category_data.keys())
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f'{asset_name}_category_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()