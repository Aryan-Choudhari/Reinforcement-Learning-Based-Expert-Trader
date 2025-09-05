import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from datetime import datetime

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
    
    def plot_price_action(self, env, dates, asset_name, save_dir):
        """Plot price action with trade signals - CORRECTED LEGEND"""
        fig, ax = plt.subplots(figsize=(15, 8))
        prices = env.data['close'].values
        
        ax.plot(dates, prices, label='Close Price', color='blue', linewidth=1.5, alpha=0.8)
        
        trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()
        
        if not trades_df.empty:
            # Track which signals actually appear and create labeled scatter plots
            
            # Buy Long signals
            buy_trades = trades_df[trades_df['type'] == 'BUY']
            if not buy_trades.empty:
                visible_buys = []
                for _, trade in buy_trades.iterrows():
                    if trade['step'] < len(dates):
                        visible_buys.append((dates[trade['step']], trade['price']))
                
                if visible_buys:
                    buy_dates, buy_prices = zip(*visible_buys)
                    ax.scatter(buy_dates, buy_prices, color='green', marker='^', 
                            s=120, alpha=0.8, zorder=5, label='Buy Long')
            
            # Sell Short signals  
            short_trades = trades_df[trades_df['type'] == 'SELL_SHORT']
            if not short_trades.empty:
                visible_shorts = []
                for _, trade in short_trades.iterrows():
                    if trade['step'] < len(dates):
                        visible_shorts.append((dates[trade['step']], trade['price']))
                
                if visible_shorts:
                    short_dates, short_prices = zip(*visible_shorts)
                    ax.scatter(short_dates, short_prices, color='red', marker='v',
                            s=120, alpha=0.8, zorder=5, label='Sell Short')
            
            # Sell Long trades (separate by profit/loss)
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            if not sell_trades.empty:
                profit_sells = []
                loss_sells = []
                
                for _, trade in sell_trades.iterrows():
                    if trade['step'] < len(dates):
                        if trade.get('profit', 0) > 0:
                            profit_sells.append((dates[trade['step']], trade['price']))
                        else:
                            loss_sells.append((dates[trade['step']], trade['price']))
                
                if profit_sells:
                    profit_dates, profit_prices = zip(*profit_sells)
                    ax.scatter(profit_dates, profit_prices, color='orange', marker='s',
                            s=100, alpha=0.8, zorder=5, label='Sell Long (Profit)')
                
                if loss_sells:
                    loss_dates, loss_prices = zip(*loss_sells)
                    ax.scatter(loss_dates, loss_prices, color='brown', marker='s',
                            s=100, alpha=0.8, zorder=5, label='Sell Long (Loss)')
            
            # Cover Short trades
            cover_trades = trades_df[trades_df['type'] == 'COVER']
            if not cover_trades.empty:
                profit_covers = []
                loss_covers = []
                
                for _, trade in cover_trades.iterrows():
                    if trade['step'] < len(dates):
                        if trade.get('profit', 0) > 0:
                            profit_covers.append((dates[trade['step']], trade['price']))
                        else:
                            loss_covers.append((dates[trade['step']], trade['price']))
                
                if profit_covers:
                    profit_dates, profit_prices = zip(*profit_covers)
                    ax.scatter(profit_dates, profit_prices, color='lime', marker='d',
                            s=100, alpha=0.8, zorder=5, label='Cover Short (Profit)')
                
                if loss_covers:
                    loss_dates, loss_prices = zip(*loss_covers)
                    ax.scatter(loss_dates, loss_prices, color='maroon', marker='d',
                            s=100, alpha=0.8, zorder=5, label='Cover Short (Loss)')
            
            # Stop Loss trades
            stop_trades = trades_df[trades_df['type'] == 'STOP-LOSS']
            if not stop_trades.empty:
                visible_stops = []
                for _, trade in stop_trades.iterrows():
                    if trade['step'] < len(dates):
                        visible_stops.append((dates[trade['step']], trade['price']))
                
                if visible_stops:
                    stop_dates, stop_prices = zip(*visible_stops)
                    ax.scatter(stop_dates, stop_prices, color='black', marker='x',
                            s=150, alpha=0.9, zorder=6, label='Stop Loss')
        
        ax.set_title(f'{asset_name} - Price Action with Trading Signals', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        
        # Let matplotlib automatically create legend from labeled elements
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        if isinstance(dates[0], (pd.Timestamp, datetime)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f'{asset_name}_price_action.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return path

    def plot_portfolio_performance(self, portfolio_dates, portfolio_values, prices, asset_name, save_dir):
        """Plot portfolio performance vs benchmark"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        benchmark_values = prices / prices[0] * portfolio_values[0]
        benchmark_portfolio = benchmark_values[:len(portfolio_values)]
        
        ax.plot(portfolio_dates, portfolio_values, 
               label='Agent Portfolio', color='green', linewidth=2.5, alpha=0.9)
        ax.plot(portfolio_dates, benchmark_portfolio,
               label='Buy & Hold', color='blue', alpha=0.7, linewidth=2)
        
        # Calculate performance metrics for title
        final_agent_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
        final_bnh_return = (benchmark_portfolio[-1] - benchmark_portfolio[0]) / benchmark_portfolio[0] * 100
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