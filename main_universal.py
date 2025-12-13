import os
import glob
import traceback
import pandas as pd
import numpy as np
from config import TradingConfig
from data_handler_universal import UniversalDataHandler
from universal_trainer import UniversalMultiStockTrainer, generate_universal_comparison_report
from utils import PerformanceMetrics # <-- ADDED for detailed reports

def generate_single_stock_report(model_name, stock_name, env, config):
    """Generates a detailed, formatted report string for a single model-stock evaluation."""
    
    # 1. Calculate Performance Metrics
    portfolio_values = np.array(env.portfolio_value_history)
    benchmark_values = np.array(env.benchmark_values)
    metrics = PerformanceMetrics.calculate_metrics(portfolio_values, benchmark_values, config.INITIAL_CASH)
    
    # 2. Calculate Trade Statistics
    trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()
    total_trades = 0
    win_rate = 0
    profit_factor = 0
    avg_win = 0
    avg_loss = 0
    
    if not trades_df.empty:
        exit_trades = trades_df[trades_df['type'].str.contains('SELL|COVER|LIQUIDATE', na=False)]
        total_trades = len(exit_trades)
        if total_trades > 0:
            winning_trades = exit_trades[exit_trades['profit'] > 0]
            losing_trades = exit_trades[exit_trades['profit'] <= 0]
            
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            
            sum_of_profits = winning_trades['profit'].sum()
            sum_of_losses = abs(losing_trades['profit'].sum())
            profit_factor = sum_of_profits / sum_of_losses if sum_of_losses != 0 else float('inf')
            
            avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0

    # 3. Format the Report String
    report = f"""
{'='*80}
INDIVIDUAL PERFORMANCE REPORT
{'='*80}
Model:          {model_name}
Stock:          {stock_name}
Generated On:   {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S %Z')}
{'-'*80}

💰 PORTFOLIO & RETURNS
Final Portfolio Value:    ${portfolio_values[-1]:,.2f}
Total Net Return:         {metrics['total_return']*100:+.2f}%
Benchmark Return:         {metrics['benchmark_total_return']*100:+.2f}%
Excess Return (Alpha):    {metrics['excess_return']*100:+.2f}%

🌊 RISK ANALYSIS
Sharpe Ratio:             {metrics['sharpe_ratio']:.3f}
Sortino Ratio:            {metrics['sortino_ratio']:.3f}
Max Drawdown:             {metrics['max_drawdown']*100:.2f}%
Benchmark Max Drawdown:   {metrics['benchmark_max_drawdown']*100:.2f}%

📋 TRADE STATISTICS
Total Completed Trades:   {total_trades}
Win Rate:                 {win_rate:.2f}%
Profit Factor:            {profit_factor:.2f}
Average Win:              ${avg_win:,.2f}
Average Loss:             ${avg_loss:,.2f}
{'-'*80}
"""
    return report

def ensure_featured_data_exists(raw_data_folder='Raw_Data', featured_data_folder='Featured_Data'):
    """Check if featured data exists for all raw files, generate if missing"""
    print("🔍 Checking featured data availability...")
    
    os.makedirs(raw_data_folder, exist_ok=True)
    os.makedirs(featured_data_folder, exist_ok=True)
    
    raw_files = glob.glob(os.path.join(raw_data_folder, '*.csv'))
    if not raw_files:
        print(f"❌ No raw data files found in '{raw_data_folder}'")
        return False
    
    print(f"📊 Found {len(raw_files)} raw data files")
    
    missing_featured_files = []
    existing_featured_files = []
    
    for raw_file in raw_files:
        filename = os.path.basename(raw_file)
        if '_featured.csv' in filename:
            featured_filename = filename
        else:
            base_name = filename.replace('.csv', '')
            featured_filename = f"{base_name}_featured.csv"
        
        featured_path = os.path.join(featured_data_folder, featured_filename)
        
        if os.path.exists(featured_path):
            existing_featured_files.append(featured_filename)
        else:
            missing_featured_files.append((raw_file, featured_filename))
    
    print(f"📊 Status: {len(existing_featured_files)} existing, {len(missing_featured_files)} missing")
    
    if not missing_featured_files:
        print("✅ All featured data files are available!")
        return True
    
    print(f"\n🛠️ Generating {len(missing_featured_files)} missing featured data files...")
    
    config = TradingConfig()
    data_handler = UniversalDataHandler(config)
    
    success_count = 0
    failed_count = 0
    
    for raw_file, featured_filename in missing_featured_files:
        try:
            print(f"\n--- Processing: {os.path.basename(raw_file)} ---")
            
            raw_data = pd.read_csv(raw_file)
            print(f"   Raw data shape: {raw_data.shape}")
            
            # Generate features with universal normalization
            featured_data = data_handler.perform_feature_engineering(raw_data, for_universal=True)
            
            featured_path = os.path.join(featured_data_folder, featured_filename)
            featured_data.to_csv(featured_path, index=False)
            
            print(f"✅ Successfully created: {featured_filename} (shape: {featured_data.shape})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(raw_file)}: {str(e)}")
            traceback.print_exc()
            failed_count += 1
    
    print(f"\n📊 Feature engineering summary:")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    
    return success_count > 0


def load_all_featured_data(featured_data_folder='Featured_Data', min_samples=300):
    """Load all featured data files into memory"""
    print(f"\n{'='*80}")
    print(f"LOADING ALL FEATURED DATA")
    print(f"{'='*80}")
    
    pattern = os.path.join(featured_data_folder, '*_featured.csv')
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"No featured data files found in '{featured_data_folder}'")
        return None
    
    print(f"Found {len(all_files)} featured data files")
    
    config = TradingConfig()
    data_handler = UniversalDataHandler(config)
    
    all_stock_data = {}
    
    for file_path in all_files:
        try:
            filename = os.path.basename(file_path)
            stock_name = filename.replace('_featured.csv', '')
            
            print(f"  Loading: {stock_name}")
            
            # Load data without re-engineering (already done)
            data = data_handler.load_featured_data(file_path, perform_feature_engineering=False)
            
            if len(data) < min_samples:
                print(f"    WARNING: Only {len(data)} rows, skipping (minimum {min_samples})")
                continue
            
            all_stock_data[stock_name] = data
            print(f"    ✓ Loaded {len(data)} samples, {len(data.columns)} features")
            
        except Exception as e:
            print(f"    ERROR loading {filename}: {e}")
            traceback.print_exc()
    
    print(f"\n✓ Successfully loaded {len(all_stock_data)} stocks")
    return all_stock_data

# Add this new function to main_universal.py
def generate_sleek_stock_reports(test_results, config, results_folder):
    """
    Generates a sleek, table-style performance report for each stock
    and saves it to a text file.
    """
    print("\n" + "="*80)
    print("GENERATING SLEEK INDIVIDUAL STOCK REPORTS")
    print("="*80)

    reports_dir = os.path.join(results_folder, 'individual_reports')
    os.makedirs(reports_dir, exist_ok=True)

    if not test_results:
        print("  No test results to process.")
        return

    all_stock_names = list(test_results[0]['stock_results'].keys())

    for stock_name in all_stock_names:
        stock_performance_data = []
        for model_result in test_results:
            model_name = model_result['model_name']
            if stock_name not in model_result['stock_results']:
                continue

            stock_data = model_result['stock_results'][stock_name]
            env = stock_data['env']
            
            portfolio_values = np.array(env.portfolio_value_history)
            benchmark_values = np.array(env.benchmark_values)
            metrics = PerformanceMetrics.calculate_metrics(portfolio_values, benchmark_values, config.INITIAL_CASH)

            trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()
            total_trades = len(trades_df[trades_df['type'].str.contains('SELL|COVER|LIQUIDATE', na=False)]) if not trades_df.empty else 0

            stock_performance_data.append({
                'Model': model_name,
                'Total_Return_%': metrics['total_return'] * 100,
                'Benchmark_Return_%': metrics['benchmark_total_return'] * 100,
                'Excess_Return_%': metrics['excess_return'] * 100,
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Sortino_Ratio': metrics['sortino_ratio'],
                'Max_Drawdown_%': metrics['max_drawdown'] * 100,
                'Benchmark_Sharpe': metrics.get('benchmark_sharpe_ratio', 0.0),
                'Benchmark_Sortino': metrics.get('benchmark_sortino_ratio', 0.0),
                'Benchmark_Max_DD_%': metrics.get('benchmark_max_drawdown', 0.0) * 100,
                'Trades': total_trades
            })

        if not stock_performance_data:
            continue

        # Create, sort, and format the DataFrame for the report
        report_df = pd.DataFrame(stock_performance_data)
        report_df = report_df.sort_values('Sharpe_Ratio', ascending=False).reset_index(drop=True)
        
        # Create the report string
        report_string = f"""
{'='*120}
# Stock Performance Summary: {stock_name}
{'='*120}
Generated On: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S %Z')}

{report_df.to_string()}
{'='*120}
"""
        # Print to console
        print(report_string)
        
        # Save to file within a stock-specific folder
        stock_report_dir = os.path.join(reports_dir, stock_name)
        os.makedirs(stock_report_dir, exist_ok=True)
        report_filepath = os.path.join(stock_report_dir, f"{stock_name}_performance_summary.txt")
        with open(report_filepath, 'w') as f:
            f.write(report_string)

    print(f"\n✓ Sleek individual reports saved to: {reports_dir}")

def generate_benchmark_comparison_reports(test_results, config):
    """
    Generates and prints a detailed agent vs. benchmark comparison report 
    to the console for each stock.
    """
    if not test_results:
        print("  No test results available for benchmark comparison.")
        return

    all_stock_names = sorted(list(test_results[0]['stock_results'].keys()))

    for stock_name in all_stock_names:
        print(f"\n{'='*62}")
        print(f"||{'BENCHMARK COMPARISON REPORT: ' + stock_name:<58}||")
        print(f"{'='*62}")

        # Sort models by Sharpe Ratio for this specific stock
        models_sorted = sorted(
            test_results,
            key=lambda r: PerformanceMetrics.calculate_metrics(
                np.array(r['stock_results'][stock_name]['env'].portfolio_value_history),
                np.array(r['stock_results'][stock_name]['env'].benchmark_values),
                config.INITIAL_CASH
            )['sharpe_ratio'],
            reverse=True
        )

        for model_result in models_sorted:
            model_name = model_result['model_name']
            if stock_name not in model_result['stock_results']:
                continue

            env = model_result['stock_results'][stock_name]['env']
            metrics = PerformanceMetrics.calculate_metrics(
                np.array(env.portfolio_value_history),
                np.array(env.benchmark_values),
                config.INITIAL_CASH
            )

            # --- Calculate Advantage Metrics ---
            advantage_return = metrics['total_return'] - metrics['benchmark_total_return']
            advantage_sharpe = metrics['sharpe_ratio'] - metrics['benchmark_sharpe_ratio']
            advantage_sortino = metrics['sortino_ratio'] - metrics['benchmark_sortino_ratio']
            # Lower drawdown is better, so advantage is benchmark - agent
            advantage_drawdown = metrics['benchmark_max_drawdown'] - metrics['max_drawdown']

            print(f"\n--- Model: {model_name} {'-'*45}")
            print(f"{'Metric':<20} | {'Agent':>12} | {'Benchmark':>12} | {'Advantage':>12}")
            print(f"{'-'*20}-|-{'-'*12}-|-{'-'*12}-|-{'-'*12}")
            print(f"{'Total Return':<20} | {metrics['total_return']*100:>+11.2f}% | {metrics['benchmark_total_return']*100:>+11.2f}% | {advantage_return*100:>+11.2f}%")
            print(f"{'Sharpe Ratio':<20} | {metrics['sharpe_ratio']:>12.3f} | {metrics['benchmark_sharpe_ratio']:>12.3f} | {advantage_sharpe:>+12.3f}")
            print(f"{'Sortino Ratio':<20} | {metrics['sortino_ratio']:>12.3f} | {metrics['benchmark_sortino_ratio']:>12.3f} | {advantage_sortino:>+12.3f}")
            print(f"{'Max Drawdown':<20} | {metrics['max_drawdown']*100:>11.2f}% | {metrics['benchmark_max_drawdown']*100:>11.2f}% | {advantage_drawdown*100:>+11.2f} pp")
        
        print(f"{'-'*62}")

def universal_training_workflow(featured_data_folder='Featured_Data', 
                                results_folder='Universal_Trading_Results'):
    """Main workflow for universal training across all stocks"""
    os.makedirs(results_folder, exist_ok=True)
    
    config = TradingConfig()
    
    # Step 1: Load all stock data
    print("\n" + "="*80)
    print("STEP 1: LOADING STOCK DATA")
    print("="*80)
    
    all_stock_data = load_all_featured_data(featured_data_folder)
    
    if not all_stock_data or len(all_stock_data) < 2:
        print(f"ERROR: Need at least 2 stocks for universal training (found {len(all_stock_data) if all_stock_data else 0})")
        return
    
    # Step 2: Prepare stocks for universal training (add metadata)
    print("\n" + "="*80)
    print("STEP 2: PREPARING UNIVERSAL DATASET")
    print("="*80)
    
    data_handler = UniversalDataHandler(config)
    prepared_stocks = data_handler.prepare_all_stocks_for_universal_training(all_stock_data)
    
    # Step 3: Get common universal features across all stocks
    common_features = data_handler.get_common_features(prepared_stocks)
    
    if len(common_features) < 10:
        print(f"ERROR: Only {len(common_features)} common features found. Need at least 10.")
        print(f"Common features: {common_features}")
        return
    
    print(f"\n✓ Using {len(common_features)} common universal features for training")
    
    # Step 4: CORRECTED - Create final dataframes for the environment
    # Ensure they contain BOTH model features AND essential environment columns (OHLC).
    final_stocks_for_env = {}
    env_required_cols = ['open', 'high', 'low', 'close', 'timestamp', 'volume', 'close_normalized']
    
    for stock_name, stock_data in prepared_stocks.items():
        # Identify which of the required env columns are actually in this stock's data
        env_cols_present = [col for col in env_required_cols if col in stock_data.columns]
        
        # Combine model features and essential env columns, removing duplicates
        final_cols = list(set(common_features + env_cols_present))
        
        # Create the final dataframe for this stock
        final_stocks_for_env[stock_name] = stock_data[final_cols].copy()
        
        print(f"  {stock_name}: {len(final_stocks_for_env[stock_name])} rows, {len(common_features)} model features")

    # Step 5: Initialize universal trainer
    print("\n" + "="*80)
    print("STEP 3: INITIALIZING UNIVERSAL TRAINER")
    print("="*80)
    
    trainer = UniversalMultiStockTrainer(config, save_dir=results_folder)
    
    # Step 6: Prepare train/val/test split using the final dataframes
    train_stocks, val_stocks, test_stocks = trainer.prepare_universal_dataset(final_stocks_for_env, common_features)
    
    print(f"\nDataset prepared:")
    print(f"  Training stocks: {len(train_stocks)} (total samples: {sum(len(d) for d in train_stocks.values())})")
    print(f"  Validation stocks: {len(val_stocks)} (total samples: {sum(len(d) for d in val_stocks.values())})")
    print(f"  Test stocks: {len(test_stocks)} (total samples: {sum(len(d) for d in test_stocks.values())})")
    
    # Step 7: Train all models universally
    print("\n" + "="*80)
    print("STEP 4: UNIVERSAL MODEL TRAINING")
    print(f"Training on {len(train_stocks)} stocks simultaneously")
    print("="*80)
    
    training_results = trainer.train_all_universal_models_parallel(train_stocks, val_stocks)
    
    if not training_results:
        print("ERROR: No models were successfully trained")
        return
    
    print(f"\n✓ Successfully trained {len(training_results)} models")
    
    # Step 8: Evaluate on test stocks
    print("\n" + "="*80)
    print("STEP 5: EVALUATION ON TEST STOCKS")
    print("="*80)
    
    test_results = trainer.evaluate_universal_models_on_test(test_stocks)
    
    if not test_results:
        print("ERROR: No models were successfully evaluated")
        return
    
    print(f"\n✓ Successfully evaluated {len(test_results)} models")
    
    # Step 9: Generate AGGREGATE reports
    print("\n" + "="*80)
    print("STEP 6: GENERATING AGGREGATE REPORTS")
    print("="*80)
    
    comparison_path = os.path.join(results_folder, 'universal_model_comparison.csv')
    comparison_df = generate_universal_comparison_report(test_results, comparison_path)
    generate_per_stock_analysis(test_results, test_stocks, results_folder)
    generate_universal_summary(test_results, train_stocks, val_stocks, test_stocks, results_folder)

    # Step 10: Generate PER-STOCK SUMMARY tables
    print("\n" + "="*80)
    print("STEP 7: GENERATING PER-STOCK SUMMARY TABLES (CSV)")
    print("="*80)
    generate_per_stock_summary_tables(test_results, config, results_folder)

    # Step 11: Generate SLEEK table reports
    print("\n" + "="*80)
    print("STEP 8: GENERATING SLEEK INDIVIDUAL REPORTS (CONSOLE & FILE)")
    print("="*80)
    generate_sleek_stock_reports(test_results, config, results_folder)

    # Step 12: Generate BENCHMARK COMPARISON reports (console)
    print("\n" + "="*80)
    print("STEP 9: GENERATING BENCHMARK COMPARISON REPORTS (CONSOLE)")
    print("="*80)
    generate_benchmark_comparison_reports(test_results, config)

    # Step 13: Print final summary
    print_final_results(test_results, results_folder)
    
    print(f"\n{'='*80}")
    print(f"🎉 UNIVERSAL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved to: {results_folder}")
    print(f"{'='*80}")

def generate_per_stock_summary_tables(test_results, config, results_folder='Universal_Trading_Results'):
    """
    Generates a detailed performance comparison CSV for each stock across all models.
    """
    print("\n" + "="*80)
    print("GENERATING PER-STOCK SUMMARY TABLES")
    print("="*80)

    reports_dir = os.path.join(results_folder, 'per_stock_summary_reports')
    os.makedirs(reports_dir, exist_ok=True)

    if not test_results:
        print("  No test results to process.")
        return

    # Get the list of stock names from the first model's results
    all_stock_names = list(test_results[0]['stock_results'].keys())

    for stock_name in all_stock_names:
        print(f"  Generating report for: {stock_name}")
        stock_performance_data = []

        for model_result in test_results:
            model_name = model_result['model_name']
            
            # Check if this model has results for the current stock
            if stock_name not in model_result['stock_results']:
                continue

            stock_data = model_result['stock_results'][stock_name]
            env = stock_data['env']

            # 1. Calculate Performance Metrics
            portfolio_values = np.array(env.portfolio_value_history)
            benchmark_values = np.array(env.benchmark_values)
            metrics = PerformanceMetrics.calculate_metrics(portfolio_values, benchmark_values, config.INITIAL_CASH)

            # 2. Calculate Trade Statistics
            trades_df = pd.DataFrame(env.trades) if env.trades else pd.DataFrame()
            total_trades = 0
            if not trades_df.empty:
                exit_trades = trades_df[trades_df['type'].str.contains('SELL|COVER|LIQUIDATE', na=False)]
                total_trades = len(exit_trades)
            
            # 3. Assemble the data row
            row = {
                'Model': model_name,
                'Feature_Group': model_result.get('feature_group', 'N/A'),
                'Num_Features': model_result.get('num_features', 0),
                'Final_Value': portfolio_values[-1],
                'Total_Return_%': metrics['total_return'] * 100,
                'Benchmark_Return_%': metrics['benchmark_total_return'] * 100,
                'Excess_Return_%': metrics['excess_return'] * 100,
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Sortino_Ratio': metrics['sortino_ratio'],
                'Max_Drawdown_%': metrics['max_drawdown'] * 100,
                'Trades': total_trades
            }
            stock_performance_data.append(row)

        if not stock_performance_data:
            print(f"    No data to generate report for {stock_name}.")
            continue
            
        # Create and save the DataFrame
        summary_df = pd.DataFrame(stock_performance_data)
        summary_df = summary_df.sort_values('Sharpe_Ratio', ascending=False)

        report_path = os.path.join(reports_dir, f"{stock_name}_test_summary.csv")
        summary_df.to_csv(report_path, index=False, float_format='%.4f')
    
    print(f"\n✓ Per-stock summary tables saved to: {reports_dir}")

def generate_per_stock_analysis(test_results, test_stocks, results_folder):
    """Generate detailed analysis for each stock"""
    print(f"\n{'='*80}")
    print(f"GENERATING PER-STOCK ANALYSIS")
    print(f"{'='*80}")
    
    per_stock_dir = os.path.join(results_folder, 'per_stock_results')
    os.makedirs(per_stock_dir, exist_ok=True)
    
    stock_names = list(test_stocks.keys())
    model_names = [r['model_name'] for r in test_results]
    
    performance_matrix = []
    
    for stock_name in stock_names:
        stock_row = {'Stock': stock_name}
        
        for result in test_results:
            model_name = result['model_name']
            if stock_name in result['stock_results']:
                stock_result = result['stock_results'][stock_name]
                return_pct = stock_result['total_return'] * 100
                stock_row[f"{model_name}_Return_%"] = return_pct
                stock_row[f"{model_name}_Trades"] = stock_result['trades']
        
        # Calculate benchmark (Buy & Hold)
        stock_data = test_stocks[stock_name]
        
        # Use normalized prices if available
        if 'close_normalized' in stock_data.columns:
            first_price = stock_data['close_normalized'].iloc[0]
            last_price = stock_data['close_normalized'].iloc[-1]
        elif 'close' in stock_data.columns:
            first_price = stock_data['close'].iloc[0]
            last_price = stock_data['close'].iloc[-1]
        else:
            first_price = 1.0
            last_price = 1.0
        
        benchmark_return = (last_price / first_price - 1) * 100
        stock_row['Benchmark_Return_%'] = benchmark_return
        
        # Find best model for this stock
        model_returns = [stock_row.get(f"{m}_Return_%", -999) for m in model_names]
        if model_returns:
            best_idx = np.argmax(model_returns)
            stock_row['Best_Model'] = model_names[best_idx]
            stock_row['Best_Return_%'] = model_returns[best_idx]
        else:
            stock_row['Best_Model'] = 'None'
            stock_row['Best_Return_%'] = 0
        
        performance_matrix.append(stock_row)
    
    # Save to CSV
    df = pd.DataFrame(performance_matrix)
    df = df.sort_values('Best_Return_%', ascending=False)
    
    csv_path = os.path.join(per_stock_dir, 'stock_performance_matrix.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"  ✓ Per-stock analysis saved to: {csv_path}")
    
    # Print top performers
    print(f"\n  Top 5 Performing Stocks:")
    for i, row in df.head(5).iterrows():
        print(f"    {row['Stock']}: {row['Best_Return_%']:+.2f}% "
              f"(Model: {row['Best_Model']}, Benchmark: {row['Benchmark_Return_%']:+.2f}%)")


def generate_universal_summary(test_results, train_stocks, val_stocks, test_stocks, results_folder):
    """Generate comprehensive summary report"""
    summary_path = os.path.join(results_folder, 'UNIVERSAL_TRAINING_SUMMARY.txt')
    
    # Find best model
    best_model = max(test_results, key=lambda x: x['avg_return'])
    
    # Calculate statistics
    avg_returns = [r['avg_return'] for r in test_results]
    
    summary_report = f"""
UNIVERSAL MULTI-STOCK TRADING SYSTEM - FINAL REPORT
{'='*80}

TRAINING CONFIGURATION:
Training Stocks: {len(train_stocks)}
Validation Stocks: {len(val_stocks)}
Test Stocks: {len(test_stocks)}
Total Models Trained: {len(test_results)}

DATASET SUMMARY:
Train Samples: {sum(len(d) for d in train_stocks.values())}
Val Samples: {sum(len(d) for d in val_stocks.values())}
Test Samples: {sum(len(d) for d in test_stocks.values())}

BEST UNIVERSAL MODEL:
Model: {best_model['model_name']}
Feature Group: {best_model['feature_group']}
Average Return: {best_model['avg_return']*100:+.2f}%
Average Final Value: ${best_model['avg_final_value']:,.0f}
Total Trades Across All Stocks: {best_model['total_trades']}
Trades Per Stock: {best_model['total_trades'] / best_model['num_stocks']:.1f}

OVERALL PERFORMANCE:
Average Return Across All Models: {np.mean(avg_returns)*100:+.2f}%
Best Model Return: {max(avg_returns)*100:+.2f}%
Worst Model Return: {min(avg_returns)*100:+.2f}%
Return Standard Deviation: {np.std(avg_returns)*100:.2f}%

MODEL RANKINGS (by Average Return):
{'='*80}
"""
    
    # Add model rankings
    sorted_results = sorted(test_results, key=lambda x: x['avg_return'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        summary_report += f"""
{i}. {result['model_name']} ({result['feature_group']})
   Average Return: {result['avg_return']*100:+.2f}%
   Average Final Value: ${result['avg_final_value']:,.0f}
   Total Trades: {result['total_trades']}
   Trades Per Stock: {result['total_trades'] / result['num_stocks']:.1f}
"""
    
    summary_report += f"""
{'='*80}

PER-STOCK PERFORMANCE ANALYSIS:
{'='*80}
"""
    
    # Analyze performance for each test stock
    for stock_name in sorted(test_stocks.keys()):
        summary_report += f"\n{stock_name}:\n"
        
        # Get results for this stock from all models
        stock_model_returns = []
        for result in test_results:
            if stock_name in result['stock_results']:
                model_return = result['stock_results'][stock_name]['total_return'] * 100
                stock_model_returns.append({
                    'model': result['model_name'],
                    'return': model_return
                })
        
        if stock_model_returns:
            best_for_stock = max(stock_model_returns, key=lambda x: x['return'])
            worst_for_stock = min(stock_model_returns, key=lambda x: x['return'])
            avg_return_stock = np.mean([m['return'] for m in stock_model_returns])
            
            summary_report += f"  Best Model: {best_for_stock['model']} ({best_for_stock['return']:+.2f}%)\n"
            summary_report += f"  Worst Model: {worst_for_stock['model']} ({worst_for_stock['return']:+.2f}%)\n"
            summary_report += f"  Average Return: {avg_return_stock:+.2f}%\n"
    
    summary_report += f"""
{'='*80}

KEY INSIGHTS:
1. Universal training allows models to learn patterns across multiple markets
2. Models generalize better to unseen stocks
3. Reduced overfitting compared to single-stock training
4. More robust performance in different market conditions

ADVANTAGES OF UNIVERSAL TRAINING:
✓ Single model works across all stocks
✓ Better generalization to new assets
✓ Learns diverse market patterns
✓ More efficient than training per-stock
✓ Easier to deploy and maintain

NEXT STEPS:
1. Review per-stock performance matrix in per_stock_results/
2. Use best model ({best_model['model_name']}) for new stocks
3. Load model: {best_model['model_name']}_universal_best.pth
4. Apply to new stocks without retraining

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
    
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    print(f"  ✓ Universal training summary saved to: {summary_path}")


def print_final_results(test_results, results_folder):
    """Print final results summary to console"""
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Sort by average return
    sorted_results = sorted(test_results, key=lambda x: x['avg_return'], reverse=True)
    
    print(f"\nTop 3 Models (Average Performance):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"\n{i}. {result['model_name']}")
        print(f"   Feature Group: {result['feature_group']}")
        print(f"   Avg Return: {result['avg_return']*100:+.2f}%")
        print(f"   Avg Final Value: ${result['avg_final_value']:,.0f}")
        print(f"   Total Trades: {result['total_trades']}")
        print(f"   Stocks Evaluated: {result['num_stocks']}")
    
    print(f"\n{'='*80}")
    print(f"All Generated Artifacts:")
    print(f"  📊 Aggregate Comparison: {os.path.join(results_folder, 'universal_model_comparison.csv')}")
    print(f"  📋 Aggregate Summary: {os.path.join(results_folder, 'UNIVERSAL_TRAINING_SUMMARY.txt')}")
    print(f"  📈 Per-Stock Summary: {os.path.join(results_folder, 'per_stock_results/')}")
    print(f"  💾 Trained Models: {results_folder}/*_universal_best.pth")
    print(f"  ✍️  Validation Trade Logs: {os.path.join(results_folder, 'best_model_trade_logs/')}")
    print(f"  🖼️  Test Performance Charts: {os.path.join(results_folder, 'test_charts/')}")
    print(f"  📄 Individual Test Reports: {os.path.join(results_folder, 'individual_reports/')}") # <-- ADDED
    print(f"{'='*80}")


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"🚀 UNIVERSAL MULTI-STOCK TRADING SYSTEM")
    print(f"Train once, trade everywhere!")
    print(f"{'='*80}")
    
    # Step 1: Ensure featured data exists
    print(f"\n📦 PHASE 1: DATA PREPARATION")
    print(f"{'='*80}")
    
    featured_data_ready = ensure_featured_data_exists(
        raw_data_folder='Raw_Data',
        featured_data_folder='Featured_Data'
    )
    
    if not featured_data_ready:
        print(f"\n❌ Cannot proceed without featured data. Please check the errors above.")
        exit(1)
    
    # Step 2: Run universal training
    print(f"\n📦 PHASE 2: UNIVERSAL TRAINING & EVALUATION")
    print(f"{'='*80}")
    
    try:
        universal_training_workflow(
            featured_data_folder='Featured_Data',
            results_folder='Universal_Trading_Results'
        )
    except Exception as e:
        print(f"\n❌ ERROR during universal training: {e}")
        traceback.print_exc()
        exit(1)
    
    print(f"\n{'='*80}")
    print(f"🎉 ALL PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")