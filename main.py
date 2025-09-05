"""
Complete main execution module for the Enhanced Trading System
"""

import os
import glob
import traceback
from config import TradingConfig
from data_handler import DataHandler
from trading_agent import PPOTradingAgent  # Changed from EnsembleTradingAgent
from visualization import TradingVisualizer
from utils import PerformanceEvaluator, ResultsSaver
from train import enhanced_walk_forward_training

def process_single_asset(file_path, asset, timeframe, config, results_folder):
    """Process a single asset with comprehensive analysis"""
    print(f"Loading data from: {file_path}")
    
    try:
        # Initialize components
        data_handler = DataHandler(config)
        
        # Load and prepare data
        data = data_handler.load_featured_data(file_path)
        train_data, val_data, test_data = data_handler.split_data(data)
        print(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Feature selection
        feature_columns = data_handler.select_features(train_data)
        scaler = data_handler.prepare_scaler(train_data, feature_columns)
        
        # DEBUG: Check actual state size
        from trading_environment import AdvancedTradingEnvironment
        temp_env = AdvancedTradingEnvironment(train_data[:100], feature_columns, scaler, config)
        temp_state = temp_env.get_state()
        actual_state_size = len(temp_state)
        
        print(f"🔍 DEBUG: Feature columns: {len(feature_columns)}")
        print(f"🔍 DEBUG: Actual state size: {actual_state_size}")
        
        # Initialize PPO trading system with correct state size
        state_size = actual_state_size
        action_size = 3
        agent = PPOTradingAgent(state_size, action_size, config)  # Changed class name
        
        # Enhanced training with more walk-forward steps
        print(f"\n🚀 Starting PPO Training for {asset}_{timeframe}")
        
        # Pass the asset-specific results folder to save models there
        episode_returns, model_path = enhanced_walk_forward_training(
            agent, train_data, feature_columns, scaler, config,
            save_dir=results_folder
        )
        
        # Load best model and test
        print(f"\n📊 Loading best model and testing...")
        agent.load_models(model_path)
        
        # Comprehensive evaluation
        evaluator = PerformanceEvaluator(config)
        test_results = evaluator.comprehensive_evaluation(agent, test_data, feature_columns, scaler)
        
        # Generate visualizations
        print(f"🎨 Generating visualizations...")
        visualizer = TradingVisualizer()
        charts_dir = os.path.join(results_folder, "charts")
        chart_paths = visualizer.create_and_save_individual_charts(
            env=test_results['ensemble']['env'],
            asset_name=f"{asset}_{timeframe}",
            save_dir=charts_dir
        )
        
        # Save comprehensive results
        results_saver = ResultsSaver()
        results_saver.save_comprehensive_results(
            test_results, asset, timeframe, results_folder,
            episode_returns, test_data, config
        )
        
        # Print final summary
        ensemble_env = test_results['ensemble']['env']
        final_value = ensemble_env.get_portfolio_value()
        total_return = (final_value - config.INITIAL_CASH) / config.INITIAL_CASH * 100
        
        print(f"\n✅ COMPLETED {asset}_{timeframe}")
        print(f" Final Portfolio Value: ${final_value:,.0f}")
        print(f" Total Return: {total_return:+.2f}%")
        print(f" Total Trades: {len(ensemble_env.trades)}")
        print(f" Charts saved to: {charts_dir}")
        print(f" Models saved to: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {asset}_{timeframe}: {str(e)}")
        traceback.print_exc()
        return False

def process_all_assets(featured_data_folder='Featured_Data', results_folder='Enhanced_PPO_Trading_Results'):
    """Process all featured data files in the folder"""
    # Create main results folder
    os.makedirs(results_folder, exist_ok=True)
    
    # Find all featured CSV files
    pattern = os.path.join(featured_data_folder, '*_featured.csv')
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"No featured data files found in '{featured_data_folder}'.")
        print("Please run S1_Feature_Engineering.py first.")
        return
    
    print(f"Found {len(all_files)} featured data files to process")
    
    # Enhanced configuration
    config = TradingConfig()
    
    successful_assets = []
    failed_assets = []
    
    # Process each file sequentially
    for i, file_path in enumerate(all_files):
        try:
            print(f"\n{'='*100}")
            print(f"PROCESSING FILE {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
            print('='*100)
            
            # Extract asset name and timeframe
            filename = os.path.basename(file_path)
            asset_timeframe = filename.replace('_featured.csv', '')
            asset_parts = asset_timeframe.split('_')
            asset = asset_parts[0]
            timeframe = '_'.join(asset_parts[1:]) if len(asset_parts) > 1 else '1day'
            
            # Create asset-specific results folder
            asset_results_folder = os.path.join(results_folder, f"{asset}_{timeframe}")
            os.makedirs(asset_results_folder, exist_ok=True)
            
            # Process single asset
            success = process_single_asset(
                file_path, asset, timeframe, config, asset_results_folder
            )
            
            if success:
                successful_assets.append(f"{asset}_{timeframe}")
                print(f"✅ Successfully processed {asset}_{timeframe}")
            else:
                failed_assets.append(f"{asset}_{timeframe}")
                print(f"❌ Failed to process {asset}_{timeframe}")
                
        except Exception as e:
            failed_assets.append(f"{asset}_{timeframe}")
            print(f"❌ Critical error processing {file_path}: {str(e)}")
            traceback.print_exc()
    
    # Generate final summary report
    generate_final_summary_report(successful_assets, failed_assets, results_folder)

def generate_final_summary_report(successful_assets, failed_assets, results_folder):
    """Generate final summary report of all processed assets"""
    summary_report = f"""
ENHANCED PPO TRADING SYSTEM - FINAL PROCESSING REPORT
{'='*80}

PROCESSING SUMMARY:
✅ Successfully Processed: {len(successful_assets)} assets
❌ Failed to Process: {len(failed_assets)} assets
📊 Total Assets Attempted: {len(successful_assets) + len(failed_assets)}
📈 Success Rate: {len(successful_assets)/(len(successful_assets) + len(failed_assets))*100:.1f}%

SUCCESSFUL ASSETS:
{chr(10).join(f" ✅ {asset}" for asset in successful_assets)}
"""

    if failed_assets:
        summary_report += f"""
FAILED ASSETS:
{chr(10).join(f" ❌ {asset}" for asset in failed_assets)}
"""

    summary_report += f"""
RESULTS LOCATION:
📁 Main Results Folder: {results_folder}
📁 Individual Asset Folders: {results_folder}/[ASSET_TIMEFRAME]/
📊 Charts: [ASSET_FOLDER]/charts/
📋 Trade Logs: [ASSET_FOLDER]/
📈 Performance Metrics: [ASSET_FOLDER]/

NEXT STEPS:
1. Review individual asset performance in their respective folders
2. Compare charts across different assets
3. Analyze trade logs for strategy insights
4. Use performance metrics for portfolio construction

Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Save summary report
    summary_path = os.path.join(results_folder, 'FINAL_PPO_PROCESSING_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    print(f"\n{'='*100}")
    print("🎯 MULTI-ASSET PPO PROCESSING COMPLETED!")
    print(f"📋 Final Summary: {len(successful_assets)} successful, {len(failed_assets)} failed")
    print(f"📁 Results saved to: {results_folder}")
    print(f"📊 Summary report: {summary_path}")
    print("="*100)

if __name__ == "__main__":
    import pandas as pd
    
    print("🚀 Starting Enhanced Multi-Asset PPO Trading System")
    print("="*100)
    
    # Process all assets
    process_all_assets(
        featured_data_folder='Featured_Data',
        results_folder='Enhanced_PPO_Trading_Results'
    )
    
    print("\n🎉 ALL PPO PROCESSING COMPLETED!")
    print("Check the 'Enhanced_PPO_Trading_Results' folder for comprehensive results")
