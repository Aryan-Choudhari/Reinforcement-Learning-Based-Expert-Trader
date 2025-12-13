import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from trading_environment import AdvancedTradingEnvironment
from utils import PerformanceMetrics, DetailedTradeLogger, DiscrepancyDetector, TradingVisualizer
from trading_agent import IndividualTradingAgent
from config import FeatureGroupConfig, TradingConfig
from tqdm import tqdm
import random
import concurrent.futures 
import traceback

# ==============================================================================
# STANDALONE WRAPPER FUNCTION FOR PARALLEL EXECUTION
# ==============================================================================
def _train_single_model_wrapper(model_name, train_stocks, val_stocks, common_features, config_dict, save_dir):
    """
    A self-contained function to train a single model. This is designed to be
    called by the ProcessPoolExecutor.
    """
    try:
        # Re-create config object from dictionary inside the worker process
        config = TradingConfig()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        print(f"[Process-{os.getpid()}] Starting training for: {model_name}")
        
        # 1. Get feature columns assigned to this model
        model_feature_assignments = FeatureGroupConfig.get_model_feature_assignments()
        group_name = model_feature_assignments[model_name]
        group_config = FeatureGroupConfig.get_group(group_name)
        model_feature_columns = FeatureGroupConfig.filter_features(
            common_features, group_config['features']
        )
        if 'close' in model_feature_columns:
            model_feature_columns.remove('close')

        if len(model_feature_columns) < 5:
            print(f"[Process-{os.getpid()}] WARNING: Only {len(model_feature_columns)} features for {model_name}, skipping.")
            return None

        # 2. Create and fit a dedicated scaler for this model's features
        all_train_features = pd.concat(
            [data[model_feature_columns] for data in train_stocks.values()],
            ignore_index=True
        )
        scaler = StandardScaler()
        scaler.fit(all_train_features)
        
        # 3. Instantiate a trainer and run the training process
        trainer_instance = UniversalMultiStockTrainer(config, save_dir)
        result = trainer_instance.train_universal_model(
            model_name, train_stocks, val_stocks,
            model_feature_columns, scaler
        )
        
        # 4. Add the scaler and feature columns to the result for later evaluation
        result['scaler'] = scaler
        result['feature_columns'] = model_feature_columns
        
        print(f"[Process-{os.getpid()}] Finished training for: {model_name}")
        return result

    except Exception as e:
        print(f"!!!!!!!!!!!!!! ERROR in worker process for model {model_name} !!!!!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None


class UniversalMultiStockTrainer:
    """Train models across multiple stocks for universal strategy learning"""
    
    def __init__(self, config, save_dir='universal_models'):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.trade_logger = DetailedTradeLogger(save_dir)
        self.discrepancy_detector = DiscrepancyDetector()
        
        self.all_models = [
            'simple_dqn', 'simple_dropout', 'simple_residual',
            'dueling', 'lstm', 'attention',
            'deep_dueling', 'transformer', 'hybrid_cnn_lstm'
        ]
        
        self.model_feature_assignments = FeatureGroupConfig.get_model_feature_assignments()
        self.universal_models = {}
        
    def prepare_universal_dataset(self, all_stock_data, common_features):
        """
        Split all stocks into train/val/test.
        """
        print(f"\n{'='*80}")
        print(f"PREPARING UNIVERSAL DATASET FROM {len(all_stock_data)} STOCKS")
        print(f"{'='*80}")
        
        train_stocks = {}
        val_stocks = {}
        test_stocks = {}
        
        for stock_name, data in all_stock_data.items():
            total_samples = len(data)
            train_end = int(total_samples * 0.60)
            val_end = int(total_samples * 0.80)
            
            train_stocks[stock_name] = data.iloc[:train_end].reset_index(drop=True)
            val_stocks[stock_name] = data.iloc[train_end:val_end].reset_index(drop=True)
            test_stocks[stock_name] = data.iloc[val_end:].reset_index(drop=True)
            
            print(f"  {stock_name}: Train={len(train_stocks[stock_name])}, "
                  f"Val={len(val_stocks[stock_name])}, Test={len(test_stocks[stock_name])}")
        
        self.common_features = common_features
        return train_stocks, val_stocks, test_stocks
    
    def create_universal_scaler(self, train_stocks, feature_columns):
        """Creates a universal scaler fitted on data from ALL training stocks."""
        all_train_features = pd.concat(
            [data[feature_columns] for data in train_stocks.values()],
            ignore_index=True
        )
        scaler = StandardScaler()
        scaler.fit(all_train_features)
        return scaler

    def train_universal_model(self, model_name, train_stocks, val_stocks, 
                             feature_columns, universal_scaler):
        """
        Train a single model across ALL stocks using experience replay mixing.
        """
        print(f"\n{'='*60}")
        print(f"UNIVERSAL TRAINING: {model_name.upper()}")
        print(f"{'='*60}")
        
        train_benchmarks = {}
        val_benchmarks = {}
        
        for stock_name in train_stocks.keys():
            train_data = train_stocks[stock_name]
            val_data = val_stocks[stock_name]
            close_col = 'close_normalized' if 'close_normalized' in train_data.columns else 'close'
            if close_col in train_data.columns and not train_data.empty:
                 train_benchmarks[stock_name] = (train_data[close_col].iloc[-1] / train_data[close_col].iloc[0] - 1) * 100
            if close_col in val_data.columns and not val_data.empty:
                val_benchmarks[stock_name] = (val_data[close_col].iloc[-1] / val_data[close_col].iloc[0] - 1) * 100
        
        train_envs = {stock_name: AdvancedTradingEnvironment(train_stocks[stock_name], feature_columns, universal_scaler, self.config) for stock_name in train_stocks.keys()}
        for env in train_envs.values():
            env.training = True
        
        first_env = list(train_envs.values())[0]
        state_size = len(first_env.get_state())
        action_size = 3
        
        agent = IndividualTradingAgent(state_size, action_size, self.config, model_name)
        
        best_universal_score = -np.inf
        best_results = None
        no_improvement_count = 0
        
        model_save_path = os.path.join(self.save_dir, f'{model_name}_universal_best.pth')
        
        # Use leave=False to clean up the progress bar after completion in parallel mode
        pbar = tqdm(range(self.config.EPISODES), desc=f"{model_name}", unit="ep", position=0, leave=False)
        
        for e in pbar:
            stock_names = list(train_stocks.keys())
            random.shuffle(stock_names)
            episode_steps = 0
            
            for stock_name in stock_names:
                env = train_envs[stock_name]
                state = env.reset()
                done = False
                agent.reset_hidden_state()
                
                while not done:
                    action = agent.act(state, training=True)
                    next_state, reward, done, _ = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    episode_steps += 1
                    
                    if len(agent.memory) > self.config.BATCH_SIZE and episode_steps % 4 == 0:
                        agent.replay()
                        agent.soft_update_target_network()
            
            if (e + 1) % self.config.VALIDATION_INTERVAL == 0:
                train_results = self._evaluate_on_all_stocks(agent, train_stocks, feature_columns, universal_scaler)
                val_results = self._evaluate_on_all_stocks(agent, val_stocks, feature_columns, universal_scaler)
                
                universal_score = self._calculate_universal_score(train_results, val_results, train_benchmarks, val_benchmarks)
                
                if universal_score > best_universal_score:
                    best_universal_score = universal_score
                    best_results = {'train': train_results, 'val': val_results, 'score': universal_score}
                    agent.save_model(model_save_path)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                pbar.set_postfix(score=f"{universal_score:.3f}", best=f"{best_universal_score:.3f}", pat=f"{no_improvement_count}/{self.config.PATIENCE}")
                
                if no_improvement_count >= self.config.PATIENCE:
                    pbar.close()
                    break
        
        return {
            'model_name': model_name,
            'feature_group': self.model_feature_assignments[model_name],
            'num_features': len(feature_columns),
            'best_universal_score': best_universal_score,
            'model_path': model_save_path,
            'best_results': best_results,
            'num_stocks': len(train_stocks)
        }
    
    def train_all_universal_models_parallel(self, train_stocks, val_stocks):
        """
        NEW METHOD: Train all models in parallel using a ProcessPoolExecutor.
        """
        print("\n" + "="*80)
        print("🚀 STARTING PARALLEL MULTI-MODEL TRAINING")
        print(f"Training {len(self.all_models)} models on {len(train_stocks)} stocks concurrently.")
        print("="*80)
        
        FeatureGroupConfig.print_group_summary()
        
        all_results = []
        config_dict = {k: v for k, v in self.config.__class__.__dict__.items() if not k.startswith('__') and k.isupper()}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_model = {
                executor.submit(
                    _train_single_model_wrapper, 
                    model_name, 
                    train_stocks, 
                    val_stocks, 
                    self.common_features, 
                    config_dict, 
                    self.save_dir
                ): model_name for model_name in self.all_models
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_model), total=len(self.all_models), desc="Overall Training Progress"):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                        self.universal_models[model_name] = result
                        # This print might interleave, but it's useful for debugging
                        # print(f"✅ Training for model '{model_name}' completed successfully.")
                    else:
                        print(f"⚠️ Training for model '{model_name}' returned no result (likely skipped or failed).")
                except Exception as exc:
                    print(f"❌ Model '{model_name}' generated an exception during training: {exc}")
                    traceback.print_exc()

        print("\n" + "="*80)
        print("🏁 PARALLEL TRAINING PHASE COMPLETE")
        print(f"Successfully trained {len(all_results)} out of {len(self.all_models)} models.")
        print("="*80)
        
        return all_results

    def _evaluate_on_all_stocks(self, agent, stock_data_dict, feature_columns, scaler):
        results = {}
        for stock_name, data in stock_data_dict.items():
            env = AdvancedTradingEnvironment(data, feature_columns, scaler, self.config)
            env.training = False
            state = env.reset()
            done = False
            agent.reset_hidden_state()
            while not done:
                action = agent.act(state, training=False)
                state, _, done, _ = env.step(action)
            self._liquidate_positions(env)
            portfolio = np.array(env.portfolio_value_history)
            final_value = portfolio[-1]
            total_return = (final_value - self.config.INITIAL_CASH) / self.config.INITIAL_CASH
            results[stock_name] = {'final_value': final_value, 'total_return': total_return, 'trades': len(env.trades), 'env': env, 'portfolio': portfolio}
        return results

    def _calculate_universal_score(self, train_results, val_results, train_benchmarks, val_benchmarks):
        train_returns = [r['total_return'] for r in train_results.values()]
        val_returns = [r['total_return'] for r in val_results.values()]
        avg_train_return = np.mean(train_returns) if train_returns else 0
        avg_val_return = np.mean(val_returns) if val_returns else 0
        train_consistency = 1 / (1 + np.std(train_returns)) if train_returns else 0
        val_consistency = 1 / (1 + np.std(val_returns)) if val_returns else 0
        generalization = 1 - abs(avg_train_return - avg_val_return)
        train_beat_rate = self._count_beating_benchmark(train_results, train_benchmarks) / len(train_results) if train_results else 0
        val_beat_rate = self._count_beating_benchmark(val_results, val_benchmarks) / len(val_results) if val_results else 0
        score = (avg_train_return * 0.25 + avg_val_return * 0.35 + train_consistency * 0.10 + val_consistency * 0.10 + generalization * 0.10 + train_beat_rate * 0.05 + val_beat_rate * 0.05)
        return score

    def _count_beating_benchmark(self, results, benchmarks):
        count = 0
        for stock_name, result in results.items():
            if stock_name in benchmarks:
                agent_return = result['total_return'] * 100
                benchmark_return = benchmarks[stock_name]
                if agent_return > benchmark_return:
                    count += 1
        return count

    def _liquidate_positions(self, env):
        if env.current_step >= len(env.data): return
        last_price = env.data.iloc[-1]['close']
        for position in list(env.long_positions):
            gross_proceeds = position['shares'] * last_price
            closing_cost = gross_proceeds * env.transaction_cost
            net_proceeds = gross_proceeds - closing_cost
            opening_cost = position['entry_price'] * position['shares'] * env.transaction_cost
            profit = (last_price - position['entry_price']) * position['shares'] - opening_cost - closing_cost
            env.cash += net_proceeds
            env.trades.append({'step': env.current_step, 'type': 'LIQUIDATE_LONG', 'shares': position['shares'], 'price': last_price, 'profit': profit, 'entry_price': position['entry_price'], 'entry_step': position.get('entry_step', env.current_step)})
        for position in list(env.short_positions):
            gross_cost = position['shares'] * last_price
            closing_cost = gross_cost * env.transaction_cost
            total_cost = gross_cost + closing_cost
            opening_cost = position['entry_price'] * position['shares'] * env.transaction_cost
            profit = (position['entry_price'] - last_price) * position['shares'] - opening_cost - closing_cost
            if env.cash >= total_cost:
                env.cash -= total_cost
                env.trades.append({'step': env.current_step, 'type': 'LIQUIDATE_SHORT', 'shares': position['shares'], 'price': last_price, 'profit': profit, 'entry_price': position['entry_price'], 'entry_step': position.get('entry_step', env.current_step)})
        env.long_positions = []
        env.short_positions = []
    
    def evaluate_universal_models_on_test(self, test_stocks):
        print("\n" + "="*80)
        print(f"EVALUATING UNIVERSAL MODELS ON {len(test_stocks)} TEST STOCKS")
        print("="*80)
        test_results = []
        for model_name, model_info in self.universal_models.items():
            try:
                print(f"\nEvaluating: {model_name}")
                state_size_at_creation = model_info['num_features']
                env_added_features = 10
                # Add the count of regime features
                regime_features_count = 7  
                agent = IndividualTradingAgent(state_size_at_creation + env_added_features + regime_features_count, 3, self.config, model_name)
                agent.load_model(model_info['model_path'])
                stock_results = self._evaluate_on_all_stocks(agent, test_stocks, model_info['feature_columns'], model_info['scaler'])
                avg_return = np.mean([r['total_return'] for r in stock_results.values()])
                avg_final_value = np.mean([r['final_value'] for r in stock_results.values()])
                total_trades = sum([r['trades'] for r in stock_results.values()])
                print(f"  Avg Portfolio: ${avg_final_value:,.0f}")
                print(f"  Avg Return: {avg_return*100:+.2f}%")
                print(f"  Total Trades: {total_trades}")
                print("  Generating performance charts...")
                visualizer = TradingVisualizer()
                for stock_name, result in stock_results.items():
                    try:
                        stock_charts_dir = os.path.join(self.save_dir, 'test_charts', stock_name)
                        os.makedirs(stock_charts_dir, exist_ok=True)
                        visualizer.create_and_save_individual_charts(result['env'], f"{model_name}", stock_charts_dir)
                    except Exception as chart_err:
                        print(f"    Could not generate chart for {stock_name} with model {model_name}: {chart_err}")
                test_results.append({'model_name': model_name, 'feature_group': model_info['feature_group'], 'num_features': len(model_info['feature_columns']), 'stock_results': stock_results, 'avg_return': avg_return, 'avg_final_value': avg_final_value, 'total_trades': total_trades, 'num_stocks': len(test_stocks)})
            except Exception as e:
                print(f"  ERROR evaluating {model_name}: {e}")
                traceback.print_exc()
        return test_results

# <-- THIS FUNCTION WAS MISSING
def generate_universal_comparison_report(test_results, save_path):
    """Generate comparison report for universal models"""
    print("\n" + "="*80)
    print("UNIVERSAL MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    for result in test_results:
        comparison_data.append({
            'Model': result['model_name'],
            'Feature_Group': result['feature_group'],
            'Num_Stocks': result['num_stocks'],
            'Avg_Final_Value': result['avg_final_value'],
            'Avg_Return_%': result['avg_return'] * 100,
            'Total_Trades': result['total_trades'],
            'Trades_Per_Stock': result['total_trades'] / result['num_stocks']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Avg_Return_%', ascending=False)
    
    print("\nRanked by Average Return:")
    print(df.to_string(index=False))
    
    df.to_csv(save_path, index=False)
    print(f"\nComparison saved to: {save_path}")
    
    return df