"""
Comprehensive results analyzer for trading system performance
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime

class ResultsAnalyzer:
    def __init__(self, results_folder):
        self.results_folder = results_folder
        
    def analyze_all_results(self):
        """Analyze results from all processed assets"""
        
        # Find all result files
        summary_files = glob.glob(os.path.join(self.results_folder, '*', '*_results_summary.json'))
        
        if not summary_files:
            print("No result files found!")
            return
        
        print(f"Found {len(summary_files)} result files to analyze")
        
        # Load all results
        all_results = []
        for file_path in summary_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not all_results:
            print("No valid results found!")
            return
        
        # Create comprehensive analysis
        results_df = pd.DataFrame(all_results)
        
        # Generate performance rankings
        self._generate_performance_rankings(results_df)
        
        # Analyze by asset type
        self._analyze_by_asset_type(results_df)
        
        # Generate correlation analysis
        self._generate_correlation_analysis(results_df)
        
        # Save comprehensive report
        self._save_comprehensive_report(results_df)
    
    def _generate_performance_rankings(self, results_df):
        """Generate performance rankings"""
        
        print("\n📊 PERFORMANCE RANKINGS")
        print("="*60)
        
        # Top performers by excess return
        top_excess = results_df.nlargest(10, 'excess_annual_return')
        print("\n🏆 TOP 10 BY EXCESS ANNUAL RETURN:")
        for idx, row in top_excess.iterrows():
            status = "✅" if row['beat_benchmark'] else "❌"
            print(f"{status} {row['asset']}_{row['timeframe']}: {row['excess_annual_return']:+6.2f}%")
        
        # Top performers by Sharpe ratio
        top_sharpe = results_df.nlargest(10, 'sharpe_ratio')
        print("\n⚡ TOP 10 BY SHARPE RATIO:")
        for idx, row in top_sharpe.iterrows():
            print(f"   {row['asset']}_{row['timeframe']}: {row['sharpe_ratio']:6.3f}")
        
        # Best risk-adjusted performance
        results_df['risk_adjusted_return'] = results_df['excess_annual_return'] / (results_df['max_drawdown'] + 0.01)
        top_risk_adj = results_df.nlargest(10, 'risk_adjusted_return')
        print("\n🛡️ TOP 10 BY RISK-ADJUSTED RETURN:")
        for idx, row in top_risk_adj.iterrows():
            print(f"   {row['asset']}_{row['timeframe']}: {row['risk_adjusted_return']:6.2f}")
    
    def _analyze_by_asset_type(self, results_df):
        """Analyze performance by asset type"""
        
        print("\n📈 PERFORMANCE BY ASSET TYPE")
        print("="*60)
        
        # Group by asset
        asset_summary = results_df.groupby('asset').agg({
            'excess_annual_return': ['mean', 'std', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'beat_benchmark': 'sum'
        }).round(3)
        
        print("\nAsset Performance Summary:")
        print(asset_summary)
    
    def _generate_correlation_analysis(self, results_df):
        """Generate correlation analysis between metrics"""
        
        numeric_cols = ['agent_annual_return', 'bnh_annual_return', 'excess_annual_return', 
                       'sharpe_ratio', 'max_drawdown']
        
        correlation_matrix = results_df[numeric_cols].corr()
        
        print("\n🔗 METRIC CORRELATIONS")
        print("="*60)
        print(correlation_matrix.round(3))
    
    def _save_comprehensive_report(self, results_df):
        """Save comprehensive analysis report"""
        
        # Performance statistics
        performance_stats = {
            'total_assets_tested': len(results_df),
            'assets_beating_benchmark': results_df['beat_benchmark'].sum(),
            'beat_rate': results_df['beat_benchmark'].mean() * 100,
            'average_excess_return': results_df['excess_annual_return'].mean(),
            'average_sharpe_ratio': results_df['sharpe_ratio'].mean(),
            'average_max_drawdown': results_df['max_drawdown'].mean(),
            'best_performer': results_df.loc[results_df['excess_annual_return'].idxmax(), 'asset'],
            'worst_performer': results_df.loc[results_df['excess_annual_return'].idxmin(), 'asset']
        }
        
        # Save detailed results
        results_df.to_csv(os.path.join(self.results_folder, 'comprehensive_results.csv'), index=False)
        
        # Save performance statistics
        with open(os.path.join(self.results_folder, 'performance_statistics.json'), 'w') as f:
            json.dump(performance_stats, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive analysis saved to: {self.results_folder}")
        print(f"   📊 Detailed results: comprehensive_results.csv")
        print(f"   📈 Performance stats: performance_statistics.json")

# Usage example
if __name__ == "__main__":
    analyzer = ResultsAnalyzer('Enhanced_Trading_Results')
    analyzer.analyze_all_results()
