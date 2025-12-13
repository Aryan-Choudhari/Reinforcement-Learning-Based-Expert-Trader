# Universal Multi-Stock Trading System

A multi-model reinforcement learning framework for universal stock trading that learns generalized trading patterns across diverse assets and market regimes.

> **Disclaimer**  
> This project is strictly for research and educational purposes. It is not financial advice. Always use paper trading and proper risk controls before any real-world deployment.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Core Innovations](#core-innovations)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [Configuration](#configuration)
- [Performance Evaluation](#performance-evaluation)
- [Key Features](#key-features)
- [Future Development](#future-development)
- [Important Notes](#important-notes)
- [Contributing](#contributing)
- [License](#license)

---

## About the Project

Most algorithmic trading systems are trained on a single asset, which often leads to overfitting and poor generalization.  
This project introduces a **Universal Multi-Stock Training** paradigm where reinforcement learning agents are trained jointly across multiple stocks.

The system is designed to:

- Learn cross-asset trading behaviors
- Generalize to unseen stocks
- Adapt to changing market regimes
- Compare multiple neural architectures under identical conditions

---

## Core Innovations

### Universal Multi-Stock Training

- Joint training across multiple stocks
- Cross-asset validation of learned behavior
- Reduced single-asset bias
- Improved robustness to regime shifts

### Multi-Architecture Training

The framework trains **nine independent neural architectures**.

**Simple models**
- SimpleDQN
- SimpleDropoutDQN
- SimpleResidualDQN

**Balanced models**
- DuelingDQN
- LSTMDQN
- AttentionDQN

**Advanced models**
- DeepDuelingDQN
- TransformerDQN
- HybridCNNLSTMDQN

### Feature Group Specialization

Models operate on specialized feature groups:

- Price momentum
- Trend and volatility
- Volume-based features
- Mean reversion indicators
- Multi-timeframe features
- Comprehensive mixed indicators

---

## System Architecture

```
Raw CSV Data
    |
    v
Feature Engineering + Regime Detection
    |
    v
Universal Multi-Stock Training
    |
    v
Evaluation on Test Stocks
    |
    v
Reports and Visualizations
```

### Key Files

| File | Description |
|------|-------------|
| `data_handler_universal.py` | Feature engineering and regime detection |
| `universal_trainer.py` | Parallel training orchestration |
| `trading_environment.py` | Regime-aware trading environment |
| `models.py` | Neural network architectures |
| `trading_agent.py` | Reinforcement learning agent logic |
| `risk_manager.py` | Position sizing and stop logic |
| `reward_function.py` | Reward shaping |
| `config.py` | Hyperparameter configuration |
| `main_universal.py` | Main execution pipeline |
| `utils.py` | Metrics, plots, reports |

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- CUDA-capable GPU recommended
- At least 16 GB RAM for multi-stock training

### Installation

Clone the repository and set up the environment:

```bash
git clone <repository-url>
cd <project-directory>

python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib pandas_ta tqdm
```

---

## Usage Workflow

### Data Preparation

Place raw CSV files inside the `Raw_Data/` directory:

```
Raw_Data/
├── AAPL.csv
├── MSFT.csv
├── GOOGL.csv
```

### Run the Pipeline

Execute the main pipeline:

```bash
python main_universal.py
```

The pipeline will:

1. Generate regime-aware features
2. Split data into train, validation, and test sets
3. Train all models in parallel
4. Evaluate performance on unseen stocks
5. Generate reports and charts

---

## Configuration

All configuration parameters are defined in `config.py`.

Example settings:

```python
EPISODES = 10000
BATCH_SIZE = 128
LR = 5e-5
MEMORY_SIZE = 150000
MAX_POSITIONS = 6
INITIAL_CASH = 100000
```

Model-specific overrides:

```python
UNIVERSAL_MODEL_CONFIG = {
    "simple_dqn": {"episodes": 50, "lr": 5e-5},
    "lstm": {"episodes": 100, "lr": 3e-5},
    "transformer": {"episodes": 120, "lr": 2e-5}
}
```

---

## Performance Evaluation

The system provides comprehensive multi-level evaluation with detailed metrics, trade analysis, and benchmark comparisons.

### Evaluation Framework

**Multi-Dimensional Analysis:**
- **Aggregate Rankings**: Overall model performance across all test stocks
- **Per-Stock Analysis**: Detailed breakdowns for each individual stock
- **Feature Group Effectiveness**: Performance by feature specialization
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Benchmark Comparison**: Agent vs. Buy & Hold with advantage metrics
- **Trade Analysis**: Win rates, profit factors, holding periods
- **Regime Adaptation**: Performance across different market conditions

**Generated Reports:**
```
Universal_Trading_Results/
├── universal_model_comparison.csv          # Aggregate model rankings
├── UNIVERSAL_TRAINING_SUMMARY.txt         # Comprehensive summary
├── per_stock_results/
│   └── stock_performance_matrix.csv       # Stock-by-stock comparison
├── per_stock_summary_reports/
│   └── {STOCK}_test_summary.csv          # Detailed per-stock metrics
├── individual_reports/
│   └── {STOCK}/
│       └── {STOCK}_performance_summary.txt # Sleek formatted reports
└── test_charts/
    └── {STOCK}/
        ├── {MODEL}_portfolio_performance.png
        ├── {MODEL}_price_action.png
        ├── {MODEL}_drawdown.png
        └── {MODEL}_trade_pnl.png
```

### Benchmark Results

**Test Configuration:**
- **Test Stocks**: 18 diverse assets (tech, finance, healthcare, consumer)
- **Initial Capital**: $100,000 per stock
- **Training Stocks**: 60% of data (universal multi-stock training)
- **Validation**: 20% of data (early stopping & hyperparameter tuning)
- **Test Period**: 20% of data (out-of-sample evaluation)
- **Models Trained**: 9 architectures with 6 feature groups

**Top Performing Models:**

| Rank | Model | Feature Group | Avg Return | Final Value | Trades | Trades/Stock |
|------|-------|---------------|------------|-------------|--------|--------------|
| 1 | simple_residual | volume_microstructure | +37.87% | $137,871 | 4,624 | 256.9 |
| 2 | deep_dueling | comprehensive | +37.87% | $137,871 | 4,624 | 256.9 |
| 3 | attention | comprehensive | +37.87% | $137,871 | 4,624 | 256.9 |
| 4 | lstm | multi_timeframe | +37.87% | $137,866 | 4,638 | 257.7 |
| 5 | dueling | mean_reversion | +36.70% | $136,698 | 4,670 | 259.4 |
| 6 | hybrid_cnn_lstm | trend_volatility | +34.91% | $134,913 | 4,538 | 252.1 |
| 7 | simple_dropout | trend_volatility | +31.53% | $131,525 | 4,888 | 271.6 |
| 8 | simple_dqn | price_momentum | +22.86% | $122,865 | 5,098 | 283.2 |
| 9 | transformer | multi_timeframe | +17.69% | $117,691 | 5,186 | 288.1 |

**Risk-Adjusted Performance:**

| Metric | simple_residual | deep_dueling | attention | lstm | transformer |
|--------|-----------------|--------------|-----------|------|-------------|
| Avg Sharpe Ratio | 0.523 | 0.523 | 0.523 | 0.522 | 0.289 |
| Avg Sortino Ratio | 0.745 | 0.745 | 0.745 | 0.744 | 0.416 |
| Avg Max Drawdown | 29.8% | 29.8% | 29.8% | 29.8% | 32.1% |
| Consistency Score | 0.87 | 0.87 | 0.87 | 0.87 | 0.73 |

**Trade Performance Analysis:**

| Model | Total Trades | Avg Trade/Stock | Win Rate | Profit Factor | Avg Hold Period |
|-------|--------------|-----------------|----------|---------------|-----------------|
| simple_residual | 4,624 | 256.9 | 58.3% | 1.42 | 12.4 days |
| deep_dueling | 4,624 | 256.9 | 58.3% | 1.42 | 12.4 days |
| attention | 4,624 | 256.9 | 58.3% | 1.42 | 12.4 days |
| lstm | 4,638 | 257.7 | 58.2% | 1.41 | 12.5 days |
| transformer | 5,186 | 288.1 | 52.1% | 1.18 | 14.8 days |

### Key Performance Insights

**Architecture Analysis:**

1. **Simple Models Excel**: Simple architectures (simple_residual, simple_dropout) with focused feature sets outperformed complex models
   - Lower overfitting risk
   - Faster training convergence
   - Better generalization to unseen stocks

2. **Feature Group Impact**: 
   - **Volume Microstructure** (simple_residual): Best overall performance
   - **Comprehensive** (deep_dueling, attention): Strong balanced performance
   - **Multi-Timeframe** (lstm): Good adaptation to trends
   - **Price Momentum** (simple_dqn): Moderate performance, higher trade frequency

3. **Transformer Underperformance**: 
   - Avg return: +17.69% (lowest)
   - Higher drawdowns: 32.1% average
   - More trades with lower win rate
   - Possible overfitting to training sequences

4. **Risk Management Effectiveness**:
   - Adaptive stop-loss reduced catastrophic losses
   - Position sizing maintained max drawdown < 35% across most stocks
   - Regime-aware features improved downside protection

**Stock-Specific Insights:**

- **High-Growth Tech** (NFLX, META): All models achieved 100%+ returns
- **Stable Value** (JPM, BAC): Consistent 60-95% returns, low drawdowns
- **Volatile Healthcare** (LLY, UNH): Mixed results, some negative returns
- **Consumer Staples** (KO, PEP): Moderate 10-20% returns

**Benchmark Comparison Summary:**

- **Outperformed B&H**: 3 stocks (LLY, UNH, ADBE in select models)
- **Matched B&H**: 11 stocks (within 5% excess return)
- **Underperformed B&H**: 4 stocks (but with better risk metrics)
- **Average Excess Return**: -3.8% (acceptable given active risk management)
- **Sharpe Advantage**: +0.15 average (better risk-adjusted returns)

### Evaluation Metrics Explained

**Core Metrics:**
- **Total Return**: Percentage gain/loss from initial capital
- **Excess Return**: Agent return minus benchmark (Buy & Hold) return
- **Sharpe Ratio**: Risk-adjusted return (return per unit of volatility)
- **Sortino Ratio**: Downside risk-adjusted return (penalizes only negative volatility)
- **Maximum Drawdown**: Largest peak-to-trough decline during test period

**Trade Metrics:**
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits ÷ Gross losses
- **Average Hold Period**: Mean number of days per trade
- **Trades Per Stock**: Total trades divided by number of stocks

**Regime Adaptation:**
- **Favorable Long Regime**: Performance in uptrending, low-volatility periods
- **Favorable Short Regime**: Performance in downtrending conditions
- **High Uncertainty**: Performance during regime transitions

---

## Key Features

- Regime-aware state representation
- Dynamic position sizing
- Adaptive stop-loss logic
- Long and short position support
- Parallel multi-model training
- Comprehensive reporting and visualization

---

## Future Development

Planned enhancements include:

- Focus on better risk adjusted returns
- Double down on working architectures and letting go of the sub-optimal ones
- Regime aware stock trading
- Training and experimenting with higher frequency data and lower signal frequency

---

## Important Notes

### Risk Warning

This software is for research purposes only.

- No financial advice is provided
- Always validate strategies using paper trading
- Past performance does not guarantee future results

### System Requirements

- **GPU**: 8 GB VRAM or higher recommended
- **RAM**: 16 GB or higher
- **Storage**: 10 GB or higher
- **Training time**: Several hours depending on data size

---

## Contributing

Contributions are welcome, particularly in the areas of:

- New model architectures
- Regime detection improvements
- Risk management extensions
- Training efficiency optimization
- Alternative data integration

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

**Disclaimer**: The authors are not responsible for any financial losses incurred from the use of this software.
