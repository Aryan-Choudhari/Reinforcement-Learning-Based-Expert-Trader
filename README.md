<div align="center">

# Expert Trader PPO Framework

**A Reinforcement Learning framework designed to emulate the nuanced decision-making, risk management, and position management strategies of expert human traders.**

</div>

> **Note**: This is a developing project. The architecture and features are subject to change. Contributions and feedback are welcome!

---

## Table of Contents

- [About The Project](#about-the-project)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [Configuration](#configuration)
- [Future Development](#future-development)
- [License](#license)

---

## About The Project

Traditional algorithmic trading models often focus on maximizing raw profit, which can lead to strategies that are overfit to historical data and perform poorly in live markets. This project takes a different approach by leveraging **Reinforcement Learning (RL)** to build an agent that learns the *process* of professional trading, not just the outcome.

The core philosophy is to create a robust agent that can achieve consistent, risk-adjusted returns by emulating the sophisticated behaviors of an expert human trader. This includes:
-   **Adapting to Market Regimes**: Understanding whether a market is trending, ranging, or in a correction, and adjusting its strategy accordingly.
-   **Dynamic Risk Management**: Actively managing open positions by adjusting stop losses, taking partial profits, and sizing new positions based on market conviction.
-   **Structured Learning**: Following a curriculum-based training approach that builds skills progressively, from basic market reading to advanced risk control.

By using an advanced **Proximal Policy Optimization (PPO)** algorithm, the agent learns a policy that balances exploration of new strategies with exploitation of known, profitable ones, all while being guided by a reward function that values professional discipline.

---

## Core Features

-   **Ensemble PPO Agent**
    -   The agent's policy is derived from an ensemble of diverse neural network architectures for enhanced robustness and to reduce model-specific biases. The ensemble includes:
        -   A standard Multi-Layer Perceptron (MLP) `ActorNetwork`.
        -   A `LSTMActorCritic` model to capture temporal patterns and sequences in market data.
        -   An `AttentionActorCritic` model to focus on the most salient market features at any given time.

-   **Expert Behavior Simulation**
    -   A rule-based `AdvancedRiskManager` works alongside the RL agent to enforce professional trading discipline. Key behaviors include:
        -   **Partial Exits**: Automatically selling a portion (e.g., 25%) of a position as it returns to breakeven to mitigate risk.
        -   **Dynamic Stop-Loss Adjustments**: Giving a losing trade more room if the overarching market trend strongly supports the original thesis (e.g., widening the stop-loss on a long position during a minor dip in a powerful uptrend).
        -   **Moving to Breakeven**: Shifting the stop-loss to the entry price after a position has been held for a minimum period, effectively creating a risk-free trade.

-   **Market Regime & Bias Analysis**
    -   The system categorizes the current market environment into distinct regimes, such as `bullish`, `bearish`, `correction_buy` (a dip in an uptrend), `correction_sell` (a rally in a downtrend), or `ranging`.
    -   This market bias is fed into the agent's state and reward function, encouraging it to take actions that are in harmony with the broader market context.

-   **Advanced Reward Engineering**
    -   The `ImprovedRewardFunction` goes far beyond simple profit-and-loss, granting rewards for specific expert actions:
        -   **High rewards** for buying into dips during strong uptrends or shorting rallies in downtrends.
        -   **Bonuses** for intelligently managing positions (e.g., successful stop-loss adjustments, holding profitable trades for longer durations).
        -   **Penalties** for taking actions that go against a strong, identified market bias (e.g., shorting in a strong bull market).

-   **Structured Walk-Forward Training**
    -   The agent is trained using a walk-forward methodology on an expanding data window, which prevents lookahead bias and promotes generalization. The training is divided into four distinct phases:
        1.  **Market Reading**: Focus on learning basic market patterns with high exploration.
        2.  **Trend Mastery**: Focus on mastering how to follow and capitalize on market trends.
        3.  **Risk Management**: Focus on expert-level risk control and defensive actions.
        4.  **Position Mastery**: Focus on advanced position management and capital allocation.

-   **Comprehensive Analytics & Visualization**
    -   The framework automatically generates a suite of reports and charts for in-depth performance analysis. This includes:
        -   Price action charts with buy/sell/stop-loss signals overlaid.
        -   Portfolio equity curve vs. a "Buy & Hold" benchmark.
        -   Drawdown analysis for both the agent and the benchmark.
        -   Distribution plots of individual trade Profit & Loss.
        -   A final performance summary report with key metrics like Sharpe Ratio, Total Return, and Win Rate.

---

## System Architecture

The project is modular, with each component responsible for a specific part of the trading and learning process. The `PPOTradingAgent` acts as the "brain," making decisions. It operates within the `AdvancedTradingEnvironment`, which simulates the market. The agent's decisions are influenced by the `AdvancedRiskManager`, a "gut feeling" expert system. The agent's learning is guided by the `ImprovedRewardFunction` and the `enhanced_walk_forward_training` curriculum.

| Component                  | File(s)                                   | Description                                                                                                                              |
| -------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Execution Orchestrator** | `main.py`                                 | The main entry point. Manages the end-to-end workflow of data loading, training, evaluation, and result saving for multiple assets.       |
| **RL Agent (The Brain)** | `trading_agent.py`, `models.py`           | The `PPOTradingAgent` containing the ensemble of Actor-Critic neural networks. It learns from experience to make trading decisions.       |
| **Market Simulator** | `trading_environment.py`                  | The `AdvancedTradingEnvironment` where the agent interacts. It simulates trade execution, manages the portfolio, and calculates the state. |
| **Expert Rules Engine** | `risk_manager.py`                         | The `AdvancedRiskManager` that provides the agent with expert-like rules for position sizing, stop placement, and trade management.      |
| **Training Curriculum** | `train.py`                                | Implements the `enhanced_walk_forward_training` logic, guiding the agent through different learning phases to develop expert skills.     |
| **Guiding Principles** | `reward_function.py`                      | The `ImprovedRewardFunction` which calculates rewards, steering the agent's learning towards desired professional trading behaviors.       |
| **Data & Features** | `data_handler.py`                         | Manages loading data, feature engineering (e.g., regime detection), feature selection, and data splitting for training and testing.      |
| **Utilities & Analysis** | `utils.py`, `visualization.py`, `result_analyzer.py` | A suite of tools for performance evaluation, generating charts, saving results, and performing cross-asset comparative analysis.           |
| **Configuration** | `config.py`                               | A centralized file for all system hyperparameters, trading parameters, and configuration settings.                                     |


---

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

-   Python 3.8 or later
-   Pip package manager
-   A C++ compiler (required for PyTorch)
-   It is highly recommended to use a virtual environment.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <your-repository-url>
    cd expert-trader-ppo-framework
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following contents:
    ```
    numpy
    pandas
    torch
    matplotlib
    scikit-learn
    ```
    Then, install them using pip:
    ```sh
    pip install -r requirements.txt
    ```
    *Note: The `data_handler.py` file has an optional dependency on `talib` for calculating technical indicators. If you wish to use it, you must install the TA-Lib library on your system before `pip install TA-Lib`.*

---

## Usage Workflow

The system is designed to run in a sequential workflow.

1.  **Prepare Your Data**
    -   The system expects pre-processed data with engineered features. As hinted in `main.py`, you should have a preliminary script (e.g., `S1_Feature_Engineering.py`) that processes raw price data.
    -   Place the final featured CSV files (e.g., `BTCUSD_1day_featured.csv`) into the `Featured_Data/` directory. The `main.py` script will automatically discover and process every file in this folder.

2.  **Run Training and Evaluation**
    -   Execute the main script from your terminal. This will kick off the entire process for all assets.
    ```sh
    python main.py
    ```
    -   During execution, the script will print progress updates for each walk-forward step and training phase.
    -   Upon completion for each asset, a results folder (e.g., `Enhanced_PPO_Trading_Results/BTCUSD_1day/`) will be created, containing saved models, trade logs, portfolio history, a JSON summary, and a `charts/` directory with visualizations.

3.  **Analyze Aggregate Results**
    -   After all assets have been processed, run the result analyzer to get a high-level performance comparison.
    ```sh
    python result_analyzer.py
    ```
    -   This script aggregates the JSON summaries from all individual asset runs, prints performance rankings to the console, and saves a `comprehensive_results.csv` file in the main results directory for further analysis.

---

## Configuration

Nearly all aspects of the agent's behavior and the training process can be modified in the **`config.py`** file. This centralizes control and makes experimentation easy.

Key parameters to consider adjusting include:
-   **`INITIAL_CASH`**: The starting capital for each simulation.
-   **`TRANSACTION_COST`**: The percentage cost per trade.
-   **`MAX_POSITIONS`**: The maximum number of concurrent open positions.
-   **`EPISODES`**: The total number of training episodes.
-   **`WALK_FORWARD_STEPS`**: The number of walk-forward training and validation windows.

---

## Future Development

This project is actively under development. Potential future enhancements include:
-   **Hyperparameter Optimization**: Integrating tools like Optuna or Ray Tune to systematically find the best model and training parameters.
-   **Live Trading Integration**: Building a bridge to connect the trained agent to a brokerage API for real-time paper or live trading.
-   **Advanced Models**: Experimenting with more complex architectures like Transformers for the agent's policy network.
-   **Portfolio-Level Management**: Expanding the agent's action space to manage risk and allocation across a portfolio of assets simultaneously.

---

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
