"""
Pure RL Trading environment with expert human trader position management
"""

import numpy as np

class AdvancedTradingEnvironment:
    def __init__(self, data, feature_columns, scaler, config):
        self.data = data.copy()
        self.config = config
        self.initial_cash = float(config.INITIAL_CASH)
        self.transaction_cost = config.TRANSACTION_COST
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.scaled_features = self.scaler.transform(self.data[self.feature_columns])
        self.max_positions = config.MAX_POSITIONS

        # Initialize components (Pure RL - No LLM components)
        from risk_manager import AdvancedRiskManager
        from reward_function import ImprovedRewardFunction
        
        self.risk_manager = AdvancedRiskManager(config)
        self.reward_function = ImprovedRewardFunction()
        
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.long_positions = []
        self.short_positions = []
        self.current_step = 40
        self.done = False
        self.trades = []
        self.portfolio_value_history = [self.initial_cash]
        self.benchmark_values = []
        return self.get_state()

    def get_total_position_value(self):
        """Calculate total value of all positions"""
        if self.current_step >= len(self.data):
            return 0
        
        current_price = self.data.iloc[self.current_step]['close']
        long_value = sum(pos['shares'] * current_price for pos in self.long_positions)
        short_value = sum(pos['shares'] * current_price for pos in self.short_positions)
        return long_value - short_value

    def get_portfolio_value(self):
        """Calculate total portfolio value"""
        position_value = self.get_total_position_value()
        return self.cash + position_value

    def get_state(self):
        """Pure RL state without LLM signals"""
        if self.current_step >= len(self.data):
            return np.zeros(len(self.feature_columns) + 15)

        market_features = self.scaled_features[self.current_step]
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.get_portfolio_value()

        # Enhanced portfolio composition
        long_value = sum(pos['shares'] * current_price for pos in self.long_positions)
        short_value = sum(pos['shares'] * current_price for pos in self.short_positions)
        net_position_value = long_value - short_value

        position_value_ratio = net_position_value / portfolio_value if portfolio_value > 0 else 0
        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 1.0
        long_position_ratio = long_value / portfolio_value if portfolio_value > 0 else 0
        short_position_ratio = short_value / portfolio_value if portfolio_value > 0 else 0
        long_count_ratio = len(self.long_positions) / self.max_positions
        short_count_ratio = len(self.short_positions) / self.max_positions

        # Market regime features (from data, not LLM)
        vol_regime = self.data.iloc[self.current_step].get('vol_regime_numeric', 1)
        trend_alignment = self.data.iloc[self.current_step].get('trend_alignment', 0)
        momentum_strength = self.data.iloc[self.current_step].get('momentum_strength', 0)
        market_structure = self.data.iloc[self.current_step].get('market_structure', 0)
        rsi = self.data.iloc[self.current_step].get('rsi_14', 50)
        rsi_normalized = (rsi - 50) / 50
        dist_to_resistance = self.data.iloc[self.current_step].get('dist_to_resistance', 0)
        dist_to_support = self.data.iloc[self.current_step].get('dist_to_support', 0)

        recent_momentum = self._calculate_momentum()
        recent_volatility = self._calculate_volatility()

        additional_features = [
            position_value_ratio, cash_ratio, long_position_ratio, short_position_ratio,
            long_count_ratio, short_count_ratio, vol_regime, trend_alignment,
            momentum_strength, market_structure, rsi_normalized, dist_to_resistance,
            dist_to_support, recent_momentum, recent_volatility
        ]

        state = np.concatenate([market_features, additional_features])
        return state

    def step(self, action):
        if self.done or self.current_step >= len(self.data) - 1:
            self.done = True
            return self.get_state(), 0, True, {}

        current_price = float(self.data.iloc[self.current_step]['close'])

        # Expert position management (same logic for both directions)
        self._process_expert_position_management(current_price)

        # Process stop losses (after position management adjustments)
        self._process_stop_losses(current_price)

        trade_occurred = False
        total_positions = len(self.long_positions) + len(self.short_positions)

        # Get market bias for intelligent position direction (Pure RL, no LLM)
        direction_prefs, market_bias = self.risk_manager.get_position_direction_preference(self)

        # Action space: 0=Hold, 1=Buy Long, 2=Sell Short
        if action == 1:
            if direction_prefs['long'] > 0.6 or market_bias in ['bullish', 'correction_buy']:
                trade_occurred = self._execute_long_trade(current_price, total_positions)
            else:
                trade_occurred = self._execute_long_trade(current_price, total_positions, size_multiplier=0.7)
                
        elif action == 2:
            if direction_prefs['short'] > 0.6 or market_bias in ['bearish', 'correction_sell']:
                trade_occurred = self._execute_short_trade(current_price, total_positions)
            else:
                trade_occurred = self._execute_short_trade(current_price, total_positions, size_multiplier=0.7)

        # Close positions logic
        if not trade_occurred:
            trade_occurred = self._close_positions_if_needed(current_price, action, market_bias)

        self.current_step += 1

        # Calculate reward with enhanced function
        reward = self.reward_function.calculate_enhanced_reward(
            self, action, trade_occurred, self.current_step - 1)

        # Update history
        portfolio_after = self.get_portfolio_value()
        self.portfolio_value_history.append(portfolio_after)
        self._update_benchmark()

        self.done = self.current_step >= len(self.data) - 1
        info = {'trades': self.trades, 'market_bias': market_bias}

        return self.get_state(), reward, self.done, info

    def _process_expert_position_management(self, current_price):
        """Expert position management - UNIFIED logic for both directions"""
        
        # Process positions with same logic - only direction differs
        for position_type, positions in [('LONG', self.long_positions), ('SHORT', self.short_positions)]:
            for i, position in enumerate(positions):
                # Check for stop loss adjustment (same logic, different conditions)
                should_adjust, new_stop = self.risk_manager.should_adjust_stop_loss(self, current_price, position)
                
                if should_adjust and not position.get('stop_adjusted', False):
                    position['stop_loss_price'] = new_stop
                    position['stop_adjusted'] = True
                    position['original_stop'] = position.get('original_stop', position['stop_loss_price'])

                # Check for partial exit at breakeven (same logic)
                elif self.risk_manager.should_exit_partial_position(self, current_price, position):
                    if not position.get('partial_exit_done', False):
                        self._execute_partial_exit(i, current_price, position_type, 0.25)
                        position['partial_exit_done'] = True

                # Check for moving stop to breakeven (same logic, different multiplier)
                elif self.risk_manager.should_move_to_breakeven(self, position):
                    multiplier = 0.999 if position_type == 'LONG' else 1.001
                    position['stop_loss_price'] = position['entry_price'] * multiplier
                    position['moved_to_breakeven'] = True
                    holding_days = self.current_step - position.get('entry_step', 0)
                    print(f"{position_type} Position {i}: Stop moved to breakeven after {holding_days} days")


    def _execute_partial_exit(self, position_index, current_price, position_type, exit_percentage):
        """FIXED: Partial exit with correct short logic"""
        if position_type == 'LONG':
            position = self.long_positions[position_index]
            shares_to_exit = int(position['shares'] * exit_percentage)
            if shares_to_exit > 0:
                proceeds = shares_to_exit * current_price * (1 - self.transaction_cost)
                profit = (current_price - position['entry_price']) * shares_to_exit - \
                        (shares_to_exit * current_price * self.transaction_cost)
                self.cash += proceeds
                position['shares'] -= shares_to_exit
                self._record_trade('PARTIAL_SELL', shares_to_exit, current_price, profit,
                                position['entry_price'], position['entry_step'])

        elif position_type == 'SHORT':
            position = self.short_positions[position_index]
            shares_to_exit = int(position['shares'] * exit_percentage)
            if shares_to_exit > 0:
                # **FIXED: Partial cover for short position**
                cost_to_cover = shares_to_exit * current_price * (1 + self.transaction_cost)
                profit = (position['entry_price'] - current_price) * shares_to_exit - \
                        (shares_to_exit * current_price * self.transaction_cost)
                self.cash -= cost_to_cover  # SUBTRACT cash for covering
                position['shares'] -= shares_to_exit
                self._record_trade('PARTIAL_COVER', shares_to_exit, current_price, profit,
                                position['entry_price'], position['entry_step'])

    def _process_stop_losses(self, current_price):
        """FIXED: Stop loss processing with minimum holding period"""
        
        # Process long position stop losses
        long_positions_to_remove = []
        for i, position in enumerate(self.long_positions):
            holding_period = self.current_step - position.get('entry_step', self.current_step)
            
            # **FIXED: Only trigger stops after minimum holding period**
            if (holding_period >= self.config.MIN_HOLDING_PERIOD and 
                current_price <= position['stop_loss_price']):
                self._close_position(i, current_price, 'LONG', 'STOP-LOSS')
                long_positions_to_remove.append(i)

        for i in sorted(long_positions_to_remove, reverse=True):
            self.long_positions.pop(i)

        # Process short position stop losses
        short_positions_to_remove = []
        for i, position in enumerate(self.short_positions):
            holding_period = self.current_step - position.get('entry_step', self.current_step)
            
            # **FIXED: Only trigger stops after minimum holding period**
            if (holding_period >= self.config.MIN_HOLDING_PERIOD and 
                current_price >= position['stop_loss_price']):
                self._close_position(i, current_price, 'SHORT', 'STOP-LOSS')
                short_positions_to_remove.append(i)

        for i in sorted(short_positions_to_remove, reverse=True):
            self.short_positions.pop(i)

    def _execute_long_trade(self, current_price, total_positions, size_multiplier=1.0):
        """Execute long trade with expert position management"""
        if total_positions < self.max_positions:
            stop_loss_price = self.risk_manager.calculate_dynamic_stop_loss(self, current_price, 'long')
            shares_to_buy = self.risk_manager.calculate_position_size(
                self, current_price, stop_loss_price, 'long')
            shares_to_buy = int(shares_to_buy * size_multiplier)

            if shares_to_buy > 0:
                total_cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    
                    position = {
                        'shares': shares_to_buy,
                        'entry_price': current_price,
                        'entry_step': self.current_step,
                        'stop_loss_price': stop_loss_price,
                        'position_type': 'LONG',
                        'original_stop': stop_loss_price,
                        'stop_adjusted': False,
                        'partial_exit_done': False,
                        'moved_to_breakeven': False
                    }
                    
                    self.long_positions.append(position)
                    self._record_trade('BUY', shares_to_buy, current_price, 0, 
                                     current_price, self.current_step)
                    return True
        return False

    def _execute_short_trade(self, current_price, total_positions, size_multiplier=1.0):
        """FIXED: Correct short trade execution"""
        if total_positions < self.max_positions:
            stop_loss_price = self.risk_manager.calculate_dynamic_stop_loss(self, current_price, 'short')
            shares_to_short = self.risk_manager.calculate_position_size(
                self, current_price, stop_loss_price, 'short')
            shares_to_short = int(shares_to_short * size_multiplier)

            if shares_to_short > 0:
                # **FIXED: Short selling - receive cash for selling shares we don't own**
                proceeds = shares_to_short * current_price * (1 - self.transaction_cost)
                self.cash += proceeds  # ADD cash (we're selling)

                position = {
                    'shares': shares_to_short,
                    'entry_price': current_price,  # Price we sold at
                    'entry_step': self.current_step,
                    'stop_loss_price': stop_loss_price,  # Above entry price
                    'position_type': 'SHORT',
                    'original_stop': stop_loss_price,
                    'stop_adjusted': False,
                    'partial_exit_done': False,
                    'moved_to_breakeven': False
                }

                self.short_positions.append(position)
                self._record_trade('SELL_SHORT', shares_to_short, current_price, 0,
                                current_price, self.current_step)
                return True
        return False

    def _close_positions_if_needed(self, current_price, action, market_bias):
        """Close positions based on market conditions - UNIFIED logic"""
        trade_occurred = False

        if action == 0:  # Hold action
            # Close positions against strong bias (only unprotected ones)
            if market_bias == 'bearish' and self.long_positions:
                for i, position in enumerate(self.long_positions):
                    if not position.get('stop_adjusted', False):
                        self._close_position(i, current_price, 'LONG')
                        self.long_positions.pop(i)
                        trade_occurred = True
                        break

            elif market_bias == 'bullish' and self.short_positions:
                for i, position in enumerate(self.short_positions):
                    if not position.get('stop_adjusted', False):
                        self._close_position(i, current_price, 'SHORT')
                        self.short_positions.pop(i)
                        trade_occurred = True
                        break

            # In ranging markets, close oldest positions
            elif market_bias == 'ranging':
                total_pos = len(self.long_positions + self.short_positions)
                if total_pos > self.max_positions // 2:
                    if self.long_positions and len(self.long_positions) >= len(self.short_positions):
                        self._close_position(0, current_price, 'LONG')
                        self.long_positions.pop(0)
                        trade_occurred = True
                    elif self.short_positions:
                        self._close_position(0, current_price, 'SHORT')
                        self.short_positions.pop(0)
                        trade_occurred = True

        return trade_occurred

    def _close_position(self, position_index, current_price, position_type, trade_type=None):
        """FIXED: Correct profit calculation for both long and short"""
        if position_type == 'LONG':
            position = self.long_positions[position_index]
            # Long: Buy low, sell high
            proceeds = position['shares'] * current_price * (1 - self.transaction_cost)
            profit = (current_price - position['entry_price']) * position['shares'] - \
                    (position['shares'] * current_price * self.transaction_cost)
            self.cash += proceeds
            
            if trade_type is None:
                trade_type = 'SELL'

        elif position_type == 'SHORT':
            position = self.short_positions[position_index]
            # **FIXED: Short: Sell high, buy low to cover**
            cost_to_cover = position['shares'] * current_price * (1 + self.transaction_cost)
            profit = (position['entry_price'] - current_price) * position['shares'] - \
                    (position['shares'] * current_price * self.transaction_cost)
            self.cash -= cost_to_cover  # SUBTRACT cash (we're buying to cover)
            
            if trade_type is None:
                trade_type = 'COVER'

        self._record_trade(trade_type, position['shares'], current_price, profit,
                        position['entry_price'], position['entry_step'])
    
    def _record_trade(self, trade_type, shares, price, profit, entry_price, entry_step):
        """Record trade with enhanced information"""
        self.trades.append({
            'step': self.current_step,
            'type': trade_type,
            'shares': shares,
            'price': price,
            'profit': profit,
            'entry_price': entry_price,
            'entry_step': entry_step
        })

    def _update_benchmark(self):
        """Fixed benchmark calculation with proper synchronization"""
        if self.current_step < len(self.data):
            current_price = self.data.iloc[self.current_step]['close']
            
            # Initialize with proper starting point
            if len(self.benchmark_values) == 0:
                start_price = self.data.iloc[40]['close']  # Match environment start
                benchmark_shares = self.initial_cash / start_price
                self.benchmark_values.append(self.initial_cash)
            else:
                # Calculate current benchmark value
                start_price = self.data.iloc[40]['close']
                benchmark_shares = self.initial_cash / start_price
                benchmark_value = benchmark_shares * current_price
                self.benchmark_values.append(benchmark_value)
            
            # **FIX: Ensure exact length matching with portfolio history**
            # Truncate benchmark to match portfolio length exactly
            if len(self.benchmark_values) > len(self.portfolio_value_history):
                self.benchmark_values = self.benchmark_values[:len(self.portfolio_value_history)]
            elif len(self.benchmark_values) < len(self.portfolio_value_history):
                # If benchmark is shorter, pad with last value
                while len(self.benchmark_values) < len(self.portfolio_value_history):
                    self.benchmark_values.append(self.benchmark_values[-1])


    @property
    def positions(self):
        """Compatibility property"""
        return self.long_positions + self.short_positions

    @positions.setter
    def positions(self, value):
        """Setter for positions property"""
        if not value:
            self.long_positions = []
            self.short_positions = []
        else:
            # Split positions by type if needed
            self.long_positions = [pos for pos in value if pos.get('position_type') == 'LONG']
            self.short_positions = [pos for pos in value if pos.get('position_type') == 'SHORT']

    # Helper methods for state calculation
    def _calculate_momentum(self):
        if self.current_step >= 5:
            current_price = self.data.iloc[self.current_step]['close']
            past_price = self.data.iloc[self.current_step-5]['close']
            return (current_price - past_price) / past_price
        return 0

    def _calculate_volatility(self):
        if self.current_step >= 10:
            recent_prices = self.data.iloc[max(0, self.current_step-10):self.current_step+1]['close']
            return np.std(recent_prices.pct_change().dropna()) if len(recent_prices) > 1 else 0
        return 0
