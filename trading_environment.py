"""
Enhanced Trading Environment with Regime-Aware State
KEEPS ORIGINAL CASH MANAGEMENT - Only adds regime features to state
"""
import numpy as np


class AdvancedTradingEnvironment:
    def __init__(self, data, feature_columns, scaler, config, lookback_context=40):
        self.data = data.copy()
        self.config = config
        self.initial_cash = float(config.INITIAL_CASH)
        self.transaction_cost = config.TRANSACTION_COST
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.max_positions = config.MAX_POSITIONS
        self.lookback_context = lookback_context
        
        from risk_manager import AdvancedRiskManager
        from reward_function import ImprovedRewardFunction
        
        self.risk_manager = AdvancedRiskManager(config)
        self._prepare_scaled_features()
        self.reward_function = ImprovedRewardFunction()
        
        # Peak tracking for dynamic exits
        self.portfolio_peak = config.INITIAL_CASH
        self.steps_since_peak = 0
        self.peak_drawdown_threshold = 0.05
        
        self.reset()

    def _prepare_scaled_features(self):
        """Pre-scale all features for efficient state retrieval"""
        self.scaled_features = self.scaler.transform(self.data[self.feature_columns])
        
    def reset(self):
        """Reset environment - START TRADING FROM STEP 0"""
        self.cash = self.initial_cash
        self.long_positions = []
        self.short_positions = []
        self.current_step = 0
        self.done = False
        self.trades = []
        self.portfolio_value_history = [self.initial_cash]
        self.benchmark_values = []
        
        # Reset peak tracking
        self.portfolio_peak = self.initial_cash
        self.steps_since_peak = 0
        
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
        """
        Calculate total portfolio value (Equity).
        Equity = (Cash + Long Position Value) - (Short Position Liability)
        """
        if self.current_step >= len(self.data):
            current_price = self.data.iloc[-1]['close']
        else:
            current_price = self.data.iloc[self.current_step]['close']
        
        long_value = sum(pos['shares'] * current_price for pos in self.long_positions)
        short_liability = sum(pos['shares'] * current_price for pos in self.short_positions)
        
        return self.cash + long_value - short_liability

    def _get_true_available_cash(self):
        """
        ORIGINAL METHOD - KEPT AS-IS
        Calculates the real available cash by subtracting the worst-case liability
        of all open short positions from the current cash balance.
        """
        # Calculate the total cash needed to buy back all shorted shares at their stop-loss prices
        short_cover_liability_at_stop = sum(
            pos['shares'] * 1.15 * pos['stop_loss_price'] * (1 + self.transaction_cost)
            for pos in self.short_positions
        )

        # The true available cash is the current balance minus the funds reserved to cover shorts
        available_cash = self.cash - short_cover_liability_at_stop
        
        return max(0, available_cash)

    def get_state(self):
        """
        ENHANCED: Get current state with regime features included
        """
        if self.current_step >= len(self.data):
            # Return zero state at end
            base_features = len(self.feature_columns)
            additional_features = 10  # Original additional features
            regime_features = 7  # New regime features
            return np.zeros(base_features + additional_features + regime_features)
        
        # Market features (scaled)
        market_features = self.scaled_features[self.current_step]
        
        # Portfolio state features
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.get_portfolio_value()
        
        position_value_ratio = self.get_total_position_value() / portfolio_value if portfolio_value > 0 else 0
        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 1.0
        position_count_ratio = len(self.positions) / self.max_positions
        
        momentum = self._calculate_momentum()
        volatility = self._calculate_volatility()
        trend = self._calculate_trend()
        
        vol_regime = self.data.iloc[self.current_step].get('vol_regime_numeric', 0)
        trend_alignment = self.data.iloc[self.current_step].get('trend_alignment', 0)
        momentum_strength = self.data.iloc[self.current_step].get('momentum_strength', 0)
        atr_pct = self.data.iloc[self.current_step].get('atr_pct', 0.02)
        
        additional_features = [
            position_value_ratio, cash_ratio, position_count_ratio, momentum,
            volatility, trend, vol_regime, trend_alignment, momentum_strength, atr_pct
        ]
        
        # ENHANCED: Add regime features to state
        regime_features = [
            self.data.iloc[self.current_step].get('vol_regime_low', 0),
            self.data.iloc[self.current_step].get('vol_regime_high', 0),
            self.data.iloc[self.current_step].get('in_uptrend', 0),
            self.data.iloc[self.current_step].get('in_downtrend', 0),
            self.data.iloc[self.current_step].get('favorable_long_regime', 0),
            self.data.iloc[self.current_step].get('favorable_short_regime', 0),
            self.data.iloc[self.current_step].get('regime_uncertainty', 0),
        ]
        
        # Concatenate all features
        state = np.concatenate([market_features, additional_features, regime_features])
        return state

    def _liquidate_positions(self, env):
        """Liquidate remaining positions at end - FIXED for trainer use"""
        if env.current_step >= len(env.data):
            return
            
        last_price = env.data.iloc[-1]['close']
        
        # Liquidate long positions
        for position in list(env.long_positions):
            gross_proceeds = position['shares'] * last_price
            closing_cost = gross_proceeds * env.transaction_cost
            net_proceeds = gross_proceeds - closing_cost
            
            opening_cost = position['entry_price'] * position['shares'] * env.transaction_cost
            profit = (last_price - position['entry_price']) * position['shares'] - opening_cost - closing_cost
            
            env.cash += net_proceeds
            env.trades.append({
                'step': env.current_step, 'type': 'LIQUIDATE_LONG',
                'shares': position['shares'], 'price': last_price,
                'profit': profit, 'entry_price': position['entry_price'],
                'entry_step': position.get('entry_step', env.current_step)
            })
        
        # Cover short positions
        for position in list(env.short_positions):
            gross_cost = position['shares'] * last_price
            closing_cost = gross_cost * env.transaction_cost
            total_cost = gross_cost + closing_cost
            
            opening_cost = position['entry_price'] * position['shares'] * env.transaction_cost
            profit = (position['entry_price'] - last_price) * position['shares'] - opening_cost - closing_cost
            
            if env.cash >= total_cost:
                env.cash -= total_cost
                env.trades.append({
                    'step': env.current_step, 'type': 'LIQUIDATE_SHORT',
                    'shares': position['shares'], 'price': last_price,
                    'profit': profit, 'entry_price': position['entry_price'],
                    'entry_step': position.get('entry_step', env.current_step)
                })
            else:
                print(f"WARNING: Insufficient cash to liquidate short at end")
        
        env.long_positions = []
        env.short_positions = []

    def _check_portfolio_peak_drawdown(self):
        """Check portfolio drawdown - works for BOTH long and short positions"""
        current_portfolio = self.get_portfolio_value()
        
        # Update peak
        if current_portfolio > self.portfolio_peak:
            self.portfolio_peak = current_portfolio
            self.steps_since_peak = 0
        else:
            self.steps_since_peak += 1
        
        # Calculate drawdown from peak
        drawdown_from_peak = (self.portfolio_peak - current_portfolio) / self.portfolio_peak
        
        # Emergency exit conditions
        if drawdown_from_peak >= self.peak_drawdown_threshold:
            return True
        
        if self.steps_since_peak > 20 and drawdown_from_peak > 0.03:
            return True
        
        # Check individual position unrealized losses
        current_price = self.data.iloc[self.current_step]['close']
        
        for position in self.long_positions:
            unrealized_loss_pct = (position['entry_price'] - current_price) / position['entry_price']
            if unrealized_loss_pct > 0.15:
                return True
        
        for position in self.short_positions:
            unrealized_loss_pct = (current_price - position['entry_price']) / position['entry_price']
            if unrealized_loss_pct > 0.15:
                return True
                
        return False

    def _liquidate_all_positions_emergency(self, current_price, reason='EMERGENCY_EXIT'):
        """Emergency liquidation of all positions"""
        liquidated_count = 0
        total_recovered = 0
        
        # Liquidate long positions
        for position in list(self.long_positions):
            gross_proceeds = position['shares'] * current_price
            closing_cost = gross_proceeds * self.transaction_cost
            net_proceeds = gross_proceeds - closing_cost
            
            opening_cost = position['entry_price'] * position['shares'] * self.transaction_cost
            profit = (current_price - position['entry_price']) * position['shares'] - opening_cost - closing_cost
            
            self.cash += net_proceeds
            total_recovered += net_proceeds
            
            self._record_trade('EMERGENCY_SELL_LONG', position['shares'], current_price, profit, 
                            position['entry_price'], position['entry_step'])
            liquidated_count += 1
        
        self.long_positions = []
        
        # Cover short positions
        for position in list(self.short_positions):
            gross_cost = position['shares'] * current_price
            closing_cost = gross_cost * self.transaction_cost
            total_cost = gross_cost + closing_cost
            
            if self.cash >= total_cost:
                opening_cost = position['entry_price'] * position['shares'] * self.transaction_cost
                profit = (position['entry_price'] - current_price) * position['shares'] - opening_cost - closing_cost
                
                self.cash -= total_cost
                
                self._record_trade('EMERGENCY_COVER_SHORT', position['shares'], current_price, profit, 
                                position['entry_price'], position['entry_step'])
                liquidated_count += 1
        
        self.short_positions = []
        
        if liquidated_count > 0 and not getattr(self, 'training', True):
            print(f"    Emergency liquidation: {liquidated_count} positions, recovered ${total_recovered:,.2f}")
        
        return liquidated_count

    def step(self, action):
        """Main step function"""
        if self.done or self.current_step >= len(self.data) - 1:
            self.done = True
            return self.get_state(), 0, True, {}

        current_price = float(self.data.iloc[self.current_step]['close'])
        portfolio_before = self.get_portfolio_value()
        
        # Process exits
        self._process_enhanced_exits(current_price)
        
        # Execute new trades
        trade_occurred = False
        if action == 1:
            trade_occurred = self._execute_long_trade_no_leverage(current_price, action)
        elif action == 2:
            trade_occurred = self._execute_short_trade_no_leverage(current_price, action)
        
        self.current_step += 1
        portfolio_after = self.get_portfolio_value()
        
        reward = self.reward_function.calculate_enhanced_reward(
            self, action, trade_occurred, self.current_step - 1
        )
        
        self.portfolio_value_history.append(portfolio_after)
        self._update_benchmark()

        self.done = self.current_step >= len(self.data) - 1
        info = {
            'trades': self.trades,
            'portfolio_before': portfolio_before,
            'portfolio_after': portfolio_after
        }

        return self.get_state(), reward, self.done, info

    def _process_enhanced_exits(self, current_price):
        """Enhanced exit logic - SYMMETRICAL for long and short"""
        long_to_remove = []
        
        for i, position in enumerate(self.long_positions):
            if 'peak_price' not in position:
                position['peak_price'] = max(position['entry_price'], current_price)
                position['peak_profit_pct'] = 0
            
            if current_price > position['peak_price']:
                position['peak_price'] = current_price
                position['peak_profit_pct'] = (current_price - position['entry_price']) / position['entry_price']
            
            current_profit_pct = (current_price - position['entry_price']) / position['entry_price']
            drawdown_from_peak = (position['peak_price'] - current_price) / position['peak_price']
            holding_period = self.current_step - position['entry_step']
            
            should_exit, exit_reason = self._evaluate_exit_rules_long(
                current_profit_pct, drawdown_from_peak, holding_period, 
                position, current_price
            )
            
            if should_exit:
                self._close_long_position(i, current_price, exit_reason)
                long_to_remove.append(i)
        
        for i in sorted(long_to_remove, reverse=True):
            if i < len(self.long_positions):
                self.long_positions.pop(i)
        
        short_to_remove = []
        
        for i, position in enumerate(self.short_positions):
            if 'trough_price' not in position:
                position['trough_price'] = min(position['entry_price'], current_price)
                position['peak_profit_pct'] = 0
            
            if current_price < position['trough_price']:
                position['trough_price'] = current_price
                position['peak_profit_pct'] = (position['entry_price'] - current_price) / position['entry_price']
            
            current_profit_pct = (position['entry_price'] - current_price) / position['entry_price']
            drawdown_from_trough = (current_price - position['trough_price']) / position['trough_price']
            holding_period = self.current_step - position['entry_step']
            
            should_exit, exit_reason = self._evaluate_exit_rules_short(
                current_profit_pct, drawdown_from_trough, holding_period,
                position, current_price
            )
            
            if should_exit:
                self._close_short_position(i, current_price, exit_reason)
                short_to_remove.append(i)
        
        for i in sorted(short_to_remove, reverse=True):
            if i < len(self.short_positions):
                self.short_positions.pop(i)

    def _evaluate_exit_rules_long(self, profit_pct, drawdown, holding_period, position, price):
        """Evaluate exit rules for long positions"""
        
        if profit_pct >= 0.20:
            return True, 'SELL_LONG_PROFIT_TARGET'
        
        if position['peak_profit_pct'] > 0.05 and drawdown >= 0.08:
            return True, 'SELL_LONG_PEAK_DECLINE'
        
        if profit_pct <= -0.10:
            return True, 'SELL_LONG_LOSS_LIMIT'
        
        if price <= position['stop_loss_price']:
            return True, 'SELL_LONG_STOP_LOSS'
        
        if holding_period > 30 and profit_pct < 0.03:
            return True, 'SELL_LONG_TIME_EXIT'
        
        if holding_period > 15 and 0.05 <= profit_pct < 0.10:
            return True, 'SELL_LONG_WEAK_PROFIT'
        
        if holding_period > 20 and profit_pct < -0.05:
            return True, 'SELL_LONG_LOSS_LIMIT'
        
        return False, None

    def _evaluate_exit_rules_short(self, profit_pct, drawdown, holding_period, position, price):
        """Evaluate exit rules for short positions"""
        
        if profit_pct >= 0.20:
            return True, 'COVER_SHORT_PROFIT_TARGET'
        
        if position['peak_profit_pct'] > 0.05 and drawdown >= 0.08:
            return True, 'COVER_SHORT_TROUGH_RISE'
        
        if profit_pct <= -0.10:
            return True, 'COVER_SHORT_LOSS_LIMIT'
        
        if price >= position['stop_loss_price']:
            return True, 'COVER_SHORT_STOP_LOSS'
        
        if holding_period > 30 and profit_pct < 0.03:
            return True, 'COVER_SHORT_TIME_EXIT'
        
        if holding_period > 15 and 0.05 <= profit_pct < 0.10:
            return True, 'COVER_SHORT_WEAK_PROFIT'
        
        if holding_period > 20 and profit_pct < -0.05:
            return True, 'COVER_SHORT_LOSS_LIMIT'
        
        return False, None

    def _execute_long_trade_no_leverage(self, current_price, action):
        """Execute long trade - ORIGINAL METHOD"""
        if self.current_step >= len(self.data):
            return False
        
        max_spendable = self._get_true_available_cash() * 0.95
        
        if max_spendable <= current_price:
            return False
        
        if len(self.long_positions) + len(self.short_positions) >= self.max_positions:
            return False

        stop_loss_price = self.risk_manager.calculate_dynamic_stop_loss(self, current_price, 'long')
        risk_per_share = current_price - stop_loss_price
        
        if risk_per_share <= 0:
            return False

        max_shares_by_cash = int(max_spendable / (current_price * (1 + self.transaction_cost)))
        
        if max_shares_by_cash < 1:
            return False

        shares_to_buy = max_shares_by_cash
        total_cost = shares_to_buy * current_price * (1 + self.transaction_cost)
        
        if total_cost > self._get_true_available_cash():
            shares_to_buy = int(self._get_true_available_cash() * 0.95 / (current_price * (1 + self.transaction_cost)))
            if shares_to_buy < 1:
                return False
            total_cost = shares_to_buy * current_price * (1 + self.transaction_cost)

        self.cash -= total_cost
        
        position = {
            'shares': shares_to_buy, 
            'entry_price': current_price,
            'entry_step': self.current_step, 
            'stop_loss_price': stop_loss_price,
            'position_type': 'LONG',
            'cost_basis': total_cost
        }
        self.long_positions.append(position)
        
        self._record_trade('BUY_LONG', shares_to_buy, current_price, 0, 
                        current_price, self.current_step)
        
        return True

    def _execute_short_trade_no_leverage(self, current_price, action):
        """Execute short trade - ORIGINAL METHOD"""
        if self.current_step >= len(self.data):
            return False

        if len(self.long_positions) + len(self.short_positions) >= self.max_positions:
            return False

        stop_loss_price = self.risk_manager.calculate_dynamic_stop_loss(self, current_price, 'short')
        risk_per_share = stop_loss_price - current_price

        if risk_per_share <= 0:
            return False

        portfolio_equity = self.get_portfolio_value()
        if portfolio_equity <= 0:
            return False
        capital_at_risk = portfolio_equity * self.config.MAX_PORTFOLIO_RISK / self.max_positions
        risk_based_shares = int(capital_at_risk / risk_per_share)

        cost_per_share_at_stop = stop_loss_price * (1 + self.transaction_cost)
        true_available_cash = self._get_true_available_cash()
        
        if cost_per_share_at_stop <= 0:
            return False
            
        cash_based_shares = int(true_available_cash / cost_per_share_at_stop)

        shares_to_short = min(risk_based_shares, cash_based_shares)

        if shares_to_short < 1:
            return False

        gross_proceeds = shares_to_short * current_price
        transaction_cost_amount = gross_proceeds * self.transaction_cost
        net_proceeds = gross_proceeds - transaction_cost_amount
        
        self.cash += net_proceeds

        position = {
            'shares': shares_to_short,
            'entry_price': current_price,
            'entry_step': self.current_step,
            'stop_loss_price': stop_loss_price,
            'position_type': 'SHORT',
            'proceeds_received': net_proceeds
        }
        self.short_positions.append(position)

        self._record_trade('SELL_SHORT', shares_to_short, current_price, 0,
                        current_price, self.current_step)

        return True

    def _calculate_momentum(self):
        """Calculate momentum with adaptive lookback"""
        lookback = min(5, self.current_step)
        if lookback < 1:
            return 0
        
        current_price = self.data.iloc[self.current_step]['close']
        past_price = self.data.iloc[self.current_step - lookback]['close']
        return (current_price - past_price) / past_price

    def _calculate_volatility(self):
        """Calculate volatility with adaptive lookback"""
        lookback = min(10, self.current_step)
        if lookback < 2:
            return 0
        
        recent_prices = self.data.iloc[max(0, self.current_step - lookback):self.current_step + 1]['close']
        if len(recent_prices) < 2:
            return 0
        return np.std(recent_prices.pct_change().dropna())

    def _calculate_trend(self):
        """Calculate trend with adaptive lookback"""
        short_lookback = min(5, self.current_step)
        long_lookback = min(20, self.current_step)
        
        if long_lookback < 2:
            return 0
        
        short_ma = self.data.iloc[max(0, self.current_step - short_lookback):self.current_step + 1]['close'].mean()
        long_ma = self.data.iloc[max(0, self.current_step - long_lookback):self.current_step + 1]['close'].mean()
        return 1 if short_ma > long_ma else -1

    def _close_long_position(self, position_index, current_price, trade_type='SELL_LONG'):
        """Close long position with exact accounting"""
        if position_index >= len(self.long_positions):
            return
        
        position = self.long_positions[position_index]
        
        gross_proceeds = position['shares'] * current_price
        closing_transaction_cost = gross_proceeds * self.transaction_cost
        net_proceeds = gross_proceeds - closing_transaction_cost
        
        cost_basis = position.get('cost_basis', position['shares'] * position['entry_price'] * (1 + self.transaction_cost))
        profit = net_proceeds - cost_basis
        
        self.cash += net_proceeds
        
        self._record_trade(trade_type, position['shares'], current_price, profit, 
                        position['entry_price'], position['entry_step'])

    def _close_short_position(self, position_index, current_price, trade_type='COVER_SHORT'):
        """Close short position with exact accounting"""
        if position_index >= len(self.short_positions):
            return
        
        position = self.short_positions[position_index]
        
        gross_cost = position['shares'] * current_price
        closing_transaction_cost = gross_cost * self.transaction_cost
        total_cost = gross_cost + closing_transaction_cost
        
        if self.cash < total_cost:
            print(f"WARNING: Insufficient cash to cover short position at step {self.current_step} for {position['shares']} shares. Difference: ${total_cost - self.cash:,.2f}")
            return
        
        proceeds_received = position.get('proceeds_received', position['shares'] * position['entry_price'] * (1 - self.transaction_cost))
        
        profit = proceeds_received - total_cost
        
        self.cash -= total_cost
        
        self._record_trade(trade_type, position['shares'], current_price, profit, 
                        position['entry_price'], position['entry_step'])
    
    def _record_trade(self, trade_type, shares, price, profit, entry_price, entry_step):
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
        """Update benchmark starting from step 0"""
        if self.current_step < len(self.data):
            benchmark_price = self.data.iloc[self.current_step]['close']
            initial_benchmark_price = self.data.iloc[0]['close']
            benchmark_shares = self.initial_cash / initial_benchmark_price
            benchmark_value = benchmark_shares * benchmark_price
            self.benchmark_values.append(benchmark_value)

    @property
    def positions(self):
        return self.long_positions + self.short_positions

    @positions.setter
    def positions(self, value):
        if not value:
            self.long_positions = []
            self.short_positions = []
        else:
            self.long_positions = []
            self.short_positions = []