#!/usr/bin/env python
# coding: utf-8

from Inclusion import *

# --------------- GATHER DATA ---------------
def get_data(start_date, end_date, interval, output_type, window_sample):

    # Maximum lookback period to set up indicators
    LOOKBACK_PERIOD = 21

    def date_disposition(date, disposition):
        # Get NYSE trading calendar
        nyse = mcal.get_calendar('NYSE')
        
        # Get holidays
        holidays = nyse.holidays().holidays
        
        # Define a custom business day, excluding weekends + NYSE holidays
        nyse_bd = CustomBusinessDay(holidays=holidays)
        
        # Find date from trading day disposition
        date = date + (disposition * nyse_bd)
    
        return date

    # Get calculated start and end date
    start_date = date_disposition(start_date, -LOOKBACK_PERIOD)
    end_date = date_disposition(end_date, window_sample)
    
    # Download gold futures data
    gold_data = yf.Ticker("GC=F")
    gold_data = gold_data.history(start=start_date, end=end_date, interval = interval)
    gold_data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
    gold_data = gold_data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    def sma_per_timeperiod(gold_data, timeperiod, output_type):

        # Variable timeperiods
        SMA_Out = 'SMA_' + str(timeperiod)
        SMA_Diff = 'SMA_Diff_' + str(timeperiod)
        
        # SMA
        gold_data[SMA_Out] = SMA(gold_data, timeperiod=timeperiod)
        
        if output_type == 1:
            gold_data[SMA_Diff] = (gold_data[SMA_Out] < gold_data['close']).astype(int)
            
        elif output_type == 0 or output_type == 2:
            # SMA_Diff (Percentage difference of SMA to close +- 100%)
            gold_data[SMA_Diff] = ((gold_data['close'] / gold_data[SMA_Out]) - 1).clip(-1,1)
            gold_data[SMA_Diff] = (gold_data[SMA_Diff] + 1) /2

        gold_data.drop(columns=[SMA_Out], inplace = True)
        gold_data = gold_data.copy()

        return gold_data
        
    def ema_per_timeperiod(gold_data, timeperiod, output_type):

        # Variable timeperiods
        EMA_Out = 'EMA_' + str(timeperiod) 
        EMA_Diff = 'EMA_Diff_' + str(timeperiod)
        
        # EMA
        gold_data[EMA_Out] = EMA(gold_data, timeperiod=timeperiod)
        
        if output_type == 1:
            gold_data[EMA_Diff] = (gold_data[EMA_Out] < gold_data['close']).astype(int)
            
        elif output_type == 0 or output_type == 2:
            # EMA_Diff (Percentage difference of EMA to close +- 100%)
            gold_data[EMA_Diff] = ((gold_data['close'] / gold_data[EMA_Out]) - 1).clip(-1,1)
            gold_data[EMA_Diff] = (gold_data[EMA_Diff] + 1) /2
    
        gold_data.drop(columns=[EMA_Out], inplace = True)
        gold_data = gold_data.copy()

        return gold_data

    def rsi_per_timeperiod(gold_data, timeperiod, output_type):

        # Variable timeperiods
        RSI_Out = 'RSI_' + str(timeperiod) 
        
        # RSI
        gold_data[RSI_Out] = RSI(gold_data, timeperiod=timeperiod)
            
        if output_type == 1 or output_type == 2:
            gold_data[RSI_Out] = np.select( [gold_data[RSI_Out] > 70, gold_data[RSI_Out] < 30], [0, 1], default=0.5)
        else:
            gold_data[RSI_Out] = gold_data[RSI_Out] / 100
            
        return gold_data

    def stochrsi_per_timeperiod(gold_data, timeperiod, output_type):

        # Variable timeperiods
        FastK_Out = 'FastK_' + str(timeperiod) 
        FastD_Out = 'FastD_' + str(timeperiod)

        # Stochastic RSI (Ranges between 0 to 100)
        gold_data[[FastK_Out, FastD_Out]] = STOCHRSI(gold_data, timeperiod=timeperiod, fastk_period=3, fastd_period=3, fastd_matype=0)
        if output_type == 1 or output_type == 2:
            gold_data[FastK_Out] = np.select( [gold_data[FastK_Out] > 80, gold_data[FastK_Out] < 20], [0, 1], default=0.5)
            gold_data[FastD_Out] = np.select( [gold_data[FastD_Out] > 80, gold_data[FastD_Out] < 20], [0, 1], default=0.5)
        else:
            gold_data[FastK_Out] = gold_data[FastK_Out] / 100
            gold_data[FastD_Out] = gold_data[FastD_Out] / 100
            
        return gold_data
    
    def adx_per_timeperiod(gold_data, timeperiod, output_type):

        # Variable timeperiods
        ADX_Out = 'ADX_' + str(timeperiod) 
        
        # ADX
        gold_data[ADX_Out] = ADX(gold_data, timeperiod=timeperiod)
            
        if output_type == 1 or output_type == 2:
            gold_data[ADX_Out] = np.select( [gold_data[ADX_Out] > 70, gold_data[ADX_Out] < 30], [0, 1], default=0.5)
        else:
            gold_data[ADX_Out] = gold_data[ADX_Out] / 100
            
        return gold_data

    def bbands_per_timeperiod(gold_data, timeperiod, output_type):

        # Variable timeperiods
        BBand_Diff = 'BBand_Diff_' + str(timeperiod) 

        # Bollinger Bands (Interquartile range to normalise +- 25% Range around each band)
        gold_data[['BBand_Upper', 'SMA', 'BBand_Lower']] = BBANDS(gold_data, timeperiod=timeperiod, nbdevup=2.0, nbdevdn=2.0, matype=0)
        gold_data['BBand_Range_Quartile'] = (gold_data['BBand_Upper'] - gold_data['BBand_Lower']) / 4 # Calculate 25% of BBand Low to High range
        
        # upper and lower bounds (+- 25% Range)
        gold_data['upper_min'] = gold_data['BBand_Upper'] - gold_data['BBand_Range_Quartile']
        gold_data['upper_max'] = gold_data['BBand_Upper'] + gold_data['BBand_Range_Quartile']
        gold_data['lower_min'] = gold_data['BBand_Lower'] - gold_data['BBand_Range_Quartile']
        gold_data['lower_max'] = gold_data['BBand_Lower'] + gold_data['BBand_Range_Quartile']
    
        # BBand_Diff single feature (Promotes longer trades?)
        choices = [
        (0 + ((gold_data['close'] - gold_data['lower_min']) / 
               (gold_data['lower_max'] - gold_data['lower_min'])) * 0.5),
        (0.5 + ((gold_data['close'] - gold_data['upper_min']) / 
             (gold_data['upper_max'] - gold_data['upper_min'])) * 0.5)
        ]
        conditions = [
        gold_data['close'] <= gold_data['lower_max'],
        gold_data['close'] >= gold_data['upper_min']
        ]
        gold_data[BBand_Diff] = np.select(conditions, 
                                            choices, 
                                            default=0.5)
    
        gold_data.drop(columns=['SMA', 'BBand_Upper', 'BBand_Lower', 'BBand_Range_Quartile', 'upper_min', 'upper_max', 'lower_min', 'lower_max'], inplace=True)
            
        return gold_data
    
    # Simple Moving Averages
    timeperiods = [14,21]
    for i in range (0, len(timeperiods)):
        gold_data = sma_per_timeperiod(gold_data, timeperiods[i], output_type)

    # Exponential Moving Averages
    timeperiods = [14,21]
    for i in range (0, len(timeperiods)):
        gold_data = ema_per_timeperiod(gold_data, timeperiods[i], output_type)

    # RSI
    timeperiods = [14,21]
    for i in range (0, len(timeperiods)):
        gold_data = rsi_per_timeperiod(gold_data, timeperiods[i], output_type)

    # ADX
    timeperiods = [14,21]
    for i in range (0, len(timeperiods)):
        gold_data = rsi_per_timeperiod(gold_data, timeperiods[i], output_type)

    # STOCHRSI
    timeperiods = [14,21]
    for i in range (0, len(timeperiods)):
        gold_data = stochrsi_per_timeperiod(gold_data, timeperiods[i], output_type)

    # BBAND
    timeperiods = [14,21]
    for i in range (0, len(timeperiods)):
        gold_data = bbands_per_timeperiod(gold_data, timeperiods[i], output_type)

    # Skip lookback period to set up indicators, rename price cols
    gold_data = gold_data.drop(gold_data.index[:LOOKBACK_PERIOD]).copy()
    gold_data = gold_data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Clip cols to avoid floating point errors
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in gold_data.columns:
        if col not in price_cols:
            gold_data[col] = gold_data[col].clip(0, 1)

    return gold_data

def pad_dummy_day(gold_data):
    
    # Get NYSE trading calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Get holidays for a date range (wide range to cover future dates)
    holidays = nyse.holidays().holidays
    
    # Define a custom business day offset excluding weekends + NYSE holidays
    nyse_bd = CustomBusinessDay(holidays=holidays)
    
    last_date = pd.to_datetime(gold_data.index[-1])
    # Add 1 actual trading days (skipping weekends and NYSE holidays)
    new_date = last_date + 1 * nyse_bd
    
    # Get the last row
    last_row = gold_data.iloc[-1].copy()

    # Create a new row with updated index
    new_row = pd.DataFrame([last_row], index=[new_date])

    # Set Open, High, Low, Close to the same value
    for col in ['Open', 'High', 'Low', 'Close']:
        new_row[col] = new_row['Close']

    # Append the new row to the original DataFrame
    gold_data = pd.concat([gold_data, new_row])

    return gold_data

# ------------ CUSTOM ENVIRIONMENT -------------
def my_process_data(self):
    start = self.frame_bound[0] - self.window_size
    end = (self.frame_bound[1])
    prices = self.df.loc[:, 'Close'].to_numpy()[start:end]

    # Exclude columns and select signal_features
    excluded_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    signal_features = self.df.loc[:, [col for col in self.df.columns if col not in excluded_columns]].to_numpy()[start:end]

    return prices, signal_features

def my_calculate_reward(self, action):
    current_price = self.prices[self._current_tick]
    last_price = self.prices[self._current_tick -1]
    last_trade_price = self.prices[self._last_trade_tick]
    holding_period = max(1, self._current_tick - self._last_trade_tick)
    step_reward = 0

    # 0 Reward if price remained the same (transition out of window in training)
    if last_trade_price == current_price:
        return step_reward
        
    # Increases in size with holding period
    sqrt_holding_penalty = np.sqrt(holding_period)
    exp_holding_penalty = np.exp(0.1 * (holding_period - 1))

    # Determine if a trade is happening
    trade = False
    if (
        (action == Actions.Buy.value and self._position == Positions.Short) or
        (action == Actions.Sell.value and self._position == Positions.Long)
    ):
        trade = True

    # Return from previous day
    if self._position == Positions.Short:
        log_return = np.log(last_price / current_price)
    elif self._position == Positions.Long:
        log_return = np.log(current_price / last_price)

    step_reward += log_return

    # On Trade produce reward based on return and hold time
    if trade:
    
        # Return from trade
        if self._position == Positions.Short:
            trade_log_return = np.log(last_trade_price / current_price)
        elif self._position == Positions.Long:
            trade_log_return = np.log(current_price / last_trade_price)
    
        # Period adjusted returns
        if trade_log_return > 0:
            period_adjusted_return = trade_log_return / sqrt_holding_penalty  # Decreasing positive reward
        elif trade_log_return < 0:
            period_adjusted_return = -abs(trade_log_return * sqrt_holding_penalty) # Increasing negative reward
    
        # Tanh squash period returns to stop extreme values from dominating
        step_reward += np.tanh(period_adjusted_return)

    return step_reward

def my_update_profit(self, action):
    trade = False

    # Determine if a trade is happening
    if (
        (action == Actions.Buy.value and self._position == Positions.Short) or
        (action == Actions.Sell.value and self._position == Positions.Long)
    ):
        trade = True

    if trade or self._truncated:
        exit_trade_price = self.prices[self._current_tick]
        entry_trade_price = self.prices[self._last_trade_tick]

        # Assume 100% of total profit/equity is being risked
        equity = self._total_profit

        # Percentage return
        if self._position == Positions.Long:
            percentage_return = (exit_trade_price - entry_trade_price) / entry_trade_price # Buy
        elif self._position == Positions.Short:
            percentage_return = (entry_trade_price - exit_trade_price) / entry_trade_price # Sell
        else:
            percentage_return = 0

        # Gross return
        gross_return = equity * (1+percentage_return)

        # Mimmicking Trading-212 overnight fees, fx fees, and small slippage
        commission = abs((equity - gross_return) * 0.005)

        if action == Actions.Buy.value:
            commission += abs(0.000139 * (equity) * (self._current_tick - self._last_trade_tick))
        elif action == Actions.Sell.value:   
            commission += abs(0.00011 * (equity) * (self._current_tick - self._last_trade_tick))

        # Entry and exit slippage
        commission += abs(equity * 0.0001)
        commission += abs((equity + gross_return) * 0.0001)

        # Net return
        net_return = gross_return - commission

        # Update total profit
        self._total_profit = net_return

def my_render_all(self, action, title=None):
    window_ticks = np.arange(len(self._position_history))
    plt.plot(self.prices)

    short_ticks = []
    long_ticks = []
    short_swap_ticks = []
    long_swap_ticks = []


    for i, tick in enumerate(window_ticks):

        # Get first and last position
        if i == self.window_size +1 or i == (len(window_ticks)) -1:
            # Initial tick position
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            if self._position_history[i] == Positions.Long:
                long_ticks.append(tick)
                
        # Determine if a trade is happening
        trade = False
        if ( 
            (self._position_history[i] == Positions.Short and self._position_history[i-1] == Positions.Long) or
            (self._position_history[i] == Positions.Long and self._position_history[i-1] == Positions.Short)
        ):
            trade = True

        # Get trade ticks
        if self._position_history[i] == Positions.Short and trade:
            short_swap_ticks.append(tick if tick > 0 else 0)
        if self._position_history[i] == Positions.Long and trade:
            long_swap_ticks.append(tick if tick > 0 else 0)

    # Plot
    plt.plot(short_ticks, self.prices[short_ticks], 'ro', markersize = 4)
    plt.plot(long_ticks, self.prices[long_ticks], 'go', markersize = 4)
    plt.plot(short_swap_ticks, self.prices[short_swap_ticks], 'rv', markersize=5, markeredgecolor='black', markeredgewidth = .5)
    plt.plot(long_swap_ticks, self.prices[long_swap_ticks], 'g^', markersize=5, markeredgecolor='black', markeredgewidth = .5)

    if title:
        plt.title(title)

    plt.suptitle(
        "Total Reward: %.3f" % self._total_reward + ' ~ ' +
        "Total Profit: %.2f" % self._total_profit + ' ~ ' +
        "Total Timeseteps: %d" % self._current_tick + ' ~ ' +
        "Last Trade Tick %d" % self._last_trade_tick + ' ~ ' +
        "Current Position: %s" % self._position_history[i].name + ' ~ ' +
        "Next Action: %s" % str(Actions(action).name)
    )

# Override env methods
class MyStocksEnv(StocksEnv):
    _process_data = my_process_data
    _calculate_reward = my_calculate_reward
    _update_profit = my_update_profit
    render_all = my_render_all

# --------------- TRAINING ---------------
def train_model(model, training_env, training_timesteps, model_save_path, n_evaluations, reward_threshold, verbose):

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold = reward_threshold, verbose = verbose)
    
    # Callback function to save best model
    monitored_env = Monitor(training_env)
    eval_callback = EvalCallback(monitored_env, callback_on_new_best = stop_callback, eval_freq = round(training_timesteps / n_evaluations), best_model_save_path = model_save_path, verbose = verbose)
    
    # Train model
    model.learn(total_timesteps= training_timesteps, progress_bar = True, log_interval = 5, callback = eval_callback)

    return
