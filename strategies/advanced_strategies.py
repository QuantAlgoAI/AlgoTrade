import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import joblib
import os
from .base_strategy import BaseStrategy
from datetime import datetime
import logging

class MLStrategy:
    def __init__(self, lookback_period=20, prediction_threshold=0.6):
        self.lookback_period = lookback_period
        self.prediction_threshold = prediction_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'models/ml_strategy.joblib'
        self.scaler_path = 'models/scaler.joblib'
        
        # Load existing model if available
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def prepare_features(self, df):
        """Prepare technical indicators as features for ML model"""
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']/df['close'].shift(1))
        
        # Trend indicators
        sma = SMAIndicator(close=df['close'], window=20)
        ema = EMAIndicator(close=df['close'], window=20)
        df['sma'] = sma.sma_indicator()
        df['ema'] = ema.ema_indicator()
        
        # Momentum indicators
        rsi = RSIIndicator(close=df['close'])
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['rsi'] = rsi.rsi()
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volatility indicators
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        # Volume-based features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        
        # Target variable (1 if price goes up in next period, 0 if down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df
    
    def train(self, df):
        """Train the ML model on historical data"""
        df = self.prepare_features(df)
        df = df.dropna()
        
        # Prepare features and target
        feature_columns = ['returns', 'log_returns', 'sma', 'ema', 'rsi', 
                          'stoch_k', 'stoch_d', 'bb_high', 'bb_low', 'bb_mid',
                          'volume_ma', 'volume_std']
        
        X = df[feature_columns]
        y = df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
    
    def predict(self, df):
        """Generate trading signals using ML model"""
        df = self.prepare_features(df)
        
        # Prepare features
        feature_columns = ['returns', 'log_returns', 'sma', 'ema', 'rsi', 
                          'stoch_k', 'stoch_d', 'bb_high', 'bb_low', 'bb_mid',
                          'volume_ma', 'volume_std']
        
        X = df[feature_columns].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Get prediction probabilities
        proba = self.model.predict_proba(X_scaled)[0]
        
        # Generate signal based on prediction threshold
        if proba[1] > self.prediction_threshold:
            return 1  # Buy signal
        elif proba[0] > self.prediction_threshold:
            return -1  # Sell signal
        return 0  # No signal

class AdaptiveStrategy:
    def __init__(self, base_strategy, volatility_threshold=0.02):
        self.base_strategy = base_strategy
        self.volatility_threshold = volatility_threshold
        self.position_size = 1.0
    
    def calculate_volatility(self, df, window=20):
        """Calculate rolling volatility"""
        returns = df['close'].pct_change()
        return returns.rolling(window=window).std()
    
    def adjust_position_size(self, volatility):
        """Adjust position size based on volatility"""
        if volatility > self.volatility_threshold:
            return 0.5  # Reduce position size in high volatility
        return 1.0
    
    def generate_signal(self, df):
        """Generate trading signal with adaptive position sizing"""
        # Get base strategy signal
        base_signal = self.base_strategy.predict(df)
        
        # Calculate current volatility
        current_volatility = self.calculate_volatility(df).iloc[-1]
        
        # Adjust position size
        self.position_size = self.adjust_position_size(current_volatility)
        
        # Return signal with adjusted position size
        return base_signal * self.position_size

class PortfolioStrategy:
    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        self.weights = weights if weights else [1/len(strategies)] * len(strategies)
    
    def generate_signal(self, df):
        """Generate combined trading signal from multiple strategies"""
        signals = []
        for strategy in self.strategies:
            signal = strategy.generate_signal(df)
            signals.append(signal)
        
        # Weighted average of signals
        weighted_signal = np.average(signals, weights=self.weights)
        
        # Normalize to -1, 0, 1
        if weighted_signal > 0.3:
            return 1
        elif weighted_signal < -0.3:
            return -1
        return 0 

class HighWinRateStrategy(BaseStrategy):
    def __init__(self, contract_hub, account_balance=100000):
        super().__init__(contract_hub, account_balance)
        self.data = pd.DataFrame()
        # Technical Indicators
        self.fast_ema_period = 9
        self.slow_ema_period = 21
        self.rsi_period = 14
        self.volume_ma_period = 20
        self.oi_ma_period = 20
        self.atr_period = 14
        # Trade Management
        self.trade_state = 'IDLE'
        self.current_trade = None
        self.last_trade_time = None
        self.trading_start_time = datetime.strptime('09:15', '%H:%M').time()
        self.trading_end_time = datetime.strptime('15:15', '%H:%M').time()
        self.max_trades_per_day = 2
        self.daily_loss_cap = 2000
        self.daily_pnl = 0
        self.trades_today = 0
        # Risk Management
        self.trailing_sl_activated = False
        self.trailing_sl_percentage = 0.20
        self.trailing_sl_distance = 0.10
        self.partial_exit_percentage = 0.50
        self.partial_exit_target = 0.20
        # Greeks thresholds
        self.max_delta = 0.7
        self.min_delta = 0.3
        self.max_theta = -0.1
        self.max_iv = 0.5
        # Volume and OI thresholds (relaxed for more trades)
        self.min_volume_increase = 1.01  # was 1.1
        self.min_oi_increase = 1.01      # was 1.05
        # Market Context
        self.market_regime = "UNKNOWN"
        self.volatility_threshold = 0.02
        self.support_levels = []
        self.resistance_levels = []

    def calculate_rsi(self, data, period=14):
        delta = data['ltp'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_volume_ma(self, data, period=20):
        return data['volume'].rolling(window=period).mean()

    def calculate_oi_ma(self, data, period=20):
        return data['oi'].rolling(window=period).mean()

    def calculate_atr(self, data, period=14):
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_vwap(self, data):
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        volume_price = typical_price * data['volume']
        cumulative_volume = data['volume'].cumsum()
        cumulative_volume_price = volume_price.cumsum()
        return cumulative_volume_price / cumulative_volume

    def find_support_resistance(self, data, window=20, threshold=0.02):
        highs = data['high'].rolling(window=window, center=True).max()
        lows = data['low'].rolling(window=window, center=True).min()
        resistance = highs[highs == data['high']]
        support = lows[lows == data['low']]
        self.resistance_levels = self.group_price_levels(resistance, threshold)
        self.support_levels = self.group_price_levels(support, threshold)

    def group_price_levels(self, levels, threshold):
        if len(levels) == 0:
            return []
        grouped = []
        current_group = [levels.iloc[0]]
        for price in levels.iloc[1:]:
            if abs(price - current_group[-1]) / current_group[-1] <= threshold:
                current_group.append(price)
            else:
                grouped.append(sum(current_group) / len(current_group))
                current_group = [price]
        if current_group:
            grouped.append(sum(current_group) / len(current_group))
        return grouped

    def analyze_market_context(self):
        if len(self.data) < self.atr_period:
            return "UNKNOWN"
        atr = self.calculate_atr(self.data, self.atr_period)
        price_range = self.data['high'].max() - self.data['low'].min()
        volatility_ratio = atr.iloc[-1] / price_range
        if volatility_ratio > self.volatility_threshold:
            return "TRENDING"
        else:
            return "RANGING"

    def update_data(self, tick_data):
        try:
            tick_df = pd.DataFrame([{
                'timestamp': datetime.fromtimestamp(int(tick_data['exchange_timestamp']) / 1000),
                'ltp': float(tick_data['last_traded_price']) / 100,
                'high': float(tick_data['high_price_of_the_day']) / 100,
                'low': float(tick_data['low_price_of_the_day']) / 100,
                'close': float(tick_data['closed_price']) / 100,
                'volume': tick_data['volume_trade_for_the_day'],
                'oi': tick_data['open_interest'],
                'oi_change': tick_data.get('open_interest_change_percentage', 0),
                'best_bid': float(tick_data['best_5_buy_data'][0]['price']) / 100 if tick_data.get('best_5_buy_data') else None,
                'best_ask': float(tick_data['best_5_sell_data'][0]['price']) / 100 if tick_data.get('best_5_sell_data') else None,
                'total_buy_qty': tick_data['total_buy_quantity'],
                'total_sell_qty': tick_data['total_sell_quantity']
            }])
            self.data = pd.concat([self.data, tick_df], ignore_index=True)
            if len(self.data) > 100:
                self.data = self.data.tail(100)
            if len(self.data) >= self.slow_ema_period:
                self.data['fast_ema'] = self.data['ltp'].ewm(span=self.fast_ema_period, adjust=False).mean()
                self.data['slow_ema'] = self.data['ltp'].ewm(span=self.slow_ema_period, adjust=False).mean()
                self.data['rsi'] = self.calculate_rsi(self.data, self.rsi_period)
                self.data['volume_ma'] = self.calculate_volume_ma(self.data, self.volume_ma_period)
                self.data['oi_ma'] = self.calculate_oi_ma(self.data, self.oi_ma_period)
                self.data['vwap'] = self.calculate_vwap(self.data)
                self.market_regime = self.analyze_market_context()
                self.find_support_resistance(self.data)
                self.generate_signals()
        except Exception as e:
            logging.error(f"Error in update_data: {str(e)}")

    def generate_signals(self):
        if len(self.data) < self.slow_ema_period:
            return pd.DataFrame()
        self.data['fast_ema'] = self.data['ltp'].ewm(span=self.fast_ema_period, adjust=False).mean()
        self.data['slow_ema'] = self.data['ltp'].ewm(span=self.slow_ema_period, adjust=False).mean()
        self.data['rsi'] = self.calculate_rsi(self.data, self.rsi_period)
        self.data['volume_ma'] = self.calculate_volume_ma(self.data, self.volume_ma_period)
        self.data['oi_ma'] = self.calculate_oi_ma(self.data, self.oi_ma_period)
        self.data['vwap'] = self.calculate_vwap(self.data)
        self.data['final_signal'] = 0
        self.data['rsi_signal'] = 0
        self.data['macd_signal'] = 0
        self.data['bb_signal'] = 0
        self.data['volume_signal'] = 0
        self.data['oi_signal'] = 0
        self.data['iv_signal'] = 0
        self.data['greeks_signal'] = 0
        current_fast = self.data['fast_ema'].iloc[-1]
        current_slow = self.data['slow_ema'].iloc[-1]
        prev_fast = self.data['fast_ema'].iloc[-2]
        prev_slow = self.data['slow_ema'].iloc[-2]
        current_rsi = self.data['rsi'].iloc[-1]
        current_volume = self.data['volume'].iloc[-1]
        current_volume_ma = self.data['volume_ma'].iloc[-1]
        current_oi = self.data['oi'].iloc[-1]
        current_oi_ma = self.data['oi_ma'].iloc[-1]
        current_price = self.data['ltp'].iloc[-1]
        current_vwap = self.data['vwap'].iloc[-1]
        if (prev_fast <= prev_slow and current_fast > current_slow and current_rsi < 70 and current_volume > current_volume_ma * self.min_volume_increase and current_oi > current_oi_ma * self.min_oi_increase and current_price > current_vwap):
            self.data.loc[self.data.index[-1], 'final_signal'] = 1
            self.data.loc[self.data.index[-1], 'rsi_signal'] = 1
            self.data.loc[self.data.index[-1], 'macd_signal'] = 1
            self.data.loc[self.data.index[-1], 'volume_signal'] = 1
            self.data.loc[self.data.index[-1], 'oi_signal'] = 1
        elif (prev_fast >= prev_slow and current_fast < current_slow and current_rsi > 30 and current_volume > current_volume_ma * self.min_volume_increase and current_oi > current_oi_ma * self.min_oi_increase and current_price < current_vwap):
            self.data.loc[self.data.index[-1], 'final_signal'] = -1
            self.data.loc[self.data.index[-1], 'rsi_signal'] = -1
            self.data.loc[self.data.index[-1], 'macd_signal'] = -1
            self.data.loc[self.data.index[-1], 'volume_signal'] = 1
            self.data.loc[self.data.index[-1], 'oi_signal'] = 1
        return self.data

    def should_exit_trade(self, current_price, entry_price):
        if not self.current_trade:
            return False, None
        profit_pct = (current_price - entry_price) / entry_price
        if (profit_pct >= self.partial_exit_target and not self.current_trade.get('partial_exit_taken', False)):
            return True, "PARTIAL_EXIT"
        if self.current_trade.get('trailing_sl_activated', False):
            if current_price <= self.current_trade['trailing_sl']:
                return True, "TRAILING_SL"
        if current_price <= self.current_trade['stop_loss']:
            return True, "STOP_LOSS"
        if current_price >= self.current_trade['target']:
            return True, "TARGET"
        return False, None

    # ... (copy any other methods needed from AlgoTrade.py) 