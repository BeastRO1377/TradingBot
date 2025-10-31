import pandas as pd
import numpy as np
import ta as ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

# Load raw data
df = pd.read_csv('historical_data.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp')

# Basic indicators
df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
df['macd'] = MACD(close=df['close']).macd()
df['macd_signal'] = MACD(close=df['close']).macd_signal()
df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()

# Bollinger Bands
bb = BollingerBands(close=df['close'], window=20)
df['bb_high'] = bb.bollinger_hband()
df['bb_low'] = bb.bollinger_lband()

# Stochastic Oscillator
so = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
df['stoch'] = so.stoch()

# Moving average crossover signal (example target)
df['target'] = np.where(df['sma_20'] > df['sma_50'], 2, 0)  # 2=BUY, 0=SELL, no HOLD for now

# Drop NaN rows (due to rolling windows)
df = df.dropna().reset_index(drop=True)

# Save feature file for training
df.to_csv('prepared_features.csv', index=False)
print('Feature file saved as prepared_features.csv')