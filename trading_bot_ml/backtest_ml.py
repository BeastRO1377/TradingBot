import pandas as pd
import numpy as np
from inference import predict_signal  # <-- from previous module
import joblib

# Load feature file
df = pd.read_csv('prepared_features.csv')
scaler = joblib.load('scaler.pkl')

# Define feature columns
feature_cols = [
    'rsi', 'macd', 'macd_signal', 'sma_20',
    'sma_50', 'bb_high', 'bb_low', 'stoch'
]

# Simple backtest parameters
cash = 10000
position = 0
trade_log = []

for idx, row in df.iterrows():
    features = row[feature_cols].values
    signal = predict_signal(features)

    price = row['close']

    if signal == 'BUY' and cash >= price:
        position += 1
        cash -= price
        trade_log.append(('BUY', price))
    elif signal == 'SELL' and position > 0:
        position -= 1
        cash += price
        trade_log.append(('SELL', price))

# Close remaining position at last price
if position > 0:
    cash += position * df.iloc[-1]['close']
    trade_log.append(('CLOSE', df.iloc[-1]['close']))

print(f"Final portfolio value: ${cash:.2f}")
print("Trade Log:")
for trade in trade_log:
    print(trade)