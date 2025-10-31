import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

base_path = '/Users/anatolytamilin/Downloads/TradingBot'

snapshots_path = os.path.join(base_path, 'golden_setup_snapshots.csv')
trades_path = os.path.join(base_path, 'trades_for_training.csv')

# Чтение файлов в DataFrame
snapshots = pd.read_csv(snapshots_path)
trades = pd.read_csv(trades_path)

# Золотой сетап
def optimize_golden_setup():
    snapshots = pd.read_csv('golden_setup_snapshots.csv')
    trades = pd.read_csv('trades_for_training.csv')
    merged_data = pd.merge_asof(
        snapshots.sort_values('timestamp'),
        trades.sort_values('datetime'),
        left_on='timestamp',
        right_on='datetime',
        by=['symbol', 'signal'],
        direction='nearest',
        tolerance=pd.Timedelta('5m')
    )

    def calculate_winrate(df, p0, v0, o0, side):
        if side == 'Buy':
            mask = (df['price_change'] >= p0) & (df['volume_change'] >= v0) & (df['oi_change'] >= o0)
        else:
            mask = (df['price_change'] <= -p0) & (df['volume_change'] >= v0) & (df['oi_change'] >= o0)
        filtered = df[mask]
        if len(filtered) < 10:
            return 0
        return (filtered['pnl_pct'] > 0).mean()

    best_params = {}
    symbols = merged_data['symbol'].unique()
    p_range = np.arange(0.1, 2.1, 0.1)
    v_range = np.arange(50, 201, 10)
    o_range = np.arange(0.1, 2.1, 0.1)

    for symbol in symbols:
        for side in ['Buy', 'Sell']:
            sub_data = merged_data[(merged_data['symbol'] == symbol) & (merged_data['signal'] == side)]
            if len(sub_data) < 30:
                continue
            best_winrate = 0
            best_p0, best_v0, best_o0 = None, None, None
            for p0, v0, o0 in product(p_range, v_range, o_range):
                winrate = calculate_winrate(sub_data, p0, v0, o0, side)
                if winrate > best_winrate:
                    best_winrate = winrate
                    best_p0, best_v0, best_o0 = p0, v0, o0
            if best_p0:
                best_params[(symbol, side)] = {
                    'price_change': best_p0,
                    'volume_change': best_v0,
                    'oi_change': best_o0,
                    'winrate': best_winrate
                }
                print(f"{symbol} {side}: p0={best_p0}, v0={best_v0}, o0={best_o0}, winrate={best_winrate:.2f}")

    params_df = pd.DataFrame([{'symbol': k[0], 'side': k[1], **v} for k, v in best_params.items()])
    params_df.to_csv('optimized_golden_params.csv', index=False)
    print("Golden Setup параметры сохранены в 'optimized_golden_params.csv'")

# Логика "от ликвидаций"
def optimize_liquidation_strategy():
    liquidations = pd.read_csv('liquidations.csv')
    trades = pd.read_csv('trades_for_training.csv')
    liquidations['timestamp'] = pd.to_datetime(liquidations['timestamp'])
    trades['datetime'] = pd.to_datetime(trades['datetime'])

    merged_data = pd.merge_asof(
        trades.sort_values('datetime'),
        liquidations.sort_values('timestamp'),
        left_on='datetime',
        right_on='timestamp',
        by='symbol',
        direction='nearest',
        tolerance=pd.Timedelta('1m')
    )

    merged_data['volatility'] = merged_data.groupby('symbol')['price'].pct_change().rolling(5).std()
    merged_data['oi_change'] = merged_data.groupby('symbol')['open_interest'].pct_change()
    features = ['value_usdt', 'volatility', 'oi_change']
    X = merged_data[features].fillna(0)
    y = (merged_data['pnl_pct'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели ликвидаций: {accuracy:.2f}")

    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    print(feature_importance)

    thresholds = merged_data['value_usdt'].quantile(np.arange(0.1, 1.0, 0.1)).values
    best_threshold, best_accuracy = None, 0
    for thresh in thresholds:
        pred = (merged_data['value_usdt'] >= thresh).astype(int)
        acc = accuracy_score(y, pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = thresh
    print(f"Лучший порог ликвидации: {best_threshold:.2f}, точность: {best_accuracy:.2f}")

# Сквиз
def optimize_squeeze_strategy():
    snapshots = pd.read_csv('golden_setup_snapshots.csv')
    trades = pd.read_csv('trades_for_training.csv')
    snapshots['is_squeeze'] = (np.abs(snapshots['price_change']) >= 5) & (snapshots['volume_change'] >= 100)

    squeeze_trades = pd.merge(
        trades,
        snapshots[snapshots['is_squeeze']],
        left_on=['symbol', 'datetime'],
        right_on=['symbol', 'timestamp'],
        how='inner'
    )

    base_winrate = (squeeze_trades['pnl_pct'] > 0).mean()
    print(f"Базовый винрейт для сквиза: {base_winrate:.2f}")

    price_thresholds = np.arange(3, 10, 1)
    volume_thresholds = np.arange(50, 200, 25)
    best_winrate = 0
    best_price_thresh = None
    best_volume_thresh = None

    for p_thresh, v_thresh in product(price_thresholds, volume_thresholds):
        temp = snapshots[(np.abs(snapshots['price_change']) >= p_thresh) & (snapshots['volume_change'] >= v_thresh)]
        temp_trades = pd.merge(trades, temp, left_on=['symbol', 'datetime'], right_on=['symbol', 'timestamp'], how='inner')
        if len(temp_trades) < 10:
            continue
        winrate = (temp_trades['pnl_pct'] > 0).mean()
        if winrate > best_winrate:
            best_winrate = winrate
            best_price_thresh = p_thresh
            best_volume_thresh = v_thresh

    print(f"Лучшие пороги сквиза: price={best_price_thresh}, volume={best_volume_thresh}, винрейт={best_winrate:.2f}")

# Запуск всех оптимизаций
if __name__ == "__main__":
    optimize_golden_setup()
    optimize_liquidation_strategy()
    optimize_squeeze_strategy()