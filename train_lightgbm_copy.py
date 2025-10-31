import pandas as pd
import lightgbm as lgb
from pathlib import Path

CSV_PATH = Path("/Users/anatolytamilin/Downloads/TradingBot/trades_for_training.csv")

# 1️⃣ читаем csv сразу с правильным разделителем и датой
df = pd.read_csv(CSV_PATH, sep=",", parse_dates=["datetime"])

# 2️⃣ гарантируем однозначное имя столбца со временем
df.rename(columns={"datetime": "timestamp"}, inplace=True)  # если был, станет timestamp

# 3️⃣ добавляем нужные фичи: side, symbol, signal
df["side_flag"] = (
    df["side"]
      .map({"Buy": 1, "Sell": 0})   # неизвестные значения → NaN
      .astype("Int8")               # допускает <NA>, не падает
)
df["symbol"]     = df["symbol"].astype("category")
df["signal"]     = df["signal"].astype("category")

# 4️⃣ список признаков для модели
FEATURES = [
    "avg_price", "volume", "open_interest",
    "price_change", "volume_change", "oi_change",
    "period_iters",               # уже был
    "side_flag", "symbol", "signal"  # новые
]

TARGET = "pnl_pct"

# 5️⃣ train/valid split (по времени, чтобы не подглядывать в будущее)
df = df.sort_values("timestamp")
train = df[df["timestamp"] < "2025-05-01"]
valid = df[df["timestamp"] >= "2025-05-01"]

lgb_train = lgb.Dataset(train[FEATURES], label=train[TARGET], categorical_feature=["symbol", "signal"])
lgb_valid = lgb.Dataset(valid[FEATURES], label=valid[TARGET], categorical_feature=["symbol", "signal"])

params = dict(
    objective       = "regression",
    metric          = "rmse",
    learning_rate   = 0.05,
    num_leaves      = 64,
    feature_fraction= 0.8,
    seed            = 42,
)

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=300,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=["train", "valid"],
    #early_stopping_round=50,
)

model.save_model("lightgbm_trades.txt")
print("✅ модель обучена и сохранена в lightgbm_trades.txt")