
from collections import defaultdict
import asyncio
import pandas as pd
from pybit.unified_trading import WebSocket

class PublicWebSocketManager:
    def __init__(self, symbols, interval="1"):
        self.symbols = symbols
        self.interval = interval
        self.ws = None
        self.candles_data = defaultdict(list)
        self.ticker_data = {}
        self.latest_open_interest = {}
        self.loop = asyncio.get_event_loop()

    async def start(self):
        def _on_message(msg):
            asyncio.run_coroutine_threadsafe(self.route_message(msg), self.loop)

        self.ws = WebSocket(
            testnet=False,
            channel_type="linear",
            ping_interval=20,
            ping_timeout=10,
        )
        self.ws.kline_stream(interval=self.interval, symbol=self.symbols, callback=_on_message)
        self.ws.ticker_stream(symbol=self.symbols, callback=_on_message)

    async def route_message(self, msg):
        topic = msg.get("topic", "")
        if topic.startswith("kline."):
            await self.handle_kline(msg)
        elif topic == "tickers":
            await self.handle_ticker(msg)

    async def handle_kline(self, msg):
        symbol = msg["topic"].split(".")[-1]
        data = msg["data"]
        if not data.get("confirm", False):
            return

        ts = pd.to_datetime(int(data["start"]), unit="ms")
        row = {
            "startTime": ts,
            "openPrice": float(data["open"]),
            "highPrice": float(data["high"]),
            "lowPrice": float(data["low"]),
            "closePrice": float(data["close"]),
            "volume": float(data["volume"]),
        }

        self.candles_data[symbol].append(row)
        if len(self.candles_data[symbol]) > 500:
            self.candles_data[symbol] = self.candles_data[symbol][-500:]

    async def handle_ticker(self, msg):
        data = msg.get("data", {})
        if isinstance(data, list):
            for ticker in data:
                symbol = ticker.get("symbol")
                if symbol:
                    self.latest_open_interest[symbol] = float(ticker.get("openInterest", 0))

import csv
import os
import asyncio
from typing import List
from pybit.unified_trading import HTTP
from collections import defaultdict

# Предположим, что PublicWebSocketManager определён в этом же файле

class TradingBot:
    def __init__(self, user_data, shared_ws):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "live")
        self.session = HTTP(
            demo=(self.mode == "demo"),
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        self.shared_ws = shared_ws
        self.symbols = shared_ws.symbols
        self.ws_private = None  # будет позже
        self.open_positions = {}
        self.loop = asyncio.get_event_loop()

    async def start(self):
        print(f"[User {self.user_id}] Бот запущен")
        # TODO: добавить приватные подписки, стратегии, трейлинг-стоп и т.д.

def load_users_from_csv(csv_path: str) -> List[dict]:
    users = []
    if not os.path.exists(csv_path):
        return users
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "user_id" in row and "api_key" in row and "api_secret" in row:
                users.append(row)
    return users

async def run_all_users():
    symbols = ["BTCUSDT", "ETHUSDT"]
    shared_ws = PublicWebSocketManager(symbols=symbols)
    await shared_ws.start()

    users = load_users_from_csv("users.csv")
    bots = [TradingBot(user_data=u, shared_ws=shared_ws) for u in users]

    tasks = [bot.start() for bot in bots]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run_all_users())

from pybit.unified_trading import WebSocket

class TradingBot:
    def __init__(self, user_data, shared_ws):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "live")
        self.session = HTTP(
            demo=(self.mode == "demo"),
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        self.shared_ws = shared_ws
        self.symbols = shared_ws.symbols
        self.ws_private = None
        self.open_positions = {}
        self.loop = asyncio.get_event_loop()

    async def start(self):
        print(f"[User {self.user_id}] Бот запущен")
        await self.setup_private_ws()

    async def setup_private_ws(self):
        def _on_private(msg):
            asyncio.run_coroutine_threadsafe(self.route_private_message(msg), self.loop)

        self.ws_private = WebSocket(
            testnet=(self.mode == "demo"),
            channel_type="private",
            api_key=self.api_key,
            api_secret=self.api_secret,
            ping_interval=20,
            ping_timeout=10,
        )
        self.ws_private.position_stream(callback=_on_private)

    async def route_private_message(self, msg):
        topic = msg.get("topic", "")
        if "position" in topic:
            await self.handle_position_update(msg)

    async def handle_position_update(self, msg):
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]

        for position in data:
            symbol = position.get("symbol")
            size = float(position.get("size", 0))
            if size > 0:
                print(f"[User {self.user_id}] Активная позиция {symbol}: {size}")
            else:
                print(f"[User {self.user_id}] Позиция {symbol} закрыта")

    async def handle_position_update(self, msg):
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]

        for position in data:
            await self.evaluate_position(position)

    async def evaluate_position(self, position):
        symbol = position.get("symbol")
        size = float(position.get("size", 0))
        side = position.get("side", "Buy")
        entry_price = float(position.get("entryPrice", 0))
        mark_price = float(position.get("markPrice", 0))
        pnl = float(position.get("unrealisedPnl", 0))

        if size == 0:
            self.open_positions.pop(symbol, None)
            print(f"[User {self.user_id}] {symbol} позиция закрыта")
            return

        self.open_positions[symbol] = position
        print(f"[User {self.user_id}] {symbol} позиция обновлена, PnL: {pnl:.2f} USDT")

        # Расчёт PnL в процентах
        if entry_price > 0:
            direction = 1 if side == "Buy" else -1
            pnl_pct = direction * ((mark_price - entry_price) / entry_price) * 100

            if pnl_pct >= 5:
                await self.set_trailing_stop(symbol, entry_price, pnl_pct, side)

    async def set_trailing_stop(self, symbol, entry_price, pnl_pct, side):
        base_trailing = 2.5
        reduction = 0

        oi = self.shared_ws.latest_open_interest.get(symbol, 0)
        if oi > 1000:
            reduction += 0.5

        candles = self.shared_ws.candles_data.get(symbol, [])
        if candles:
            recent_close = candles[-1]["closePrice"]
            price_change = abs(recent_close - entry_price) / entry_price * 100
            if price_change > 1:
                reduction += 0.5

        final_trailing = max(base_trailing - reduction, 0.5)
        stop_pct = pnl_pct - final_trailing

        if side == "Buy":
            stop_price = entry_price * (1 + stop_pct / 100)
        else:
            stop_price = entry_price * (1 - stop_pct / 100)

        print(f"[User {self.user_id}] Установка трейлинг-стопа {symbol} на {stop_price:.4f}")

        try:
            result = await asyncio.to_thread(lambda: self.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                side=side,
                stopLoss=str(stop_price),
                timeInForce="GoodTillCancel"
            ))
            print(f"[User {self.user_id}] StopLoss результат: {result}")
        except Exception as e:
            print(f"[User {self.user_id}] Ошибка установки стопа: {e}")

import numpy as np
import lightgbm as lgb

class TradingBot:
    # ...
    def load_model(self):
        try:
            self.model = lgb.Booster(model_file="model.txt")
            print(f"[User {self.user_id}] ML-модель загружена")
        except Exception as e:
            self.model = None
            print(f"[User {self.user_id}] Ошибка загрузки модели: {e}")

    def prepare_features_for_model(self, df):
        df["rsi"] = ta.rsi(df["closePrice"], length=14)
        df["ema_20"] = df["closePrice"].ewm(span=20).mean()
        df["std_20"] = df["closePrice"].rolling(window=20).std()
        df["upper_band"] = df["ema_20"] + 2 * df["std_20"]
        df["lower_band"] = df["ema_20"] - 2 * df["std_20"]
        df = df.dropna()
        if df.empty:
            return None
        features = df[["rsi", "ema_20", "std_20", "upper_band", "lower_band"]].iloc[-1:]
        return features

    async def evaluate_ml_signal(self, symbol):
        df = pd.DataFrame(self.shared_ws.candles_data.get(symbol, []))
        if df.empty or df.shape[0] < 30:
            return

        features = self.prepare_features_for_model(df)
        if features is None or self.model is None:
            return

        prediction = self.model.predict(features)[0]
        print(f"[User {self.user_id}] ML сигнал по {symbol}: {prediction:.4f}")

        if prediction > 0.8:
            await self.place_order(symbol, "Buy", qty=0.01, reason="ML_signal")
        elif prediction < 0.2:
            await self.place_order(symbol, "Sell", qty=0.01, reason="ML_signal")

    async def place_order(self, symbol, side, qty, reason=""):
        order = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "Market"
        }

        if self.monitoring == "http" or self.mode == "demo":
            try:
                result = await asyncio.to_thread(lambda: self.session.place_order(**order))
                print(f"[User {self.user_id}] Ордер через REST ({reason}): {result}")
            except Exception as e:
                print(f"[User {self.user_id}] Ошибка REST-ордера: {e}")
        else:
            try:
                await self.ws_private.send_cmd("order.place", order)
                print(f"[User {self.user_id}] Ордер через WS ({reason}) отправлен")
            except Exception as e:
                print(f"[User {self.user_id}] Ошибка WS-ордера: {e}")


    def prepare_features_for_model(self, df, symbol=None):
        df["rsi"] = ta.rsi(df["closePrice"], length=14)
        df["ema_20"] = df["closePrice"].ewm(span=20).mean()
        df["std_20"] = df["closePrice"].rolling(window=20).std()
        df["upper_band"] = df["ema_20"] + 2 * df["std_20"]
        df["lower_band"] = df["ema_20"] - 2 * df["std_20"]
        df["volume"] = df["volume"].rolling(window=5).mean()

        if symbol and symbol in self.shared_ws.latest_open_interest:
            df["open_interest"] = self.shared_ws.latest_open_interest[symbol]
        else:
            df["open_interest"] = 0.0

        df = df.dropna()
        if df.empty:
            return None
        features = df[["rsi", "ema_20", "std_20", "upper_band", "lower_band", "volume", "open_interest"]].iloc[-1:]
        return features

    async def execute_golden_setup_websocket(self, symbol):
        for side in ["Buy", "Sell"]:
            key = (symbol, side)
            if key not in self.golden_param_store:
                continue

            params = self.golden_param_store[key]
            period = int(params["period_iters"])
            df = pd.DataFrame(self.shared_ws.candles_data.get(symbol, []))
            if df.empty or len(df) < period + 1:
                continue

            recent = df.tail(period + 1)
            close_now = recent["closePrice"].iloc[-1]
            close_prev = recent["closePrice"].iloc[0]
            price_delta_pct = (close_now - close_prev) / close_prev * 100

            volume_now = recent["volume"].iloc[-1]
            volume_avg = recent["volume"].iloc[:-1].mean()
            volume_ratio = (volume_now / volume_avg) if volume_avg > 0 else 0

            oi_now = self.shared_ws.latest_open_interest.get(symbol, 0)
            oi_hist = volume_avg  # Прокси для динамики OI
            oi_delta_pct = (oi_now - oi_hist) / oi_hist * 100 if oi_hist else 0

            trigger = False
            if side == "Buy" and price_delta_pct > params["price_change"] and                volume_ratio > (params["volume_change"] / 1000) and                oi_delta_pct > params["oi_change"]:
                trigger = True
            elif side == "Sell" and price_delta_pct < params["price_change"] and                  volume_ratio > (params["volume_change"] / 1000) and                  oi_delta_pct > params["oi_change"]:
                trigger = True

            if trigger:
                print(f"[User {self.user_id}] ⚡ GOLDEN {side} SETUP {symbol} — ΔP: {price_delta_pct:.2f}%, vol×{volume_ratio:.1f}, ΔOI: {oi_delta_pct:.2f}%")
                await self.place_order(symbol, side, qty=0.01, reason="GoldenSetup")

                # Подготовка признаков и логирование
                df["rsi"] = ta.rsi(df["closePrice"], length=14)
                df["ema_20"] = df["closePrice"].ewm(span=20).mean()
                df["std_20"] = df["closePrice"].rolling(window=20).std()
                df["upper_band"] = df["ema_20"] + 2 * df["std_20"]
                df["lower_band"] = df["ema_20"] - 2 * df["std_20"]
                df["volume_avg"] = df["volume"].rolling(window=5).mean()
                df = df.dropna()

                if not df.empty:
                    latest = df.iloc[-1]
                    save_training_snapshot({
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "price_change": price_delta_pct,
                        "volume_ratio": volume_ratio,
                        "oi_change": oi_delta_pct,
                        "rsi": latest["rsi"],
                        "ema_20": latest["ema_20"],
                        "std_20": latest["std_20"],
                        "upper_band": latest["upper_band"],
                        "lower_band": latest["lower_band"],
                        "volume": latest["volume_avg"],
                        "open_interest": oi_now,
                        "strategy": "golden",
                        "result": ""
                    })

    def compute_supertrend(self, df, period=10, multiplier=3):
        hl2 = (df["highPrice"] + df["lowPrice"]) / 2
        atr = ta.atr(df["highPrice"], df["lowPrice"], df["closePrice"], length=period)
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        supertrend = pd.Series(index=df.index, dtype=bool)

        in_uptrend = True
        for current in range(1, len(df)):
            if df["closePrice"].iloc[current] > upperband.iloc[current - 1]:
                in_uptrend = True
            elif df["closePrice"].iloc[current] < lowerband.iloc[current - 1]:
                in_uptrend = False
            supertrend.iloc[current] = in_uptrend
        return supertrend

    async def run_all_strategies(self, symbol):
        df = pd.DataFrame(self.shared_ws.candles_data.get(symbol, []))
        if df.empty or len(df) < 50:
            return

        df["macd"], df["macd_signal"], _ = ta.macd(df["closePrice"])
        df["rsi"] = ta.rsi(df["closePrice"], length=14)
        df["supertrend"] = self.compute_supertrend(df)

        # Простейший сигнал MACD
        if df["macd"].iloc[-2] < df["macd_signal"].iloc[-2] and df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]:
            print(f"[User {self.user_id}] 📈 MACD сигнал BUY по {symbol}")
            await self.place_order(symbol, "Buy", qty=0.01, reason="MACD_cross")

        # RSI выход из зоны перепроданности
        if df["rsi"].iloc[-2] < 30 and df["rsi"].iloc[-1] >= 30:
            print(f"[User {self.user_id}] 📈 RSI сигнал BUY по {symbol}")
            await self.place_order(symbol, "Buy", qty=0.01, reason="RSI_rebound")

        # SuperTrend смена тренда
        if df["supertrend"].iloc[-2] is False and df["supertrend"].iloc[-1] is True:
            print(f"[User {self.user_id}] 🔁 SuperTrend смена на ВВЕРХ по {symbol}")
            await self.place_order(symbol, "Buy", qty=0.01, reason="SuperTrend_up")

        # Drift – упрощённая динамика: стабильный рост
        closes = df["closePrice"].tail(5).values
        if all(x < y for x, y in zip(closes, closes[1:])):
            print(f"[User {self.user_id}] 🟢 Drift: стабильный рост по {symbol}")
            await self.place_order(symbol, "Buy", qty=0.01, reason="Drift_up")

    async def strategy_macd(self, symbol):
        df = pd.DataFrame(self.shared_ws.candles_data.get(symbol, []))
        if df.empty or len(df) < 50:
            return

        df["macd"], df["macd_signal"], _ = ta.macd(df["closePrice"])
        if df["macd"].iloc[-2] < df["macd_signal"].iloc[-2] and df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]:
            print(f"[User {self.user_id}] 📈 MACD сигнал BUY по {symbol}")
            await self.place_order(symbol, "Buy", qty=0.01, reason="MACD_cross")

    async def strategy_rsi(self, symbol):
        df = pd.DataFrame(self.shared_ws.candles_data.get(symbol, []))
        if df.empty or len(df) < 20:
            return

        df["rsi"] = ta.rsi(df["closePrice"], length=14)
        if df["rsi"].iloc[-2] < 30 and df["rsi"].iloc[-1] >= 30:
            print(f"[User {self.user_id}] 📈 RSI сигнал BUY по {symbol}")
            await self.place_order(symbol, "Buy", qty=0.01, reason="RSI_rebound")

    async def strategy_supertrend(self, symbol):
        df = pd.DataFrame(self.shared_ws.candles_data.get(symbol, []))
        if df.empty or len(df) < 20:
            return

        df["supertrend"] = self.compute_supertrend(df)
        if df["supertrend"].iloc[-2] is False and df["supertrend"].iloc[-1] is True:
            print(f"[User {self.user_id}] 🔁 SuperTrend смена на ВВЕРХ по {symbol}")
            await self.place_order(symbol, "Buy", qty=0.01, reason="SuperTrend_up")

    async def strategy_drift(self, symbol):
        df = pd.DataFrame(self.shared_ws.candles_data.get(symbol, []))
        if df.empty or len(df) < 6:
            return

        closes = df["closePrice"].tail(5).values
        if all(x < y for x, y in zip(closes, closes[1:])):
            print(f"[User {self.user_id}] 🟢 Drift: стабильный рост по {symbol}")
            await self.place_order(symbol, "Buy", qty=0.01, reason="Drift_up")

import asyncio
import logging
from telegram_fsm_v12 import dp, bot

async def run_all():
    bot_task = asyncio.create_task(run_all_users())
    tg_task = asyncio.create_task(dp.start_polling(bot))
    await asyncio.gather(bot_task, tg_task)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_all())


import os

class TradingBot:
    # ...
    def load_model(self):
        model_path = "model.txt"
        if os.path.exists(model_path):
            try:
                self.model = lgb.Booster(model_file=model_path)
                print(f"[User {self.user_id}] ✅ ML-модель загружена из {model_path}")
            except Exception as e:
                self.model = None
                print(f"[User {self.user_id}] ❌ Ошибка загрузки модели: {e}")
        else:
            self.model = None
            print(f"[User {self.user_id}] ⚠️ model.txt не найден — ML сигналы отключены")
