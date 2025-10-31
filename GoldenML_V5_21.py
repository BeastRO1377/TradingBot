# -*- coding: utf-8 -*-
import os
import asyncio
import logging
import hmac
import hashlib
import json
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pandas_ta as ta
from pybit.unified_trading import HTTP, WebSocket
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
class Config:
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    # Trading parameters
    MAX_TOTAL_VOLUME = Decimal("500")
    POSITION_VOLUME = Decimal("100")
    TARGET_LOSS_AVERAGING = Decimal("16.0")
    TRAILING_GAP_PERCENT = Decimal("0.008")
    PROFIT_LEVEL = Decimal("0.008")
    OPERATION_MODE = "ST_cross2"
    HEDGE_MODE = True
    ASSET_SELECTION_INTERVAL = 3600  # 1 hour
    INTERVAL = "1"
    SUPER_TREND_TIMEFRAME = "1"

# State Management
class TradingState:
    def __init__(self):
        self.open_positions: Dict[str, Dict] = {}
        self.averaging_positions: Dict[str, Dict] = {}
        self.total_volume = Decimal("0")
        self.drift_history = defaultdict(list)
        self.oi_history = defaultdict(list)
        self.volume_history = defaultdict(list)
        self.selected_symbols: List[str] = []
        self.last_asset_selection = datetime.now() - timedelta(days=1)
        self.lock = asyncio.Lock()

# WebSocket Manager
class BybitWSManager:
    def __init__(self, config: Config, state: TradingState):
        self.config = config
        self.state = state
        self.ws = None
        self._running = False
        self._reconnect_attempts = 0
        self.MAX_RECONNECT = 5
        self.BASE_RECONNECT_DELAY = 5

    async def start(self, callback: Callable):
        self._running = True
        while self._running and self._reconnect_attempts < self.MAX_RECONNECT:
            try:
                self.ws = WebSocket(
                    testnet=False,
                    channel_type="private",
                    api_key=self.config.BYBIT_API_KEY,
                    api_secret=self.config.BYBIT_API_SECRET,
                    reconnect_on_close=False
                )
                
                self.ws.position_stream(callback=callback)
                await self._monitor_connection()
                self._reconnect_attempts = 0
                
            except Exception as e:
                logging.error(f"WS connection error: {e}")
                await self._handle_reconnect()

    async def _monitor_connection(self):
        while self._running and self.ws.is_connected():
            await asyncio.sleep(5)

    async def _handle_reconnect(self):
        self._reconnect_attempts += 1
        delay = min(self.BASE_RECONNECT_DELAY * 2 ** self._reconnect_attempts, 60)
        logging.info(f"Reconnecting in {delay} seconds...")
        await asyncio.sleep(delay)

    def stop(self):
        self._running = False
        if self.ws:
            try:
                self.ws.exit()
            except Exception as e:
                logging.error(f"Error closing WebSocket: {e}")

# API Client
class BybitAPIClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = HTTP(
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
            timeout=30
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_positions(self) -> List[Dict]:
        response = self.session.get_positions(category="linear", settleCoin="USDT")
        if response['retCode'] != 0:
            raise Exception(f"API Error: {response['retMsg']}")
        return [p for p in response['result']['list'] if float(p['size']) > 0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def place_order(self, symbol: str, side: str, qty: Decimal) -> Optional[Dict]:
        symbol_info = await self.get_symbol_info(symbol)
        qty = self._adjust_quantity(symbol_info, qty)
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "positionIdx": 1 if side.lower() == "buy" else 2
        }
        
        response = self.session.place_order(**params)
        if response['retCode'] != 0:
            raise Exception(f"Order failed: {response['retMsg']}")
        return response

    async def get_symbol_info(self, symbol: str) -> Dict:
        response = self.session.get_instruments_info(category="linear", symbol=symbol)
        if response['retCode'] != 0:
            raise Exception(f"Symbol info error: {response['retMsg']}")
        return response['result']['list'][0]

    def _adjust_quantity(self, symbol_info: Dict, qty: Decimal) -> Decimal:
        lot_size = symbol_info['lotSizeFilter']
        step = Decimal(lot_size['qtyStep'])
        return (qty // step) * step

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_historical_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        response = self.session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        return self._process_kline_data(response)

    def _process_kline_data(self, response: Dict) -> pd.DataFrame:
        if response['retCode'] != 0:
            raise Exception(f"Kline error: {response['retMsg']}")
        
        data = response['result']['list']
        columns = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(data, columns=columns)
        
        df['startTime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        return df[["startTime", "open", "high", "low", "close", "volume"]]

# Trading Logic
class TradeExecutor:
    def __init__(self, config: Config, state: TradingState, api: BybitAPIClient):
        self.config = config
        self.state = state
        self.api = api

    async def run_strategy(self, symbol: str):
        if self.config.OPERATION_MODE == "ST_cross2":
            await self._process_st_cross2(symbol)

    async def _process_st_cross2(self, symbol: str):
        df = await self.api.get_historical_data(symbol, self.config.INTERVAL, 200)
        if len(df) < 10:
            return

        df_fast = self._calculate_supertrend(df, 3, 1.0)
        df_slow = self._calculate_supertrend(df, 8, 3.0)

        prev_fast = df_fast['supertrend'].iloc[-2]
        curr_fast = df_fast['supertrend'].iloc[-1]
        prev_slow = df_slow['supertrend'].iloc[-2]
        curr_slow = df_slow['supertrend'].iloc[-1]

        prev_diff = prev_fast - prev_slow
        curr_diff = curr_fast - curr_slow
        last_close = df_fast['close'].iloc[-1]

        prev_diff_pct = (Decimal(prev_diff) / Decimal(last_close)) * 100
        curr_diff_pct = (Decimal(curr_diff) / Decimal(last_close)) * 100

        if prev_diff_pct <= Decimal("-0.3") and curr_diff_pct >= Decimal("0.3"):
            await self._open_position(symbol, "Buy")
        elif prev_diff_pct >= Decimal("0.3") and curr_diff_pct <= Decimal("-0.3"):
            await self._open_position(symbol, "Sell")

    def _calculate_supertrend(self, df: pd.DataFrame, length: int, multiplier: float) -> pd.DataFrame:
        high = df['high']
        low = df['low']
        close = df['close']
        
        hl2 = (high + low) / 2
        atr = ta.atr(high, low, close, length=length)
        
        df['basic_ub'] = hl2 + multiplier * atr
        df['basic_lb'] = hl2 - multiplier * atr
        
        df['supertrend'] = 0.0
        for i in range(1, len(df)):
            df['supertrend'].iloc[i] = (
                df['basic_ub'].iloc[i] if close.iloc[i] > df['basic_ub'].iloc[i-1]
                else df['basic_lb'].iloc[i] if close.iloc[i] < df['basic_lb'].iloc[i-1]
                else df['supertrend'].iloc[i-1]
            )
        return df

    async def _open_position(self, symbol: str, side: str):
        async with self.state.lock:
            if symbol in self.state.open_positions:
                return

            try:
                price = await self._get_last_price(symbol)
                qty = self.config.POSITION_VOLUME / Decimal(str(price))
                
                await self.api.place_order(symbol, side, qty)
                self.state.open_positions[symbol] = {
                    "side": side,
                    "entry_price": price,
                    "qty": qty,
                    "trailing_stop_set": False,
                    "opened_at": datetime.now()
                }
                self.state.total_volume += self.config.POSITION_VOLUME
                logging.info(f"Opened {side} position on {symbol}")
                
                await self._set_trailing_stop(symbol, qty, side)
                
            except Exception as e:
                logging.error(f"Failed to open position: {e}")

    async def _set_trailing_stop(self, symbol: str, qty: Decimal, side: str):
        try:
            trailing_gap = self.config.TRAILING_GAP_PERCENT
            await self.api.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                side=side,
                trailingStop=str(trailing_gap),
                qty=str(qty),
                positionIdx=1 if side.lower() == "buy" else 2
            )
            async with self.state.lock:
                self.state.open_positions[symbol]['trailing_stop_set'] = True
            logging.info(f"Trailing stop set for {symbol}")
        except Exception as e:
            logging.error(f"Failed to set trailing stop: {e}")

    async def check_averaging(self):
        async with self.state.lock:
            for symbol, pos in list(self.state.open_positions.items()):
                current_price = await self._get_last_price(symbol)
                entry_price = Decimal(str(pos['entry_price']))
                pnl = self._calculate_pnl(pos['side'], entry_price, current_price, pos['qty'])
                
                if pnl <= -self.config.TARGET_LOSS_AVERAGING:
                    await self._execute_averaging(symbol, pos)

    async def _execute_averaging(self, symbol: str, position: Dict):
        try:
            avg_qty = position['qty'] * Decimal('2')
            await self.api.place_order(symbol, position['side'], avg_qty)
            
            async with self.state.lock:
                self.state.averaging_positions[symbol] = {
                    **position,
                    'averaged': True,
                    'avg_qty': avg_qty
                }
            logging.info(f"Averaging position for {symbol}")
        except Exception as e:
            logging.error(f"Failed to average position: {e}")

    async def _get_last_price(self, symbol: str) -> Decimal:
        df = await self.api.get_historical_data(symbol, self.config.INTERVAL, 1)
        return Decimal(str(df['close'].iloc[-1]))

    def _calculate_pnl(self, side: str, entry: Decimal, current: Decimal, qty: Decimal) -> Decimal:
        if side.lower() == "buy":
            return (current - entry) / entry * 100
        return (entry - current) / entry * 100

# Telegram Integration
class TelegramBot:
    def __init__(self, config: Config):
        self.config = config
        self.bot = None
        self.queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(10)

    async def start(self):
        from aiogram import Bot
        self.bot = Bot(token=self.config.TELEGRAM_TOKEN)
        asyncio.create_task(self._message_sender())

    async def send_message(self, text: str):
        await self.queue.put(text)

    async def _message_sender(self):
        while True:
            try:
                msg = await self.queue.get()
                async with self.semaphore:
                    await self.bot.send_message(
                        chat_id=self.config.TELEGRAM_CHAT_ID,
                        text=msg,
                        parse_mode="Markdown"
                    )
            except Exception as e:
                logging.error(f"Telegram error: {e}")

# Main Application
class TradingBot:
    def __init__(self):
        self.config = Config()
        self.state = TradingState()
        self.api = BybitAPIClient(self.config)
        self.ws_manager = BybitWSManager(self.config, self.state)
        self.executor = TradeExecutor(self.config, self.state, self.api)
        self.tg_bot = TelegramBot(self.config)

    async def run(self):
        try:
            await self.tg_bot.start()
            await asyncio.gather(
                self.ws_manager.start(self._handle_ws_message),
                self._main_loop(),
                self._asset_selection_loop()
            )
        except KeyboardInterrupt:
            self.stop()

    async def _main_loop(self):
        while True:
            try:
                await self._update_positions()
                await self.executor.check_averaging()
                await self._process_symbols()
                await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                await self.tg_bot.send_message(f"⚠️ Main loop error: {e}")

    async def _handle_ws_message(self, message):
        try:
            if "data" in message:
                await self._process_position_update(message["data"])
        except Exception as e:
            logging.error(f"WS message error: {e}")
            await self.tg_bot.send_message(f"⚠️ WS error: {e}")

    async def _process_position_update(self, data: List[Dict]):
        async with self.state.lock:
            for pos in data:
                symbol = pos['symbol']
                if float(pos['size']) == 0:
                    if symbol in self.state.open_positions:
                        del self.state.open_positions[symbol]
                else:
                    self.state.open_positions[symbol] = {
                        'side': pos['side'],
                        'entry_price': pos['avgPrice'],
                        'qty': pos['size'],
                        'trailing_stop_set': False
                    }

    async def _update_positions(self):
        try:
            positions = await self.api.get_positions()
            async with self.state.lock:
                self.state.open_positions = {
                    p['symbol']: {
                        'side': p['side'],
                        'entry_price': p['avgPrice'],
                        'qty': p['size'],
                        'trailing_stop_set': False
                    } for p in positions
                }
        except Exception as e:
            logging.error(f"Position update error: {e}")
            await self.tg_bot.send_message(f"⚠️ Position update error: {e}")

    async def _process_symbols(self):
        symbols = await self._get_active_symbols()
        for symbol in symbols:
            await self.executor.run_strategy(symbol)

    async def _asset_selection_loop(self):
        while True:
            try:
                if (datetime.now() - self.state.last_asset_selection).seconds > self.config.ASSET_SELECTION_INTERVAL:
                    await self._select_assets()
                    await self.tg_bot.send_message("🔄 Asset selection updated")
            except Exception as e:
                logging.error(f"Asset selection error: {e}")
            await asyncio.sleep(60)

    async def _select_assets(self):
        try:
            response = self.api.session.get_tickers(category="linear")
            tickers = [t for t in response['result']['list'] if "USDT" in t['symbol']]
            
            filtered = [
                t for t in tickers
                if Decimal(t['turnover24h']) >= Decimal("2000000") and 
                Decimal(t['volume24h']) >= Decimal("2000000")
            ]
            
            self.state.selected_symbols = [t['symbol'] for t in filtered[:300]]
            self.state.last_asset_selection = datetime.now()
            
        except Exception as e:
            logging.error(f"Asset selection failed: {e}")

    async def _get_active_symbols(self) -> List[str]:
        if not self.state.selected_symbols:
            await self._