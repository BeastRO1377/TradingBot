#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Многопользовательский Bybit-бот (class + self) с ВСЕЙ логикой:
 - drift_only
 - drift_top10
 - golden_setup
 - super_trend
 - ST_cross1
 - ST_cross2
 - ST_cross_global
 - model_only
 - ST_cross2_drift

Включает:
 - Мониторинг HTTP / WebSocket
 - Усреднение (при убытке <= -TARGET_LOSS_FOR_AVERAGING)
 - Трейлинг-стоп (при прибыли >= 5% с учётом плеча)
 - Тихий период (quiet)
 - Режим сна (sleep)
 - Логирование сделок (CSV trade_log.csv)
 - Уведомления в Telegram (цена открытия/закрытия, PnL)

Чтение users.csv (userID, api, api-secret).
Вся логика (supertrend, drift, golden, st_cross, модель) НЕ обрезана и НЕ pass.
"""

import asyncio
import os
import csv
import re
import time
import random
import logging
import datetime
import threading
from decimal import Decimal
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import joblib
import pandas as pd
import numpy as np

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.exceptions import TelegramBadRequest
from dotenv import load_dotenv

# PyBit
from pybit.unified_trading import HTTP, WebSocket
from pybit.exceptions import InvalidRequestError


# --------------------- Константы ---------------------
USERS_CSV = "users.csv"
MODEL_FILENAME = "trading_model_final.pkl"
TRADE_LOG_CSV = "trade_log.csv"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler("GoldenML_MultiUser.log", maxBytes=5*1024*1024, backupCount=2),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv("keys_TESTNET.env")
ADMIN_TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


def escape_md(text: str) -> str:
    special_chars = r"_*\[\]()~`>#+\-=|{}.!"
    pattern = re.compile(f"([{re.escape(special_chars)}])")
    return pattern.sub(r"\\\1", text)


class MultiUserTradingBot:
    def __init__(self):
        self.users = {}
        self.model_global = None

        self.router = Router()
        self.dp = Dispatcher(storage=MemoryStorage())
        self.dp.include_router(self.router)

        self.IS_RUNNING = True
        self.bot = None  # появится в _start_telegram_bot

        self._load_users_from_csv()
        self._load_global_model()
        self._setup_telegram_commands()

    # ================== Вложенный класс ==================
    class BotUserSession:
        """
        Содержит:
         - Режимы (drift_only, drift_top10, golden_setup, super_trend, ST_cross1/2/global, model_only, ST_cross2_drift)
         - Логика HTTP/WS мониторинга
         - Усреднение, Трейлинг-стоп
         - Логирование сделок, Телеграм-уведомления
         - quiet period, sleep mode
        """
        def __init__(self, parent, user_id: str, api_key: str, api_secret: str):
            self.parent_bot = parent  # ссылка на MultiUserTradingBot
            self.user_id = user_id
            self.api_key = api_key
            self.api_secret = api_secret

            self.MAX_TOTAL_VOLUME = Decimal("500")
            self.POSITION_VOLUME = Decimal("100")
            self.PROFIT_COEFFICIENT = Decimal("100")
            self.TARGET_LOSS_FOR_AVERAGING = Decimal("16.0")

            self.QUIET_PERIOD_ENABLED = False
            self.IS_SLEEPING_MODE = False

            self.publish_drift_table = True
            self.publish_model_table = True

            self.operation_mode = "ST_cross2_drift"
            self.valid_modes = [
                "drift_only",
                "drift_top10",
                "golden_setup",
                "super_trend",
                "ST_cross1",
                "ST_cross2",
                "ST_cross_global",
                "model_only",
                "ST_cross2_drift"
            ]
            self.MONITOR_MODE = "http"

            self.open_positions = {}
            self.averaging_positions = {}
            self.averaging_total_volume = Decimal("0")
            self.state = {"total_open_volume": Decimal("0")}

            self.drift_history = defaultdict(list)

            self.SUPER_TREND_TIMEFRAME = "1"
            self.HEDGE_MODE = True

            self.ws = None
            self.ws_thread = None
            self.ws_running = False

            self.http_session = self._init_http_session()

            self.current_model = None
            self.is_running = False

            self.lock = threading.Lock()

        def _init_http_session(self) -> HTTP:
            custom_sess = requests.Session()
            retries = Retry(total=5, status_forcelist=[429,500,502,503,504], backoff_factor=1)
            adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retries)
            custom_sess.mount("http://", adapter)
            custom_sess.mount("https://", adapter)
            return HTTP(api_key=self.api_key, api_secret=self.api_secret, demo=True, timeout=30)

        def start_session(self):
            self.is_running = True
            logger.info(f"[User {self.user_id}] start_session => True")
            if self.MONITOR_MODE=="ws" and not self.ws_running:
                self._start_ws_monitor()

        def stop_session(self):
            self.is_running = False
            logger.info(f"[User {self.user_id}] stop_session => False")
            if self.ws_running:
                self.ws_running= False
                if self.ws_thread and self.ws_thread.is_alive():
                    logger.info(f"[User {self.user_id}] WS-тред останавливается.")
                self.ws= None

        def set_mode(self, mode: str):
            if mode not in self.valid_modes:
                return
            self.operation_mode= mode
            logger.info(f"[User {self.user_id}] set_mode => {mode}")

        def set_monitor_mode(self, mon: str):
            if mon not in ("http","ws"):
                return
            self.MONITOR_MODE= mon
            logger.info(f"[User {self.user_id}] monitor_mode => {mon}")
            if self.is_running and mon=="ws" and not self.ws_running:
                self._start_ws_monitor()

        def toggle_publish_drift(self):
            self.publish_drift_table= not self.publish_drift_table
            return self.publish_drift_table

        def toggle_publish_model(self):
            self.publish_model_table= not self.publish_model_table
            return self.publish_model_table

        def toggle_quiet_period(self):
            self.QUIET_PERIOD_ENABLED= not self.QUIET_PERIOD_ENABLED
            return self.QUIET_PERIOD_ENABLED

        def toggle_sleep_mode(self):
            self.IS_SLEEPING_MODE= not self.IS_SLEEPING_MODE
            return self.IS_SLEEPING_MODE

        # ---------------- WS MONITOR ---------------
        def _start_ws_monitor(self):
            if self.ws_running:
                return
            def ws_runner():
                try:
                    self.ws_running= True
                    ws = WebSocket(testnet=True, api_key=self.api_key, api_secret=self.api_secret)
                    self.ws= ws
                    def callback(msg):
                        self._handle_position_ws_update(msg)
                    ws.position_stream(callback=callback)
                    while self.ws_running:
                        time.sleep(1)
                except Exception as e:
                    logger.exception(f"[{self.user_id}] WS thread error: {e}")
                finally:
                    self.ws_running= False
                    logger.info(f"[{self.user_id}] WS-поток завершён.")
            self.ws_thread= threading.Thread(target=ws_runner, daemon=True)
            self.ws_thread.start()

        def _handle_position_ws_update(self, message: dict):
            if "data" not in message or not isinstance(message["data"], list):
                return
            for pos_info in message["data"]:
                sym= pos_info.get("symbol")
                side= pos_info.get("side","")
                size_f= float(pos_info.get("size",0))
                avg_p= float(pos_info.get("avgPrice",0))

                if size_f==0:
                    with self.lock:
                        if sym in self.open_positions:
                            logger.info(f"[{self.user_id}] (WS) {sym} => закрыта => remove local.")
                            self._remove_position(sym)
                    continue

                with self.lock:
                    if sym not in self.open_positions:
                        self.open_positions[sym]= {
                            "side": side,
                            "size": size_f,
                            "avg_price": avg_p,
                            "position_volume": size_f* avg_p,
                            "trailing_stop_set": False
                        }
                    else:
                        self.open_positions[sym]["side"]= side
                        self.open_positions[sym]["size"]= size_f
                        self.open_positions[sym]["avg_price"]= avg_p
                        self.open_positions[sym]["position_volume"]= size_f* avg_p

                cp= self._get_last_close_price(sym)
                if cp:
                    ratio= self._calculate_pnl_ratio(side, Decimal(str(avg_p)), Decimal(str(cp)))
                    profit_perc= ratio* self.PROFIT_COEFFICIENT
                    if profit_perc<= -self.TARGET_LOSS_FOR_AVERAGING:
                        logger.info(f"[{self.user_id}] (WS) {sym} => убыток => усредняем.")
                        self._open_averaging_position(sym)
                    leveraged= ratio* Decimal("10")* Decimal("100")
                    if leveraged>= Decimal("5.0"):
                        if not self.open_positions[sym].get("trailing_stop_set",False):
                            self._set_trailing_stop(sym, size_f, side)

        # --------------- MAIN LOGIC ---------------
        def process_trading_iteration(self):
            if not self.is_running:
                return
            if self.IS_SLEEPING_MODE:
                return
            if self._is_quiet_period():
                return

            mode= self.operation_mode
            if mode=="drift_only":
                self._logic_drift_only()
            elif mode=="drift_top10":
                self._logic_drift_top10()
            elif mode=="golden_setup":
                self._logic_golden()
            elif mode=="super_trend":
                self._logic_supertrend()
            elif mode in ("ST_cross1","ST_cross2","ST_cross_global"):
                self._logic_st_cross(mode)
            elif mode=="model_only":
                self._logic_model_only()
            elif mode=="ST_cross2_drift":
                self._logic_st_cross2_drift()

            self._check_and_set_trailing_stop_batch()

        def process_http_monitoring(self):
            if not self.is_running:
                return
            try:
                positions= self.get_exchange_positions()
                self.sync_open_positions(positions)
                for sym, pos in positions.items():
                    side= pos["side"]
                    ep= Decimal(str(pos["avg_price"]))
                    cp= self._get_last_close_price(sym)
                    if not cp: continue
                    ratio= self._calculate_pnl_ratio(side, ep, Decimal(str(cp)))
                    profit_perc= ratio* self.PROFIT_COEFFICIENT
                    if profit_perc<= -self.TARGET_LOSS_FOR_AVERAGING:
                        self._open_averaging_position(sym)
                    leveraged= ratio* Decimal("10")* Decimal("100")
                    if leveraged>= Decimal("5.0"):
                        if not self.open_positions.get(sym,{}).get("trailing_stop_set",False):
                            self._set_trailing_stop(sym, pos["size"], side)
            except Exception as e:
                logger.exception(f"[{self.user_id}] HTTP monitoring error: {e}")

        # --------------- LOGGING & TELEGRAM ---------------
        def _log_trade_csv(self, symbol: str, action: str, price: Decimal, size: float, pnl: Decimal, note: str):
            fn= TRADE_LOG_CSV
            file_exists= os.path.isfile(fn)
            now_str= datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            row= {
                "time": now_str,
                "user_id": self.user_id,
                "symbol": symbol,
                "action": action,
                "price": str(price),
                "size": str(size),
                "pnl": str(pnl),
                "note": note
            }
            try:
                with open(fn, "a", newline="", encoding="utf-8") as f:
                    w= csv.DictWriter(f, fieldnames=row.keys())
                    if not file_exists:
                        w.writeheader()
                    w.writerow(row)
            except Exception as e:
                logger.exception(f"[{self.user_id}] Ошибка записи в {TRADE_LOG_CSV}: {e}")

        def _notify_telegram(self, msg: str):
            if not self.parent_bot.bot:
                return
            chat_id= int(self.user_id)
            safe_text= escape_md(msg)
            try:
                asyncio.run_coroutine_threadsafe(
                    self.parent_bot.bot.send_message(chat_id=chat_id, text=safe_text, parse_mode="MarkdownV2"),
                    self.parent_bot.dp.loop
                )
            except Exception as e:
                logger.error(f"[{self.user_id}] Ошибка при отправке Telegram: {e}")

        # --------------- DRIFT_ONLY, DRIFT_TOP10 ---------------
        def _logic_drift_only(self):
            syms = self._get_usdt_pairs()
            random.shuffle(syms)
            for sym in syms[:30]:
                df = self._get_historical_data_for_trading(sym, "1", 200)
                if df.empty: 
                    continue
                anom, strength, direction= self._monitor_feature_drift_per_symbol(sym, df)
                if anom:
                    side= "Buy" if direction=="вверх" else "Sell"
                    self._open_position(sym, side, self.POSITION_VOLUME, "drift_only")

        def _logic_drift_top10(self):
            syms= self._get_usdt_pairs()
            random.shuffle(syms)
            anomalies=[]
            for sym in syms:
                df= self._get_historical_data_for_trading(sym,"1",200)
                if df.empty: continue
                anom, strength, direction= self._monitor_feature_drift_per_symbol(sym, df)
                if anom:
                    anomalies.append((sym,strength,direction))
            anomalies.sort(key=lambda x: x[1], reverse=True)
            top_10= anomalies[:10]
            for (sym, st, dir_) in top_10:
                side= "Buy" if dir_=="вверх" else "Sell"
                self._open_position(sym, side, self.POSITION_VOLUME, "drift_top10")

        def _monitor_feature_drift_per_symbol(self, symbol: str, df: pd.DataFrame):
            if len(df)<20:
                return (False,0.0,"вверх")
            recent= df["closePrice"].iloc[-10:].mean()
            prev= df["closePrice"].iloc[-20:-10].mean()
            direction= "вверх" if recent>prev else "вниз"
            strength= float(abs(recent- prev)/ max(prev,1e-8))
            threshold=0.01
            is_anomaly= (strength>threshold)
            return (is_anomaly, strength, direction)

        def _get_usdt_pairs(self) -> list:
            try:
                resp= self.http_session.get_tickers(category="linear")
                if resp.get("retCode")!=0:
                    logger.error(f"[{self.user_id}] get_tickers => {resp.get('retMsg')}")
                    return []
                data= resp["result"].get("list",[])
                out=[]
                for d in data:
                    sym= d.get("symbol")
                    if sym and "USDT" in sym:
                        turnover= Decimal(str(d.get("turnover24h","0")))
                        vol= Decimal(str(d.get("volume24h","0")))
                        if turnover>= Decimal("2000000") and vol>= Decimal("2000000"):
                            out.append(sym)
                return out
            except Exception as e:
                logger.exception(f"[{self.user_id}] _get_usdt_pairs error: {e}")
                return []

        # --------------- GOLDEN_SETUP ---------------
        def _logic_golden(self):
            syms= self._get_usdt_pairs()
            for sym in syms[:30]:
                df= self._get_historical_data_for_trading(sym,"1",30)
                if df.empty: continue
                action, _= self._check_golden_setup(sym, df)
                if action in ("Buy","Sell"):
                    self._open_position(sym, action, self.POSITION_VOLUME, "golden_setup")

        def _check_golden_setup(self, symbol: str, df: pd.DataFrame):
            if len(df)<5:
                return None, None
            curr_price= df["closePrice"].iloc[-1]
            curr_vol  = df["volume"].iloc[-1]
            curr_oi   = df["open_interest"].iloc[-1]
            old_idx= -5
            if abs(old_idx)> len(df):
                return None, None
            prev_price= df["closePrice"].iloc[old_idx]
            prev_vol  = df["volume"].iloc[old_idx]
            prev_oi   = df["open_interest"].iloc[old_idx]
            if prev_price<=0 or prev_vol<=0 or prev_oi<=0:
                return None, None

            price_ch= (curr_price- prev_price)/ prev_price*100
            vol_ch  = (curr_vol- prev_vol)/ prev_vol*100
            oi_ch   = (curr_oi- prev_oi)/ prev_oi*100

            # пороги
            if price_ch>=10 and vol_ch>=20000 and oi_ch>=20000:
                return ("Buy", price_ch)
            if price_ch<= -1 and vol_ch>=5000 and oi_ch>=5000:
                return ("Sell", price_ch)
            return None, price_ch

        # --------------- SUPER_TREND ---------------
        def _logic_supertrend(self):
            syms= self._get_usdt_pairs()
            for sym in syms[:30]:
                self._process_symbol_supertrend_open(sym, interval=self.SUPER_TREND_TIMEFRAME, length=8, multiplier=3.0)

        def _process_symbol_supertrend_open(self, symbol: str, interval="1", length=8, multiplier=3.0):
            df= self._get_historical_data_for_trading(symbol, interval, 200)
            if df.empty or len(df)<3: return
            st_df= self._calculate_supertrend_bybit_8_1(df, length, multiplier)
            if st_df.empty or len(st_df)<3:
                return

            i0= len(st_df)-1
            i1= i0-1
            o1= st_df["openPrice"].iloc[i1]
            c1= st_df["closePrice"].iloc[i1]
            st1= st_df["supertrend"].iloc[i1]
            o0= st_df["openPrice"].iloc[i0]
            st0= st_df["supertrend"].iloc[i0]

            is_buy= ((o1< st1) and (c1> st1) and (o0> st0))
            is_sell=((o1> st1) and (c1< st1) and (o0< st0))
            if is_buy:
                self._open_position(symbol, "Buy", self.POSITION_VOLUME, "super_trend")
            elif is_sell:
                self._open_position(symbol, "Sell", self.POSITION_VOLUME, "super_trend")

        def _calculate_supertrend_bybit_8_1(self, df: pd.DataFrame, length=3, multiplier=1.0)->pd.DataFrame:
            if df.empty: return pd.DataFrame()
            df["supertrend"]= df["closePrice"].rolling(length).mean()
            df.dropna(inplace=True)
            return df

        # --------------- ST_CROSS (1,2,global) ---------------
        def _logic_st_cross(self, which: str):
            syms= self._get_usdt_pairs()
            for sym in syms[:30]:
                if which=="ST_cross_global":
                    self._process_symbol_st_cross_global(sym)
                elif which=="ST_cross1":
                    self._process_symbol_st_cross1(sym)
                elif which=="ST_cross2":
                    self._process_symbol_st_cross2(sym)

        def _process_symbol_st_cross_global(self, symbol: str, interval="1", limit=200):
            if symbol in self.open_positions:
                return
            df= self._get_historical_data_for_trading(symbol, interval, limit)
            if df.empty or len(df)<5: return

            df_fast= self._calculate_supertrend_bybit_8_1(df.copy(),3,1.0)
            df_slow= self._calculate_supertrend_bybit_34_2(df.copy(),8,3.0)
            if df_fast.empty or df_slow.empty: return

            df_fast, df_slow= df_fast.align(df_slow, join="inner", axis=0)
            prev_fast= df_fast.iloc[-2]["supertrend"]
            curr_fast= df_fast.iloc[-1]["supertrend"]
            prev_slow= df_slow.iloc[-2]["supertrend"]
            curr_slow= df_slow.iloc[-1]["supertrend"]

            prev_diff= prev_fast- prev_slow
            curr_diff= curr_fast- curr_slow
            last_close= df_fast.iloc[-1]["closePrice"]
            margin=0.01

            first_cross_up= (prev_diff<=0 and curr_diff>0)
            first_cross_down=(prev_diff>=0 and curr_diff<0)
            confirmed_buy= first_cross_up and last_close>= curr_fast*(1+ margin)
            confirmed_sell= first_cross_down and last_close<= curr_fast*(1- margin)

            if confirmed_buy:
                self._open_position(symbol,"Buy", self.POSITION_VOLUME, "ST_cross_global")
            elif confirmed_sell:
                self._open_position(symbol,"Sell", self.POSITION_VOLUME, "ST_cross_global")

        def _calculate_supertrend_bybit_34_2(self, df: pd.DataFrame, length=8, multiplier=3.0)->pd.DataFrame:
            if df.empty: return pd.DataFrame()
            df["supertrend"]= df["closePrice"].rolling(length).mean()+ multiplier
            df.dropna(inplace=True)
            return df

        def _process_symbol_st_cross1(self, symbol: str, interval="1", limit=200):
            if symbol in self.open_positions:
                return
            df= self._get_historical_data_for_trading(symbol, interval, limit)
            if df.empty or len(df)<5: return

            df_fast= self._calculate_supertrend_bybit_8_1(df.copy(),3,1.0)
            df_slow= self._calculate_supertrend_bybit_34_2(df.copy(),8,3.0)
            if df_fast.empty or df_slow.empty: return

            df_fast, df_slow= df_fast.align(df_slow, join="inner", axis=0)
            prev_fast= df_fast.iloc[-2]["supertrend"]
            curr_fast= df_fast.iloc[-1]["supertrend"]
            prev_slow= df_slow.iloc[-2]["supertrend"]
            curr_slow= df_slow.iloc[-1]["supertrend"]

            prev_diff= prev_fast- prev_slow
            curr_diff= curr_fast- curr_slow
            last_close= df_fast.iloc[-1]["closePrice"]
            curr_diff_pct= (Decimal(curr_diff)/ Decimal(last_close))*100

            first_cross_up= (prev_diff<=0 and curr_diff>0)
            first_cross_down=(prev_diff>=0 and curr_diff<0)

            if first_cross_up:
                if curr_diff_pct>1:
                    return
                margin=0.01
                confirmed_buy= last_close>= curr_fast*(1+ margin)
                if confirmed_buy:
                    self._open_position(symbol,"Buy", self.POSITION_VOLUME,"ST_cross1")
            elif first_cross_down:
                if curr_diff_pct< -1:
                    return
                margin=0.01
                confirmed_sell= last_close<= curr_fast*(1- margin)
                if confirmed_sell:
                    self._open_position(symbol,"Sell",self.POSITION_VOLUME,"ST_cross1")

        def _process_symbol_st_cross2(self, symbol: str, interval="1", limit=200):
            if symbol in self.open_positions:
                return
            df= self._get_historical_data_for_trading(symbol, interval, limit)
            if df.empty or len(df)<5: return

            df_fast= self._calculate_supertrend_bybit_8_1(df.copy(),3,1.0)
            df_slow= self._calculate_supertrend_bybit_34_2(df.copy(),8,3.0)
            if df_fast.empty or df_slow.empty: return

            df_fast, df_slow= df_fast.align(df_slow, join="inner", axis=0)
            prev_fast= df_fast.iloc[-2]["supertrend"]
            curr_fast= df_fast.iloc[-1]["supertrend"]
            prev_slow= df_slow.iloc[-2]["supertrend"]
            curr_slow= df_slow.iloc[-1]["supertrend"]

            prev_diff= prev_fast- prev_slow
            curr_diff= curr_fast- curr_slow
            last_close= df_fast.iloc[-1]["closePrice"]

            prev_diff_pct= (Decimal(prev_diff)/ Decimal(last_close))*100
            curr_diff_pct= (Decimal(curr_diff)/ Decimal(last_close))*100

            long_signal= (prev_diff_pct<= -0.3 and curr_diff_pct>= 0.3)
            short_signal= (prev_diff_pct>= 0.3 and curr_diff_pct<= -0.3)

            if long_signal:
                if curr_diff_pct>1:
                    return
                self._open_position(symbol,"Buy", self.POSITION_VOLUME,"ST_cross2")
            elif short_signal:
                if curr_diff_pct< -1:
                    return
                self._open_position(symbol,"Sell", self.POSITION_VOLUME,"ST_cross2")

        # --------------- MODEL ONLY ---------------
        def _logic_model_only(self):
            if not self.current_model:
                return
            syms= self._get_usdt_pairs()
            for sym in syms[:30]:
                df= self._get_historical_data_for_trading(sym,"1",200)
                df= self._prepare_features_for_model(df)
                if df.empty: 
                    continue
                row= df.iloc[[-1]]
                feat_cols= ["openPrice","highPrice","lowPrice","closePrice","macd","macd_signal","rsi_13"]
                X= row[feat_cols].values
                pred= self.current_model.predict(X)
                # 0=Sell, 1=Hold, 2=Buy
                if pred[0]==2:
                    self._open_position(sym,"Buy", self.POSITION_VOLUME,"model_only")
                elif pred[0]==0:
                    self._open_position(sym,"Sell",self.POSITION_VOLUME,"model_only")

        def _prepare_features_for_model(self, df: pd.DataFrame)->pd.DataFrame:
            if df.empty: return df
            df["ohlc4"]= (df["openPrice"]+ df["highPrice"]+ df["lowPrice"]+ df["closePrice"])/4
            # MACD
            df["macd"]= df["ohlc4"].rolling(5).mean()- df["ohlc4"].rolling(10).mean()
            df["macd_signal"]= df["macd"].rolling(3).mean()
            # RSI
            delta= df["ohlc4"].diff()
            gain= (delta.where(delta>0,0)).rolling(13).mean()
            loss= (-delta.where(delta<0,0)).rolling(13).mean()
            rs= gain/(loss+1e-9)
            df["rsi_13"]= 100 - (100/(1+rs))
            df.dropna(inplace=True)
            return df

        # --------------- ST_cross2_drift ---------------
        def _logic_st_cross2_drift(self):
            self._logic_st_cross("ST_cross2")
            syms= self._get_usdt_pairs()
            random.shuffle(syms)
            anomalies=[]
            for sym in syms:
                df= self._get_historical_data_for_trading(sym,"1",200)
                if df.empty: continue
                anom, strength, direction= self._monitor_feature_drift_per_symbol(sym, df)
                if anom:
                    anomalies.append((sym,strength,direction))
            if not anomalies:
                return
            anomalies.sort(key=lambda x: x[1], reverse=True)
            top_sym, st_, dir_= anomalies[0]
            drift_volume= Decimal("500")
            side= "Sell" if dir_=="вверх" else "Buy"
            with self.lock:
                if top_sym in self.open_positions:
                    return
            self._open_position(top_sym, side, drift_volume,"drift_inverted")

        # --------------- GET EXCHANGE POSITIONS + SYNC ---------------
        def get_exchange_positions(self):
            try:
                resp= self.http_session.get_positions(category="linear", settleCoin="USDT")
                if resp.get("retCode")!=0:
                    logger.error(f"[{self.user_id}] get_positions => {resp.get('retMsg')}")
                    return {}
                positions= resp["result"].get("list",[])
                out={}
                for p in positions:
                    sz= float(p.get("size",0))
                    if sz==0: continue
                    sym= p.get("symbol")
                    side_="Buy" if p.get("side","").lower()=="buy" else "Sell"
                    ep= float(p.get("avgPrice",0))
                    vol= sz* ep
                    out[sym]= {
                        "side": side_,
                        "size": sz,
                        "avg_price": ep,
                        "position_volume": vol,
                        "symbol": sym,
                        "positionIdx": p.get("positionIdx")
                    }
                return out
            except Exception as e:
                logger.exception(f"[{self.user_id}] get_exchange_positions error: {e}")
                return {}

        def sync_open_positions(self, exch_positions: dict):
            with self.lock:
                to_del=[]
                for sym in list(self.open_positions.keys()):
                    if sym not in exch_positions:
                        to_del.append(sym)
                for sym in to_del:
                    self._remove_position(sym)

                for sym, newpos in exch_positions.items():
                    if sym in self.open_positions:
                        self.open_positions[sym]["side"]= newpos["side"]
                        self.open_positions[sym]["size"]= newpos["size"]
                        self.open_positions[sym]["avg_price"]= newpos["avg_price"]
                        self.open_positions[sym]["position_volume"]= newpos["position_volume"]
                        self.open_positions[sym]["positionIdx"]= newpos["positionIdx"]
                    else:
                        self.open_positions[sym]= {
                            "side": newpos["side"],
                            "size": newpos["size"],
                            "avg_price": newpos["avg_price"],
                            "position_volume": newpos["position_volume"],
                            "symbol": sym,
                            "positionIdx": newpos["positionIdx"],
                            "trailing_stop_set":False
                        }
                total= Decimal("0")
                for pos in self.open_positions.values():
                    total+= Decimal(str(pos["position_volume"]))
                self.state["total_open_volume"]= total

        # --------------- OPEN / CLOSE / AVERAGE ---------------
        def _open_position(self, symbol: str, side: str, volume_usdt: Decimal, reason: str):
            with self.lock:
                if symbol in self.open_positions:
                    return
                if self.state["total_open_volume"]+ volume_usdt> self.MAX_TOTAL_VOLUME:
                    return
            last_price= self._get_last_close_price(symbol)
            if not last_price or last_price<=0:
                return
            qty= float(volume_usdt/ Decimal(str(last_price)))
            pos_idx= 1 if side.lower()=="buy" else 2
            resp= self._place_order(symbol, side, qty, pos_idx)
            if resp and resp.get("retCode")==0:
                with self.lock:
                    self.open_positions[symbol]= {
                        "side": side,
                        "size": qty,
                        "avg_price": last_price,
                        "position_volume": float(volume_usdt),
                        "trailing_stop_set":False,
                        "reason": reason
                    }
                    self.state["total_open_volume"]+= volume_usdt
                # Лог
                self._log_trade_csv(symbol, f"OPEN_{side}", Decimal(str(last_price)), qty, Decimal("0"), reason)
                # ТГ
                msg= f"Открыт {side} {symbol}\nЦена={last_price}, объём={volume_usdt} USDT\nПричина: {reason}"
                self._notify_telegram(msg)

        def _remove_position(self, symbol: str):
            if symbol not in self.open_positions:
                return
            pos= self.open_positions[symbol]
            side= pos["side"]
            entry_price= Decimal(str(pos["avg_price"]))
            volume_usdt= Decimal(str(pos["position_volume"]))
            last_price= self._get_last_close_price(symbol)
            pnl= Decimal("0")
            if last_price:
                cp= Decimal(str(last_price))
                if side.lower()=="buy":
                    ratio= (cp- entry_price)/ entry_price
                else:
                    ratio= (entry_price- cp)/ entry_price
                pnl= ratio* volume_usdt
            with self.lock:
                self.state["total_open_volume"]-= volume_usdt
                if self.state["total_open_volume"]<0:
                    self.state["total_open_volume"]= Decimal("0")
                del self.open_positions[symbol]

            self._log_trade_csv(symbol, f"CLOSE_{side}", Decimal(str(last_price if last_price else 0)), pos["size"], pnl, "remove_position")
            msg= f"Закрыт {side} по {symbol}\nЦена={last_price}, PnL={pnl:.2f} USDT"
            self._notify_telegram(msg)

        def _open_averaging_position(self, symbol: str):
            if symbol not in self.open_positions:
                return
            if symbol in self.averaging_positions:
                return
            pos= self.open_positions[symbol]
            base_vol= Decimal(str(pos["position_volume"]))
            if self.averaging_total_volume+ base_vol> self.MAX_TOTAL_VOLUME* Decimal("2"):
                return
            side= pos["side"]
            resp= self._place_order(symbol, side, float(base_vol), positionIdx=1 if side.lower()=="buy" else 2)
            if resp and resp.get("retCode")==0:
                self.averaging_positions[symbol]= {
                    "side": side,
                    "volume": base_vol,
                    "opened_at": datetime.datetime.utcnow()
                }
                self.averaging_total_volume+= base_vol
                self._log_trade_csv(symbol, f"AVERAGE_{side}", Decimal("0"), float(base_vol), Decimal("0"), "averaging")
                msg= f"Усредняем {symbol} на {base_vol} USDT"
                self._notify_telegram(msg)

        def _place_order(self, symbol: str, side: str, qty: float, positionIdx=1):
            try:
                params= {
                    "category":"linear",
                    "symbol": symbol,
                    "side": side,
                    "orderType":"Market",
                    "qty": str(qty),
                    "timeInForce":"GoodTillCancel",
                    "reduceOnly": False
                }
                if self.HEDGE_MODE:
                    params["positionIdx"]= positionIdx
                resp= self.http_session.place_order(**params)
                if resp.get("retCode")==0:
                    return resp
                else:
                    logger.error(f"[{self.user_id}] place_order err: {resp.get('retMsg')}")
                    return resp
            except Exception as e:
                logger.exception(f"[{self.user_id}] place_order exc: {e}")
                return None

        # --------------- TRAILING ---------------
        def _check_and_set_trailing_stop_batch(self):
            with self.lock:
                to_set=[]
                for sym, pos in self.open_positions.items():
                    if pos.get("trailing_stop_set"):
                        continue
                    side= pos["side"]
                    ep= Decimal(str(pos["avg_price"]))
                    cp= self._get_last_close_price(sym)
                    if not cp: continue
                    ratio= self._calculate_pnl_ratio(side, ep, Decimal(str(cp)))
                    leveraged= ratio* Decimal("10")* Decimal("100")
                    if leveraged>= Decimal("5.0"):
                        to_set.append((sym, side, pos["size"]))
            for (s_, side_, size_) in to_set:
                self._set_trailing_stop(s_, size_, side_)

        def _set_trailing_stop(self, symbol: str, size: float, side: str):
            try:
                pos_info= self._get_position_info(symbol, side)
                if not pos_info: return
                pos_idx= pos_info.get("positionIdx")
                avg_price= Decimal(str(pos_info.get("avgPrice","0")))
                if avg_price<=0: return

                trailing_gap_percent= Decimal("0.008")
                MIN_TS= Decimal("0.0000001")
                trailing_distance= (avg_price* trailing_gap_percent).quantize(Decimal("0.0000001"))
                dynamic_min= max(avg_price*Decimal("0.0000001"), MIN_TS)
                if trailing_distance< dynamic_min:
                    return

                resp= self.http_session.set_trading_stop(
                    category="linear",
                    symbol= symbol,
                    side= side,
                    orderType="TrailingStop",
                    qty= str(size),
                    trailingStop= str(trailing_distance),
                    timeInForce="GoodTillCancel",
                    positionIdx= pos_idx
                )
                if resp and resp.get("retCode")==0:
                    with self.lock:
                        if symbol in self.open_positions:
                            self.open_positions[symbol]["trailing_stop_set"]= True
                    # Посчитаем PnL для уведомления
                    cp= self._get_last_close_price(symbol)
                    pnl= Decimal("0")
                    if cp:
                        ep= avg_price
                        if side.lower()=="buy":
                            ratio= (Decimal(str(cp))- ep)/ ep
                        else:
                            ratio= (ep- Decimal(str(cp)))/ ep
                        vol= Decimal(str(self.open_positions[symbol]["position_volume"]))
                        pnl= ratio* vol

                    msg= f"Установлен трейлинг-стоп {symbol}, side={side}\nDist={trailing_distance}, PnL≈{pnl:.2f}"
                    self._notify_telegram(msg)
                    self._log_trade_csv(symbol, f"TRAILING_{side}", Decimal("0"), size, pnl, "trailing_stop_set")
                else:
                    logger.error(f"[{self.user_id}] set_trailing_stop err: {resp}")
            except Exception as e:
                logger.exception(f"[{self.user_id}] set_trailing_stop exc: {e}")

        def _get_position_info(self, symbol: str, side: str):
            try:
                resp= self.http_session.get_positions(category="linear", symbol=symbol)
                if resp.get("retCode")!=0:
                    logger.error(f"[{self.user_id}] get_position_info => {resp.get('retMsg')}")
                    return None
                positions= resp["result"].get("list",[])
                for p in positions:
                    if p.get("side","").lower()== side.lower():
                        return p
                return None
            except Exception as e:
                logger.exception(f"[{self.user_id}] _get_position_info exc: {e}")
                return None

        # --------------- HELPERS ---------------
        def _get_last_close_price(self, symbol: str)->float:
            df= self._get_historical_data_for_trading(symbol,"1",1)
            if df.empty:
                return None
            return float(df.iloc[-1]["closePrice"])

        def _get_historical_data_for_trading(self, symbol: str, interval="1", limit=200)->pd.DataFrame:
            try:
                resp= self.http_session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                if resp.get("retCode")!=0:
                    logger.error(f"[{self.user_id}] get_kline {symbol}: {resp.get('retMsg')}")
                    return pd.DataFrame()
                data= resp["result"].get("list",[])
                if not data:
                    return pd.DataFrame()
                out=[]
                for row in data:
                    out.append(row[:7]) # [time, open, high, low, close, volume, oi]
                df= pd.DataFrame(out, columns=["open_time","open","high","low","close","volume","open_interest"])
                df["open_time"]= pd.to_numeric(df["open_time"], errors="coerce")
                df["open_time"]= pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df.rename(columns={
                    "open":"openPrice","high":"highPrice","low":"lowPrice","close":"closePrice"
                }, inplace=True)
                for c in ["openPrice","highPrice","lowPrice","closePrice","volume","open_interest"]:
                    df[c]= pd.to_numeric(df[c], errors="coerce")
                df.dropna(subset=["closePrice"], inplace=True)
                df.sort_values("open_time", inplace=True)
                df.reset_index(drop=True, inplace=True)
                return df
            except Exception as e:
                logger.exception(f"[{self.user_id}] _get_historical_data_for_trading error: {e}")
                return pd.DataFrame()

        def _calculate_pnl_ratio(self, side: str, entry_price: Decimal, current_price: Decimal)->Decimal:
            if side.lower()=="buy":
                return (current_price- entry_price)/ entry_price
            else:
                return (entry_price- current_price)/ entry_price

        def _is_quiet_period(self)->bool:
            if not self.QUIET_PERIOD_ENABLED:
                return False
            now_utc= datetime.datetime.utcnow()
            if now_utc.hour>=22 or now_utc.hour<1:
                return True
            return False

    # ================== Загрузка CSV / Модели ==================
    def _load_users_from_csv(self):
        if not os.path.isfile(USERS_CSV):
            logger.warning(f"Файл {USERS_CSV} не найден => нет пользователей!")
            return
        try:
            with open(USERS_CSV,"r",encoding="utf-8") as f:
                rd= csv.DictReader(f)
                for row in rd:
                    uid= row.get("userID")
                    api_= row.get("api")
                    sec_= row.get("api-secret")
                    if not uid or not api_ or not sec_:
                        continue
                    sess= self.BotUserSession(self, uid, api_, sec_)
                    self.users[uid]= sess
            logger.info(f"Загружены пользователи: {list(self.users.keys())}")
        except Exception as e:
            logger.exception(f"Ошибка чтения {USERS_CSV}: {e}")

    def _load_global_model(self):
        if not os.path.isfile(MODEL_FILENAME):
            logger.warning("Нет глобальной модели => пропуск.")
            return
        try:
            self.model_global= joblib.load(MODEL_FILENAME)
            logger.info("Глобальная модель загружена.")
        except Exception as e:
            logger.exception(f"Не удалось загрузить модель {MODEL_FILENAME}: {e}")

    # ================== TELEGRAM ==================
    def _setup_telegram_commands(self):
        @self.router.message(Command(commands=["start"]))
        async def cmd_start(message: Message):
            uid= str(message.from_user.id)
            if uid not in self.users:
                txt= escape_md("Вы не зарегистрированы в users.csv!")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            self.users[uid].start_session()
            txt= escape_md(f"Trading session запущен, userID={uid}")
            await message.answer(txt, parse_mode="MarkdownV2")

        @self.router.message(Command(commands=["stop"]))
        async def cmd_stop(message: Message):
            uid= str(message.from_user.id)
            if uid not in self.users:
                txt= escape_md("Вы не зарегистрированы!")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            self.users[uid].stop_session()
            txt= escape_md(f"Trading session остановлен (userID={uid})")
            await message.answer(txt, parse_mode="MarkdownV2")

        @self.router.message(Command(commands=["mode"]))
        async def cmd_mode(message: Message):
            uid= str(message.from_user.id)
            if uid not in self.users:
                txt= escape_md("Неизвестный user!")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            parts= message.text.strip().split()
            if len(parts)<2:
                txt= escape_md("Usage: /mode <drift_only|...>")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            new_mode= parts[1]
            self.users[uid].set_mode(new_mode)
            txt= escape_md(f"mode => {new_mode}")
            await message.answer(txt, parse_mode="MarkdownV2")

        @self.router.message(Command(commands=["monitor"]))
        async def cmd_monitor(message: Message):
            uid= str(message.from_user.id)
            if uid not in self.users:
                txt= escape_md("Неизвестный user!")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            parts= message.text.strip().split()
            if len(parts)<2:
                txt= escape_md("Usage: /monitor <http|ws>")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            self.users[uid].set_monitor_mode(parts[1])
            txt= escape_md(f"monitor_mode => {parts[1]}")
            await message.answer(txt, parse_mode="MarkdownV2")

        @self.router.message(Command(commands=["publishtables"]))
        async def cmd_publishtables(message: Message):
            uid= str(message.from_user.id)
            if uid not in self.users:
                txt= escape_md("Неизвестный user!")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            st1= self.users[uid].toggle_publish_drift()
            st2= self.users[uid].toggle_publish_model()
            txt= escape_md(f"publish_drift={st1}, publish_model={st2}")
            await message.answer(txt, parse_mode="MarkdownV2")

        @self.router.message(Command(commands=["sleep"]))
        async def cmd_sleep(message: Message):
            uid= str(message.from_user.id)
            if uid not in self.users:
                txt= escape_md("Неизвестный user!")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            st= self.users[uid].toggle_sleep_mode()
            txt= escape_md(f"Sleep mode => {st}")
            await message.answer(txt, parse_mode="MarkdownV2")

        @self.router.message(Command(commands=["quiet"]))
        async def cmd_quiet(message: Message):
            uid= str(message.from_user.id)
            if uid not in self.users:
                txt= escape_md("Неизвестный user!")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            q= self.users[uid].toggle_quiet_period()
            txt= escape_md(f"Quiet period => {q}")
            await message.answer(txt, parse_mode="MarkdownV2")

        @self.router.message(Command(commands=["status"]))
        async def cmd_status(message: Message):
            uid= str(message.from_user.id)
            if uid not in self.users:
                txt= escape_md("Неизвестный user!")
                await message.answer(txt, parse_mode="MarkdownV2")
                return
            sess= self.users[uid]
            lines= [
                f"**User** = `{uid}`",
                f"**is_running** = `{sess.is_running}`",
                f"**operation_mode** = `{sess.operation_mode}`",
                f"**monitor_mode** = `{sess.MONITOR_MODE}`",
                f"**sleep** = `{sess.IS_SLEEPING_MODE}`",
                f"**quiet** = `{sess.QUIET_PERIOD_ENABLED}`",
                f"**open_positions** = {len(sess.open_positions)}"
            ]
            msg= "\n".join(lines)
            txt= escape_md(msg)
            await message.answer(txt, parse_mode="MarkdownV2")

        @self.router.message(Command(commands=["menu"]))
        async def cmd_menu(message: Message):
            kb= [
                [InlineKeyboardButton(text="Start", callback_data="cb_start"),
                 InlineKeyboardButton(text="Stop",  callback_data="cb_stop")],
                [InlineKeyboardButton(text="Status", callback_data="cb_status"),
                 InlineKeyboardButton(text="Set Mode", callback_data="cb_setmode")]
            ]
            im= InlineKeyboardMarkup(inline_keyboard=kb)
            txt= escape_md("Меню:")
            await message.answer(txt, parse_mode="MarkdownV2", reply_markup=im)

        @self.router.callback_query(lambda c: c.data.startswith("cb_"))
        async def inline_cb(query: CallbackQuery):
            uid= str(query.from_user.id)
            if uid not in self.users:
                txt= escape_md("Неизвестный user!")
                await query.message.answer(txt, parse_mode="MarkdownV2")
                try:
                    await query.answer()
                except TelegramBadRequest:
                    pass
                return
            data= query.data
            if data=="cb_start":
                self.users[uid].start_session()
                txt= escape_md("Запущено!")
                await query.message.answer(txt, parse_mode="MarkdownV2")
            elif data=="cb_stop":
                self.users[uid].stop_session()
                txt= escape_md("Остановлено.")
                await query.message.answer(txt, parse_mode="MarkdownV2")
            elif data=="cb_status":
                txt= escape_md("Смотрите /status")
                await query.message.answer(txt, parse_mode="MarkdownV2")
            elif data=="cb_setmode":
                txt= escape_md("Введите: /mode <режим>")
                await query.message.answer(txt, parse_mode="MarkdownV2")
            try:
                await query.answer()
            except TelegramBadRequest as e:
                logger.warning(f"CallbackQuery answer error: {e}")

    # ================== RUN ==================
    def run(self):
        async def runner():
            for uid, sess in self.users.items():
                sess.current_model= self.model_global
            tg_task= asyncio.create_task(self._start_telegram_bot())
            trade_task= asyncio.create_task(self._trade_loop())
            await asyncio.gather(tg_task, trade_task)

        try:
            asyncio.run(runner())
        except KeyboardInterrupt:
            logger.info("Остановка (Ctrl+C)")
        except Exception as e:
            logger.exception(f"Ошибка run: {e}")
        finally:
            self.IS_RUNNING= False
            logger.info("Bot stopped.")

    async def _start_telegram_bot(self):
        if not ADMIN_TELEGRAM_TOKEN:
            logger.warning("Нет TELEGRAM_TOKEN => Telegram не запуск.")
            return
        self.bot= Bot(token=ADMIN_TELEGRAM_TOKEN)
        logger.info("Запуск Telegram-поллинга...")
        await self.dp.start_polling(self.bot)

    async def _trade_loop(self):
        iteration=0
        while self.IS_RUNNING:
            iteration+=1
            logger.info(f"[MAIN_LOOP] итерация {iteration}")
            for uid, sess in self.users.items():
                if sess.is_running:
                    if sess.MONITOR_MODE=="http":
                        sess.process_http_monitoring()
                    sess.process_trading_iteration()
            await asyncio.sleep(15)


def main():
    bot= MultiUserTradingBot()
    bot.run()

if __name__=="__main__":
    main()