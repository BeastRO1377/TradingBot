# strategies.py
import logging
import time
import asyncio
import pandas as pd
import numpy as np

import utils
import ai_ml
import config 

logger = logging.getLogger(__name__)

# ======================================================================
# 1. ДИСПЕТЧЕРЫ СТРАТЕГИЙ
# ======================================================================

async def high_frequency_dispatcher(bot, symbol: str):
    """
    Диспетчер для быстрых, тиковых стратегий.
    """
    if not await _prereqs_check(bot, symbol):
        return

    mode = bot.strategy_mode
    
    if mode in ("full", "liq_squeeze", "liquidation_only"):
        if await liquidation_strategy(bot, symbol):
            return
        
        # Условие для сквизов:
    if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
        await squeeze_strategy(bot, symbol)


async def low_frequency_dispatcher(bot, symbol: str):
    """
    Диспетчер для медленных стратегий, основанных на закрытых свечах.
    """
    if not await _prereqs_check(bot, symbol):
        return

    mode = bot.strategy_mode
    

    if mode in ("full", "golden_only", "golden_squeeze"):
        await golden_strategy(bot, symbol)

# ======================================================================
# 2. ПРОВЕРКА УСЛОВИЙ
# ======================================================================

async def _prereqs_check(bot, symbol: str) -> bool:
    """
    Общая предварительная проверка для всех стратегий.
    """
    if time.time() < bot.strategy_cooldown_until.get(symbol, 0):
        return False
    if symbol in bot.open_positions or symbol in bot.pending_orders:
        return False
    
    # Этот вызов теперь быстрый благодаря кэшированию
    age = await bot.listing_age_minutes(symbol)
    if age < bot.listing_age_min:
        return False
        
    if symbol in bot.failed_orders and time.time() - bot.failed_orders.get(symbol, 0) < 600:
        return False
        
    return True


# ======================================================================
# 3. РЕАЛИЗАЦИЯ СТРАТЕГИЙ
# ======================================================================

async def liquidation_strategy(bot, symbol: str) -> bool:
    """
    [ИСПРАВЛЕННАЯ ВЕРСИЯ] Проверяет кластеры ликвидаций и входит ПРОТИВ них.
    """
    signal_key = None
    try:
        liq_buffer = bot.liq_buffers.get(symbol)
        if not liq_buffer: return False

        now = time.time()
        time_window_sec = 10.0
        
        recent_events = [evt for evt in liq_buffer if now - evt['ts'] <= time_window_sec]
        if len(recent_events) < 2: return False # Нужен кластер, а не одно событие

        # Считаем суммарный объем по каждой стороне
        buy_liq_value = sum(evt['value'] for evt in recent_events if evt['side'] == 'Buy')
        sell_liq_value = sum(evt['value'] for evt in recent_events if evt['side'] == 'Sell')
        
        threshold = bot.shared_ws.get_liq_threshold(symbol)
        
        entry_side = None
        cluster_value = 0
        dominant_side = ''

        # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ЛОГИКИ ---
        if buy_liq_value >= threshold:
            # Ликвидируют шорты (рынок покупает), цена растет. Мы хотим ПРОДАТЬ.
            entry_side = "Sell"
            cluster_value = buy_liq_value
            dominant_side = "Buy" # Ликвидации были бай-сайдовые
        elif sell_liq_value >= threshold:
            # Ликвидируют лонги (рынок продает), цена падает. Мы хотим КУПИТЬ.
            entry_side = "Buy"
            cluster_value = sell_liq_value
            dominant_side = "Sell" # Ликвидации были селл-сайдовые

        if not entry_side:
            return False # Нет сигнала

        # Сигнал обнаружен!
        signal_key = (symbol, entry_side, 'liquidation_cluster')
        
        if signal_key in bot.active_signals: return True
        bot.active_signals.add(signal_key)

        # Устанавливаем кулдаун
        bot.shared_ws.last_liq_trade_time[symbol] = time.time()
        
        logger.info(f"💧 [{symbol}] ОБНАРУЖЕН КЛАСТЕР ЛИКВИДАЦИЙ ({dominant_side})! Объем: ${cluster_value:,.0f}. Вход в {entry_side}. Передано AI.")

        features = await bot.extract_realtime_features(symbol)
        if not features:
            bot.active_signals.discard(signal_key)
            return False

        candidate = {
            'symbol': symbol, 'side': entry_side, 'source': 'liquidation_cascade',
            'base_metrics': {'liquidation_value_usd': cluster_value, 'liquidation_side': dominant_side}
        }
        
        asyncio.create_task(bot._process_signal(candidate, features, signal_key))
        return True

    except Exception as e:
        logger.error(f"[liquidation_strategy] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
    
    return False



async def squeeze_strategy(bot, symbol: str) -> bool:
    """
    [ИНТРА-СВЕЧНАЯ ВЕРСИЯ] Ищет аномально большую ОДНУ минутную свечу
    и немедленно запускает анализ для входа на откате.
    """
    signal_key = None
    try:
        # Используем тот же кулдаун, чтобы не спамить сигналами
        if not bot._squeeze_allowed(symbol):
            return False
        
        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        # Нам нужна хотя бы одна закрытая свеча
        if len(candles) < 1:
            return False

        # --- ГЛАВНОЕ ИЗМЕНЕНИЕ: АНАЛИЗИРУЕМ ПОСЛЕДНЮЮ МИНУТНУЮ СВЕЧУ ---
        last_candle = candles[-1]
        open_price = utils.safe_to_float(last_candle.get("openPrice"))
        close_price = utils.safe_to_float(last_candle.get("closePrice"))

        if open_price == 0: return False

        # Рассчитываем процентное изменение ТЕЛА свечи
        pct_change_1m_body = ((close_price - open_price) / open_price) * 100.0

        # Порог для минутной свечи должен быть меньше, чем для 5-минутной.
        # 1.5% - 2.5% - хороший диапазон для начала.
        SQUEEZE_1M_THRESHOLD = 2.0 

        if abs(pct_change_1m_body) < SQUEEZE_1M_THRESHOLD:
            return False # Свеча не аномальная, выходим

        # --- СИГНАЛ ОБНАРУЖЕН! ---
        bot.last_squeeze_ts[symbol] = time.time()
        
        impulse_dir = "up" if pct_change_1m_body > 0 else "down"
        side = "Sell" if impulse_dir == "up" else "Buy"
        signal_key = (symbol, side, 'squeeze_1m') # Меняем ключ для ясности

        if signal_key in bot.active_signals:
            return True
        bot.active_signals.add(signal_key)
        
        logger.info(f"🔥 [{symbol}] ОБНАРУЖЕН ИНТРА-СВЕЧНОЙ СКВИЗ! Движение 1m: {pct_change_1m_body:.2f}%. Передано AI-аналитику.")
        
        full_features = await bot.extract_realtime_features(symbol)
        if not full_features:
            bot.active_signals.discard(signal_key)
            return False

        candidate = {
            "symbol": symbol, "side": side, "source": "squeeze",
            # Передаем метрику, которая вызвала сигнал
            "base_metrics": {'pct_1m_body': pct_change_1m_body} 
        }
        
        # Передаем сигнал в стандартный обработчик, который запустит "Охотника"
        asyncio.create_task(bot._process_signal(candidate, full_features, signal_key))
        return True
        
    except Exception as e:
        logger.error(f"[_squeeze_logic_1m] Критическая ошибка для {symbol}: {e}", exc_info=True)
        if signal_key:
            bot.active_signals.discard(signal_key)
        return False



# async def golden_strategy(bot, symbol: str):
#     """
#     Проверяет условия для стратегии Golden Setup и отправляет кандидата на анализ.
#     """
#     signal_key = None
#     try:
#         minute_candles = bot.shared_ws.candles_data.get(symbol, [])
#         recent = bot._aggregate_candles_5m(minute_candles)
#         vol_hist_5m = bot._aggregate_series_5m(list(bot.shared_ws.volume_history.get(symbol, [])), method="sum")
#         oi_hist_5m  = bot._aggregate_series_5m(list(bot.shared_ws.oi_history.get(symbol, [])), method="last")
#         cvd_hist_5m = bot._aggregate_series_5m(list(bot.shared_ws.cvd_history.get(symbol, [])), method="sum")

#         if not recent: return

#         buy_params = await bot._get_golden_thresholds(symbol, "Buy")
#         sell_params = await bot._get_golden_thresholds(symbol, "Sell")
#         period_iters = max(int(buy_params["period_iters"]), int(sell_params["period_iters"]))

#         if not all(len(hist) > period_iters for hist in [recent, vol_hist_5m, oi_hist_5m, cvd_hist_5m]):
#             return

#         action = None
#         price_change_pct, volume_change_pct, oi_change_pct = 0.0, 0.0, 0.0

#         sell_period = int(sell_params["period_iters"])
#         price_change_pct_sell = (recent[-1]["closePrice"] / recent[-1 - sell_period]["closePrice"] - 1) * 100
#         volume_change_pct_sell = (vol_hist_5m[-1] / vol_hist_5m[-1 - sell_period] - 1) * 100 if vol_hist_5m[-1 - sell_period] else 0
#         oi_change_pct_sell = (oi_hist_5m[-1] / oi_hist_5m[-1 - sell_period] - 1) * 100 if oi_hist_5m[-1 - sell_period] else 0
        
#         if (price_change_pct_sell <= -sell_params["price_change"] and
#             volume_change_pct_sell >= sell_params["volume_change"] and
#             oi_change_pct_sell >= sell_params["oi_change"]):
#             action = "Sell"
#             price_change_pct, volume_change_pct, oi_change_pct = price_change_pct_sell, volume_change_pct_sell, oi_change_pct_sell

#         if action is None:
#             buy_period = int(buy_params["period_iters"])
#             price_change_pct_buy = (recent[-1]["closePrice"] / recent[-1 - buy_period]["closePrice"] - 1) * 100
#             volume_change_pct_buy = (vol_hist_5m[-1] / vol_hist_5m[-1 - buy_period] - 1) * 100 if vol_hist_5m[-1 - buy_period] else 0
#             oi_change_pct_buy = (oi_hist_5m[-1] / oi_hist_5m[-1 - buy_period] - 1) * 100 if oi_hist_5m[-1 - buy_period] else 0

#             if (price_change_pct_buy >= buy_params["price_change"] and
#                 volume_change_pct_buy >= buy_params["volume_change"] and
#                 oi_change_pct_buy >= buy_params["oi_change"]):
#                 action = "Buy"
#                 price_change_pct, volume_change_pct, oi_change_pct = price_change_pct_buy, volume_change_pct_buy, oi_change_pct_buy

#         if action:
#             signal_key = (symbol, action, 'golden_setup')
#             if signal_key in bot.active_signals: return
#             bot.active_signals.add(signal_key)

#             features = await bot.extract_realtime_features(symbol)
#             if not features: return

#             cvd_change_pct = (cvd_hist_5m[-1] - cvd_hist_5m[-1 - period_iters]) / abs(cvd_hist_5m[-1 - period_iters] or 1) * 100
            
#             candidate = {
#                 'symbol': symbol, 'side': action, 'source': 'golden_setup',
#                 'base_metrics': {
#                     'pct_5m': price_change_pct, 'vol_change_pct': volume_change_pct,
#                     'oi_change_pct': oi_change_pct, 'cvd_change_pct': cvd_change_pct
#                 }
#             }
#             funding_snap = bot._apply_funding_to_features(symbol, features)
#             bot._apply_funding_to_candidate(candidate, funding_snap)
            
#             asyncio.create_task(bot._process_signal(candidate, features, signal_key))

#     except Exception as e:
#         logger.error(f"[_golden_logic] unexpected error for {symbol}: {e}", exc_info=True)
#         if signal_key:
#             bot.active_signals.discard(signal_key)


async def golden_strategy(bot, symbol: str):
    """
    [GOLDEN SETUP V3 - COILED SPRING] Ищет признаки скрытого накопления:
    рост ОИ при стабильной цене, с последующим всплеском объема.
    """
    signal_key = None
    try:
        # Кулдаун, чтобы не спамить по одной и той же ситуации
        if time.time() < bot._last_golden_ts.get(symbol, 0) + 300: # Кулдаун 5 минут
            return

        # Нам нужны данные хотя бы за 5 минут для анализа
        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        oi_history = list(bot.shared_ws.oi_history.get(symbol, []))
        if len(candles) < 5 or len(oi_history) < 5:
            return

        # --- Шаг 1: Анализ фазы накопления (последние 5 минут) ---
        
        # Условие 1: Стабильность цены
        price_start = utils.safe_to_float(candles[-5].get("openPrice"))
        price_end = utils.safe_to_float(candles[-1].get("closePrice"))
        if price_start == 0: return
        price_change_5m_pct = abs((price_end - price_start) / price_start) * 100.0
        is_price_stable = price_change_5m_pct < 0.8 # Цена за 5 мин изменилась менее чем на 0.8%

        # Условие 2: Рост Открытого Интереса
        oi_start = utils.safe_to_float(oi_history[-5])
        oi_end = utils.safe_to_float(oi_history[-1])
        if oi_start == 0: return
        oi_change_5m_pct = ((oi_end - oi_start) / oi_start) * 100.0
        is_oi_growing = oi_change_5m_pct > 1.0 # ОИ за 5 мин вырос более чем на 1%

        # --- Шаг 2: Поиск "искры" на последней минуте ---
        
        # Условие 3: Всплеск объема на последней свече
        last_candle_volume = utils.safe_to_float(candles[-1].get("volume"))
        avg_volume_prev_4m = np.mean([utils.safe_to_float(c.get("volume", 0)) for c in candles[-5:-1]])
        if avg_volume_prev_4m == 0: return
        VOLUME_MULTIPLIER = 2.0
        is_volume_spike = last_candle_volume > (avg_volume_prev_4m * VOLUME_MULTIPLIER)

        # --- Шаг 3: Принятие решения ---
        if not (is_price_stable and is_oi_growing and is_volume_spike):
            return # Если хотя бы одно условие не выполнено, это не наш сетап

        # СИГНАЛ ОБНАРУЖЕН!
        bot._last_golden_ts[symbol] = time.time()
        
        # Направление определяем по цвету последней "искровой" свечи
        side = "Buy" if price_end >= candles[-1]['openPrice'] else "Sell"
        signal_key = (symbol, side, 'golden_setup_v3')

        if signal_key in bot.active_signals: return
        bot.active_signals.add(signal_key)

        logger.info(f"🏆 [{symbol}] GOLDEN SETUP V3 (Сжатая пружина)! OIΔ(5m): {oi_change_5m_pct:+.2f}%, ЦенаΔ(5m): {price_change_5m_pct:.2f}%, Vol Spike: x{(last_candle_volume/avg_volume_prev_4m):.1f}. Направление: {side}.")

        full_features = await bot.extract_realtime_features(symbol)
        if not full_features:
            bot.active_signals.discard(signal_key)
            return

        candidate = {
            "symbol": symbol, "side": side, "source": "golden_setup",
            "base_metrics": { 'oi_change_5m_pct': oi_change_5m_pct }
        }
        
        asyncio.create_task(bot._process_signal(candidate, full_features, signal_key))

    except Exception as e:
        logger.error(f"[golden_strategy_v3] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key:
            bot.active_signals.discard(signal_key)



# async def golden_strategy(bot, symbol: str):
#     """
#     [НОВАЯ ВЕРСИЯ] Ищет признаки накопления/распределения:
#     аномальный объем при малом изменении цены.
#     """
#     signal_key = None
#     try:
#         # Используем отдельный, более частый кулдаун для этой стратегии
#         if time.time() < bot._last_golden_ts.get(symbol, 0) + 180: # Кулдаун 3 минуты
#             return

#         candles = list(bot.shared_ws.candles_data.get(symbol, []))
#         # Нам нужно хотя бы 30 свечей для расчета среднего объема
#         if len(candles) < 30:
#             return

#         # --- Шаг 1: Анализ последней минутной свечи ---
#         last_candle = candles[-1]
#         open_price = utils.safe_to_float(last_candle.get("openPrice"))
#         close_price = utils.safe_to_float(last_candle.get("closePrice"))
#         current_volume = utils.safe_to_float(last_candle.get("volume"))

#         if open_price == 0 or current_volume == 0:
#             return

#         # --- Шаг 2: Проверка условий ---
#         # Условие 1: Аномальный объем
#         # Считаем средний объем за предыдущие 29 минут
#         avg_volume_29m = np.mean([utils.safe_to_float(c.get("volume", 0)) for c in candles[-30:-1]])
#         if avg_volume_29m == 0: return # Избегаем деления на ноль

#         # Объем последней свечи должен быть как минимум в 2.5 раза больше среднего
#         VOLUME_MULTIPLIER = 2.5
#         is_volume_anomaly = current_volume > (avg_volume_29m * VOLUME_MULTIPLIER)

#         # Условие 2: Малое изменение цены
#         price_change_pct = abs((close_price - open_price) / open_price) * 100.0
#         is_price_stable = price_change_pct < 0.7 # Цена изменилась менее чем на 0.7%

#         # --- Шаг 3: Принятие решения о сигнале ---
#         if not (is_volume_anomaly and is_price_stable):
#             return # Если оба условия не выполнены, выходим

#         # Сигнал обнаружен!
#         bot._last_golden_ts[symbol] = time.time() # Устанавливаем кулдаун

#         # Определяем направление по цвету свечи
#         side = "Buy" if close_price >= open_price else "Sell"
#         signal_key = (symbol, side, 'golden_setup_v2')

#         if signal_key in bot.active_signals: return
#         bot.active_signals.add(signal_key)

#         logger.info(f"🏆 [{symbol}] GOLDEN SETUP! Аномальный объем {current_volume:,.0f} (x{current_volume/avg_volume_29m:.1f}) при ΔЦены {price_change_pct:.2f}%. Направление: {side}. Передано AI.")

#         full_features = await bot.extract_realtime_features(symbol)
#         if not full_features:
#             bot.active_signals.discard(signal_key)
#             return

#         candidate = {
#             "symbol": symbol, "side": side, "source": "golden_setup",
#             "base_metrics": {
#                 'price_change_1m': price_change_pct,
#                 'volume_anomaly_vs_30m': current_volume / avg_volume_29m
#             }
#         }
        
#         # Передаем сигнал в стандартный обработчик для финального одобрения AI
#         asyncio.create_task(bot._process_signal(candidate, full_features, signal_key))

#     except Exception as e:
#         logger.error(f"[golden_strategy_v2] Ошибка для {symbol}: {e}", exc_info=True)
#         if signal_key:
#             bot.active_signals.discard(signal_key)
