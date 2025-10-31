# strategies.py
import logging
import time
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta

import utils
import ai_ml
import config 

logger = logging.getLogger(__name__)

async def high_frequency_dispatcher(bot, symbol: str):
    """
    [ИЗМЕНЕННАЯ ВЕРСИЯ] Работает ТОЛЬКО с монетами из "горячего списка".
    """
    # Главный фильтр: если монеты нет в "горячем списке", игнорируем ее.
    if symbol not in bot.shared_ws.watchlist:
        return

    mode = bot.strategy_mode
    if mode == "scalp_only":
        await flea_strategy(bot, symbol)
        return
        
    if not await _prereqs_check(bot, symbol):
        return
        
    if mode in ("full", "liq_squeeze", "liquidation_only"):
        if await liquidation_strategy(bot, symbol):
            return
            
    if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
        await squeeze_strategy(bot, symbol)

async def low_frequency_dispatcher(bot, symbol: str):
    """
    [ИЗМЕНЕННАЯ ВЕРСИЯ] Диспетчер для медленных стратегий (если появятся),
    также работает по "горячему списку".
    """
    if symbol not in bot.shared_ws.watchlist:
        return

    mode = bot.strategy_mode
    if mode == "scalp_only":
        return
        
    if not await _prereqs_check(bot, symbol):
        return
    # Сюда можно будет добавлять другие медленные стратегии в будущем

async def _prereqs_check(bot, symbol: str) -> bool:
    """
    [ИЗМЕНЕННАЯ ВЕРСИЯ] Добавлена проверка нахождения в "горячем списке".
    """
    # Дублируем проверку для максимальной безопасности.
    if symbol not in bot.shared_ws.watchlist:
        return False
    
    now = time.time()
    if now < bot.strategy_cooldown_until.get(symbol, 0):
        return False
    if symbol in bot.open_positions or symbol in bot.pending_orders:
        return False
    if symbol in bot.recently_closed:
        cooldown_period = 900 
        if now - bot.recently_closed[symbol] < cooldown_period:
            return False
    age = await bot.listing_age_minutes(symbol)
    if age < bot.listing_age_min:
        return False
    if symbol in bot.failed_orders and now - bot.failed_orders.get(symbol, 0) < 600:
        return False
    return True

async def liquidation_strategy(bot, symbol: str) -> bool:
    signal_key = None
    try:
        liq_buffer = bot.liq_buffers.get(symbol)
        if not liq_buffer: return False
        now = time.time()
        time_window_sec = 10.0
        recent_events = [evt for evt in liq_buffer if now - evt['ts'] <= time_window_sec]
        if len(recent_events) < 2: return False
        buy_liq_value = sum(evt['value'] for evt in recent_events if evt['side'] == 'Buy')
        sell_liq_value = sum(evt['value'] for evt in recent_events if evt['side'] == 'Sell')
        threshold = bot.shared_ws.get_liq_threshold(symbol)
        entry_side = None
        cluster_value = 0
        dominant_side = ''
        if buy_liq_value >= threshold:
            entry_side = "Sell"
            cluster_value = buy_liq_value
            dominant_side = "Buy"
        elif sell_liq_value >= threshold:
            entry_side = "Buy"
            cluster_value = sell_liq_value
            dominant_side = "Sell"
        if not entry_side:
            return False
        signal_key = (symbol, entry_side, 'liquidation_cluster')
        if signal_key in bot.active_signals: return True
        bot.active_signals.add(signal_key)
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
    signal_key = None
    try:
        if not bot._squeeze_allowed(symbol):
            return False
        features = await bot.extract_realtime_features(symbol)
        if not features: return False
        rsi_val = features.get('rsi14', 50.0)
        if not ((rsi_val > 75) or (rsi_val < 25)):
            return False
        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < 1: return False
        current_candle = candles[-1]
        open_price = utils.safe_to_float(current_candle.get("openPrice"))
        last_price = features.get("price")
        if open_price == 0 or last_price == 0: return False
        pct_change_intra_minute = ((last_price - open_price) / open_price) * 100.0
        SQUEEZE_INTRA_MINUTE_THRESHOLD = 2.0
        if abs(pct_change_intra_minute) < SQUEEZE_INTRA_MINUTE_THRESHOLD:
            return False
        side = "Sell" if rsi_val > 75 else "Buy"
        is_logical_impulse = (side == "Sell" and pct_change_intra_minute > 0) or \
                             (side == "Buy" and pct_change_intra_minute < 0)
        if not is_logical_impulse:
            return False
        bot.last_squeeze_ts[symbol] = time.time()
        signal_key = (symbol, side, 'squeeze_intra_minute')
        if signal_key in bot.active_signals: return True
        bot.active_signals.add(signal_key)
        logger.info(f"🔥 [{symbol}] ОБНАРУЖЕН ИНТРА-СВЕЧНОЙ СКВИЗ! ΔЦена 1m: {pct_change_intra_minute:.2f}%. Передано AI.")
        candidate = {
            "symbol": symbol, "side": side, "source": "squeeze",
            "base_metrics": {'pct_1m_intra_candle': pct_change_intra_minute} 
        }
        asyncio.create_task(bot._process_signal(candidate, features, signal_key))
        return True
    except Exception as e:
        logger.error(f"[_squeeze_logic_v4] Критическая ошибка для {symbol}: {e}", exc_info=True)
        if signal_key:
            bot.active_signals.discard(signal_key)
        return False

async def golden_strategy(bot, symbol: str):
    """
    [GOLDEN SETUP 2.0] Ищет сетап "Сжатая пружина" с учетом контекста рынка.
    Вызывается проактивным сканером.
    """
    signal_key = None
    try:
        # Предварительные проверки (делаем до тяжелого вызова extract_realtime_features)
        if time.time() < bot._last_golden_ts.get(symbol, 0) + 300: return
        if symbol in bot.open_positions or symbol in bot.pending_orders: return
        if symbol in bot.recently_closed: return

        features = await bot.extract_realtime_features(symbol)
        if not features: return

        # --- Контекстный фильтр ---
        CONTEXT_WINDOW_MIN = 240 # 4 часа
        MAX_CONTEXT_CHANGE_PCT = 6.0 # Не входим, если цена уже улетела на 6%
        
        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < CONTEXT_WINDOW_MIN + 1: return
            
        context_change_pct = utils.compute_pct(candles, CONTEXT_WINDOW_MIN)

        # Определяем сторону
        last_candle = candles[-1]
        price_end = utils.safe_to_float(last_candle.get("closePrice"))
        open_price = utils.safe_to_float(last_candle.get("openPrice"))
        if open_price == 0: return
        side = "Buy" if price_end >= open_price else "Sell"

        # Применяем фильтр
        if side == "Buy" and context_change_pct > MAX_CONTEXT_CHANGE_PCT:
            return
        if side == "Sell" and context_change_pct < -MAX_CONTEXT_CHANGE_PCT:
            return
        
        # --- Основная логика сетапа ---
        is_price_stable = abs(features.get('pct5m', 0.0)) < 0.8
        is_oi_growing = features.get('dOI5m', 0.0) * 100.0 > 1.0
        
        last_candle_volume = features.get('vol1m', 0)
        avg_volume_prev_4m = features.get('avg_volume_prev_4m', 0)
        if avg_volume_prev_4m == 0: return
        VOLUME_MULTIPLIER = 2.0
        is_volume_spike = last_candle_volume > (avg_volume_prev_4m * VOLUME_MULTIPLIER)

        if not (is_price_stable and is_oi_growing and is_volume_spike):
            return

        # --- Сигнал подтвержден, отправляем ---
        bot._last_golden_ts[symbol] = time.time()
        signal_key = (symbol, side, 'golden_setup_v2')
        if signal_key in bot.active_signals: return
        bot.active_signals.add(signal_key)

        vol_spike_ratio = (last_candle_volume / avg_volume_prev_4m)
        logger.info(f"🏆 [{symbol}] GOLDEN SETUP 2.0! Контекст 4ч: {context_change_pct:+.2f}%. Vol Spike: x{vol_spike_ratio:.1f}. Направление: {side}.")
        
        candidate = {
            "symbol": symbol, "side": side, "source": "golden_setup",
            "base_metrics": { 'oi_change_5m_pct': features.get('dOI5m', 0.0) * 100.0 }
        }
        asyncio.create_task(bot._process_signal(candidate, features, signal_key))

    except Exception as e:
        logger.error(f"[golden_strategy_v2] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key:
            bot.active_signals.discard(signal_key)



async def flea_strategy(bot, symbol: str):
    cfg = bot.user_data.get("flea_settings", config.FLEA_STRATEGY)
    if not cfg.get("ENABLED", False):
        return
        
    now = time.time()
    if bot.flea_positions_count >= cfg.get("MAX_OPEN_POSITIONS", 15): return
    if now < bot.flea_cooldown_until.get(symbol, 0): return
    bot.flea_cooldown_until[symbol] = now + 5 
    if symbol in bot.open_positions or symbol in bot.pending_orders: return
    if symbol in bot.recently_closed and now - bot.recently_closed[symbol] < 300: return

    try:
        # Единственный тяжелый вызов
        features = await bot.extract_realtime_features(symbol)
        if not features: return

        # Никаких Pandas! Только получаем готовые значения.
        fast_ema = features.get('fast_ema', 0.0)
        slow_ema = features.get('slow_ema', 0.0)
        fast_ema_prev = features.get('fast_ema_prev', 0.0)
        slow_ema_prev = features.get('slow_ema_prev', 0.0)
        rsi = features.get('rsi14', 50.0)
        atr = features.get('atr14', 0.0)
        trend_ema = features.get('trend_ema', 0.0)
        last_price = features.get('price', 0.0)

        if any(v == 0.0 for v in [fast_ema, slow_ema, trend_ema, last_price, atr]):
             return

        is_uptrend = last_price > trend_ema
        is_downtrend = last_price < trend_ema
        
        atr_pct = (atr / last_price) * 100
        volatility_ok = cfg.get("MIN_ATR_PCT", 0.05) < atr_pct < cfg.get("MAX_ATR_PCT", 1.5)
        
        side = None
        if fast_ema_prev < slow_ema_prev and fast_ema > slow_ema and rsi > 51 and is_uptrend:
            side = "Buy"
        elif fast_ema_prev > slow_ema_prev and fast_ema < slow_ema and rsi < 49 and is_downtrend:
            side = "Sell"

        if not side or not volatility_ok: return
            
        tp_offset = atr * cfg.get("TP_ATR_MULTIPLIER", 1.5)
        sl_offset = atr * cfg.get("STOP_LOSS_ATR_MULTIPLIER", 1.0)
        
        if side == "Buy":
            tp_price = last_price + tp_offset
            sl_price = last_price - sl_offset
        else:
            tp_price = last_price - tp_offset
            sl_price = last_price + sl_offset
            
        candidate = {
            'symbol': symbol, 'side': side, 'source': 'flea_v2.1',
            'take_profit_price': tp_price, 'stop_loss_price': sl_price,
            'max_hold_minutes': cfg.get("MAX_HOLD_MINUTES", 10)
        }
        logger.info(f"🦟 [{symbol}] 'Блоха 2.1' поймала сигнал в {side}. TP={tp_price:.6f}, SL={sl_price:.6f}")
        await bot.execute_flea_trade(candidate)
        
    except Exception as e:
        logger.error(f"[flea_strategy_v2.1] Ошибка для {symbol}: {e}", exc_info=True)
