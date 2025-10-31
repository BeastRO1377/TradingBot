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
    Диспетчер для быстрых, тиковых стратегий. Вызывается на каждом обновлении тикера.
    """
    if not await _prereqs_check(bot, symbol):
        return

    mode = bot.strategy_mode
    
    if mode in ("full", "liq_squeeze", "liquidation_only"):
        if await liquidation_strategy(bot, symbol):
            return

    if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
        if await squeeze_strategy(bot, symbol):
            return

async def low_frequency_dispatcher(bot, symbol: str):
    """
    Диспетчер для медленных стратегий, основанных на закрытых свечах.
    Вызывается раз в минуту.
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
    Проверяет наличие кластеров ликвидаций в реальном времени.
    """
    signal_key = None
    try:
        liq_buffer = bot.liq_buffers.get(symbol)
        if not liq_buffer: return False

        now = time.time()
        time_window_sec = 10.0
        
        recent_events = [evt for evt in liq_buffer if now - evt['ts'] <= time_window_sec]
        if not recent_events: return False

        dominant_side = recent_events[-1]['side']
        cluster_value = sum(evt['value'] for evt in recent_events if evt['side'] == dominant_side)
        threshold = bot.shared_ws.get_liq_threshold(symbol)

        if cluster_value >= threshold:
            # --- ИСПРАВЛЕННАЯ ЛОГИКА ДЛЯ КОНТР-ТРЕНДА ---
            # Если ликвидируют лонги (Buy), мы хотим купить на дне.
            # Если ликвидируют шорты (Sell), мы хотим продать на пике.
            entry_side = "Buy" if dominant_side == "Buy" else "Sell" 
            # Примечание: на Bybit ликвидация лонга имеет side "Buy", а шорта - "Sell".
            # Для контр-тренда нам нужно войти в ту же сторону, что и ликвидации.
            # Значит, первоначальная логика entry_side = dominant_side была верной
            # для контр-тренда в контексте API Bybit.
            # Моя предыдущая заметка была неверной, возвращаем как было.
            entry_side = dominant_side
            # --- КОНЕЦ ПОЯСНЕНИЯ ---

            signal_key = (symbol, entry_side, 'liquidation_cluster')
            
            if signal_key in bot.active_signals: return True
            bot.active_signals.add(signal_key)

            bot.shared_ws.last_liq_trade_time[symbol] = time.time()
            
            logger.info(f"💧 [{symbol}] ОБНАРУЖЕН КЛАСТЕР ЛИКВИДАЦИЙ! Объем: ${cluster_value:,.0f}. Передано AI.")

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
        logger.error(f"[liquidation_strategy] Критическая ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
    
    return False

async def squeeze_strategy(bot, symbol: str) -> bool:
    """
    Ищет сквиз, используя надежный кулдаун, чтобы избежать спама сигналами.
    """
    signal_key = None
    try:
        if not bot._squeeze_allowed(symbol):
            return False
        
        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < 5:
            return False

        pct_change_5m = utils.compute_pct(candles, 5)

        if abs(pct_change_5m) < config.SQUEEZE_THRESHOLD_PCT:
            return False

        bot.last_squeeze_ts[symbol] = time.time()
        
        impulse_dir = "up" if pct_change_5m > 0 else "down"
        side = "Sell" if impulse_dir == "up" else "Buy"
        signal_key = (symbol, side, 'squeeze')

        if signal_key in bot.active_signals:
            return True
        bot.active_signals.add(signal_key)
        
        logger.info(f"🔥 [{symbol}] ОБНАРУЖЕН СКВИЗ! Движение: {pct_change_5m:.2f}%. Передано Охотнику.")
        
        full_features = await bot.extract_realtime_features(symbol)
        if not full_features:
            bot.active_signals.discard(signal_key)
            return False

        candidate = {
            "symbol": symbol, "side": side, "source": "squeeze",
            "base_metrics": {'pct_5m': pct_change_5m}
        }
        
        asyncio.create_task(bot._hunt_squeeze_entry_point(candidate, full_features, signal_key))
        return True
        
    except Exception as e:
        logger.error(f"[_squeeze_logic] Критическая ошибка для {symbol}: {e}", exc_info=True)
        if signal_key:
            bot.active_signals.discard(signal_key)
        return False

async def golden_strategy(bot, symbol: str):
    """
    Проверяет условия для стратегии Golden Setup и отправляет кандидата на анализ.
    """
    signal_key = None
    try:
        minute_candles = bot.shared_ws.candles_data.get(symbol, [])
        recent = bot._aggregate_candles_5m(minute_candles)
        vol_hist_5m = bot._aggregate_series_5m(list(bot.shared_ws.volume_history.get(symbol, [])), method="sum")
        oi_hist_5m  = bot._aggregate_series_5m(list(bot.shared_ws.oi_history.get(symbol, [])), method="last")
        cvd_hist_5m = bot._aggregate_series_5m(list(bot.shared_ws.cvd_history.get(symbol, [])), method="sum")

        if not recent: return

        buy_params = await bot._get_golden_thresholds(symbol, "Buy")
        sell_params = await bot._get_golden_thresholds(symbol, "Sell")
        period_iters = max(int(buy_params["period_iters"]), int(sell_params["period_iters"]))

        if not all(len(hist) > period_iters for hist in [recent, vol_hist_5m, oi_hist_5m, cvd_hist_5m]):
            return

        action = None
        price_change_pct, volume_change_pct, oi_change_pct = 0.0, 0.0, 0.0

        sell_period = int(sell_params["period_iters"])
        price_change_pct_sell = (recent[-1]["closePrice"] / recent[-1 - sell_period]["closePrice"] - 1) * 100
        volume_change_pct_sell = (vol_hist_5m[-1] / vol_hist_5m[-1 - sell_period] - 1) * 100 if vol_hist_5m[-1 - sell_period] else 0
        oi_change_pct_sell = (oi_hist_5m[-1] / oi_hist_5m[-1 - sell_period] - 1) * 100 if oi_hist_5m[-1 - sell_period] else 0
        
        if (price_change_pct_sell <= -sell_params["price_change"] and
            volume_change_pct_sell >= sell_params["volume_change"] and
            oi_change_pct_sell >= sell_params["oi_change"]):
            action = "Sell"
            price_change_pct, volume_change_pct, oi_change_pct = price_change_pct_sell, volume_change_pct_sell, oi_change_pct_sell

        if action is None:
            buy_period = int(buy_params["period_iters"])
            price_change_pct_buy = (recent[-1]["closePrice"] / recent[-1 - buy_period]["closePrice"] - 1) * 100
            volume_change_pct_buy = (vol_hist_5m[-1] / vol_hist_5m[-1 - buy_period] - 1) * 100 if vol_hist_5m[-1 - buy_period] else 0
            oi_change_pct_buy = (oi_hist_5m[-1] / oi_hist_5m[-1 - buy_period] - 1) * 100 if oi_hist_5m[-1 - buy_period] else 0

            if (price_change_pct_buy >= buy_params["price_change"] and
                volume_change_pct_buy >= buy_params["volume_change"] and
                oi_change_pct_buy >= buy_params["oi_change"]):
                action = "Buy"
                price_change_pct, volume_change_pct, oi_change_pct = price_change_pct_buy, volume_change_pct_buy, oi_change_pct_buy

        if action:
            signal_key = (symbol, action, 'golden_setup')
            if signal_key in bot.active_signals: return
            bot.active_signals.add(signal_key)

            features = await bot.extract_realtime_features(symbol)
            if not features: return

            cvd_change_pct = (cvd_hist_5m[-1] - cvd_hist_5m[-1 - period_iters]) / abs(cvd_hist_5m[-1 - period_iters] or 1) * 100
            
            candidate = {
                'symbol': symbol, 'side': action, 'source': 'golden_setup',
                'base_metrics': {
                    'pct_5m': price_change_pct, 'vol_change_pct': volume_change_pct,
                    'oi_change_pct': oi_change_pct, 'cvd_change_pct': cvd_change_pct
                }
            }
            funding_snap = bot._apply_funding_to_features(symbol, features)
            bot._apply_funding_to_candidate(candidate, funding_snap)
            
            asyncio.create_task(bot._process_signal(candidate, features, signal_key))

    except Exception as e:
        logger.error(f"[_golden_logic] unexpected error for {symbol}: {e}", exc_info=True)
        if signal_key:
            bot.active_signals.discard(signal_key)