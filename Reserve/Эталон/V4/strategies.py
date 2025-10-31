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

# --- НАЧАЛО ИЗМЕНЕНИЙ: НОВАЯ СТРАТЕГИЯ ---

# --- strategies.py ---

async def intelligent_dom_strategy(bot, symbol: str):
    """
    [V4 - ЖЕЛЕЗНЫЙ ФИЛЬТР МОМЕНТА] Использует ADX+CVD для блокировки контр-трендовых
    сделок против зарождающегося сильного импульса.
    """
    cfg = bot.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY)
    if not cfg.get("ENABLED", False): return False
    
    signal_key = None
    try:
        # --- Шаг 1: Получаем все необходимые данные ---
        features = await bot.extract_realtime_features(symbol)
        if not features: return False

        # --- Шаг 2: Определяем потенциальный сигнал, который будем проверять ---
        trade_history = bot.shared_ws.trade_history.get(symbol)
        if not trade_history: return False
        
        current_time_ms = int(time.time() * 1000)
        recent_trades = [t for t in trade_history if t['t'] >= current_time_ms - cfg.get("SQUEEZE_LOOKBACK_MS", 5000)]
        if len(recent_trades) < 10: return False
        
        start_price = recent_trades[0]['p']
        end_price = recent_trades[-1]['p']
        if start_price <= 0: return False
        price_change_percent = ((end_price - start_price) / start_price) * 100
        threshold = cfg.get("SQUEEZE_THRESHOLD_PERCENT", 0.2)
        
        potential_side = None
        if price_change_percent > threshold: potential_side = "Sell" # Сквиз вверх, ищем шорт
        elif price_change_percent < -threshold: potential_side = "Buy" # Сквиз вниз, ищем лонг
        if not potential_side: return False

        # --- Шаг 3: "Железный фильтр момента" (ADX + CVD + Price Momentum) ---
        if bot.user_data.get("MOMENTUM_FILTER_ENABLED", True):
            adx_threshold = bot.user_data.get("MOMENTUM_FILTER_ADX_THRESHOLD", 25.0)
            pct5m_threshold = bot.user_data.get("MOMENTUM_FILTER_PCT5M_THRESHOLD", 1.5)
            
            adx = features.get('adx14', 0.0)
            pct5m = features.get('pct5m', 0.0)
            cvd5m = features.get('CVD5m', 0.0)

            # ЗАПРЕТ НА ШОРТ, если это сильный бычий пробой
            if potential_side == "Sell" and adx > adx_threshold and pct5m > pct5m_threshold and cvd5m > 0:
                logger.debug(f"⛔️ [{symbol}] Шорт-сигнал заблокирован фильтром МОМЕНТУМА (ADX={adx:.1f}, Pct5m={pct5m:.2f}%, CVD5m={cvd5m:.0f})")
                return False

            # ЗАПРЕТ НА ЛОНГ, если это сильный медвежий пробой
            if potential_side == "Buy" and adx > adx_threshold and pct5m < -pct5m_threshold and cvd5m < 0:
                logger.debug(f"⛔️ [{symbol}] Лонг-сигнал заблокирован фильтром МОМЕНТУМА (ADX={adx:.1f}, Pct5m={pct5m:.2f}%, CVD5m={cvd5m:.0f})")
                return False

        # --- Шаг 4: Если фильтр пройден, продолжаем с логикой "липкой" стены ---
        validated_wall = await _validate_sticky_wall(bot, symbol, potential_side, cfg)
        if not validated_wall: return False

        side_to_enter = validated_wall["side"]
        wall_price = validated_wall["price"]
        
        signal_key = (symbol, side_to_enter, 'intelligent_dom_squeeze_v4')
        if signal_key in bot.active_signals: return True
        
        bot.active_signals.add(signal_key)
        bot.strategy_cooldown_until[symbol] = time.time() + cfg.get("COOLDOWN_SECONDS", 300)

        logger.info(f"🧱✅ [{symbol}] INTELLIGENT DOM V4: Сигнал прошел фильтр момента. Запускаю Аналитика...")
        
        candidate = {'symbol': symbol, 'side': side_to_enter, 'source': 'dom_squeeze_v4', 'wall_price': wall_price}
        asyncio.create_task(bot._initiate_dom_hunt(candidate, signal_key))
        return True

    except Exception as e:
        logger.error(f"[intelligent_dom_strategy] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
        return False

async def _validate_sticky_wall(bot, symbol: str, side: str, cfg: dict) -> dict | None:
    """
    Вспомогательная функция для проверки "липкости" стены.
    Принимает уже определенное направление (side).
    """
    WALL_VALIDATION_DURATION_SEC = 10 # Стена должна простоять 10 секунд
    
    # Находим потенциальную стену для заданного направления
    potential_wall = await _find_potential_wall(bot, symbol, side, cfg)
    if not potential_wall: return None
        
    wall_price = potential_wall["price"]
    wall_initial_size = potential_wall["size"]
    
    # Наблюдаем за стеной в течение WALL_VALIDATION_DURATION_SEC
    for _ in range(WALL_VALIDATION_DURATION_SEC):
        await asyncio.sleep(1)
        orderbook = bot.shared_ws.orderbooks.get(symbol)
        if not orderbook: return None

        dom_to_scan = orderbook.get('asks', {}) if side == "Sell" else orderbook.get('bids', {})
        current_wall_size = dom_to_scan.get(wall_price, 0)
        
        # Если стена исчезла или уменьшилась более чем на 30% - это спуфинг
        if current_wall_size < wall_initial_size * 0.7:
            logger.debug(f"[{symbol}] Стена на {wall_price} оказалась 'спуф'-заявкой. Отмена.")
            return None
            
    return {"price": wall_price, "size": wall_initial_size, "side": side}

async def _find_potential_wall(bot, symbol: str, side: str, cfg: dict) -> dict | None:
    """Находит ближайшую крупную стену в стакане."""
    orderbook = bot.shared_ws.orderbooks.get(symbol)
    if not orderbook or 'bids' not in orderbook: return None

    if side == "Sell":
        dom_to_scan = orderbook.get('asks', {})
        sorted_levels = sorted(dom_to_scan.items()) 
    else: # Buy
        dom_to_scan = orderbook.get('bids', {})
        sorted_levels = sorted(dom_to_scan.items(), reverse=True)
    
    scan_depth = cfg.get("WALL_SCAN_DEPTH", 20)
    if len(sorted_levels) < scan_depth: return None

    top_levels = sorted_levels[:scan_depth]
    avg_size = sum(qty for _, qty in top_levels) / len(top_levels) if top_levels else 0
    if avg_size == 0: return None

    wall_multiplier = cfg.get("WALL_MULTIPLIER", 10)
    wall_threshold = avg_size * wall_multiplier

    for price, size in top_levels:
        if size > wall_threshold:
            return {"price": price, "size": size}
    return None




async def high_frequency_dispatcher(bot, symbol: str):
    if symbol not in bot.shared_ws.watchlist:
        return

    mode = bot.strategy_mode
    if mode == "scalp_only":
        await flea_strategy(bot, symbol)
        return
        
    if not await _prereqs_check(bot, symbol):
        return

    # 1. Стратегия по стакану (DOM Squeeze)
    # Запускается в режимах: full, squeeze_only (общий сквиз), dom_squeeze_only (только она)
    if mode in ("full", "squeeze_only", "dom_squeeze_only", "liq_squeeze"):
        if await intelligent_dom_strategy(bot, symbol):
            return # Если сработала, остальные ВЧ-стратегии не нужны

    # Если мы в режиме "только DOM", то дальше не идем
    if mode == "dom_squeeze_only":
        return

    # 2. Стратегия по ликвидациям
    # Запускается в режимах: full, liquidation_only, liq_squeeze
    if mode in ("full", "liquidation_only", "liq_squeeze"):
        if await liquidation_strategy(bot, symbol):
            return
            
    # 3. Стратегия по свечам (Candle Squeeze)
    # Запускается в режимах: full, squeeze_only, golden_squeeze, liq_squeeze
    if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
        await squeeze_strategy(bot, symbol)



async def low_frequency_dispatcher(bot, symbol: str):
    if symbol not in bot.shared_ws.watchlist:
        return

    mode = bot.strategy_mode
    if mode == "scalp_only":
        return
        
    if not await _prereqs_check(bot, symbol):
        return

async def _prereqs_check(bot, symbol: str) -> bool:
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
    signal_key = None
    try:
        if time.time() < bot._last_golden_ts.get(symbol, 0) + 300: return
        if symbol in bot.open_positions or symbol in bot.pending_orders: return
        if symbol in bot.recently_closed: return

        features = await bot.extract_realtime_features(symbol)
        if not features: return

        CONTEXT_WINDOW_MIN = 240
        MAX_CONTEXT_CHANGE_PCT = 6.0
        
        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < CONTEXT_WINDOW_MIN + 1: return
            
        context_change_pct = utils.compute_pct(candles, CONTEXT_WINDOW_MIN)

        last_candle = candles[-1]
        price_end = utils.safe_to_float(last_candle.get("closePrice"))
        open_price = utils.safe_to_float(last_candle.get("openPrice"))
        if open_price == 0: return
        side = "Buy" if price_end >= open_price else "Sell"

        if side == "Buy" and context_change_pct > MAX_CONTEXT_CHANGE_PCT:
            return
        if side == "Sell" and context_change_pct < -MAX_CONTEXT_CHANGE_PCT:
            return
        
        is_price_stable = abs(features.get('pct5m', 0.0)) < 0.8
        is_oi_growing = features.get('dOI5m', 0.0) * 100.0 > 1.0
        
        last_candle_volume = features.get('vol1m', 0)
        avg_volume_prev_4m = features.get('avg_volume_prev_4m', 0)
        if avg_volume_prev_4m == 0: return
        VOLUME_MULTIPLIER = 2.0
        is_volume_spike = last_candle_volume > (avg_volume_prev_4m * VOLUME_MULTIPLIER)

        if not (is_price_stable and is_oi_growing and is_volume_spike):
            return

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
        features = await bot.extract_realtime_features(symbol)
        if not features: return

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