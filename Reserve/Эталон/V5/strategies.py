# strategies.py
import logging
import time
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
from collections import defaultdict

import utils
import ai_ml
import config 

logger = logging.getLogger(__name__)


# --- Вспомогательные функции для V6 ---

async def _update_wall_memory(bot, symbol: str):
    if symbol not in bot.wall_watch_list: return
    wall_price, side, timestamp = bot.wall_watch_list[symbol]
    MEMORY_UPDATE_DELAY_SEC = 300
    if time.time() - timestamp > MEMORY_UPDATE_DELAY_SEC:
        try:
            candles = list(bot.shared_ws.candles_data.get(symbol, []))
            if not candles: 
                del bot.wall_watch_list[symbol]
                return
            relevant_candles = [c for c in candles if pd.to_datetime(c['startTime']).timestamp() > timestamp]
            price_tick = bot.price_tick_map.get(symbol, 0.0001)
            if price_tick == 0: price_tick = 0.0001
            price_cluster = round(wall_price / (price_tick * 100)) * (price_tick * 100)
            # --- НАЧАЛО ИЗМЕНЕНИЙ: Безопасный доступ к словарю ---
            if symbol not in bot.dom_wall_memory:
                bot.dom_wall_memory[symbol] = {}
            if price_cluster not in bot.dom_wall_memory[symbol]:
                bot.dom_wall_memory[symbol][price_cluster] = {'holds': 0, 'breaches': 0, 'last_seen': 0}
            memory = bot.dom_wall_memory[symbol][price_cluster]
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            was_breached = False
            for candle in relevant_candles:
                high, low = utils.safe_to_float(candle['highPrice']), utils.safe_to_float(candle['lowPrice'])
                if side == "Sell" and high > wall_price: was_breached = True; break
                if side == "Buy" and low < wall_price: was_breached = True; break
            if was_breached:
                memory['breaches'] += 1
                logger.info(f"🧠 [Память Стен] {symbol}: Уровень {price_cluster:.6f} был ПРОБИТ. Рейтинг понижен.")
            else:
                memory['holds'] += 1
                logger.info(f"🧠 [Память Стен] {symbol}: Уровень {price_cluster:.6f} УДЕРЖАЛ цену. Рейтинг повышен.")
            memory['last_seen'] = time.time()
        except Exception as e:
            logger.error(f"Ошибка при обновлении памяти стен для {symbol}: {e}", exc_info=True)
        finally:
            del bot.wall_watch_list[symbol]

def _get_rsi_from_candles(bot, symbol: str) -> float | None:
    candles = list(bot.shared_ws.candles_data.get(symbol, []))
    if len(candles) < 20: return None
    try:
        close_prices = pd.Series([utils.safe_to_float(c.get("closePrice")) for c in candles])
        rsi = ta.rsi(close_prices, length=14)
        if rsi is not None and not rsi.empty:
            return rsi.iloc[-1]
    except Exception as e:
        logger.debug(f"Ошибка при быстром расчете RSI для {symbol}: {e}")
    return None

async def _find_closest_wall(bot, symbol: str, cfg: dict) -> dict | None:
    orderbook = bot.shared_ws.orderbooks.get(symbol)
    last_price = utils.safe_to_float(bot.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
    if not orderbook or not last_price > 0: return None

    scan_depth = cfg.get("WALL_SCAN_DEPTH", 20)
    wall_multiplier = cfg.get("WALL_MULTIPLIER", 10)

    # Scan Asks (resistance)
    asks = sorted(orderbook.get('asks', {}).items())
    if len(asks) > scan_depth:
        top_asks = asks[:scan_depth]
        avg_ask_size = sum(qty for _, qty in top_asks) / len(top_asks)
        if avg_ask_size > 0:
            ask_wall_threshold = avg_ask_size * wall_multiplier
            for price, size in top_asks:
                if size > ask_wall_threshold:
                    ask_wall = {"price": price, "size": size, "side": "Sell", "distance": abs(price - last_price)}
                    break
            else: ask_wall = None
    else: ask_wall = None

    # Scan Bids (support)
    bids = sorted(orderbook.get('bids', {}).items(), reverse=True)
    if len(bids) > scan_depth:
        top_bids = bids[:scan_depth]
        avg_bid_size = sum(qty for _, qty in top_bids) / len(top_bids)
        if avg_bid_size > 0:
            bid_wall_threshold = avg_bid_size * wall_multiplier
            for price, size in top_bids:
                if size > bid_wall_threshold:
                    bid_wall = {"price": price, "size": size, "side": "Buy", "distance": abs(price - last_price)}
                    break
            else: bid_wall = None
    else: bid_wall = None

    if ask_wall and bid_wall:
        return ask_wall if ask_wall['distance'] < bid_wall['distance'] else bid_wall
    return ask_wall or bid_wall

async def _validate_sticky_wall(bot, symbol: str, wall_data: dict) -> dict | None:
    wall_price = wall_data["price"]
    side = wall_data["side"]
    
    price_tick = bot.price_tick_map.get(symbol, 0.0001)
    if price_tick == 0: price_tick = 0.0001
    price_cluster = round(wall_price / (price_tick * 100)) * (price_tick * 100)
    memory = bot.dom_wall_memory.get(symbol, {}).get(price_cluster, {'holds': 0, 'breaches': 0})
    rating = memory['holds'] - (memory['breaches'] * 2)

    MIN_RATING_THRESHOLD = 0
    if rating < MIN_RATING_THRESHOLD:
        logger.debug(f"[{symbol}] Стена на {wall_price:.6f} проигнорирована. Отрицательный рейтинг: {rating}")
        return None

    WALL_VALIDATION_DURATION_SEC, wall_initial_size = 10, wall_data["size"]
    for _ in range(WALL_VALIDATION_DURATION_SEC):
        await asyncio.sleep(1)
        orderbook = bot.shared_ws.orderbooks.get(symbol)
        if not orderbook: return None
        dom_to_scan = orderbook.get('asks', {}) if side == "Sell" else orderbook.get('bids', {})
        if dom_to_scan.get(wall_price, 0) < wall_initial_size * 0.7:
            logger.debug(f"[{symbol}] Стена на {wall_price:.6f} оказалась 'спуф'-заявкой. Отмена.")
            return None
            
    wall_data['rating'] = rating
    return wall_data

# --- Основные стратегии ---

async def intelligent_dom_strategy(bot, symbol: str):
    """
    [V6 - НАДЕЖНЫЙ ТРИГГЕР] Постоянно ищет ближайшую сильную стену и затем
    оценивает рыночный контекст, вместо того чтобы ждать редкого импульса в ленте.
    """
    cfg = bot.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY)
    if not cfg.get("ENABLED", False): return False
    signal_key = None
    try:
        await _update_wall_memory(bot, symbol)

        # Шаг 1: Найти ближайшую потенциальную стену
        closest_wall = await _find_closest_wall(bot, symbol, cfg)
        if not closest_wall: return False

        # Шаг 2: Проверить ее "липкость" и историческую репутацию
        validated_wall = await _validate_sticky_wall(bot, symbol, closest_wall)
        if not validated_wall: return False
        
        potential_side = validated_wall["side"]
        
        # Шаг 3: Только теперь, когда есть надежная стена, запускаем полный анализ
        features = await bot.extract_realtime_features(symbol)
        if not features: return False

        # Шаг 4: Применяем "Железный фильтр момента"
        if bot.user_data.get("MOMENTUM_FILTER_ENABLED", True):
            adx_threshold = bot.user_data.get("MOMENTUM_FILTER_ADX_THRESHOLD", 25.0)
            pct5m_threshold = bot.user_data.get("MOMENTUM_FILTER_PCT5M_THRESHOLD", 1.5)
            adx, pct5m, cvd5m = features.get('adx14',0), features.get('pct5m',0), features.get('CVD5m',0)
            
            # ЗАПРЕТ НА КОНТРТРЕНД против сильного импульса
            if (potential_side == "Sell" and adx > adx_threshold and pct5m > pct5m_threshold and cvd5m > 0) or \
               (potential_side == "Buy" and adx > adx_threshold and pct5m < -pct5m_threshold and cvd5m < 0):
                logger.debug(f"⛔️ [{symbol}] DOM сигнал на {potential_side} заблокирован фильтром МОМЕНТУМА.")
                return False

        # Шаг 5: Если все проверки пройдены, создаем сигнал
        side_to_enter, wall_price = validated_wall["side"], validated_wall["price"]
        signal_key = (symbol, side_to_enter, 'intelligent_dom_squeeze_v6')
        if signal_key in bot.active_signals: return True
        
        bot.active_signals.add(signal_key)
        bot.strategy_cooldown_until[symbol] = time.time() + cfg.get("COOLDOWN_SECONDS", 300)
        bot.wall_watch_list[symbol] = (wall_price, side_to_enter, time.time())
        
        logger.info(f"🧱✅ [{symbol}] DOM V6: Найдена надежная стена ({side_to_enter}) @ {wall_price:.6f}. Рейтинг: {validated_wall['rating']}. Запускаю Охотника-Аналитика...")
        candidate = {
            'symbol': symbol, 
            'side': side_to_enter, 
            'source': 'dom_squeeze_v6', 
            'wall_price': wall_price,
            'wall_rating': validated_wall['rating'] # <--- ДОБАВЛЕНО ЭТО ПОЛЕ
        }
        asyncio.create_task(bot._hunt_reversal(candidate, features, signal_key))

        return True

    except Exception as e:
        logger.error(f"[intelligent_dom_strategy_v6] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
        return False

async def squeeze_strategy(bot, symbol: str):
    """
    [V5 - ОПТИМИЗИРОВАННАЯ] Сначала быстро проверяет RSI, и только потом
    запускает полный расчет фичей, чтобы избежать перегрузки CPU.
    """
    signal_key = None
    try:
        if not bot._squeeze_allowed(symbol): return False
        rsi_val = _get_rsi_from_candles(bot, symbol)
        if rsi_val is None or not ((rsi_val > 75) or (rsi_val < 25)):
            return False
        
        features = await bot.extract_realtime_features(symbol)
        if not features: return False

        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < 1: return False
        
        last_price = features.get("price")
        pct1m = features.get("pct1m")
        if last_price is None or pct1m is None: return False

        SQUEEZE_INTRA_MINUTE_THRESHOLD = 2.0
        if abs(pct1m) < SQUEEZE_INTRA_MINUTE_THRESHOLD:
            return False

        side = "Sell" if rsi_val > 75 else "Buy"
        if not ((side == "Sell" and pct1m > 0) or (side == "Buy" and pct1m < 0)): return False

        bot.last_squeeze_ts[symbol] = time.time()
        signal_key = (symbol, side, 'squeeze_intra_minute')
        if signal_key in bot.active_signals: return True
        bot.active_signals.add(signal_key)
        
        logger.info(f"🔥 [{symbol}] ОБНАРУЖЕН ИНТРА-СВЕЧНОЙ СКВИЗ! ΔЦена 1m: {pct1m:.2f}%. Передано AI.")
        candidate = {"symbol": symbol, "side": side, "source": "squeeze", "base_metrics": {'pct_1m_intra_candle': pct1m}}
        asyncio.create_task(bot._process_signal(candidate, features, signal_key))
        return True
    except Exception as e:
        logger.error(f"[squeeze_strategy_v5] Критическая ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
        return False

# --- Диспетчеры и остальные стратегии (без изменений) ---

async def high_frequency_dispatcher(bot, symbol: str):
    if symbol not in bot.shared_ws.watchlist: return
    mode = bot.strategy_mode
    if mode == "scalp_only":
        await flea_strategy(bot, symbol)
        return
    if not await _prereqs_check(bot, symbol): return
    if mode in ("full", "squeeze_only", "dom_squeeze_only", "liq_squeeze"):
        if await intelligent_dom_strategy(bot, symbol): return
    if mode == "dom_squeeze_only": return
    if mode in ("full", "liquidation_only", "liq_squeeze"):
        if await liquidation_strategy(bot, symbol): return
    if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
        await squeeze_strategy(bot, symbol)

async def low_frequency_dispatcher(bot, symbol: str):
    if symbol not in bot.shared_ws.watchlist: return
    mode = bot.strategy_mode
    if mode == "scalp_only": return
    if not await _prereqs_check(bot, symbol): return

async def _prereqs_check(bot, symbol: str) -> bool:
    if symbol not in bot.shared_ws.watchlist: return False
    now = time.time()
    if now < bot.strategy_cooldown_until.get(symbol, 0): return False
    if symbol in bot.open_positions or symbol in bot.pending_orders: return False
    if symbol in bot.recently_closed and (now - bot.recently_closed[symbol] < 900): return False
    age = await bot.listing_age_minutes(symbol)
    if age < bot.listing_age_min: return False
    if symbol in bot.failed_orders and (now - bot.failed_orders.get(symbol, 0) < 600): return False
    return True

async def liquidation_strategy(bot, symbol: str) -> bool:
    signal_key = None
    try:
        liq_buffer = bot.liq_buffers.get(symbol)
        if not liq_buffer: return False
        now = time.time()
        recent_events = [evt for evt in liq_buffer if now - evt['ts'] <= 10.0]
        if len(recent_events) < 2: return False
        buy_liq_value = sum(evt['value'] for evt in recent_events if evt['side'] == 'Buy')
        sell_liq_value = sum(evt['value'] for evt in recent_events if evt['side'] == 'Sell')
        threshold = bot.shared_ws.get_liq_threshold(symbol)
        entry_side, cluster_value, dominant_side = None, 0, ''
        if buy_liq_value >= threshold:
            entry_side, cluster_value, dominant_side = "Sell", buy_liq_value, "Buy"
        elif sell_liq_value >= threshold:
            entry_side, cluster_value, dominant_side = "Buy", sell_liq_value, "Sell"
        if not entry_side: return False
        signal_key = (symbol, entry_side, 'liquidation_cluster')
        if signal_key in bot.active_signals: return True
        bot.active_signals.add(signal_key)
        bot.shared_ws.last_liq_trade_time[symbol] = time.time()
        logger.info(f"💧 [{symbol}] КЛАСТЕР ЛИКВИДАЦИЙ ({dominant_side})! ${cluster_value:,.0f}. Вход в {entry_side}.")
        features = await bot.extract_realtime_features(symbol)
        if not features:
            bot.active_signals.discard(signal_key)
            return False
        candidate = {'symbol': symbol, 'side': entry_side, 'source': 'liquidation_cascade', 'base_metrics': {'liquidation_value_usd': cluster_value, 'liquidation_side': dominant_side}}
        asyncio.create_task(bot._process_signal(candidate, features, signal_key))
        return True
    except Exception as e:
        logger.error(f"[liquidation_strategy] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
    return False

async def golden_strategy(bot, symbol: str):
    signal_key = None
    try:
        if time.time() < bot._last_golden_ts.get(symbol, 0) + 300: return
        if symbol in bot.open_positions or symbol in bot.pending_orders or symbol in bot.recently_closed: return
        features = await bot.extract_realtime_features(symbol)
        if not features: return
        CONTEXT_WINDOW_MIN, MAX_CONTEXT_CHANGE_PCT = 240, 6.0
        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < CONTEXT_WINDOW_MIN + 1: return
        context_change_pct = utils.compute_pct(candles, CONTEXT_WINDOW_MIN)
        price_end = utils.safe_to_float(candles[-1].get("closePrice"))
        open_price = utils.safe_to_float(candles[-1].get("openPrice"))
        if open_price == 0: return
        side = "Buy" if price_end >= open_price else "Sell"
        if (side == "Buy" and context_change_pct > MAX_CONTEXT_CHANGE_PCT) or \
           (side == "Sell" and context_change_pct < -MAX_CONTEXT_CHANGE_PCT): return
        is_price_stable = abs(features.get('pct5m', 0.0)) < 0.8
        is_oi_growing = features.get('dOI5m', 0.0) * 100.0 > 1.0
        last_candle_volume = features.get('vol1m', 0)
        avg_volume_prev_4m = features.get('avg_volume_prev_4m', 0)
        if avg_volume_prev_4m == 0: return
        is_volume_spike = last_candle_volume > (avg_volume_prev_4m * 2.0)
        if not (is_price_stable and is_oi_growing and is_volume_spike): return
        bot._last_golden_ts[symbol] = time.time()
        signal_key = (symbol, side, 'golden_setup_v2')
        if signal_key in bot.active_signals: return
        bot.active_signals.add(signal_key)
        vol_spike_ratio = last_candle_volume / avg_volume_prev_4m
        logger.info(f"🏆 [{symbol}] GOLDEN SETUP 2.0! Контекст 4ч: {context_change_pct:+.2f}%. Vol Spike: x{vol_spike_ratio:.1f}. Направление: {side}.")
        candidate = {"symbol": symbol, "side": side, "source": "golden_setup", "base_metrics": { 'oi_change_5m_pct': features.get('dOI5m', 0.0) * 100.0 }}
        asyncio.create_task(bot._process_signal(candidate, features, signal_key))
    except Exception as e:
        logger.error(f"[golden_strategy_v2] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)

async def flea_strategy(bot, symbol: str):
    cfg = bot.user_data.get("flea_settings", config.FLEA_STRATEGY)
    if not cfg.get("ENABLED", False): return
    now = time.time()
    if bot.flea_positions_count >= cfg.get("MAX_OPEN_POSITIONS", 15): return
    if now < bot.flea_cooldown_until.get(symbol, 0): return
    bot.flea_cooldown_until[symbol] = now + 5 
    if symbol in bot.open_positions or symbol in bot.pending_orders or (symbol in bot.recently_closed and now - bot.recently_closed[symbol] < 300): return
    try:
        features = await bot.extract_realtime_features(symbol)
        if not features: return
        fast_ema, slow_ema = features.get('fast_ema', 0), features.get('slow_ema', 0)
        fast_ema_prev, slow_ema_prev = features.get('fast_ema_prev', 0), features.get('slow_ema_prev', 0)
        rsi, atr, trend_ema, last_price = features.get('rsi14', 50), features.get('atr14', 0), features.get('trend_ema', 0), features.get('price', 0)
        if any(v == 0 for v in [fast_ema, slow_ema, trend_ema, last_price, atr]): return
        is_uptrend, is_downtrend = last_price > trend_ema, last_price < trend_ema
        volatility_ok = cfg.get("MIN_ATR_PCT", 0.05) < (atr / last_price * 100) < cfg.get("MAX_ATR_PCT", 1.5)
        side = None
        if fast_ema_prev < slow_ema_prev and fast_ema > slow_ema and rsi > 51 and is_uptrend: side = "Buy"
        elif fast_ema_prev > slow_ema_prev and fast_ema < slow_ema and rsi < 49 and is_downtrend: side = "Sell"
        if not side or not volatility_ok: return
        tp_offset = atr * cfg.get("TP_ATR_MULTIPLIER", 1.5)
        sl_offset = atr * cfg.get("STOP_LOSS_ATR_MULTIPLIER", 1.0)
        tp_price = last_price + tp_offset if side == "Buy" else last_price - tp_offset
        sl_price = last_price - sl_offset if side == "Buy" else last_price + sl_offset
        candidate = {'symbol': symbol, 'side': side, 'source': 'flea_v2.1', 'take_profit_price': tp_price, 'stop_loss_price': sl_price, 'max_hold_minutes': cfg.get("MAX_HOLD_MINUTES", 10)}
        logger.info(f"🦟 [{symbol}] 'Блоха 2.1' поймала сигнал в {side}. TP={tp_price:.6f}, SL={sl_price:.6f}")
        await bot.execute_flea_trade(candidate)
    except Exception as e:
        logger.error(f"[flea_strategy_v2.1] Ошибка для {symbol}: {e}", exc_info=True)