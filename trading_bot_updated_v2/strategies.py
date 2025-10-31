from config import DOM_SQUEEZE_STRATEGY
from utils import SimpleClusterAnalyzer
from ai_ml import MLXPredictorManager
import logging

logger = logging.getLogger(__name__)


async def low_frequency_dispatcher(bot, symbol: str) -> bool:
    """
    Вызывает доступные стратегии с низкой частотой обновления.
    """
    try:
        return await mlx_enhanced_dom_strategy(bot, symbol)
    except Exception as e:
        logger.error(f"[{symbol}] Ошибка в low_frequency_dispatcher: {e}")
        return False


async def mlx_enhanced_dom_strategy(bot, symbol: str) -> bool:
    cfg = bot.user_data.get("dom_squeeze_settings", DOM_SQUEEZE_STRATEGY)
    if not cfg.get("ENABLED", False):
        return False

    if not hasattr(bot, 'mlx_predictor'):
        bot.mlx_predictor = MLXPredictorManager()
        bot.mlx_predictor.load_model()

    if not hasattr(bot, 'simple_cluster_analyzer'):
        bot.simple_cluster_analyzer = SimpleClusterAnalyzer()

    try:
        await _update_wall_memory(bot, symbol)
        closest_wall = await _find_closest_wall_zscore(bot, symbol, cfg)
        if not closest_wall:
            return False

        validated_wall = await _validate_sticky_wall(bot, symbol, closest_wall)
        if not validated_wall:
            return False

        orderbook = bot.shared_ws.orderbooks.get(symbol)
        last_price = _dom_get_last_price(bot, symbol)
        if not orderbook or not (last_price and last_price > 0):
            return False

        cluster = bot.simple_cluster_analyzer.analyze(orderbook)
        wall_price = validated_wall['price']
        wall_side  = validated_wall['side']
        wall_size  = validated_wall['qty']
        wall_z     = validated_wall.get('zscore', 0)

        # simple volume logic
        if wall_side == "Sell":
            mode = "breakout" if wall_price < last_price and wall_z >= 1.5 else "fade"
            side_to_enter = "Buy" if mode == "breakout" else "Sell"
        else:
            mode = "breakout" if wall_price > last_price and wall_z >= 1.5 else "fade"
            side_to_enter = "Sell" if mode == "breakout" else "Buy"

        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features:
            return False

        ok_guard, reason_guard = await bot._entry_guard(symbol, side_to_enter, mkt_features, {"source": f"mlx_dom_{mode}"})
        if not ok_guard:
            logger.info(f"[{symbol}] Сигнал заблокирован guard: {reason_guard}")
            return False

        signal_key = (symbol, side_to_enter, f"mlx_dom_{mode}")
        if signal_key in bot.active_signals:
            return True

        bot.active_signals.add(signal_key)
        logger.info(f"✅ DOM-СИГНАЛ [{symbol}] → режим: {mode.upper()}, вход: {side_to_enter}, стена: {wall_side} @ {wall_price:.6f}")

        candidate = {
            "symbol": symbol,
            "side": side_to_enter,
            "source": f"mlx_dom_{mode}",
            "wall_price": float(wall_price),
            "wall_rating": validated_wall.get("score", 0.5),
        }
        await bot._process_signal(candidate, mkt_features, signal_key)
        return True

    except Exception as e:
        logger.error(f"[{symbol}] Ошибка в mlx_enhanced_dom_strategy: {e}")
        return False