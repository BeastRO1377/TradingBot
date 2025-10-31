# strategies.py
import logging
import time
import asyncio
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pandas_ta as ta
import pickle
import os
import utils
import ai_ml
import config 
import bot_core

import math
import statistics
logger = logging.getLogger(__name__)

def init_bot_memory(bot):
    """Инициализация всей памяти бота"""
    if not hasattr(bot, 'dom_wall_memory'):
        bot.dom_wall_memory = {}
    
    if not hasattr(bot, 'wall_watch_list'):
        bot.wall_watch_list = {}
        
    load_wall_memory(bot)
    
    logger.info(f"🧠 Память инициализирована: {len(bot.dom_wall_memory)} символов")

def _dom_get_last_price(bot, symbol: str) -> float:
    """Единообразно достаём last_price для стратегий DOM."""
    td = (getattr(bot.shared_ws, "ticker_data", {}) or {}).get(symbol, {}) or {}
    lp = td.get("last_price") or td.get("lastPrice") or td.get("markPrice") or td.get("indexPrice")
    return utils.safe_to_float(lp)

class MLXWallBreakoutPredictor(nn.Module):
    def __init__(self, input_dim=14, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2)) # 0 - Hold, 1 - Breakout
        self.layers = nn.Sequential(*layers)
    
    def __call__(self, x):
        return self.layers(x)

class MLXPredictorManager:
    def __init__(self, model_path="wall_breakout_mlx.npz"):
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.losses.cross_entropy
        self.is_trained = False
        self.model_path = model_path
        self.feature_columns = [
            'wall_size_ratio', 'volume_imbalance', 'rsi', 'price_distance_pct',
            'market_depth_ratio', 'recent_volatility', 'cluster_strength',
            'time_of_day', 'wall_rating', 'momentum_5m', 'adx_strength',
            'volume_ratio', 'orderbook_imbalance', 'spread_pct'
        ]
        self.training_buffer = deque(maxlen=2000)
        self.init_model()

    def init_model(self):
        try:
            self.model = MLXWallBreakoutPredictor(input_dim=len(self.feature_columns))
            self.optimizer = optim.Adam(learning_rate=0.001)
            logger.info("🧠 MLX модель инициализирована.")
        except Exception as e:
            logger.error(f"Ошибка инициализации MLX модели: {e}")

    def prepare_mkt_features(self, wall_data: dict, market_context: dict, cluster_data: dict = None):
        mkt_features = {}
        mkt_features['wall_size_ratio'] = min(wall_data.get('size_ratio', 1.0) / 20.0, 1.0)
        mkt_features['wall_rating'] = wall_data.get('rating', 0) / 10.0
        mkt_features['price_distance_pct'] = min(wall_data.get('distance_pct', 0) / 0.5, 1.0)
        mkt_features['rsi'] = market_context.get('rsi', 50) / 100.0
        mkt_features['adx_strength'] = min(market_context.get('adx', 0) / 100.0, 1.0)
        mkt_features['momentum_5m'] = np.clip(market_context.get('pct5m', 0) / 5.0, -1.0, 1.0)
        mkt_features['recent_volatility'] = min(market_context.get('atr_pct', 0) / 5.0, 1.0)
        mkt_features['volume_imbalance'] = market_context.get('volume_imbalance', 0)
        mkt_features['market_depth_ratio'] = market_context.get('depth_ratio', 1.0)
        mkt_features['orderbook_imbalance'] = market_context.get('orderbook_imbalance', 0)
        mkt_features['spread_pct'] = min(market_context.get('spread_pct', 0) / 1.0, 1.0)
        mkt_features['volume_ratio'] = min(market_context.get('volume_ratio', 1.0), 5.0) / 5.0
        mkt_features['cluster_strength'] = min(cluster_data.get('cluster_strength', 0) / 1000.0, 1.0) if cluster_data else 0
        hour = time.localtime().tm_hour
        mkt_features['time_of_day'] = (hour * 60 + time.localtime().tm_min) / 1440.0
        feature_vector = [mkt_features.get(col, 0.0) for col in self.feature_columns]
        return mx.array(feature_vector, dtype=mx.float32).reshape(1, -1)

    async def predict_breakout_probability(self, mkt_features: mx.array) -> float:
        if not self.is_trained or self.model is None:
            return 0.5
        try:
            logits = self.model(mkt_features)
            probs = mx.softmax(logits, axis=-1)
            mx.eval(probs)
            prob_breakout = probs[0, 1].item()
            return max(0.0, min(1.0, prob_breakout))
        except Exception as e:
            logger.debug(f"Ошибка MLX прогнозирования: {e}")
            return 0.5

    def save_model(self):
        if self.model and self.is_trained:
            try:
                self.model.save_weights(self.model_path)
                logger.info(f"MLX модель сохранена: {self.model_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения MLX модели: {e}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Файл модели {self.model_path} не найден. Модель не будет загружена.")
            return
        try:
            if self.model is None: self.init_model()
            self.model.load_weights(self.model_path)
            self.is_trained = True
            logger.info(f"MLX модель успешно загружена из {self.model_path}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить MLX модель: {e}")


class EnhancedVolumeClusterAnalyzer:
    def __init__(self, window_size=12):
        self.window_size = window_size
        self.volume_clusters = defaultdict(lambda: deque(maxlen=window_size))

    def update_cluster_data(self, symbol: str, orderbook: dict):
        try:
            bids = orderbook.get('bids', {})
            asks = orderbook.get('asks', {})

            top_bids = dict(sorted(bids.items(), key=lambda kv: kv[0], reverse=True)[:60])
            top_asks = dict(sorted(asks.items(), key=lambda kv: kv[0])[:60])

            bid_clusters = self._find_volume_clusters(top_bids, "Buy")
            ask_clusters = self._find_volume_clusters(top_asks, "Sell")

            best_bid = max(bids.keys()) if bids else None
            best_ask = min(asks.keys()) if asks else None
            q_bid    = bids.get(best_bid, 0.0) if best_bid is not None else 0.0
            q_ask    = asks.get(best_ask, 0.0) if best_ask is not None else 0.0

            self.volume_clusters[symbol].append({
                'ts': time.time(),
                'bid_clusters': bid_clusters,
                'ask_clusters': ask_clusters,
                'best_bid': best_bid, 'best_ask': best_ask,
                'q_bid': q_bid, 'q_ask': q_ask,
                'bids': top_bids, 'asks': top_asks,
            })
            return len(bid_clusters) + len(ask_clusters)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Ошибка обновления кластеров {symbol}: {e}")
            return 0

    def _find_volume_clusters(self, levels: dict, side: str, min_cluster_size=3, price_gap_threshold=2):
        if not levels: return []
        prices = sorted(levels.keys(), reverse=(side=="Buy"))
        clusters, cur = [], []
        for i, p in enumerate(prices):
            v = levels[p]
            if not cur:
                cur.append((p, v)); continue
            last_p = cur[-1][0]
            if abs(p - last_p) <= price_gap_threshold * max(1e-9, abs(prices[0]-prices[-1]) / max(1, len(prices)-1)):
                cur.append((p, v))
            else:
                if len(cur) >= min_cluster_size: clusters.append(self._analyze_cluster(cur, side))
                cur = [(p, v)]
        if len(cur) >= min_cluster_size: clusters.append(self._analyze_cluster(cur, side))
        return clusters

    def _analyze_cluster(self, cluster: list, side: str):
        prices = [p for p,_ in cluster]
        vols   = [v for _,v in cluster]
        tot = sum(vols); n = len(vols)
        center = sum(p*v for p,v in cluster) / tot if tot>0 else np.mean(prices)
        density = tot / max(1, n)
        diffs = np.diff(vols) if n>1 else np.array([0.0])
        density_gradient = float(np.mean(diffs))
        return {
            'side': side,
            'price_range': (min(prices), max(prices)),
            'center_price': center,
            'total_volume': float(tot),
            'num_levels': int(n),
            'density': float(density),
            'density_gradient': density_gradient,
        }

    def calculate_advanced_metrics(self, symbol: str, current_price: float):
        dq = self.volume_clusters.get(symbol)
        if not dq: return {}

        latest = dq[-1]
        clusters = latest.get('bid_clusters', []) + latest.get('ask_clusters', [])
        if not clusters: return {'confidence_score': 0.0}

        def dist_to_price(c): 
            lo, hi = c['price_range']
            if lo <= current_price <= hi: return 0.0
            return min(abs(current_price-lo), abs(current_price-hi))
        clusters_sorted = sorted(clusters, key=dist_to_price)
        top = clusters_sorted[:3]

        avg_density = float(np.mean([c['density'] for c in top])) if top else 0.0
        spread = (latest['best_ask'] - latest['best_bid']) if latest['best_ask'] and latest['best_bid'] else 0.0
        price_spread_pct = (spread / current_price * 100.0) if current_price>0 and spread>0 else 0.0

        tick = 1e-6
        now = time.time()
        near_any = 0.0
        if len(dq) >= 2:
            snaps = list(dq)[-6:]
            last_ts = snaps[-1]['ts']
            for s in snaps:
                cls = s.get('bid_clusters', []) + s.get('ask_clusters', [])
                if any(dist_to_price(c) <= 3*tick for c in cls):
                    near_any += 1
            persistence_secs = (now - snaps[0]['ts']) * (near_any / max(1,len(snaps)))
        else:
            persistence_secs = 0.0

        bids = latest.get('bids', {})
        asks = latest.get('asks', {})
        def cum_behind(side, wall_price):
            if side == "Sell":
                keys = sorted([p for p in asks.keys() if p > wall_price])[:12]
                return sum(asks[k] for k in keys)
            else:
                keys = sorted([p for p in bids.keys() if p < wall_price], reverse=True)[:12]
                return sum(bids[k] for k in keys)

        dominant = max(top, key=lambda c: c['total_volume']) if top else None
        if dominant:
            wall_price = dominant['center_price']
            side = dominant['side']
            behind = cum_behind(side, wall_price)
            baseline = sum((asks if side=="Sell" else bids).values()) / 10.0 if (asks or bids) else 1.0
            liquidity_vacuum_ratio = float(behind) / max(1e-9, baseline)
        else:
            liquidity_vacuum_ratio = 1.0

        refill_rate = 0.0
        if len(dq) >= 2 and dominant:
            prev = dq[-2]
            def nearest_volume(snap):
                pools = (snap.get('bid_clusters', []) + snap.get('ask_clusters', []))
                if not pools: return 0.0
                pools = sorted(pools, key=lambda c: abs(c['center_price'] - dominant['center_price']))
                return float(pools[0]['total_volume'])
            dv = nearest_volume(latest) - nearest_volume(prev)
            dt = latest['ts'] - prev['ts']
            refill_rate = dv / max(1e-6, dt)

        best_bid, best_ask = latest.get('best_bid'), latest.get('best_ask')
        q_bid, q_ask = latest.get('q_bid', 0.0), latest.get('q_ask', 0.0)
        if best_bid and best_ask and (q_bid + q_ask) > 0:
            mid = 0.5*(best_bid+best_ask)
            micro = (best_ask*q_bid + best_bid*q_ask) / (q_bid + q_ask)
            microprice_tilt = (micro - mid)
        else:
            microprice_tilt = 0.0

        dens_score = np.tanh(avg_density / (1e4))
        pers_score = np.tanh(persistence_secs / 10.0)
        refill_score = np.tanh(max(0.0, refill_rate) / 5e3)
        vac_score = np.clip(1.0 - np.tanh(liquidity_vacuum_ratio), 0, 1)
        micro_score = np.tanh(abs(microprice_tilt) / 1e-3)

        confidence = 0.35*dens_score + 0.25*pers_score + 0.2*vac_score + 0.15*refill_score + 0.05*micro_score
        return {
            'avg_density': avg_density,
            'price_spread_pct': price_spread_pct,
            'persistence_secs': float(persistence_secs),
            'liquidity_vacuum_ratio': float(liquidity_vacuum_ratio),
            'refill_rate': float(refill_rate),
            'microprice_tilt': float(microprice_tilt),
            'confidence_score': float(np.clip(confidence, 0, 1)),
        }


async def _update_wall_memory(bot, symbol: str):
    if symbol not in bot.wall_watch_list: return
    wall_price, side, timestamp = bot.wall_watch_list[symbol]
    if time.time() - timestamp > 300:
        try:
            candles = list(bot.shared_ws.candles_data.get(symbol, []))
            if not candles: del bot.wall_watch_list[symbol]; return
            relevant_candles = [c for c in candles if pd.to_datetime(c['startTime']).timestamp() > timestamp]
            price_tick = bot.price_tick_map.get(symbol, 1e-4)
            if price_tick == 0: price_tick = 1e-4
            price_cluster = round(wall_price / (price_tick * 100)) * (price_tick * 100)
            if symbol not in bot.dom_wall_memory: bot.dom_wall_memory[symbol] = {}
            if price_cluster not in bot.dom_wall_memory[symbol]: bot.dom_wall_memory[symbol][price_cluster] = {'holds': 0, 'breaches': 0, 'last_seen': 0}
            memory = bot.dom_wall_memory[symbol][price_cluster]
            was_breached = False
            for candle in relevant_candles:
                high, low = utils.safe_to_float(candle['highPrice']), utils.safe_to_float(candle['lowPrice'])
                if side == "Sell" and high > wall_price: was_breached = True; break
                if side == "Buy" and low < wall_price: was_breached = True; break
            if was_breached: memory['breaches'] += 1; logger.info(f"🧠💥 [{symbol}] Уровень {price_cluster:.6f} ПРОБИТ.")
            else: memory['holds'] += 1; logger.info(f"🧠✅ [{symbol}] Уровень {price_cluster:.6f} УДЕРЖАН.")
            memory['last_seen'] = time.time()
        except Exception as e: logger.error(f"Ошибка обновления памяти стен: {e}", exc_info=True)
        finally: del bot.wall_watch_list[symbol]


def _get_rsi_from_candles(bot, symbol: str) -> float | None:
    candles = list(bot.shared_ws.candles_data.get(symbol, []))
    if len(candles) < 20: return None
    try:
        close_prices = pd.Series([utils.safe_to_float(c.get("closePrice")) for c in candles])
        rsi = ta.rsi(close_prices, length=14)
        return rsi.iloc[-1] if rsi is not None and not rsi.empty else None
    except: return None

async def _find_closest_wall_zscore(bot, symbol: str, cfg: dict):
    orderbook = bot.shared_ws.orderbooks.get(symbol)
    last_price = utils.safe_to_float(bot.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
    if not orderbook or not (last_price and last_price > 0):
        return None
        
    n = int(cfg.get("ZSCORE_LEVELS_N", 20))
    z_min = float(cfg.get("ZSCORE_MIN", 2.0))
    tick = bot.price_tick_map.get(symbol, 1e-4) or 1e-4

    def top_levels(side: str):
        levels_dict = orderbook.get('asks' if side == "Sell" else 'bids', {})
        if side == "Sell":
            return sorted(levels_dict.items())[:n]
        else:
            return sorted(levels_dict.items(), reverse=True)[:n]

    def best_wall_for_side(side: str):
        levels = top_levels(side)
        if not levels or len(levels) < 2: return None
            
        sizes = [q for _, q in levels]
        mean = statistics.fmean(sizes)
        std = statistics.pstdev(sizes) or 1.0
        
        candidates = []
        for p, q in levels:
            z = (q - mean) / std if std > 0 else 0
            if z >= z_min:
                dist_ticks = abs(p - last_price) / tick
                candidates.append({
                    "side": side, "price": float(p), "size": float(q),
                    "zscore": float(z), "rating": float(z), "dist_ticks": dist_ticks
                })
        
        if not candidates: return None
        candidates.sort(key=lambda c: (-c["zscore"], c["dist_ticks"]))
        return candidates[0]

    sell_wall = best_wall_for_side("Sell")
    buy_wall = best_wall_for_side("Buy")

    if sell_wall and buy_wall:
        s, b = sell_wall, buy_wall
        return s if (s["zscore"], -s["dist_ticks"]) >= (b["zscore"], -b["dist_ticks"]) else b
        
    return sell_wall or buy_wall


async def _validate_sticky_wall(bot, symbol: str, wall_data: dict) -> dict | None:
    wall_price, side = wall_data["price"], wall_data["side"]
    price_tick = bot.price_tick_map.get(symbol, 1e-4)
    if price_tick == 0: price_tick = 1e-4
    price_cluster = round(wall_price / (price_tick * 100)) * (price_tick * 100)
    memory = bot.dom_wall_memory.get(symbol, {}).get(price_cluster, {'holds': 0, 'breaches': 0})
    rating = memory['holds'] - (memory['breaches'] * 2)
    if rating < 0:
        logger.debug(f"[{symbol}] Стена на {wall_price:.6f} проигнорирована. Отрицательный рейтинг: {rating}")
        return None
    wall_initial_size = wall_data["size"]
    for _ in range(10):
        await asyncio.sleep(1)
        orderbook = bot.shared_ws.orderbooks.get(symbol)
        if not orderbook: return None
        dom_to_scan = orderbook.get('asks', {}) if side == "Sell" else orderbook.get('bids', {})
        if dom_to_scan.get(wall_price, 0) < wall_initial_size * 0.7:
            logger.debug(f"[{symbol}] Стена на {wall_price:.6f} оказалась 'спуф'-заявкой.")
            return None
    wall_data['rating'] = rating
    return wall_data



async def _prepare_mlx_market_context(bot, symbol: str, mkt_features: dict, orderbook: dict) -> dict:
    last_price = mkt_features.get('price', 0)
    orderbook_imbalance = 0
    if orderbook:
        bids_volume = sum(orderbook.get('bids', {}).values())
        asks_volume = sum(orderbook.get('asks', {}).values())
        total_volume = bids_volume + asks_volume
        if total_volume > 0:
            orderbook_imbalance = (bids_volume - asks_volume) / total_volume
    spread_pct = 0
    if orderbook and last_price > 0:
        bids = sorted(orderbook.get('bids', {}).keys())
        asks = sorted(orderbook.get('asks', {}).keys())
        if bids and asks:
            spread = asks[0] - bids[0]
            spread_pct = (spread / last_price) * 100
    return {
        'rsi': mkt_features.get('rsi14', 50),
        'adx': mkt_features.get('adx14', 0),
        'pct5m': mkt_features.get('pct5m', 0),
        'atr_pct': (mkt_features.get('atr14', 0) / last_price * 100) if last_price > 0 else 0,
        'volume_imbalance': mkt_features.get('volume_imbalance', 0),
        'depth_ratio': mkt_features.get('depth_ratio', 1.0),
        'orderbook_imbalance': orderbook_imbalance,
        'spread_pct': spread_pct,
        'volume_ratio': mkt_features.get('volume_ratio', 1.0)
    }


async def _mlx_enhanced_filters(bot, symbol: str, side: str, mkt_features: dict, 
                              mlx_confidence: float, cluster_metrics: dict) -> bool:
    base_adx_threshold = 25.0
    base_momentum_threshold = 1.5
    confidence_boost = mlx_confidence * 0.5
    adx_threshold = base_adx_threshold * (1 + confidence_boost)
    momentum_threshold = base_momentum_threshold * (1 + confidence_boost)
    adx, pct5m, cvd5m = mkt_features.get('adx14', 0), mkt_features.get('pct5m', 0), mkt_features.get('CVD5m', 0)
    if (side == "Sell" and adx > adx_threshold and pct5m > momentum_threshold and cvd5m > 0):
        logger.debug(f"⛔️ [{symbol}] Заблокировано MLX-фильтром (ADX: {adx:.1f}, MLX: {mlx_confidence:.2f})")
        return False
    if (side == "Buy" and adx > adx_threshold and pct5m < -momentum_threshold and cvd5m < 0):
        logger.debug(f"⛔️ [{symbol}] Заблокировано MLX-фильтром (ADX: {adx:.1f}, MLX: {mlx_confidence:.2f})")
        return False
    if cluster_metrics.get('strength_trend', 0) < -0.3:
        logger.debug(f"⛔️ [{symbol}] Отрицательный тренд силы кластеров")
        return False
    return True


async def _confirm_tape_touch(bot, symbol: str, wall_price: float, side: str, cfg: dict, last_price: float):
    lb_secs = int(cfg.get("TAPE_TOUCH_LB_SECS", 3))
    max_dist_ticks = int(cfg.get("TAPE_TOUCH_MAX_DIST_TICKS", 3))
    min_trades = int(cfg.get("TAPE_TOUCH_MIN_TRADES", 6))
    min_absorb_qty = float(cfg.get("TAPE_TOUCH_MIN_ABSORB_QTY", 12000))
    tick = bot.price_tick_map.get(symbol, 1e-4) or 1e-4

    trades = list(getattr(bot.shared_ws, "trade_history", {}).get(symbol, []))
    if not trades: 
        return False, {"reason": "no_tape"}

    now = time.time()
    def near(px): return abs(px - wall_price)/tick <= max_dist_ticks

    total, qty, buys_qty, sells_qty = 0, 0.0, 0.0, 0.0
    ofi = 0.0
    max_px, min_px = -1e300, 1e300

    for t in trades:
        ts = float(t.get('ts') or t.get('timestamp') or 0.0)
        if ts < now - lb_secs: continue
        px = float(t.get('price') or 0.0)
        q  = float(t.get('qty') or t.get('size') or 0.0)
        if px <= 0 or q <= 0: continue
        if not near(px): continue

        side_t = (t.get('side') or "").lower()
        total += 1; qty += q
        if side_t == "buy": buys_qty += q; ofi += q
        elif side_t == "sell": sells_qty += q; ofi -= q

        if px > max_px: max_px = px
        if px < min_px: min_px = px

    metrics = {
        "total_trades": total, "total_qty": qty,
        "buys_qty": buys_qty, "sells_qty": sells_qty, "OFI": ofi,
        "max_px": None if max_px < -1e290 else max_px,
        "min_px": None if min_px > 1e290 else min_px,
    }

    if total < min_trades or qty < min_absorb_qty:
        metrics["reason"] = "insufficient_prints_or_qty"
        return False, metrics

    if side == "Sell":
        if metrics["max_px"] is not None and metrics["max_px"] > wall_price + max_dist_ticks * tick:
            metrics["reason"] = "overrun_above_sell_wall"; return False, metrics
        absorb_ratio = buys_qty / max(1e-9, qty)
    else:
        if metrics["min_px"] is not None and metrics["min_px"] < wall_price - max_dist_ticks * tick:
            metrics["reason"] = "overrun_below_buy_wall"; return False, metrics
        absorb_ratio = sells_qty / max(1e-9, qty)

    metrics["absorb_ratio"] = absorb_ratio
    min_absorb_ratio = float(cfg.get("TAPE_ABSORB_RATIO_MIN", 0.62))
    if absorb_ratio < min_absorb_ratio:
        metrics["reason"] = "weak_absorption"; return False, metrics

    metrics["reason"] = "ok"
    return True, metrics


def _format_dom_decision_line(symbol: str, mode: str, side: str,
                              prob_breakout: float, prob_hold: float,
                              mkt_features: dict, market_context: dict,
                              cluster_conf: float, dist_pct: float,
                              tape_ok, reasons):
    adx = float(mkt_features.get("adx14", 0.0))
    pct5m = float(mkt_features.get("pct5m", 0.0))
    cvd5m = float(mkt_features.get("CVD5m", 0.0))
    ob_imb = float((market_context or {}).get("orderbook_imbalance", 0.0))
    parts = [
        f"{symbol} mode={mode}", f"side={side}", f"pb={prob_breakout:.2f}", f"ph={prob_hold:.2f}",
        f"ADX={adx:.1f}", f"pct5m={pct5m:.2f}", f"CVD5m={cvd5m:.2f}", f"obImb={ob_imb:.2f}",
        f"cluster={cluster_conf:.2f}", f"dist={dist_pct:.3f}%",
    ]
    if tape_ok is not None:
        parts.append(f"tape={'ok' if tape_ok else 'no'}")
    if reasons:
        parts.append("why=" + "|".join(reasons))
    return " ".join(parts)


async def mlx_enhanced_dom_strategy(bot, symbol: str) -> bool:
    cfg = bot.user_data.get("dom_squeeze_settings", config.DOM_SQUEEZE_STRATEGY)
    if not cfg.get("ENABLED", False): return False

    if not hasattr(bot, 'mlx_predictor'):
        bot.mlx_predictor = MLXPredictorManager()
        bot.mlx_predictor.load_model()
    if not hasattr(bot, 'enhanced_volume_analyzer'):
        bot.enhanced_volume_analyzer = EnhancedVolumeClusterAnalyzer()

    signal_key = None
    try:
        await _update_wall_memory(bot, symbol)
        closest_wall = await _find_closest_wall_zscore(bot, symbol, cfg)
        if not closest_wall: return False
        validated_wall = await _validate_sticky_wall(bot, symbol, closest_wall)
        if not validated_wall: return False

        orderbook = bot.shared_ws.orderbooks.get(symbol)
        last_price = _dom_get_last_price(bot, symbol)
        if not orderbook or not (last_price and last_price > 0): return False

        bot.enhanced_volume_analyzer.update_cluster_data(symbol, orderbook)
        cluster_metrics = bot.enhanced_volume_analyzer.calculate_advanced_metrics(symbol, last_price) or {}
        cluster_confidence = float(cluster_metrics.get('confidence_score', 0.5))
        
        FADE_CONFIDENCE_MAX = cfg.get("FADE_CONFIDENCE_MAX", 0.30)
        BREAKOUT_CONFIDENCE_MIN = cfg.get("BREAKOUT_CONFIDENCE_MIN", 0.70)
        
        mode = None
        if cluster_confidence <= FADE_CONFIDENCE_MAX and cfg.get("FADE_TRADING_ENABLED", False):
            mode = "fade"
        elif cluster_confidence >= BREAKOUT_CONFIDENCE_MIN and cfg.get("BREAKOUT_TRADING_ENABLED", True):
            mode = "breakout"
        else:
            logger.debug(f"🧱 [{symbol}] Cluster confidence {cluster_confidence:.2f} в серой зоне. Сигнала нет.")
            return False

        logger.info(f"🧱 [{symbol}] Условие выполнено! Cluster confidence: {cluster_confidence:.2f}. Режим: {mode.upper()}")

        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features: return False
        
        side_to_enter = validated_wall['side'] if mode == "fade" else ("Buy" if validated_wall['side'] == "Sell" else "Sell")

        ok_guard, reason_guard = await bot._entry_guard(symbol, side_to_enter, mkt_features, {"source": f"mlx_dom_{mode}"})
        if not ok_guard:
            logger.info(f"🛡️ [{symbol}] DOM-сигнал ({mode}) заблокирован guard: {reason_guard}")
            return False

        signal_key = (symbol, side_to_enter, f'mlx_dom_{mode}')
        if signal_key in bot.active_signals: return True
        
        bot.active_signals.add(signal_key)
        
        logger.info(f"✅🚀 [{symbol}] СИГНАЛ СФОРМИРОВАН! Режим: {mode.upper()}, Вход: {side_to_enter}, Стена: {validated_wall['side']} @ {validated_wall['price']:.6f}")

        candidate = {
            'symbol': symbol, 'side': side_to_enter, 'source': f'mlx_dom_{mode}',
            'wall_price': float(validated_wall['price']), 'wall_rating': validated_wall.get('rating', 0),
            'cluster_confidence': cluster_confidence
        }
        
        asyncio.create_task(bot._process_signal(candidate, mkt_features, signal_key))
        return True

    except Exception as e:
        logger.error(f"[mlx_enhanced_dom_strategy] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
        return False


async def high_frequency_dispatcher(bot, symbol: str):
    if symbol not in bot.shared_ws.watchlist: return
    mode = bot.strategy_mode
    
    if mode == "scalp_only":
        await flea_strategy(bot, symbol); return
        
    if not await _prereqs_check(bot, symbol): return
    
    if mode == "dom_squeeze_only":
        await mlx_enhanced_dom_strategy(bot, symbol)
        return
                    
    if mode in ("full", "liquidation_only", "liq_squeeze"):
        if await liquidation_strategy(bot, symbol): return
            
    if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
        await squeeze_strategy(bot, symbol)

async def low_frequency_dispatcher(bot, symbol: str):
    if symbol not in bot.shared_ws.watchlist: return
    mode = bot.strategy_mode
    if mode == "scalp_only": return
    if not await _prereqs_check(bot, symbol): return
    if mode in ("full", "golden_only", "golden_squeeze"):
        await golden_strategy(bot, symbol)


def init_mlx_components(bot):
    if not hasattr(bot, 'mlx_predictor'):
        bot.mlx_predictor = MLXPredictorManager()
        bot.mlx_predictor.load_model()
    
    if not hasattr(bot, 'enhanced_volume_analyzer'):
        bot.enhanced_volume_analyzer = EnhancedVolumeClusterAnalyzer()
    
    logger.info("🧠 MLX компоненты инициализированы")

async def save_mlx_models(bot):
    if hasattr(bot, 'mlx_predictor') and bot.mlx_predictor.is_trained:
        bot.mlx_predictor.save_model()
        logger.info("MLX модели сохранены при завершении работы")

async def _prereqs_check(bot, symbol: str) -> bool:
    now = time.time()
    if symbol not in bot.shared_ws.watchlist: return False
    if now < bot.strategy_cooldown_until.get(symbol, 0): return False
    if symbol in bot.open_positions or symbol in bot.pending_orders: return False
    if symbol in bot.recently_closed and (now - bot.recently_closed[symbol] < 900): return False
    if await bot.listing_age_minutes(symbol) < bot.listing_age_min: return False
    if symbol in bot.failed_orders and (now - bot.failed_orders.get(symbol, 0) < 600): return False
    return True

async def squeeze_strategy(bot, symbol: str):
    signal_key = None
    try:
        if not bot._squeeze_allowed(symbol): return False
        rsi_val = _get_rsi_from_candles(bot, symbol)
        if rsi_val is None or not ((rsi_val > 75) or (rsi_val < 25)):
            return False
        
        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features: return False

        pct1m = mkt_features.get("pct1m")
        if pct1m is None: return False

        SQUEEZE_INTRA_MINUTE_THRESHOLD = 2.0
        if abs(pct1m) < SQUEEZE_INTRA_MINUTE_THRESHOLD:
            return False

        side = "Sell" if rsi_val > 75 else "Buy"
        if not ((side == "Sell" and pct1m > 0) or (side == "Buy" and pct1m < 0)): return False

        bot.last_squeeze_ts[symbol] = time.time()
        signal_key = (symbol, side, 'squeeze_intra_minute')
        if signal_key in bot.active_signals: return True
        bot.active_signals.add(signal_key)
        
        logger.info(f"🔥 [{symbol}] ОБНАРУЖЕН ИНТРА-СВЕЧНОЙ СКВИЗ! ΔЦена 1m: {pct1m:.2f}%.")
        candidate = {"symbol": symbol, "side": side, "source": "squeeze", "base_metrics": {'pct_1m_intra_candle': pct1m}}
        asyncio.create_task(bot._process_signal(candidate, mkt_features, signal_key))
        return True
    except Exception as e:
        logger.error(f"[squeeze_strategy] Критическая ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
        return False

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
        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features:
            bot.active_signals.discard(signal_key)
            return False
        candidate = {'symbol': symbol, 'side': entry_side, 'source': 'liquidation_cascade', 'base_metrics': {'liquidation_value_usd': cluster_value, 'liquidation_side': dominant_side}}
        asyncio.create_task(bot._process_signal(candidate, mkt_features, signal_key))
        return True
    except Exception as e:
        logger.error(f"[liquidation_strategy] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key: bot.active_signals.discard(signal_key)
    return False

# --- ИСПРАВЛЕННАЯ ВЕРСИЯ ---
# Логика `is_price_stable` изменена для корректной работы.
async def golden_strategy(bot, symbol: str):
    signal_key = None
    try:
        if time.time() < bot._last_golden_ts.get(symbol, 0) + 300: return
        if not await _prereqs_check(bot, symbol): return

        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features: return

        CONTEXT_WINDOW_MIN, MAX_CONTEXT_CHANGE_PCT = 240, 6.0
        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < CONTEXT_WINDOW_MIN + 5: return # Нужно хотя бы 5 свечей для анализа

        context_change_pct = utils.compute_pct(candles, CONTEXT_WINDOW_MIN)
        
        last_candle = candles[-1]
        price_end = utils.safe_to_float(last_candle.get("closePrice"))
        open_price = utils.safe_to_float(last_candle.get("openPrice"))
        if open_price == 0: return
        
        side = "Buy" if price_end >= open_price else "Sell"
        if (side == "Buy" and context_change_pct > MAX_CONTEXT_CHANGE_PCT) or \
           (side == "Sell" and context_change_pct < -MAX_CONTEXT_CHANGE_PCT): return

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ЛОГИКИ ---
        # Проверяем стабильность цены за 4 минуты ДО последней, "сигнальной" минуты.
        stability_candles = candles[-5:-1] 
        price_range_pct = (max(c['highPrice'] for c in stability_candles) - min(c['lowPrice'] for c in stability_candles)) / price_end * 100
        is_price_stable = price_range_pct < 0.8 # Цена была в диапазоне 0.8%

        is_oi_growing = mkt_features.get('dOI5m', 0.0) * 100.0 > 1.0
        
        last_candle_volume = mkt_features.get('vol1m', 0)
        avg_volume_prev_4m = np.mean([c['volume'] for c in stability_candles]) if stability_candles else 0
        if avg_volume_prev_4m == 0: return
        
        is_volume_spike = last_candle_volume > (avg_volume_prev_4m * 2.0)
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        if not (is_price_stable and is_oi_growing and is_volume_spike): return
        
        bot._last_golden_ts[symbol] = time.time()
        signal_key = (symbol, side, 'golden_setup_v2')
        if signal_key in bot.active_signals: return
        bot.active_signals.add(signal_key)

        vol_spike_ratio = last_candle_volume / avg_volume_prev_4m
        logger.info(f"🏆 [{symbol}] GOLDEN SETUP 2.0! Контекст 4ч: {context_change_pct:+.2f}%. Vol Spike: x{vol_spike_ratio:.1f}. Направление: {side}.")
        candidate = {"symbol": symbol, "side": side, "source": "golden_setup"}
        asyncio.create_task(bot._process_signal(candidate, mkt_features, signal_key))
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
    if not await _prereqs_check(bot, symbol): return
    try:
        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features: return
        fast_ema, slow_ema = mkt_features.get('fast_ema', 0), mkt_features.get('slow_ema', 0)
        fast_ema_prev, slow_ema_prev = mkt_features.get('fast_ema_prev', 0), mkt_features.get('slow_ema_prev', 0)
        rsi, atr, trend_ema, last_price = mkt_features.get('rsi14', 50), mkt_features.get('atr14', 0), mkt_features.get('trend_ema', 0), mkt_features.get('price', 0)
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
        logger.error(f"[flea_strategy] Ошибка для {symbol}: {e}", exc_info=True)

def save_wall_memory(bot, filename=None):
    try:
        filename = filename or str(config.WALL_MEMORY_FILE)
        with open(filename, 'wb') as f:
            pickle.dump(bot.wall_memory, f)
        total_levels = sum(len(v) for v in bot.wall_memory.values())
        logger.info(f"🧠 Память стен сохранена в {filename} (всего уровней: {total_levels})")
    except Exception as e:
        logger.error(f"Ошибка сохранения памяти стен: {e}")

def load_wall_memory(bot, filename=None):
    try:
        filename = filename or str(config.WALL_MEMORY_FILE)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                bot.wall_memory = pickle.load(f)
            total_levels = sum(len(v) for v in bot.wall_memory.values())
            logger.info(f"🧠 Память стен загружена из {filename} (символов: {len(bot.wall_memory)}, уровней: {total_levels})")
        else:
            bot.wall_memory = {}
            logger.info(f"🧠 Файл памяти стен не найден: {filename}. Создана новая память.")
    except Exception as e:
        bot.wall_memory = {}
        logger.error(f"Ошибка загрузки памяти стен: {e}")
