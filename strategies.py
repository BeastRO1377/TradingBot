# strategies.py
import logging
import time
import asyncio
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import datetime as dt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from ta_compat import ta
import pickle
import os
import utils
import ai_ml
import config
import bot_core

import math
import statistics
from ai_ml import score_from_feature_dict, score_signal_probability


logger = logging.getLogger(__name__)


def _golden_dynamic_volume_threshold(side: str, pct1m: float, pct4m: float) -> float:
    """
    Подбирает минимально допустимое отношение объема к среднему
    в зависимости от силы импульса. Для резких пампов Golden Setup
    позволяет чуть меньший фильтр, чтобы не пропускать ранние фазы.
    """
    side = (side or "").capitalize()
    base = 2.2
    if side == "Buy":
        if pct1m >= 8.0 or pct4m >= 10.0:
            return 1.6
        if pct1m >= 5.0 or pct4m >= 7.0:
            return 1.8
        if pct1m >= 3.0 or pct4m >= 5.0:
            return 2.0
        return base
    # Для шортов сохраняем более строгий фильтр, но допускаем послабление при
    # экстремальных сливов, чтобы фиксировать "истекание".
    if pct1m <= -3.5 or pct4m <= -5.0:
        return 1.8
    if pct1m <= -2.0 or pct4m <= -3.0:
        return 2.0
    return base


def _golden_context_limits(side: str, volume_spike: float, pct1m: float, pct4m: float) -> tuple[float, float]:
    """
    Возвращает (max_positive_context, max_negative_context) для Golden Setup.
    Позволяет игнорировать слишком жёсткий предел 6%, когда налицо
    действительно сильный импульс с поддержкой объема.
    """
    side = (side or "").capitalize()
    max_up = 6.0
    max_down = -6.0

    # Чем выше объем и импульс, тем больше допускаем перекупленность контекста.
    if side == "Buy":
        if volume_spike >= 3.0 or pct1m >= 8.0 or pct4m >= 12.0:
            max_up = 28.0
        elif volume_spike >= 2.3 or pct1m >= 6.0 or pct4m >= 9.0:
            max_up = 20.0
        elif volume_spike >= 1.8 or pct1m >= 4.0 or pct4m >= 6.5:
            max_up = 12.0
    else:  # Sell
        if volume_spike >= 3.0 or pct1m <= -4.0 or pct4m <= -6.0:
            max_down = -20.0
        elif volume_spike >= 2.3 or pct1m <= -3.0 or pct4m <= -5.0:
            max_down = -15.0
        elif volume_spike >= 1.8 or pct1m <= -2.0 or pct4m <= -3.5:
            max_down = -10.0

    return max_up, max_down

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
        self.bootstrap_mode = False
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
            logger.info(f"Файл модели {self.model_path} не найден. Создаю базовый шаблон весов.")
            self._bootstrap_default_weights("missing file")
            return
        try:
            if self.model is None: self.init_model()
            self.model.load_weights(self.model_path)
            self.is_trained = True
            self.bootstrap_mode = False
            logger.info(f"MLX модель успешно загружена из {self.model_path}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить MLX модель ({e}). Использую базовый шаблон.")
            self._bootstrap_default_weights("load failure")

    def _bootstrap_default_weights(self, reason: str):
        if self.model is None:
            self.init_model()
        try:
            self.model.save_weights(self.model_path)
            self.is_trained = False
            self.bootstrap_mode = True
            logger.info(f"Базовая модель wall_breakout создана ({reason}).")
        except Exception as err:
            logger.error(f"Не удалось создать базовую модель wall_breakout: {err}", exc_info=True)


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

async def _find_closest_wall_zscore(bot, symbol: str, cfg: dict) -> dict | None:
    """
    Ищет ближайшую «аномально крупную» DOM-стену и возвращает:
      {'price': float, 'side': 'Buy'|'Sell', 'size': float, 'zscore': float}
    Идея: считаем z-score для ближайших к цене уровней по bids/asks и берём максимально значимые.
    """
    ob = getattr(bot.shared_ws, "orderbooks", {}).get(symbol)
    if not ob:
        return None

    last_price = _dom_get_last_price(bot, symbol)
    if not (last_price and last_price > 0):
        return None

    bids: dict = ob.get("bids", {}) or {}
    asks: dict = ob.get("asks", {}) or {}

    tick = float(getattr(bot, "price_tick_map", {}).get(symbol, 1e-4) or 1e-4)
    max_levels      = int(cfg.get("MAX_SCAN_LEVELS", 80))
    max_dist_ticks  = int(cfg.get("MAX_DISTANCE_TICKS", 150))
    min_wall_zscore = float(cfg.get("MIN_WALL_ZSCORE", 1.0))

    def near_levels(levels: dict, side: str):
        out = []
        for p, q in levels.items():
            try:
                pf = float(p)
                qf = float(q)
            except Exception:
                continue
            dist_ticks = abs(pf - last_price) / tick
            if dist_ticks <= max_dist_ticks:
                if side == "bid" and pf <= last_price:
                    out.append((pf, qf, dist_ticks))
                elif side == "ask" and pf >= last_price:
                    out.append((pf, qf, dist_ticks))
        out.sort(key=lambda x: x[2])
        return out[:max_levels]

    bids_near = near_levels(bids, "bid")
    asks_near = near_levels(asks, "ask")

    def with_zscore(levels):
        if not levels:
            return []
        sizes = [q for _, q, _ in levels]
        mean = sum(sizes) / len(sizes)
        var  = sum((q - mean) ** 2 for q in sizes) / len(sizes)
        std  = math.sqrt(var)
        out = []
        for p, q, d in levels:
            z = (q - mean) / std if std > 1e-12 else 0.0
            out.append((p, q, d, z))
        return out

    bzs = with_zscore(bids_near)  # (price, size, dist, z)
    azs = with_zscore(asks_near)

    best_bid = max(bzs, key=lambda x: (x[3], -x[2]), default=None)
    best_ask = max(azs, key=lambda x: (x[3], -x[2]), default=None)

    candidates = []
    if best_bid and best_bid[3] >= min_wall_zscore:
        candidates.append(("Buy", best_bid))   # крупный BID под ценой
    if best_ask and best_ask[3] >= min_wall_zscore:
        candidates.append(("Sell", best_ask))  # крупный ASK над ценой

    if not candidates:
        return None

    side, (p, q, dist, z) = min(candidates, key=lambda t: t[1][2])  # ближе к цене
    return {"price": float(p), "side": side, "size": float(q), "zscore": float(z)}


def _get_wall_memory_rating(bot, symbol: str, wall_price: float, side: str, tick: float, cfg: dict) -> int:
    """
    Надёжно достаёт рейтинг стены из bot.dom_wall_memory с учётом разных видов ключей:
    - (side, price_cluster) / (side, cluster_idx)
    - price_cluster / cluster_idx (float/int/str)
    + fallback: поиск ближайшего кластера по расстоянию в тиках.

    Возвращает: rating = holds - 2 * breaches  (int)
    """
    try:
        sym_mem = getattr(bot, "dom_wall_memory", {}).get(symbol, {})
        if not sym_mem:
            return 0

        side = (side or "").strip()
        tick = float(tick or 1e-6)
        cluster_ticks = int((cfg or {}).get("MEM_CLUSTER_TICKS", 100))

        # Базовые представления кластера
        cluster_price = round(wall_price / (tick * cluster_ticks)) * (tick * cluster_ticks)
        idx = int(round(wall_price / tick))
        cluster_idx = idx // cluster_ticks

        # Кандидаты ключей в порядке убывания «вероятности»
        key_candidates = [
            (side, cluster_price),
            (side.lower(), cluster_price),
            (side.upper(), cluster_price),
            (side, cluster_idx),
            (side.lower(), cluster_idx),
            cluster_price,
            round(cluster_price, 8),      # сгладим флоат
            str(cluster_price),
            cluster_idx,
            str(cluster_idx),
        ]

        def _extract_rating(node) -> int:
            holds = int(node.get("holds", 0))
            breaches = int(node.get("breaches", 0))
            return holds - 2 * breaches

        # Прямые попадания по ключам
        for k in key_candidates:
            if k in sym_mem:
                return _extract_rating(sym_mem[k])

        # Fuzzy: пробежимся по ключам и найдём ближайший кластер в тиках.
        best = None  # (dist_ticks, node)
        for k, node in sym_mem.items():
            k_side = None
            k_val = None

            if isinstance(k, tuple) and len(k) == 2:
                # Возможные форматы: (side, price_cluster) или (price_cluster, side)
                if isinstance(k[0], str):
                    k_side, k_val = k[0], k[1]
                elif isinstance(k[1], str):
                    k_side, k_val = k[1], k[0]
                else:
                    k_val = k[0]
            else:
                k_val = k

            if k_side and str(k_side).lower() != side.lower():
                continue

            # Попробуем привести k_val к числу
            try:
                kv = float(k_val)
            except Exception:
                continue

            # Сравниваем по цене кластера (а не по индексу)
            dist_ticks = abs(kv - cluster_price) / tick if tick > 0 else abs(kv - cluster_price)
            # ограничимся одной «полосой» кластера
            if best is None or dist_ticks < best[0]:
                best = (dist_ticks, node)

        if best and best[0] <= cluster_ticks:  # в пределах одной кластерной ширины
            return _extract_rating(best[1])

        return 0
    except Exception:
        return 0



async def _validate_sticky_wall(bot, symbol: str, wall_data: dict) -> dict | None:
    """
    Проверяет «липкость» стены:
      1) рейтинг из памяти (holds/breaches) — если < 0, отбрасываем,
      2) 10 секунд не «тает» (<70% от начального размера) — иначе отбрасываем.
    Возвращает wall_data с полем 'rating' или None.
    """
    wall_price = float(wall_data["price"])
    side = str(wall_data["side"])
    price_tick = float(getattr(bot, "price_tick_map", {}).get(symbol, 1e-4) or 1e-4)

    # Достаём рейтинг через устойчивый резолвер (см. выше)
    cfg = getattr(bot, "user_data", {}).get("dom_squeeze_settings", {}) or {}
    rating = _get_wall_memory_rating(bot, symbol, wall_price, side, price_tick, cfg)

    if rating < 0:
        logger.debug(f"[{symbol}] Стена {side} @ {wall_price:.6f} проигнорирована (rating={rating} < 0).")
        return None

    wall_initial_size = float(wall_data.get("size") or wall_data.get("qty") or 0.0)

    # 10-секундный контроль «не спуф» (размер не тает более чем до 70%)
    for _ in range(10):
        await asyncio.sleep(1)
        ob = getattr(bot.shared_ws, "orderbooks", {}).get(symbol)
        if not ob:
            return None
        dom_side = ob.get("asks", {}) if side == "Sell" else ob.get("bids", {})
        # ключи в DOM могут быть строками — аккуратно достанем по float
        current = 0.0
        if wall_price in dom_side:
            try:
                current = float(dom_side.get(wall_price, 0.0))
            except Exception:
                current = 0.0
        else:
            # попробуем точное совпадение по строке и лёгкий epsilon
            as_float_pairs = []
            for kp, qv in dom_side.items():
                try:
                    as_float_pairs.append((float(kp), float(qv)))
                except Exception:
                    continue
            # найдём ближайший уровень по цене (до одного тика)
            if as_float_pairs:
                as_float_pairs.sort(key=lambda t: abs(t[0] - wall_price))
                nearest_p, nearest_q = as_float_pairs[0]
                if abs(nearest_p - wall_price) <= price_tick + 1e-12:
                    current = nearest_q

        if current < wall_initial_size * 0.7:
            logger.debug(f"[{symbol}] Стена {side} @ {wall_price:.6f} испарилась (<70%).")
            return None

    wall_data["rating"] = int(rating)
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
    """
    Улучшенная DOM-логика:
      - ищет аномальную стену (zscore),
      - проверяет «липкость» (rating из памяти + не тает),
      - сравнивает wall_size с встречным кластером,
      - переводит в режим fade/breakout с фильтром по wall_z при пробое.
    """
    # берём настройки из user_data или глобального конфига, иначе дефолты
    user_cfg = getattr(bot, "user_data", {}).get("dom_squeeze_settings")
    cfg = user_cfg if isinstance(user_cfg, dict) and user_cfg else config.DOM_SQUEEZE_STRATEGY

    if not cfg.get("ENABLED", False):
        return False

    # MLX/кластер-аналитика — ленивая инициализация
    if not hasattr(bot, "mlx_predictor"):
        from ai_ml import MLXPredictorManager
        bot.mlx_predictor = MLXPredictorManager()
        bot.mlx_predictor.load_model()

    if not hasattr(bot, "simple_cluster_analyzer"):
        import utils
        bot.simple_cluster_analyzer = utils.SimpleClusterAnalyzer()

    try:
        # обновим память и найдём стену
        await _update_wall_memory(bot, symbol)
        closest_wall = await _find_closest_wall_zscore(bot, symbol, cfg)
        if not closest_wall:
            return False

        validated_wall = await _validate_sticky_wall(bot, symbol, closest_wall)
        if not validated_wall:
            return False

        ob = getattr(bot.shared_ws, "orderbooks", {}).get(symbol)
        last_price = _dom_get_last_price(bot, symbol)
        if not ob or not (last_price and last_price > 0):
            return False

        cluster = bot.simple_cluster_analyzer.analyze(ob)

        wall_price  = float(validated_wall.get("price", 0.0))
        wall_side   = str(validated_wall.get("side", "Buy"))
        wall_size   = float(validated_wall.get("size") or validated_wall.get("qty") or 0.0)
        wall_z      = float(validated_wall.get("zscore", 0.0))   # ← тот самый wall_z
        wall_rating = int(validated_wall.get("rating", 0))       # используется с MIN_WALL_RATING

        # Пороги с дефолтами (если нет в конфиге)
        min_wall_rating    = int(cfg.get("MIN_WALL_RATING", 0))        # ← новый порог; можно не добавлять в config.py
        wall_cluster_ratio = float(cfg.get("WALL_CLUSTER_RATIO", 1.25))
        z_min_for_breakout = float(cfg.get("Z_MIN_FOR_BREAKOUT", 1.5)) # ← порог z-score для пробоя

        # Сила стены vs встречный кластер
        if wall_side == "Sell":
            cluster_size = float(cluster.get("top_bid_size", 0.0))
        else:
            cluster_size = float(cluster.get("top_ask_size", 0.0))

        if cluster_size > 0.0 and wall_size < cluster_size * wall_cluster_ratio:
            logger.debug(
                f"[{symbol}] Стена {wall_side} слабее встречного кластера "
                f"({wall_size:.0f} < {cluster_size:.0f} * {wall_cluster_ratio}). Пропуск."
            )
            return False

        # Фильтр по «памяти» стены
        if wall_rating < min_wall_rating:
            logger.debug(f"[{symbol}] Рейтинг стены {wall_rating} < {min_wall_rating}. Пропуск.")
            return False

        # Режим: fade/breakout
        if wall_side == "Sell":
            mode = "breakout" if (wall_price < last_price and wall_z >= z_min_for_breakout) else "fade"
            side_to_enter = "Buy" if mode == "breakout" else "Sell"
        else:  # Buy-стена
            mode = "breakout" if (wall_price > last_price and wall_z >= z_min_for_breakout) else "fade"
            side_to_enter = "Sell" if mode == "breakout" else "Buy"

        # рыночные фичи + guard
        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features:
            return False

        try:
            ml_conf = score_from_feature_dict(mkt_features)
        except Exception as e:
            logger.warning(f"[golden_setup] ML score failed: {e}")
        mkt_features["ml_confidence"] = ml_conf

        ok_guard, reason_guard = await bot._entry_guard(symbol, side_to_enter, mkt_features, {"source": f"mlx_dom_{mode}"})
        if not ok_guard:
            logger.info(f"[{symbol}] Сигнал заблокирован guard: {reason_guard}")
            return False

        # дедуп сигнала
        signal_key = (symbol, side_to_enter, f"mlx_dom_{mode}")
        if signal_key in bot.active_signals:
            return True

        bot.active_signals.add(signal_key)
        logger.info(
            f"✅ DOM-СИГНАЛ [{symbol}] → {mode.upper()} | вход: {side_to_enter}, "
            f"стена: {wall_side} @ {wall_price:.6f} | z={wall_z:.2f}, rating={wall_rating}, "
            f"size={wall_size:.0f}"
        )

        candidate = {
            "symbol": symbol,
            "side": side_to_enter,
            "source": f"mlx_dom_{mode}",
            "wall_price": wall_price,
            "wall_rating": wall_rating,
            "wall_zscore": wall_z,
            "wall_size": wall_size,
            "cluster_size": float(cluster_size),
            "wall_hold_ratio": utils.safe_to_float(validated_wall.get("hold_ratio"), 1.0),
        }
        await bot._process_signal(candidate, mkt_features, signal_key)
        return True

    except Exception as e:
        logger.error(f"[{symbol}] Ошибка в mlx_enhanced_dom_strategy: {e}")
        return False


async def high_frequency_dispatcher(bot, symbol: str):
    if symbol not in bot.shared_ws.watchlist: return
    mode = bot.strategy_mode
    
    if mode == "scalp_only":
        await flea_strategy(bot, symbol); return
        
    if not await _prereqs_check(bot, symbol): return
    
    # if mode == "dom_squeeze_only":
    #     await mlx_enhanced_dom_strategy(bot, symbol)
    #     return

    if mode == "dom_squeeze_only":
        await simplified_dom_strategy(bot, symbol)
        return


    if mode in ("full", "liquidation_only", "liq_squeeze"):
        if await liquidation_strategy(bot, symbol): return
            
    if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
        await squeeze_strategy(bot, symbol)

    if mode in ("full", "golden_only", "golden_squeeze"):
        await golden_strategy(bot, symbol)

async def low_frequency_dispatcher(bot, symbol: str):
    if symbol not in bot.shared_ws.watchlist: return
    mode = bot.strategy_mode
    if mode == "scalp_only": return
    if not await _prereqs_check(bot, symbol): return


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

def _has_bearish_divergence(candles: list, lookback: int = 60) -> bool:
    """
    Проверяет наличие простой медвежьей дивергенции (price HH, RSI/MACD HL).
    Используем последние `lookback` минутных свечей.
    """
    if len(candles) < max(lookback, 30):
        return False

    recent = candles[-lookback:]
    close_prices = [utils.safe_to_float(c.get("closePrice", 0.0)) for c in recent]
    if any(p <= 0 for p in close_prices):
        return False

    price_series = pd.Series(close_prices, dtype="float64")
    rsi_series = ta.rsi(price_series, length=14)
    macd_df = ta.macd(price_series, fast=12, slow=26, signal=9)
    if rsi_series is None or macd_df is None or macd_df.empty:
        return False
    macd_hist_series = macd_df.iloc[:, 2]

    def _local_maxima(values: list, window: int = 2) -> list:
        maxima = []
        n = len(values)
        for idx in range(window, n - window):
            segment = values[idx - window: idx + window + 1]
            center = values[idx]
            if center == max(segment) and center > values[idx - 1] and center >= values[idx + 1]:
                maxima.append(idx)
        return maxima

    maxima_indices = _local_maxima(close_prices, window=2)
    if len(maxima_indices) < 2:
        return False

    idx1, idx2 = maxima_indices[-2], maxima_indices[-1]
    price1, price2 = close_prices[idx1], close_prices[idx2]
    if price2 <= price1 * 1.001:  # нет нового high
        return False

    rsi1 = rsi_series.iloc[idx1]
    rsi2 = rsi_series.iloc[idx2]
    macd1 = macd_hist_series.iloc[idx1]
    macd2 = macd_hist_series.iloc[idx2]

    bearish_rsi = pd.notna(rsi1) and pd.notna(rsi2) and rsi2 < rsi1 - 1.5
    if pd.notna(macd1) and pd.notna(macd2):
        delta_macd = macd1 - macd2
        threshold = max(abs(macd1) * 0.25, 1e-4)
        bearish_macd = delta_macd > threshold
    else:
        bearish_macd = False

    return bearish_rsi or bearish_macd

async def golden_strategy(bot, symbol: str):
    signal_key = None
    try:
        # cooldown на повторный сигнал
        if time.time() < bot._last_golden_ts.get(symbol, 0) + 300:
            return
        if not await _prereqs_check(bot, symbol):
            return

        await bot.ensure_recent_candles(symbol, lookback=180, max_age_sec=180)

        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features:
            return

        # === ML: скоринг уверенности ===
        ML_FLOOR = getattr(config, "ML_CONFIDENCE_FLOOR", 0.65)
        try:
            ml_conf = score_from_feature_dict(mkt_features)
        except Exception as e:
            logger.warning(f"[golden_setup] ML score failed: {e}")
            ml_conf = 0.5
        mkt_features["ml_confidence"] = ml_conf
        ml_floor = max(0.0, min(1.0, ML_FLOOR))
        logger.debug(f"[golden_setup] {symbol} ml_conf={ml_conf:.5f}")
        # ===============================

        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < 260:  # 240m контекст + расчетные свечи
            return

        PRICE_IMPULSE_1M = 0.35
        PRICE_IMPULSE_4M = 1.5
        VOLUME_SPIKE_FACTOR = 2.2
        OI_GRADUAL_MIN = 1.25   # % за 4 минуты
        OI_EXTREME_MIN = 0.8    # % за 1-5 минут
        CONTEXT_WINDOW_MIN, MAX_CONTEXT_CHANGE_PCT = 240, 10.0

        context_change_pct = utils.compute_pct(candles, CONTEXT_WINDOW_MIN)
        price_change_1m = utils.safe_to_float(mkt_features.get("pct1m", 0.0))
        price_change_4m = utils.safe_to_float(mkt_features.get("GS_pct4m", 0.0))
        price_change_5m = utils.safe_to_float(mkt_features.get("pct5m", 0.0))
        price_change_15m = utils.safe_to_float(mkt_features.get("pct15m", 0.0))
        adx14 = utils.safe_to_float(mkt_features.get("adx14", 0.0))
        cvd_1m = utils.safe_to_float(mkt_features.get("CVD1m", 0.0))

        side = None
        if price_change_1m >= PRICE_IMPULSE_1M or price_change_4m >= PRICE_IMPULSE_4M:
            side = "Buy"
        elif price_change_1m <= -PRICE_IMPULSE_1M or price_change_4m <= -PRICE_IMPULSE_4M:
            side = "Sell"
        elif price_change_5m >= PRICE_IMPULSE_4M:
            side = "Buy"
        elif price_change_5m <= -PRICE_IMPULSE_4M:
            side = "Sell"
        else:
            return

        if side == "Buy" and _has_bearish_divergence(candles):
            logger.debug(f"[golden_setup] Пропуск {symbol}: обнаружена медвежья дивергенция RSI/MACD.")
            return

        price_exhaust_cfg = getattr(bot, "golden_price_exhaustion_filter", getattr(config, "GOLDEN_PRICE_EXHAUSTION_FILTER", {}))
        if price_exhaust_cfg.get("ENABLED", True):
            max_abs_pct5m = utils.safe_to_float(price_exhaust_cfg.get("MAX_ABS_PCT5M", 0.0))
            max_abs_pct15m = utils.safe_to_float(price_exhaust_cfg.get("MAX_ABS_PCT15M", 0.0))
            max_abs_gs_pct4m = utils.safe_to_float(price_exhaust_cfg.get("MAX_ABS_GS_PCT4M", 0.0))
            if max_abs_pct5m > 0 and abs(price_change_5m) >= max_abs_pct5m:
                logger.debug(
                    f"[golden_setup] Пропуск {symbol}: |pct5m|={abs(price_change_5m):.2f}% ≥ {max_abs_pct5m:.2f}% — поздняя стадия движения."
                )
                return
            if max_abs_pct15m > 0 and abs(price_change_15m) >= max_abs_pct15m:
                logger.debug(
                    f"[golden_setup] Пропуск {symbol}: |pct15m|={abs(price_change_15m):.2f}% ≥ {max_abs_pct15m:.2f}% — поздняя стадия движения."
                )
                return
            if max_abs_gs_pct4m > 0 and abs(price_change_4m) >= max_abs_gs_pct4m:
                logger.debug(
                    f"[golden_setup] Пропуск {symbol}: |GS_pct4m|={abs(price_change_4m):.2f}% ≥ {max_abs_gs_pct4m:.2f}% — поздняя стадия движения."
                )
                return

        adx_cfg = getattr(bot, "golden_adx_filter", getattr(config, "GOLDEN_ADX_FILTER", {}))
        if adx_cfg.get("ENABLED", True):
            min_adx = utils.safe_to_float(adx_cfg.get("MIN_ADX", 0.0))
            if adx14 < min_adx:
                logger.debug(f"[golden_setup] Пропуск {symbol}: ADX {adx14:.2f} < {min_adx:.2f}.")
                return

        pct1m_cfg = getattr(bot, "golden_pct1m_filter", getattr(config, "GOLDEN_PCT1M_FILTER", {}))
        if pct1m_cfg.get("ENABLED", True):
            max_pullback_long = utils.safe_to_float(pct1m_cfg.get("MAX_PULLBACK_LONG", 0.0))
            max_surge_short = utils.safe_to_float(pct1m_cfg.get("MAX_SURGE_SHORT", 0.0))
            if side == "Buy" and max_pullback_long > 0 and price_change_1m <= -max_pullback_long:
                logger.debug(
                    f"[golden_setup] Пропуск {symbol}: pct1m={price_change_1m:+.2f}% ниже допуска {-max_pullback_long:.2f}% для лонга."
                )
                return
            if side == "Sell" and max_surge_short > 0 and price_change_1m >= max_surge_short:
                logger.debug(
                    f"[golden_setup] Пропуск {symbol}: pct1m={price_change_1m:+.2f}% выше допуска {max_surge_short:.2f}% для шорта."
                )
                return

        cvd_cfg = getattr(bot, "golden_cvd_filter", getattr(config, "GOLDEN_CVD_FILTER", {}))
        if cvd_cfg.get("ENABLED", True):
            if side == "Buy":
                max_cvd = utils.safe_to_float(cvd_cfg.get("MAX_CVD1M_LONG", 0.0))
                if cvd_1m > max_cvd:
                    logger.debug(
                        f"[golden_setup] Пропуск {symbol}: CVD1m={cvd_1m:.0f} > {max_cvd:.0f} для лонга."
                    )
                    return
            else:
                min_cvd = utils.safe_to_float(cvd_cfg.get("MIN_CVD1M_SHORT", 0.0))
                if cvd_1m < min_cvd:
                    logger.debug(
                        f"[golden_setup] Пропуск {symbol}: CVD1m={cvd_1m:.0f} < {min_cvd:.0f} для шорта."
                    )
                    return

        def _extract_price(bar: dict, key_candidates: tuple[str, ...]) -> float:
            for key in key_candidates:
                if key in bar:
                    val = utils.safe_to_float(bar.get(key))
                    if val > 0:
                        return val
            return 0.0

        last_close = _extract_price(candles[-1], ("closePrice", "close", "c"))
        last_high = _extract_price(candles[-1], ("highPrice", "high", "h"))
        last_low = _extract_price(candles[-1], ("lowPrice", "low", "l"))
        if last_close <= 0 or last_high <= 0 or last_low <= 0:
            return

        PEAK_LOOKBACK = 6
        BREAKOUT_MIN_PCT = 0.15
        RSI_OVERBOUGHT = 87.0
        RSI_OVERSOLD = 10.0
        recent_slice = candles[-PEAK_LOOKBACK:]
        recent_high = max(_extract_price(bar, ("highPrice", "high", "h")) for bar in recent_slice)
        recent_low = min(_extract_price(bar, ("lowPrice", "low", "l")) for bar in recent_slice)
        prior_high = max(_extract_price(bar, ("highPrice", "high", "h")) for bar in recent_slice[:-1]) if len(recent_slice) > 1 else recent_high
        prior_low = min(_extract_price(bar, ("lowPrice", "low", "l")) for bar in recent_slice[:-1]) if len(recent_slice) > 1 else recent_low
        breakout_up_pct = ((last_close - prior_high) / max(prior_high, 1e-9) * 100.0) if prior_high and prior_high > 0 else 0.0
        breakout_down_pct = ((prior_low - last_close) / max(prior_low, 1e-9) * 100.0) if prior_low and prior_low > 0 else 0.0

        rsi_14 = utils.safe_to_float(mkt_features.get("rsi14", 50.0))

        score_hits = {"price": False, "volume": False, "oi": False}

        # === Оценка breakout ===
        price_strength = breakout_up_pct if side == "Buy" else breakout_down_pct
        base_breakout = BREAKOUT_MIN_PCT
        soft_breakout = 0.5 * base_breakout
        if side == "Buy":
            if breakout_up_pct >= base_breakout or (ml_conf >= ml_floor and breakout_up_pct >= soft_breakout):
                score_hits["price"] = True
            else:
                logger.debug(f"[golden_setup] {symbol}: breakout_up {breakout_up_pct:.2f}% ниже порога {base_breakout:.2f}% (ml={ml_conf:.2f}).")
            if rsi_14 >= RSI_OVERBOUGHT and price_change_1m >= PRICE_IMPULSE_1M * 1.3 and ml_conf < (ml_floor + 0.1):
                logger.debug(f"[golden_setup] Пропуск {symbol}: RSI={rsi_14:.1f} при импульсе {price_change_1m:.2f}% — риск разворота.")
                return
        else:
            if breakout_down_pct >= base_breakout or (ml_conf >= ml_floor and breakout_down_pct >= soft_breakout):
                score_hits["price"] = True
            else:
                logger.debug(f"[golden_setup] {symbol}: breakout_down {breakout_down_pct:.2f}% ниже порога {base_breakout:.2f}% (ml={ml_conf:.2f}).")
            if rsi_14 <= RSI_OVERSOLD and price_change_1m <= -PRICE_IMPULSE_1M * 1.3 and ml_conf < (ml_floor + 0.1):
                logger.debug(f"[golden_setup] Пропуск {symbol}: RSI={rsi_14:.1f} при падении {price_change_1m:.2f}% — риск разворота.")
                return

        last_candle = candles[-1]
        last_candle_volume = utils.safe_to_float(last_candle.get("volume", 0.0))
        avg_volume_prev_4m = utils.safe_to_float(mkt_features.get("avg_volume_prev_4m", 0.0))
        if avg_volume_prev_4m <= 0.0 and len(candles) >= 5:
            avg_volume_prev_4m = float(np.mean([utils.safe_to_float(c.get("volume", 0.0)) for c in candles[-5:-1]]))
        if avg_volume_prev_4m <= 0.0:
            return

        volume_spike_ratio = last_candle_volume / avg_volume_prev_4m if avg_volume_prev_4m else 0.0
        min_volume_spike = _golden_dynamic_volume_threshold(side, price_change_1m, price_change_4m)

        # === Оценка объёма ===
        if volume_spike_ratio >= min_volume_spike:
            score_hits["volume"] = True
        elif ml_conf >= ml_floor and volume_spike_ratio >= (0.7 * min_volume_spike):
            score_hits["volume"] = True
            logger.info(f"[golden_setup] ML override volume: x{volume_spike_ratio:.2f} < {min_volume_spike:.2f}, ml={ml_conf:.2f}")
        else:
            logger.debug(
                f"[golden_setup] {symbol}: объем x{volume_spike_ratio:.2f} < min {min_volume_spike:.2f} "
                f"(side={side}, pct1m={price_change_1m:.2f}%, pct4m={price_change_4m:.2f}%)."
            )

        max_ctx_up, max_ctx_down = _golden_context_limits(side, volume_spike_ratio, price_change_1m, price_change_4m)

        # === ML-override для контекста ===
        if side == "Buy" and context_change_pct > max_ctx_up:
            if ml_conf >= ML_FLOOR and context_change_pct <= (max_ctx_up * 1.5):
                logger.info(f"[golden_setup] ML override context up: ctx={context_change_pct:.2f} > {max_ctx_up:.2f}, ml={ml_conf:.2f}")
            else:
                logger.debug(f"[golden_setup] Пропуск {symbol}: контекст {context_change_pct:.2f}% превышает допуск {max_ctx_up:.2f}%.")
                return
        if side == "Sell" and context_change_pct < max_ctx_down:
            if ml_conf >= ML_FLOOR and context_change_pct >= (max_ctx_down * 1.5):
                logger.info(f"[golden_setup] ML override context down: ctx={context_change_pct:.2f} < {max_ctx_down:.2f}, ml={ml_conf:.2f}")
            else:
                logger.debug(f"[golden_setup] Пропуск {symbol}: контекст {context_change_pct:.2f}% ниже допуска {max_ctx_down:.2f}%.")
                return
        # ================================

        oi_change_1m_pct = utils.safe_to_float(mkt_features.get("dOI1m", 0.0)) * 100.0
        oi_change_5m_pct = utils.safe_to_float(mkt_features.get("dOI5m", 0.0)) * 100.0
        oi_change_4m_pct = utils.safe_to_float(mkt_features.get("GS_dOI4m", 0.0)) * 100.0
        max_oi_pct = max(oi_change_1m_pct, oi_change_5m_pct, oi_change_4m_pct)
        min_oi_pct = min(oi_change_1m_pct, oi_change_5m_pct, oi_change_4m_pct)

        if side == "Buy":
            directional_oi = max_oi_pct
            if directional_oi <= 0:
                logger.debug(f"[golden_setup] {symbol}: OI не поддерживает лонг (maxOI={directional_oi:.2f}%).")
            has_gradual_oi = oi_change_4m_pct >= OI_GRADUAL_MIN
            has_extreme_oi = (
                oi_change_1m_pct >= OI_EXTREME_MIN
                or oi_change_5m_pct >= OI_EXTREME_MIN
            )
            if directional_oi > 0 and (has_gradual_oi or has_extreme_oi):
                score_hits["oi"] = True
            elif directional_oi > 0 and ml_conf >= ml_floor and directional_oi >= 0.30:
                score_hits["oi"] = True
                logger.info(f"[golden_setup] ML override OI: maxOI={directional_oi:.2f}%, ml={ml_conf:.2f}")
            else:
                logger.debug(f"[golden_setup] {symbol}: OI сигнала нет (maxOI={directional_oi:.2f}%).")
        else:  # Sell: требуется снижение открытого интереса
            directional_oi = min_oi_pct
            if directional_oi >= 0:
                logger.debug(f"[golden_setup] {symbol}: OI не поддерживает шорт (minOI={directional_oi:.2f}%).")
                return
            has_gradual_oi = oi_change_4m_pct <= -OI_GRADUAL_MIN
            has_extreme_oi = (
                oi_change_1m_pct <= -OI_EXTREME_MIN
                or oi_change_5m_pct <= -OI_EXTREME_MIN
            )
            if directional_oi < 0 and (has_gradual_oi or has_extreme_oi):
                score_hits["oi"] = True
            elif directional_oi < 0 and ml_conf >= ml_floor and abs(directional_oi) >= 0.30:
                score_hits["oi"] = True
                logger.info(f"[golden_setup] ML override OI: minOI={directional_oi:.2f}%, ml={ml_conf:.2f}")
            else:
                logger.debug(f"[golden_setup] {symbol}: OI сигнала нет (minOI={directional_oi:.2f}%).")

        score_total = sum(int(v) for v in score_hits.values())
        if score_total == 0:
            logger.debug(f"[golden_setup] Пропуск {symbol}: нет подтверждений (price/vol/OI).")
            return

        prev_state = bot.signal_prev_state.get(symbol, {}) if hasattr(bot, "signal_prev_state") else {}
        curr_cvd1 = utils.safe_to_float(mkt_features.get("CVD1m", 0.0))
        curr_cvd5 = utils.safe_to_float(mkt_features.get("CVD5m", 0.0))
        curr_pct5 = utils.safe_to_float(mkt_features.get("pct5m", 0.0))
        prev_cvd1 = float(prev_state.get("CVD1m", curr_cvd1))
        prev_cvd5 = float(prev_state.get("CVD5m", curr_cvd5))
        prev_pct5 = float(prev_state.get("pct5m", curr_pct5))
        delta_cvd1 = curr_cvd1 - prev_cvd1
        delta_cvd5 = curr_cvd5 - prev_cvd5
        delta_pct5 = curr_pct5 - prev_pct5
        cvd1_sign_change = 0.0
        if prev_cvd1 <= 0 < curr_cvd1:
            cvd1_sign_change = 1.0
        elif prev_cvd1 >= 0 > curr_cvd1:
            cvd1_sign_change = -1.0
        cvd5_sign_change = 0.0
        if prev_cvd5 <= 0 < curr_cvd5:
            cvd5_sign_change = 1.0
        elif prev_cvd5 >= 0 > curr_cvd5:
            cvd5_sign_change = -1.0

        enriched_features = dict(mkt_features)
        enriched_features.update({
            "delta_cvd1m": delta_cvd1,
            "delta_cvd5m": delta_cvd5,
            "prev_cvd1m": prev_cvd1,
            "prev_cvd5m": prev_cvd5,
            "cvd1m_sign_change": cvd1_sign_change,
                "cvd5m_sign_change": cvd5_sign_change,
            "delta_pct5m": delta_pct5,
        })

        signal_prob = None
        if getattr(bot, "signal_model_enabled", True):
            signal_prob = score_signal_probability(symbol, side, enriched_features)
            if signal_prob is not None:
                threshold = getattr(bot, "signal_model_threshold", None)
                if threshold is None or not math.isfinite(threshold):
                    threshold = getattr(ai_ml.get_signal_model(), "threshold", 0.75)
                if signal_prob < threshold:
                    if hasattr(bot, "signal_prev_state"):
                        bot.signal_prev_state[symbol] = {
                            "CVD1m": curr_cvd1,
                            "CVD5m": curr_cvd5,
                            "pct5m": curr_pct5,
                        }
                    logger.debug(
                        f"[golden_setup] Пропуск {symbol}: signal_prob={signal_prob:.2f} < threshold {threshold:.2f}."
                    )
                    return

        if hasattr(bot, "signal_prev_state"):
            bot.signal_prev_state[symbol] = {
                "CVD1m": curr_cvd1,
                "CVD5m": curr_cvd5,
                "pct5m": curr_pct5,
            }

        PRE_RUN_LOOKBACK_MIN = 10
        if len(candles) >= PRE_RUN_LOOKBACK_MIN:
            last_slice = candles[-PRE_RUN_LOOKBACK_MIN:]
            base_close = utils.safe_to_float(last_slice[0].get("closePrice", 0.0))
            if base_close > 0:
                recent_close = utils.safe_to_float(last_slice[-1].get("closePrice", 0.0))
                pre_run_pct = ((recent_close - base_close) / base_close) * 100.0
                pre_run_cap = 1.8
                if side == "Buy" and pre_run_pct >= pre_run_cap:
                    if ml_conf >= (ml_floor + 0.15):
                        logger.info(f"[golden_setup] ML override pre-run BUY: {pre_run_pct:.2f}% ≥ {pre_run_cap}%, ml={ml_conf:.2f}")
                    else:
                        logger.debug(f"[golden_setup] Пропуск {symbol}: pre-run {pre_run_pct:.2f}% ≥ {pre_run_cap}%.")
                        return
                if side == "Sell" and pre_run_pct <= -pre_run_cap:
                    if ml_conf >= (ml_floor + 0.15):
                        logger.info(f"[golden_setup] ML override pre-run SELL: {pre_run_pct:.2f}% ≤ -{pre_run_cap}%, ml={ml_conf:.2f}")
                    else:
                        logger.debug(f"[golden_setup] Пропуск {symbol}: pre-run {pre_run_pct:.2f}% ≤ -{pre_run_cap}%.")
                        return

        liquidity_cfg = getattr(config, "GOLDEN_LIQUIDITY_FILTER", {})
        min_turnover = utils.safe_to_float(liquidity_cfg.get("MIN_TURNOVER_24H", 0.0))
        min_avg_vol = utils.safe_to_float(liquidity_cfg.get("MIN_AVG_VOL_1M", 0.0))
        if min_turnover > 0:
            turnover = 0.0
            if hasattr(bot, "_get_symbol_turnover"):
                turnover = bot._get_symbol_turnover(symbol)
            else:
                ticker = (bot.shared_ws.ticker_data.get(symbol, {}) if getattr(bot, "shared_ws", None) else {}) or {}
                for key in ("turnover24h", "turnover24hUsd", "turnover24hUSDT", "turnover24hQuote", "turnover24"):
                    turnover = utils.safe_to_float(ticker.get(key))
                    if turnover:
                        break
            if turnover and turnover < min_turnover:
                logger.debug(f"[golden_setup] Пропуск {symbol}: turnover24h={turnover:.0f} < {min_turnover:.0f}")
                return

        if min_avg_vol > 0 and avg_volume_prev_4m > 0 and avg_volume_prev_4m < min_avg_vol:
            logger.debug(f"[golden_setup] Пропуск {symbol}: avg_volume_prev_4m={avg_volume_prev_4m:.1f} < {min_avg_vol:.1f}")
            return

        strict_cfg = getattr(bot, "ml_strict_filters", getattr(config, "ML_STRICT_FILTERS", {}))
        strict_enabled = bool(strict_cfg.get("ENABLED", True))
        strict_min = utils.safe_to_float(strict_cfg.get("MIN_WORKING_ML", 0.33))
        strict_max = utils.safe_to_float(strict_cfg.get("MAX_WORKING_ML", 0.85))

        direct_entry = score_total == 3 or (score_total >= 2 and ml_conf >= (ml_floor + 0.05))
        if strict_enabled:
            working_range = strict_min <= ml_conf <= strict_max
            if not working_range and direct_entry:
                logger.info(
                    f"[golden_setup] ML strict mode активен: прямой вход отключён (ml={ml_conf:.2f}, допустимый диапазон {strict_min:.2f}-{strict_max:.2f})."
                )
            if not working_range:
                direct_entry = False

        bot._last_golden_ts[symbol] = time.time()
        signal_key = (symbol, side, 'golden_setup_v3')
        if signal_key in bot.active_signals:
            return
        bot.active_signals.add(signal_key)

        logger.info(
            f"🏆 [{symbol}] GOLDEN SETUP v3 | side={side} | score={score_total}/3 | priceΔ1m={price_change_1m:+.2f}% "
            f"| OI1m={oi_change_1m_pct:+.2f}% OI4m={oi_change_4m_pct:+.2f}% "
            f"| vol spike x{volume_spike_ratio:.1f} | context 4h={context_change_pct:+.2f}% | ml={ml_conf:.2f} "
            f"| mode={'DIRECT' if direct_entry else 'HUNT'}"
        )

        # Передаём ml_conf в candidate
        candidate = {
            "symbol": symbol,
            "side": side,
            "source": "golden_setup",
            "ml_conf": ml_conf,
            "score_total": score_total,
            "score_hits": score_hits,
        }
        if direct_entry:
            candidate["direct_entry"] = True
            candidate["entry_mode"] = "direct"
        asyncio.create_task(bot._process_signal(candidate, mkt_features, signal_key))

    except Exception as e:
        logger.error(f"[golden_strategy_v3] Ошибка для {symbol}: {e}", exc_info=True)
        if signal_key:
            bot.active_signals.discard(signal_key)


async def aggressive_golden_setup(bot, symbol: str):
    cfg = getattr(bot, "aggressive_golden_cfg", {}) or {}
    if not getattr(bot, "aggressive_golden_enabled", False):
        return

    now = time.time()
    cooldown = float(cfg.get("COOLDOWN_SEC", 180.0))
    last_fire = bot._last_aggressive_ts.get(symbol, 0.0)
    if now < last_fire + cooldown:
        return

    signal_key = None
    try:
        if not await _prereqs_check(bot, symbol):
            return

        await bot.ensure_recent_candles(symbol, lookback=int(cfg.get("LOOKBACK_MINUTES", 120)), max_age_sec=180)

        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features:
            return

        try:
            ml_conf = score_from_feature_dict(mkt_features)
        except Exception:
            ml_conf = float(mkt_features.get("ml_confidence", 0.5) or 0.5)
        mkt_features["ml_confidence"] = ml_conf

        candles = list(bot.shared_ws.candles_data.get(symbol, []))
        if len(candles) < 10:
            return

        last_candle = candles[-1]
        last_volume = utils.safe_to_float(last_candle.get("volume", 0.0))

        avg_volume_prev_4m = utils.safe_to_float(mkt_features.get("avg_volume_prev_4m", 0.0))
        if avg_volume_prev_4m <= 0.0 and len(candles) >= 5:
            avg_volume_prev_4m = float(np.mean([utils.safe_to_float(c.get("volume", 0.0)) for c in candles[-5:-1]]))
        if avg_volume_prev_4m <= 0:
            return

        volume_ratio = last_volume / avg_volume_prev_4m if avg_volume_prev_4m else 0.0
        volume_threshold = float(cfg.get("VOLUME_FACTOR", 3.0))
        if volume_ratio < volume_threshold:
            return

        min_volume_anom = float(cfg.get("MIN_VOLUME_ANOMALY", 0.0))
        if min_volume_anom > 0:
            volume_anomaly = utils.safe_to_float(mkt_features.get("volume_anomaly", 0.0))
            if volume_anomaly <= 0:
                volume_anomaly = utils.safe_to_float(mkt_features.get("volume_anomaly_prev", 0.0))
            if volume_anomaly <= 0:
                volume_anomaly = volume_ratio
            if volume_anomaly < min_volume_anom:
                logger.debug(
                    f"[agg_golden] {symbol}: volume anomaly {volume_anomaly:.2f} ниже порога {min_volume_anom:.2f}."
                )
                return

        price_change_1m = utils.safe_to_float(mkt_features.get("pct1m", 0.0))
        price_change_4m = utils.safe_to_float(mkt_features.get("GS_pct4m", 0.0))

        price_thr_1m = float(cfg.get("PRICE_DELTA_1M", 0.9))
        price_thr_4m = float(cfg.get("PRICE_DELTA_4M", 1.8))

        side = None
        if price_change_1m >= price_thr_1m or price_change_4m >= price_thr_4m:
            side = "Buy"
        elif price_change_1m <= -price_thr_1m or price_change_4m <= -price_thr_4m:
            side = "Sell"
        else:
            return

        max_spread_pct = float(cfg.get("MAX_SPREAD_PCT", 0.0))
        if max_spread_pct > 0:
            spread_pct = abs(utils.safe_to_float(mkt_features.get("spread_pct", 0.0)))
            if spread_pct > max_spread_pct:
                logger.debug(
                    f"[agg_golden] {symbol}: спред {spread_pct:.4f} превышает лимит {max_spread_pct:.4f}."
                )
                return

        pct15m = utils.safe_to_float(mkt_features.get("pct15m", 0.0))
        min_align_pct15m = float(cfg.get("MIN_ALIGN_PCT15M", 0.0))
        if min_align_pct15m > 0:
            if side == "Buy" and pct15m < min_align_pct15m:
                logger.debug(
                    f"[agg_golden] {symbol}: 15m импульс {pct15m:.2f}% против покупки (мин {min_align_pct15m:.2f}%)."
                )
                return
            if side == "Sell" and pct15m > -min_align_pct15m:
                logger.debug(
                    f"[agg_golden] {symbol}: 15m импульс {pct15m:.2f}% против продажи (мин {min_align_pct15m:.2f}%)."
                )
                return

        # ограничение контекста
        context_window = int(cfg.get("LOOKBACK_MINUTES", 120))
        context_delta = utils.compute_pct(candles, context_window)
        max_context = float(cfg.get("MAX_CONTEXT_ABS", 0.0))
        if max_context > 0 and abs(context_delta) > max_context:
            logger.debug(f"[agg_golden] {symbol}: контекст {context_delta:.2f}% вне лимита {max_context:.2f}%.")
            return

        bot._last_aggressive_ts[symbol] = now

        signal_key = (symbol, side, "aggressive_golden_setup")
        if signal_key in bot.active_signals:
            return
        bot.active_signals.add(signal_key)

        score_hits = {"price": True, "volume": True, "oi": True}
        candidate = {
            "symbol": symbol,
            "side": side,
            "source": "aggressive_golden_setup",
            "ml_conf": ml_conf,
            "score_total": 3,
            "score_hits": score_hits,
            "direct_entry": True,
            "entry_mode": "direct",
            "justification": "Агрессивный золотой сетап",
            "label": "Aggressive Golden Setup",
        }

        candidate["trailing_mode"] = bot.user_data.get("trailing_mode") or config.ACTIVE_TRAILING_MODE
        candidate["volume_ratio"] = volume_ratio
        candidate["price_impulse_1m"] = price_change_1m
        candidate["price_impulse_4m"] = price_change_4m

        logger.info(
            f"[agg_golden] {symbol} {side} | vol x{volume_ratio:.2f} | pct1m {price_change_1m:+.2f}% | pct4m {price_change_4m:+.2f}%"
        )

        asyncio.create_task(bot._process_signal(candidate, mkt_features, signal_key))
    except Exception as exc:
        logger.error(f"[aggressive_golden] Ошибка для {symbol}: {exc}", exc_info=True)
        if signal_key:
            bot.active_signals.discard(signal_key)

# async def golden_strategy(bot, symbol: str):
#     """
#     GOLDEN DIRECT:
#       - 3/3: price + volume + dOI1m  => немедленный вход
#       - PV : price + volume          => немедленный вход
#     ML и CVD используются как бонус к скору/логам и не блокируют вход.
#     """
#     import time
#     import numpy as np

#     signal_key = None
#     try:
#         # --- 0) Cooldown и prereqs ---
#         COOLDOWN_SEC = int(getattr(config, "GOLDEN_COOLDOWN_SEC", 60))  # по умолчанию 60с, чтобы не спамить
#         if time.time() < bot._last_golden_ts.get(symbol, 0) + COOLDOWN_SEC:
#             return
#         if not await _prereqs_check(bot, symbol):
#             return

#         await bot.ensure_recent_candles(symbol, lookback=180, max_age_sec=180)

#         # --- 1) Фичи рынка ---
#         mkt_features = await bot.extract_realtime_features(symbol)
#         if not mkt_features:
#             return

#         # ML — только бонус/мета (не блокирует)
#         ML_FLOOR = getattr(config, "ML_CONFIDENCE_FLOOR", 0.65)
#         try:
#             ml_conf = score_from_feature_dict(mkt_features)
#         except Exception as e:
#             logger.warning(f"[golden_setup] ML score failed: {e}")
#             ml_conf = 0.5
#         mkt_features["ml_confidence"] = ml_conf

#         # --- 2) Данные свечей (нужны текущая и предыдущая 1м) ---
#         candles = list((bot.shared_ws.candles_data.get(symbol, [])) or [])
#         if len(candles) < 3:
#             return

#         def _extract_price(bar: dict, keys: tuple[str, ...]) -> float:
#             for k in keys:
#                 if k in bar:
#                     v = utils.safe_to_float(bar.get(k))
#                     if v > 0:
#                         return v
#             return 0.0

#         last_candle = candles[-1]
#         prev_candle = candles[-2]

#         last_close = _extract_price(last_candle, ("closePrice", "close", "c"))
#         prev_close = _extract_price(prev_candle, ("closePrice", "close", "c"))
#         last_high  = _extract_price(last_candle, ("highPrice", "high", "h"))
#         last_low   = _extract_price(last_candle, ("lowPrice", "low", "l"))

#         if min(last_close, prev_close, last_high, last_low) <= 0:
#             return

#         last_vol = utils.safe_to_float(last_candle.get("volume", 0.0))
#         prev_vol = utils.safe_to_float(prev_candle.get("volume", 0.0))
#         if prev_vol <= 0:
#             # попытка подстраховаться: взять объём со свечи -3
#             if len(candles) >= 3:
#                 prev_vol = utils.safe_to_float(candles[-3].get("volume", 0.0))
#             if prev_vol <= 0:
#                 return

#         # --- 3) Базовые метрики 1м ---
#         price_change_1m = (last_close / prev_close - 1.0) * 100.0
#         vol_ratio_1m = last_vol / prev_vol
#         oi_change_1m_pct = utils.safe_to_float(mkt_features.get("dOI1m", 0.0)) * 100.0

#         # Пороги (можно править в config.py)
#         # ВНИМАНИЕ: PRICE_REQ_1M_PCT = 0.35 по умолчанию (0.35%), а не 20.0 (%),
#         # иначе сигналов почти не будет. Если нужно — поменяйте в конфиге.
#         PRICE_REQ_1M_PCT = float(getattr(config, "GOLDEN_REQ_PRICE_DELTA_1M_PCT", 0.35))
#         VOL_REQ_RATIO_1M  = float(getattr(config, "GOLDEN_REQ_VOLUME_RATIO_1M", 2.0))
#         OI_REQ_1M_PCT     = float(getattr(config, "GOLDEN_REQ_DOI1M_PCT", 0.35))

#         meet_price  = abs(price_change_1m) >= PRICE_REQ_1M_PCT
#         meet_volume = vol_ratio_1m >= VOL_REQ_RATIO_1M
#         meet_oi     = oi_change_1m_pct >= OI_REQ_1M_PCT

#         # --- 4) Контекстные расчёты для логов/защит ---
#         # avg_volume_prev_4m -> если нет в фичах, считаем из 4 пред. свечей
#         avg_volume_prev_4m = utils.safe_to_float(mkt_features.get("avg_volume_prev_4m", 0.0))
#         if avg_volume_prev_4m <= 0 and len(candles) >= 5:
#             avg_volume_prev_4m = float(np.mean([utils.safe_to_float(c.get("volume", 0.0)) for c in candles[-5:-1]]))
#         volume_spike_ratio4m = (last_vol / avg_volume_prev_4m) if avg_volume_prev_4m > 0 else 0.0

#         GS_pct4m = utils.safe_to_float(mkt_features.get("GS_pct4m", 0.0))
#         dyn_thresh = _golden_dynamic_volume_threshold("Buy" if price_change_1m > 0 else "Sell",
#                                                       price_change_1m, GS_pct4m)

#         CONTEXT_WINDOW_MIN = 240
#         context_change_pct = utils.compute_pct(candles, CONTEXT_WINDOW_MIN)
#         max_ctx_up, max_ctx_down = _golden_context_limits("Buy" if price_change_1m > 0 else "Sell",
#                                                           volume_spike_ratio4m, price_change_1m, GS_pct4m)

#         # --- 5) Определение режима и стороны ---
#         side_3of3 = None
#         side_pv   = None
#         label     = None

#         if meet_price and meet_volume and meet_oi:
#             side_3of3 = "Buy" if price_change_1m > 0 else "Sell"
#             label = "GOLDEN 3/3 DIRECT"
#         elif meet_price and meet_volume:
#             side_pv = "Buy" if price_change_1m > 0 else "Sell"
#             label = "GOLDEN PV DIRECT"
#         else:
#             unmet = "price" if not meet_price else ("volume" if not meet_volume else "dOI1m")
#             logger.info(
#                 f"🔔 [GOLDEN WATCH] {symbol} | Δ1м={price_change_1m:+.2f}% | "
#                 f"Vol1м x{vol_ratio_1m:.2f} | dOI1м={oi_change_1m_pct:+.2f}% | unmet={unmet}"
#             )
#             return

#         side_to_trade = side_3of3 or side_pv

#         # --- 6) Защитные фильтры (минимальные и быстрые) ---
#         # 6.1 RSI/MACD дивергенция против Buy — мягкий стоппер
#         if side_to_trade == "Buy" and _has_bearish_divergence(candles):
#             logger.debug(f"[golden_setup] Пропуск {symbol}: медвежья дивергенция RSI/MACD.")
#             return

#         # 6.2 Контекст 4h: отсечём экстремальные случаи
#         if side_to_trade == "Buy" and context_change_pct > max_ctx_up:
#             logger.debug(f"[golden_setup] Пропуск {symbol}: контекст 4h {context_change_pct:.2f}% > {max_ctx_up:.2f}%.")
#             return
#         if side_to_trade == "Sell" and context_change_pct < max_ctx_down:
#             logger.debug(f"[golden_setup] Пропуск {symbol}: контекст 4h {context_change_pct:.2f}% < {max_ctx_down:.2f}%.")
#             return

#         # 6.3 Pre-run 10м: по умолчанию 6.5%
#         PRE_RUN_LOOKBACK_MIN = 10
#         pre_run_cap = float(getattr(config, "GOLDEN_PRERUN_CAP_PCT", 6.5))
#         if len(candles) >= PRE_RUN_LOOKBACK_MIN:
#             last_slice = candles[-PRE_RUN_LOOKBACK_MIN:]
#             base_close = utils.safe_to_float(last_slice[0].get("closePrice", 0.0))
#             recent_close = utils.safe_to_float(last_slice[-1].get("closePrice", 0.0))
#             if base_close > 0 and recent_close > 0:
#                 pre_run_pct = (recent_close / base_close - 1.0) * 100.0
#                 if side_to_trade == "Buy" and pre_run_pct >= pre_run_cap:
#                     logger.debug(f"[golden_setup] Пропуск {symbol}: pre-run {pre_run_pct:.2f}% ≥ {pre_run_cap}%.")
#                     return
#                 if side_to_trade == "Sell" and pre_run_pct <= -pre_run_cap:
#                     logger.debug(f"[golden_setup] Пропуск {symbol}: pre-run {pre_run_pct:.2f}% ≤ -{pre_run_cap}%.")
#                     return

#         # 6.4 Ликвидность
#         liquidity_cfg = getattr(config, "GOLDEN_LIQUIDITY_FILTER", {})
#         min_turnover = utils.safe_to_float(liquidity_cfg.get("MIN_TURNOVER_24H", 0.0))
#         min_avg_vol = utils.safe_to_float(liquidity_cfg.get("MIN_AVG_VOL_1M", 0.0))
#         if min_turnover > 0:
#             turnover = 0.0
#             if hasattr(bot, "_get_symbol_turnover"):
#                 turnover = bot._get_symbol_turnover(symbol)
#             else:
#                 ticker = (bot.shared_ws.ticker_data.get(symbol, {}) if getattr(bot, "shared_ws", None) else {}) or {}
#                 for key in ("turnover24h", "turnover24hUsd", "turnover24hUSDT", "turnover24hQuote", "turnover24"):
#                     turnover = utils.safe_to_float(ticker.get(key))
#                     if turnover:
#                         break
#             if turnover and turnover < min_turnover:
#                 logger.debug(f"[golden_setup] Пропуск {symbol}: turnover24h={turnover:.0f} < {min_turnover:.0f}")
#                 return
#         if min_avg_vol > 0 and avg_volume_prev_4m > 0 and avg_volume_prev_4m < min_avg_vol:
#             logger.debug(f"[golden_setup] Пропуск {symbol}: avg_volume_prev_4m={avg_volume_prev_4m:.1f} < {min_avg_vol:.1f}")
#             return

#         # --- 7) Бонусный скоринг (не влияет на вход) ---
#         score_bonus = 0.0

#         # Volume spike бонус (к среднему за 4м)
#         if volume_spike_ratio4m >= dyn_thresh:
#             score_bonus += 10.0
#             if volume_spike_ratio4m >= max(3.0, 1.5 * dyn_thresh):
#                 score_bonus += 10.0

#         # ML бонус: (ml-0.5)*20 -> 0..10
#         score_bonus += max(0.0, (ml_conf - 0.5) * 20.0)

#         # CVD согласованность
#         cvd_1m = utils.safe_to_float(mkt_features.get("CVD1m", 0.0))
#         if (side_to_trade == "Buy" and cvd_1m > 0) or (side_to_trade == "Sell" and cvd_1m < 0):
#             score_bonus += 5.0

#         # --- 8) Лог и запуск входа (без следопыта) ---
#         logger.info(
#             f"🏆 [{symbol}] {label} | side={side_to_trade} | Δ1м={price_change_1m:+.2f}% "
#             f"| Vol1м x{vol_ratio_1m:.2f} | dOI1м={oi_change_1m_pct:+.2f}% "
#             f"| vol_spike4м x{volume_spike_ratio4m:.2f} (dyn={dyn_thresh:.2f}) "
#             f"| ctx={context_change_pct:+.2f}% | ml={ml_conf:.2f} | cvd1м={cvd_1m:.0f} "
#             f"| bonus={score_bonus:.1f}"
#         )

#         # запомним ts (анти-спам)
#         bot._last_golden_ts[symbol] = time.time()

#         # ВХОД В СДЕЛКУ (прямой вызов)
#         await bot.execute_trade_entry_golden_setup(
#             symbol=symbol,
#             side=side_to_trade,
#             reason=label,
#             mkt_features=mkt_features,
#         )

#     except Exception as e:
#         logger.error(f"[golden_strategy_direct] Ошибка для {symbol}: {e}", exc_info=True)



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
            pickle.dump(bot.dom_wall_memory, f)
        total_levels = sum(len(v) for v in bot.dom_wall_memory.values())
        logger.info(f"🧠 Память стен сохранена в {filename} (всего уровней: {total_levels})")
    except Exception as e:
        logger.error(f"Ошибка сохранения памяти стен: {e}")

def load_wall_memory(bot, filename=None):
    try:
        filename = filename or str(config.WALL_MEMORY_FILE)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                bot.dom_wall_memory = pickle.load(f)
            total_levels = sum(len(v) for v in bot.dom_wall_memory.values())
            logger.info(f"🧠 Память стен загружена из {filename} (символов: {len(bot.dom_wall_memory)}, уровней: {total_levels})")
        else:
            bot.dom_wall_memory = {}
            logger.info(f"🧠 Файл памяти стен не найден: {filename}. Создана новая память.")
    except Exception as e:
        bot.dom_wall_memory = {}
        logger.error(f"Ошибка загрузки памяти стен: {e}")


async def simplified_dom_strategy(bot, symbol: str) -> bool:
    """
    Упрощенная DOM-логика с учётом пользовательских настроек
    """
    base_cfg = dict(config.DOM_SQUEEZE_STRATEGY)
    user_cfg = getattr(bot, "user_data", {}).get("dom_squeeze_settings", {}) or {}
    cfg = {**base_cfg, **user_cfg}

    if not cfg.get("ENABLED", False):
        return False

    try:
        wall_data = await _find_and_validate_wall(bot, symbol, cfg)
        if not wall_data:
            return False

        wall_price = wall_data["price"]
        wall_side = wall_data["side"]
        wall_size = wall_data["size"]
        wall_z = wall_data["zscore"]
        wall_rating = wall_data["rating"]
        wall_hold_ratio = utils.safe_to_float(wall_data.get("hold_ratio"), 1.0)

        confidence_level = _calculate_wall_confidence(wall_z, wall_rating, wall_size, cfg)
        if confidence_level == "LOW":
            logger.debug(f"[{symbol}] Слабая стена (z:{wall_z:.1f}, rating:{wall_rating}) — пропуск")
            return False

        last_price = _dom_get_last_price(bot, symbol)
        trade_decision = _make_trade_decision(wall_side, wall_price, last_price, wall_z, confidence_level, cfg)
        if not trade_decision:
            return False

        side_to_enter = trade_decision["side"]
        mode = trade_decision["mode"]
        reason = trade_decision["reason"]

        fade_min_rating = int(cfg.get("FADE_MIN_RATING", cfg.get("MIN_WALL_RATING", 0)))
        breakout_min_rating = int(cfg.get("BREAKOUT_MIN_RATING", cfg.get("MIN_WALL_RATING", 0)))
        fade_min_hold = float(cfg.get("FADE_MIN_HOLD_RATIO", cfg.get("REQUIRED_HOLD_RATIO", 0.0)))
        breakout_min_hold = float(cfg.get("BREAKOUT_MIN_HOLD_RATIO", 0.0))

        if mode == "fade":
            if wall_rating < fade_min_rating:
                logger.debug(f"[{symbol}] Fade-пропуск: rating {wall_rating} < {fade_min_rating}.")
                return False
            if wall_hold_ratio < fade_min_hold:
                logger.debug(f"[{symbol}] Fade-пропуск: hold {wall_hold_ratio:.2f} < {fade_min_hold:.2f}.")
                return False
        else:  # breakout
            if wall_rating < breakout_min_rating:
                logger.debug(f"[{symbol}] Breakout-пропуск: rating {wall_rating} < {breakout_min_rating}.")
                return False
            if breakout_min_hold > 0.0 and wall_hold_ratio < breakout_min_hold:
                logger.debug(f"[{symbol}] Breakout-пропуск: hold {wall_hold_ratio:.2f} < {breakout_min_hold:.2f}.")
                return False

        mkt_features = await bot.extract_realtime_features(symbol)
        if not mkt_features:
            return False

        if not await _quick_market_check(bot, symbol, side_to_enter, mode, confidence_level, mkt_features, cfg):
            return False

        if not _passes_retest(bot, symbol, wall_price, wall_side, side_to_enter, mode, wall_data, cfg):
            logger.debug(f"[{symbol}] Ретест стены не подтвердился. Пропуск сигнала {mode}.")
            return False

        ml_score = None
        ml_hold_score = None
        if hasattr(bot, "mlx_predictor") and getattr(bot.mlx_predictor, "is_trained", False):
            try:
                base_size = float(cfg.get("WALL_BASE_SIZE", 1.0) or 1.0)
                price_diff_pct = abs(wall_price - last_price) / max(last_price, 1e-9) * 100.0
                wall_mlx = {
                    "size_ratio": (wall_size / base_size) if base_size else 1.0,
                    "rating": wall_rating,
                    "distance_pct": price_diff_pct
                }
                market_context = {
                    "rsi": utils.safe_to_float(mkt_features.get("rsi14"), 50.0),
                    "adx": utils.safe_to_float(mkt_features.get("adx14"), 20.0),
                    "pct5m": utils.safe_to_float(mkt_features.get("pct5m"), 0.0),
                    "atr_pct": utils.safe_to_float(mkt_features.get("atr14")) / max(utils.safe_to_float(mkt_features.get("price"), 1.0), 1e-9) * 100.0,
                    "volume_imbalance": utils.safe_to_float(mkt_features.get("orderbook_imbalance"), 0.0),
                    "depth_ratio": utils.safe_to_float(mkt_features.get("depth_ratio"), 1.0),
                    "orderbook_imbalance": utils.safe_to_float(mkt_features.get("orderbook_imbalance"), 0.0),
                    "spread_pct": utils.safe_to_float(mkt_features.get("spread_pct"), 0.0),
                    "volume_ratio": utils.safe_to_float(mkt_features.get("volume_ratio"), 1.0),
                    "momentum_5m": utils.safe_to_float(mkt_features.get("pct5m"), 0.0),
                }
                feature_vec = bot.mlx_predictor.prepare_mkt_features(wall_mlx, market_context)
                ml_score = await bot.mlx_predictor.predict_breakout_probability(feature_vec)
                ml_hold_score = 1.0 - ml_score if ml_score is not None else None
                if mode == "breakout":
                    threshold = max(float(cfg.get("ML_BREAKOUT_THRESHOLD", 0.6)), float(cfg.get("BREAKOUT_MIN_ML_SCORE", 0.55)))
                    if ml_score is not None and ml_score < threshold and confidence_level != "HIGH":
                        logger.debug(f"[{symbol}] ML фильтр отклоняет breakout: score={ml_score:.2f} < {threshold}")
                        return False
                else:
                    threshold = max(float(cfg.get("ML_FADE_THRESHOLD", 0.4)), float(cfg.get("FADE_MIN_ML_HOLD", 0.50)))
                    if ml_hold_score is not None and ml_hold_score < threshold and confidence_level != "HIGH":
                        logger.debug(f"[{symbol}] ML фильтр отклоняет fade: confidence={ml_hold_score:.2f} < {threshold}")
                        return False
            except Exception as ml_err:
                logger.debug(f"[{symbol}] ML фильтр недоступен: {ml_err}")

        signal_id = utils.new_cid()

        candidate = {
            "symbol": symbol,
            "side": side_to_enter,
            "source": f"dom_{mode}",
            "signal_id": signal_id,
            "wall_price": wall_price,
            "wall_rating": wall_rating,
            "confidence": confidence_level,
            "wall_size": wall_size,
            "wall_hold_ratio": wall_hold_ratio,
            "ml_score": ml_score,
        }

        base_size = float(cfg.get("WALL_BASE_SIZE", 1.0) or 1.0)
        wall_size_ratio = (wall_size / base_size) if base_size else 0.0
        price_diff_pct = abs(wall_price - last_price) / max(last_price, 1e-9) * 100.0
        price = utils.safe_to_float(mkt_features.get("price"), 0.0)
        atr_value = utils.safe_to_float(mkt_features.get("atr14"), 0.0)
        atr_pct = (atr_value / price) * 100.0 if price > 0 else 0.0
        spread_pct = utils.safe_to_float(mkt_features.get("spread_pct"), 0.0) * 100.0
        market_snapshot = {
            "price": price,
            "pct1m": utils.safe_to_float(mkt_features.get("pct1m")),
            "pct5m": utils.safe_to_float(mkt_features.get("pct5m")),
            "adx14": utils.safe_to_float(mkt_features.get("adx14")),
            "rsi14": utils.safe_to_float(mkt_features.get("rsi14")),
            "atr_pct": atr_pct,
            "spread_pct": spread_pct,
            "volume_ratio": utils.safe_to_float(mkt_features.get("volume_ratio"), 1.0),
            "volume_imbalance": utils.safe_to_float(mkt_features.get("volume_imbalance")),
            "orderbook_imbalance": utils.safe_to_float(mkt_features.get("orderbook_imbalance")),
            "depth_ratio": utils.safe_to_float(mkt_features.get("depth_ratio"), 1.0),
            "CVD1m": utils.safe_to_float(mkt_features.get("CVD1m")),
            "CVD5m": utils.safe_to_float(mkt_features.get("CVD5m")),
            "vol1m": utils.safe_to_float(mkt_features.get("vol1m")),
            "vol5m": utils.safe_to_float(mkt_features.get("vol5m")),
            "timestamp": dt.datetime.utcnow().isoformat()
        }
        feature_snapshot = {
            "wall_size_ratio": wall_size_ratio,
            "wall_rating": wall_rating,
            "price_distance_pct": price_diff_pct,
            "volume_imbalance": market_snapshot["volume_imbalance"],
            "market_depth_ratio": market_snapshot["depth_ratio"],
            "recent_volatility": atr_pct,
            "cluster_strength": wall_data.get("cluster_strength", 0.0),
            "time_of_day": (time.localtime().tm_hour * 60 + time.localtime().tm_min) / 1440.0,
            "momentum_5m": utils.safe_to_float(mkt_features.get("pct5m")),
            "adx_strength": market_snapshot["adx14"],
            "volume_ratio": market_snapshot["volume_ratio"],
            "orderbook_imbalance": market_snapshot["orderbook_imbalance"],
            "spread_pct": spread_pct
        }
        training_payload = {
            "signal_id": signal_id,
            "timestamp": dt.datetime.utcnow().isoformat(),
            "symbol": symbol,
            "mode": mode,
            "side": side_to_enter,
            "wall_side": wall_side,
            "confidence": confidence_level,
            "wall_price": wall_price,
            "last_price": last_price,
            "wall_size": wall_size,
            "wall_rating": wall_rating,
            "wall_zscore": wall_z,
            "wall_hold_ratio": wall_hold_ratio,
            "cluster_size": wall_data.get("cluster_size"),
            "ml_score": ml_score,
            "feature_snapshot": feature_snapshot,
            "market_snapshot": market_snapshot
        }
        await bot.register_dom_signal(training_payload)

        signal_key = (symbol, side_to_enter, f"dom_{mode}")
        if signal_key in bot.active_signals:
            return True

        bot.active_signals.add(signal_key)
        ml_repr = f"{ml_score:.2f}" if isinstance(ml_score, float) else "n/a"
        logger.info(
            f"✅ DOM-СИГНАЛ [{symbol}] {side_to_enter} | {mode} | "
            f"Стена: {wall_side}@{wall_price:.6f} | "
            f"z:{wall_z:.1f} rating:{wall_rating} hold:{wall_data.get('hold_ratio', 1.0):.2f} | "
            f"{reason} | ML:{ml_repr}"
        )

        await bot._process_signal(candidate, mkt_features, signal_key)
        return True

    except Exception as e:
        logger.error(f"[{symbol}] Ошибка в simplified_dom_strategy: {e}", exc_info=True)
        # Добавляем короткий cooldown при ошибке для предотвращения спама
        if hasattr(bot, 'strategy_cooldown_until'):
            bot.strategy_cooldown_until[symbol] = time.time() + 30
        return False


async def _find_and_validate_wall(bot, symbol: str, cfg: dict) -> dict | None:
    """
    Найти и проверить стену за один проход
    """
    # Найти стену
    wall = await _find_closest_wall_zscore(bot, symbol, cfg)
    if not wall:
        return None

    # Быстрая проверка на спуф (3 секунды вместо 10)
    if not await _quick_wall_validation(bot, symbol, wall, cfg):
        return None

    # Проверить рейтинг из памяти
    price_tick = bot.price_tick_map.get(symbol, 1e-4)
    rating = _get_wall_memory_rating(bot, symbol, wall["price"], wall["side"], price_tick, cfg)
    
    if rating < cfg.get("MIN_WALL_RATING", -1):
        return None

    hold_ratio = 1.0
    cluster = None
    if price_tick:
        cluster = round(wall["price"] / (price_tick * 100)) * (price_tick * 100)
    if cluster is not None:
        cluster_mem = bot.dom_wall_memory.get(symbol, {}).get(cluster, {})
        holds = cluster_mem.get("holds", 0)
        breaches = cluster_mem.get("breaches", 0)
        total = holds + breaches
        if total > 0:
            hold_ratio = holds / total
    min_hold_ratio = cfg.get("REQUIRED_HOLD_RATIO", 0.5)
    if hold_ratio < min_hold_ratio:
        logger.debug(f"[{symbol}] Стена отклонена: hold_ratio={hold_ratio:.2f} < {min_hold_ratio:.2f}")
        return None

    wall["rating"] = rating
    wall["hold_ratio"] = hold_ratio
    wall["validated_ts"] = time.time()
    return wall


# async def _quick_wall_validation(bot, symbol: str, wall: dict) -> bool:
#     """
#     Быстрая 3-секундная проверка стены на спуф + обновление памяти
#     """
#     wall_price = wall["price"]
#     wall_side = wall["side"]
#     initial_size = wall["size"]
#     tick_size = bot.price_tick_map.get(symbol, 1e-4)
#     cluster = round(wall_price / (tick_size * 100)) * (tick_size * 100)

#     for i in range(3):
#         await asyncio.sleep(1)
#         ob = getattr(bot.shared_ws, "orderbooks", {}).get(symbol)
#         if not ob:
#             return False

#         current_size = _get_wall_size(ob, wall_side, wall_price)
#         if current_size < initial_size * 0.8:
#             # Запись breach
#             bot.dom_wall_memory.setdefault(symbol, {}).setdefault(cluster, {'holds': 0, 'breaches': 0})
#             bot.dom_wall_memory[symbol][cluster]['breaches'] += 1
#             await asyncio.to_thread(bot._save_wall_memory)

#             return False

#     # Запись hold
#     bot.dom_wall_memory.setdefault(symbol, {}).setdefault(cluster, {'holds': 0, 'breaches': 0})
#     bot.dom_wall_memory[symbol][cluster]['holds'] += 1
#     await asyncio.to_thread(bot._save_wall_memory)

#     return True

async def _quick_wall_validation(bot, symbol: str, wall: dict, cfg: dict) -> bool:
    """
    Быстрая 3-секундная проверка стены на спуф + потокобезопасное обновление памяти
    """
    wall_price = wall["price"]
    wall_side = wall["side"]
    initial_size = wall["size"]
    tick_size = bot.price_tick_map.get(symbol, 1e-4) or 1e-4
    cluster = round(wall_price / (tick_size * 100)) * (tick_size * 100)

    for i in range(3):
        await asyncio.sleep(1)
        ob = getattr(bot.shared_ws, "orderbooks", {}).get(symbol)
        if not ob:
            return False

        current_size = _get_wall_size(ob, wall_side, wall_price)
        # Ранний выход при значительном уменьшении стены
        if current_size < initial_size * 0.8:
            async with bot.wall_memory_lock:
                mem_cluster = bot.dom_wall_memory.setdefault(symbol, {}).setdefault(
                    cluster, {'holds': 0, 'breaches': 0}
                )
                mem_cluster['breaches'] += 1
                await bot._save_wall_memory_locked()
            cooldown = cfg.get("BREACH_COOLDOWN_SEC", 0)
            if cooldown:
                bot.strategy_cooldown_until[symbol] = max(bot.strategy_cooldown_until.get(symbol, 0), time.time() + cooldown)
            return False

    async with bot.wall_memory_lock:
        mem_cluster = bot.dom_wall_memory.setdefault(symbol, {}).setdefault(
            cluster, {'holds': 0, 'breaches': 0}
        )
        mem_cluster['holds'] += 1
        await bot._save_wall_memory_locked()
    return True



# def _calculate_wall_confidence(wall_z: float, wall_rating: int, wall_size: float, cfg: dict) -> str:
#     """
#     Простая оценка уверенности в стене (использует cfg из config.py)
#     """
#     min_z_high = cfg.get("MIN_Z_HIGH", 2.0)
#     min_z_medium = cfg.get("MIN_Z_MEDIUM", 1.5)
#     min_rating_high = cfg.get("MIN_RATING_HIGH", 2)
    
#     if wall_z >= min_z_high and wall_rating >= min_rating_high:
#         return "HIGH"
#     elif wall_z >= min_z_medium and wall_rating >= 0:
#         return "MEDIUM" 
#     else:
#         return "LOW"


def _calculate_wall_confidence(wall_z: float, wall_rating: int, wall_size: float, cfg: dict) -> str:
    """
    Простая оценка уверенности в стене (с учётом размера)
    """
    min_z_high = cfg.get("MIN_Z_HIGH", 2.0)
    min_z_medium = cfg.get("MIN_Z_MEDIUM", 1.5)
    min_rating_high = cfg.get("MIN_RATING_HIGH", 2)
    base_size = float(cfg.get("WALL_BASE_SIZE", 10000))
    size_factor = wall_size / base_size if base_size > 0 else 1.0

    if wall_z >= min_z_high and wall_rating >= min_rating_high and size_factor >= 1.0:
        return "HIGH"
    if wall_z >= min_z_medium and wall_rating >= 0 and size_factor >= 0.5:
        return "MEDIUM"
    return "LOW"



def _make_trade_decision(wall_side: str, wall_price: float, last_price: float, 
                        wall_z: float, confidence: str, cfg: dict) -> dict | None:
    """
    Четкие правила для принятия торгового решения (использует cfg из config.py)
    """
    min_z_breakout = cfg.get("MIN_Z_BREAKOUT", 1.8)
    min_z_fade = cfg.get("MIN_Z_FADE", 1.3)
    
    # Для HIGH confidence можно снижать требования
    if confidence == "HIGH":
        min_z_breakout *= 0.9
        min_z_fade *= 0.9

    price_diff_pct = abs(wall_price - last_price) / last_price * 100
    max_distance_pct = cfg.get("MAX_DISTANCE_PCT", 0.3)

    if price_diff_pct > max_distance_pct:
        return None

    if wall_side == "Sell":
        if last_price > wall_price and wall_z >= min_z_breakout:
            return {"side": "Buy", "mode": "breakout", "reason": "breakout_above_sell_wall"}
        elif last_price < wall_price and wall_z >= min_z_fade:
            return {"side": "Sell", "mode": "fade", "reason": "fade_sell_wall_resistance"}
    
    else:  # Buy wall
        if last_price < wall_price and wall_z >= min_z_breakout:
            return {"side": "Sell", "mode": "breakout", "reason": "breakout_below_buy_wall"}
        elif last_price > wall_price and wall_z >= min_z_fade:
            return {"side": "Buy", "mode": "fade", "reason": "fade_buy_wall_support"}

    return None


def _passes_retest(bot, symbol: str, wall_price: float, wall_side: str, side: str,
                   mode: str, wall_data: dict, cfg: dict) -> bool:
    if not cfg.get("RETEST_REQUIRED", False):
        return True

    tick_size = bot.price_tick_map.get(symbol, 1e-4) or 1e-4
    retest_ticks = max(1, int(cfg.get("RETEST_TICKS", 2)))
    threshold = retest_ticks * tick_size

    best_bid = utils.safe_to_float(bot.best_bid_map.get(symbol)) or utils.safe_to_float(
        (bot.shared_ws.ticker_data.get(symbol, {}) or {}).get("bestBidPrice"), 0.0
    )
    best_ask = utils.safe_to_float(bot.best_ask_map.get(symbol)) or utils.safe_to_float(
        (bot.shared_ws.ticker_data.get(symbol, {}) or {}).get("bestAskPrice"), 0.0
    )
    last_price = _dom_get_last_price(bot, symbol)

    if best_bid <= 0 or best_ask <= 0 or last_price <= 0:
        return False

    if mode == "breakout":
        if wall_side == "Sell" and not (best_bid >= wall_price + threshold and last_price >= wall_price + threshold):
            return False
        if wall_side == "Buy" and not (best_ask <= wall_price - threshold and last_price <= wall_price - threshold):
            return False
    else:  # fade
        if wall_side == "Sell" and not (best_ask <= wall_price - threshold and last_price <= wall_price - threshold):
            return False
        if wall_side == "Buy" and not (best_bid >= wall_price + threshold and last_price >= wall_price + threshold):
            return False

    max_wait = cfg.get("RETEST_MAX_SECONDS", 15)
    validated_ts = wall_data.get("validated_ts", time.time())
    if time.time() - validated_ts > max_wait:
        return False

    return True


async def _quick_market_check(bot, symbol: str, side: str, mode: str, confidence: str,
                              mkt_features: dict, cfg: dict) -> bool:
    """
    Быстрая проверка рыночных условий
    """
    pct5m = utils.safe_to_float(mkt_features.get("pct5m"))
    pct1m = utils.safe_to_float(mkt_features.get("pct1m"))
    spread_pct = utils.safe_to_float(mkt_features.get("spread_pct"))
    adx14 = utils.safe_to_float(mkt_features.get("adx14"))
    cvd1m = utils.safe_to_float(mkt_features.get("CVD1m"))
    cvd5m = utils.safe_to_float(mkt_features.get("CVD5m"))
    volume_ratio = utils.safe_to_float(mkt_features.get("volume_ratio"), 1.0)
    vol1m = utils.safe_to_float(mkt_features.get("vol1m"))
    vol5m = utils.safe_to_float(mkt_features.get("vol5m"))
    price = utils.safe_to_float(mkt_features.get("price"), 0.0)
    atr14 = utils.safe_to_float(mkt_features.get("atr14"))

    max_spread = cfg.get("MAX_SPREAD_PCT", 0.25)
    if spread_pct and spread_pct > max_spread:
        return False

    if price > 0:
        atr_pct = (atr14 / price) * 100
        if atr_pct > 3.0 and confidence != "HIGH":
            return False
    else:
        atr_pct = 0.0

    trend_threshold = 1.8
    if side == "Buy" and pct5m < -trend_threshold:
        return False
    if side == "Sell" and pct5m > trend_threshold:
        return False

    min_cvd = cfg.get("MIN_CVD_DELTA", 0.0)
    min_vol_ratio = cfg.get("MIN_VOL_RATIO", 0.6)

    if mode == "breakout":
        if side == "Buy":
            if cvd1m < min_cvd and cvd5m < min_cvd * 0.5:
                return False
        else:
            if cvd1m > -min_cvd and cvd5m > -min_cvd * 0.5:
                return False
        if volume_ratio < min_vol_ratio and vol1m + vol5m <= 0:
            return False
        if adx14 < 18.0 and confidence != "HIGH":
            return False
        if pct1m and side == "Buy" and pct1m < 0:
            return False
        if pct1m and side == "Sell" and pct1m > 0:
            return False
    else:  # fade
        if adx14 > 38.0 and confidence != "HIGH":
            return False
        if side == "Sell" and pct1m > 0 and pct5m > 0:
            return False
        if side == "Buy" and pct1m < 0 and pct5m < 0:
            return False
        if volume_ratio < min_vol_ratio * 0.8 and confidence != "HIGH":
            return False

    # базовый guard bot-level
    ok, reason = await bot._entry_guard(symbol, side, mkt_features, {"source": f"dom_{mode}"})
    if not ok:
        logger.debug(f"[{symbol}] Guard блокировка: {reason}")
        return False

    return True


def _get_wall_size(ob: dict, side: str, price: float) -> float:
    """
    Достать текущий размер стены из стакана
    """
    try:
        if side == "Sell":
            levels = ob.get("asks", {})
        else:
            levels = ob.get("bids", {})
        
        # Поиск точного соответствия цены
        for p, q in levels.items():
            if abs(float(p) - price) < 1e-8:
                return float(q)
                
        return 0.0
    except Exception:
        return 0.0

async def update_wall_memory(bot, symbol: str, wall_price: float, side: str, result: str):
    """
    Обновляет рейтинг стены в зависимости от того, удержалась ли она (hold) или была пробита (breach)
    """
    try:
        price_tick = bot.price_tick_map.get(symbol, 1e-4)
        if price_tick == 0:
            price_tick = 1e-4
        cluster = round(wall_price / (price_tick * 100)) * (price_tick * 100)

        mem = bot.dom_wall_memory.setdefault(symbol, {})
        cluster_data = mem.setdefault(cluster, {'holds': 0, 'breaches': 0})

        if result == "hold":
            cluster_data['holds'] += 1
        elif result == "breach":
            cluster_data['breaches'] += 1

        # Сохраняем на диск
        bot.save_wall_memory()

        logger.info(f"[{symbol}] Обновлена память стен: {result} @ {wall_price:.6f} → {cluster_data}")
    except Exception as e:
        logger.error(f"[{symbol}] Ошибка обновления памяти стен: {e}")
