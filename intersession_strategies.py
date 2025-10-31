# intersession_strategies.py
import logging
import numpy as np
import pandas as pd
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple

import config
import utils

REGIME_RANGE = "RANGE"
logger = logging.getLogger(__name__)

def is_intersession_utc(now: Optional[dt.datetime] = None) -> bool:
    """True, если текущий UTC-временной момент в окне 21:00..00:00."""
    if now is None:
        now = dt.datetime.utcnow()
    t = now.time()
    return (t >= dt.time(21, 0, 0)) or (t < dt.time(0, 0, 0))

def is_quiet_market_window(now: Optional[dt.datetime] = None) -> bool:
    """True, если межсессионное окно или выходные (до воскресенья 22:00 UTC)."""
    if now is None:
        now = dt.datetime.utcnow()

    if config.WEEKEND_INTERSESSION_ENABLED:
        weekday = now.weekday()  # Monday=0 ... Sunday=6
        if weekday == 5:  # Saturday
            return True
        if weekday == 6 and now.time() < dt.time(22, 0, 0):  # Sunday before 22:00 UTC
            return True

    return is_intersession_utc(now)

def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["high"]) + pd.to_numeric(df["low"]) + pd.to_numeric(df["close"])) / 3.0

def micro_vwap(df: pd.DataFrame, minutes: int = 10) -> Tuple[float, float]:
    """Микро-VWAP+сигма (по OHLCV как фоллбэк)."""
    s = df.tail(minutes)
    tp = _typical_price(s)
    vol = pd.to_numeric(s["volume"]).replace(0, np.nan).fillna(method="ffill").fillna(0)
    vwap = float((tp * vol).sum() / (vol.sum() or 1.0))
    r = np.log(pd.to_numeric(s["close"]).pct_change().add(1.0).replace([np.inf, -np.inf], np.nan).fillna(0) + 1.0)
    sigma = float(r.std(ddof=0) * max(len(s),1) ** 0.5)
    return vwap, max(sigma, 1e-9)

def detect_intraquiet(df: pd.DataFrame) -> bool:
    """Low-vol, узкий диапазон, пила (NSR высокий)."""
    if len(df) < 60:
        return False
    last30 = df.tail(30)
    last60 = df.tail(60)
    hhll_30 = float(np.max(pd.to_numeric(last30["high"])) - np.min(pd.to_numeric(last30["low"])))
    atr_5m = float(pd.to_numeric(last60["high"]).rolling(5).max().iloc[-1] - pd.to_numeric(last60["low"]).rolling(5).min().iloc[-1]) / 5.0
    total_move = float(np.abs(pd.to_numeric(last30["close"]).diff()).sum())
    net_move = float(abs(float(last30["close"].iloc[-1]) - float(last30["close"].iloc[0])))
    nsr = (total_move / net_move) if net_move > 0 else 10.0
    rv = float(np.sqrt(np.sum(np.log(pd.to_numeric(last30["close"]).pct_change().add(1.0).fillna(1.0)) ** 2)))

    HHLL_COEFF = 1.0
    NSR_MIN = 1.8
    RV_PERCENTILE = 45
    COND_MIN = 2

    conds = 0
    conds += int(hhll_30 <= HHLL_COEFF * atr_5m if atr_5m > 0 else False)
    conds += int(nsr >= NSR_MIN)
    rv_threshold = np.nanpercentile(
        np.log(pd.to_numeric(df["close"]).pct_change().add(1.0).fillna(1.0)).abs(),
        RV_PERCENTILE
    )
    conds += int(rv <= rv_threshold)
    logger.debug(
        "[quiet] hhll=%.4f atr5=%.4f nsr=%.2f rv=%.4f conds=%d",
        hhll_30,
        atr_5m,
        nsr,
        rv,
        conds,
    )
    return conds >= COND_MIN

def signal_mr_vwap(symbol: str, df: pd.DataFrame, z_entry: float = 1.4) -> Optional[Dict[str, Any]]:
    """MR к micro-VWAP."""
    if len(df) < 15: return None
    vwap, sigma = micro_vwap(df, minutes=min(len(df), 10))
    last = float(df["close"].iloc[-1])
    z = (last - vwap) / (sigma or 1e-9)
    if abs(z) < z_entry: return None
    side = "Buy" if z < 0 else "Sell"
    return {
        "symbol": symbol,
        "side": side,
        "source": "INTRAQUIET_MR_VWAP",
        "justification": f"microVWAP={vwap:.6f}, z={z:.2f}",
        "entry_hint": {"type": "market"}
    }

def _wick_ratio(row: pd.Series) -> float:
    high = float(row["high"]); low = float(row["low"]); open_ = float(row.get("open", row["close"]))
    close = float(row["close"]); rng = max(high - low, 1e-9); body = abs(close - open_)
    return (rng - body) / rng

def signal_sweep_fade(symbol: str, df: pd.DataFrame, z_jump: float = 3.2, wick_thr: float = 0.6) -> Optional[Dict[str, Any]]:
    """Фейд свипа."""
    if len(df) < 6: return None
    cl = np.log(pd.to_numeric(df["close"]).pct_change().add(1.0).fillna(1.0))
    mu = float(cl.rolling(50).mean().iloc[-1] or 0)
    sd = float(cl.rolling(50).std(ddof=0).iloc[-1] or 1e-9)
    last_r = float(cl.iloc[-1]); z = (last_r - mu) / (sd or 1e-9)
    wick = _wick_ratio(df.iloc[-1])
    prev = df.iloc[-2]
    prev_body_top = max(float(prev["open"]), float(prev["close"]))
    prev_body_bot = min(float(prev["open"]), float(prev["close"]))
    close_back_in = prev_body_bot <= float(df["close"].iloc[-1]) <= prev_body_top
    if abs(z) >= z_jump and wick >= wick_thr and close_back_in:
        side = "Sell" if z > 0 else "Buy"
        return {
            "symbol": symbol,
            "side": side,
            "source": "INTRAQUIET_SWEEP_FADE",
            "justification": f"z_jump={z:.2f}, wick={wick:.2f}, re-entry",
            "entry_hint": {"type": "market"}
        }
    return None

def grid_levels_from_range(df: pd.DataFrame, step_atr_mult: float = 0.2, levels: int = 3) -> Optional[Tuple[List[float], List[float]]]:
    """Симметричная сетка уровней по 30-мин диапазону."""
    if len(df) < 30: return None
    last30 = df.tail(30)
    lo = float(np.min(pd.to_numeric(last30["low"]))); hi = float(np.max(pd.to_numeric(last30["high"])))
    atr_proxy = (hi - lo) / 14.0 if (hi - lo) > 0 else 0.0
    if atr_proxy <= 0: return None
    step = max(atr_proxy * step_atr_mult, 1e-9); mid = (hi + lo) / 2.0
    buys  = [mid - (i+1)*step for i in range(levels)]
    sells = [mid + (i+1)*step for i in range(levels)]
    return buys, sells

def generate_intersession_signals(
    symbol: str,
    df: pd.DataFrame,
    now: Optional[dt.datetime] = None,
    market_snapshot: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not is_quiet_market_window(now): return out
    if market_snapshot:
        spread_pct = utils.safe_to_float(
            (market_snapshot.get("orderbook") or {}).get("orderbook_implied_spread_pct", 0.0)
            or market_snapshot.get("spread_pct", 0.0)
        )
        depth_ratio = utils.safe_to_float(
            (market_snapshot.get("orderbook") or {}).get("depth_ratio", 0.0)
        )
        config_liq = getattr(config, "INTERSESSION_CONFIG", {}).get("LIQUIDITY", {})
        if spread_pct and spread_pct > float(config_liq.get("SPREAD_PCT_Q", 0.75)) * 100:
            return out
        if depth_ratio and depth_ratio < float(config_liq.get("DEPTH_RATIO_MIN", 0.35)):
            return out
    if not detect_intraquiet(df): return out
    s1 = signal_mr_vwap(symbol, df);       s2 = signal_sweep_fade(symbol, df); gl = grid_levels_from_range(df)
    if s1: out.append(s1)
    if s2: out.append(s2)
    if gl:
        out.append({"symbol": symbol, "side": "GRID", "source": "INTRAQUIET_GRID_HINT", "justification": "range grid", "grid_levels": gl})
    return out
