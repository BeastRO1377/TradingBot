"""
research_intersession.py

Исследование порогов межсессионной логики.
Загружает history.pkl, перебирает сетку порогов и считает:
    * сколько «тихих» окон найдено (условие cond_min);
    * сколько сигналов (MR VWAP / Sweep Fade / Grid);
    * среднюю доходность сигналов через 5/10/30 минут.

Пример запуска:
    source venv/bin/activate
    python research_intersession.py --start 2025-10-12 --end 2025-10-19 --max-symbols 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from intersession_strategies import (
    is_quiet_market_window,
    signal_mr_vwap,
    signal_sweep_fade,
    grid_levels_from_range,
)

WINDOW = 60
LOOKAHEADS = [5, 10, 30]


@dataclass(frozen=True)
class ParamSet:
    hhll_coeff: float
    nsr_min: float
    rv_percentile: float
    cond_min: int


def load_history(path: Path) -> Dict[str, pd.DataFrame]:
    with path.open("rb") as f:
        raw = pickle.load(f)
    candles_map = raw.get("candles", raw)
    out: Dict[str, pd.DataFrame] = {}
    for symbol, rows in candles_map.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "startTime" not in df.columns:
            continue
        df["startTime"] = pd.to_datetime(df["startTime"])
        df.sort_values("startTime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        out[symbol] = df
    return out


def normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "open": ["open", "openPrice", "o"],
        "high": ["high", "highPrice", "h"],
        "low": ["low", "lowPrice", "l"],
        "close": ["close", "closePrice", "c"],
        "volume": ["volume", "turnover", "v"],
    }
    out = df.copy()
    for target, candidates in mapping.items():
        if target in out.columns:
            continue
        for src in candidates:
            if src in out.columns:
                out[target] = out[src]
                break
    return out


def compute_returns(df: pd.DataFrame, idx: int, side: str, price: float) -> Dict[int, Optional[float]]:
    res: Dict[int, Optional[float]] = {}
    for horizon in LOOKAHEADS:
        j = idx + horizon
        if j >= len(df):
            res[horizon] = None
            continue
        future_price = float(df.iloc[j]["close"])
        if price <= 0 or future_price <= 0:
            res[horizon] = None
            continue
        if side == "Buy":
            res[horizon] = (future_price / price - 1.0) * 100.0
        else:
            res[horizon] = (price / future_price - 1.0) * 100.0
    return res


def iter_param_sets(
    hhll_coeffs: Iterable[float],
    nsr_values: Iterable[float],
    rv_percentiles: Iterable[float],
    cond_values: Iterable[int],
) -> List[ParamSet]:
    combos = []
    for hhll, nsr, rv, cond in itertools.product(hhll_coeffs, nsr_values, rv_percentiles, cond_values):
        combos.append(ParamSet(hhll, nsr, rv, cond))
    return combos


def evaluate_symbol(
    symbol: str,
    df_raw: pd.DataFrame,
    params: List[ParamSet],
    start_ts: Optional[dt.datetime],
    end_ts: Optional[dt.datetime],
) -> Dict[ParamSet, Dict[str, object]]:
    stats: Dict[ParamSet, Dict[str, object]] = {
        p: {
            "quiet": 0,
            "signals": 0,
            "source_counts": Counter(),
            "returns_sum": {h: 0.0 for h in LOOKAHEADS},
            "returns_n": {h: 0 for h in LOOKAHEADS},
        }
        for p in params
    }

    df = normalise_ohlcv(df_raw)
    df_len = len(df)
    if df_len <= WINDOW:
        return stats

    close = pd.to_numeric(df["close"])
    high = pd.to_numeric(df["high"])
    low = pd.to_numeric(df["low"])

    for idx in range(WINDOW, df_len):
        ts = df.iloc[idx]["startTime"].to_pydatetime()
        if start_ts and ts < start_ts:
            continue
        if end_ts and ts > end_ts:
            break
        if not is_quiet_market_window(ts):
            continue

        slice30 = slice(idx - 30, idx + 1)
        slice60 = slice(idx - 60, idx + 1)

        hhll_30 = float(high.iloc[slice30].max() - low.iloc[slice30].min())
        hi_roll = high.iloc[slice60].rolling(5).max().iloc[-1]
        lo_roll = low.iloc[slice60].rolling(5).min().iloc[-1]
        atr_5m = float((hi_roll - lo_roll) / 5.0) if not np.isnan(hi_roll) and not np.isnan(lo_roll) else 0.0
        abs_diff = close.iloc[slice30].diff().abs()
        total_move = float(abs_diff.sum())
        net_move = float(abs(float(close.iloc[idx]) - float(close.iloc[idx - 30])) if idx - 30 >= 0 else 0.0)
        nsr = (total_move / net_move) if net_move > 0 else 10.0
        log_returns = np.log(close.iloc[slice30].pct_change().add(1.0).fillna(1.0))
        rv = float(np.sqrt(np.sum(log_returns**2)))

        window_df = df.iloc[idx - WINDOW : idx + 1].reset_index(drop=True)
        price = float(window_df.iloc[-1]["close"])

        for param in params:
            conds = 0
            if atr_5m > 0 and hhll_30 <= param.hhll_coeff * atr_5m:
                conds += 1
            if nsr >= param.nsr_min:
                conds += 1
            percentile_threshold = np.nanpercentile(np.abs(log_returns.fillna(0.0)), param.rv_percentile)
            if rv <= percentile_threshold:
                conds += 1

            if conds < param.cond_min:
                continue

            entry = stats[param]
            entry["quiet"] += 1

            signal_candidates: List[Dict[str, object]] = []
            s1 = signal_mr_vwap(symbol, window_df)
            if s1:
                signal_candidates.append(s1)
            s2 = signal_sweep_fade(symbol, window_df)
            if s2:
                signal_candidates.append(s2)
            gl = grid_levels_from_range(window_df)
            if gl:
                signal_candidates.append({"symbol": symbol, "side": "GRID", "source": "INTRAQUIET_GRID_HINT", "grid_levels": gl})

            selected: Optional[Dict[str, object]] = None
            for candidate in signal_candidates:
                if candidate.get("side") in ("Buy", "Sell"):
                    selected = candidate
                    entry["signals"] += 1
                    source = str(candidate.get("source", "unknown"))
                    entry["source_counts"][source] += 1
                    returns = compute_returns(df, idx, str(candidate["side"]), price)
                    for horizon, value in returns.items():
                        if value is None:
                            continue
                        entry["returns_sum"][horizon] += value
                        entry["returns_n"][horizon] += 1
                    break

    return stats


def summarise(all_stats: Dict[ParamSet, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for param, data in all_stats.items():
        quiet = data["quiet"]
        signals = data["signals"]
        row = {
            "hhll_coeff": param.hhll_coeff,
            "nsr_min": param.nsr_min,
            "rv_percentile": param.rv_percentile,
            "cond_min": param.cond_min,
            "quiet_windows": quiet,
            "signals": signals,
        }
        for horizon in LOOKAHEADS:
            n = data["returns_n"][horizon]
            avg = data["returns_sum"][horizon] / n if n else np.nan
            row[f"avg_ret_{horizon}m"] = avg
            row[f"n_ret_{horizon}m"] = n
        rows.append(row)
    df = pd.DataFrame(rows)
    df.sort_values(["signals", "quiet_windows"], ascending=False, inplace=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research thresholds for intersession strategy.")
    parser.add_argument("--history", type=Path, default=config.HISTORY_FILE)
    parser.add_argument("--start", type=str, default=None, help="UTC start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="UTC end date (YYYY-MM-DD)")
    parser.add_argument("--max-symbols", type=int, default=40, help="Limit number of symbols (по алфавиту)")
    parser.add_argument(
        "--hhll",
        type=str,
        default="[0.8, 1.0, 1.2]",
        help="JSON список коэффициентов для диапазона hhll/ATR5",
    )
    parser.add_argument(
        "--nsr",
        type=str,
        default="[1.8, 2.0, 2.2]",
        help="JSON список порогов для NSR",
    )
    parser.add_argument(
        "--rv",
        type=str,
        default="[35, 45, 55]",
        help="JSON список перцентилей для rv",
    )
    parser.add_argument(
        "--cond",
        type=str,
        default="[1, 2]",
        help="JSON список минимального числа выполненных условий",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.history.exists():
        raise SystemExit(f"Не найден файл истории: {args.history}")

    try:
        hhll_values = json.loads(args.hhll)
        nsr_values = json.loads(args.nsr)
        rv_values = json.loads(args.rv)
        cond_values = json.loads(args.cond)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Ошибка чтения параметров JSON: {exc}") from exc

    params = iter_param_sets(hhll_values, nsr_values, rv_values, cond_values)
    print(f"Комбинаций порогов: {len(params)}")

    history = load_history(args.history)
    symbols = sorted(history.keys())
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    start_ts = dt.datetime.fromisoformat(args.start) if args.start else None
    end_ts = dt.datetime.fromisoformat(args.end) if args.end else None

    agg_stats: Dict[ParamSet, Dict[str, object]] = {
        p: {
            "quiet": 0,
            "signals": 0,
            "source_counts": Counter(),
            "returns_sum": {h: 0.0 for h in LOOKAHEADS},
            "returns_n": {h: 0 for h in LOOKAHEADS},
        }
        for p in params
    }

    for symbol in symbols:
        df = history[symbol]
        symbol_stats = evaluate_symbol(symbol, df, params, start_ts, end_ts)
        for param in params:
            target = agg_stats[param]
            source = symbol_stats[param]
            target["quiet"] += source["quiet"]
            target["signals"] += source["signals"]
            target["source_counts"].update(source["source_counts"])
            for horizon in LOOKAHEADS:
                target["returns_sum"][horizon] += source["returns_sum"][horizon]
                target["returns_n"][horizon] += source["returns_n"][horizon]
        print(f"{symbol}: окна={sum(symbol_stats[p]['quiet'] for p in params)}")

    summary_df = summarise(agg_stats)
    print("\n--- Итоги ---")
    if summary_df.empty:
        print("Сигналов не найдено ни для одной комбинации порогов.")
    else:
        print(summary_df.head(20).to_string(index=False))
        out_path = Path("research_intersession_summary.csv")
        summary_df.to_csv(out_path, index=False)
        print(f"\nВся таблица сохранена в {out_path.resolve()}")


if __name__ == "__main__":
    main()
