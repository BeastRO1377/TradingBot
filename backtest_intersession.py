"""
backtest_intersession.py

Псевдо-бэктест межсессионной логики.
Использует сохранённую историю из history.pkl и прогоняет генерацию сигналов
по «тихим» окнам (межсессия + выходные). Для каждого обнаруженного сигнала
проверяет доходность через 5/10/30 минут.

Запуск (из каталога TradingBot):
    source venv/bin/activate
    python backtest_intersession.py --start 2025-10-12 --end 2025-10-19
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from intersession_strategies import (
    generate_intersession_signals,
    is_quiet_market_window,
)

WINDOW = 60          # глубина окна минутных свечей
LOOKAHEADS = [5, 10, 30]  # горизонты оценки, минут


@dataclass
class SignalStat:
    symbol: str
    timestamp: dt.datetime
    source: str
    side: str
    price: float
    returns: Dict[int, Optional[float]]


def load_history(path: Path) -> Dict[str, pd.DataFrame]:
    with path.open("rb") as f:
        raw = pickle.load(f)
    candles_map = raw.get("candles", raw)
    result: Dict[str, pd.DataFrame] = {}
    for symbol, rows in candles_map.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "startTime" not in df.columns:
            continue
        df["startTime"] = pd.to_datetime(df["startTime"])
        df.sort_values("startTime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        result[symbol] = df
    return result


def compute_returns(df: pd.DataFrame, idx: int, side: str, price: float) -> Dict[int, Optional[float]]:
    out: Dict[int, Optional[float]] = {}
    for horizon in LOOKAHEADS:
        target_idx = idx + horizon
        if target_idx >= len(df):
            out[horizon] = None
            continue
        future_price = float(df.iloc[target_idx]["close"])
        if price <= 0 or future_price <= 0:
            out[horizon] = None
            continue
        if side == "Buy":
            ret = (future_price / price - 1.0) * 100.0
        else:
            ret = (price / future_price - 1.0) * 100.0
        out[horizon] = ret
    return out


def normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит DataFrame к колонкам open/high/low/close/volume, если они названы иначе.
    Возвращает новую копию.
    """
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


def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    start_ts: Optional[dt.datetime],
    end_ts: Optional[dt.datetime],
    throttle_sec: float,
) -> List[SignalStat]:
    stats: List[SignalStat] = []
    last_signal_time: Dict[str, dt.datetime] = {}

    df = normalise_ohlcv(df)

    for idx in range(WINDOW, len(df)):
        ts = df.iloc[idx]["startTime"].to_pydatetime()
        if start_ts and ts < start_ts:
            continue
        if end_ts and ts > end_ts:
            break
        if not is_quiet_market_window(ts):
            continue

        if last_signal_time.get(symbol):
            delta = (ts - last_signal_time[symbol]).total_seconds()
            if delta < throttle_sec:
                continue

        window_df = df.iloc[idx - WINDOW : idx + 1].reset_index(drop=True)
        window_df = normalise_ohlcv(window_df)
        signals = generate_intersession_signals(symbol, window_df, now=ts)
        if not signals:
            continue

        last_signal_time[symbol] = ts

        price = float(window_df.iloc[-1]["close"])
        for sig in signals:
            if sig.get("side") not in ("Buy", "Sell"):
                continue
            returns = compute_returns(df, idx, sig["side"], price)
            stats.append(
                SignalStat(
                    symbol=symbol,
                    timestamp=ts,
                    source=str(sig.get("source", "")),
                    side=str(sig.get("side")),
                    price=price,
                    returns=returns,
                )
            )
            break  # учитываем только один сигнальный тип на минуту

    return stats


def summarise(stats: List[SignalStat]) -> pd.DataFrame:
    if not stats:
        return pd.DataFrame()
    rows = []
    for s in stats:
        row = {
            "symbol": s.symbol,
            "timestamp": s.timestamp,
            "source": s.source,
            "side": s.side,
            "price": s.price,
        }
        for horizon, value in s.returns.items():
            row[f"ret_{horizon}m"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def format_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "Сигналов не обнаружено."
    lines = []
    lines.append(f"Всего сигналов: {len(df)}")
    for source, group in df.groupby("source"):
        lines.append(f"\nИсточник: {source} (n={len(group)})")
        for horizon in LOOKAHEADS:
            col = f"ret_{horizon}m"
            valid = group[col].dropna()
            if valid.empty:
                lines.append(f"  {horizon}m: нет данных")
                continue
            avg = valid.mean()
            med = valid.median()
            pos_ratio = (valid > 0).mean() * 100.0
            lines.append(
                f"  {horizon}m: средняя {avg:+.2f}% | медиана {med:+.2f}% | "
                f"доля >0: {pos_ratio:.1f}%"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Псевдо-бэктест межсессионных сигналов.")
    parser.add_argument("--history", type=Path, default=config.HISTORY_FILE, help="Путь к history.pkl")
    parser.add_argument("--start", type=str, default=None, help="Начало периода (UTC, YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Конец периода (UTC, YYYY-MM-DD)")
    parser.add_argument("--throttle", type=float, default=float(config.INTERSESSION_THROTTLE_SEC), help="Троттлинг между сигналами по символу (секунды)")
    parser.add_argument("--symbols", type=str, nargs="*", default=None, help="Ограничить список символов")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.history.exists():
        raise SystemExit(f"Файл истории не найден: {args.history}")

    print(f"Загружаю историю из {args.history} ...")
    history = load_history(args.history)
    if args.symbols:
        history = {sym: history[sym] for sym in args.symbols if sym in history}
        missing = set(args.symbols) - set(history.keys())
        if missing:
            print(f"Предупреждение: отсутствуют данные для {', '.join(sorted(missing))}")

    start_ts = dt.datetime.fromisoformat(args.start) if args.start else None
    end_ts = dt.datetime.fromisoformat(args.end) if args.end else None

    all_stats: List[SignalStat] = []
    for symbol, df in history.items():
        stats = backtest_symbol(symbol, df, start_ts, end_ts, args.throttle)
        all_stats.extend(stats)
        print(f"{symbol}: найдено {len(stats)} сигналов")

    summary_df = summarise(all_stats)
    print("\n--- Итоги ---")
    print(format_summary(summary_df))

    output_path = Path("backtest_intersession_results.csv")
    if not summary_df.empty:
        summary_df.to_csv(output_path, index=False)
        print(f"\nПодробности сохранены в {output_path.resolve()}")


if __name__ == "__main__":
    main()
