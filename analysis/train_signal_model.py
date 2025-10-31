#!/usr/bin/env python3
"""
Train a tabular classifier on trades_unified.csv to predict profitable trades.

The script:
  1. Pairs OPEN and CLOSE rows from trades_unified.csv.
  2. Builds a feature dict per trade (numeric indicators + categorical symbol/side).
  3. Splits data into train/test.
  4. Tries to train a LightGBM model; if LightGBM is unavailable, falls back to
     sklearn's HistGradientBoostingClassifier.
  5. Prints evaluation metrics (accuracy, precision/recall/F1, ROC-AUC) and top
     feature importances.
  6. Saves the fitted model and vectorizer into analysis/models/.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    from sklearn.ensemble import HistGradientBoostingClassifier

    LIGHTGBM_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parents[1]
TRADES_FILE = BASE_DIR / "trades_unified.csv"
MODEL_DIR = BASE_DIR / "analysis" / "models"
FEATURE_SNAPSHOT_FILE = BASE_DIR / "analysis" / "trade_feature_snapshots.csv"

FEATURE_COLUMNS = [
    "pct1m",
    "pct5m",
    "pct15m",
    "dOI1m",
    "dOI5m",
    "GS_dOI4m",
    "CVD1m",
    "CVD5m",
    "volume_anomaly",
    "vol1m",
    "vol5m",
    "vol15m",
    "rsi14",
    "adx14",
]


def parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def float_or_nan(raw: str | None) -> float:
    try:
        val = float(raw) if raw not in (None, "") else math.nan
        if math.isfinite(val):
            return val
    except (TypeError, ValueError):
        pass
    return math.nan


def load_trade_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Trades file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def pair_trades(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    open_map: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    completed: List[Dict[str, object]] = []
    prev_feature_map: Dict[str, Dict[str, float]] = defaultdict(dict)

    for row in rows:
        ts = parse_timestamp(row.get("timestamp"))
        if not ts:
            continue
        event = (row.get("event") or "").lower()
        symbol = row.get("symbol", "").strip()
        side = (row.get("side") or "").strip().capitalize()
        if not symbol or not side:
            continue
        key = (symbol, side)

        if event == "open":
            prev_state = prev_feature_map.get(symbol, {}).copy()
            open_map[key].append({"open_ts": ts, "row": row, "prev_state": prev_state})
            prev_feature_map[symbol] = {
                "CVD1m": float_or_nan(row.get("CVD1m")),
                "CVD5m": float_or_nan(row.get("CVD5m")),
                "pct5m": float_or_nan(row.get("pct5m")),
            }
        elif event == "close":
            queue = open_map.get(key)
            if not queue:
                continue
            entry = queue.pop(0)
            entry["close_ts"] = ts
            entry["symbol"] = symbol
            entry["side"] = side
            entry["pnl_usdt"] = float_or_nan(row.get("pnl_usdt"))
            entry["pnl_pct"] = float_or_nan(row.get("pnl_pct"))
            entry["close_row"] = row
            completed.append(entry)

    return completed


def build_dataset(trades: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, DictVectorizer]:
    samples: List[Dict[str, object]] = []
    labels: List[int] = []

    for trade in trades:
        open_row: Dict[str, str] = trade["row"]
        prev_state = trade.get("prev_state") or {}
        close_ts: datetime = trade.get("close_ts")
        open_ts: datetime = trade.get("open_ts")

        if not math.isfinite(trade.get("pnl_usdt", math.nan)):
            continue

        feature_dict: Dict[str, object] = {
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "duration_sec": (close_ts - open_ts).total_seconds() if close_ts and open_ts else 0.0,
        }

        for col in FEATURE_COLUMNS:
            feature_dict[f"feature_{col}"] = float_or_nan(open_row.get(col))

        curr_cvd1 = float_or_nan(open_row.get("CVD1m"))
        curr_cvd5 = float_or_nan(open_row.get("CVD5m"))
        prev_cvd1 = prev_state.get("CVD1m", curr_cvd1)
        prev_cvd5 = prev_state.get("CVD5m", curr_cvd5)
        delta_cvd1 = curr_cvd1 - prev_cvd1
        delta_cvd5 = curr_cvd5 - prev_cvd5
        feature_dict["feature_delta_cvd1m"] = delta_cvd1
        feature_dict["feature_delta_cvd5m"] = delta_cvd5
        feature_dict["feature_prev_cvd1m"] = prev_cvd1
        feature_dict["feature_prev_cvd5m"] = prev_cvd5
        sign_change_cvd1 = 0.0
        if not math.isnan(prev_cvd1):
            if prev_cvd1 <= 0 < curr_cvd1:
                sign_change_cvd1 = 1.0
            elif prev_cvd1 >= 0 > curr_cvd1:
                sign_change_cvd1 = -1.0
        feature_dict["feature_cvd1m_sign_change"] = sign_change_cvd1
        sign_change_cvd5 = 0.0
        if not math.isnan(prev_cvd5):
            if prev_cvd5 <= 0 < curr_cvd5:
                sign_change_cvd5 = 1.0
            elif prev_cvd5 >= 0 > curr_cvd5:
                sign_change_cvd5 = -1.0
        feature_dict["feature_cvd5m_sign_change"] = sign_change_cvd5
        curr_pct5 = float_or_nan(open_row.get("pct5m"))
        prev_pct5 = prev_state.get("pct5m", curr_pct5)
        feature_dict["feature_delta_pct5m"] = curr_pct5 - prev_pct5

        samples.append(feature_dict)
        labels.append(1 if trade["pnl_usdt"] > 0 else 0)

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(samples)
    y = np.array(labels, dtype=np.int8)
    return X, y, vectorizer


def train_model(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if LIGHTGBM_AVAILABLE:
        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:  # fallback
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            random_state=42,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    print("Model:", model.__class__.__name__)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))

    if y_score is not None and len(np.unique(y_test)) > 1:
        try:
            roc = roc_auc_score(y_test, y_score)
            print("ROC-AUC:", roc)
        except ValueError:
            pass

    return model, (X_train, X_test, y_train, y_test)


def display_feature_importance(model, vectorizer: DictVectorizer) -> None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_") and model.coef_.ndim == 1:
        importances = np.abs(model.coef_)
    else:
        print("Model does not expose feature importances.")
        return

    features = vectorizer.get_feature_names_out()
    pairs = sorted(zip(features, importances), key=lambda x: abs(x[1]), reverse=True)[:25]
    print("\nTop feature importances:")
    for name, value in pairs:
        print(f"  {name:25s} {value:.4f}")


def save_artifacts(model, vectorizer: DictVectorizer) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / "signal_model.pkl").open("wb") as fp:
        pickle.dump(model, fp)
    with (MODEL_DIR / "signal_vectorizer.pkl").open("wb") as fp:
        pickle.dump(vectorizer, fp)
    print(f"\nArtifacts saved to {MODEL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a trade outcome classifier.")
    parser.add_argument("--start", help="ISO timestamp; only use trades after this time.")
    parser.add_argument("--end", help="ISO timestamp; only use trades before this time.")
    args = parser.parse_args()

    rows = load_trade_rows(TRADES_FILE)
    paired = pair_trades(rows)

    if args.start:
        start_dt = parse_timestamp(args.start)
        paired = [t for t in paired if t.get("open_ts") and t["open_ts"] >= start_dt]
    if args.end:
        end_dt = parse_timestamp(args.end)
        paired = [t for t in paired if t.get("open_ts") and t["open_ts"] <= end_dt]

    if not paired:
        raise SystemExit("No trades available after filtering. Nothing to train on.")

    X, y, vectorizer = build_dataset(paired)
    model, datasets = train_model(X, y)
    display_feature_importance(model, vectorizer)
    save_artifacts(model, vectorizer)


if __name__ == "__main__":
    main()
