# autotune.py
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

ROOT     = Path(__file__).resolve().parent
PNL_OK   = 1.5         # сделку считаем «удачной», если PnL ≥ 1.5 %
MIN_ROWS = 25          # минимум наблюдений на (symbol, side)

# ----------------------------------------------------------------------
# 1. Загрузка и предварительная очистка
# ----------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    """Читаем CSV как строки и конвертируем колонку timestamp в datetime."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)

    # унифицируем ISO-строки вида “…Z”  →  “…+00:00”
    if "timestamp" in df.columns:
        df["timestamp"] = (
            df["timestamp"]
            .str.replace("Z", "+00:00", regex=False)
            .pipe(pd.to_datetime, errors="coerce", utc=True)
            .dt.tz_localize(None)                 # делаем naïve-UTC
        )
        df.dropna(subset=["timestamp"], inplace=True)
        df["t_key"] = df["timestamp"].dt.floor("min")

    return df


def _load():
    trades = _read_csv(ROOT / "trades_unified.csv")
    snaps  = _read_csv(ROOT / "golden_setup_snapshots.csv")
    sqz    = _read_csv(ROOT / "squeeze_snapshots.csv")
    liq    = _read_csv(ROOT / "liquidation_snapshots.csv")
    return trades, snaps, sqz, liq


# ----------------------------------------------------------------------
# 2. Сшивка снапшотов c фактом сделки
# ----------------------------------------------------------------------
# --- внутри autotune.py -----------------------------------------------
def _join(snaps: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Совмещаем снапшоты и сделки:
      • по symbol + t_key (минутное окно)
      • по nap.direction  ↔  trades.side          (оба Buy/Sell)
    """
    if snaps.empty or trades.empty:
        return pd.DataFrame()

    # trades могут содержать 'side' вместо 'signal' – унифицируем
    if "signal" not in trades.columns and "side" in trades.columns:
        trades = trades.rename(columns={"side": "signal"})
    if "signal" not in snaps.columns and "side" in snaps.columns:
        snaps  = snaps.rename(columns={"side": "signal"})

    # на всякий случай приводим регистр
    for df in (snaps, trades):
        if "signal" in df.columns:
            df["signal"] = df["signal"].str.capitalize()   # buy → Buy

    df = (
        snaps
        .merge(trades, on=["symbol", "t_key", "signal"], suffixes=("_snap", "_trade"))
        .dropna(subset=["pnl"])
    )
    df["label"] = (df["pnl"].astype(float) >= PNL_OK).astype(int)
    return df

# ----------------------------------------------------------------------
# 3. Обучение логистической регрессии + сохранение весов
# ----------------------------------------------------------------------
def _fit(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Для каждого (symbol, side) обучаем LR-CV и сохраняем нормированные
    веса признаков.  Отбрасываем пары, где мало данных или нет 0/1.
    """
    out = []
    for (sym, side), grp in df.groupby(["symbol", "signal"]):
        if len(grp) < MIN_ROWS or grp["label"].nunique() < 2:
            continue

        X = grp[feature_cols].astype(float)
        y = grp["label"].values
        mdl = LogisticRegressionCV(cv=4, penalty="l2").fit(X, y)

        coef = abs(mdl.coef_[0])
        weights = coef / coef.sum()

        out.append({
            "symbol": sym,
            "side":   side,
            **{f"w_{c}": round(weights[i], 4) for i, c in enumerate(feature_cols)},
            "samples": len(grp),
        })

    return pd.DataFrame(out)


def retrain_all():
    trades, snaps, sqz, liq = _load()

    # ---------- Golden setup -----------------------------------------
    g_df = _join(snaps, trades)
    g_w  = _fit(g_df, ["price_change", "volume_change", "oi_change"])
    if not g_w.empty:
        g_w.to_csv(ROOT / "golden_feature_weights.csv", index=False)

    # ---------- Squeeze ----------------------------------------------
    if not sqz.empty:
        s_df = _join(sqz, trades)
        s_w  = _fit(s_df, ["pct_5m", "vol_change_pct", "squeeze_power"])
        if not s_w.empty:
            s_w.to_csv(ROOT / "squeeze_feature_weights.csv", index=False)

    # ---------- Liquidations -----------------------------------------
    if not liq.empty:
        l_df = _join(liq, trades)
        l_w  = _fit(l_df, ["liq_value", "delta_oi"])
        if not l_w.empty:
            l_w.to_csv(ROOT / "liq_feature_weights.csv", index=False)

    print(f"{datetime.utcnow():%Y-%m-%d %H:%M:%S}  ✓ autotune done")


if __name__ == "__main__":
    retrain_all()