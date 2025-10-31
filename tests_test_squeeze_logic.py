"""Unit‑tests for squeeze_logic_refactor.

Run:
    pytest -q
Requires:
    pip install pytest hypothesis  # hypothesis optional but recommended
"""

from __future__ import annotations

import math
import typing as _t

import pytest
from hypothesis import given, strategies as st

from squeeze_logic_refactor import (
    Action,
    SqueezeFeatures,
    SqueezeParams,
    decide_squeeze,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def params() -> SqueezeParams:  # noqa: D401
    """Global params instance – override YAML watcher in tests."""
    return SqueezeParams()


@pytest.fixture()
def base_feat() -> SqueezeFeatures:  # noqa: D401
    return SqueezeFeatures(price=100.0, atr_1m=0.3, ret_1m=0.0)


# ---------------------------------------------------------------------------
# Deterministic behaviour tests
# ---------------------------------------------------------------------------


def test_low_score_cancels(base_feat: SqueezeFeatures, params: SqueezeParams) -> None:
    f = base_feat
    f.ret_1m = 0.0005  # 0.05 % move → very low score
    d = decide_squeeze(f, side="Buy", params=params)
    assert d["action"] == Action.CANCEL
    assert d["score"] < params.squeeze_min_score


@pytest.mark.parametrize(
    "ret,expected",
    [
        (0.015, Action.ENTER),  # huge move → ENTER
        (0.0005, Action.CANCEL),  # tiny move → CANCEL
        (0.005, Action.WAIT),  # mid‑move → WAIT
    ],
)
def test_action_matrix(
    base_feat: SqueezeFeatures, params: SqueezeParams, ret: float, expected: Action
) -> None:
    f = base_feat
    f.ret_1m = ret
    f.liq_sigma = 3.0  # medium liquidity → neutral
    d = decide_squeeze(f, side="Buy", params=params)
    assert d["action"] == expected


def test_stop_loss_direction(base_feat: SqueezeFeatures, params: SqueezeParams) -> None:
    f = base_feat
    f.ret_1m = 0.015
    d_long = decide_squeeze(f, side="Buy", params=params)
    d_short = decide_squeeze(f, side="Sell", params=params)
    assert d_long["stop_loss"] < f.price  # long stop below
    assert d_short["stop_loss"] > f.price  # short stop above


def test_risk_size_zero_when_not_enter(
    base_feat: SqueezeFeatures, params: SqueezeParams
) -> None:
    f = base_feat
    f.ret_1m = 0.002  # ambiguous
    d = decide_squeeze(f, side="Buy", params=params)
    if d["action"] != Action.ENTER:
        assert d["risk_size"] == 0.0


# ---------------------------------------------------------------------------
# Property‑based fuzzing – should not crash or emit NaN
# ---------------------------------------------------------------------------


@given(
    price=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    atr=st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False),
    ret=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
)
def test_decision_stability(
    price: float, atr: float, ret: float, params: SqueezeParams
) -> None:
    f = SqueezeFeatures(price=price, atr_1m=atr, ret_1m=ret)
    d = decide_squeeze(f, side="Buy", params=params)
    # score must be finite and within [0,1]
    assert math.isfinite(d["score"]) and 0.0 <= d["score"] <= 1.0
    # risk_size non‑negative
    assert d["risk_size"] >= 0.0
