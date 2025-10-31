"""
Refactored squeeze decision module.

Key design goals
----------------
1. **Pure‑functional core** – no side effects, easy unit testing.
2. **Data classes** – explicit contracts for all inputs/outputs.
3. **Pluggable thresholds** – everything configurable via `SqueezeParams`.
4. **Vectorised maths** – single pass over features, no branching spaghetti.
5. **Detailed diagnostics** – every decision contains full context for logging/ML.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, TypedDict


class Action(str, enum.Enum):
    ENTER = "ENTER"
    WAIT = "WAIT"
    CANCEL = "CANCEL"


class Style(str, enum.Enum):
    LIMIT_PULLBACK = "limit_pullback"
    MARKET_CONTINUATION = "market_continuation"


class Decision(TypedDict):
    action: Action
    style: Style
    side: str               # "Buy" | "Sell"
    score: float            # 0‑1 squeeze strength
    exhaustion: float       # 0‑1
    continuation: float     # 0‑1
    justification: str


@dataclass(slots=True)
class SqueezeFeatures:
    price: float
    atr_1m: float
    ret_1m: float                 # absolute return, e.g. 0.012 = 1.2 %
    volume_pctl_1m: Optional[float] = None
    volume_z_1m: Optional[float] = None
    oi_delta_1m_pct: float = 0.0
    liq_sigma: float = 0.0
    impulse_dir: str = "up"       # "up" | "down"
    hi_5m: float = math.nan
    lo_5m: float = math.nan
    vwap_5m: float = math.nan


@dataclass(slots=True)
class SqueezeParams:
    # score weights
    w_move: float = 0.35
    w_vol: float = 0.25
    w_oi: float = 0.25
    w_liq: float = 0.15

    # thresholds
    squeeze_min_score: float = 0.55
    exhaustion_enter_thr: float = 0.15
    continuation_follow_thr: float = 0.40

    # risk/money management
    atr_k: float = 0.4

    # pullback fibonacci levels
    fib_levels: tuple[float, float, float] = (0.382, 0.50, 0.618)


def _lin01(x: float, lo: float, hi: float) -> float:
    """Linear mapping to [0,1] with clipping."""
    if hi <= lo:
        raise ValueError("lo must be < hi")
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def calc_squeeze_score(f: SqueezeFeatures, p: SqueezeParams) -> float:
    """Composite squeeze score 0‑1."""
    move_score = _lin01(abs(f.ret_1m), 0.003, 0.012)  # 0.3 % → 0, 1.2 % → 1
    if f.volume_pctl_1m is not None:
        vol_score = f.volume_pctl_1m
    elif f.volume_z_1m is not None:
        vol_score = _lin01(f.volume_z_1m, -0.5, 2.0)
    else:
        vol_score = 0.0
    oi_score = _lin01(abs(f.oi_delta_1m_pct) / 100, 0.1, 0.75)
    liq_score = _lin01(f.liq_sigma, 2.0, 5.0)

    score = (
        p.w_move * move_score
        + p.w_vol * vol_score
        + p.w_oi * oi_score
        + p.w_liq * liq_score
    )
    # hard‑clip to [0,1]
    return max(0.0, min(1.0, score))


def calc_exhaustion_continuation(f: SqueezeFeatures) -> tuple[float, float]:
    """Very crude heuristics – replace with ML later."""
    # exhaustion = rapid reversal vs impulse
    exh = _lin01(abs(f.ret_1m), 0.0, 0.008) * (1.0 - _lin01(f.liq_sigma, 2.0, 6.0))
    # continuation = strong momentum & liquidity
    cont = _lin01(abs(f.ret_1m), 0.004, 0.015) * _lin01(f.liq_sigma, 2.0, 6.0)
    return exh, cont


def decide_squeeze(f: SqueezeFeatures, p: SqueezeParams, side: str) -> Decision:
    """Return a deterministic decision dict."""
    score = calc_squeeze_score(f, p)
    exhaustion, continuation = calc_exhaustion_continuation(f)

    # 1) Preliminary filter
    if score < p.squeeze_min_score:
        return Decision(
            action=Action.CANCEL,
            style=Style.LIMIT_PULLBACK,
            side=side,
            score=score,
            exhaustion=exhaustion,
            continuation=continuation,
            justification=f"squeeze_score {score:.2f} below min {p.squeeze_min_score:.2f}",
        )

    # 2) Exhaustion vs continuation
    if exhaustion > p.exhaustion_enter_thr and exhaustion > continuation:
        action = Action.ENTER
        style = Style.LIMIT_PULLBACK
        justification = f"exhaustion {exhaustion:.2f} > continuation {continuation:.2f}"
    elif continuation > p.continuation_follow_thr:
        action = Action.ENTER
        style = Style.MARKET_CONTINUATION
        justification = f"continuation {continuation:.2f} strong"
    else:
        action = Action.WAIT
        style = Style.LIMIT_PULLBACK
        justification = "ambiguous – waiting"

    return Decision(
        action=action,
        style=style,
        side=side,
        score=score,
        exhaustion=exhaustion,
        continuation=continuation,
        justification=justification,
    )


# --- Risk plan utils --------------------------------------------------------


@dataclass(slots=True)
class PullbackPlanLeg:
    price: float
    size_frac: float


def build_pullback_plan(f: SqueezeFeatures, p: SqueezeParams) -> list[PullbackPlanLeg]:
    """Return 3‑leg limit ladder based on impulse hi/lo."""
    rng = max(f.hi_5m - f.lo_5m, 1e-9)
    legs = []
    if f.impulse_dir == "up":
        base = f.hi_5m
        signs = (-1, -1, -1)
    else:
        base = f.lo_5m
        signs = (1, 1, 1)

    for lvl, sign in zip(p.fib_levels, signs):
        price = base + sign * lvl * rng
        legs.append(PullbackPlanLeg(price=price, size_frac=1 / len(p.fib_levels)))
    return legs
