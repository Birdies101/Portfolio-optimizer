"""Mean-variance portfolio optimization.

Minimum variance portfolio, tangency (max Sharpe) portfolio and the efficient
frontier, solved as convex programs with cvxpy. Pure functions only: no
Streamlit, no network, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}

# Tiny diagonal ridge so near-duplicate assets (e.g. SPY + VOO) keep the
# covariance matrix numerically positive definite.
_RIDGE = 1e-10
_OK = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


class InfeasibleConstraintsError(ValueError):
    """The weight constraints admit no fully invested portfolio."""


@dataclass(frozen=True)
class Constraints:
    """Per-asset weight bounds. Long-only: [0, max_weight]; shorting: [-max_weight, max_weight]."""

    allow_short: bool = False
    max_weight: float = 1.0

    @property
    def lower(self) -> float:
        return -self.max_weight if self.allow_short else 0.0

    @property
    def upper(self) -> float:
        return self.max_weight


@dataclass
class Portfolio:
    label: str
    weights: pd.Series
    ret: float
    vol: float
    sharpe: float
    is_fallback: bool = False


@dataclass
class OptimizationResult:
    mvp: Portfolio
    tangency: Portfolio
    frontier: pd.DataFrame  # columns: ret, vol, sharpe, then one weight column per asset
    rf: float
    checks: dict[str, bool]
    warnings: list[str] = field(default_factory=list)


def estimate_moments(
    returns: pd.DataFrame, periods_per_year: int, estimator: str = "ledoit_wolf"
) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized mean returns and covariance from periodic simple returns.

    mean_annual = mean_periodic * N, cov_annual = cov_periodic * N.
    estimator: "ledoit_wolf" (shrinkage) or "sample".
    """
    mu = returns.mean() * periods_per_year
    if estimator == "ledoit_wolf":
        periodic = LedoitWolf().fit(returns.to_numpy()).covariance_
    elif estimator == "sample":
        periodic = returns.cov().to_numpy()
    else:
        raise ValueError(f"Unknown covariance estimator: {estimator!r}")
    cov = pd.DataFrame(periodic * periods_per_year, index=returns.columns, columns=returns.columns)
    return mu, cov


def apply_views(mu: pd.Series, views: dict[str, tuple[float, float]]) -> pd.Series:
    """Blend estimated returns with user views: μᵢ + confidenceᵢ·(viewᵢ − μᵢ).

    A Black-Litterman-lite: views maps ticker -> (annual expected return, confidence in [0, 1]).
    Confidence 1 replaces the estimate, 0 ignores the view. Unknown tickers are ignored.
    """
    out = mu.astype(float).copy()
    for t, (view, conf) in views.items():
        if t in out.index:
            out[t] += min(max(conf, 0.0), 1.0) * (view - out[t])
    return out


def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    """(expected return, volatility, Sharpe ratio) of weights w."""
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


def asset_summary(mu: pd.Series, cov: pd.DataFrame, rf: float) -> pd.DataFrame:
    """Per-asset annualized return, volatility and Sharpe ratio."""
    vol = pd.Series(np.sqrt(np.diag(cov)), index=mu.index)
    return pd.DataFrame({"ret": mu, "vol": vol, "sharpe": (mu - rf) / vol})


def correlation(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def near_duplicates(corr: pd.DataFrame, threshold: float = 0.99) -> list[tuple[str, str]]:
    """Asset pairs whose correlation exceeds threshold.

    Pass the raw return correlation: shrinkage estimators pull correlations down and can hide twins.
    """
    names, c = list(corr.columns), corr.to_numpy()
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names)) if c[i, j] > threshold]


def check_feasible(n_assets: int, cons: Constraints) -> None:
    """Raise if no weights within bounds can sum to 1."""
    if cons.max_weight <= 0:
        raise InfeasibleConstraintsError("Max weight per asset must be positive.")
    if n_assets * cons.upper < 1 - 1e-9:
        raise InfeasibleConstraintsError(
            f"With {n_assets} assets and a max weight of {cons.upper:.0%}, the weights can add up to at most "
            f"{n_assets * cons.upper:.0%}, so a fully invested portfolio is impossible. "
            f"Raise the max weight to at least {1 / n_assets:.0%} or add assets."
        )


def _psd(cov: np.ndarray) -> cp.Constant:
    sym = (cov + cov.T) / 2 + _RIDGE * np.eye(len(cov))
    return cp.psd_wrap(sym)


def _polish(w: np.ndarray, cons: Constraints) -> np.ndarray:
    """Remove solver noise: zero out dust, clip to bounds, renormalize to sum to 1."""
    w = np.clip(np.asarray(w, dtype=float).ravel(), cons.lower, cons.upper)
    w[np.abs(w) < 1e-8] = 0.0
    return w / w.sum()


def _weight_constraints(w: cp.Variable, cons: Constraints) -> list:
    return [cp.sum(w) == 1, w >= cons.lower, w <= cons.upper]


def min_variance_weights(cov: np.ndarray, cons: Constraints) -> np.ndarray:
    """Minimize w'Σw subject to sum(w) = 1 and the weight bounds."""
    n = len(cov)
    check_feasible(n, cons)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, _psd(cov))), _weight_constraints(w, cons))
    prob.solve()
    if prob.status not in _OK:
        raise InfeasibleConstraintsError(f"Minimum variance problem could not be solved ({prob.status}).")
    return _polish(w.value, cons)


def max_return(mu: np.ndarray, cons: Constraints) -> float:
    """Highest expected return reachable under the weight bounds (an LP)."""
    w = cp.Variable(len(mu))
    prob = cp.Problem(cp.Maximize(mu @ w), _weight_constraints(w, cons))
    prob.solve()
    if prob.status not in _OK:
        raise InfeasibleConstraintsError(f"Maximum return problem could not be solved ({prob.status}).")
    return float(prob.value)


def tangency_weights(mu: np.ndarray, cov: np.ndarray, rf: float, cons: Constraints) -> np.ndarray | None:
    """Max Sharpe weights via the standard convex reformulation, or None if undefined.

    With y = κw (κ > 0): minimize y'Σy  s.t.  (μ - rf)'y = 1,  sum(y) = κ,
    lower·κ <= y <= upper·κ. Then w = y / κ. Infeasible exactly when no
    allowed portfolio has a positive excess return.
    """
    excess = mu - rf
    if not cons.allow_short and excess.max() <= 0:
        return None
    n = len(mu)
    y = cp.Variable(n)
    k = cp.Variable(nonneg=True)
    constraints = [excess @ y == 1, cp.sum(y) == k, y >= cons.lower * k, y <= cons.upper * k]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, _psd(cov))), constraints)
    prob.solve()
    if prob.status not in _OK or k.value is None or k.value < 1e-9:
        return None
    return _polish(np.asarray(y.value).ravel() / float(k.value), cons)


def max_return_at_vol(mu: pd.Series, cov: pd.DataFrame, target_vol: float, cons: Constraints) -> np.ndarray | None:
    """Frontier portfolio for a target volatility: maximize μ'w s.t. √(w'Σw) <= target_vol.

    Returns None if target_vol is below the minimum achievable volatility.
    """
    mu_v, cov_v = mu.to_numpy(dtype=float), cov.loc[mu.index, mu.index].to_numpy(dtype=float)
    lt = np.linalg.cholesky((cov_v + cov_v.T) / 2 + _RIDGE * np.eye(len(mu_v))).T
    w = cp.Variable(len(mu_v))
    # Tiny slack so a target exactly at the MVP's volatility stays feasible numerically.
    prob = cp.Problem(
        cp.Maximize(mu_v @ w), _weight_constraints(w, cons) + [cp.norm(lt @ w) <= target_vol * (1 + 1e-6)]
    )
    try:
        prob.solve()
    except cp.error.SolverError:
        return None
    if prob.status not in _OK or w.value is None:
        return None
    return _polish(w.value, cons)


def cml_allocation(tangency: Portfolio, rf: float, target_vol: float) -> tuple[float, float, float]:
    """Mix of tangency portfolio and cash on the Capital Market Line for a target volatility.

    Returns (weight in tangency portfolio, weight in cash, expected return). Cash below 0 means
    borrowing at the risk-free rate to hold more than 100% of the tangency portfolio.
    """
    a = target_vol / tangency.vol
    return a, 1.0 - a, rf + a * (tangency.ret - rf)


def bootstrap_tangency(
    returns: pd.DataFrame,
    periods_per_year: int,
    rf: float,
    cons: Constraints,
    estimator: str = "ledoit_wolf",
    views: dict[str, tuple[float, float]] | None = None,
    n_samples: int = 200,
    seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """Tangency weights re-estimated on bootstrap resamples of the return history.

    Each sample draws len(returns) rows with replacement (iid bootstrap), re-estimates μ and Σ,
    applies the same views, and re-solves the tangency problem. Returns (one row of weights per
    successful sample, number of samples where the tangency portfolio was undefined).
    """
    assets = list(returns.columns)
    n, x = len(assets), returns.to_numpy()
    check_feasible(n, cons)
    rng = np.random.default_rng(seed)

    # Build the problem once with parameters; re-solving only swaps in the data.
    lt, excess = cp.Parameter((n, n)), cp.Parameter(n)
    y, k = cp.Variable(n), cp.Variable(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(lt @ y)),
        [excess @ y == 1, cp.sum(y) == k, y >= cons.lower * k, y <= cons.upper * k],
    )
    rows, failed = [], 0
    for _ in range(n_samples):
        sample = pd.DataFrame(x[rng.integers(0, len(x), len(x))], columns=assets)
        mu, cov = estimate_moments(sample, periods_per_year, estimator)
        if views:
            mu = apply_views(mu, views)
        ex = mu.to_numpy() - rf
        if not cons.allow_short and ex.max() <= 0:
            failed += 1
            continue
        try:
            lt.value = np.linalg.cholesky(cov.to_numpy() + 1e-12 * np.eye(n)).T
            excess.value = ex
            prob.solve()
        except (np.linalg.LinAlgError, cp.error.SolverError):
            failed += 1
            continue
        if prob.status in _OK and k.value is not None and k.value > 1e-9:
            rows.append(_polish(np.asarray(y.value).ravel() / float(k.value), cons))
        else:
            failed += 1
    return pd.DataFrame(rows, columns=assets), failed


def efficient_frontier(
    mu: np.ndarray, cov: np.ndarray, cons: Constraints, ret_min: float, n_points: int = 100
) -> list[np.ndarray]:
    """Min-variance weights for target returns from ret_min (the MVP) up to the max achievable.

    Uses mu'w >= target, which binds on the efficient (upper) branch and is more
    robust than an equality at the endpoints. Infeasible targets are skipped.
    """
    ret_max = max_return(mu, cons)
    if ret_max - ret_min < 1e-10:
        return []
    n = len(mu)
    w = cp.Variable(n)
    target = cp.Parameter()
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, _psd(cov))),
        _weight_constraints(w, cons) + [mu @ w >= target],
    )
    out = []
    for t in np.linspace(ret_min, ret_max, n_points):
        target.value = t
        try:
            prob.solve(warm_start=True)
        except cp.error.SolverError:
            continue
        if prob.status in _OK and w.value is not None:
            out.append(_polish(w.value, cons))
    return out


def optimize(
    mu: pd.Series, cov: pd.DataFrame, rf: float, cons: Constraints, n_points: int = 100
) -> OptimizationResult:
    """Compute the MVP, tangency portfolio and efficient frontier, plus sanity checks."""
    assets = list(mu.index)
    mu_v, cov_v = mu.to_numpy(dtype=float), cov.loc[assets, assets].to_numpy(dtype=float)
    warnings: list[str] = []

    eig = np.linalg.eigvalsh((cov_v + cov_v.T) / 2)
    if eig.min() <= 1e-10 * eig.max():
        warnings.append(
            "The covariance matrix is singular: there are too few return observations for this many "
            "assets (or two assets are exact copies). The optimizer can then build combinations that look "
            "almost riskless, so the minimum variance volatility and the tangency Sharpe ratio are not "
            "reliable. Switch to Ledoit-Wolf shrinkage, lengthen the lookback, or use a higher frequency."
        )

    def make(label: str, w: np.ndarray, fallback: bool = False) -> Portfolio:
        r, v, s = portfolio_stats(w, mu_v, cov_v, rf)
        return Portfolio(label, pd.Series(w, index=assets), r, v, s, fallback)

    mvp = make("Minimum Variance", min_variance_weights(cov_v, cons))

    frontier_w = efficient_frontier(mu_v, cov_v, cons, mvp.ret, n_points)
    rows = [(*portfolio_stats(w, mu_v, cov_v, rf), *w) for w in frontier_w]
    frontier = pd.DataFrame(rows, columns=["ret", "vol", "sharpe", *assets])
    if frontier.empty:
        frontier = pd.DataFrame([[mvp.ret, mvp.vol, mvp.sharpe, *mvp.weights]], columns=frontier.columns)

    tan_w = tangency_weights(mu_v, cov_v, rf, cons)
    if tan_w is not None:
        tangency = make("Tangency (Max Sharpe)", tan_w)
    else:
        best = frontier["sharpe"].idxmax()
        tangency = make("Max Sharpe (fallback)", frontier.loc[best, assets].to_numpy(dtype=float), fallback=True)
        warnings.append(
            f"No allowed portfolio earns more than the risk-free rate ({rf:.2%}), so the tangency portfolio "
            "is undefined: the Capital Market Line would slope downward. Showing the frontier portfolio with "
            "the highest (least negative) Sharpe ratio instead. Treat it with caution."
        )

    tol = 1e-4
    checks = {
        "Weights sum to 100%": bool(
            abs(mvp.weights.sum() - 1) < 1e-6 and abs(tangency.weights.sum() - 1) < 1e-6
        ),
        "Weights within bounds": bool(
            all(
                (p.weights >= cons.lower - 1e-6).all() and (p.weights <= cons.upper + 1e-6).all()
                for p in (mvp, tangency)
            )
        ),
        "MVP has the lowest volatility on the frontier": bool(mvp.vol <= frontier["vol"].min() + tol),
        "Tangency has the highest Sharpe on the frontier": bool(
            tangency.sharpe >= frontier["sharpe"].max() - tol
        ),
    }
    return OptimizationResult(mvp, tangency, frontier, rf, checks, warnings)
