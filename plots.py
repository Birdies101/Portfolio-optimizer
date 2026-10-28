"""Plotly figures for the app."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from optimizer import OptimizationResult

MVP_COLOR = "#2a9d8f"
TAN_COLOR = "#e76f51"
FRONTIER_COLOR = "#264653"
ASSET_COLOR = "#8d99ae"
TARGET_COLOR = "#7b5ea7"


def frontier_chart(
    res: OptimizationResult, assets: pd.DataFrame, target: tuple[float, float] | None = None
) -> go.Figure:
    """Efficient frontier, individual assets, MVP, tangency portfolio and Capital Market Line.

    target: optional (volatility, return) of the target-volatility portfolio to mark.
    """
    f, mvp, tan = res.frontier, res.mvp, res.tangency
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=f["vol"], y=f["ret"], mode="lines", name="Efficient frontier",
            line=dict(color=FRONTIER_COLOR, width=3),
            hovertemplate="Vol %{x:.2%}<br>Return %{y:.2%}<extra>Frontier</extra>",
        )
    )

    if not tan.is_fallback:
        x_end = max(f["vol"].max(), assets["vol"].max(), tan.vol) * 1.1
        fig.add_trace(
            go.Scatter(
                x=[0, x_end], y=[res.rf, res.rf + tan.sharpe * x_end], mode="lines",
                name="Capital Market Line", line=dict(color=TAN_COLOR, dash="dash", width=2),
                hovertemplate="Vol %{x:.2%}<br>Return %{y:.2%}<extra>CML</extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=assets["vol"], y=assets["ret"], mode="markers+text", name="Assets",
            text=assets.index, textposition="top center",
            marker=dict(color=ASSET_COLOR, size=10, line=dict(color="white", width=1)),
            customdata=assets["sharpe"],
            hovertemplate="<b>%{text}</b><br>Vol %{x:.2%}<br>Return %{y:.2%}<br>Sharpe %{customdata:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0], y=[res.rf], mode="markers", name=f"Risk-free ({res.rf:.2%})",
            marker=dict(color=TAN_COLOR, size=8, symbol="circle-open", line=dict(width=2)),
            hovertemplate="Risk-free rate %{y:.2%}<extra></extra>",
        )
    )
    for p, color in ((mvp, MVP_COLOR), (tan, TAN_COLOR)):
        fig.add_trace(
            go.Scatter(
                x=[p.vol], y=[p.ret], mode="markers", name=p.label,
                marker=dict(color=color, size=18, symbol="star", line=dict(color="white", width=1)),
                hovertemplate=f"<b>{p.label}</b><br>Vol %{{x:.2%}}<br>Return %{{y:.2%}}<br>Sharpe {p.sharpe:.2f}<extra></extra>",
            )
        )

    if target is not None:
        fig.add_trace(
            go.Scatter(
                x=[target[0]], y=[target[1]], mode="markers", name="Target volatility",
                marker=dict(color=TARGET_COLOR, size=14, symbol="diamond", line=dict(color="white", width=1)),
                hovertemplate="<b>Target</b><br>Vol %{x:.2%}<br>Return %{y:.2%}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Annualized volatility", yaxis_title="Annualized expected return",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
    )
    fig.update_xaxes(rangemode="tozero")
    return fig


def weights_chart(weights: pd.Series, color: str) -> go.Figure:
    """Horizontal bar chart of weights (handles negative/short weights)."""
    w = weights[weights.abs() > 1e-4].sort_values()
    fig = go.Figure(
        go.Bar(
            x=w.values, y=w.index, orientation="h", marker_color=color,
            text=[f"{v:.1%}" for v in w.values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    lo, hi = min(0.0, w.min()), max(0.0, w.max())
    pad = 0.2 * (hi - lo)  # room for the outside labels
    fig.update_layout(
        xaxis_range=[lo - pad if lo < 0 else 0, hi + pad],
        xaxis_tickformat=".0%", height=max(220, 34 * len(w) + 60),
        margin=dict(l=10, r=40, t=10, b=10), showlegend=False,
    )
    return fig


def bootstrap_box(samples: pd.DataFrame, point: pd.Series) -> go.Figure:
    """Box plot of bootstrap tangency weights per asset, with the full-sample weight marked."""
    order = point.sort_values(ascending=False).index
    fig = go.Figure()
    for t in order:
        fig.add_trace(
            go.Box(
                y=samples[t], name=t, marker_color=TAN_COLOR, boxpoints="outliers", showlegend=False,
                hovertemplate=f"{t}: %{{y:.1%}}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(order), y=point[order], mode="markers", name="Full-sample weight",
            marker=dict(color=FRONTIER_COLOR, size=11, symbol="x"),
            hovertemplate="%{x} full-sample: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title="Tangency weight", yaxis_tickformat=".0%", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(height=max(320, 40 * len(corr) + 120), margin=dict(l=10, r=10, t=10, b=10))
    return fig
