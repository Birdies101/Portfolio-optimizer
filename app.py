"""Streamlit UI: minimum variance and tangency portfolios for user-chosen tickers."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import data
import optimizer as opt
import plots

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

LOOKBACKS = {"1 year": 1, "3 years": 3, "5 years": 5, "10 years": 10, "Custom dates": None}
DEFAULT_TICKERS = "AAPL, MSFT, JNJ, XOM, TLT, AGG, GLD"
N_BOOTSTRAP = 200
BOND_HELP = (
    "Separate tickers with commas, spaces or new lines. Any Yahoo Finance symbol works: stocks, ETFs, "
    "mutual funds, crypto (e.g. BTC-USD).\n\nBonds: free data sources don't carry price histories for "
    "individual bonds, so use bond ETFs as proxies, e.g. AGG/BND (aggregate), SHY/IEF/TLT "
    "(short/intermediate/long Treasuries), LQD (investment-grade corporate), HYG (high yield), "
    "TIP (inflation-protected), MUB (municipal)."
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return data.download_prices(list(tickers), start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_rf() -> float | None:
    return data.fetch_risk_free_rate()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_optimize(mu: pd.Series, cov: pd.DataFrame, rf: float, allow_short: bool, max_weight: float):
    return opt.optimize(mu, cov, rf, opt.Constraints(allow_short=allow_short, max_weight=max_weight))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_bootstrap(
    returns: pd.DataFrame, periods: int, rf: float, allow_short: bool, max_weight: float,
    estimator: str, views: tuple[tuple[str, tuple[float, float]], ...],
):
    return opt.bootstrap_tangency(
        returns, periods, rf, opt.Constraints(allow_short, max_weight), estimator, dict(views), N_BOOTSTRAP
    )


def pct_table(df: pd.DataFrame, pct_cols: list[str]) -> None:
    """Render a dataframe with selected columns shown as percentages."""
    shown = df.copy()
    shown[pct_cols] = shown[pct_cols] * 100
    config = {c: st.column_config.NumberColumn(c, format="%.2f%%") for c in pct_cols}
    config.update({c: st.column_config.NumberColumn(c, format="%.2f") for c in df.columns if c not in pct_cols})
    st.dataframe(shown, column_config=config, width="stretch")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Inputs")
    if "tickers_text" not in st.session_state:  # first load: take tickers from a shared/bookmarked URL
        shared = data.parse_tickers(st.query_params.get("tickers", ""))
        st.session_state.tickers_text = ", ".join(shared) if shared else DEFAULT_TICKERS
    tickers_text = st.text_area(
        "Tickers", key="tickers_text", height=110, help=BOND_HELP, placeholder="AAPL, MSFT, TLT, AGG, GLD",
    )
    st.caption(
        "Press Ctrl+Enter or click outside the box to update. The page URL saves your tickers, "
        "so you can bookmark or share it."
    )

    lookback = st.selectbox("Lookback period", list(LOOKBACKS), index=2)
    today = date.today()
    if LOOKBACKS[lookback] is None:
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", today - timedelta(days=5 * 365), max_value=today)
        end = c2.date_input("End", today, max_value=today)
    else:
        end = today
        start = today - timedelta(days=round(LOOKBACKS[lookback] * 365.25))

    freq = st.selectbox(
        "Return frequency", list(opt.PERIODS_PER_YEAR), index=0, format_func=str.capitalize,
        help="Daily uses the most data; weekly/monthly reduce noise from non-synchronous trading.",
    )

    st.subheader("Risk-free rate")
    rf_mode = st.radio("Source", ["Auto: 3-month T-bill (^IRX)", "Manual"], label_visibility="collapsed")
    manual_rf = None
    if rf_mode == "Manual":
        manual_rf = st.number_input("Annual risk-free rate (%)", value=4.0, step=0.25, format="%.2f") / 100

    st.subheader("Constraints")
    allow_short = st.toggle("Allow short selling", value=False, help="Off = long-only (no negative weights).")
    max_weight = st.slider(
        "Max weight per asset", 5, 100, 100, step=5, format="%d%%",
        help="Caps each position. With shorting on, also caps each short at the same size.",
    ) / 100

    estimator_label = st.radio(
        "Covariance estimator", ["Ledoit-Wolf shrinkage", "Sample covariance"],
        help="Shrinkage pulls the covariance toward a structured target, giving more stable weights.",
    )
    estimator = "ledoit_wolf" if estimator_label.startswith("Ledoit") else "sample"

# ---------------------------------------------------------------- pipeline
st.title("Portfolio Optimizer")
st.caption("Minimum variance and maximum Sharpe (tangency) portfolios on the mean-variance efficient frontier.")

tickers = data.parse_tickers(tickers_text)
if tickers:
    st.query_params["tickers"] = ",".join(tickers)
elif "tickers" in st.query_params:
    del st.query_params["tickers"]
if len(tickers) < 2:
    st.info("Enter at least two tickers in the sidebar to get started.")
    st.stop()
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

with st.spinner(f"Downloading prices for {len(tickers)} tickers…"):
    try:
        raw = cached_prices(tuple(tickers), start, end)
    except data.DataError as exc:
        st.error(str(exc))
        st.stop()

clean = data.clean_prices(raw, start, freq)
for t, reason in clean.dropped.items():
    st.warning(f"Skipped **{t}**: {reason}.")
for w in clean.warnings:
    st.warning(w)
if clean.returns.shape[1] < 2 or len(clean.returns) < 3:
    st.error("At least two tickers with overlapping price history are needed. Check the tickers above.")
    st.stop()

if manual_rf is not None:
    rf = manual_rf
else:
    fetched = cached_rf()
    if fetched is None:
        rf = data.DEFAULT_RF
        st.warning(f"Couldn't fetch the T-bill rate (^IRX); using a default of {rf:.2%}. Set it manually in the sidebar.")
    else:
        rf = fetched

periods = opt.PERIODS_PER_YEAR[freq]
hist_mu, cov = opt.estimate_moments(clean.returns, periods, estimator)
cons = opt.Constraints(allow_short=allow_short, max_weight=max_weight)

# ---------------------------------------------------------------- views (Black-Litterman-lite)
with st.expander("Your return views (Black-Litterman-lite)"):
    st.caption(
        "Expected returns start at their historical averages. Type your own annual return for any asset. "
        "Confidence blends it with history: 100% uses your view, 50% goes halfway, 0% ignores it."
    )
    editor_key = "views_" + "_".join(hist_mu.index)  # fresh table whenever the asset list changes
    base = pd.DataFrame(
        {"Historical": hist_mu * 100, "Your view": hist_mu * 100, "Confidence": 100.0}, index=hist_mu.index
    )
    base.index.name = "Ticker"
    edited = st.data_editor(
        base, key=editor_key, disabled=["Historical"], width="stretch",
        column_config={
            "Historical": st.column_config.NumberColumn("Historical", format="%.2f%%"),
            "Your view": st.column_config.NumberColumn("Your view", format="%.2f%%", step=0.01),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%", min_value=0, max_value=100, step=10),
        },
    )
    if st.button("Reset to historical"):
        st.session_state.pop(editor_key, None)
        st.rerun()
views = {
    t: (row["Your view"] / 100, row["Confidence"] / 100)
    for t, row in edited.iterrows()
    if pd.notna(row["Your view"]) and pd.notna(row["Confidence"]) and abs(row["Your view"] - row["Historical"]) > 1e-9
}
mu = opt.apply_views(hist_mu, views)

try:
    res = cached_optimize(mu, cov, rf, allow_short, max_weight)
except opt.InfeasibleConstraintsError as exc:
    st.error(str(exc))
    st.stop()
for w in res.warnings:
    st.warning(w)
twins = opt.near_duplicates(clean.returns.corr())
if twins:
    st.warning(
        f"Near-identical assets (return correlation above 0.99): {', '.join(f'{a} & {b}' for a, b in twins)}. "
        "They behave like one holding, so how the weight is split between them is arbitrary. "
        "Consider keeping just one of each pair."
    )

st.caption(
    f"Data: **{clean.start:%b %d, %Y} – {clean.end:%b %d, %Y}** · {len(clean.returns):,} {freq} returns · "
    f"{len(mu)} assets · risk-free {rf:.2%} · {'shorting allowed' if allow_short else 'long-only'}, "
    f"max {max_weight:.0%} per asset · {estimator_label.lower()}"
)
if views:
    st.info(f"Using your return views for **{', '.join(views)}**. Everything below reflects them.")

# ---------------------------------------------------------------- target volatility
with st.sidebar:
    st.subheader("Target volatility")
    modes = ["Frontier: risky assets only"]
    if not res.tangency.is_fallback:
        modes.append("CML: tangency portfolio + cash")
    tv_mode = st.radio(
        "Read off", modes,
        help="Frontier: the best risky-only portfolio at this volatility. CML: mix the tangency portfolio "
        "with cash (or borrow at the risk-free rate to lever up), which beats the frontier everywhere "
        "except at the tangency point.",
    )
    use_cml = tv_mode.startswith("CML")
    hi = math.floor(max(res.frontier["vol"].max(), res.tangency.vol) * 1000) / 10
    lo = 0.0 if use_cml else min(math.ceil(res.mvp.vol * 1000) / 10, hi)
    if hi - lo < 0.1:
        target_vol = None
        st.caption("The frontier is a single point here, so there's nothing to choose.")
    else:
        default = min(max(round(res.tangency.vol * 100, 1), lo), hi)
        target_vol = st.slider("Annual volatility", lo, hi, default, step=0.1, format="%.1f%%") / 100

target = None
if target_vol is not None:
    if use_cml:
        scale, cash, t_ret = opt.cml_allocation(res.tangency, rf, target_vol)
        t_weights = pd.concat([res.tangency.weights * scale, pd.Series({"Cash (risk-free)": cash})])
        t_vol = target_vol
    else:
        cash = 0.0
        w = opt.max_return_at_vol(mu, cov, max(target_vol, res.mvp.vol), cons)
        t_weights = pd.Series(w, index=mu.index) if w is not None else None
        if w is not None:
            t_ret, t_vol, _ = opt.portfolio_stats(w, mu.to_numpy(), cov.to_numpy(), rf)
    if t_weights is not None:
        target = (t_vol, t_ret)

# ---------------------------------------------------------------- results
assets = opt.asset_summary(mu, cov, rf)
if views:
    assets.insert(1, "hist", hist_mu)
c1, c2 = st.columns(2)
for col, p, color in ((c1, res.mvp, plots.MVP_COLOR), (c2, res.tangency, plots.TAN_COLOR)):
    with col:
        st.subheader(p.label)
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected return", f"{p.ret:.2%}")
        m2.metric("Volatility", f"{p.vol:.2%}")
        m3.metric("Sharpe ratio", f"{p.sharpe:.2f}")

st.plotly_chart(plots.frontier_chart(res, assets, target), width="stretch")

st.subheader("Portfolio weights")
c1, c2 = st.columns(2)
for col, p, color in ((c1, res.mvp, plots.MVP_COLOR), (c2, res.tangency, plots.TAN_COLOR)):
    with col:
        st.markdown(f"**{p.label}**")
        st.plotly_chart(plots.weights_chart(p.weights, color), width="stretch")

weight_cols = {res.mvp.label: res.mvp.weights, res.tangency.label: res.tangency.weights}
if target is not None:
    t_label = f"Target {target_vol:.1%} vol ({'CML' if use_cml else 'frontier'})"
    weight_cols[t_label] = t_weights
    st.subheader(f"Target volatility portfolio: {target_vol:.1%}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected return", f"{t_ret:.2%}")
    m2.metric("Volatility", f"{t_vol:.2%}")
    m3.metric("Sharpe ratio", f"{(t_ret - rf) / t_vol:.2f}" if t_vol > 0 else "n/a")
    m4.metric("Cash (risk-free)" if cash >= 0 else "Borrowed at risk-free", f"{abs(cash):.1%}")
    if use_cml and cash < 0:
        st.caption(
            f"Above the tangency portfolio's {res.tangency.vol:.1%} volatility, the CML borrows {-cash:.1%} at the "
            "risk-free rate to hold more than 100% in the tangency portfolio. Real borrowing costs more, so this is optimistic."
        )
    elif not use_cml and target_vol > t_vol + 1e-4:
        st.caption(f"The riskiest frontier portfolio only reaches {t_vol:.1%} volatility under these constraints.")
    st.plotly_chart(plots.weights_chart(t_weights, plots.TARGET_COLOR), width="stretch")

weights = pd.DataFrame(weight_cols).fillna(0.0)
weights.index.name = "Ticker"
pct_table(weights, list(weights.columns))

summary = pd.DataFrame(
    {p.label: {"Expected return": p.ret, "Volatility": p.vol, "Sharpe ratio": p.sharpe} for p in (res.mvp, res.tangency)}
).T
pct_table(summary, ["Expected return", "Volatility"])

tab_assets, tab_corr, tab_frontier, tab_checks = st.tabs(
    ["Asset summary", "Correlations", "Frontier data", "Sanity checks"]
)
with tab_assets:
    shown = assets.rename(columns={
        "ret": "Annual return used" if views else "Annual return", "hist": "Historical return",
        "vol": "Annual volatility", "sharpe": "Sharpe ratio",
    })
    pct_table(shown, [c for c in shown.columns if c != "Sharpe ratio"])
    st.caption("Historical averages. Past returns are a noisy guide to future returns.")
with tab_corr:
    st.plotly_chart(plots.correlation_heatmap(opt.correlation(cov)), width="stretch")
with tab_frontier:
    frontier = res.frontier.rename(columns={"ret": "Return", "vol": "Volatility", "sharpe": "Sharpe"})
    pct_table(frontier, [c for c in frontier.columns if c != "Sharpe"])
with tab_checks:
    for name, ok in res.checks.items():
        st.markdown(f"{'✅' if ok else '❌'} {name}")

# ---------------------------------------------------------------- bootstrap stability
st.subheader("How stable are the tangency weights?")
if res.tangency.is_fallback:
    st.info("The tangency portfolio is undefined for these inputs, so there's nothing to resample.")
elif st.toggle(
    f"Run {N_BOOTSTRAP} bootstrap resamples",
    help="Re-draws the return history with replacement, re-estimates returns and covariance (keeping your "
    "views), and re-solves the tangency portfolio each time. Wide boxes mean the weights are mostly noise.",
):
    with st.spinner(f"Re-optimizing {N_BOOTSTRAP} resampled histories…"):
        samples, failed = cached_bootstrap(
            clean.returns, periods, rf, allow_short, max_weight, estimator, tuple(sorted(views.items()))
        )
    if samples.empty:
        st.warning("The tangency portfolio was undefined in every resample.")
    else:
        st.plotly_chart(plots.bootstrap_box(samples, res.tangency.weights), width="stretch")
        stats = pd.DataFrame({
            "Full sample": res.tangency.weights,
            "Median": samples.median(),
            "5th percentile": samples.quantile(0.05),
            "95th percentile": samples.quantile(0.95),
            "Held in % of samples": (samples.abs() > 0.01).mean(),
        }).sort_values("Full sample", ascending=False)
        stats.index.name = "Ticker"
        pct_table(stats, list(stats.columns))
        note = f" In {failed} resample(s) no portfolio beat the risk-free rate; those are left out." if failed else ""
        st.caption(
            "Boxes show the middle 50% of weights across resamples; whiskers the typical range. "
            "'Held' counts samples where the weight is above 1% in size." + note
        )

d1, d2 = st.columns(2)
d1.download_button("Download weights (CSV)", weights.to_csv().encode(), "portfolio_weights.csv", "text/csv")
d2.download_button("Download frontier (CSV)", res.frontier.to_csv(index=False).encode(), "efficient_frontier.csv", "text/csv")

with st.expander("Method and caveats"):
    st.markdown(
        f"""
- **Returns:** simple returns from split- and dividend-adjusted prices, annualized as mean × {opt.PERIODS_PER_YEAR[freq]}
  and covariance × {opt.PERIODS_PER_YEAR[freq]}.
- **Minimum variance:** minimizes portfolio variance subject to fully invested weights and the bounds.
- **Tangency:** maximizes (return − risk-free) / volatility, solved exactly as a convex problem.
  The Capital Market Line runs from the risk-free rate through it.
- **Your views:** each asset's expected return becomes historical + confidence × (view − historical).
  This is a simplified Black-Litterman: no equilibrium prior and no relative views.
- **Target volatility:** the frontier option maximizes return subject to volatility ≤ target. The CML option
  holds (target ÷ tangency volatility) in the tangency portfolio and the rest in cash, or borrows above 100%.
- **Bootstrap:** {N_BOOTSTRAP} iid resamples of the return history with the same estimator, views and constraints.
  Resampling single periods ignores autocorrelation and volatility clustering, so the true uncertainty is likely larger.
- **Caveat:** expected returns from history are very noisy, so the tangency weights can swing a lot with
  small changes in the window. The minimum variance portfolio is more stable because it ignores expected returns.
"""
    )
