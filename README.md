# Portfolio Optimizer

Type in some tickers and get the **minimum variance portfolio**, the **tangency (maximum Sharpe) portfolio**, the efficient frontier and the Capital Market Line, all computed automatically from Yahoo Finance price history.

## Setup and run

Requires Python 3.11–3.13 (3.12 recommended; some numerical dependencies may lack wheels for newer Pythons).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The first load takes roughly 20–30 seconds (library imports and the price download). After that, downloads are cached for an hour and changes recompute in about a second.

Run the tests:

```bash
python -m pytest
```

No API keys or paid services are needed.

## Using it

- **Tickers:** separate them with commas, spaces or new lines. Case and duplicates don't matter. Any Yahoo Finance symbol works (stocks, ETFs, mutual funds, crypto such as `BTC-USD`).
- **Bonds:** free data sources don't have price histories for individual bonds, so use bond ETFs as proxies: `AGG`/`BND` (aggregate), `SHY`/`IEF`/`TLT` (short/intermediate/long Treasuries), `LQD` (investment-grade corporate), `HYG` (high yield), `TIP` (inflation-protected), `MUB` (municipal).
- **Sidebar options:** lookback (1/3/5/10 years or custom), return frequency, risk-free rate (auto or manual), short selling, max weight per asset, and covariance estimator.
- **Outputs:** frontier chart with the CML, metrics, weight charts and tables, per-asset statistics, a correlation heatmap, the frontier data, sanity checks, and CSV downloads.
- **Your return views (Black-Litterman-lite):** an expandable table above the results lists each asset's historical return. Type your own annual return for any asset and set a confidence: the return used is `historical + confidence × (view − historical)`. "Reset to historical" clears your views. Views affect the tangency portfolio, the frontier, the target-volatility portfolio and the bootstrap. They don't affect the minimum variance portfolio, which ignores expected returns.
- **Target volatility (sidebar):** pick an annual volatility and read off the allocation.
  - *Frontier:* the highest-return risky-only portfolio at that volatility (maximize `μᵀw` subject to `√(wᵀΣw) ≤ target`).
  - *CML:* hold `target ÷ σ_tangency` in the tangency portfolio and the rest in cash. Above the tangency volatility, the cash weight turns negative, meaning borrowing at the risk-free rate.
  - The portfolio is marked on the chart with a purple diamond and added to the weights CSV.
- **Weight stability (bootstrap):** turn on "Run 200 bootstrap resamples" to re-draw the return history with replacement 200 times. Each time it re-estimates returns and covariance with the same estimator, views and constraints, and re-solves the tangency portfolio. A box plot per asset shows how much the weights move, and a table gives the median, the 5th–95th percentile range and how often each asset is held. Wide boxes mean the tangency weights are mostly noise. Single periods are resampled independently, which ignores volatility clustering, so the real uncertainty is likely even larger.
- **Bookmark or share:** the page URL always carries your tickers (e.g. `http://localhost:8501/?tickers=AAPL,TLT,GLD`). Opening such a link loads those tickers. Other settings use their defaults.

## Defaults chosen

| Setting | Default | Why |
|---|---|---|
| Lookback | 5 years | Enough data for covariance estimates without going too far back. |
| Frequency | Daily | Most observations. |
| Risk-free rate | Latest `^IRX` (13-week T-bill) | Free, current. Falls back to 4.00% with a warning if the fetch fails. |
| Constraints | Long-only, no weight cap | Matches what most individual investors can actually hold. |
| Shorting | Off. When on, each weight is bounded to ±max weight (±100% by default) | Unbounded shorting gives extreme, unrealistic leverage. |
| Covariance | Ledoit-Wolf shrinkage | More stable weights than the sample covariance. |
| Frontier | 100 points | Smooth curve. |

## Math conventions

- **Returns:** simple returns `P_t / P_{t-1} - 1` on split- and dividend-adjusted closes.
- **Annualization:** N = 252 (daily), 52 (weekly), 12 (monthly). Annual mean = periodic mean × N. Annual covariance = periodic covariance × N.
- **Risk-free rate:** `^IRX` is quoted in percent, so it's divided by 100.
- **Minimum variance:** minimize `wᵀΣw` subject to `Σw = 1` and the weight bounds.
- **Tangency:** maximize `(wᵀμ − rf) / √(wᵀΣw)`. This is solved exactly with the standard convex reformulation: with `y = κw`, minimize `yᵀΣy` subject to `(μ − rf)ᵀy = 1`, `Σy = κ`, `κ ≥ 0` and `lower·κ ≤ y ≤ upper·κ`. Then `w = y / κ`.
  - **Why cvxpy, not SLSQP:** the reformulation is a convex quadratic program, so the solver (Clarabel) finds the global optimum every time, handles weight bounds exactly, and needs no random restarts. SLSQP on the Sharpe ratio itself is non-convex and can stop at local optima or fail near the bounds.
- **When the tangency portfolio is undefined:** if no allowed portfolio beats the risk-free rate, the problem is infeasible. The app then shows the frontier portfolio with the highest (least negative) Sharpe ratio, labels it as a fallback, explains why, and hides the CML.
- **Efficient frontier:** minimize variance subject to `μᵀw ≥ target` for 100 targets from the MVP return up to the maximum achievable return (found with a linear program). Only the upper, efficient branch is drawn. Infeasible targets are skipped.
- **Capital Market Line:** from `(0, rf)` through the tangency portfolio, with slope equal to its Sharpe ratio.
- **Self-checks** (Sanity checks tab): weights sum to 1 and respect the bounds; the MVP has the lowest volatility on the frontier; the tangency portfolio has the highest Sharpe ratio on it (tolerance 1e-4).
- **Numerics:** a 1e-10 ridge keeps near-duplicate assets (e.g., SPY and VOO) solvable. Solver output is clipped to the bounds and renormalized to remove ~1e-6 noise.

## Data handling

- Prices are adjusted closes from `yfinance` (`auto_adjust=True`).
- Tickers with no data are skipped, and each is named in a warning. Dotted tickers like `BRK.B` get a hint to try the dash form (`BRK-B`).
- **Trading calendar:** when stocks and 24/7 assets like crypto are mixed, the stocks' calendar is used. Weekends and exchange holidays are dropped, so stocks never get fake 0% returns on days their market was closed. A crypto-only portfolio keeps weekdays.
- **Bad assets are dropped before the window is set.** That covers assets that stopped trading (likely delisted or merged) and assets missing more than 5% of days over their own history. A dropped asset can't shorten the window for the others. Gaps of up to 5 days are forward-filled.
- **Window start:** the analysis window starts when the youngest remaining asset starts trading. If that cuts the lookback short by more than 30 days, the app names the asset(s) responsible.
- **Window end:** the window ends on the last day *every* asset has a price. If today's crypto bar isn't in yet, today isn't forward-filled into a fake 0% return.
- **Weekly/monthly:** a final week or month that's still in progress is dropped, so every return covers a full period. The date range shown is the actual first and last trading day used.
- **Warnings:** you get one when there are fewer than 60 return observations, when the covariance matrix is singular (more assets than observations with the sample estimator), and when two assets have a return correlation above 0.99 (e.g., SPY and VOO), since the split between them is arbitrary.

## Assumptions and known limitations

- **Expected returns are estimated from history and are very noisy.** Five years of data pins down a mean return only to within several percentage points. The tangency weights are highly sensitive to this. The minimum variance portfolio ignores expected returns and is much more stable.
- **Past performance isn't predictive.** This is a mean-variance analysis of history, not a forecast or investment advice.
- **Bond ETFs are proxies.** They have their own fees, duration drift and tracking differences relative to holding bonds directly.
- **yfinance is unofficial.** It scrapes Yahoo Finance, so it can break, rate-limit, or return gaps. Yahoo's terms limit the data to personal use.
- **The model is static.** There are no transaction costs, taxes, rebalancing or borrowing costs for shorts, and borrowing and lending happen at the same risk-free rate.
- **Mean-variance assumptions.** Volatility is treated as the only risk, which understates tail risk for assets like crypto.
- **Intraday prices:** during market hours, Yahoo's latest bar is today's live price, not a close.

## Suggested next improvements

1. **Better expected returns:** a Black-Litterman or CAPM-implied prior, so the tangency portfolio doesn't depend only on noisy historical means. Optionally add your own views on top.
2. **Weight stability diagnostics:** bootstrap the return history and show the spread of the tangency weights per asset, plus resampled-frontier (Michaud) weights.
3. **Out-of-sample backtest:** rolling-window re-optimization with transaction costs, comparing the MVP, tangency and equal-weight portfolios.

## Project layout

```
portfolio-optimizer/
├── app.py            # Streamlit UI only
├── data.py           # yfinance download, ticker parsing, cleaning/alignment
├── optimizer.py      # pure optimization functions (cvxpy), no Streamlit
├── plots.py          # Plotly figures
├── tests/
│   ├── test_optimizer.py
│   ├── test_data.py
│   ├── test_features.py   # views, bootstrap, target volatility
│   └── test_app.py        # Streamlit AppTest UI tests (offline, faked prices)
├── requirements.txt
├── pytest.ini
└── README.md
```
