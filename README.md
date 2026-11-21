# risk_kit

Portfolio risk analysis and construction toolkit, with Yahoo Finance data.

## Setup

    pip install -r requirements.txt

## Quick start

    import risk_kit as rk

    report = rk.portfolio_report(["SPY", "AGG", "GLD"], weights=[0.6, 0.3, 0.1],
                                 benchmark="SPY", start="2010-01-01")
    report["summary"]        # return, volatility, Sharpe, Sortino, drawdowns, VaR, CVaR
    report["stress"]         # historical crisis replay
    report["capm"]           # alpha and beta against the benchmark

## Dashboard

A Streamlit web app built on the toolkit. Enter tickers and weights, and it explains your portfolio's risk in plain English:

Seven tabs, simplest first; technical detail lives under **Details**, and the **My funds** tab appears only for mutual fund portfolios. Settings (benchmark, dates, safe rate, VaR level) are tucked into the sidebar's Settings panel.

| Tab | What it shows |
| --- | --- |
| Summary | One screen: a plain-English verdict, four numbers in everyday words (growth, ups and downs, worst fall, what your money became), growth vs the market, what's working / what to watch, and the top 3 next steps (`commentary.py`) |
| My funds | For Indian mutual funds. With a CAS or transaction file: your real return (XIRR) per fund and overall, compared with the same cash flows put into a Nifty 50 index fund. For every fund: rank in its SEBI category on 1/3/5-year return, Sharpe and worst fall, and the measured yearly cost of a regular plan vs its direct plan (`peers.py`). Tax if you sold today, tax-free LTCG harvesting room, exit-load and ELSS lock-in flags (`transactions.py`) |
| Markets | Today's conditions for India (default), Global or US, picked from your holdings: market mood (trend, distance from the 52-week high, volatility), headline indices, sectors, Indian fund categories (via index ETFs), rates, currency, commodities (`market.py`) |
| Try ideas › Change my portfolio | Trade simulator: buy or sell units (or amounts) of your holdings, add new funds by name or stocks/ETFs by ticker, see estimated capital-gains tax and exit loads on sells (with a CAS or transaction file), and compare before vs after (return, risk, worst fall, crisis behaviour, 5-year bad case, money and risk shares); plus textbook mixes and a real-time backtest |
| Action plan | A prioritised action plan from simple rules (`advisor.py`), plus a chat with Claude acting as a senior adviser that sees the dashboard's numbers (needs `ANTHROPIC_API_KEY`, set in the environment or pasted in the tab) |
| Try ideas › Build a new one | Build a new portfolio by **name**: search any company, ETF or Indian mutual fund ("Reliance", "Apple", "Parag Parikh Flexi Cap") or pick from popular Indian and US names, set an amount for each, and see how it would have done over 3–15 years against the Nifty 50 TR or S&P 500, its 1–10 year good / middle / bad case, money vs risk share, behaviour in past crises and plain warnings (`stocks.py`) |
| Future | 1–10 year slider: good, middle and bad case for your money, simulated by replaying blocks of the portfolio's own past returns |
| Details › Risk | VaR five ways, return distribution, VaR backtest (Kupiec test), rolling and EWMA volatility |
| Details › All statistics | Key ratios (Sharpe, VaR), drawdown chart and the full statistics table |
| Details › Diversification | Money vs risk contribution, correlation heatmap, beta / alpha / capture ratios |
| Details › Crises | Historical crises replayed on your portfolio, plus a custom shock scenario |
| Try ideas (below the simulator) | Efficient frontier, max Sharpe / min vol / risk parity mixes, a real-time backtest with costs |

Run it locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

**Use your own portfolio:** export your positions from your broker's website (CSV, Excel or a PDF statement) and drop the file on
*Upload holdings from your broker* in the sidebar. The app finds the ticker and value/shares columns
(tested with Fidelity, Schwab, Vanguard and Zerodha formats), leaves out cash, totals and notes, and lets you
correct the column choices and exchange (e.g. `.NS` for India) if needed. Files are only used in your session
and never saved. PDFs are read from their text (tables with or without lines, across pages; password-protected statements prompt for the password); scanned image PDFs can't be read. Logic lives in `holdings_import.py`.

**Indian mutual funds:** statements that list funds by name (e.g. a distributor's capital summary or a CAS) work too.
Each fund name is matched to its AMFI scheme code, and the match is verified by checking that value ÷ units on the
statement equals a NAV published by AMFI. NAV history comes from [mfapi.in](https://www.mfapi.in) (free, no key).
Funds appear as `MF:<scheme code>` tickers, so you can also type them in by hand. Matching lives in `mutual_funds.py`. All calculations live in `risk_kit.py`; `app.py` only handles inputs, charts and explanations.

**Deploy for free:** push this folder to a public GitHub repo, sign in at share.streamlit.io, and point it at `app.py`.

### Uploads

- Broker holdings exports (CSV, Excel, PDF), several at once.
- **CAMS / KFintech detailed CAS** (PDF, password is usually your PAN in capitals): exact AMFI codes, every transaction, XIRR and tax lots. Read with `casparser`; the investor's name, PAN and contact details are dropped on import and never shown or sent anywhere.
- A transaction file (CSV/Excel) with columns Date, Fund, Units, Amount, and optionally Type and AMFI code.

## Tests

    python -m pytest -q

## Data

- `get_prices` downloads adjusted prices from Yahoo Finance and caches them in `data/cache` for 24 hours. It removes short bad-price glitches (e.g. a few days quoted at 1/100 of the real price) and repairs clear unadjusted splits.
- Indian mutual funds: NAV history from mfapi.in, cached on disk for a day; AMFI's daily `NAVAll.txt` for the scheme master and SEBI categories.
- Benchmark default for ₹ portfolios: **Nifty 50 TR**, the NAV of UTI Nifty 50 Index Fund (direct, growth), which includes dividends, unlike the `^NSEI` price index.
- Risk-free rate default: India, the median 3-month annualised return of direct-plan liquid funds (≈ 91-day T-bill); US, the 13-week T-bill yield (`^IRX`).
- Tax estimates follow the rules for sales from 23 July 2024 and leave out surcharge, carried-forward losses and exit loads. They are estimates, not tax advice.
- The EDHEC course loaders (`get_returns`, `get_hfi_returns`, `get_ind_returns`, ...) expect their CSV files in `data/`.

Yahoo Finance data is provided through the unofficial `yfinance` library. Check Yahoo's terms of use before
shipping a commercial product built on it.
