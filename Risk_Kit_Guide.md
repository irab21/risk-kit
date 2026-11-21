# Risk Kit: A Beginner's Guide to Portfolio Risk & Construction

Nov 18, 2026 · @Ira

## What this guide covers

`risk_kit.py` is a Python toolkit that answers three questions an investor asks: how risky is this investment, how should I combine investments, and how do I protect my money from big losses. This guide explains every concept in it from scratch, with small numeric examples you can check by hand and the matching code.

No finance background is assumed. You only need to know that a **return** is the percentage gain or loss over a period: a stock bought at 100 and worth 105 a month later returned 5%, written as 0.05.

The guide follows the toolkit in the order you would learn it:

1. **Measuring the past:** returns, wealth, drawdowns (section 1).
2. **Describing risk:** volatility, downside risk, skewness and fat tails (section 2).
3. **Putting a number on bad days:** Value at Risk and CVaR (section 3).
4. **Comparing investments fairly:** annualizing and the Sharpe ratio (section 4).
5. **Building portfolios:** diversification, the efficient frontier, optimal portfolios (sections 5–6).
6. **Protecting a portfolio:** the CPPI insurance strategy (section 7).
7. **Imagining the future:** Monte Carlo simulation of prices (section 8).
8. **Real data:** downloading prices from Yahoo Finance (section 9).
9. **Fuller performance and benchmark analysis:** Sortino, Calmar, beta, alpha, tracking error (sections 10–11).
10. **Risk that changes over time** and better VaR models with backtests (sections 12–13).
11. **Robust portfolio construction** and realistic backtests with costs (sections 14–15).
12. **Bonds, pensions and stress tests** (sections 16–17).
13. **Turning it into a dashboard** you can ship (section 18).

### Setting up

Install the libraries once with `pip install -r requirements.txt` (numpy, pandas, scipy, matplotlib, yfinance, pytest). The toolkit lives in `~/Documents/risk_kit/risk_kit.py`. The data loaders expect the course CSV files in a `data/` folder next to it:

```
risk_kit/
  risk_kit.py
  data/
    Portfolios_Formed_on_ME_monthly_EW.csv
    edhec-hedgefundindices.csv
    ind30_m_vw_rets.csv
    ind30_m_size.csv
    ind30_m_nfirms.csv
```

Then, from a Jupyter notebook or Python session in the `risk_kit` folder:

```python
import risk_kit as rk

rets = rk.get_returns()        # small-cap vs large-cap US stocks, monthly
hfi = rk.get_hfi_returns()     # 13 hedge fund strategy indices, monthly
ind = rk.get_ind_returns()     # 30 US industry portfolios, monthly
```

Every data loader returns a pandas DataFrame: one row per month, one column per asset, values as decimals (0.05 = 5%).

To work with real investments instead of the course files, download prices from Yahoo Finance (section 9):

```python
prices = rk.get_prices(["SPY", "AGG", "GLD"], start="2010-01-01")
monthly = rk.prices_to_returns(prices, "M")
rk.summary_stats(monthly)
```

The folder also holds `tests/test_risk_kit.py`, a suite of 32 automated tests. Run `python -m pytest -q` in the `risk_kit` folder to check that everything works.

## 1. Returns, wealth index and drawdowns

A drawdown measures how far an investment has fallen from its highest point so far. It is the loss an investor actually feels, and it is often the number that makes people panic and sell.

### Returns compound, they don't add

If you gain 10% and then lose 10%, you do not break even. 100 becomes 110, then 110 × 0.9 = 99. Returns multiply. To combine returns over several periods, you multiply the growth factors (1 + return):

```latex
\text{Total return} = (1+r_1)(1+r_2)\cdots(1+r_n) - 1
```

### The wealth index

The **wealth index** answers: if I had invested 1,000 at the start, what would it be worth each month? The toolkit computes it as `1000 * (1 + returns).cumprod()`. `cumprod` is the running product, exactly the compounding above.

### Previous peak and drawdown

The **previous peak** is the highest wealth reached so far (`cummax`, the running maximum). The **drawdown** is how far today's wealth sits below that peak:

```latex
\text{Drawdown}_t = \frac{\text{Wealth}_t - \text{Peak}_t}{\text{Peak}_t}
```

A drawdown is always zero or negative. Zero means you are at a new high. The **maximum drawdown** is the worst value over the whole history: the biggest peak-to-trough loss.

### Worked example

Six months of returns for a fund, starting with 1,000:

| Month | Return | Wealth | Previous peak | Drawdown |
| --- | --- | --- | --- | --- |
| Jan | +10% | 1,100 | 1,100 | 0% |
| Feb | +5% | 1,155 | 1,155 | 0% |
| Mar | −20% | 924 | 1,155 | −20.0% |
| Apr | −10% | 832 | 1,155 | −28.0% |
| May | +15% | 956 | 1,155 | −17.2% |
| Jun | +25% | 1,195 | 1,195 | 0% |

The maximum drawdown is −28%, in April. Notice the recovery: a 28% loss needed a gain of about 39% (832 → 1,155) to get back to the peak. In general, a loss of L needs a gain of L / (1 − L) to recover. A 50% loss needs a 100% gain.

```python
import pandas as pd
import risk_kit as rk

r = pd.Series([0.10, 0.05, -0.20, -0.10, 0.15, 0.25])
dd = rk.drawdown(r)
print(dd)                    # Wealth, Previous Peak, Drawdown columns
print(dd["Drawdown"].min())  # -0.28, the maximum drawdown
print(dd["Drawdown"].idxmin()) # when it happened

# On real data, plotting wealth and its peak together:
rets = rk.get_returns()
dd = rk.drawdown(rets["LargeCap"])
dd[["Wealth", "Previous Peak"]].plot()
```

### Real-life applications

- **Fund factsheets:** almost every mutual fund and hedge fund reports its maximum drawdown next to its returns. Two funds with the same average return can have very different drawdowns.
- **Retirement planning:** a retiree withdrawing money during a 40% drawdown sells assets at low prices and may never recover. Drawdown, not average return, drives this risk.
- **Risk limits:** trading desks and fund managers often have a hard rule, such as "cut positions if the drawdown passes 15%".
- **Investor behavior:** most people abandon a strategy during a deep drawdown. Knowing a strategy's historical maximum drawdown helps you judge whether you could stick with it. US stocks fell about 50% in 2008–09 and about 34% in early 2020.

## 2. Describing risk: volatility, semideviation, skewness, kurtosis

An average return tells you nothing about how bumpy the ride is. These four statistics describe the shape of the returns: how spread out they are, whether losses are worse than gains, and how often extreme months happen.

### Volatility (standard deviation)

**Volatility** is the standard deviation of returns: the typical distance of a month's return from the average month. It is the most common measure of risk.

```latex
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(r_i - \bar{r})^2}
```

Example: Fund A returns +1%, +1%, +1%, +1% (volatility 0). Fund B returns +6%, −4%, +6%, −4% (volatility 5%). Both average +1% a month, but B is far riskier. In pandas: `returns.std()`.

### Semideviation: only the bad side

Volatility punishes big gains as much as big losses, which investors don't actually mind. **Semideviation** (`rk.semideviation`) is the standard deviation of the negative returns only. It measures downside risk.

Example: returns of +8%, +12%, −2%, −4%. Volatility is large because of the big gains. Semideviation looks only at −2% and −4%: their standard deviation is 1%. A fund with high volatility but low semideviation has mostly "good" volatility.

### Skewness: are the surprises good or bad?

**Skewness** (`rk.skewness`) measures asymmetry. It is the average cubed deviation divided by volatility cubed:

```latex
S = \frac{E[(R - \bar{R})^3]}{\sigma^3}
```

- **Skewness = 0:** gains and losses are symmetric, like a bell curve.
- **Negative skewness:** many small gains with occasional large losses. Selling insurance looks like this: you collect small premiums every month until a disaster costs you a lot.
- **Positive skewness:** many small losses with occasional large gains. A lottery ticket or a venture capital fund looks like this.

Cubing keeps the sign, so large losses pull skewness negative. Many hedge fund strategies in the EDHEC data have negative skewness, which an average-and-volatility summary hides.

### Kurtosis: how fat are the tails?

**Kurtosis** (`rk.kurtosis`) measures how often extreme returns happen. It uses the fourth power, which magnifies big deviations:

```latex
K = \frac{E[(R - \bar{R})^4]}{\sigma^4}
```

A normal (bell-curve) distribution has kurtosis of exactly 3. Anything above 3 has **fat tails**: crashes and booms happen more often than a bell curve predicts. Stock returns typically have kurtosis well above 3. Note that `rk.kurtosis` returns raw kurtosis, while `scipy.stats.kurtosis` returns *excess* kurtosis (kurtosis minus 3) by default.

Why it matters: under a normal distribution, a daily move of 5 standard deviations should happen roughly once every several thousand years. The S&P 500 fell about 20% in one day on 19 October 1987, a move far beyond what a bell curve allows.

### Testing for normality: Jarque-Bera

Many simple risk models assume returns are normal. `rk.is_normal(r)` runs the **Jarque-Bera test**, which checks whether the skewness is close to 0 and the kurtosis close to 3 at the same time.

It returns `True` if normality cannot be rejected at the 1% level (the default `level=0.01`) and `False` if the returns are clearly not normal. Pass a DataFrame and you get one answer per column.

```python
hfi = rk.get_hfi_returns()

pd.DataFrame({
    "Volatility": hfi.std(ddof=0),
    "Semideviation": rk.semideviation(hfi),
    "Skewness": rk.skewness(hfi),
    "Kurtosis": rk.kurtosis(hfi),
    "Normal?": rk.is_normal(hfi),
}).sort_values("Skewness")
```

On the EDHEC hedge fund indices, most strategies fail the normality test, with negative skewness and kurtosis well above 3.

### Real-life applications

- **Choosing funds:** two funds with the same return and volatility can differ greatly in skewness. A negatively skewed fund can look safe for years and then lose heavily.
- **Options and insurance:** strategies that sell options or insurance earn steady income with negative skewness. Investors need to see that hidden risk.
- **Regulation:** banks are expected to model fat tails rather than assume normal returns, after the 2008 crisis showed bell-curve models underestimating losses.
- **The Sortino ratio:** a popular variant of the Sharpe ratio uses semideviation instead of volatility, so it does not penalize upside surprises.

## 3. Value at Risk (VaR) and Conditional VaR (CVaR)

Value at Risk answers one plain question: on a bad month, how much could I lose? A 5% monthly VaR of 7% means: in 95 months out of 100, you should lose less than 7%. In the worst 5 months out of 100, you lose more.

VaR is reported as a positive number for a loss, which is why every VaR function in the toolkit puts a minus sign in front.

### Method 1: Historic VaR

`rk.var_historic(r, level=5)` uses only past data. Sort the returns from worst to best and read off the return at the 5th percentile.

Example: you have 100 months of returns. Sorted, the five worst are −15%, −12%, −10%, −9%, −7%. The 5% historic VaR is about 7%: only 5% of months were worse than that. (`np.percentile` interpolates, so the exact output can differ slightly.)

- **Pro:** no assumptions about the shape of returns.
- **Con:** it depends entirely on the past sample. If the past 10 years had no crash, VaR will look reassuringly small.

### Method 2: Gaussian (parametric) VaR

`rk.var_gaussian(r, level=5)` assumes returns are normally distributed and uses only the mean and volatility:

```latex
\text{VaR} = -(\mu + z\,\sigma)
```

Here z is the z-score of the chosen level: about −1.645 for 5% and −2.33 for 1%.

Example: mean monthly return 1%, volatility 4%. The 5% VaR = −(0.01 − 1.645 × 0.04) = 0.0558, a 5.6% loss.

- **Pro:** simple, smooth, needs only two numbers.
- **Con:** real returns have fat tails and negative skew (section 2), so Gaussian VaR usually **understates** the risk.

### Method 3: Cornish-Fisher (modified) VaR

`rk.var_cornish_fisher(r, level=5)` keeps the Gaussian formula but adjusts z using the actual skewness S and kurtosis K of the returns:

```latex
z_{cf} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)(K-3)}{24} - \frac{(2z^3-5z)S^2}{36}
```

Negative skewness and fat tails push z further into the loss side, so the VaR gets bigger. It is a practical middle ground: it reacts to the shape of the returns without relying only on a handful of past disasters.

### CVaR: how bad is it beyond the VaR?

VaR tells you where the bad 5% starts, but not how bad things get inside it. **Conditional VaR** (also called **Expected Shortfall**) is the average loss in those worst cases. `rk.cvar_historic(r, level=5)` averages every return at or below the historic VaR.

Using the same five worst months, the 5% CVaR = average of 15%, 12%, 10%, 9% and 7% = 10.6%. So: "in the worst 5% of months, expect to lose about 10.6% on average."

Two funds can share a VaR of 7% while one has a CVaR of 9% and the other 25%. CVaR exposes the difference.

### Comparing the methods

```python
hfi = rk.get_hfi_returns()

comparison = pd.concat([
    rk.var_gaussian(hfi, level=5),
    rk.var_cornish_fisher(hfi, level=5),
    rk.var_historic(hfi, level=5),
    rk.cvar_historic(hfi, level=5),
], axis=1)
comparison.columns = ["Gaussian", "Cornish-Fisher", "Historic", "CVaR"]
comparison.plot.bar(title="Hedge fund indices: 5% monthly VaR")
```

For most hedge fund strategies, Gaussian VaR comes out smallest, Cornish-Fisher larger, and CVaR largest.

### Real-life applications

- **Bank capital:** banks have long used VaR to decide how much capital to hold against trading losses. The Basel rules for trading books now use Expected Shortfall (CVaR), because VaR ignores the size of losses past the cutoff.
- **Risk limits:** a trading desk may be told "your 1-day 99% VaR must stay below $10 million".
- **Fund reporting:** many funds publish VaR in their risk disclosures.
- **Personal finance:** a 5% monthly VaR of 8% on a $50,000 portfolio means a $4,000 loss in a bad month is normal, not a reason to panic.

## 4. Annualizing and the Sharpe ratio

To compare a fund measured monthly with one measured daily, convert both to yearly figures. The Sharpe ratio then tells you how much return you earn per unit of risk, which is the fairest single-number comparison of two investments.

### Annualizing returns

`rk.annualize_rets(r, periods_per_year)` compounds the returns and scales the result to one year:

```latex
R_{annual} = \left(\prod_{i=1}^{N}(1+r_i)\right)^{P/N} - 1
```

P is the number of periods per year (12 for monthly, 252 for daily trading days, 4 for quarterly) and N is the number of periods you have.

Example: an average of 1% a month for a year gives (1.01)^12 − 1 = 12.68%, not 12%, because of compounding.

### Annualizing volatility

Risk grows with the square root of time, not with time itself. `rk.annualize_vol(r, periods_per_year)` multiplies by the square root of P:

```latex
\sigma_{annual} = \sigma_{period} \times \sqrt{P}
```

Example: monthly volatility of 4% becomes 4% × √12 = 13.9% a year. Daily volatility of 1% becomes 1% × √252 = 15.9% a year.

The square root comes from random ups and downs partly cancelling out over time. Over 12 months the spread of total outcomes grows by √12 ≈ 3.5 times, not 12 times.

### The Sharpe ratio

An investment should be judged by the return it earns **above** a risk-free alternative such as a government Treasury bill, divided by the risk taken. `rk.sharpe_ratio(r, riskfree_rate, periods_per_year)` computes:

```latex
\text{Sharpe} = \frac{R_{annual} - R_f}{\sigma_{annual}}
```

The function first converts the annual risk-free rate to a monthly one, subtracts it from every month, then annualizes the excess return and divides by annualized volatility.

Example: a fund averages 1% a month with 4% monthly volatility, and Treasury bills pay 3% a year. The annual excess return is about 9.4%, the annual volatility 13.9%, so the Sharpe ratio is about 0.68.

Compare two funds:

| Fund | Annual return | Annual volatility | Sharpe (rf = 3%) |
| --- | --- | --- | --- |
| Steady Fund | 8% | 6% | 0.83 |
| Rocket Fund | 15% | 25% | 0.48 |

The Rocket Fund earns more, but the Steady Fund pays you better for each unit of risk. With borrowing, you could lever up the Steady Fund to match the Rocket Fund's risk and earn more than 15%. That is why professionals compare Sharpe ratios, not raw returns.

Rough guide: below 0.5 is weak, 0.5–1 is decent, above 1 is very good, above 2 is rare and deserves suspicion over long periods.

```python
ind = rk.get_ind_returns()["1995":"2000"]

summary = pd.DataFrame({
    "Return": rk.annualize_rets(ind, 12),
    "Volatility": rk.annualize_vol(ind, 12),
    "Sharpe": rk.sharpe_ratio(ind, 0.03, 12),
}).sort_values("Sharpe", ascending=False)
summary.head(10)
```

### Real-life applications

- **Fund selection:** fund databases such as Morningstar report Sharpe ratios so investors can compare funds with different risk levels.
- **Manager pay:** hedge fund and trading desk performance is often judged on risk-adjusted return, not raw profit.
- **Strategy research:** quantitative researchers test new strategies by their Sharpe ratio after costs.
- **Comparing across frequencies:** a crypto fund reporting daily data and a pension fund reporting quarterly data can only be compared after annualizing.

## 5. Portfolio math and diversification

A portfolio's return is simply the weighted average of its assets' returns, but its risk is usually **lower** than the weighted average of their risks. That gap is diversification, often called the only free lunch in finance.

### Weights

A portfolio is described by its **weights**: the share of money in each asset. Investing 6,000 in stocks and 4,000 in bonds gives weights of 0.6 and 0.4. Weights add up to 1.

### Portfolio return

`rk.portfolio_returns(weights, returns)` computes the weighted average, written in matrix form as wᵀR:

```latex
R_p = w_1 R_1 + w_2 R_2 + \cdots + w_n R_n = w^T R
```

Example: 60% in stocks expected to return 10% and 40% in bonds expected to return 4%. The portfolio's expected return is 0.6 × 10% + 0.4 × 4% = 7.6%.

### Covariance and correlation

Risk does not average so simply, because assets don't always move together. **Correlation** (ρ) measures how two assets move together, from −1 to +1:

- **+1:** they always move in the same direction. No diversification benefit.
- **0:** unrelated. Some benefit.
- **−1:** they always move in opposite directions. Maximum benefit; risk can even be cancelled out.

**Covariance** is correlation scaled by both volatilities: Cov(A, B) = ρ × σA × σB. A **covariance matrix** holds the covariance of every pair, with each asset's variance (σ²) on the diagonal. In pandas: `returns.cov()`.

### Portfolio volatility

`rk.portfolio_vol(weights, covmat)` computes the portfolio's volatility from the covariance matrix:

```latex
\sigma_p = \sqrt{w^T \Sigma w}
```

For two assets this expands to:

```latex
\sigma_p = \sqrt{w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2 w_1 w_2 \rho\, \sigma_1 \sigma_2}
```

### Worked example: correlation is what matters

Asset A returns 10% with 20% volatility. Asset B returns 6% with 10% volatility. Put 50% in each. The expected return is 8% whatever the correlation, but the risk changes a lot:

| Correlation ρ | Portfolio return | Portfolio volatility |
| --- | --- | --- |
| +1.0 | 8% | 15.0% |
| 0.0 | 8% | 11.2% |
| −0.5 | 8% | 8.7% |

With correlation of +1, the portfolio volatility is the plain average (15%). Lower correlation cuts risk without giving up any return. At −0.5 the portfolio is less risky than asset B alone, while earning more than B.

```python
import numpy as np
import risk_kit as rk

ind = rk.get_ind_returns()["1996":"2000"]
er = rk.annualize_rets(ind, 12)   # expected returns
cov = ind.cov()                   # covariance matrix

assets = ["Food", "Beer", "Smoke", "Coal"]
w = np.repeat(1/4, 4)             # equal weights
print(rk.portfolio_returns(w, er[assets]))
print(rk.portfolio_vol(w, cov.loc[assets, assets]))
```

### Real-life applications

- **The 60/40 portfolio:** the classic pension and retirement mix of 60% stocks and 40% bonds relies on stocks and bonds often having low or negative correlation.
- **Global investing:** owning shares in several countries reduces risk because markets don't all fall at once.
- **A warning from crises:** in a panic, correlations tend to jump towards +1, so diversification works least when you need it most. In 2022, stocks and bonds fell together.
- **Concentration risk:** an employee holding most of their savings in their employer's stock has no diversification. If the company fails, they lose their job and savings together.

## 6. The efficient frontier and optimal portfolios

For any set of assets, there is a curve of best possible portfolios: the highest expected return for each level of risk. This curve is the **efficient frontier**, the core idea of Harry Markowitz's Modern Portfolio Theory (1952), which won him a Nobel Prize. Any portfolio below the curve is wasteful: you could earn more for the same risk.

### Two assets: plot\_ef2

With two assets, every split from 100% A to 100% B is a portfolio. `rk.plot_ef2(n_points, er, cov)` tries `n_points` splits and plots the return and volatility of each.

Using assets A (10% return, 20% volatility) and B (6% return, 10% volatility) with correlation 0.2:

&#91;embedded content: computed with rk.portfolio\_vol, rk.gmv and rk.msr · illustrative assets A and B\]

The curve bends to the left: some mixes are *less* risky than B alone, even though A is twice as volatile as B. That is diversification at work. Everything on the lower part of the curve, below the minimum-volatility point, is inefficient: a point straight above it offers more return for the same risk.

### Many assets: minimize\_vol and optimal\_weights

With 30 assets you can't try every combination by hand. The toolkit uses an optimizer (`scipy.optimize.minimize` with the SLSQP method) instead:

- `rk.minimize_vol(target_return, er, cov)` finds the weights with the lowest volatility that still reach the target return. Weights must be between 0 and 1 (no short selling) and add up to 1.
- `rk.optimal_weights(n_points, er, cov)` repeats this for evenly spaced targets between the lowest and highest asset return.
- `rk.plot_ef(n_points, er, cov)` plots the resulting frontier.

### The Global Minimum Variance (GMV) portfolio

`rk.gmv(cov)` finds the leftmost point on the frontier: the least risky portfolio possible. It needs only the covariance matrix, not the expected returns. That matters because expected returns are very hard to estimate, while covariances are much more stable. In the chart, the GMV portfolio is 14% in A and 86% in B.

### The Maximum Sharpe Ratio (MSR) portfolio and the capital market line

If you can also hold a risk-free asset, the best risky portfolio is the one with the highest Sharpe ratio. `rk.msr(riskfree_rate, er, cov)` finds it by minimizing the negative Sharpe ratio.

Draw a straight line from the risk-free rate on the vertical axis so it just touches the frontier. That line is the **capital market line** (CML), and the touching point is the MSR portfolio. Every point on the CML is a mix of the MSR portfolio and cash, and each one beats the frontier at the same risk. A cautious investor holds some MSR plus cash; an aggressive one borrows to hold more MSR. Everyone holds the same risky mix: this is **two-fund separation**.

### Equal weight (EW)

The simplest portfolio puts 1/N in each asset. It needs no estimates at all. Research has found that equal weighting often performs about as well as optimized portfolios out of sample, because optimizers chase estimation errors.

### Putting it together

```python
ind = rk.get_ind_returns()["1996":"2000"]
er = rk.annualize_rets(ind, 12)
cov = ind.cov() * 12          # annualize the covariance to match er

rk.plot_ef(25, er, cov,
           show_cml=True, riskfree_rate=0.03,
           show_ew=True, show_gmv=True)

w_msr = rk.msr(0.03, er, cov)
pd.Series(w_msr, index=er.index).sort_values(ascending=False).head()
```

The expected returns and the covariance matrix must be in the same units. `er` is annual, so multiply the monthly covariance by 12.

### The big caveat: garbage in, garbage out

The MSR portfolio is very sensitive to the expected returns you feed it. Change one asset's estimate by 1% and the weights can swing from 0% to 50%. Because past returns are a poor guide to future returns, the MSR optimizer often produces extreme, concentrated portfolios. This is why practitioners often prefer GMV or equal weights, or shrink their estimates.

### Real-life applications

- **Robo-advisors:** services that build portfolios from a risk questionnaire place you on an efficient frontier of index funds.
- **Pension funds and endowments:** they use mean-variance optimization to decide how much to put in stocks, bonds, real estate and other asset classes.
- **Minimum volatility ETFs:** funds that track minimum variance indices apply the GMV idea to stocks.
- **The case for index funds:** the CML idea underpins the argument that most investors should hold one broad market portfolio and adjust risk with cash or bonds.

## 7. CPPI: insuring a portfolio against big losses

CPPI (Constant Proportion Portfolio Insurance) is a rule for moving money between a risky asset and a safe asset so the portfolio never falls below a chosen **floor**, while still capturing much of the upside. `rk.run_cppi` backtests this rule month by month.

### The idea in plain words

Decide the minimum you are willing to end up with, say 80% of your starting money. The amount above that floor is your **cushion**: money you can afford to risk. Put a multiple of the cushion into the risky asset and the rest into the safe asset. As the portfolio falls, the cushion shrinks and you automatically sell risky assets. As it rises, you buy more.

### The rule, each month

1. **Cushion** = (account value − floor value) ÷ account value.
2. **Risky weight** = m × cushion, kept between 0% and 100%. m is the **multiplier**.
3. **Safe weight** = 1 − risky weight.
4. Apply this month's risky and safe returns to get the new account value.
5. Repeat next month with the new value.

### Worked example

Start with 1,000, floor at 80% (800), multiplier m = 3. For simplicity the safe asset earns 0%.

| Month | Start value | Cushion | Risky weight | Risky amount | Market return | End value | 100% stocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,000 | 20.0% | 60.0% | 600 | −10% | 940 | 900 |
| 2 | 940 | 14.9% | 44.7% | 420 | −10% | 898 | 810 |
| 3 | 898 | 10.9% | 32.7% | 294 | +20% | 957 | 972 |

After two bad months the CPPI portfolio is at 898 against 810 for a pure stock investor, and well above the 800 floor. In the strong third month CPPI gains less, because it had cut back on stocks. That is the price of the insurance: less upside after losses.

### Choosing the multiplier

A higher m means more upside but more risk. The floor can be broken only if the risky asset drops more than 1/m in a single step before you can rebalance. With m = 3 that is a 33% one-month crash; with m = 5 only 20%. This is called **gap risk**. In October 1987 portfolio insurance strategies of this kind were blamed for adding selling pressure during the crash.

### The drawdown version

With `drawdown=0.25`, the floor is no longer fixed. It rises with the portfolio's highest value: floor = peak × (1 − 0.25). The strategy then aims to never lose more than 25% from any previous high, which locks in gains as the portfolio grows.

### Using run\_cppi

```python
ind = rk.get_ind_returns()["2000":]
risky = ind[["Steel", "Fin", "Beer"]]

bt = rk.run_cppi(risky, m=3, start=1000, floor=0.8, riskfree_rate=0.03)

ax = bt["Wealth"]["Fin"].plot(label="CPPI", legend=True, figsize=(12, 6))
bt["Risky Wealth"]["Fin"].plot(ax=ax, style="k:", label="100% Fin", legend=True)
bt["floor_value"]["Fin"].plot(ax=ax, style="r--", label="Floor", legend=True)

# Drawdown-based floor
bt_dd = rk.run_cppi(risky, drawdown=0.25)
rk.drawdown(bt_dd["Wealth"]["Fin"])["Drawdown"].min()   # stays near -25%
```

The function returns a dictionary:

| Key | What it holds |
| --- | --- |
| `Wealth` | The CPPI portfolio's value each month |
| `Risky Wealth` | What 100% in the risky asset would have been worth |
| `Risk Budget` | The cushion each month |
| `Risky Allocation` | The risky weight each month |
| `floor_value` | The floor each month (rises over time with `drawdown`) |
| `peak` | The highest value reached so far |
| `m`, `start`, `floor`, `drawdown`, `risky_r`, `safe_r` | The inputs, for reference |

If no `safe_r` is given, the safe asset earns `riskfree_rate / 12` each month.

### Real-life applications

- **Capital-protected products:** banks sell structured products and guaranteed funds that promise to return at least, say, 90% of your money after five years. CPPI is one common engine behind them.
- **Target-date and retirement products:** some use CPPI-style rules to protect a minimum income level near retirement.
- **Personal risk rules:** "never let my savings drop below the amount I need for a house deposit" is a CPPI floor.
- **Drawdown control at funds:** many managers cut risk as losses mount, which is CPPI with a drawdown floor in practice.

## 8. Simulating the future: Geometric Brownian Motion and Monte Carlo

History gives you only one path. Monte Carlo simulation creates thousands of possible futures, so you can ask "in how many of them do I reach my goal?" `rk.gbm` generates these futures with the most common model of stock prices, Geometric Brownian Motion (GBM).

### The model in plain words

Each month, the price grows by a predictable **drift** (the expected return, mu) plus a random shock whose size is set by the **volatility** (sigma). Because each month's change multiplies the price, the price can never go below zero, just like a real stock.

```latex
\frac{S_{t+1}}{S_t} = 1 + \mu\,\Delta t + \sigma\sqrt{\Delta t}\,\xi, \qquad \xi \sim N(0,1)
```

Here Δt is the length of one step in years (1/12 for monthly). The volatility is scaled by √Δt, the same square-root-of-time rule from section 4. The toolkit uses (1 + mu)^Δt as the mean growth instead of 1 + mu × Δt, which compounds exactly to mu per year.

### Parameters

| Parameter | Meaning | Default |
| --- | --- | --- |
| `n_years` | How many years to simulate | 10 |
| `n_scenarios` | How many possible futures | 1,000 |
| `mu` | Expected annual return | 0.07 (7%) |
| `sigma` | Annual volatility | 0.15 (15%) |
| `steps_per_year` | 12 = monthly, 252 = daily | 12 |
| `s_0` | Starting price or amount | 100 |
| `prices` | `True` returns prices; `False` returns the per-step returns | True |

The result is a DataFrame with one row per time step (starting at s\_0 in row 0) and one column per scenario.

### Worked example: will my savings double?

You invest 10,000 for 10 years in a fund expected to return 7% a year with 15% volatility. At exactly 7% a year, you would end with about 19,700. But returns are random:

```python
import numpy as np
import risk_kit as rk

np.random.seed(42)  # makes the run repeatable
prices = rk.gbm(n_years=10, n_scenarios=5000, mu=0.07, sigma=0.15, s_0=10_000)

final = prices.iloc[-1]
print(final.median())                  # typical outcome
print(final.quantile([0.05, 0.95]))    # bad case and good case
print((final >= 20_000).mean())        # chance of doubling
print((final < 10_000).mean())         # chance of losing money

prices.iloc[:, :50].plot(legend=False, alpha=0.4)  # 50 sample paths
```

One run of this code (seed 42) gave a median of about 17,600, below the 19,700 you would get at a steady 7%. The 5% bad case was about 8,000 and the 95% good case about 37,400. About 39% of paths doubled, 12% lost money after 10 years, and 12% more than tripled. The median sits below the average because volatility drags down compounded growth: losing 15% and gaining 15% leaves you at 97.75%, not 100%.

### Combining GBM with CPPI

Simulated returns can feed straight into `run_cppi` to test the insurance strategy across thousands of futures rather than one history:

```python
rets = rk.gbm(n_years=5, n_scenarios=1000, mu=0.07, sigma=0.20, prices=False)
rets = pd.DataFrame(rets).iloc[1:]   # drop the starting row
bt = rk.run_cppi(rets, m=3, floor=0.8, riskfree_rate=0.03)
final = bt["Wealth"].iloc[-1]
print((final < 800).mean())          # how often the floor was broken
```

### Real-life applications

- **Retirement planners:** financial planning software reports a "probability of success", such as "an 85% chance your money lasts until age 95". That number comes from Monte Carlo simulation.
- **Option pricing:** the Black-Scholes model for pricing options assumes stock prices follow GBM. Banks price complex options by simulating GBM paths.
- **Pension funding:** pension funds simulate assets and liabilities to estimate the chance of a funding shortfall.
- **Stress testing a strategy:** rather than trusting one backtest, you can see how a strategy behaves across thousands of futures.

**Limits:** GBM assumes normal returns with constant volatility. Real markets have fat tails, volatility clusters and crashes (section 2), so GBM tends to understate extreme outcomes.

## 9. Loading real data from Yahoo Finance

The toolkit can now download prices for any stock, ETF, index or currency on Yahoo Finance and turn them into returns. Everything in this guide then works on real investments you choose.

### Prices versus returns

Yahoo gives you **prices**: what one share was worth each day. Risk measures need **returns**: the percentage change from one period to the next.

```latex
r_t = \frac{P_t}{P_{t-1}} - 1
```

Example: SPY closes at 500 on Monday and 505 on Tuesday. Tuesday's return is 505 / 500 − 1 = 1%.

### Adjusted prices: don't forget dividends

When a company pays a dividend, its share price drops by about that amount, but shareholders haven't lost money: they received cash. A stock split (one share becomes two) halves the price without any loss at all. **Adjusted prices** correct for both, so their changes are the true **total return**. `get_prices` always downloads adjusted prices.

Ignoring this is a classic beginner mistake: a stock paying a 4% dividend would look 4% a year worse than it really was.

### The functions

| Function | What it does |
| --- | --- |
| `get_prices(tickers, start, end)` | Daily adjusted prices, one column per ticker |
| `prices_to_returns(prices, freq)` | Returns at daily ("D"), weekly ("W"), monthly ("M"), quarterly ("Q") or yearly ("Y") frequency |
| `get_yahoo_returns(tickers, start, end, freq)` | Both steps in one call |
| `infer_periods_per_year(r)` | Works out 252, 52, 12, 4 or 1 from the dates |
| `clean_returns(r)` | Drops empty rows and warns about suspicious returns above 100% |
| `load_returns_csv(path)` | Loads returns from your own CSV file |

```python
import risk_kit as rk

prices = rk.get_prices(["SPY", "AGG", "GLD"], start="2010-01-01")
monthly = rk.prices_to_returns(prices, "M")
rk.summary_stats(monthly)    # periods per year is detected automatically
```

Tickers to try: `SPY` (US stocks), `QQQ` (Nasdaq 100), `AGG` (US bonds), `GLD` (gold), `EFA` (international stocks), `^GSPC` (the S&P 500 index), `AAPL` (Apple), `BTC-USD` (Bitcoin).

### Daily or monthly?

Daily data gives many more observations, which helps with VaR and rolling statistics. Monthly data is less noisy and better for long-term portfolio decisions. Note that the same investment shows different VaR at different frequencies: a 1-day VaR of 2% is not the same as a 1-month VaR of 2%.

### Caching

Downloads are saved in `data/cache` and reused for 24 hours. This makes a dashboard fast and avoids hitting Yahoo repeatedly. Pass `use_cache=False` to force a fresh download.

### Things to know before you ship

- **Missing history:** an ETF launched in 2004 has no data for 2000. Statistics then cover different periods for different assets. `portfolio_report` drops the dates where any holding is missing.
- **Bad tickers:** a ticker Yahoo doesn't recognize is dropped with a warning instead of crashing.
- **Survivorship bias:** Yahoo only lists companies that still trade. Testing a strategy on today's S&P 500 members ignores the companies that went bust, which makes the past look better than it was.
- **Terms of use:** `yfinance` is an unofficial library that reads Yahoo's public website. It can break when Yahoo changes its site, and Yahoo's terms restrict commercial use. For a paid product, plan for a licensed data provider; keeping all downloads inside `get_prices` makes that switch a one-function change.

## 10. More performance measures

The Sharpe ratio treats every bump the same way. Three extra measures look at the pain investors actually feel: the Sortino ratio (losses only), the Calmar ratio (worst drawdown) and drawdown duration (how long you wait to recover). `summary_stats` puts them all in one table.

### Sortino ratio

`rk.sortino_ratio(r, riskfree_rate)` is the Sharpe ratio with **downside deviation** in place of volatility. Downside deviation only counts returns below the risk-free rate; months above it count as zero.

```latex
\text{Sortino} = \frac{R_{annual} - R_f}{\text{Downside deviation}}
```

Example: two funds both earn 9% with 15% volatility. Fund A's volatility comes from big up-months; Fund B's from big down-months. Their Sharpe ratios are equal, but Fund A has a much higher Sortino ratio. Sortino ratios are usually higher than Sharpe ratios because downside deviation is smaller than volatility.

### Calmar ratio

`rk.calmar_ratio(r)` divides the annual return by the size of the maximum drawdown. A fund earning 10% a year with a worst drawdown of −25% has a Calmar ratio of 0.4. It answers: how many years of average return does the worst loss wipe out? Hedge funds and trend-following funds quote it often.

### Drawdown duration

`rk.drawdown_duration(r)` counts how many periods the investment spent below its previous high. `Longest` is the longest such stretch; `Current` is how long it has been underwater as of today (0 at a new high). Depth and length are different kinds of pain: US stocks took about 5½ years (2007 to 2013) to regain their 2007 high.

### summary\_stats: the whole picture in one table

`rk.summary_stats(r, riskfree_rate=0.03)` reports, for every column: annual return, volatility, Sharpe, Sortino, Calmar, max drawdown, longest drawdown, skewness, kurtosis, historic and Cornish-Fisher VaR, CVaR, best and worst period, % positive periods and the number of periods.

A real example, using daily Yahoo Finance data from January 2005 to October 2025 and a 3% risk-free rate. The portfolio is 40% SPY (US stocks), 20% QQQ (Nasdaq 100), 30% AGG (bonds) and 10% GLD (gold):

| Measure | SPY (stocks) | Portfolio |
| --- | --- | --- |
| Annual return | 10.7% | 10.0% |
| Annual volatility | 19.1% | 12.1% |
| Sharpe ratio | 0.39 | 0.56 |
| Sortino ratio | 0.55 | 0.79 |
| Calmar ratio | 0.19 | 0.30 |
| Max drawdown | −55.2% | −33.4% |
| Longest drawdown | 1,223 trading days | 596 trading days |
| 5% daily VaR | 1.8% | 1.1% |

The portfolio gave up 0.7% a year of return but cut volatility by more than a third and the worst loss from 55% to 33%. Every risk-adjusted measure improved. This is diversification at work on real data.

```python
prices = rk.get_prices(["SPY", "QQQ", "AGG", "GLD"], start="2005-01-01")
r = rk.prices_to_returns(prices, "D")
r["Portfolio"] = rk.portfolio_return_series(r, {"SPY": .4, "QQQ": .2, "AGG": .3, "GLD": .1})
rk.summary_stats(r, riskfree_rate=0.03).T.round(3)
```

### Real-life applications

- **Fund factsheets** report most of these numbers; the summary table is the core of a risk dashboard.
- **Investor suitability:** someone who must withdraw money in two years cares more about drawdown duration than Sharpe ratio.
- **Manager selection:** consultants screen managers on Sortino and Calmar, which reward controlling losses.

## 11. Risk against a benchmark: beta, alpha, CAPM, tracking error

Most investments are judged against a **benchmark**: a simple alternative you could have bought instead, usually a market index such as the S&P 500 (SPY). These measures answer two questions: how much does my investment just follow the market, and did it add anything on top?

### Beta: sensitivity to the market

`rk.beta(r, benchmark)` measures how much an asset moves, on average, when the market moves 1%.

```latex
\beta = \frac{\text{Cov}(R, R_m)}{\text{Var}(R_m)}
```

- **Beta = 1:** moves with the market, like an index fund.
- **Beta = 1.5:** amplifies the market. If the market falls 10%, expect about −15%.
- **Beta = 0.5:** half as sensitive, like a utility company.
- **Beta near 0:** unrelated to the market, like gold or cash.
- **Negative beta:** tends to rise when the market falls. Rare, and valuable as protection.

### CAPM and alpha

The **Capital Asset Pricing Model** says an asset's fair expected return depends only on its beta:

```latex
E[R] = R_f + \beta\,(E[R_m] - R_f)
```

Example: risk-free rate 3%, expected market return 8%, beta 1.2. Fair expected return = 3% + 1.2 × 5% = 9% (`rk.capm_expected_return(1.2, 0.03, 0.08)`).

**Alpha** is the return above what beta explains: the manager's skill (or luck). `rk.capm(r, benchmark, riskfree_rate)` runs a regression of each asset's excess return on the market's excess return and reports four numbers per asset:

| Output | Meaning |
| --- | --- |
| Alpha (annual) | Extra return per year not explained by market exposure |
| Beta | Market sensitivity |
| R-squared | Share of the asset's ups and downs explained by the market (0 to 1) |
| Alpha t-stat | Whether alpha is distinguishable from luck; above about 2 is usually called significant |

Real results (daily data, January 2005 to October 2025, SPY as benchmark): QQQ had a beta of 1.04, an R-squared of 0.85 and alpha of about 4.1% a year (t-stat 2.1), reflecting the big technology companies' run. Gold (GLD) had a beta of just 0.05: its returns had almost nothing to do with stocks, which is why it diversified so well. Note that "alpha" in a single asset like gold or QQQ is a historical fact, not proof of future outperformance.

### Tracking error and information ratio

**Tracking error** (`rk.tracking_error`) is the annual volatility of the difference between your return and the benchmark's. An index fund has a tracking error near 0%; an active fund usually 2–8%. The 40/20/30/10 portfolio from section 10 had a tracking error of 8% against SPY: it behaves quite differently from the stock market.

The **information ratio** (`rk.information_ratio`) is the Sharpe ratio of beating the benchmark: average yearly outperformance divided by tracking error. Above 0.5 is good for an active manager. The 40/20/30/10 portfolio's information ratio was −0.22, because it trailed an all-stock benchmark in a rising market. That is expected: it was built for lower risk, not to beat stocks. Choosing the right benchmark matters.

### Up and down capture

`rk.up_down_capture(r, benchmark)` compares average returns in up-markets and down-markets. An up capture of 0.9 and a down capture of 0.6 means the investment captured 90% of the market's gains but only 60% of its losses: an attractive profile.

### Real-life applications

- **Active vs passive debate:** most active funds show near-zero or negative alpha after fees, which is the main argument for index funds.
- **Hedging:** a portfolio with beta 1.2 and value $1 million can be made market-neutral by shorting $1.2 million of index futures.
- **Cost of capital:** companies use CAPM to estimate the return shareholders expect, which sets the hurdle for new projects.
- **Fund mandates:** institutional investors often cap tracking error, such as "stay within 3% of the MSCI World".

## 12. Risk over time: rolling statistics and EWMA

Risk is not a fixed number. A single volatility figure for 20 years hides calm years and panics. Rolling statistics and EWMA show how risk changes over time, which is what a live dashboard needs.

&#91;embedded content: Yahoo Finance daily SPY prices, computed with rk.ewma\_vol · Apr 2005 to Oct 2025\]

The same index can be a calm 6% investment (2017) or a wild 80% one (late 2008). A risk figure computed over the whole period would describe neither.

### Rolling statistics

A **rolling** statistic is recalculated over a moving window: for example, the volatility of the last 63 trading days (about three months), recomputed every day. The toolkit has:

| Function | What it shows over time |
| --- | --- |
| `rolling_vol(r, window)` | Annualized volatility |
| `rolling_sharpe(r, window, riskfree_rate)` | Sharpe ratio |
| `rolling_beta(r, benchmark, window)` | Beta against a benchmark |
| `rolling_corr(r, benchmark, window)` | Correlation with a benchmark |
| `rolling_max_drawdown(r, window)` | Worst drawdown inside each window |

Choosing the window is a trade-off. A short window (21 days) reacts quickly but jumps around. A long window (252 days) is smooth but slow to notice danger. Common choices: 63 days for daily data, 36 months for monthly data.

`rolling_corr` is especially useful: it shows diversification failing in real time, as correlations between stocks rise towards 1 in a crisis.

### EWMA volatility: recent days count more

A rolling window has two flaws: every day in the window counts equally, and a big shock drops out suddenly when it leaves the window. The **exponentially weighted moving average** (EWMA) fixes both by giving each day a weight that fades smoothly with age:

```latex
\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2
```

In words: today's variance estimate = 94% of yesterday's estimate + 6% of yesterday's squared return (with λ = 0.94). The 0.94 is the standard from J.P. Morgan's RiskMetrics system for daily data; `rk.ewma_vol(r, lam=0.97)` is a common choice for monthly data. `rk.ewma_cov(r)` builds a whole covariance matrix the same way.

Example from 2020: at the end of May 2020, the 63-day rolling volatility of SPY still read 59%, because the March crash was still inside its window. The EWMA estimate had already fallen to 33%, reflecting the calmer market of May. EWMA reacts faster in both directions.

```python
prices = rk.get_prices("SPY", start="2005-01-01")
r = rk.prices_to_returns(prices, "D")["SPY"]

pd.DataFrame({
    "Rolling 63-day": rk.rolling_vol(r, 63),
    "EWMA (0.94)": rk.ewma_vol(r, 0.94),
}).plot(title="SPY volatility")
```

### Real-life applications

- **Volatility targeting:** many funds hold more stocks when EWMA volatility is low and fewer when it spikes, aiming for steady risk.
- **Risk dashboards:** banks and asset managers show rolling volatility, beta and correlation charts every morning.
- **Early warning:** a sudden jump in rolling correlations or EWMA volatility is often the first sign of market stress.
- **The VIX:** the "fear index" is the market's own forecast of the S&P 500's volatility over the next 30 days, derived from option prices.

## 13. Better VaR: parametric CVaR, Monte Carlo VaR and backtesting

Section 3 introduced three ways to estimate VaR. This section adds two more estimators and, most importantly, a way to check whether any VaR model actually works. On 20 years of real data, a standard VaR model failed that check.

### Parametric (Gaussian) CVaR

`rk.cvar_gaussian(r, level)` gives the expected loss beyond the VaR, assuming normal returns:

```latex
\text{CVaR} = -\left(\mu - \sigma\,\frac{\phi(z)}{\alpha}\right)
```

Here φ is the height of the bell curve at z, and α is the level as a decimal (0.05). Example: mean 1%, volatility 4% a month. The 5% Gaussian VaR is 5.6%, and the 5% Gaussian CVaR is −(0.01 − 0.04 × 0.1031 / 0.05) = 7.2%. Like Gaussian VaR, it understates risk when returns have fat tails.

### Monte Carlo VaR

`rk.var_monte_carlo(r, level, n_sims=100_000)` simulates many possible next periods and reads VaR and CVaR from them. Two methods:

- **Bootstrap** (default): draws past days at random, with replacement. It keeps real fat tails without assuming any distribution. For several assets it draws whole days together, so the correlations between assets are kept.
- **Normal:** draws from a normal distribution with the same means and covariances. Useful for comparison.

Monte Carlo becomes essential for portfolios with options or other non-linear positions, where simple formulas break down.

### Scaling VaR to longer horizons

`rk.scale_var(var, horizon)` uses the square-root-of-time rule: a 1-day VaR of 2% becomes about 2% × √10 = 6.3% over 10 days. Bank regulators historically used exactly this 10-day scaling. It assumes returns are independent from day to day, which is roughly true in calm markets and less so in crises.

### Backtesting: does the VaR model work?

A 5% VaR should be exceeded on about 5% of days. If losses beat it far more often, the model is underestimating risk. `rk.var_backtest(r, level, window, method)` checks this honestly:

1. On each day, estimate VaR using only the previous `window` days (for example 250, about one year).
2. Compare that forecast with the actual return the next day.
3. Mark a **breach** whenever the loss was bigger than the VaR.

`rk.kupiec_test(breaches, level)` then asks whether the number of breaches is consistent with the level, or too many (or too few) to be bad luck. It reports the expected and actual breaches and a p-value. A p-value below 0.05 means the model is rejected.

### A real result

The 40/20/30/10 portfolio from section 10, daily data from January 2005 to October 2025, 5% historic VaR on a rolling 250-day window:

|  | Value |
| --- | --- |
| Days tested | 4,990 |
| Expected breaches (5%) | 250 |
| Actual breaches | 292 (5.9%) |
| p-value | 0.007 |
| Model OK? | No |

The model was breached 17% more often than it promised. The reason is the volatility clustering from section 12: after a calm year, the 250-day window holds no bad days, so VaR is too low just as a crisis begins. A practical fix is to compute VaR on returns scaled by EWMA volatility, or simply to report CVaR alongside VaR.

```python
r = rep["portfolio"]                      # from rk.portfolio_report(...)
bt = rk.var_backtest(r, level=5, window=250, method="historic")
rk.kupiec_test(bt["Breach"], level=5)

ax = (-bt["VaR"]).plot(label="-VaR forecast")
bt["Return"].plot(ax=ax, alpha=0.4, label="Actual return")
```

### Real-life applications

- **Bank regulation:** under the Basel "traffic light" system, banks count their VaR breaches over the past 250 days. Too many and regulators force them to hold more capital.
- **Model validation:** risk teams at funds backtest every model before trusting it, and repeat the test regularly.
- **Dashboard honesty:** showing breaches next to the VaR line lets users judge for themselves how reliable the number is.

## 14. More portfolio construction: risk parity, maximum diversification, shrinkage, weight caps

Section 6 showed that the Maximum Sharpe Ratio portfolio depends heavily on expected returns, which are almost impossible to predict. The methods here avoid that problem: they use only risk (the covariance matrix), and they make the estimates and weights more stable.

### Money is not risk: risk contribution

`rk.risk_contribution(weights, cov)` shows what share of the portfolio's total risk comes from each asset. The shares add up to 100%. Because stocks are far more volatile than bonds, a portfolio can be balanced in money but completely unbalanced in risk:

&#91;embedded content: Yahoo Finance SPY, QQQ, AGG, GLD monthly returns, computed with rk.risk\_contribution · Jan 2005 to Oct 2025\]

The classic 60/40 portfolio is really a stock portfolio with a little bond seasoning: 92% of its ups and downs come from stocks.

### Risk parity (equal risk contribution)

`rk.risk_parity(cov)` finds weights where every asset contributes the same share of risk. Calm assets get more money, volatile assets less. With SPY, QQQ, AGG and GLD, that means about 56% in bonds and 12–17% in each of the others.

A simpler cousin, `rk.inverse_vol_weights(cov)`, sets weights in proportion to 1 / volatility. It ignores correlations but gets close to risk parity: 55% bonds here.

Risk parity portfolios have low volatility (6.7% a year here, against 9.5% for 60/40) and return less. Funds that use risk parity often borrow to scale the whole portfolio up to a target risk level.

### Maximum diversification

The **diversification ratio** (`rk.diversification_ratio(w, cov)`) is the weighted average volatility of the assets divided by the portfolio's volatility. It equals 1 when there is no diversification at all, and grows as diversification cuts risk. `rk.max_diversification(cov)` finds the weights with the highest ratio, favouring assets that move differently from the rest.

### Covariance shrinkage: better inputs

A covariance matrix estimated from a few years of data contains noise. With 30 assets there are 435 correlations to estimate, and the optimizer treats every random blip as real, leading to extreme weights. **Shrinkage** blends the noisy sample estimate with a simple, stable one:

```latex
\Sigma_{shrunk} = \delta\,\Sigma_{constant\ correlation} + (1-\delta)\,\Sigma_{sample}
```

`rk.cov_shrinkage(r, delta=0.5)` pulls every pair's correlation halfway towards the average correlation. δ = 0 is the raw sample; δ = 1 assumes all correlations are equal (`rk.cov_constant_correlation`). On the four ETFs, shrinking by half moved the MSR portfolio from 58% QQQ / 42% gold with nothing else, to 50% QQQ, 36% gold and 15% SPY.

### Weight caps

Every optimizer (`minimize_vol`, `msr`, `gmv`, `risk_parity`, `max_diversification`) now accepts `min_weight` and `max_weight`. For example, `rk.msr(0.03, er, cov, max_weight=0.4)` caps every holding at 40%. On the four ETFs this changed the MSR portfolio from two assets to four: 40% QQQ, 40% gold, 13% SPY and 7% bonds. Caps that cannot add up to 100% (five assets capped at 15%) raise a clear error.

### Comparing the methods

| Method | SPY | QQQ | AGG | GLD | Volatility | Needs return forecasts? |
| --- | --- | --- | --- | --- | --- | --- |
| 60/40 | 60% | 0% | 40% | 0% | 9.5% | No |
| Equal weight | 25% | 25% | 25% | 25% | 9.8% | No |
| Inverse volatility | 17% | 14% | 55% | 15% | 6.9% | No |
| Risk parity | 15% | 13% | 56% | 17% | 6.7% | No |
| Max diversification | 9% | 15% | 55% | 21% | 6.8% | No |
| Global min variance | 2% | 0% | 98% | 0% | 4.5% | No |
| Max Sharpe ratio | 0% | 58% | 0% | 42% | — | Yes |

Weights from the full-period covariance of monthly returns, January 2005 to October 2025. These are in-sample: section 15 tests the methods fairly, using only past data at each step.

### Real-life applications

- **Risk parity funds:** Bridgewater's All Weather fund popularized the idea; many pension funds now run risk-parity allocations.
- **Minimum volatility and equal-weight ETFs** apply GMV and 1/N to the stock market.
- **Investment policy limits:** real mandates nearly always cap single holdings (for example 5% per stock, 25% per sector), which is what weight caps model.
- **Shrinkage in practice:** Ledoit-Wolf shrinkage, a close relative of this method, is standard in quantitative asset management.

## 15. Rebalancing and transaction costs: backtesting realistically

A **backtest** replays a strategy on past data to see how it would have done. A good backtest copies what a real investor would face: weights drift as prices move, trading back to target costs money, and decisions can only use data available at the time. `rk.backtest_portfolio` does all three.

### Drift and rebalancing

Say you start with 60% stocks and 40% bonds. If stocks rise 20% and bonds stay flat, you now hold 64% stocks: your portfolio has quietly become riskier. **Rebalancing** means selling some of what went up and buying what went down to get back to 60/40.

Example: 10,000 split 6,000 stocks / 4,000 bonds. Stocks rise to 7,200. Total = 11,200, so stocks are 64.3%. To rebalance, sell 480 of stocks and buy 480 of bonds, giving 6,720 / 4,480 = 60/40 again.

### Turnover and costs

**Turnover** is the total of all weight changes when trading. In the example, stocks fell 4.3 percentage points and bonds rose 4.3, so turnover is 8.6%. Each trade costs something: commissions, the bid-ask spread and market impact. Costs are measured in **basis points** (bps): 1 bp = 0.01%, so 10 bps = 0.10% of the amount traded.

```latex
\text{Cost} = \text{Turnover} \times \frac{\text{cost\_bps}}{10{,}000}
```

### Using backtest\_portfolio

```python
m = rk.get_yahoo_returns(["SPY", "AGG"], start="2005-01-01", freq="M")

bt = rk.backtest_portfolio(m, [0.6, 0.4], rebalance="Q", cost_bps=10)
bt[["Return", "Wealth", "Turnover", "Cost"]]   # one row per month
bt.attrs["weights"]                               # weights actually held each month
```

`rebalance` can be `"D"`, `"W"`, `"M"`, `"Q"`, `"Y"` or `None` (buy and hold, never rebalance).

### How often to rebalance? A real test

60% SPY and 40% AGG, monthly data from January 2005 to October 2025:

| Rebalancing | Annual return | Turnover per year | Max drawdown | Stock weight at the end |
| --- | --- | --- | --- | --- |
| Monthly | 7.95% | 24% | −32.3% | 60% |
| Quarterly | 8.05% | 17% | −31.7% | 60% |
| Yearly | 8.11% | 11% | −30.4% | 62% |
| Never (buy and hold) | 8.87% | — | −30.8% | 87% |

Buy and hold earned the most because stocks beat bonds strongly over this period, but by the end it was an 87% stock portfolio: far riskier than the investor chose. Rebalancing is about **keeping the risk you signed up for**, not about maximizing return. For a simple ETF portfolio, costs are small: at 50 bps, monthly rebalancing lost only about 0.1% a year.

### Strategies that re-estimate: weighting functions

Instead of fixed weights, you can pass a **weighting function**. At each rebalance it receives only the previous `estimation_window` periods of returns and decides the new weights. This avoids **look-ahead bias**, the most common way backtests lie: using information you could not have had at the time.

Built-in weighting functions: `weight_ew`, `weight_inverse_vol`, `weight_gmv`, `weight_risk_parity` and `weight_max_div`. The last three accept `cov_estimator=rk.cov_shrinkage` for shrunk covariances. You can also write your own: any function that takes a DataFrame of returns and returns weights.

`rk.compare_strategies` runs several at once and adds an annual turnover column to the summary table:

```python
m = rk.get_yahoo_returns(["SPY", "QQQ", "AGG", "GLD"], start="2005-01-01", freq="M")

stats, rets = rk.compare_strategies(m, {
    "60/40 fixed": [0.6, 0, 0.4, 0],
    "Equal weight": rk.weight_ew,
    "Risk parity": rk.weight_risk_parity,
    "GMV (shrunk)": lambda r: rk.weight_gmv(r, cov_estimator=rk.cov_shrinkage, delta=0.5),
}, rebalance="Q", cost_bps=10, estimation_window=36)
```

Results, February 2008 to October 2025 (the first 36 months are used only for estimation):

| Strategy | Annual return | Volatility | Sharpe (rf 3%) | Max drawdown | Turnover per year |
| --- | --- | --- | --- | --- | --- |
| Equal weight | 10.4% | 10.2% | 0.70 | −23.4% | 17% |
| Risk parity | 7.6% | 6.8% | 0.66 | −17.0% | 22% |
| 60/40 fixed | 8.3% | 10.1% | 0.52 | −29.2% | 13% |
| GMV (shrunk) | 3.1% | 4.7% | 0.02 | −16.8% | 10% |

Equal weight did best on return and Sharpe, a common finding. Risk parity had the smallest loss of the high-Sharpe options. GMV went almost entirely into bonds and barely beat cash. One period and four ETFs prove nothing general: the point is that the toolkit lets you test such questions honestly.

### Real-life applications

- **Robo-advisors** rebalance automatically, usually when weights drift past a threshold such as 5%.
- **Pension funds** set rebalancing policies in their investment rules and account for trading costs.
- **Strategy research:** any new strategy should be backtested out of sample, with costs, before real money goes in. A strategy that only works with zero costs does not work.

## 16. Bonds and liability-driven investing

Many investors don't just want growth: they have to pay for something specific later, such as pensions, a mortgage, or school fees. **Liability-driven investing** (LDI) manages a portfolio against those future payments rather than against a market index. The key tools are present value, bond pricing and duration.

### Present value: money later is worth less today

If you can earn 4% a year, 1,000 today becomes 1,040 next year. Turned around, 1,040 next year is worth 1,000 today. **Discounting** turns a future payment into today's value:

```latex
PV = \frac{C}{(1+r)^t}
```

`rk.discount(t, r)` gives the value today of 1 paid in t years; `rk.pv(flows, r)` adds up the present value of a whole schedule of payments (a Series indexed by time in years).

### Funding ratio: can we pay what we owe?

`rk.funding_ratio(assets, liabilities, r)` divides assets by the present value of liabilities. Above 1 means fully funded.

Example: a small pension plan must pay 100,000 a year in years 10 to 14 and has 400,000 in assets. At a 4% interest rate the liabilities are worth 312,779 today, so the funding ratio is 1.28. If interest rates fall to 3%, the same promises are worth more today and the ratio drops to 1.14, without the assets changing at all. **Falling interest rates hurt pension plans**, which is exactly what happened to many plans in the 2010s.

### Bond prices move opposite to interest rates

A bond pays fixed coupons and returns its face value at maturity. `rk.bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)` lists its payments; `rk.bond_price(...)` discounts them.

A bond with face value 1,000 and a 5% coupon paid twice a year:

| Maturity | Price if rates are 5% | Price if rates rise to 6% | Change |
| --- | --- | --- | --- |
| 2 years | 1,000 | 981 | −1.9% |
| 10 years | 1,000 | 926 | −7.4% |
| 30 years | 1,000 | 862 | −13.8% |

When rates rise, existing bonds paying the old, lower coupon become less attractive, so their price falls. The longer the bond, the bigger the fall. This is why bond funds lost heavily in 2022 when rates jumped.

### Duration: one number for interest-rate risk

**Macaulay duration** (`rk.macaulay_duration(flows, rate)`) is the average time until you receive your money, weighting each payment by its present value. A 10-year bond with a 5% annual coupon has a duration of about 8.1 years: shorter than 10, because the coupons arrive early.

Duration is also a rule of thumb for risk: a bond's price changes by roughly duration × the change in rates. Duration 8 means about −8% if rates rise 1 percentage point.

### Duration matching: immunizing a liability

If your assets have the same duration as your liabilities, a change in interest rates moves both by about the same amount, so the funding ratio stays stable. `rk.match_durations(target, short_bond, long_bond, rate)` gives the share to put in a short bond so that a short/long mix matches the target's duration.

Example at 4%: a liability due in 10 years (duration 10), a 5-year bond (duration 4.6) and a 20-year bond (duration 13.5). Put 39% in the short bond and 61% in the long bond: 0.39 × 4.6 + 0.61 × 13.5 = 10.

### Simulating interest rates: the CIR model

To test a liability-driven plan, you need many possible paths of interest rates. `rk.cir(n_years, n_scenarios, a, b, sigma, r_0)` uses the **Cox-Ingersoll-Ross** model, in which rates are pulled back towards a long-run level b at speed a, with random shocks that shrink as rates approach zero, so rates stay positive. It returns the simulated rates and the price of a zero-coupon bond along each path.

With a long-run rate of 3%, speed 0.05 and volatility 0.05, 1,000 simulated paths ended 10 years later anywhere from 0.3% (5th percentile) to 7.8% (95th percentile), with a median of 2.5%.

```python
rates, zc_prices = rk.cir(n_years=10, n_scenarios=1000, a=0.05, b=0.03, sigma=0.05, r_0=0.03, seed=0)
rates.iloc[:, :20].plot(legend=False, title="20 simulated interest-rate paths")
```

### Real-life applications

- **Pension funds and insurers** run LDI programs that hold long bonds to match the duration of their promises.
- **The 2022 UK gilt crisis:** many UK pension LDI funds had borrowed to boost their duration. When gilt yields jumped, they had to sell bonds quickly to meet collateral calls, and the Bank of England stepped in.
- **Personal goals:** to pay a known bill in 7 years, a bond (or bond fund) with a duration of about 7 years largely removes interest-rate risk.
- **Banks** measure the duration gap between their loans and deposits to manage interest-rate risk.

## 17. Stress testing: replaying historical crises

VaR and volatility describe normal times. A **stress test** asks a blunter question: if a known disaster happened again, what would it do to the portfolio I hold today? `rk.stress_test` replays real crises on your current weights.

&#91;embedded content: Yahoo Finance daily adjusted prices, computed with rk.stress\_test · six crises, 2007 to 2022\]

In five crises out of six, holding bonds and gold roughly halved the damage. In 2022, inflation and rising rates pushed stocks and bonds down together, so diversification failed: the portfolio lost 21.5% against 24.1% for stocks. A stress test reveals this kind of weakness; a single correlation number would not.

### Historical scenarios

`rk.CRISES` lists seven peak-to-trough windows of the US stock market, from the dot-com crash (March 2000 to October 2002) to the 2022 rate shock. `rk.stress_test(daily_returns, weights)` reports for each one:

- the portfolio's total return over the window,
- its maximum drawdown within the window,
- each asset's own return,
- `Covered`: whether the data fully covers the window. QQQ and SPY existed in 2000, but AGG (launched 2003) and GLD (2004) did not, so a portfolio holding them cannot be tested on the dot-com crash. Uncovered rows are flagged rather than silently computed on partial data.

You can add your own scenarios:

```python
prices = rk.get_prices(["SPY", "QQQ", "AGG", "GLD"], start="2005-01-01")
daily = rk.prices_to_returns(prices, "D")
weights = [0.4, 0.2, 0.3, 0.1]

my_scenarios = {**rk.CRISES, "Aug 2024 yen carry unwind": ("2024-07-16", "2024-08-05")}
rk.stress_test(daily, weights, my_scenarios)
```

Stress tests should use **daily** data: a crash like March 2020 lasted only five weeks, and monthly data would blur it.

### Hypothetical scenarios

History never repeats exactly. `rk.shock_test(weights, shocks)` applies made-up moves to each asset and shows the portfolio result and each asset's contribution:

```python
rk.shock_test({"Stocks": 0.6, "Bonds": 0.4}, {"Stocks": -0.30, "Bonds": 0.05})
# Portfolio Return -0.16: stocks contribute -18%, bonds +2%
```

Useful hypothetical scenarios: stocks −30% with bonds +5% (a classic recession), stocks −20% with bonds −10% (an inflation shock like 2022), or a single holding −50% (a company-specific disaster).

### Real-life applications

- **Bank regulation:** the US Federal Reserve and the European Central Bank run yearly stress tests with severe hypothetical scenarios. Banks that fail must hold more capital or cut dividends.
- **Pension and insurance oversight:** regulators require funds to show they could survive set market shocks.
- **Personal planning:** "could I live with a 30% loss like in 2020?" is a better question for most people than "what is my VaR?"
- **Dashboards:** a stress-test table is one of the most intuitive risk views for non-specialists, because everyone remembers 2008 and 2020.

## 18. From toolkit to dashboard

The toolkit is built so a dashboard can stay simple: one function call returns everything to display, and all calculations stay in `risk_kit.py`, where they are tested.

&#91;embedded content: architecture of a risk dashboard built on risk\_kit.py\]

The user picks holdings and dates; the dashboard passes them to risk\_kit and draws what comes back. Keeping calculations out of the dashboard code means every number on screen comes from a tested function.

### One call: portfolio\_report

```python
report = rk.portfolio_report(
    tickers=["SPY", "QQQ", "AGG", "GLD"],
    weights=[0.4, 0.2, 0.3, 0.1],
    benchmark="SPY",
    start="2010-01-01",
    freq="D",              # or "M" for monthly statistics
    riskfree_rate=0.03,
)
```

It returns a dictionary. Each key maps naturally to one panel of the dashboard:

| Key | Contents | Suggested panel |
| --- | --- | --- |
| `summary` | Every statistic from section 10, per asset, portfolio and benchmark | Headline tiles and a comparison table |
| `wealth` | Growth of 1 invested in the portfolio | Line chart |
| `drawdown` | Drawdown over time | Shaded area chart under the wealth chart |
| `rolling_vol`, `ewma_vol` | Risk over time | Line chart |
| `risk_contribution` | Share of risk from each holding | Bar chart next to the weights |
| `correlation` | Correlation matrix of holdings | Heatmap |
| `capm`, `tracking_error`, `information_ratio` | Benchmark comparison | Table |
| `var_backtest`, `kupiec` | VaR line, actual returns, breaches | Chart with breaches marked, plus a pass/fail badge |
| `stress` | Historical crisis replay | Table or bar chart |
| `weights`, `prices`, `returns`, `portfolio` | The inputs and raw series | Download buttons |

### Choosing a framework

- **Streamlit:** the fastest way to turn Python into a web app. Widgets for picking tickers and weights take a few lines, and `st.cache_data` caches results. Best for a first version.
- **Plotly Dash:** more control over layout and interactions; better for a polished multi-page product.
- **A web front end (React) with a Python API (FastAPI):** most flexible and most work; worth it when the dashboard is a product with user accounts.

A minimal Streamlit version:

```python
# app.py  -- run with: streamlit run app.py
import streamlit as st
import risk_kit as rk

st.title("Portfolio risk dashboard")
tickers = st.text_input("Tickers", "SPY, QQQ, AGG, GLD").replace(" ", "").split(",")
weights = [st.number_input(t, 0.0, 1.0, 1/len(tickers)) for t in tickers]

@st.cache_data(ttl=3600)
def load(tickers, weights):
    return rk.portfolio_report(tickers, weights, benchmark="SPY", start="2010-01-01")

rep = load(tuple(tickers), tuple(weights))
cols = st.columns(4)
s = rep["summary"].loc["Portfolio"]
cols[0].metric("Annual return", f"{s['Annualized Return']:.1%}")
cols[1].metric("Volatility", f"{s['Annualized Vol']:.1%}")
cols[2].metric("Sharpe", f"{s['Sharpe Ratio']:.2f}")
cols[3].metric("Max drawdown", f"{s['Max Drawdown']:.1%}")
st.line_chart(rep["wealth"])
st.area_chart(rep["drawdown"])
st.dataframe(rep["stress"])
```

### Before you ship: a checklist

- [ ] **Data licence.** `yfinance` reads Yahoo's website unofficially; Yahoo's terms limit commercial use and the library can break without warning. For a paid product, budget for a licensed provider (for example Polygon, Tiingo, EOD Historical Data or Alpha Vantage) and swap it in inside `get_prices` only.
- [ ] **Disclaimer.** State clearly that the dashboard shows historical analysis, not investment advice, and that past performance does not predict future results. Check the rules where you sell: giving personalized advice can require a licence.
- [ ] **Input checks.** Weights that don't add up to 1, unknown tickers and very short histories should produce a clear message, not a crash. `portfolio_report` normalizes weights and drops tickers with no data; surface those warnings to the user.
- [ ] **Explain every number.** Beginners need a one-line tooltip per metric. The glossary in section 19 is a starting point.
- [ ] **Show data coverage.** Display the date range actually used and flag stress scenarios marked `Covered = False`.
- [ ] **Performance.** Cache downloads (built in) and results (`st.cache_data`). Optimizers and VaR backtests on daily data take seconds; run them only when inputs change.
- [ ] **Tests.** Run `python -m pytest` before every release. The suite has 32 tests covering every module.
- [ ] **Pin versions.** Use the `requirements.txt` in the folder, and pin exact versions in production so a library update can't change your numbers.

## 19. Function reference, glossary and fixes

### Every function in risk\_kit.py

| Function | What it does | Section |
| --- | --- | --- |
| `get_prices(tickers, start, end)` | Daily adjusted prices from Yahoo Finance, cached 24 hours | 9 |
| `prices_to_returns(prices, freq)` | Daily, weekly, monthly, quarterly or yearly returns | 9 |
| `get_yahoo_returns(tickers, start, end, freq)` | Download and convert in one call | 9 |
| `infer_periods_per_year(r)` | 252, 52, 12, 4 or 1 from the dates | 9 |
| `clean_returns(r)` / `load_returns_csv(path)` | Data checks / load your own CSV | 9 |
| `get_returns()`, `get_hfi_returns()`, `get_ind_returns()`, `get_ind_size()`, `get_ind_nfirms()` | EDHEC course datasets from the `data/` folder | 1–4 |
| `drawdown(r)` | Wealth index, previous peak and drawdown | 1 |
| `max_drawdown(r)` | Worst peak-to-trough loss | 10 |
| `drawdown_duration(r)` | Longest and current time underwater | 10 |
| `semideviation(r)`, `skewness(r)`, `kurtosis(r)`, `is_normal(r)` | Shape of the return distribution | 2 |
| `var_historic`, `var_gaussian`, `var_cornish_fisher`, `cvar_historic` | VaR and CVaR, three ways | 3 |
| `cvar_gaussian(r, level)` | CVaR assuming normal returns | 13 |
| `var_monte_carlo(r, level, method)` | VaR and CVaR by simulation | 13 |
| `scale_var(var, horizon)` | Square-root-of-time scaling | 13 |
| `var_backtest(r, level, window)` / `kupiec_test(breaches)` | Does the VaR model hold up? | 13 |
| `annualize_rets`, `annualize_vol`, `sharpe_ratio` | Yearly figures and Sharpe ratio | 4 |
| `downside_deviation`, `sortino_ratio`, `calmar_ratio` | Downside-focused ratios | 10 |
| `summary_stats(r, riskfree_rate)` | Every key statistic in one table | 10 |
| `beta`, `capm`, `capm_expected_return` | Market sensitivity and alpha | 11 |
| `tracking_error`, `information_ratio`, `up_down_capture` | Performance against a benchmark | 11 |
| `rolling_vol`, `rolling_sharpe`, `rolling_beta`, `rolling_corr`, `rolling_max_drawdown` | Statistics over a moving window | 12 |
| `ewma_vol(r, lam)` / `ewma_cov(r, lam)` | Volatility and covariance weighting recent data more | 12 |
| `portfolio_returns`, `portfolio_vol`, `portfolio_return_series` | Portfolio return and risk | 5 |
| `risk_contribution(w, cov)` | Share of risk from each asset | 14 |
| `cov_sample`, `cov_constant_correlation`, `cov_shrinkage` | Covariance estimators | 14 |
| `plot_ef2`, `plot_ef`, `efficient_frontier` | Efficient frontier as a plot or as data | 6 |
| `minimize_vol`, `optimal_weights`, `msr`, `gmv` | Mean-variance optimizers, with weight caps | 6, 14 |
| `equal_weights`, `inverse_vol_weights`, `risk_parity`, `max_diversification`, `diversification_ratio` | Risk-based portfolios | 14 |
| `weight_ew`, `weight_inverse_vol`, `weight_gmv`, `weight_risk_parity`, `weight_max_div` | Weighting functions for backtests | 15 |
| `backtest_portfolio(r, weights, rebalance, cost_bps)` | Backtest with drift, rebalancing and costs | 15 |
| `compare_strategies(r, strategies)` | Several backtests side by side | 15 |
| `run_cppi(risky_r, ...)` | CPPI backtest | 7 |
| `gbm(...)` | Monte Carlo price paths | 8 |
| `cir(...)`, `inst_to_ann`, `ann_to_inst` | Interest-rate simulation and rate conversions | 16 |
| `discount`, `pv`, `funding_ratio` | Present values and funding | 16 |
| `bond_cash_flows`, `bond_price`, `macaulay_duration`, `match_durations` | Bond maths and immunization | 16 |
| `CRISES`, `stress_test(r, weights)`, `shock_test(weights, shocks)` | Historical and hypothetical stress tests | 17 |
| `portfolio_report(tickers, weights, benchmark)` | Everything a dashboard needs, in one dictionary | 18 |

### Glossary

- **Return:** percentage gain or loss over a period, as a decimal (0.05 = 5%).
- **Volatility:** standard deviation of returns; the usual measure of risk.
- **Drawdown:** fall from the highest value reached so far.
- **Skewness:** whether surprises are mostly on the loss side (negative) or gain side (positive).
- **Kurtosis:** how often extreme returns happen; above 3 means fat tails.
- **VaR:** the loss that is exceeded only in the worst x% of periods.
- **CVaR / Expected Shortfall:** the average loss in those worst x% of periods.
- **Risk-free rate:** return on a safe asset such as a Treasury bill.
- **Sharpe ratio:** excess return over the risk-free rate, divided by volatility.
- **Correlation:** how closely two assets move together, from −1 to +1.
- **Covariance matrix:** table of how every pair of assets moves together.
- **Efficient frontier:** best return available for each level of risk.
- **GMV:** the lowest-risk portfolio on the frontier.
- **MSR / tangency portfolio:** the highest-Sharpe portfolio on the frontier.
- **Capital market line:** mixes of cash and the MSR portfolio.
- **CPPI:** a rule that shifts between risky and safe assets to protect a floor.
- **Multiplier (m):** how aggressively CPPI invests the cushion.
- **Monte Carlo:** estimating outcomes by simulating many random futures.
- **GBM:** the standard random-walk model of stock prices.

* **Adjusted price:** a price corrected for dividends and splits, so its changes are total returns.
* **Sortino ratio:** like the Sharpe ratio, but counting only downside volatility.
* **Calmar ratio:** annual return divided by the maximum drawdown.
* **Benchmark:** the simple alternative an investment is compared with, such as the S&P 500.
* **Beta:** how much an asset moves when the market moves 1%.
* **Alpha:** return above what beta explains.
* **Tracking error:** volatility of the difference from the benchmark.
* **Information ratio:** outperformance per unit of tracking error.
* **EWMA:** an average that weights recent data more heavily.
* **VaR backtest / breach:** checking VaR forecasts against what happened; a breach is a loss bigger than the VaR.
* **Risk contribution:** the share of portfolio risk coming from one asset.
* **Risk parity:** weights that give every asset an equal share of risk.
* **Shrinkage:** blending a noisy estimate with a simpler, stable one.
* **Rebalancing / turnover:** trading back to target weights / the total amount traded.
* **Basis point (bp):** 0.01%.
* **Look-ahead bias:** a backtest using information that was not available at the time.
* **Present value:** what a future payment is worth today.
* **Duration:** average time to a bond's payments; also its sensitivity to interest rates.
* **Funding ratio:** assets divided by the present value of liabilities.
* **Stress test:** replaying a crisis or a made-up shock on a portfolio.

### Fixes made to the original script

The file was renamed from `edhec_risk_kit (1).py` to `risk_kit.py`, and these bugs were fixed:

| Problem | Effect before | Fix |
| --- | --- | --- |
| `get_returns()` had no `return` | Always returned `None` | Returns the DataFrame, with a monthly period index |
| Hard-coded Windows paths (two different usernames) | Every data loader failed on Mac | All loaders read from a `data/` folder next to the script |
| `plot_ef` hard-coded `rf=0.1` for the MSR point | Capital market line drawn wrong unless the rate was 10% | Uses the `riskfree_rate` argument |
| `plot_ef2` checked `er` twice | A wrong-sized covariance matrix was not caught | Checks `er` and `cov` |
| `run_cppi` used the key `"floor"` twice | The floor parameter was silently overwritten | Floor history is now `"floor_value"` |
| `drawdown()` drew three plots on every call | Unwanted charts in calculations | Pure calculation; plot the result yourself |
| Duplicate `drawdown1` and repeated imports | Clutter | Removed; imports grouped at the top |
| `is_normal` accepted only a Series | Errors on a DataFrame | Works per column on a DataFrame |
| Hedge fund dates read as month-first | Possible wrong dates | Read day-first, matching the EDHEC file |

The toolkit was then extended with everything in sections 9–18. The test suite (`tests/test_risk_kit.py`, 32 tests) checks every module against known answers on simulated data, and the Yahoo Finance functions were run live on 20 years of real prices. The EDHEC course loaders were not run, because those CSV files are not yet in the `data/` folder.
