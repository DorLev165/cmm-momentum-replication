# CMM Replication — Issues and Remediation Plan

This document catalogs the gaps between the current implementation in this
repository and the methodology of Beckmeyer & Wiedemann (2025),
*"All Days Are Not Created Equal: Understanding Momentum by Learning to
Weight Past Returns"* (Journal of Banking and Finance 181, 107565).

Each section states:

- **What the paper does**
- **What this codebase does**
- **Why it matters**
- **What needs to change** (design-level; implementation is left to you)

Items are ordered by severity. The first four are blockers — without fixing
them, the strategy cannot reproduce the paper's phenomenon (sparse learned
weights, underreaction-driven profitability, crash resistance).

---

## 1. Universe is survivorship-biased (⚠️ BLOCKER)

### What the paper does
Uses the full CRSP universe of U.S. common stocks from 1973–2022 with
**point-in-time** membership. Every stock that was ever listed appears in
the cross-section during the months it was actually trading; delistings,
mergers, and bankruptcies are reflected.

### What this codebase does
`cmm/fetch_data.py::get_sp500_tickers` scrapes **today's** S&P 500
constituents from Wikipedia and applies that membership list back to 1990.

### Why it matters
- Every stock that was dropped from the index between 1990 and today is
  **silently excluded**. This is pure survivorship bias.
- The universe is ~500 mega-caps. The paper sorts the entire cross-section
  (≈3,000–5,000 stocks per month) into deciles using NYSE breakpoints. With
  a 500-stock universe, NYSE breakpoints are meaningless and the decile
  portfolios are tiny.
- Momentum's short leg depends critically on bad-news stocks. Those are the
  ones most likely to have been dropped from the index — i.e., the exact
  stocks your universe is missing.

### What needs to change
- **Data source**: CRSP (through WRDS), or a CRSP-equivalent that provides
  point-in-time listings with delisting returns. `yfinance` is not suitable
  for this purpose — it has no delisted-ticker history.
- **Membership**: For each month-end `t`, the investable universe is every
  stock with valid price and characteristic data at `t`, **not** a
  static ticker list.
- **Delisting returns**: CRSP provides `DLRET`; paper implicitly uses these.
  Without them, performance-by-decile is biased.
- If CRSP is unavailable, at minimum use a historical S&P 1500 or Russell
  3000 constituents file with entry/exit dates, and accept the remaining
  bias as a known limitation.

---

## 2. Only 11 price-based characteristics vs. 153 (⚠️ BLOCKER)

### What the paper does
Uses the **153 firm characteristics from Jensen, Kelly & Pedersen (2023)**
— a mix of accounting ratios (book-to-market, profitability, investment,
accruals, leverage, etc.) and market-based signals (past returns at
multiple horizons, volatility, liquidity, beta, etc.). These are fed
through a 3-hidden-layer FFN to produce the scalar `ẑ`.

The paper's own Table 15 shows that a **returns-only CMM does not beat
standard momentum**. Accounting data is necessary for the strategy to
work; accounting *alone* is more informative than market data alone; and
the combination is required for the published results.

### What this codebase does
`cmm/fetch_data.py::_compute_chars_simple` produces 11 characteristics,
**all derived from past prices**: log price, multi-horizon cumulative
returns, multi-horizon volatility, skewness, drawdown, 5-day reversal.

### Why it matters
This is structurally the same as the paper's "Ret. CMM" ablation. It
cannot outperform standard momentum — the paper tells you this
explicitly. The FFN has nothing to learn from beyond what's already in
the 231-day return vector, so `ẑ` collapses and the softmax weights stay
near-uniform (see §4).

### What needs to change
- **Source**: the JKP characteristic dataset is publicly posted on Bryan
  Kelly's / Theis Jensen's website as monthly panel data. Download once,
  merge on `(permno, yyyymm)`.
- **At minimum**, the implementation needs accounting ratios:
  book-to-market, operating profitability, asset growth (investment),
  gross profits-to-assets, accruals, net share issuance, ROE.
- **Point-in-time discipline**: accounting data must be lagged by the
  paper's convention — typically the fiscal year ending before June of
  year `t` is assumed known for July `t` onward. `yfinance.info` returns
  *current* values and is therefore unusable.
- **Preprocessing**: the JKP dataset is already rank-transformed into
  `[-0.5, 0.5]` per month. If you use it, don't re-standardize those
  columns; but see §4 for what you should standardize and what you
  shouldn't.

---

## 3. "Size" is price, not market cap (⚠️ BLOCKER for portfolio weights)

### What the paper does
Value-weights stocks within each decile by **market capitalization**
(shares outstanding × price). This is the standard definition in the
Fama–French / CRSP factor literature.

### What this codebase does
`cmm/fetch_data.py` lines 189, 205, 220: stores `p0 = float(px.iloc[me_loc])`
— the last close price — as `size`. `cmm/portfolio.py::value_weighted_return`
then uses this price as the weight.

### Why it matters
Share price has no economic meaning across firms (it's an artifact of
splits and IPO choices). A $700,000 Berkshire A share would dominate a
decile; a $5 stock with a $50B market cap would be near-zero weight.
Even within the S&P 500 this distorts weights meaningfully — and for
a cross-sectional sort into deciles it corrupts the HML return you
report.

### What needs to change
- Fetch **shares outstanding** as a panel variable (CRSP: `SHROUT`;
  Compustat: `cshoq`; JKP: already provides `me` = market equity).
- `size = shares_outstanding × price` at the portfolio formation date
  (end of month `t`).
- Rename the parameter throughout (`size` → `market_cap`) to prevent
  this bug from recurring.

---

## 4. FFN sees standardized returns, not raw returns (⚠️ BLOCKER — silent bug)

### What the paper does
The FFN takes firm characteristics `z` as input and outputs a scalar `ẑ`.
The importance score is `Score_{t-d} = ẑ · r_{t-d}` where `r` is the
**raw daily log return**. The softmax and subsequent weighted sum
`E_CMM = Σ w · r` also use raw `r`. Only the characteristics are
standardized / rank-transformed.

### What this codebase does
Two layers of standardization get applied to the returns:

1. `cmm/model.py::CMMModel.fit` lines 86–88: creates a `StandardScaler()`
   and calls `fit_transform(X)` on the **entire** concatenated
   `[chars, returns]` matrix. The 231 daily-return columns are
   z-scored across the training sample.
2. The standardized `X` is then passed to `CMMFFNWrapper` with
   `scale_features=False`, so it isn't rescaled again — but the damage
   is done.

Inside `cmm/ffn.py::CMMModule.forward`:

```python
r = x[:, -self.n_ret:]      # these are STANDARDIZED returns, not raw
scores = z.unsqueeze(-1) * r
weights = torch.softmax(scores, dim=-1)
e_cmm = (weights * r).sum(dim=-1)   # also standardized returns
```

So `E_CMM` is a weighted sum of standardized returns, not raw returns.
The cross-sectional ordering it produces is not the same as
`Σ w · r_raw`, and more importantly **the economic interpretation of
`w` (weights on actual daily returns of the stock) is broken**.

### Why it matters
- The FFN's job is to learn when `|r|` is informative. Standardizing
  `r` across the training sample destroys the magnitude signal —
  a "large return" for a low-vol stock becomes indistinguishable from
  a "medium return" for a high-vol stock.
- The paper's Section 4.1 test — correlation between `|r|` and `w` to
  diagnose under- vs. over-reaction — becomes meaningless because
  `|r|` has been rescaled.
- Empirically, your own plots in `weights_train_*.png` show weights
  ranging ~0.0042–0.0058 around the equal-weight baseline of
  `1/231 ≈ 0.00433`. The paper reports that two of 231 days capture
  30% of total weight. Your softmax is near-uniform because `ẑ · r`
  is tiny when `r` is z-scored. **This is the single clearest symptom
  that the strategy is not CMM.**

### What needs to change
Separate the preprocessing of the two feature blocks:

- **Characteristics `z`**: standardize or rank-transform across the
  training sample, as the paper does. Cache the scaler fit on training
  data; apply to validation and test unchanged.
- **Daily returns `r`**: pass through **unchanged**. No scaling, no
  clipping (beyond basic NaN handling).
- In `CMMModule.forward`, the last `n_ret` columns must be raw
  returns at all times.

Concretely, the scaler in `CMMModel.fit` should either:

- Only touch columns `[0:n_char]`, or
- Be moved entirely to the data preparation stage and never applied
  to the return block.

After the fix, you should verify that `ẑ.abs().mean()` is O(1) or
larger, not O(0.01), and that the softmax weights show the sparsity
pattern from the paper's Figure 2.

---

## 5. Training window is too short

### What the paper does
First training window: 1973–1982 (10 years) → test 1983–1984. Expands
forward every 2 years. By the time predictions reach 2022, the model
has been trained on ~40 years of panel data across ~5,000 stocks per
month.

### What this codebase does
`main.py` sets `start_date="1990-01-01"` and
`initial_train_end_year=1999`. After warmup (252 + 22 trading days), the
first usable training month is ~January 1991, giving ~9 years × 500
stocks ≈ 50,000 stock-months for the first fit. Paper's first fit has
roughly 10× that.

### Why it matters
Neural networks with ~100k parameters (your architecture:
`(153+231)→256→128→64→1` ≈ 130k params, and you've shrunk the input to
`11+231=242` so it's slightly fewer but still large) need a lot of
training data to avoid overfitting `ẑ`. On 50k samples with 30 epochs
you are almost certainly in a noisy-fit regime, and the subsequent
flat-weights output (§4) may be partly the FFN collapsing to a
near-constant output because it can't learn anything robust.

### What needs to change
- Start training in 1973 (or whatever the earliest date the JKP panel
  supports — typically 1952 for U.S. equities).
- Keep the paper's 2-year test windows and 2-year expansion cadence.
- The warmup period (first ~13 months of returns needed for the 252-day
  formation window) should come *before* the training start — i.e., load
  data from 1972 to build features for 1973.

---

## 6. Breakpoints use today's NYSE listings

### What the paper does
NYSE breakpoints for decile sorting are computed each month from the
**stocks listed on NYSE in that month** (CRSP `exchcd == 1`). Membership
changes over time.

### What this codebase does
`cmm/portfolio.py::get_nyse_tickers` fetches a GitHub-hosted CSV of
*current* NYSE listings and uses that set for every month.

### Why it matters
- Many stocks that were NYSE-listed in 1995 are now delisted, merged,
  or moved to Nasdaq. They vanish from your breakpoint calculation.
- Breakpoints shift upward over time (inflation, survivorship), but
  your set doesn't reflect the 1990s composition.
- Secondary concern compared to the universe issue, but it compounds
  the survivorship problem from §1.

### What needs to change
- Use CRSP's `exchcd` field per month.
- If working off a constituents file, ensure it includes exchange at
  time `t`, not current exchange.

---

## 7. `yfinance` is the wrong data source for a backtest

### Why it's a problem
- No delisted tickers → survivorship bias (§1).
- `auto_adjust=True` produces prices adjusted for splits and dividends,
  which is fine for daily returns. But the same endpoint does not
  provide **shares outstanding** historically, which you need for
  market cap (§3).
- `.info` fields are current snapshots, so accounting data is not
  point-in-time (you've already flagged this in the code comments —
  good).
- Rate limits and silent failures on individual tickers can produce
  panels with holes that are hard to audit.

### What needs to change
- Primary recommendation: **WRDS/CRSP + Compustat** if you have
  academic access, or the **JKP panel** (Jensen–Kelly–Pedersen
  replication dataset on Bryan Kelly's website) which already packages
  everything needed.
- Backup: a commercial provider with point-in-time delisted data
  (Sharadar, Norgate, Polygon's historical with delisted flag).
- Keep the daily return pipeline if you like, but source
  characteristics and market cap from a proper panel dataset.

---

## 8. Minor methodology gaps

These won't block replication but will make your results not quite
comparable to the paper's numbers.

### 8.1 Portfolio is high-minus-low **decile**, paper uses deciles
You're already doing this correctly; just flagging that the table at the
end of `main.py` reports "mean IC" rather than Sharpe. IC is useful but
not the paper's headline metric — make sure your output prioritizes the
Sharpe and MDD of the HML return series, which is what the paper reports
in Table 1.

### 8.2 No transaction cost analysis
Paper's Table 8 shows CMM survives after bid-ask spreads. You could
either skip this (clearly label "gross of costs") or add a half-spread
deduction. Without it, don't claim the strategy "works" in any practical
sense.

### 8.3 Validation set is a time-suffix of training
`train_val_split_for_window` takes the last 20% of training rows by row
order. If the input is already sorted by (date, stock), this is a proper
time-series validation split. If it's not (and I don't think you
guarantee this), validation leaks. Either sort by date before splitting
or split by unique date as `cmm/data.py::train_val_split_by_time`
already implements — then use the latter everywhere.

### 8.4 No early stopping / no validation-set model selection
Training runs for a fixed 30 epochs regardless of validation loss.
Paper uses validation for early stopping (standard Gu-Kelly-Xiu
practice). Fixed epochs means you may be overfit or underfit depending
on the window.

### 8.5 Refitting cadence
Paper refits every 2 years. Your `shift_years=2` matches. Good. Just
confirm that within a 2-year test period you are *not* updating
weights — one fit per window only. (Looking at `training.py`, this
appears correct.)

---

## Suggested remediation order

1. **Fix the scaler bug (§4)** first — it's a 10-line change and will
   immediately show whether your existing pipeline can produce
   non-uniform weights on the current data. Even with the wrong universe
   and thin characteristics, after this fix the learned weights should
   be visibly sparser than equal-weight.
2. **Replace `size` with market cap (§3)** — small change, large
   correctness impact.
3. **Switch data source** to CRSP + JKP characteristics (§1, §2, §6,
   §7). This is the bulk of the work and where the real performance
   unlock is.
4. **Extend training history to 1973** (§5).
5. **Add early stopping + validate the train/val split is time-based**
   (§8.3, §8.4).
6. **Add transaction cost layer** (§8.2) before claiming practical
   viability.

After steps 1–4, compare your results to Table 1 of the paper:
- Target: annualized return ~18.5%, Sharpe ~1.47, MDD ~−27.6% over
  1983–2022.
- If you're still well below this after fixing the universe, the scaler,
  and the characteristics, check (in order):
  - Is `ẑ` producing meaningful magnitudes? (`ẑ.abs().mean() > 1`?)
  - Do the softmax weights pass the Figure 2 sanity check (top 2 days
    ≈ 30% of mass)?
  - Is the correlation between `|r|` and `w` positive for 60–90% of
    stock-months? (Paper's Figure 3.)
  - Is the model trained long enough? (Adam loss should plateau.)

---

## What is already correct and should not be changed

- `CMMModule` architecture in `cmm/ffn.py` — matches the paper exactly.
- Softmax boundary conditions and the tests in
  `tests/test_softmax_weights.py`.
- Cross-sectional normalization of the target
  (`cmm/data.py::cross_sectional_normalize`).
- Expanding window protocol with 2-year shifts
  (`cmm/training.py::expanding_window_splits`).
- Stock-level dollar-neutral HML construction in
  `cmm/portfolio.py::build_hml_portfolio` — this is more rigorous than
  most replications.
- GELU activation (paper footnote 2 specifies GELU).
- Decision to comment out `yfinance.info` fundamentals for look-ahead
  reasons — correct instinct, keep it.

These are the bones of a correct implementation. The flesh — data,
characteristics, and the scaling discipline — is where the replication
breaks.
