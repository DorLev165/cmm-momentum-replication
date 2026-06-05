"""
One-shot: populate data/spx/spx_monthly.csv from CRSP's monthly market
index (crsp.msi via WRDS). Fast (~10 seconds) and reliable — bypasses
yfinance entirely.

Rationale
---------
The CMM pipeline uses a trailing-market-return signal for regime
classification and D interactions. yfinance is the default source but
has been observed to hang for 15+ minutes on single-ticker requests.
Running this script once populates the disk cache that
`cmm.regime.fetch_market_returns` reads first, so the main pipeline
never calls yfinance.

Data source: CRSP `crsp.msi` table, column `vwretd` — the CRSP
value-weighted market return including dividends. This is the
academic industry standard "market return" and the basis of the
Fama-French MKT factor. Virtually every paper using CRSP market
returns (Daniel-Moskowitz 2016, Moreira-Muir 2017, and similar
momentum-crash literature) uses this series.

Internal consistency: the CMM strategy trades the full CRSP universe
(~3–5k stocks/month), so vwretd — computed on that same universe —
is the right regime signal. Using SPX (`sprtrn`, top-500 only) would
create a mismatch between "what we trade" and "how we classify
regime." We pull `sprtrn` alongside but only use it as a last-resort
fallback for months where vwretd is missing.

Usage
-----
    cd "C:\\Projects\\Momentum trading"
    python scripts/cache_spx_from_wrds.py

After this runs, `python main.py` will load regime signal from the
cache in <1 second instead of calling yfinance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wrds


OUT = Path(__file__).resolve().parent.parent / "data" / "spx" / "spx_monthly.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("Connecting to WRDS...")
    conn = wrds.Connection()

    print("Querying crsp.msi for monthly market returns (vwretd, academic standard)...")
    df = conn.raw_sql(
        """
        SELECT date, vwretd, sprtrn
        FROM crsp.msi
        WHERE date >= '1970-01-01'
          AND (vwretd IS NOT NULL OR sprtrn IS NOT NULL)
        ORDER BY date
        """,
        date_cols=["date"],
    )
    conn.close()

    if df is None or len(df) == 0:
        print("ERROR: CRSP returned no rows. Check your WRDS/CRSP access.")
        sys.exit(1)

    # Use vwretd (CRSP value-weighted market return, academic standard).
    # Fall back to sprtrn (S&P 500 total return) for any month where vwretd
    # is missing — should be virtually never post-1970.
    vwretd = df["vwretd"].astype(float)
    sprtrn = df["sprtrn"].astype(float)
    combined = vwretd.copy()
    fallback_mask = vwretd.isna() & sprtrn.notna()
    n_fallback = int(fallback_mask.sum())
    if n_fallback:
        print(f"  [note] {n_fallback} months fall back to sprtrn (vwretd missing)")
        combined[fallback_mask] = sprtrn[fallback_mask]

    # Convert simple returns to log returns (CMM pipeline uses log throughout)
    # Column name kept as "sp500_logret" for backward-compat with cache readers
    # in cmm.regime, even though the series is now vwretd (MKT factor basis).
    df["sp500_logret"] = np.log1p(combined)
    df = df.dropna(subset=["sp500_logret"])

    # Normalize date to calendar month-end (matches pd.offsets.MonthEnd(0))
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")

    # Dedupe if any (shouldn't happen)
    df = df.sort_values("date").drop_duplicates("date", keep="last")

    out = df[["date", "sp500_logret"]]
    out.to_csv(OUT, index=False)

    print(f"Wrote {len(out):,} monthly observations to {OUT}")
    print(f"Date range: {out['date'].min().date()} → {out['date'].max().date()}")
    print(f"Value range: [{out['sp500_logret'].min():+.4f}, {out['sp500_logret'].max():+.4f}]")
    print()
    print("Done. The main pipeline (python main.py) will now read this cache")
    print("instead of calling yfinance.")


if __name__ == "__main__":
    main()
