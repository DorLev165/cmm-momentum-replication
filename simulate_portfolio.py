import pandas as pd

prices = pd.read_csv(
    r"C:\Projects\Momentum trading\adj_close_prices.csv",
    index_col="Date", parse_dates=["Date"],
)

SHARES = {"MSFT": 500, "AAPL": 600, "SPY": -100}
CASH_END = 100_000.0  # Cash on Apr 17, 2026 per the problem statement
RATE = 0.02
DAILY_FACTOR = 1 + RATE / 365

df = prices.copy()

# Position values
df["MSFT_val"] = SHARES["MSFT"] * df["MSFT"]
df["AAPL_val"] = SHARES["AAPL"] * df["AAPL"]
df["SPY_val"]  = SHARES["SPY"]  * df["SPY"]

# Cash starts at $100,000 on Jan 4, 2010 and compounds daily at 2%/365 (calendar days)
CASH_0 = 100_000.0
days_elapsed = (df.index - df.index[0]).days
df["Cash"] = CASH_0 * (DAILY_FACTOR ** days_elapsed)

# NAV
df["NAV"] = df["MSFT_val"] + df["AAPL_val"] + df["SPY_val"] + df["Cash"]

# Exposures
df["Long_Exposure"]  = df["MSFT_val"] + df["AAPL_val"]
df["Short_Exposure"] = df["SPY_val"].abs()

# Leverage
df["Gross_Leverage"] = (df["Long_Exposure"] + df["Short_Exposure"]) / df["NAV"]
df["Net_Leverage"]   = (df["Long_Exposure"] - df["Short_Exposure"]) / df["NAV"]

cols = ["MSFT", "AAPL", "SPY",
        "MSFT_val", "AAPL_val", "SPY_val",
        "Cash", "NAV",
        "Long_Exposure", "Short_Exposure",
        "Gross_Leverage", "Net_Leverage"]
df = df[cols].round(4)

out_csv = r"C:\Projects\Momentum trading\portfolio_simulation_final.csv"
df.to_csv(out_csv)

out_xlsx = r"C:\Projects\Momentum trading\portfolio_simulation_final.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    df.to_excel(w, sheet_name="Simulation")

print("First row (Jan 4, 2010):")
print(df.iloc[0])
print("\nLast row (Apr 17, 2026):")
print(df.iloc[-1])
print(f"\nRows: {len(df)}")
print(f"CSV : {out_csv}")
print(f"XLSX: {out_xlsx}")
