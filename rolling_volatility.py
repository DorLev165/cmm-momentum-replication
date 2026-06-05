import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prices = pd.read_csv(
    r"C:\Projects\Momentum trading\adj_close_prices.csv",
    index_col="Date", parse_dates=["Date"],
)

SHARES = {"MSFT": 500, "AAPL": 600, "SPY": -100}
CASH_0 = 100_000.0
RATE = 0.02
DAILY_FACTOR = 1 + RATE / 365
WINDOW = 90
ANNUAL_FACTOR = np.sqrt(250)

df = prices.copy()
days_elapsed = (df.index - df.index[0]).days
df["Cash"] = CASH_0 * (DAILY_FACTOR ** days_elapsed)
df["NAV"] = (SHARES["MSFT"] * df["MSFT"]
             + SHARES["AAPL"] * df["AAPL"]
             + SHARES["SPY"]  * df["SPY"]
             + df["Cash"])

# Daily simple returns of NAV
df["Return"] = df["NAV"].pct_change()

# 90-day rolling 1-day vol (std of daily returns) and annualized
df["Vol_1day"]      = df["Return"].rolling(WINDOW).std()
df["Vol_Annualized"] = df["Vol_1day"] * ANNUAL_FACTOR

# Restrict output window: Jun 1, 2010 -> Apr 17, 2026
out = df.loc["2010-06-01":"2026-04-17", ["Cash", "NAV", "Return", "Vol_1day", "Vol_Annualized"]]

out_csv  = r"C:\Projects\Momentum trading\rolling_volatility_final.csv"
out_xlsx = r"C:\Projects\Momentum trading\rolling_volatility_final.xlsx"
out.to_csv(out_csv)
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    out.to_excel(w, sheet_name="RollingVol")

# Plot
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(out.index, out["Vol_1day"], color="steelblue", linewidth=0.9)
ax[0].set_title("90-Day Rolling 1-Day Volatility of Portfolio NAV")
ax[0].set_ylabel("1-Day Vol")
ax[0].grid(alpha=0.3)

ax[1].plot(out.index, out["Vol_Annualized"], color="darkorange", linewidth=0.9)
ax[1].set_title("Annualized Volatility (1-Day Vol x sqrt(250))")
ax[1].set_ylabel("Annualized Vol")
ax[1].set_xlabel("Date")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = r"C:\Projects\Momentum trading\rolling_volatility_plot_final.png"
plt.savefig(plot_path, dpi=130)
plt.close()

print(f"Rows in output window: {len(out)}")
print(f"\nFirst 3 rows:\n{out.head(3)}")
print(f"\nLast 3 rows:\n{out.tail(3)}")
print(f"\nMin annualized vol: {out['Vol_Annualized'].min():.4f}  on {out['Vol_Annualized'].idxmin().date()}")
print(f"Max annualized vol: {out['Vol_Annualized'].max():.4f}  on {out['Vol_Annualized'].idxmax().date()}")
print(f"\nCSV : {out_csv}")
print(f"XLSX: {out_xlsx}")
print(f"PNG : {plot_path}")
