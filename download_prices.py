import yfinance as yf
import pandas as pd

tickers = ["MSFT", "AAPL", "SPY"]

data = yf.download(
    tickers,
    start="2010-01-04",
    end="2026-04-18",  # end is exclusive, so this gets through Apr 17
    auto_adjust=False,
    actions=False,
)

adj_close = data["Adj Close"][tickers]
adj_close.index.name = "Date"
adj_close.columns.name = None

print(f"Date range: {adj_close.index[0].date()} to {adj_close.index[-1].date()}")
print(f"Total trading days: {len(adj_close)}")
print(f"\nFirst 3 rows:\n{adj_close.head(3)}")
print(f"\nLast 3 rows:\n{adj_close.tail(3)}")
print(f"\nAny missing values:\n{adj_close.isna().sum()}")

out_path = r"C:\Projects\Momentum trading\adj_close_prices.csv"
adj_close.to_csv(out_path)
print(f"\nSaved to: {out_path}")
