import yfinance as yf
import pandas as pd
import os

# List of stocks we want to predict
STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

def download_stock_data(ticker, period='5y'):
    print(f"Downloading data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    df.to_csv(f"data/{ticker}.csv", index=False)
    print(f"Saved {ticker}.csv — {len(df)} rows of data")
    return df

def download_all():
    os.makedirs('data', exist_ok=True)
    for stock in STOCKS:
        download_stock_data(stock)
    print("\nAll stock data downloaded successfully!")

if __name__ == "__main__":
    download_all()
