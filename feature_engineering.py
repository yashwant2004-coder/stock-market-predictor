import pandas as pd
import numpy as np
import os

def add_technical_indicators(df):
    # Moving Averages (trend direction)
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # RSI - Relative Strength Index (overbought/oversold)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD - Moving Average Convergence Divergence
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (volatility)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)

    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    # Price changes
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

    # Lag features (yesterday, 2 days ago, 5 days ago prices)
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_5'] = df['Close'].shift(5)

    # Target column - what we want to predict
    # 1 = price went UP next day, 0 = price went DOWN
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop rows with missing values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def process_all_stocks():
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    for ticker in stocks:
        print(f"Processing {ticker}...")
        df = pd.read_csv(f"data/{ticker}.csv")
        df = add_technical_indicators(df)
        df.to_csv(f"data/{ticker}_features.csv", index=False)
        print(f"Saved {ticker}_features.csv — {len(df)} rows, {len(df.columns)} columns")

    print("\nFeature engineering done for all stocks!")

if __name__ == "__main__":
    process_all_stocks()
