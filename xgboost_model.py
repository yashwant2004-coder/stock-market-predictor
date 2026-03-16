import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

FEATURES = [
    'Close', 'Volume', 'MA_20', 'MA_50', 'RSI',
    'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
    'Volume_Ratio', 'Daily_Return', 'Volatility',
    'Lag_1', 'Lag_2', 'Lag_5'
]

def train_xgboost(ticker):
    print(f"\nTraining XGBoost for {ticker}...")

    df = pd.read_csv(f"data/{ticker}_features.csv")

    X = df[FEATURES].values
    y = df['Target'].values

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Detailed Report:")
    print(classification_report(y_test, preds,
          target_names=['Down', 'Up']))

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    with open(f"models/{ticker}_xgboost.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"models/{ticker}_xgb_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"  XGBoost model saved for {ticker}!")
    return model, acc

if __name__ == "__main__":
    results = {}
    for ticker in STOCKS:
        model, acc = train_xgboost(ticker)
        results[ticker] = acc

    print("\n" + "="*40)
    print("XGBoost Results Summary:")
    print("="*40)
    for ticker, acc in results.items():
        print(f"  {ticker}: {acc:.2%}")
    print("="*40)
    print("\nAll XGBoost models trained and saved!")