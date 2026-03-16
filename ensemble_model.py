import numpy as np
import pandas as pd
import torch
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from lstm_model import LSTMModel, SEQUENCE_LENGTH, FEATURES

STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

# ── Load LSTM predictions ────────────────────────────────────
def get_lstm_predictions(ticker, X_test):
    model = LSTMModel(input_size=len(FEATURES))
    model.load_state_dict(torch.load(
        f"models/{ticker}_lstm.pth",
        map_location='cpu', weights_only=True
    ))
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        probs = model(X_tensor).squeeze().numpy()
    return probs

# ── Load XGBoost predictions ─────────────────────────────────
def get_xgboost_predictions(ticker, X_test_raw):
    with open(f"models/{ticker}_xgboost.pkl", 'rb') as f:
        xgb_model = pickle.load(f)
    with open(f"models/{ticker}_xgb_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_test_raw)
    return xgb_model.predict_proba(X_scaled)[:, 1]

# ── Prepare data ─────────────────────────────────────────────
def prepare_ensemble_data(ticker):
    df = pd.read_csv(f"data/{ticker}_features.csv")
    feature_data = df[FEATURES].values
    targets      = df['Target'].values

    # LSTM scaler
    with open(f"models/{ticker}_scaler.pkl", 'rb') as f:
        lstm_scaler = pickle.load(f)
    scaled = lstm_scaler.transform(feature_data)

    # Build sequences for LSTM
    X_seq = []
    for i in range(SEQUENCE_LENGTH, len(scaled)):
        X_seq.append(scaled[i - SEQUENCE_LENGTH:i])
    X_seq = np.array(X_seq)

    # Raw features for XGBoost (aligned with sequences)
    X_raw = feature_data[SEQUENCE_LENGTH:]
    y     = targets[SEQUENCE_LENGTH:]

    # Train/test split
    split      = int(len(X_seq) * 0.8)
    X_seq_test = X_seq[split:]
    X_raw_test = X_raw[split:]
    y_test     = y[split:]

    # Also get train split for meta-learner training
    X_seq_train = X_seq[:split]
    X_raw_train = X_raw[:split]
    y_train     = y[:split]

    return (X_seq_train, X_raw_train, y_train,
            X_seq_test,  X_raw_test,  y_test)

# ── Train ensemble ───────────────────────────────────────────
def train_ensemble(ticker):
    print(f"\nBuilding ensemble for {ticker}...")

    (X_seq_train, X_raw_train, y_train,
     X_seq_test,  X_raw_test,  y_test) = prepare_ensemble_data(ticker)

    # Get predictions from both models on TRAIN set
    lstm_train = get_lstm_predictions(ticker, X_seq_train)
    xgb_train  = get_xgboost_predictions(ticker, X_raw_train)

    # Stack predictions as features for meta-learner
    meta_train = np.column_stack([lstm_train, xgb_train])

    # Train meta-learner
    meta_model = LogisticRegression()
    meta_model.fit(meta_train, y_train)

    # Evaluate on TEST set
    lstm_test = get_lstm_predictions(ticker, X_seq_test)
    xgb_test  = get_xgboost_predictions(ticker, X_raw_test)
    meta_test = np.column_stack([lstm_test, xgb_test])

    # Individual accuracies
    lstm_acc = accuracy_score(y_test, (lstm_test > 0.5).astype(int))
    xgb_acc  = accuracy_score(y_test, (xgb_test  > 0.5).astype(int))

    # Ensemble accuracy
    ensemble_preds = meta_model.predict(meta_test)
    ensemble_acc   = accuracy_score(y_test, ensemble_preds)

    print(f"  LSTM Accuracy:     {lstm_acc:.2%}")
    print(f"  XGBoost Accuracy:  {xgb_acc:.2%}")
    print(f"  Ensemble Accuracy: {ensemble_acc:.2%} ⭐")
    print(classification_report(y_test, ensemble_preds,
          target_names=['Down', 'Up']))

    # Save ensemble
    with open(f"models/{ticker}_ensemble.pkl", 'wb') as f:
        pickle.dump(meta_model, f)
    print(f"  Ensemble saved for {ticker}!")

    return lstm_acc, xgb_acc, ensemble_acc

# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}
    for ticker in STOCKS:
        results[ticker] = train_ensemble(ticker)

    print("\n" + "="*50)
    print(f"{'Stock':<8} {'LSTM':>8} {'XGBoost':>10} {'Ensemble':>10}")
    print("="*50)
    for ticker, (l, x, e) in results.items():
        print(f"{ticker:<8} {l:>8.2%} {x:>10.2%} {e:>10.2%}")
    print("="*50)
    print("\nAll ensemble models saved!")
