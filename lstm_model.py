import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import os
import pickle

# ── Configuration ──────────────────────────────────────────
SEQUENCE_LENGTH = 60   # look back 60 days to predict next day
EPOCHS          = 50
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
STOCKS          = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

FEATURES = [
    'Close', 'Volume', 'MA_20', 'MA_50', 'RSI',
    'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
    'Volume_Ratio', 'Daily_Return', 'Volatility',
    'Lag_1', 'Lag_2', 'Lag_5'
]

# ── LSTM Model Definition ───────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # take last timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)

# ── Data Preparation ────────────────────────────────────────
def prepare_data(ticker):
    df = pd.read_csv(f"data/{ticker}_features.csv")

    feature_data = df[FEATURES].values
    targets      = df['Target'].values

    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(feature_data)

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(feature_data)):
        X.append(feature_data[i - SEQUENCE_LENGTH:i])
        y.append(targets[i])

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler

# ── Training ────────────────────────────────────────────────
def train_model(ticker):
    print(f"\nTraining LSTM for {ticker}...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(ticker)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t  = torch.FloatTensor(X_test)

    model     = LSTMModel(input_size=len(FEATURES))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train_t), BATCH_SIZE):
            xb = X_train_t[i:i + BATCH_SIZE]
            yb = y_train_t[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = (model(X_test_t).squeeze().numpy() > 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss:.4f} — Test Accuracy: {acc:.2%}")

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f"models/{ticker}_lstm.pth")
    with open(f"models/{ticker}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Model saved for {ticker}!")
    return model

# ── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for ticker in STOCKS:
        train_model(ticker)
    print("\nAll LSTM models trained and saved!")
