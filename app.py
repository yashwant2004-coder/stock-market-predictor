import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(
    page_title="Live Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .predict-up {
        background: linear-gradient(135deg, #0d4f2e, #1a7a4a);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #00ff88;
        border: 2px solid #00ff88;
        margin: 10px 0;
    }
    .predict-down {
        background: linear-gradient(135deg, #4f0d0d, #7a1a1a);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #ff4444;
        border: 2px solid #ff4444;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

STOCKS = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet (Google)',
    'MSFT': 'Microsoft',
    'TSLA': 'Tesla',
    'AMZN': 'Amazon'
}

FEATURES = [
    'Close', 'Volume', 'MA_20', 'MA_50', 'RSI',
    'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
    'Volume_Ratio', 'Daily_Return', 'Volatility',
    'Lag_1', 'Lag_2', 'Lag_5'
]

def add_indicators(df):
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + std * 2
    df['BB_Lower'] = df['BB_Middle'] - std * 2
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(20).std()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_5'] = df['Close'].shift(5)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='2y')
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = add_indicators(df)
    return df

@st.cache_data(ttl=300)
def fetch_quote(ticker):
    info = yf.Ticker(ticker).fast_info
    return {
        'price':  round(info.last_price, 2),
        'open':   round(info.open, 2),
        'high':   round(info.day_high, 2),
        'low':    round(info.day_low, 2),
        'volume': int(info.last_volume)
    }

@st.cache_resource
def train_model(ticker, df):
    X = df[FEATURES].values
    y = df['Target'].values
    split = int(len(X) * 0.8)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=42
    )
    model.fit(X_scaled[:split], y[:split])
    return model, scaler

def make_prediction(df, model, scaler):
    X_latest = scaler.transform(df[FEATURES].values[-1:])
    prob = model.predict_proba(X_latest)[0][1]
    pred = model.predict(X_latest)[0]
    return prob, pred

def plot_chart(df, show_indicators, days):
    df_plot = df.tail(days)
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.plot(df_plot['Date'], df_plot['Close'],
            color='#00d4ff', linewidth=2, label='Close Price')
    colors = {
        'MA_20': '#ffaa00', 'MA_50': '#ff6600',
        'BB_Upper': '#00ff88', 'BB_Lower': '#ff4444'
    }
    for ind in show_indicators:
        ax.plot(df_plot['Date'], df_plot[ind],
                color=colors[ind], linewidth=1,
                linestyle='--', label=ind, alpha=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_color('#444')
    ax.tick_params(colors='white')
    ax.set_ylabel('Price (USD)', color='white')
    ax.legend(facecolor='#1e2130', labelcolor='white')
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

def plot_rsi(df, days):
    df_plot = df.tail(days)
    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.plot(df_plot['Date'], df_plot['RSI'],
            color='#aa88ff', linewidth=1.5)
    ax.axhline(70, color='#ff4444', linestyle='--', alpha=0.7, label='Overbought')
    ax.axhline(30, color='#00ff88', linestyle='--', alpha=0.7, label='Oversold')
    ax.fill_between(df_plot['Date'], df_plot['RSI'], 50,
                    where=df_plot['RSI'] >= 50, alpha=0.1, color='green')
    ax.fill_between(df_plot['Date'], df_plot['RSI'], 50,
                    where=df_plot['RSI'] < 50, alpha=0.1, color='red')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_color('#444')
    ax.tick_params(colors='white')
    ax.set_ylabel('RSI', color='white')
    ax.legend(facecolor='#1e2130', labelcolor='white')
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

def main():
    st.markdown("# 📈 Live Stock Market Predictor")
    st.markdown("**Real-time AI predictions using Gradient Boosting Ensemble**")
    st.markdown("---")

    st.sidebar.title("⚙️ Settings")
    ticker = st.sidebar.selectbox(
        "Select Stock",
        list(STOCKS.keys()),
        format_func=lambda x: f"{x} — {STOCKS[x]}"
    )
    show_indicators = st.sidebar.multiselect(
        "Show Indicators",
        ['MA_20', 'MA_50', 'BB_Upper', 'BB_Lower'],
        default=['MA_20', 'MA_50']
    )
    days = st.sidebar.slider("Days to Display", 30, 365, 180)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 About")
    st.sidebar.markdown(
        "Fetches **live data** from Yahoo Finance "
        "and predicts tomorrow's price direction."
    )

    with st.spinner(f"⏳ Fetching live data for {ticker}..."):
        try:
            df    = fetch_data(ticker)
            quote = fetch_quote(ticker)
            st.success(f"✅ Live data loaded — {len(df)} trading days")
        except Exception as e:
            st.error(f"Error: {e}")
            return

    latest   = df.iloc[-1]
    prev     = df.iloc[-2]
    price_ch = latest['Close'] - prev['Close']
    pct_ch   = (price_ch / prev['Close']) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("💰 Live Price", f"${quote['price']}",
                  f"{price_ch:+.2f} ({pct_ch:+.2f}%)")
    with c2:
        st.metric("📂 Open",   f"${quote['open']}")
    with c3:
        st.metric("📈 High",   f"${quote['high']}")
    with c4:
        st.metric("📉 Low",    f"${quote['low']}")
    with c5:
        st.metric("📊 Volume", f"{quote['volume']:,}")

    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    st.markdown("---")

    st.subheader(f"📊 {ticker} — {STOCKS[ticker]}")
    st.pyplot(plot_chart(df, show_indicators, days))

    st.subheader("📉 RSI Indicator")
    st.pyplot(plot_rsi(df, days))

    st.markdown("---")
    st.subheader("🤖 AI Prediction for Tomorrow")

    with st.spinner("Training model on live data..."):
        model, scaler = train_model(ticker, df)
        prob, pred    = make_prediction(df, model, scaler)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("🎯 Confidence",
                  f"{prob:.2%}" if pred == 1 else f"{1-prob:.2%}",
                  "Bullish 📈" if pred == 1 else "Bearish 📉")
    with c2:
        st.metric("📊 RSI Signal",
                  f"{latest['RSI']:.1f}",
                  "Overbought ⚠️" if latest['RSI'] > 70
                  else "Oversold 💡" if latest['RSI'] < 30
                  else "Neutral ➡️")

    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown(
            f'<div class="predict-up">🚀 PREDICTION: PRICE WILL GO UP TOMORROW!'
            f'<br><small>Confidence: {prob:.1%}</small></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="predict-down">⚠️ PREDICTION: PRICE WILL GO DOWN TOMORROW!'
            f'<br><small>Confidence: {1-prob:.1%}</small></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

    st.caption("⚠️ For educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()
