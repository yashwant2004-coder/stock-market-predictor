import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from lstm_model import LSTMModel, SEQUENCE_LENGTH, FEATURES
from feature_engineering import add_technical_indicators

# ── Page Config ──────────────────────────────────────────────
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
    .live-badge {
        background: #ff4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        animation: pulse 1.5s infinite;
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

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models(ticker):
    lstm = LSTMModel(input_size=len(FEATURES))
    lstm.load_state_dict(torch.load(
        f"models/{ticker}_lstm.pth",
        map_location='cpu', weights_only=True
    ))
    lstm.eval()
    with open(f"models/{ticker}_xgboost.pkl", 'rb') as f:
        xgb = pickle.load(f)
    with open(f"models/{ticker}_ensemble.pkl", 'rb') as f:
        ensemble = pickle.load(f)
    with open(f"models/{ticker}_scaler.pkl", 'rb') as f:
        lstm_scaler = pickle.load(f)
    with open(f"models/{ticker}_xgb_scaler.pkl", 'rb') as f:
        xgb_scaler = pickle.load(f)
    return lstm, xgb, ensemble, lstm_scaler, xgb_scaler

# ── Fetch Live Data ──────────────────────────────────────────
def fetch_live_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='2y')
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = add_technical_indicators(df)
    return df

# ── Fetch Live Quote ─────────────────────────────────────────
def fetch_live_quote(ticker):
    stock = yf.Ticker(ticker)
    info  = stock.fast_info
    return {
        'price':  round(info.last_price, 2),
        'open':   round(info.open, 2),
        'high':   round(info.day_high, 2),
        'low':    round(info.day_low, 2),
        'volume': int(info.last_volume)
    }

# ── Make Prediction ──────────────────────────────────────────
def make_prediction(ticker, df):
    lstm, xgb_model, ensemble, lstm_scaler, xgb_scaler = load_models(ticker)

    feature_data = df[FEATURES].values
    scaled = lstm_scaler.transform(feature_data)

    # LSTM
    X_seq = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(FEATURES))
    with torch.no_grad():
        lstm_prob = lstm(torch.FloatTensor(X_seq)).item()

    # XGBoost
    X_raw    = xgb_scaler.transform(feature_data[-1:])
    xgb_prob = xgb_model.predict_proba(X_raw)[0][1]

    # Ensemble
    meta_input    = np.array([[lstm_prob, xgb_prob]])
    ensemble_prob = ensemble.predict_proba(meta_input)[0][1]
    ensemble_pred = ensemble.predict(meta_input)[0]

    return {
        'lstm_prob':     lstm_prob,
        'xgb_prob':      xgb_prob,
        'ensemble_prob': ensemble_prob,
        'is_up':         ensemble_pred == 1
    }

# ── Plot Chart ───────────────────────────────────────────────
def plot_chart(df, ticker, show_indicators, days):
    df_plot = df.tail(days).copy()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.plot(df_plot['Date'], df_plot['Close'],
            color='#00d4ff', linewidth=2, label='Close Price')

    colors = {
        'MA_20':    '#ffaa00',
        'MA_50':    '#ff6600',
        'BB_Upper': '#00ff88',
        'BB_Lower': '#ff4444'
    }
    for ind in show_indicators:
        if ind in df_plot.columns:
            ax.plot(df_plot['Date'], df_plot[ind],
                    color=colors[ind], linewidth=1,
                    linestyle='--', label=ind, alpha=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#444')
    ax.tick_params(colors='white')
    ax.set_ylabel('Price (USD)', color='white')
    ax.legend(facecolor='#1e2130', labelcolor='white')
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

# ── Plot RSI ─────────────────────────────────────────────────
def plot_rsi(df, days):
    df_plot = df.tail(days).copy()
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
                    where=df_plot['RSI'] < 50,  alpha=0.1, color='red')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#444')
    ax.tick_params(colors='white')
    ax.set_ylabel('RSI', color='white')
    ax.legend(facecolor='#1e2130', labelcolor='white')
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

# ── Main ─────────────────────────────────────────────────────
def main():
    # Header
    col_title, col_badge = st.columns([6, 1])
    with col_title:
        st.markdown("# 📈 Live Stock Market Predictor")
        st.markdown("**Real-time AI predictions using LSTM + XGBoost Ensemble**")
    with col_badge:
        st.markdown('<br><span class="live-badge">🔴 LIVE</span>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar
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
    auto_refresh = st.sidebar.toggle("Auto Refresh (5 min)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 About")
    st.sidebar.markdown(
        "Fetches **live data** from Yahoo Finance "
        "and runs it through the trained ensemble model "
        "to predict tomorrow's price direction."
    )

    # Auto refresh
    if auto_refresh:
        st.sidebar.success("Auto-refresh ON")
        import time
        time.sleep(300)
        st.rerun()

    # Fetch live data
    with st.spinner(f"⏳ Fetching live data for {ticker}..."):
        try:
            df   = fetch_live_data(ticker)
            quote = fetch_live_quote(ticker)
            st.success(f"✅ Live data loaded — {len(df)} trading days")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

    # Live quote metrics
    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    price_ch = latest['Close'] - prev['Close']
    pct_ch   = (price_ch / prev['Close']) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("💰 Live Price",
                  f"${quote['price']}",
                  f"{price_ch:+.2f} ({pct_ch:+.2f}%)")
    with c2:
        st.metric("📂 Open",  f"${quote['open']}")
    with c3:
        st.metric("📈 High",  f"${quote['high']}")
    with c4:
        st.metric("📉 Low",   f"${quote['low']}")
    with c5:
        st.metric("📊 Volume", f"{quote['volume']:,}")

    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    st.markdown("---")

    # Charts
    st.subheader(f"📊 {ticker} — {STOCKS[ticker]}")
    st.pyplot(plot_chart(df, ticker, show_indicators, days))

    st.subheader("📉 RSI Indicator")
    st.pyplot(plot_rsi(df, days))

    st.markdown("---")

    # AI Prediction
    st.subheader("🤖 AI Prediction for Tomorrow")
    with st.spinner("Running AI models on live data..."):
        try:
            pred = make_prediction(ticker, df)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("🧠 LSTM",
                          f"{pred['lstm_prob']:.2%}",
                          "Bullish 📈" if pred['lstm_prob'] > 0.5
                          else "Bearish 📉")
            with c2:
                st.metric("🌲 XGBoost",
                          f"{pred['xgb_prob']:.2%}",
                          "Bullish 📈" if pred['xgb_prob'] > 0.5
                          else "Bearish 📉")
            with c3:
                st.metric("⚡ Ensemble",
                          f"{pred['ensemble_prob']:.2%}",
                          "Bullish 📈" if pred['ensemble_prob'] > 0.5
                          else "Bearish 📉")

            st.markdown("<br>", unsafe_allow_html=True)

            if pred['is_up']:
                st.markdown(
                    '<div class="predict-up">'
                    '🚀 PREDICTION: PRICE WILL GO UP TOMORROW!'
                    f'<br><small>Confidence: {pred["ensemble_prob"]:.1%}</small>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="predict-down">'
                    '⚠️ PREDICTION: PRICE WILL GO DOWN TOMORROW!'
                    f'<br><small>Confidence: {1 - pred["ensemble_prob"]:.1%}</small>'
                    '</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "⚠️ Disclaimer: For educational purposes only. "
        "Not financial advice."
    )

if __name__ == "__main__":
    main()
