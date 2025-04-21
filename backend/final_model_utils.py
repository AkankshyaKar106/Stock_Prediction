import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from keras.metrics import MeanSquaredError
import talib
import pytz
import sqlite3
import gdown
import os


def load_lstm_model():
    try:
        lstm_model = load_model(
            "D:/Qwegle/Qwegle/Algorithmic_Trading/lstm.h5", custom_objects={
                "mse": MeanSquaredError()})
        lstm_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        with open(
                'D:/Qwegle/Qwegle/Algorithmic_Trading/tokenizer1.pkl', 'rb'
                ) as f:
            tokenizer = pickle.load(f)
        return lstm_model, tokenizer
    except Exception as e:
        print(f"Error loading LSTM model or tokenizer: {e}")
        return None, None


def load_rf_model():
    try:
        rf_url = (
            "https://drive.google.com/uc?id=1O5_ynexhyLDYGiOBpzNibI5KqGiiNXKJ")
        temp_file = "temp_rf.pkl"  # Temporary file

        # Download the model temporarily
        gdown.download(rf_url, temp_file, quiet=False)

        # Load the model from the temp file
        rf_model = joblib.load(temp_file)

        # Remove temp file after loading
        os.remove(temp_file)
        with open(
                'D:/Qwegle/Qwegle/Algorithmic_Trading/scaler1.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(
                'D:/Qwegle/Qwegle/Algorithmic_Trading/target_scaler1.pkl', 'rb'
                ) as f:
            target_scaler = pickle.load(f)
        with open(
                'D:/Qwegle/Qwegle/Algorithmic_Trading/sentiment_scaler1.pkl',
                'rb') as f:
            sentiment_scaler = pickle.load(f)
        return rf_model, scaler, target_scaler, sentiment_scaler
    except Exception as e:
        print(f"Error loading Random Forest model or scaler: {e}")
        return None, None, None, None


def fetch_tickers(url, table_index, ticker_column, suffix_check=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return []
        selected_table = tables[table_index]
        rows = selected_table.find_all("tr")
        tickers = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) > ticker_column:
                ticker = cols[ticker_column].text.strip()
                if suffix_check and suffix_check in ticker:
                    tickers.append(ticker.split('.')[0])
                else:
                    tickers.append(ticker)
        return sorted(list(set(tickers)))
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return []


def fetch_bse_sensex_tickers():
    url = "https://en.wikipedia.org/wiki/BSE_SENSEX"
    return fetch_tickers(
        url, table_index=2, ticker_column=0, suffix_check=".BO")


def fetch_nse_nifty_500_tickers():
    url = "https://en.wikipedia.org/wiki/NIFTY_500"
    return fetch_tickers(url, table_index=4, ticker_column=3)


def fetch_stock_data(ticker, prediction_date):
    try:
        if isinstance(prediction_date, datetime):
            prediction_date = prediction_date.strftime('%Y-%m-%d')

        start_with_offset = (
            pd.to_datetime(prediction_date) - pd.DateOffset(
                days=100)).strftime('%Y-%m-%d')

        for suffix in [".NS", ".BO", ""]:
            try:
                stock = yf.Ticker(ticker + suffix)
                data = stock.history(
                    start=start_with_offset, end=prediction_date)

                if not data.empty:
                    data = data.reset_index()

                    required_columns = [
                        'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_columns:
                        if col not in data.columns:
                            st.warning(
                                f"Missing column {col} in data for {ticker}")
                            return None

                    numeric_columns = [
                        'Open', 'High', 'Low', 'Close', 'Volume']
                    for col in numeric_columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')

                    data['Ticker'] = ticker + suffix
                    return data

            except Exception as e:
                print(f"Failed to fetch data for {ticker + suffix}: {str(e)}")
                continue

        st.warning(f"No data found for {ticker} with any suffix")
        return None

    except Exception as e:
        st.error(f"Error in fetch_stock_data for {ticker}: {str(e)}")
        return None


def collect_sentiment_data(ticker):
    try:
        query = f"{ticker} stock news"
        url = f"https://www.google.com/search?q={query}&tbm=nws"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [headline.text for headline in soup.select("h3")]
        return headlines
    except Exception as e:
        st.error(f"Error collecting sentiment data for {ticker}: {e}")
        return []


def preprocess_sentiment_text(headlines, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences(headlines)
    return pad_sequences(
        sequences, maxlen=max_len, padding='post', truncating='post')


def calculate_sentiment_score(headlines, lstm_model, tokenizer):
    if not headlines:
        return 0.0

    headlines = [headline.lower().strip() for headline in headlines]
    x = preprocess_sentiment_text(headlines, tokenizer)

    predictions = lstm_model.predict(x).flatten()

    min_pred = np.min(predictions)
    max_pred = np.max(predictions)
    if max_pred - min_pred != 0:
        scaled_scores = 2 * (predictions - min_pred) / (max_pred - min_pred)-1
    else:
        scaled_scores = np.zeros_like(predictions)

    final_scores = scaled_scores * 100
    return np.mean(final_scores)


def add_previous_close_column(stock_data):
    stock_data['Previous_Close'] = stock_data['Close'].shift(1)
    return stock_data


def compute_technical_indicators(stock_data):
    if stock_data is None or stock_data.empty:
        raise ValueError("Stock data is empty or None.")

    try:
        df = stock_data.copy()

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])

        if len(df) == 0:
            raise ValueError("No valid Close prices after cleaning data")

        close_array = df['Close'].values
        high_array = df['High'].values
        low_array = df['Low'].values

        df['SMA_20'] = talib.SMA(close_array, timeperiod=20)
        df['SMA_50'] = talib.SMA(close_array, timeperiod=50)
        df['RSI'] = talib.RSI(close_array, timeperiod=14)
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(close_array)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
            close_array)
        df['Price_Change'] = df['Close'].pct_change()

        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

        df['ATR'] = talib.ATR(
            high_array, low_array, close_array, timeperiod=14)
        df['ROC'] = talib.ROC(close_array, timeperiod=10)
        df['MOM'] = talib.MOM(close_array, timeperiod=10)

        df = df.ffill().bfill()

        return df

    except Exception as e:
        raise Exception(f"Error in compute_technical_indicators: {str(e)}")


def predict_next_hour_price(
        stock_data, sentiment_score, rf_model, scaler,
        sentiment_scaler, target_scaler):
    try:
        feature_columns = [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "SMA_50", "RSI", "MACD", "BB_Middle", "ATR", "ROC", "MOM"
        ]

        if stock_data is None or stock_data.empty:
            raise ValueError("Stock data is empty or None")

        stock_data = stock_data.copy()

        stock_data['Returns'] = stock_data['Close'].pct_change()
        volatility = stock_data['Returns'].ewm(
            span=10, min_periods=10).std().iloc[-1] * np.sqrt(6.5)
        hourly_volatility = volatility / np.sqrt(6.5)

        stock_data['EMA_5'] = stock_data['Close'].ewm(
            span=5, adjust=False).mean()

        for period in [1, 2, 3, 4, 6, 8]:  # Hours instead of days
            stock_data[f'{period}h_return'] = stock_data['Close'].pct_change(
                periods=period)

        trend_weights = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04]
        trend_signals = [
            stock_data['1h_return'].iloc[-1],
            stock_data['2h_return'].iloc[-1],
            stock_data['3h_return'].iloc[-1],
            stock_data['4h_return'].iloc[-1],
            stock_data['6h_return'].iloc[-1],
            stock_data['8h_return'].iloc[-1]
        ]
        weighted_trend = sum(
            w * t for w, t in zip(trend_weights, trend_signals))

        last_prices = stock_data['Close'].tail(7)
        avg_price = last_prices.mean()
        median_price = last_prices.median()
        last_close = stock_data['Close'].iloc[-1]

        for col in feature_columns:
            if col not in stock_data.columns:
                raise ValueError(f"Missing required column: {col}")
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

        last_hour = stock_data[feature_columns].iloc[-1].values
        scaled_features = scaler.transform(last_hour.reshape(1, -1))
        sentiment_scaled = sentiment_scaler.transform(
            np.array([[sentiment_score]]))
        combined_features = np.hstack([scaled_features, sentiment_scaled])

        raw_prediction = rf_model.predict(combined_features)[0]

        if target_scaler is not None:
            predicted_price = target_scaler.inverse_transform(
                [[raw_prediction]])[0][0]
        else:
            predicted_price = raw_prediction

        predicted_change = (predicted_price - last_close) / last_close

        volatility_multiplier = 1.5
        max_hourly_move = (
            last_close * hourly_volatility * volatility_multiplier)

        trend_weight = 0.2
        trend_adjustment = weighted_trend * trend_weight

        predicted_change = predicted_change * (1 + trend_adjustment)

        if abs(predicted_change) > 0.5 * hourly_volatility:
            dampening_factor = np.exp(
                -3 * abs(predicted_change) / hourly_volatility)
            predicted_change *= dampening_factor

        capped_change = np.clip(
            predicted_change, -max_hourly_move / last_close,
            max_hourly_move / last_close)

        preliminary_price = last_close * (1 + capped_change)

        sentiment_impact = 0.00001 * sentiment_score
        sentiment_adjusted_price = preliminary_price * (1 + sentiment_impact)

        weighted_avg_price = (0.5 * sentiment_adjusted_price +
                              0.2 * last_close +
                              0.2 * avg_price +
                              0.1 * median_price)

        max_allowed_change = 0.01
        final_predicted_price = np.clip(
            weighted_avg_price,
            last_close * (1 - max_allowed_change),
            last_close * (1 + max_allowed_change)
        )

        ma_boundary = 0.02
        ema5_price = stock_data['EMA_5'].iloc[-1]

        if abs((final_predicted_price - ema5_price) / ema5_price
               ) > ma_boundary:
            final_predicted_price = ema5_price * (
                1 + np.sign(final_predicted_price - ema5_price) * ma_boundary)

        return final_predicted_price

    except Exception as e:
        st.error(f"Error predicting stock price for next hour: {e}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)
        return None


def calculate_difference_and_signal(
        predicted_price, stock_data, prediction_time=None):
    try:
        last_close = stock_data["Close"].iloc[-1]

        rsi = stock_data["RSI"].iloc[-1]
        macd = stock_data["MACD"].iloc[-1]
        macd_signal = stock_data["MACD_Signal"].iloc[-1]
        volume = stock_data["Volume"].iloc[-1]
        avg_volume = stock_data["Volume"].ewm(span=10).mean().iloc[-1]

        pred_change = ((predicted_price - last_close) / last_close) * 100

        signals = []

        # Price movement signal
        if abs(pred_change) > 0.5:
            signals.append(1 if pred_change > 0 else -1)

        # RSI signal
        if rsi < 30:
            signals.append(1)
        elif rsi > 70:
            signals.append(-1)

        # MACD signal
        if macd > macd_signal:
            signals.append(1)
        elif macd < macd_signal:
            signals.append(-1)

        # Volume signal
        if volume > avg_volume * 1.2:
            signals.append(1 if pred_change > 0 else -1)

        # Calculate final signal
        signal_strength = sum(signals) / len(signals) if signals else 0

        if signal_strength >= 0.7:
            signal = "Strong Buy"
        elif signal_strength >= 0.3:
            signal = "Buy"
        elif signal_strength <= -0.7:
            signal = "Strong Sell"
        elif signal_strength <= -0.3:
            signal = "Sell"
        else:
            signal = "Hold"

        # Define these variables before the reasons list
        rsi_status = ('oversold' if rsi < 30
                      else 'overbought' if rsi > 70 else 'neutral')
        volume_diff = (volume / avg_volume * 100 - 100)

        reasons = [
            f"Predicted price movement: {pred_change:.2f}%",
            f"RSI: {rsi:.1f} ({rsi_status})",
            f"MACD: {'bullish' if macd > macd_signal else 'bearish'} trend",
            f"Volume: {volume_diff:.1f}% compared to average"
        ]

        analysis = " | ".join(reasons)

        return signal, analysis

    except Exception as e:
        st.error(f"Error analyzing prediction: {e}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)
        return None, None


def predict_next_day_price(
        stock_data, sentiment_score, rf_model, scaler,
        sentiment_scaler, target_scaler):
    try:
        feature_columns = [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "SMA_50", "RSI", "MACD", "BB_Middle", "ATR", "ROC", "MOM"
        ]

        if stock_data is None or stock_data.empty:
            raise ValueError("Stock data is empty or None")

        stock_data = stock_data.copy()

        stock_data['Returns'] = stock_data['Close'].pct_change()
        volatility = stock_data['Returns'].ewm(
            span=10, min_periods=10).std().iloc[-1] * np.sqrt(252)
        daily_volatility = volatility / np.sqrt(252)

        stock_data['EMA_5'] = stock_data['Close'].ewm(
            span=5, adjust=False).mean()

        for period in [1, 2, 3, 5, 7, 10]:
            stock_data[f'{period}d_return'] = stock_data['Close'].pct_change(
                periods=period)

        trend_weights = [0.35, 0.25, 0.20, 0.10, 0.06, 0.04, ]
        trend_signals = [
            stock_data['1d_return'].iloc[-1],
            stock_data['2d_return'].iloc[-1],
            stock_data['3d_return'].iloc[-1],
            stock_data['5d_return'].iloc[-1],
            stock_data['7d_return'].iloc[-1],
            stock_data['10d_return'].iloc[-1],

        ]
        weighted_trend = sum(
            w * t for w, t in zip(trend_weights, trend_signals))

        last_prices = stock_data['Close'].tail(8)
        avg_price = last_prices.mean()
        median_price = last_prices.median()
        last_close = stock_data['Close'].iloc[-1]

        for col in feature_columns:
            if col not in stock_data.columns:
                raise ValueError(f"Missing required column: {col}")
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

        last_day = stock_data[feature_columns].iloc[-1].values
        scaled_features = scaler.transform(last_day.reshape(1, -1))
        sentiment_scaled = sentiment_scaler.transform(
            np.array([[sentiment_score]]))
        combined_features = np.hstack([scaled_features, sentiment_scaled])

        raw_prediction = rf_model.predict(combined_features)[0]

        if target_scaler is not None:
            predicted_price = target_scaler.inverse_transform(
                [[raw_prediction]])[0][0]
        else:
            predicted_price = raw_prediction

        predicted_change = (predicted_price - last_close) / last_close

        volatility_multiplier = 1.5
        max_daily_move = last_close * daily_volatility * volatility_multiplier

        trend_weight = 0.2
        trend_adjustment = weighted_trend * trend_weight

        predicted_change = predicted_change * (1 + trend_adjustment)

        if abs(predicted_change) > 0.5 * daily_volatility:
            dampening_factor = np.exp(
                -3 * abs(predicted_change) / daily_volatility)
            predicted_change *= dampening_factor

        capped_change = np.clip(predicted_change, -max_daily_move / last_close,
                                max_daily_move / last_close)

        preliminary_price = last_close * (1 + capped_change)

        sentiment_impact = 0.00001 * sentiment_score
        sentiment_adjusted_price = preliminary_price * (1 + sentiment_impact)

        weighted_avg_price = (0.5 * sentiment_adjusted_price +
                              0.2 * last_close +
                              0.2 * avg_price +
                              0.1 * median_price)

        max_allowed_change = 0.01
        final_predicted_price = np.clip(
            weighted_avg_price,
            last_close * (1 - max_allowed_change),
            last_close * (1 + max_allowed_change)
        )

        ma_boundary = 0.02
        ema5_price = stock_data['EMA_5'].iloc[-1]

        if abs((final_predicted_price - ema5_price
                ) / ema5_price) > ma_boundary:
            final_predicted_price = ema5_price * (
                1 + np.sign(final_predicted_price - ema5_price) * ma_boundary)

        return final_predicted_price

    except Exception as e:
        st.error(f"Error predicting stock price for next day: {e}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)
        return None


def process_ticker_data(
        ticker, adjusted_datetime, prediction_created_at, stock_data,
        lstm_model, tokenizer, rf_model, scaler, sentiment_scaler,
        target_scaler, prediction_type):

    stock_data = compute_technical_indicators(stock_data)
    stock_data = add_previous_close_column(stock_data)
    sentiment_data = collect_sentiment_data(ticker)
    sentiment_score = calculate_sentiment_score(
        sentiment_data, lstm_model, tokenizer)
    stock_data['Sentiment_Score'] = sentiment_score

    if prediction_type == "Next Day":
        predicted_price = predict_next_day_price(
            stock_data,
            sentiment_score,
            rf_model,
            scaler,
            sentiment_scaler,
            target_scaler)
    else:
        predicted_price = predict_next_hour_price(
            stock_data,
            sentiment_score,
            rf_model,
            scaler,
            sentiment_scaler,
            target_scaler)

    if predicted_price is not None:
        signal, reason = calculate_difference_and_signal(
            predicted_price, stock_data)

        prediction_data = {
            "Ticker": ticker,
            "Prediction Type": prediction_type,
            "Prediction Created": prediction_created_at.strftime(
                '%Y-%m-%d %H:%M:%S IST'),
            "Target Prediction Date": adjusted_datetime.strftime('%Y-%m-%d'),
            "Target Prediction Time": adjusted_datetime.strftime('%H:%M IST'),
            "Predicted Price": round(
                predicted_price,
                2),
            "Signal": signal,
            "Reason": reason,
            "Sentiment Score": round(
                sentiment_score,
                2)}
        return prediction_data
    return None


def is_market_open(dt):

    ist_dt = dt.astimezone(pytz.timezone('Asia/Kolkata'))

    if ist_dt.weekday() >= 5:
        return False

    market_start = ist_dt.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = ist_dt.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_start <= ist_dt <= market_end


def get_next_market_datetime(dt, prediction_type="Next Hour"):

    ist_dt = dt.astimezone(pytz.timezone('Asia/Kolkata'))
    date_str = ist_dt.strftime('%Y-%m-%d %H:%M')
    message = ""

    if ist_dt.weekday() >= 5:
        days_to_monday = (7 - ist_dt.weekday()) + 0
        ist_dt = ist_dt + timedelta(days=days_to_monday)
        if prediction_type == "Next Day":
            ist_dt = ist_dt.replace(hour=9, minute=15, second=0, microsecond=0)
            message = (f"Market is closed on weekends."
                       f"Prediction adjusted to Monday {date_str} IST")

        else:
            ist_dt = ist_dt.replace(hour=9, minute=15, second=0, microsecond=0)
            message = (f"Market is closed on weekends."
                       f"Prediction adjusted to Monday {date_str} IST")

        return ist_dt, message

    market_end = ist_dt.replace(hour=15, minute=30, second=0, microsecond=0)
    next_market_start = ist_dt.replace(
        hour=9, minute=15, second=0, microsecond=0)

    if prediction_type == "Next Day":
        ist_dt = (
            ist_dt +
            timedelta(
                days=1)).replace(
            hour=9,
            minute=15,
            second=0,
            microsecond=0)

        while ist_dt.weekday() >= 5:
            ist_dt = ist_dt + timedelta(days=1)
        message = f"Next day prediction set for {date_str} IST"
    else:
        if ist_dt > market_end:
            ist_dt = (
                ist_dt +
                timedelta(
                    days=1)).replace(
                hour=9,
                minute=15,
                second=0,
                microsecond=0)
            while ist_dt.weekday() >= 5:
                ist_dt = ist_dt + timedelta(days=1)
            message = (
                f"Market is closed. Prediction adjusted to next trading day"
                f"{date_str} IST")
        elif ist_dt < next_market_start:
            ist_dt = next_market_start
            message = (
                f"Market is not open yet. Prediction adjusted to market open"
                f"{date_str} IST")

    return ist_dt, message


def create_predictions_table():
    """Create SQLite table to store predictions if it doesn't exist"""
    conn = sqlite3.connect('stock_predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ticker TEXT,
                  prediction_type TEXT,
                  prediction_created TIMESTAMP,
                  target_date DATE,
                  target_time TIME,
                  predicted_price REAL,
                  signal TEXT,
                  reason TEXT,
                  sentiment_score REAL)''')
    conn.commit()
    conn.close()


def save_prediction(prediction_data):
    """Save a single prediction to the database with enhanced debugging"""
    conn = None
    try:
        st.write("Saving prediction with data:")
        st.write(prediction_data)

        conn = sqlite3.connect('stock_predictions.db')
        c = conn.cursor()

        prediction_created = datetime.strptime(
            prediction_data['Prediction Created'],
            '%Y-%m-%d %H:%M:%S IST')
        target_date = datetime.strptime(
            prediction_data['Target Prediction Date'],
            '%Y-%m-%d').date()
        target_time = datetime.strptime(
            prediction_data['Target Prediction Time'],
            '%H:%M IST').time()

        st.write("Converted dates:")
        st.write(f"Prediction Created: {prediction_created}")
        st.write(f"Target Date: {target_date}")
        st.write(f"Target Time: {target_time}")

        insert_query = '''INSERT INTO predictions
                         (ticker, prediction_type, prediction_created,
                          target_date, target_time, predicted_price, signal,
                          reason, sentiment_score)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''

        values = (prediction_data['Ticker'],
                  prediction_data['Prediction Type'],
                  prediction_created,
                  target_date,
                  target_time,
                  prediction_data['Predicted Price'],
                  prediction_data['Signal'],
                  prediction_data['Reason'],
                  prediction_data['Sentiment Score'])

        st.write("Executing SQL with values:")
        st.write(values)

        c.execute(insert_query, values)
        conn.commit()

        c.execute(
            (
                "SELECT * FROM predictions WHERE ticker=?"
                "AND prediction_created=?"),
            (prediction_data['Ticker'],
             prediction_created))
        saved_record = c.fetchone()

        if saved_record:
            ticker_name = prediction_data['Ticker']
            st.success(f"Successfully saved prediction for {ticker_name}")
        else:
            st.warning("Insert succeeded but verification failed")

        conn.close()

    except Exception as e:
        st.error(f"Error saving prediction: {str(e)}")
        if conn:
            conn.close()
